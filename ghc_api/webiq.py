"""
Microsoft Web IQ REST and MCP proxy, with Web Search as an LLM-callable tool.

Design
------
The proxy does NOT search on the model's behalf and does NOT inject anything
into prompts. It exposes the six official Web IQ v3 REST endpoints, plus the
Web IQ Streamable HTTP MCP endpoint, using the server-held API key.

The REST contracts are Microsoft's official Web IQ v3 contracts:
https://webiq.microsoft.ai/documentation/api-reference/

It is a transparent proxy in the strict sense -- the request body is forwarded
as the raw bytes the client sent, and the upstream status, headers and body
come back untouched. There is deliberately no parameter whitelist, no local
defaulting, no clamping and no schema validation, because every one of those
is a way for this proxy to disagree with the service it claims to be. A
parameter Microsoft adds tomorrow works here today, and an invalid request
gets the authoritative upstream error rather than an imitation of it.

The only thing that is not passed through is the API key: the client's own
``x-apikey``/``Authorization`` headers are never forwarded (that is the whole
point of key custody) and never persisted (see ``auth.redact_auth_headers``).
Upstream authentication failures become 503 because they describe *this
server's* credentials, not the caller's token. Browse 403 is preserved because
Microsoft also uses it for URL/content policy rejection, not only key access.

The caller (the built-in chat UI, or any API client) declares ``webiq_search``
as a normal function tool, lets the model decide whether and what to search,
then executes the tool call against that endpoint and feeds the result back.
That tool schema is a prompt surface, not the API: it stays deliberately
narrow (query plus a result count) because handing a model eight knobs makes
it fill them in badly. The client is what turns those narrow arguments into a
full official request -- see ``scripts/webiq_search_demo.py``.

This keeps the LLM proxy path a pure pass-through: no hidden extra upstream
calls, no silent tool rewriting, no search failure escalating into a failed
chat request.
"""

from typing import Any, Dict, List, Mapping, Optional, Tuple

import requests

TOOL_NAME = "webiq_search"

# Retired one-shot option. It used to make the proxy search on the client's
# behalf and splice results into the system prompt. It is now rejected rather
# than silently ignored, so callers cannot believe a search happened when it
# did not.
LEGACY_OPTION_KEY = "webiq_search_options"

LEGACY_OPTION_MESSAGE = (
    "'webiq_search_options' has been removed. Declare the 'webiq_search' "
    "function tool instead, and execute the model's tool calls against "
    "POST /v3/search/web."
)

TOOL_DESCRIPTION = (
    "Search the public web and return ranked passages with source URLs. "
    "Use it for facts that are recent, niche, or that you are not confident "
    "about, and cite the returned sources in your answer."
)

TOOL_PARAMETERS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "A focused search-engine query (keywords, not a full "
                "sentence). Resolve pronouns from the conversation first."
            ),
        },
        "max_results": {
            "type": "integer",
            "description": "How many results to return (1-10).",
            "minimum": 1,
            "maximum": 10,
        },  # kept snake_case: this is the model-facing tool, not the HTTP API
    },
    "required": ["query"],
    "additionalProperties": False,
}

# The complete public Web IQ v3 surface. Keeping this as an allowlist prevents
# a caller from turning the server-held key into a credential for an arbitrary
# path while still allowing additive request fields to pass through untouched.
API_BASE_URL = "https://api.microsoft.ai"
WEB_PATH = "/v3/search/web"
MCP_PATH = "/v3/mcp"
API_PATHS: Dict[str, str] = {
    WEB_PATH: "web",
    "/v3/search/videos": "videos",
    "/v3/browse": "browse",
    "/v3/search/news": "news",
    "/v3/search/images": "images",
    "/v3/search/classic": "classic",
}
ALL_PATHS = frozenset((*API_PATHS, MCP_PATH))

# Standard HTTP fields used by Streamable HTTP plus the open-ended MCP header
# namespace. MCP 2026-07-28 added Mcp-Method, Mcp-Name and Mcp-Param-*; keeping
# a fixed list here would silently break every later protocol addition. Client
# authentication is still replaced by the server-held x-apikey, and requests
# owns Host, framing and connection headers.
MCP_STANDARD_REQUEST_HEADERS = frozenset({
    "accept", "content-type", "last-event-id",
})

# Kept as a public compatibility constant for clients/tests that imported the
# original web-only endpoint.
ENDPOINT = API_BASE_URL + WEB_PATH

# Response headers that must not be copied from upstream to the client.
#
# * Hop-by-hop headers are forbidden for WSGI applications by PEP 3333
#   (waitress answers 500 if one appears) and describe the upstream connection,
#   not this one.
# * Content-Length/Content-Encoding describe a body framing that Flask
#   recomputes for its own response.
# * Date and Server are generated by this server. Copying them too makes the
#   WSGI layer emit a comma-joined "Date: <ours>, <upstream's>", which is both
#   wrong and unparseable.
# * Alt-Svc and Strict-Transport-Security are claims about *upstream's* origin
#   ("this origin also speaks h3 on port N", "this origin is HTTPS-only").
#   Re-emitted here they become claims about this proxy's origin, which are not
#   true and which a browser would act on.
#
# Everything else survives, so Retry-After and the x-ms-* diagnostics reach the
# caller.
DROPPED_RESPONSE_HEADERS = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "trailers", "transfer-encoding", "upgrade",
    "content-length", "content-encoding",
    "date", "server",
    "alt-svc", "strict-transport-security",
})


class WebIQError(RuntimeError):
    """A safe-to-return Web IQ failure carrying an HTTP status code.

    Raised only when there is no upstream response to pass through: this
    server is unconfigured, the connection failed, or upstream rejected this
    server's key. Everything else is returned verbatim, not raised.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 502,
        *,
        upstream_status: Optional[int] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.upstream_status = upstream_status


def pop_legacy_option(payload: Dict) -> bool:
    """Strip the retired option from a payload in place.

    Always removes the key so it can never be forwarded upstream as an unknown
    parameter. Returns True when the caller actually asked for a search, so the
    route can answer with a migration error instead of quietly dropping it.
    """
    if not isinstance(payload, dict) or LEGACY_OPTION_KEY not in payload:
        return False
    option = payload.pop(LEGACY_OPTION_KEY)
    if isinstance(option, dict):
        return bool(option.get("enabled", True))
    return bool(option)


def is_configured(settings: Any) -> bool:
    """Whether Web IQ is enabled and has a key."""
    return bool(getattr(settings, "enable_webiq_search", False)
                and getattr(settings, "webiq_api_key", ""))


def tool_definition(api_style: str) -> Dict[str, Any]:
    """Return the webiq_search tool schema in Chat or Responses shape."""
    if api_style == "responses":
        return {
            "type": "function",
            "name": TOOL_NAME,
            "description": TOOL_DESCRIPTION,
            "parameters": TOOL_PARAMETERS,
        }
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": TOOL_DESCRIPTION,
            "parameters": TOOL_PARAMETERS,
        },
    }


def endpoint_for(settings: Any, api_path: str = WEB_PATH) -> str:
    """Return the upstream URL for one allowlisted Web IQ path.

    ``webiq_base_url`` is the all-service override and should be an origin (or
    reverse-proxy prefix), for example ``http://127.0.0.1:9000``. The old
    ``webiq_endpoint`` setting remains compatible for Web Search. If that old
    value is the standard-shaped ``.../v3/search/web`` URL, its base is also
    safely derivable for the other services.
    """
    if api_path not in ALL_PATHS:
        raise ValueError(f"Unsupported Web IQ API path: {api_path}")

    base_url = getattr(settings, "webiq_base_url", "")
    if base_url:
        return base_url.rstrip("/") + api_path

    legacy_endpoint = getattr(settings, "webiq_endpoint", "")
    if legacy_endpoint:
        if api_path == WEB_PATH:
            return legacy_endpoint
        if legacy_endpoint.rstrip("/").endswith(WEB_PATH):
            normalized = legacy_endpoint.rstrip("/")
            return normalized[:-len(WEB_PATH)].rstrip("/") + api_path
        # Do not silently split traffic between a legacy Web Search mock and
        # production Web IQ. That can spend real quota and disclose test data.
        raise ValueError(
            "webiq_endpoint is a Web Search-only URL and cannot safely route "
            f"{api_path}; configure webiq_base_url for all Web IQ services"
        )

    return API_BASE_URL + api_path


def timeout_for(settings: Any, api_path: str) -> int:
    """Resolve the REST read timeout, with slower services independently tunable."""
    field = {
        "/v3/browse": "webiq_browse_timeout",
        "/v3/search/classic": "webiq_classic_timeout",
    }.get(api_path)
    if field:
        configured = getattr(settings, field, None)
        if configured is not None:
            return max(1, int(configured))
    return max(1, int(getattr(settings, "webiq_timeout", 30)))


def passthrough_headers(headers: Any) -> List[Tuple[str, str]]:
    """Upstream response headers that are safe to hand back to the client.

    Everything survives except the connection-, framing- and origin-scoped ones
    listed in DROPPED_RESPONSE_HEADERS, so ``Retry-After`` on a 429 -- the one
    header a client needs in order to back off correctly against a shared paid
    key -- reaches the caller.
    """
    if not headers:
        return []
    return [(name, value) for name, value in headers.items()
            if name.lower() not in DROPPED_RESPONSE_HEADERS]


def search(
    body: bytes,
    settings: Any,
    *,
    content_type: str = "application/json",
    api_path: str = WEB_PATH,
) -> requests.Response:
    """Forward one Web IQ REST request and return the raw upstream response.

    ``body`` is handed over as received; this function never parses, validates,
    defaults or rewrites it. The caller gets the ``requests.Response`` so it
    can pass status, headers and content through verbatim.

    Raises WebIQError only when there is nothing to pass through: not
    configured, connection failed, or upstream rejected this server's key.
    """
    if not is_configured(settings):
        raise WebIQError("Microsoft Web IQ is not configured on this server.", 503)

    try:
        upstream_url = endpoint_for(settings, api_path)
    except ValueError as exc:
        raise WebIQError(str(exc), 503) from exc

    try:
        response = requests.post(
            upstream_url,
            headers={
                # The client's own x-apikey/Authorization are deliberately not
                # forwarded: this server's key is the only one that is used.
                # Host is not set here either; urllib3 derives it from the URL.
                "x-apikey": settings.webiq_api_key,
                "content-type": content_type or "application/json",
            },
            data=body,
            timeout=timeout_for(settings, api_path),
        )
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
        raise WebIQError(f"Web IQ connection failed: {type(exc).__name__}", 504) from exc
    except requests.RequestException as exc:
        raise WebIQError(f"Web IQ request failed: {type(exc).__name__}") from exc

    # Browse also uses 403 for blocked URLs and filtered content, which is an
    # authoritative result the caller needs. Its credential failure is
    # ambiguous, so only the unambiguous 401 is rewritten on that endpoint.
    rejected_server_key = response.status_code == 401 or (
        response.status_code == 403 and api_path != "/v3/browse"
    )
    if rejected_server_key:
        detail = (response.text or "").strip()
        if len(detail) > 500:
            detail = detail[:500] + "..."
        raise WebIQError(
            "Web IQ rejected this server's API key "
            f"(upstream HTTP {response.status_code}). "
            "Check webiq_api_key in config.yaml."
            + (f" Upstream said: {detail}" if detail else ""),
            503,
            upstream_status=response.status_code,
        )

    return response


def mcp_request(
    method: str,
    body: bytes,
    settings: Any,
    *,
    request_headers: Mapping[str, str],
) -> requests.Response:
    """Open a streaming request to the official Web IQ MCP server.

    MCP transport headers are end-to-end protocol state and must survive the
    proxy. Authentication and connection/framing headers do not: the former is
    replaced with this server's key and ``requests`` owns the latter.
    """
    if not is_configured(settings):
        raise WebIQError("Microsoft Web IQ is not configured on this server.", 503)

    forwarded_headers = {"x-apikey": settings.webiq_api_key}
    for raw_name, value in request_headers.items():
        name = raw_name.lower().strip()
        if not value:
            continue
        if name in MCP_STANDARD_REQUEST_HEADERS or name.startswith("mcp-"):
            forwarded_headers[name] = value

    try:
        upstream_url = endpoint_for(settings, MCP_PATH)
    except ValueError as exc:
        raise WebIQError(str(exc), 503) from exc

    normalized_method = method.upper()
    connect_timeout = max(1, int(getattr(settings, "webiq_timeout", 30)))
    if normalized_method == "GET":
        # GET is the long-lived server-to-client event channel. The route-level
        # concurrency cap prevents these intentionally unbounded reads from
        # consuming the entire waitress thread pool.
        timeout = (connect_timeout, None)
    else:
        # A stuck tools/call or DELETE must eventually release its WSGI thread.
        read_timeout = max(1, int(getattr(
            settings, "webiq_mcp_timeout", connect_timeout)))
        timeout = (connect_timeout, read_timeout)

    try:
        response = requests.request(
            normalized_method,
            upstream_url,
            headers=forwarded_headers,
            data=body if normalized_method not in ("GET", "DELETE") else None,
            timeout=timeout,
            stream=True,
        )
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
        raise WebIQError(f"Web IQ MCP connection failed: {type(exc).__name__}", 504) from exc
    except requests.RequestException as exc:
        raise WebIQError(f"Web IQ MCP request failed: {type(exc).__name__}") from exc

    if response.status_code in (401, 403):
        response.close()
        raise WebIQError(
            "Web IQ rejected this server's API key "
            f"(upstream HTTP {response.status_code}). "
            "Check webiq_api_key in config.yaml.",
            503,
            upstream_status=response.status_code,
        )
    return response


def result_count(body: Any, api_path: str = WEB_PATH) -> Optional[int]:
    """Number of primary results in a parsed REST response, if applicable.

    Used for logging only; response bodies are never changed. Classic combines
    verticals, so its count is the sum of all top-level ``*Results`` arrays.
    Browse has one document rather than a result array and therefore returns
    ``None``.
    """
    if not isinstance(body, dict):
        return None
    if api_path == "/v3/search/classic":
        arrays = [value for key, value in body.items()
                  if key.endswith("Results") and isinstance(value, list)]
        return sum(len(value) for value in arrays) if arrays else None
    result_key = {
        WEB_PATH: "webResults",
        "/v3/search/videos": "videoResults",
        "/v3/search/news": "newsResults",
        "/v3/search/images": "imageResults",
    }.get(api_path)
    results = body.get(result_key) if result_key else None
    return len(results) if isinstance(results, list) else None
