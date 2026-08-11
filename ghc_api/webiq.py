"""
Microsoft Web IQ search, exposed as an LLM-callable tool.

Design
------
The proxy does NOT search on the model's behalf and does NOT inject anything
into prompts. It only exposes one plain endpoint (``POST /v3/search/web``)
that runs a search with the server-held API key.

That path, its request body and its response body are the official Microsoft
Web Search v3 contract, verbatim. A client written against api.microsoft.ai
works against this proxy by changing the base URL and nothing else, exactly
like the OpenAI- and Anthropic-shaped endpoints elsewhere in this project.
What the proxy adds is key custody, an auth gate, logging and quota caps --
none of which need a bespoke schema.

The rule that keeps compatibility verifiable:

    all policy is applied to the request; the response is passed through.

Defaults and caps therefore clamp the *outgoing* body, and whatever upstream
answers is returned to the client untouched -- including ``traceId``,
``contentTier``, ``clickUrl`` and the other fields a caller may be
contractually required to act on.

The caller (the built-in chat UI, or any API client) declares ``webiq_search``
as a normal function tool, lets the model decide whether and what to search,
then executes the tool call against that endpoint and feeds the result back.
That tool schema is a prompt surface, not the API: it stays deliberately
narrow (query plus a result count) because handing a model eight knobs makes
it fill them in badly.

This keeps the LLM proxy path a pure pass-through: no hidden extra upstream
calls, no silent tool rewriting, no search failure escalating into a failed
chat request.
"""

import re
from typing import Any, Dict, Optional

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

# The endpoint the Web Search v3 specification defines, used unless config.yaml
# overrides it via webiq_endpoint (for a mock, a recording proxy, or a regional
# deployment), matching the github_api_base_url/copilot_api_base_url overrides
# elsewhere in this project.
ENDPOINT = "https://api.microsoft.ai/v3/search/web"

# Ceilings defined by the Web Search v3 specification. A request that exceeds
# one of these is rejected with 400 rather than quietly trimmed: silently
# truncating a query would mangle trailing "site:" operators into a different,
# still-plausible search.
QUERY_MAX_CHARS = 1000
RESULTS_MAX = 50
LENGTH_MAX = 500000
CONTENT_FORMATS = ("passage", "text", "html", "markdown")
SAFE_SEARCH_VALUES = ("off", "strict")

# "lat:<float>;long:<float>"
LOCATION_PATTERN = re.compile(r"^lat:-?\d+(\.\d+)?;long:-?\d+(\.\d+)?$")
LANGUAGE_PATTERN = re.compile(r"^[A-Za-z]{2}$")
REGION_PATTERN = re.compile(r"^[A-Za-z]{2}$")

# Every field the official request body accepts. Anything else is rejected so
# that a typo -- or the snake_case names this endpoint used before it was made
# spec-compatible -- fails loudly instead of being silently ignored.
SUPPORTED_PARAMS = (
    "query", "maxResults", "language", "region", "location",
    "contentFormat", "maxLength", "safeSearch",
)

# Old snake_case spellings, mapped to their replacements for the error message.
RETIRED_PARAMS = {
    "max_results": "maxResults",
    "max_length": "maxLength",
    "content_format": "contentFormat",
    "safe_search": "safeSearch",
}

# Upstream statuses that describe the *client's* request or the upstream
# service, and so mean the same thing when re-emitted by this proxy.
PASSTHROUGH_STATUSES = frozenset({400, 410, 415, 429, 500, 503, 504})


class WebIQError(RuntimeError):
    """A safe-to-return Web IQ failure carrying an HTTP status code."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


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


def normalize_query(raw: Any) -> str:
    """Validate a client-supplied query against the official contract.

    Collapses surrounding and repeated whitespace, which is cosmetic and cannot
    change the meaning of a search, but rejects rather than truncates an
    over-long query.
    """
    if not isinstance(raw, str):
        raise WebIQError("'query' must be a string.", 400)
    query = " ".join(raw.split())
    if not query:
        raise WebIQError("'query' must not be empty.", 400)
    if len(query) > QUERY_MAX_CHARS:
        raise WebIQError(
            f"'query' must be at most {QUERY_MAX_CHARS} characters "
            f"(got {len(query)}).",
            400,
        )
    return query


def _bounded_int(raw: Any, name: str, default: int, spec_max: int, cap: int) -> int:
    """Resolve one numeric parameter: default, spec check, then local cap.

    Out-of-spec values are an error, because the official API would reject them
    too and a client deserves to know its request was wrong. A value that is
    valid but above this server's own cap is clamped instead: that cap is local
    policy protecting a shared paid key, not part of the contract. The clamp is
    visible in the logged upstream request.
    """
    if raw is None:
        value = int(default)
    else:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise WebIQError(f"'{name}' must be an integer.", 400)
        value = raw
        if value < 1 or value > spec_max:
            raise WebIQError(f"'{name}' must be between 1 and {spec_max}.", 400)
    return max(1, min(value, int(cap), spec_max))


def _enum(raw: Any, name: str, default: str, allowed: tuple) -> str:
    value = default if raw is None else raw
    if not isinstance(value, str) or value not in allowed:
        raise WebIQError(f"'{name}' must be one of: {', '.join(allowed)}.", 400)
    return value


def _code(raw: Any, name: str, default: str, pattern: "re.Pattern") -> str:
    value = default if raw is None else raw
    if not isinstance(value, str) or not pattern.match(value):
        raise WebIQError(f"'{name}' must be a 2-letter code.", 400)
    return value


def build_upstream_request(payload: Dict[str, Any], settings: Any) -> Dict[str, Any]:
    """Turn a client request into an official Web Search v3 request body.

    Server configuration supplies the default for every optional parameter, so
    an operator can tune the deployment without locking clients out of the
    contract: any request may override any of them.
    """
    if not isinstance(payload, dict):
        raise WebIQError("Request body must be a JSON object.", 400)

    for key in payload:
        if key in SUPPORTED_PARAMS:
            continue
        if key in RETIRED_PARAMS:
            raise WebIQError(
                f"'{key}' is not a Web Search v3 parameter; "
                f"use '{RETIRED_PARAMS[key]}'.",
                400,
            )
        raise WebIQError(
            f"Unknown parameter '{key}'. Supported: {', '.join(SUPPORTED_PARAMS)}.",
            400,
        )

    body = {
        "query": normalize_query(payload.get("query")),
        "maxResults": _bounded_int(
            payload.get("maxResults"), "maxResults",
            settings.webiq_max_results, RESULTS_MAX, settings.webiq_max_results_cap,
        ),
        "language": _code(
            payload.get("language"), "language", settings.webiq_language, LANGUAGE_PATTERN),
        "region": _code(
            payload.get("region"), "region", settings.webiq_region, REGION_PATTERN),
        "maxLength": _bounded_int(
            payload.get("maxLength"), "maxLength",
            settings.webiq_max_length, LENGTH_MAX, settings.webiq_max_length_cap,
        ),
        "contentFormat": _enum(
            payload.get("contentFormat"), "contentFormat",
            settings.webiq_content_format, CONTENT_FORMATS),
        "safeSearch": _enum(
            payload.get("safeSearch"), "safeSearch",
            settings.webiq_safe_search, SAFE_SEARCH_VALUES),
    }

    location = payload.get("location")
    if location is not None:
        if not isinstance(location, str) or not LOCATION_PATTERN.match(location):
            raise WebIQError(
                "'location' must look like 'lat:<float>;long:<float>'.", 400)
        body["location"] = location

    return body


def _upstream_failure(response: Any) -> WebIQError:
    """Translate a failed upstream response into a WebIQError.

    Statuses that describe the client's request or the upstream service are
    re-emitted unchanged, so an official error code keeps its official meaning.
    The exception is 401/403: those mean *this server's* Web IQ credentials
    were rejected, and re-emitting them would be indistinguishable from this
    proxy rejecting the caller's own token. They surface as 503 -- the service
    is unusable due to server-side configuration -- with an explicit message.
    """
    status = response.status_code
    detail = (response.text or "").strip()
    if len(detail) > 500:
        detail = detail[:500] + "..."

    if status in (401, 403):
        return WebIQError(
            "Web IQ rejected this server's API key "
            f"(upstream HTTP {status}). Check webiq_api_key in config.yaml."
            + (f" Upstream said: {detail}" if detail else ""),
            503,
        )

    mapped = status if status in PASSTHROUGH_STATUSES else 502
    return WebIQError(
        f"Web IQ returned HTTP {status}." + (f" Upstream said: {detail}" if detail else ""),
        mapped,
    )


def search(
    payload: Dict[str, Any],
    settings: Any,
    *,
    trace: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one Web Search v3 request and return the upstream body verbatim.

    Raises WebIQError; never returns partial or fabricated data.

    The returned dict is exactly what upstream sent, so ``traceId``,
    ``contentTier``, ``clickUrl``, ``instrumentationSuffix``, ``crawledAt`` and
    everything else reach the caller. Dropping any of them would make click
    instrumentation impossible, hide premium-tier billing, and leave support
    tickets without a trace id.

    ``trace`` is an optional dict filled in place with what actually happened
    upstream (endpoint, the request body sent, the HTTP status). Callers use it
    to log the exchange, including for the error paths that raise below. The
    API key travels in a header and is deliberately never written there.
    """
    if trace is None:
        trace = {}
    endpoint = getattr(settings, "webiq_endpoint", "") or ENDPOINT
    trace["endpoint"] = endpoint

    # Validated before the configuration check: a malformed request is the
    # client's bug and stays the client's bug whether or not this server
    # happens to hold a usable key.
    upstream_request = build_upstream_request(payload, settings)
    trace["request"] = upstream_request

    if not is_configured(settings):
        raise WebIQError("Microsoft Web IQ is not configured on this server.", 503)

    try:
        response = requests.post(
            endpoint,
            headers={
                # Host is not set here: HTTP requires it and urllib3 derives it
                # from the URL. The spec lists it only as HTTP boilerplate.
                "x-apikey": settings.webiq_api_key,
                "content-type": "application/json",
            },
            json=upstream_request,
            timeout=settings.webiq_timeout,
        )
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
        raise WebIQError(f"Web IQ connection failed: {type(exc).__name__}", 504) from exc
    except requests.RequestException as exc:
        raise WebIQError(f"Web IQ request failed: {type(exc).__name__}") from exc

    trace["status_code"] = getattr(response, "status_code", None)

    if not response.ok:
        raise _upstream_failure(response)

    try:
        body = response.json()
    except ValueError as exc:
        raise WebIQError("Web IQ returned invalid JSON.") from exc

    if not isinstance(body, dict):
        raise WebIQError("Web IQ returned an unexpected response shape.")

    trace["result_count"] = len(web_results(body))
    return body


def web_results(body: Dict[str, Any]) -> list:
    """The webResults array of a response, or [] if absent or malformed.

    Only for logging and counting. The response itself is never rewritten.
    """
    results = body.get("webResults") if isinstance(body, dict) else None
    return results if isinstance(results, list) else []
