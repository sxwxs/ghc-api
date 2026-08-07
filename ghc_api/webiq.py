"""
Microsoft Web IQ search, exposed as an LLM-callable tool.

Design
------
The proxy does NOT search on the model's behalf and does NOT inject anything
into prompts. It only exposes one plain endpoint (``POST /v1/webiq/search``)
that runs a search with the server-held API key.

The caller (the built-in chat UI, or any API client) declares ``webiq_search``
as a normal function tool, lets the model decide whether and what to search,
then executes the tool call against that endpoint and feeds the result back.

This keeps the LLM proxy path a pure pass-through: no hidden extra upstream
calls, no silent tool rewriting, no search failure escalating into a failed
chat request.
"""

from typing import Any, Dict, List, Optional

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
    "POST /v1/webiq/search."
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
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}

# Guard rails applied to whatever the model or client asks for.
MAX_QUERY_CHARS = 400
MAX_RESULTS_LIMIT = 10


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
    """Validate and trim a model-supplied query."""
    if not isinstance(raw, str):
        raise WebIQError("'query' must be a string.", 400)
    query = " ".join(raw.split())
    if not query:
        raise WebIQError("'query' must not be empty.", 400)
    return query[:MAX_QUERY_CHARS]


def normalize_max_results(raw: Any, default: int) -> int:
    """Clamp a requested result count into the allowed range."""
    if raw is None:
        return max(1, min(MAX_RESULTS_LIMIT, int(default)))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise WebIQError("'max_results' must be an integer.", 400) from None
    return max(1, min(MAX_RESULTS_LIMIT, value))


def search(query: str, settings: Any, *, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
    """Run one Web IQ v3 web search and return normalized results.

    Raises WebIQError; never returns partial or fabricated data.
    """
    if not is_configured(settings):
        raise WebIQError("Microsoft Web IQ is not configured on this server.", 503)

    query = normalize_query(query)
    count = normalize_max_results(max_results, settings.webiq_max_results)

    try:
        response = requests.post(
            settings.webiq_endpoint,
            headers={
                "x-apikey": settings.webiq_api_key,
                "content-type": "application/json",
            },
            json={
                "query": query,
                "maxResults": count,
                "language": settings.webiq_language,
                "region": settings.webiq_region,
                "maxLength": settings.webiq_max_length,
                "contentFormat": settings.webiq_content_format,
                "safeSearch": settings.webiq_safe_search,
            },
            timeout=settings.webiq_timeout,
        )
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
        raise WebIQError(f"Web IQ connection failed: {type(exc).__name__}", 504) from exc
    except requests.RequestException as exc:
        raise WebIQError(f"Web IQ request failed: {type(exc).__name__}") from exc

    if not response.ok:
        raise WebIQError(
            f"Web IQ returned HTTP {response.status_code}.",
            429 if response.status_code == 429 else 502,
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise WebIQError("Web IQ returned invalid JSON.") from exc

    raw_results = body.get("webResults") if isinstance(body, dict) else None
    return _normalize_results(raw_results, settings.webiq_max_length)


def _normalize_results(raw_results: Any, max_length: int) -> List[Dict[str, Any]]:
    """Keep only the fields the model needs, and cap content length locally.

    Web IQ's maxLength is a server-side hint; clamping here bounds the token
    cost of a tool result regardless of what upstream actually returns.
    """
    if not isinstance(raw_results, list):
        return []
    limit = max(1, int(max_length))
    results = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        content = item.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        results.append({
            "title": item.get("title") or "Untitled",
            "url": item.get("url") or "",
            "content": content[:limit],
        })
    return results
