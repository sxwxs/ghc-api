"""
Web search proxy preprocessing.

When enabled for a request containing the web_search tool, this module calls
an external search proxy, injects the results into the system prompt, removes
the unsupported tool, and returns the payload to send to Copilot.
"""

from typing import Any, Dict, List

import requests


SEARCH_QUERY_PREFIX = "Perform a web search for the query:"


def has_web_search_tool(payload: Dict) -> bool:
    """Return True if the payload contains any web_search-type tool."""
    for tool in payload.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type", "")
        if isinstance(tool_type, str) and tool_type.startswith("web_search"):
            return True
    return False


def _normalize_search_query(text: str) -> str:
    """Remove Claude Code's web-search wrapper from a query string."""
    query = text.strip()
    if query.casefold().startswith(SEARCH_QUERY_PREFIX.casefold()):
        query = query[len(SEARCH_QUERY_PREFIX):].lstrip()
    return query[:200]


def extract_search_query(payload: Dict) -> str:
    """Extract a search query from the last user message."""
    for msg in reversed(payload.get("messages", [])):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return _normalize_search_query(content)
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return _normalize_search_query(" ".join(parts))
    return ""


def call_search_proxy(query: str, endpoint: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Call the external search proxy and return a list of result dicts."""
    if not query or not endpoint:
        return []
    endpoint = endpoint.rstrip("/")
    try:
        resp = requests.get(
            f"{endpoint}/search",
            params={"keyword": query, "limit": limit},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            results = data
        elif isinstance(data, dict):
            results = data.get("results")
            if not isinstance(results, list):
                results = data.get("items")
            if not isinstance(results, list):
                results = data.get("data")
        else:
            results = None
        if not isinstance(results, list):
            return []
        return [item for item in results if isinstance(item, dict)]
    except Exception as e:
        print(f"[WebSearch] Search proxy call failed: {type(e).__name__}: {e}")
        return []


def format_search_results(query: str, results: List[Dict[str, Any]]) -> str:
    """Format search results as a readable text block."""
    if not results:
        return f'[Web Search Results]\nNo results found for "{query}".'

    lines = [
        "[Web Search Results]",
        f'Search results for "{query}":',
        "",
    ]
    for i, result in enumerate(results, 1):
        title = result.get("title") or "Untitled"
        url = result.get("link") or result.get("url")
        description = (
            result.get("description")
            or result.get("snippet")
            or result.get("content")
        )
        lines.append(f"{i}. {title}")
        if url:
            lines.append(f"   URL: {url}")
        if description:
            lines.append(f"   {description}")
        lines.append("")
    return "\n".join(lines)


def inject_search_results_into_payload(payload: Dict, search_results_text: str) -> Dict:
    """Return a new payload with search results injected into the system field."""
    system = payload.get("system")

    if system is None:
        new_system = [{"type": "text", "text": search_results_text}]
    elif isinstance(system, str):
        new_system = system + "\n\n" + search_results_text
    elif isinstance(system, list):
        new_system = list(system) + [{"type": "text", "text": search_results_text}]
    else:
        new_system = [{"type": "text", "text": search_results_text}]

    return {**payload, "system": new_system}


def remove_web_search_tools(payload: Dict) -> Dict:
    """Return a new payload with web_search tools removed."""
    tools = payload.get("tools")
    if not tools:
        return payload

    filtered = [
        tool
        for tool in tools
        if not (
            isinstance(tool, dict)
            and isinstance(tool.get("type", ""), str)
            and tool.get("type", "").startswith("web_search")
        )
    ]

    new_payload = dict(payload)
    if filtered:
        new_payload["tools"] = filtered
    else:
        new_payload.pop("tools", None)
        new_payload.pop("tool_choice", None)
    return new_payload


def apply_web_search_fallback(payload: Dict, endpoint: str) -> Dict:
    """Search locally, inject the results, and remove web_search tools."""
    from .counters import counters
    counters.incr("mod.web_search_fallback")
    query = extract_search_query(payload)
    print(f"[WebSearch] Extracted search query: {query!r}")

    results = call_search_proxy(query, endpoint)
    print(f"[WebSearch] Got {len(results)} search results")

    formatted = format_search_results(query, results)
    payload = inject_search_results_into_payload(payload, formatted)
    payload = remove_web_search_tools(payload)
    return payload
