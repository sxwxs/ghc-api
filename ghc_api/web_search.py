"""
Web search proxy fallback.

When the Copilot backend rejects a request containing the web_search tool,
this module calls an external search proxy, injects the results into the
system prompt, removes the web_search tool, and returns a modified payload
ready for retry.
"""

from typing import Any, Dict, List

import requests


def is_web_search_unsupported_error(status_code: int, response_text: str) -> bool:
    """Return True if the backend response indicates web_search is unsupported."""
    if status_code not in (400, 422):
        return False
    text_lower = response_text.lower()
    return "web search" in text_lower and ("unsupported" in text_lower or "not supported" in text_lower)


def has_web_search_tool(payload: Dict) -> bool:
    """Return True if the payload contains any web_search-type tool."""
    for tool in payload.get("tools") or []:
        tool_type = tool.get("type", "")
        if isinstance(tool_type, str) and tool_type.startswith("web_search"):
            return True
    return False


def extract_search_query(payload: Dict) -> str:
    """Extract a search query from the last user message."""
    for msg in reversed(payload.get("messages", [])):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content[:200].strip()
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            text = " ".join(parts).strip()
            return text[:200]
    return ""


def call_search_proxy(query: str, endpoint: str, limit: int = 5) -> List[Dict[str, Any]]:
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
        return data.get("results", [])
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
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.get('title', 'Untitled')}")
        if r.get("link"):
            lines.append(f"   URL: {r['link']}")
        if r.get("description"):
            lines.append(f"   {r['description']}")
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

    filtered = [t for t in tools if not (isinstance(t.get("type", ""), str) and t["type"].startswith("web_search"))]

    new_payload = dict(payload)
    if filtered:
        new_payload["tools"] = filtered
    else:
        new_payload.pop("tools", None)
        new_payload.pop("tool_choice", None)
    return new_payload


def apply_web_search_fallback(payload: Dict, endpoint: str) -> Dict:
    """Orchestrate the full web search fallback: search, inject results, remove tools."""
    query = extract_search_query(payload)
    print(f"[WebSearch] Extracted search query: {query!r}")

    results = call_search_proxy(query, endpoint)
    print(f"[WebSearch] Got {len(results)} search results")

    formatted = format_search_results(query, results)
    payload = inject_search_results_into_payload(payload, formatted)
    payload = remove_web_search_tools(payload)
    return payload
