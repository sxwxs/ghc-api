"""
Web IQ search endpoint.

Exposes the server-held Web IQ key as a callable search API so that clients
can execute a ``webiq_search`` tool call without ever seeing the key.

The path, the request body and the response body are the official Microsoft
Web Search v3 contract (``POST /v3/search/web``), verbatim. Point any client
written against ``api.microsoft.ai`` at this proxy by changing the base URL --
the same deal the OpenAI- and Anthropic-shaped endpoints in this project
offer. Server configuration supplies defaults and hard caps for the outgoing
request; the upstream response is returned untouched.

Every call is recorded twice: in the dedicated Web IQ log (a daily ``.jl``
file plus the ring buffer behind the dashboard's Web IQ panel) and in the
shared request cache, so searches appear in the request list alongside the
LLM requests.
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from flask import Blueprint, g, jsonify, request

from ..auth import ANONYMOUS_USER_ID, redact_auth_headers
from ..cache import cache
from ..counters import counters
from ..state import state
from ..utils import get_client_ip
from ..webiq import WebIQError, search, web_results
from ..webiq_log import STATE_COMPLETED, STATE_ERROR, webiq_log

webiq_bp = Blueprint("webiq", __name__)

SEARCH_PATH = "/v3/search/web"

# Shown as the "model" of a search in the request list, so Web IQ traffic is
# distinguishable at a glance from LLM traffic.
REQUEST_LIST_MODEL = "webiq_search"


def _record_search(
    *,
    request_id: str,
    started: float,
    request_body: Any,
    trace: Dict[str, Any],
    response_body: Optional[Dict[str, Any]] = None,
    status_code: int = 200,
    error: Optional[str] = None,
) -> None:
    """Record one search in both logs. Never raises into the request path.

    Two destinations on purpose:
      * webiq_log - the dedicated Web IQ log (daily .jl file plus the ring
        buffer behind the dashboard's Web IQ panel), which keeps the search
        specifics: query, upstream payload, upstream status.
      * cache - the shared request cache, so a search also shows up in the
        request list, full-text search, detail view and export next to the
        LLM requests.

    The query recorded is the one that actually went upstream (from ``trace``),
    which is also the one the client sent: normalization only collapses
    whitespace, and anything it would have to alter is rejected instead.
    """
    client_ip = get_client_ip(request)
    user_id = getattr(g, "user_id", ANONYMOUS_USER_ID) or ANONYMOUS_USER_ID
    duration = time.time() - started
    upstream_request = trace.get("request") or {}
    query = upstream_request.get("query") if isinstance(upstream_request, dict) else None
    logged_body: Dict[str, Any] = (
        {"error": {"message": error, "type": "webiq_search_error"}}
        if error is not None
        else (response_body or {})
    )
    results: List[Any] = [] if error is not None else web_results(response_body or {})

    try:
        webiq_log.record({
            "id": request_id,
            "timestamp": int(started),
            "endpoint": SEARCH_PATH,
            "client_ip": client_ip,
            "user_id": user_id,
            "query": query,
            # What the client asked for, and what actually went upstream. The
            # two differ whenever a default or a cap kicked in.
            "request_body": request_body,
            "upstream": {
                "endpoint": trace.get("endpoint"),
                "request": trace.get("request"),
                "status_code": trace.get("status_code"),
            },
            "results": results,
            "result_count": len(results),
            # Kept out of "results" so the dashboard keeps rendering a plain
            # list, but preserved because a trace id is what upstream support
            # asks for first.
            "trace_id": (response_body or {}).get("traceId") if response_body else None,
            "status_code": status_code,
            "error": error,
            "duration_ms": int(duration * 1000),
            "state": STATE_COMPLETED if error is None else STATE_ERROR,
        })
    except Exception as exc:  # pragma: no cover - logging must never break search
        print(f"[WebIQ] failed to record search: {exc}")

    try:
        cache.add_request(request_id, {
            "client_ip": client_ip,
            "request_headers": redact_auth_headers(dict(request.headers)),
            "request_body": request_body,
            "response_body": logged_body,
            "model": REQUEST_LIST_MODEL,
            "endpoint": SEARCH_PATH,
            "status_code": status_code,
            "request_size": len(json.dumps(request_body)) if request_body is not None else 0,
            "response_size": len(json.dumps(logged_body, default=str)),
            "duration": round(duration, 3),
            "user_id": user_id,
            # A search spends Web IQ quota, not model tokens; leaving these at
            # zero keeps the token statistics about the LLM only.
            "input_tokens": 0,
            "output_tokens": 0,
        })
    except Exception as exc:  # pragma: no cover
        print(f"[WebIQ] failed to add the search to the request cache: {exc}")


@webiq_bp.route(SEARCH_PATH, methods=["POST"])
def webiq_search():
    """Run one Web Search v3 request with the server-held key.

    Request and response are the official contract; see webiq_web.md. All
    validation, defaulting and capping happens on the way out, so the body
    returned here is exactly what upstream produced.
    """
    started = time.time()
    request_id = str(uuid.uuid4())
    payload = request.get_json(silent=True)

    trace: Dict[str, Any] = {}
    try:
        body = search(payload, state, trace=trace)
    except WebIQError as exc:
        counters.incr("webiq.search_error")
        print(f"[WebIQ] search failed: {exc}")
        _record_search(
            request_id=request_id,
            started=started,
            request_body=payload,
            trace=trace,
            status_code=exc.status_code,
            error=str(exc),
        )
        return jsonify({"error": {
            "message": str(exc),
            "type": "webiq_search_error",
        }}), exc.status_code

    counters.incr("webiq.search")
    elapsed = time.time() - started
    _record_search(
        request_id=request_id,
        started=started,
        request_body=payload,
        trace=trace,
        response_body=body,
    )
    results = web_results(body)
    query = (trace.get("request") or {}).get("query")
    print(f"[WebIQ] query={query!r} results={len(results)} in {elapsed:.2f}s")
    return jsonify(body)
