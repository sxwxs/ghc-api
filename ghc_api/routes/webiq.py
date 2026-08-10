"""
Web IQ search endpoint.

Exposes the server-held Web IQ key as a callable search API so that clients
can execute a ``webiq_search`` tool call without ever seeing the key.

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
from ..webiq import WebIQError, normalize_query, search
from ..webiq_log import STATE_COMPLETED, STATE_ERROR, webiq_log

webiq_bp = Blueprint("webiq", __name__)

# Shown as the "model" of a search in the request list, so Web IQ traffic is
# distinguishable at a glance from LLM traffic.
REQUEST_LIST_MODEL = "webiq_search"


def _record_search(
    *,
    request_id: str,
    started: float,
    request_body: Any,
    query: Optional[str],
    trace: Dict[str, Any],
    results: Optional[List[Dict[str, Any]]] = None,
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
    """
    client_ip = get_client_ip(request)
    user_id = getattr(g, "user_id", ANONYMOUS_USER_ID) or ANONYMOUS_USER_ID
    duration = time.time() - started
    response_body: Dict[str, Any] = (
        {"error": {"message": error, "type": "webiq_search_error"}}
        if error is not None
        else {"query": query, "results": results or []}
    )

    try:
        webiq_log.record({
            "id": request_id,
            "timestamp": int(started),
            "endpoint": "/v1/webiq/search",
            "client_ip": client_ip,
            "user_id": user_id,
            "query": query,
            # What the client asked for, and what actually went upstream. The
            # two differ whenever normalization or clamping kicked in.
            "request_body": request_body,
            "upstream": {
                "endpoint": trace.get("endpoint"),
                "request": trace.get("request"),
                "status_code": trace.get("status_code"),
            },
            "results": results if results is not None else [],
            "result_count": len(results) if results is not None else 0,
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
            "response_body": response_body,
            "model": REQUEST_LIST_MODEL,
            "endpoint": "/v1/webiq/search",
            "status_code": status_code,
            "request_size": len(json.dumps(request_body)) if request_body is not None else 0,
            "response_size": len(json.dumps(response_body)),
            "duration": round(duration, 3),
            "user_id": user_id,
            # A search spends Web IQ quota, not model tokens; leaving these at
            # zero keeps the token statistics about the LLM only.
            "input_tokens": 0,
            "output_tokens": 0,
        })
    except Exception as exc:  # pragma: no cover
        print(f"[WebIQ] failed to add the search to the request cache: {exc}")


@webiq_bp.route("/v1/webiq/search", methods=["POST"])
def webiq_search():
    """Run one web search and return normalized results.

    Request:  {"query": "...", "max_results": 5}
    Response: {"query": "...", "results": [{"title", "url", "content"}]}
    """
    started = time.time()
    request_id = str(uuid.uuid4())
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        message = "Request body must be a JSON object."
        _record_search(
            request_id=request_id,
            started=started,
            request_body=payload,
            query=None,
            trace={},
            status_code=400,
            error=message,
        )
        return jsonify({"error": {
            "message": message,
            "type": "invalid_request_error",
        }}), 400

    # Normalize up front so the echoed query, the log line and the query that
    # actually reaches upstream are all the same bounded, single-line string.
    query = None
    trace: Dict[str, Any] = {}
    try:
        query = normalize_query(payload.get("query"))
        results = search(
            query,
            state,
            max_results=payload.get("max_results"),
            trace=trace,
        )
    except WebIQError as exc:
        counters.incr("webiq.search_error")
        print(f"[WebIQ] search failed: {exc}")
        _record_search(
            request_id=request_id,
            started=started,
            request_body=payload,
            query=query,
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
        query=query,
        trace=trace,
        results=results,
    )
    print(f"[WebIQ] query={query!r} results={len(results)} in {elapsed:.2f}s")
    return jsonify({
        "query": query,
        "results": results,
        "elapsed_ms": int(elapsed * 1000),
    })
