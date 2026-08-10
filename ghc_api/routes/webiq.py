"""
Web IQ search endpoint.

Exposes the server-held Web IQ key as a callable search API so that clients
can execute a ``webiq_search`` tool call without ever seeing the key.

Every call is recorded (request + response) through ``webiq_log``: appended to
a daily ``.jl`` file and kept in the in-memory ring buffer the dashboard reads.
"""

import time
from typing import Any, Dict, List, Optional

from flask import Blueprint, g, jsonify, request

from ..auth import ANONYMOUS_USER_ID
from ..counters import counters
from ..state import state
from ..utils import get_client_ip
from ..webiq import WebIQError, normalize_query, search
from ..webiq_log import STATE_COMPLETED, STATE_ERROR, webiq_log

webiq_bp = Blueprint("webiq", __name__)


def _record_search(
    *,
    started: float,
    request_body: Any,
    query: Optional[str],
    trace: Dict[str, Any],
    results: Optional[List[Dict[str, Any]]] = None,
    status_code: int = 200,
    error: Optional[str] = None,
) -> None:
    """Append one search to the Web IQ log. Never raises into the request path."""
    try:
        webiq_log.record({
            "timestamp": int(started),
            "endpoint": "/v1/webiq/search",
            "client_ip": get_client_ip(request),
            "user_id": getattr(g, "user_id", ANONYMOUS_USER_ID) or ANONYMOUS_USER_ID,
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
            "duration_ms": int((time.time() - started) * 1000),
            "state": STATE_COMPLETED if error is None else STATE_ERROR,
        })
    except Exception as exc:  # pragma: no cover - logging must never break search
        print(f"[WebIQ] failed to record search: {exc}")


@webiq_bp.route("/v1/webiq/search", methods=["POST"])
def webiq_search():
    """Run one web search and return normalized results.

    Request:  {"query": "...", "max_results": 5}
    Response: {"query": "...", "results": [{"title", "url", "content"}]}
    """
    started = time.time()
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        message = "Request body must be a JSON object."
        _record_search(
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
