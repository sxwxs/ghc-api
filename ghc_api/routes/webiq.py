"""
Web IQ search endpoint.

Exposes the server-held Web IQ key as a callable search API so that clients
can execute a ``webiq_search`` tool call without ever seeing the key.
"""

import time

from flask import Blueprint, jsonify, request

from ..counters import counters
from ..state import state
from ..webiq import WebIQError, search

webiq_bp = Blueprint("webiq", __name__)


@webiq_bp.route("/v1/webiq/search", methods=["POST"])
def webiq_search():
    """Run one web search and return normalized results.

    Request:  {"query": "...", "max_results": 5}
    Response: {"query": "...", "results": [{"title", "url", "content"}]}
    """
    started = time.time()
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": {
            "message": "Request body must be a JSON object.",
            "type": "invalid_request_error",
        }}), 400

    try:
        results = search(
            payload.get("query"),
            state,
            max_results=payload.get("max_results"),
        )
    except WebIQError as exc:
        counters.incr("webiq.search_error")
        print(f"[WebIQ] search failed: {exc}")
        return jsonify({"error": {
            "message": str(exc),
            "type": "webiq_search_error",
        }}), exc.status_code

    counters.incr("webiq.search")
    elapsed = time.time() - started
    query = payload.get("query")
    print(f"[WebIQ] query={query!r} results={len(results)} in {elapsed:.2f}s")
    return jsonify({
        "query": query,
        "results": results,
        "elapsed_ms": int(elapsed * 1000),
    })
