"""
Web IQ search endpoint.

Exposes the server-held Web IQ key as a callable search API so that clients
can execute a ``webiq_search`` tool call without ever seeing the key.

``POST /v3/search/web`` is the official Microsoft Web Search v3 contract:
https://webiq.microsoft.ai/documentation/api-reference/web/

It is a transparent proxy. The request body is forwarded as the raw bytes the
client sent, and the upstream status, headers and body are returned verbatim,
including error responses and ``Retry-After``. Point any client written
against ``api.microsoft.ai`` at this proxy by changing the base URL -- the
same deal the OpenAI- and Anthropic-shaped endpoints in this project offer.

What the proxy adds is key custody (the client's own ``x-apikey`` is ignored
and redacted, never forwarded), the optional user-token auth gate, and
logging. Each call is written to the shared request cache -- so it appears in
the request list, full-text search, detail view and export next to the LLM
requests, under the model name ``webiq_search`` -- and appended in full to
``<config_dir>/webiq/YYYY-MM-DD.jl``, which is the only untruncated copy.
"""

import json
import time
import uuid
from typing import Any, Optional, Tuple

from flask import Blueprint, Response, g, jsonify, request

from ..auth import ANONYMOUS_USER_ID, redact_auth_headers
from ..cache import cache
from ..counters import counters
from ..state import state
from ..utils import get_client_ip
from ..webiq import (
    WebIQError,
    endpoint_for,
    passthrough_headers,
    result_count,
    search,
)
from ..webiq_log import STATE_COMPLETED, STATE_ERROR, record_search_to_file

webiq_bp = Blueprint("webiq", __name__)

SEARCH_PATH = "/v3/search/web"

# Shown as the "model" of a search in the request list, so Web IQ traffic is
# distinguishable at a glance from LLM traffic.
REQUEST_LIST_MODEL = "webiq_search"


def _decode(raw: bytes) -> Any:
    """Best-effort view of a body for logging only.

    Parsed JSON when it is JSON, the decoded text otherwise. Logging must
    never depend on the body being well-formed -- the whole point of the
    transparent proxy is that upstream, not this server, decides that.
    """
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw.decode("utf-8", errors="replace")


def _record_search(
    *,
    request_id: str,
    started: float,
    request_bytes: bytes,
    status_code: int,
    response_bytes: bytes = b"",
    response_body: Any = None,
    upstream_status: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    """Record one search in both places. Never raises into the request path.

    Two destinations on purpose:
      * the daily .jl file - full fidelity, nothing truncated;
      * the shared request cache - so a search shows up in the request list,
        full-text search, detail view and export next to the LLM requests.
        That copy is subject to cache_max_request_size like any other entry.
    """
    client_ip = get_client_ip(request)
    user_id = getattr(g, "user_id", ANONYMOUS_USER_ID) or ANONYMOUS_USER_ID
    duration = time.time() - started
    request_body = _decode(request_bytes)
    query = request_body.get("query") if isinstance(request_body, dict) else None
    state_label = STATE_COMPLETED if status_code < 400 else STATE_ERROR

    entry = {
        "id": request_id,
        "timestamp": int(started),
        "type": "webiq_search",
        "endpoint": SEARCH_PATH,
        "client_ip": client_ip,
        "user_id": user_id,
        "query": query,
        "request_body": request_body,
        "upstream": {
            "endpoint": endpoint_for(state),
            # None when the request never reached upstream (unconfigured
            # server, connection failure), which is itself worth seeing.
            "status_code": upstream_status,
        },
        "response_body": response_body,
        "result_count": result_count(response_body),
        # Kept as its own field because a trace id is what upstream support
        # asks for first, and it should be greppable in the .jl file.
        "trace_id": response_body.get("traceId") if isinstance(response_body, dict) else None,
        "status_code": status_code,
        "error": error,
        "duration_ms": int(duration * 1000),
        "state": state_label,
    }

    try:
        record_search_to_file(entry)
    except Exception as exc:  # pragma: no cover - logging must never break search
        print(f"[WebIQ] failed to record search: {exc}")

    try:
        cache.add_request(request_id, {
            "client_ip": client_ip,
            "request_headers": redact_auth_headers(dict(request.headers)),
            "request_body": request_body,
            "response_body": response_body,
            "model": REQUEST_LIST_MODEL,
            "endpoint": SEARCH_PATH,
            "status_code": status_code,
            "request_size": len(request_bytes),
            "response_size": len(response_bytes),
            "duration": round(duration, 3),
            "user_id": user_id,
            # A search spends Web IQ quota, not model tokens; leaving these at
            # zero keeps the token statistics about the LLM only.
            "input_tokens": 0,
            "output_tokens": 0,
        })
    except Exception as exc:  # pragma: no cover
        print(f"[WebIQ] failed to add the search to the request cache: {exc}")


def _error_response(message: str, status_code: int) -> Tuple[Response, int]:
    return jsonify({"error": {
        "message": message,
        "type": "webiq_search_error",
    }}), status_code


@webiq_bp.route(SEARCH_PATH, methods=["POST"])
def webiq_search():
    """Proxy one Web Search v3 request using the server-held key.

    The body goes upstream as received and the upstream response comes back
    verbatim, so the contract is Microsoft's, not this project's:
    https://webiq.microsoft.ai/documentation/api-reference/web/
    """
    started = time.time()
    request_id = str(uuid.uuid4())
    request_bytes = request.get_data()

    try:
        upstream = search(
            request_bytes,
            state,
            content_type=request.headers.get("content-type", "application/json"),
        )
    except WebIQError as exc:
        # Only reached when there is no upstream response to pass through.
        counters.incr("webiq.search_error")
        print(f"[WebIQ] search failed: {exc}")
        response, status = _error_response(str(exc), exc.status_code)
        _record_search(
            request_id=request_id,
            started=started,
            request_bytes=request_bytes,
            status_code=status,
            response_bytes=response.get_data(),
            response_body=response.get_json(),
            error=str(exc),
        )
        return response, status

    response_bytes = upstream.content
    body = _decode(response_bytes)

    if upstream.ok:
        counters.incr("webiq.search")
    else:
        counters.incr("webiq.search_error")

    _record_search(
        request_id=request_id,
        started=started,
        request_bytes=request_bytes,
        status_code=upstream.status_code,
        response_bytes=response_bytes,
        response_body=body,
        upstream_status=upstream.status_code,
    )

    count = result_count(body)
    query = request.get_json(silent=True)
    query = query.get("query") if isinstance(query, dict) else None
    print(f"[WebIQ] query={query!r} status={upstream.status_code} "
          f"results={count if count is not None else '-'} "
          f"in {time.time() - started:.2f}s")

    return Response(
        response_bytes,
        status=upstream.status_code,
        headers=passthrough_headers(upstream.headers),
    )
