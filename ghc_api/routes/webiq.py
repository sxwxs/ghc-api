"""
Web IQ REST and MCP endpoints.

Exposes the server-held Web IQ key through all six official REST APIs and the
Streamable HTTP MCP server. Clients can call Web IQ without seeing the key.

The official Web IQ v3 contracts are documented at:
https://webiq.microsoft.ai/documentation/api-reference/

It is a transparent proxy. The request body is forwarded as the raw bytes the
client sent, and the upstream status, headers and body are returned verbatim,
including error responses and ``Retry-After``. Point any client written
against ``api.microsoft.ai`` at this proxy by changing the base URL -- the
same deal the OpenAI- and Anthropic-shaped endpoints in this project offer.

What the proxy adds is key custody (the client's own ``x-apikey`` is ignored
and redacted, never forwarded), the optional user-token auth gate, and
logging. Each REST call is written to the shared request cache under a
service-specific model name and appended in full to
``<config_dir>/webiq/YYYY-MM-DD.jl``, which is the only untruncated copy.
Streaming MCP bodies are not persisted.
"""

import json
import threading
import time
import uuid
from typing import Any, Optional, Tuple

from flask import Blueprint, Response, g, jsonify, request, stream_with_context

from ..auth import ANONYMOUS_USER_ID, redact_auth_headers
from ..cache import cache
from ..counters import counters
from ..state import state
from ..utils import get_client_ip
from ..webiq import (
    API_PATHS,
    MCP_PATH,
    WEB_PATH,
    WebIQError,
    endpoint_for,
    mcp_request,
    passthrough_headers,
    result_count,
    search,
)
from ..webiq_log import (
    STATE_CANCELLED,
    STATE_COMPLETED,
    STATE_ERROR,
    record_search_to_file,
)

webiq_bp = Blueprint("webiq", __name__)

# Compatibility names retained for the original web-only route.
SEARCH_PATH = WEB_PATH
REQUEST_LIST_MODEL = "webiq_search"


class _MCPStreamLimiter:
    """Non-blocking process-local cap for thread-occupying MCP streams."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0

    def try_acquire(self, limit: int) -> bool:
        with self._lock:
            if self._active >= max(1, int(limit)):
                return False
            self._active += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._active > 0:
                self._active -= 1

    def active(self) -> int:
        with self._lock:
            return self._active


_mcp_stream_limiter = _MCPStreamLimiter()


def _request_list_model(api_path: str) -> str:
    service = API_PATHS[api_path]
    return REQUEST_LIST_MODEL if api_path == WEB_PATH else f"webiq_{service}"


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
    api_path: str = WEB_PATH,
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
    url = (
        request_body.get("url")
        if api_path == "/v3/browse" and isinstance(request_body, dict)
        else None
    )
    state_label = STATE_COMPLETED if status_code < 400 else STATE_ERROR
    model_name = _request_list_model(api_path)

    entry = {
        "id": request_id,
        "timestamp": int(started),
        "type": model_name,
        "endpoint": api_path,
        "client_ip": client_ip,
        "user_id": user_id,
        "query": query,
        "url": url,
        "request_body": request_body,
        "upstream": {
            "endpoint": _audit_endpoint(api_path),
            # None when the request never reached upstream (unconfigured
            # server, connection failure), which is itself worth seeing.
            "status_code": upstream_status,
        },
        "response_body": response_body,
        "result_count": result_count(response_body, api_path),
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
            "model": model_name,
            "endpoint": api_path,
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


def _audit_endpoint(api_path: str) -> Optional[str]:
    """Best-effort upstream URL for audit records; configuration errors are data."""
    try:
        return endpoint_for(state, api_path)
    except ValueError:
        return None


def _error_response(
    message: str,
    status_code: int,
    service: str,
) -> Tuple[Response, int]:
    return jsonify({"error": {
        "message": message,
        "type": f"webiq_{service}_error",
    }}), status_code


@webiq_bp.route("/v3/search/classic", methods=["POST"])
@webiq_bp.route("/v3/search/images", methods=["POST"])
@webiq_bp.route("/v3/search/news", methods=["POST"])
@webiq_bp.route("/v3/browse", methods=["POST"])
@webiq_bp.route("/v3/search/videos", methods=["POST"])
@webiq_bp.route(SEARCH_PATH, methods=["POST"])
def webiq_search():
    """Transparently proxy one of the six Web IQ v3 REST APIs."""
    started = time.time()
    request_id = str(uuid.uuid4())
    request_bytes = request.get_data()
    api_path = request.path
    service = API_PATHS[api_path]
    counter_name = "search" if api_path == WEB_PATH else service

    try:
        upstream = search(
            request_bytes,
            state,
            content_type=request.headers.get("content-type", "application/json"),
            api_path=api_path,
        )
    except WebIQError as exc:
        # Only reached when there is no upstream response to pass through.
        counters.incr(f"webiq.{counter_name}_error")
        print(f"[WebIQ] {service} failed: {exc}")
        error_service = "search" if api_path == WEB_PATH else service
        response, status = _error_response(
            str(exc), exc.status_code, error_service)
        _record_search(
            request_id=request_id,
            started=started,
            request_bytes=request_bytes,
            status_code=status,
            response_bytes=response.get_data(),
            response_body=response.get_json(),
            error=str(exc),
            api_path=api_path,
        )
        return response, status

    response_bytes = upstream.content
    body = _decode(response_bytes)

    if upstream.ok:
        counters.incr(f"webiq.{counter_name}")
    else:
        counters.incr(f"webiq.{counter_name}_error")

    _record_search(
        request_id=request_id,
        started=started,
        request_bytes=request_bytes,
        status_code=upstream.status_code,
        response_bytes=response_bytes,
        response_body=body,
        upstream_status=upstream.status_code,
        api_path=api_path,
    )

    count = result_count(body, api_path)
    request_view = request.get_json(silent=True)
    identifier_name = "url" if api_path == "/v3/browse" else "query"
    identifier = (
        request_view.get(identifier_name)
        if isinstance(request_view, dict) else None
    )
    print(f"[WebIQ] service={service} {identifier_name}={identifier!r} "
          f"status={upstream.status_code} "
          f"results={count if count is not None else '-'} "
          f"in {time.time() - started:.2f}s")

    return Response(
        response_bytes,
        status=upstream.status_code,
        headers=passthrough_headers(upstream.headers),
    )


def _header_value(headers: Any, name: str) -> Optional[str]:
    if not headers:
        return None
    wanted = name.lower()
    for key, value in headers.items():
        if key.lower() == wanted:
            return value
    return None


def _start_mcp_audit(body: bytes, headers: dict) -> dict:
    """Create a pending cache entry without retaining the MCP body."""
    audit_headers = redact_auth_headers(dict(request.headers))
    # Mcp-Param-* values mirror selected tool arguments from the body. Keeping
    # them would defeat the body-free audit promise, so retain their presence
    # but not their value.
    for name in list(audit_headers):
        if name.lower().startswith("mcp-param-"):
            audit_headers[name] = "***REDACTED***"

    audit = {
        "id": str(uuid.uuid4()),
        "started": time.time(),
        "client_ip": get_client_ip(request),
        "user_id": getattr(g, "user_id", ANONYMOUS_USER_ID) or ANONYMOUS_USER_ID,
        "http_method": request.method,
        "mcp_method": headers.get("mcp-method"),
        "mcp_name": headers.get("mcp-name"),
        "request_session_id": headers.get("mcp-session-id"),
        "request_size": len(body),
        "request_headers": audit_headers,
    }
    request_summary = {
        "body_logged": False,
        "http_method": audit["http_method"],
        "mcp_method": audit["mcp_method"],
        "mcp_name": audit["mcp_name"],
        "mcp_session_id": audit["request_session_id"],
    }
    audit["request_summary"] = request_summary
    try:
        cache.start_request(audit["id"], {
            "client_ip": audit["client_ip"],
            "request_headers": audit["request_headers"],
            "request_body": request_summary,
            "model": "webiq_mcp",
            "endpoint": MCP_PATH,
            "request_size": audit["request_size"],
            "user_id": audit["user_id"],
        })
    except Exception as exc:  # pragma: no cover - auditing must not break MCP
        print(f"[WebIQ] failed to start MCP audit record: {exc}")
    return audit


def _complete_mcp_audit(
    audit: dict,
    *,
    status_code: int,
    upstream_status: Optional[int],
    response_size: int,
    response_session_id: Optional[str],
    stream_completed: bool,
    completion_state: str,
    error: Optional[str] = None,
) -> None:
    """Complete a body-free MCP audit record in the file and request cache."""
    duration = time.time() - audit["started"]
    response_summary = {
        "body_logged": False,
        "stream_completed": stream_completed,
        "stream_state": completion_state,
        "mcp_session_id": response_session_id,
        "error": error,
    }
    entry = {
        "id": audit["id"],
        "timestamp": int(audit["started"]),
        "type": "webiq_mcp",
        "endpoint": MCP_PATH,
        "client_ip": audit["client_ip"],
        "user_id": audit["user_id"],
        "method": audit["http_method"],
        "mcp_method": audit["mcp_method"],
        "mcp_name": audit["mcp_name"],
        "request_session_id": audit["request_session_id"],
        "response_session_id": response_session_id,
        "request_size": audit["request_size"],
        "response_size": response_size,
        "body_logged": False,
        "stream_completed": stream_completed,
        "upstream": {
            "endpoint": _audit_endpoint(MCP_PATH),
            "status_code": upstream_status,
        },
        "status_code": status_code,
        "error": error,
        "duration_ms": int(duration * 1000),
        "state": completion_state,
    }
    try:
        record_search_to_file(entry)
    except Exception as exc:  # pragma: no cover
        print(f"[WebIQ] failed to record MCP audit: {exc}")
    try:
        cache.complete_request(audit["id"], {
            "request_body": audit["request_summary"],
            "response_body": response_summary,
            "status_code": status_code,
            "request_size": audit["request_size"],
            "response_size": response_size,
            "duration": round(duration, 3),
            "user_id": audit["user_id"],
            "stream_completed": stream_completed,
            "stream_state": completion_state,
            "error": error,
            # Override RequestCache's status-only default: an HTTP 200 stream
            # can still be truncated or cancelled after headers were sent.
            "state": completion_state,
        })
    except Exception as exc:  # pragma: no cover
        print(f"[WebIQ] failed to complete MCP cache record: {exc}")


@webiq_bp.route(MCP_PATH, methods=["GET", "POST", "DELETE"])
def webiq_mcp():
    """Transparently proxy the Web IQ Streamable HTTP MCP transport."""
    body = request.get_data()
    headers = {name.lower(): value for name, value in request.headers.items()}
    audit = _start_mcp_audit(body, headers)

    if not _mcp_stream_limiter.try_acquire(
        state.webiq_mcp_max_concurrent_streams
    ):
        counters.incr("webiq.mcp_error")
        message = (
            "Web IQ MCP concurrency limit reached; retry after an active "
            "stream closes."
        )
        response, status = _error_response(message, 503, "mcp")
        response.headers["Retry-After"] = "1"
        _complete_mcp_audit(
            audit,
            status_code=status,
            upstream_status=None,
            response_size=len(response.get_data()),
            response_session_id=None,
            stream_completed=False,
            completion_state=STATE_ERROR,
            error=message,
        )
        return response, status

    try:
        upstream = mcp_request(
            request.method,
            body,
            state,
            request_headers=headers,
        )
    except WebIQError as exc:
        _mcp_stream_limiter.release()
        counters.incr("webiq.mcp_error")
        response, status = _error_response(str(exc), exc.status_code, "mcp")
        _complete_mcp_audit(
            audit,
            status_code=status,
            upstream_status=None,
            response_size=len(response.get_data()),
            response_session_id=None,
            stream_completed=False,
            completion_state=STATE_ERROR,
            error=str(exc),
        )
        return response, status
    except Exception as exc:  # pragma: no cover - defensive slot cleanup
        _mcp_stream_limiter.release()
        counters.incr("webiq.mcp_error")
        message = f"Web IQ MCP proxy failed: {type(exc).__name__}"
        response, status = _error_response(message, 502, "mcp")
        _complete_mcp_audit(
            audit,
            status_code=status,
            upstream_status=None,
            response_size=len(response.get_data()),
            response_session_id=None,
            stream_completed=False,
            completion_state=STATE_ERROR,
            error=message,
        )
        return response, status

    response_session_id = _header_value(upstream.headers, "mcp-session-id")

    @stream_with_context
    def generate():
        response_size = 0
        stream_completed = False
        completion_state = STATE_ERROR
        stream_error = None
        try:
            for chunk in upstream.iter_content(chunk_size=64 * 1024):
                if chunk:
                    response_size += len(chunk)
                    yield chunk
            stream_completed = True
            if upstream.ok:
                completion_state = STATE_COMPLETED
            else:
                stream_error = f"Upstream MCP returned HTTP {upstream.status_code}"
        except GeneratorExit:
            completion_state = STATE_CANCELLED
            stream_error = "Client disconnected before the MCP stream completed"
            raise
        except Exception as exc:
            completion_state = STATE_ERROR
            detail = str(exc).strip()
            stream_error = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
            if len(stream_error) > 500:
                stream_error = stream_error[:500] + "..."
            raise
        finally:
            try:
                upstream.close()
            except Exception as exc:  # pragma: no cover - requests close is normally safe
                if completion_state == STATE_COMPLETED:
                    completion_state = STATE_ERROR
                    stream_completed = False
                    stream_error = f"Failed to close upstream MCP stream: {type(exc).__name__}"
            _mcp_stream_limiter.release()
            if completion_state == STATE_COMPLETED:
                counters.incr("webiq.mcp")
            elif completion_state == STATE_CANCELLED:
                counters.incr("webiq.mcp_cancelled")
            else:
                counters.incr("webiq.mcp_error")
            _complete_mcp_audit(
                audit,
                status_code=upstream.status_code,
                upstream_status=upstream.status_code,
                response_size=response_size,
                response_session_id=response_session_id,
                stream_completed=stream_completed,
                completion_state=completion_state,
                error=stream_error,
            )

    response = Response(
        generate(),
        status=upstream.status_code,
        headers=passthrough_headers(upstream.headers),
        direct_passthrough=True,
    )
    content_type = _header_value(upstream.headers, "content-type") or ""
    if content_type.split(";", 1)[0].strip().lower() == "text/event-stream":
        response.headers["X-Accel-Buffering"] = "no"
        response.headers["Cache-Control"] = "no-cache"
    return response
