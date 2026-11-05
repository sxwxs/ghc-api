"""Three native AnySearch REST APIs with key custody and safe per-service audit."""

import time
import uuid

from flask import Blueprint, Response, g, request

from ..anysearch import (
    API_PATHS, EXTRACT_PATH, SEARCH_PATH, SUB_DOMAINS_PATH, decode_body,
    is_success, request_api, result_count, safe_query_string,
)
from ..anysearch_log import record_search_to_file
from ..auth import ANONYMOUS_USER_ID, redact_auth_headers
from ..cache import cache
from ..counters import counters
from ..state import state
from ..utils import get_client_ip

anysearch_bp = Blueprint("anysearch", __name__)


@anysearch_bp.route(SEARCH_PATH, methods=["POST"])
@anysearch_bp.route(EXTRACT_PATH, methods=["POST"])
@anysearch_bp.route(SUB_DOMAINS_PATH, methods=["GET"])
def anysearch_search():
    started = time.time()
    clock = time.monotonic()
    request_id = str(uuid.uuid4())
    api_path = request.path
    service = API_PATHS[api_path]
    model = f"anysearch_{service}"
    raw_request = request.get_data() if request.method == "POST" else b""
    query = safe_query_string(request.query_string) if api_path == SUB_DOMAINS_PATH else b""
    # The transport is the security boundary. Raw 402 bodies must never reach
    # Flask, its logs, the shared cache, exports, or the browser/model history.
    response = request_api(raw_request, state, api_path=api_path, local_request_id=request_id,
                           content_type=request.headers.get("Content-Type"),
                           content_encoding=request.headers.get("Content-Encoding"), query_string=query)
    body = decode_body(response.content)
    success = is_success(response.status_code, body, api_path)
    counters.incr(f"anysearch.{service}" + ("" if success else "_error"))
    duration = time.monotonic() - clock
    request_body = decode_body(raw_request) if raw_request else None
    count = result_count(body, api_path) if success else None
    client_ip = get_client_ip(request)
    user_id = getattr(g, "user_id", ANONYMOUS_USER_ID) or ANONYMOUS_USER_ID
    entry = {
        "id": request_id,
        "timestamp": int(started),
        "type": model,
        "endpoint": api_path,
        "client_ip": client_ip,
        "user_id": user_id,
        "query": request_body.get("query") if isinstance(request_body, dict) else None,
        "url": request_body.get("url") if isinstance(request_body, dict) else None,
        "request_query": query.decode("utf-8", errors="replace"),
        # Capability discovery is an API request, but not search quota usage.
        "uses_search_quota": api_path != SUB_DOMAINS_PATH,
        "request_body": request_body,
        "response_body": body,
        "status_code": response.status_code,
        "upstream": {"status_code": response.upstream_status,
                     "request_id": response.upstream_request_id},
        "business_code": body.get("code") if isinstance(body, dict) else None,
        "result_count": count,
        "duration_ms": int(duration * 1000),
        "state": "completed" if success else "error",
    }
    try:
        record_search_to_file(entry)
    except Exception:
        print("[AnySearch] Failed to record request")
    try:
        cache.add_request(request_id, {
            "client_ip": client_ip,
            "user_id": user_id,
            "request_headers": redact_auth_headers(dict(request.headers)),
            "request_body": request_body,
            "response_body": body,
            "model": model,
            "endpoint": api_path,
            "request_query": entry["request_query"],
            "uses_search_quota": entry["uses_search_quota"],
            "status_code": response.status_code,
            "state": entry["state"],
            "upstream_status": response.upstream_status,
            "upstream_request_id": response.upstream_request_id,
            "business_code": entry["business_code"],
            "result_count": count,
            "request_size": len(raw_request) + len(query),
            "response_size": len(response.content),
            "duration": round(duration, 3),
            "input_tokens": 0,
            "output_tokens": 0,
        })
    except Exception:
        print("[AnySearch] Failed to add request to cache")
    print(f"[AnySearch] service={service} request_id={request_id} status={response.status_code} "
          f"success={success} results={count} duration_ms={int(duration * 1000)}")
    return Response(response.content, status=response.status_code, headers=response.headers)
