"""
Recording of Web IQ searches to disk.

Every call to one of the six Web IQ REST endpoints is appended as one JSON
Lines record to ``<config_dir>/webiq/YYYY-MM-DD.jl`` (on by default, controlled
by ``state.log_webiq_requests``). Failures are recorded too. MCP calls append
method, status, duration, session and user metadata, but never their streaming
request or response bodies.

This file is the only full-fidelity record of a REST call. Each REST call and
MCP metadata record is also added to the shared request cache, which feeds the
dashboard, request list, full-text search and export. The cache replaces any
REST body over ``cache_max_request_size`` with a placeholder, both in memory
and in ``requests/YYYY-MM-DD.jl``. Nothing in this Web IQ log is truncated.

There is deliberately no in-memory buffer: keeping a second copy of every
search in RAM next to the cache's copy bought nothing but memory.

The Web IQ API key is never part of a record. The server's key travels in a
header that is never logged, and a client's own ``x-apikey`` is redacted by
``auth.redact_auth_headers`` before the request headers reach the cache.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict

STATE_COMPLETED = "completed"
STATE_ERROR = "error"
STATE_CANCELLED = "cancelled"


def log_dir() -> str:
    """Directory holding the daily Web IQ search logs."""
    from .utils import get_config_dir
    return os.path.join(get_config_dir(), "webiq")


def format_jsonl_line(entry: Dict[str, Any]) -> str:
    return json.dumps(entry, ensure_ascii=False, default=str) + "\n"


def record_search_to_file(entry: Dict[str, Any]) -> None:
    """Append one Web IQ REST or body-free MCP record to today's log file.

    Recording must never break a request that already succeeded, so disk
    failures are reported and swallowed.
    """
    try:
        from .state import state
        if not getattr(state, "log_webiq_requests", True):
            return

        directory = log_dir()
        os.makedirs(directory, exist_ok=True)
        daily_file = os.path.join(directory, f"{datetime.now().strftime('%Y-%m-%d')}.jl")
        with open(daily_file, "a", encoding="utf-8") as f:
            f.write(format_jsonl_line(entry))
    except Exception as exc:
        print(f"[WebIQ Logging] Failed to append search record: {exc}")
