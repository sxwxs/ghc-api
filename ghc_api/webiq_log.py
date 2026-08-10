"""
Recording of Web IQ searches.

Every call that reaches ``POST /v1/webiq/search`` is recorded twice:

* appended as one JSON Lines record to ``<config_dir>/webiq/YYYY-MM-DD.jl``
  (on by default, controlled by ``state.log_webiq_requests``), and
* kept in a small in-memory ring buffer (the most recent
  ``state.webiq_log_max_entries`` entries, 20 by default) so the dashboard can
  show recent searches without touching disk.

Failures are recorded too: a search that never reached upstream, or that came
back 429/502, is exactly the thing you want in the log. The Web IQ API key is
sent as a header and is never part of a recorded payload.

The files live in their own directory rather than under ``requests/`` so the
request-statistics indexer keeps seeing only LLM proxy traffic.
"""

import json
import os
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

# Ring-buffer size. Also the ceiling accepted from config, so a misconfigured
# value cannot turn this into an unbounded in-memory log.
DEFAULT_MAX_ENTRIES = 20
MAX_MEMORY_ENTRIES_LIMIT = 500

STATE_COMPLETED = "completed"
STATE_ERROR = "error"


class WebIQLog:
    """Thread-safe recorder for Web IQ search request/response pairs."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES):
        self.lock = threading.Lock()
        self._entries: Deque[Dict[str, Any]] = deque(maxlen=max_entries)
        self.total_count = 0
        self.error_count = 0

    @property
    def max_entries(self) -> int:
        return self._entries.maxlen or DEFAULT_MAX_ENTRIES

    def set_max_entries(self, max_entries: int) -> None:
        """Resize the ring buffer, keeping the newest entries."""
        size = max(1, min(MAX_MEMORY_ENTRIES_LIMIT, int(max_entries)))
        with self.lock:
            if size == self._entries.maxlen:
                return
            self._entries = deque(self._entries, maxlen=size)

    def record(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Record one search. Returns the stored entry (with id/timestamp filled in).

        Recording must never break a search that already succeeded, so disk
        failures are reported and swallowed.
        """
        stored = dict(entry)
        stored.setdefault("id", uuid.uuid4().hex)
        stored.setdefault("timestamp", int(time.time()))
        stored.setdefault("type", "webiq_search")
        stored.setdefault("state", STATE_ERROR if stored.get("error") else STATE_COMPLETED)

        with self.lock:
            self._entries.append(stored)
            self.total_count += 1
            if stored.get("state") == STATE_ERROR:
                self.error_count += 1

        self._append_to_daily_file(stored)
        return stored

    def recent(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return recorded searches, newest first."""
        with self.lock:
            items = list(reversed(self._entries))
        if limit is not None and limit >= 0:
            return items[:limit]
        return items

    def get(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Return one buffered entry by id, if it has not been evicted yet."""
        with self.lock:
            for entry in reversed(self._entries):
                if entry.get("id") == entry_id:
                    return dict(entry)
        return None

    def stats(self) -> Dict[str, int]:
        with self.lock:
            return {
                "total_searches": self.total_count,
                "failed_searches": self.error_count,
                "buffered": len(self._entries),
                "max_entries": self.max_entries,
            }

    def clear(self) -> None:
        with self.lock:
            self._entries.clear()
            self.total_count = 0
            self.error_count = 0

    @staticmethod
    def format_jsonl_line(entry: Dict[str, Any]) -> str:
        return json.dumps(entry, ensure_ascii=False, default=str) + "\n"

    def log_dir(self) -> str:
        from .utils import get_config_dir
        return os.path.join(get_config_dir(), "webiq")

    def _append_to_daily_file(self, entry: Dict[str, Any]) -> None:
        try:
            from .state import state
            if not getattr(state, "log_webiq_requests", True):
                return

            log_dir = self.log_dir()
            os.makedirs(log_dir, exist_ok=True)
            daily_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.jl")
            with open(daily_file, "a", encoding="utf-8") as f:
                f.write(self.format_jsonl_line(entry))
        except Exception as exc:
            print(f"[WebIQ Logging] Failed to append search record: {exc}")


# Global Web IQ log instance.
webiq_log = WebIQLog()
