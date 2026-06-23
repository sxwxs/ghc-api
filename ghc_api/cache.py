"""
Thread-safe cache for storing API requests and responses
"""

import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


ANONYMOUS_USER_ID = "anonymous"


def _coerce_user_id(value: Any) -> str:
    """Normalize a user_id value to a non-empty string, defaulting to ANONYMOUS."""
    if value is None:
        return ANONYMOUS_USER_ID
    text = str(value).strip()
    return text if text else ANONYMOUS_USER_ID


class RequestCache:
    """Thread-safe cache for storing API requests and responses"""

    # Request states
    STATE_PENDING = "pending"
    STATE_SENDING = "sending"
    STATE_RECEIVING = "receiving"
    STATE_COMPLETED = "completed"
    STATE_ERROR = "error"

    def __init__(self, max_entries: int = 1000, max_request_size: int = 1024 * 1024):
        self.max_entries = max_entries
        self.max_request_size = max_request_size
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.Lock()
        self.request_count = 0
        self.bytes_sent = 0
        self.bytes_received = 0
        self.model_stats: Dict[str, Dict] = {}
        self.endpoint_stats: Dict[str, Dict] = {}
        # Per-user accumulators. Same shape as model_stats / endpoint_stats but
        # nested under a user_id key. The global model_stats / endpoint_stats
        # above are kept in parallel so the dashboard's "all users" view stays
        # cheap (no per-user aggregation on read).
        self.user_model_stats: Dict[str, Dict[str, Dict]] = {}
        self.user_endpoint_stats: Dict[str, Dict[str, Dict]] = {}
        self.user_totals: Dict[str, Dict[str, int]] = {}

    @staticmethod
    def _current_timestamp() -> int:
        return int(time.time())

    def _truncate_oversize_bodies(self, entry: Dict[str, Any]) -> None:
        """Replace request/response bodies with a placeholder when they exceed the configured size limit."""
        if not self.max_request_size or self.max_request_size <= 0:
            return
        limit = self.max_request_size
        if entry.get("request_size", 0) and entry["request_size"] > limit:
            placeholder = {"_truncated": True, "_size": entry["request_size"], "_reason": f"request body exceeded cache_max_request_size ({limit} bytes)"}
            entry["request_body"] = placeholder
            entry["original_request_body"] = placeholder
        if entry.get("response_size", 0) and entry["response_size"] > limit:
            placeholder = {"_truncated": True, "_size": entry["response_size"], "_reason": f"response body exceeded cache_max_request_size ({limit} bytes)"}
            if entry.get("raw_events") is not None:
                entry["raw_events"] = [json.dumps(placeholder)]
            else:
                entry["response_body"] = placeholder

    @classmethod
    def _normalize_import_timestamp(cls, value: Any) -> int:
        """Normalize imported timestamps to Unix seconds."""
        if isinstance(value, (int, float)):
            return int(value)

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return cls._current_timestamp()

            try:
                return int(float(stripped))
            except ValueError:
                try:
                    return int(datetime.fromisoformat(stripped.replace("Z", "+00:00")).timestamp())
                except ValueError:
                    return cls._current_timestamp()

        return cls._current_timestamp()

    def start_request(self, request_id: str, data: Dict) -> None:
        """Start tracking a new request (before sending to upstream)"""
        with self.lock:
            if len(self.cache) >= self.max_entries:
                self.cache.popitem(last=False)

            self.cache[request_id] = {
                "id": request_id,
                "timestamp": self._current_timestamp(),
                "client_ip": data.get("client_ip"),
                "request_headers": data.get("request_headers"),
                "original_request_body": data.get("original_request_body"),
                "request_body": data.get("request_body"),
                "response_body": None,
                "raw_events": None,
                "model": data.get("model", "unknown"),
                "translated_model": data.get("translated_model"),
                "endpoint": data.get("endpoint", "unknown"),
                "status_code": None,
                "request_size": data.get("request_size", 0),
                "response_size": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "duration": 0,
                "state": self.STATE_PENDING,
                "user_id": _coerce_user_id(data.get("user_id")),
            }
            self._truncate_oversize_bodies(self.cache[request_id])

    def update_request_state(self, request_id: str, state: str, **kwargs) -> None:
        """Update the state and optional fields of an existing request"""
        with self.lock:
            if request_id in self.cache:
                self.cache[request_id]["state"] = state
                for key, value in kwargs.items():
                    if key in self.cache[request_id]:
                        self.cache[request_id][key] = value

    def complete_request(self, request_id: str, data: Dict) -> None:
        """Complete a request with response data and update statistics"""
        entry_snapshot = None
        with self.lock:
            if request_id in self.cache:
                # Update existing entry
                entry = self.cache[request_id]
                if "client_ip" in data:
                    entry["client_ip"] = data.get("client_ip")
                if "request_headers" in data:
                    entry["request_headers"] = data.get("request_headers")
                if "original_request_body" in data:
                    entry["original_request_body"] = data.get("original_request_body")
                if "request_body" in data:
                    entry["request_body"] = data.get("request_body")
                if "raw_events" in data:
                    entry["raw_events"] = data.get("raw_events")
                    entry["response_body"] = None
                else:
                    entry["response_body"] = data.get("response_body")
                entry["status_code"] = data.get("status_code", 200)
                entry["request_size"] = data.get("request_size", entry.get("request_size", 0))
                entry["response_size"] = data.get("response_size", 0)
                entry["input_tokens"] = data.get("input_tokens", 0)
                entry["output_tokens"] = data.get("output_tokens", 0)
                entry["cache_creation_input_tokens"] = data.get("cache_creation_input_tokens", 0)
                entry["cache_read_input_tokens"] = data.get("cache_read_input_tokens", 0)
                entry["duration"] = data.get("duration", 0)
                entry["state"] = self.STATE_COMPLETED if data.get("status_code", 200) < 400 else self.STATE_ERROR
                # Defensive: if user_id wasn't set during start_request (legacy
                # path or external caller forgot), fall back to anonymous.
                if not entry.get("user_id"):
                    entry["user_id"] = _coerce_user_id(data.get("user_id"))
                # Pass through any unrecognized keys so subclasses can surface
                # sidecar fields (e.g. recovered_content for tool-call recovery).
                _KNOWN = {
                    "client_ip", "request_headers", "original_request_body",
                    "request_body", "response_body", "raw_events", "status_code",
                    "request_size", "response_size", "input_tokens", "output_tokens",
                    "cache_creation_input_tokens", "cache_read_input_tokens",
                    "duration", "user_id", "model", "translated_model", "endpoint",
                }
                for key, value in data.items():
                    if key not in _KNOWN:
                        entry[key] = value
                self._truncate_oversize_bodies(entry)
            else:
                # Fallback: create new entry if somehow missing
                if len(self.cache) >= self.max_entries:
                    self.cache.popitem(last=False)

                self.cache[request_id] = {
                    "id": request_id,
                    "timestamp": self._current_timestamp(),
                    "client_ip": data.get("client_ip"),
                    "request_headers": data.get("request_headers"),
                    "original_request_body": data.get("original_request_body"),
                    "request_body": data.get("request_body"),
                    "response_body": data.get("response_body") if "raw_events" not in data else None,
                    "raw_events": data.get("raw_events"),
                    "model": data.get("model", "unknown"),
                    "translated_model": data.get("translated_model"),
                    "endpoint": data.get("endpoint", "unknown"),
                    "status_code": data.get("status_code", 200),
                    "request_size": data.get("request_size", 0),
                    "response_size": data.get("response_size", 0),
                    "input_tokens": data.get("input_tokens", 0),
                    "output_tokens": data.get("output_tokens", 0),
                    "cache_creation_input_tokens": data.get("cache_creation_input_tokens", 0),
                    "cache_read_input_tokens": data.get("cache_read_input_tokens", 0),
                    "duration": data.get("duration", 0),
                    "state": self.STATE_COMPLETED if data.get("status_code", 200) < 400 else self.STATE_ERROR,
                    "user_id": _coerce_user_id(data.get("user_id")),
                }
                self._truncate_oversize_bodies(self.cache[request_id])

            self.request_count += 1
            self.bytes_sent += data.get("request_size", 0)
            self.bytes_received += data.get("response_size", 0)

            user_id = _coerce_user_id(self.cache[request_id].get("user_id"))

            # Update model stats using translated name when available.
            model = data.get("translated_model") or data.get("model", "unknown")
            if model not in self.model_stats:
                self.model_stats[model] = {
                    "request_count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "bytes_sent": 0,
                    "bytes_received": 0,
                }
            self.model_stats[model]["request_count"] += 1
            self.model_stats[model]["input_tokens"] += data.get("input_tokens", 0)
            self.model_stats[model]["output_tokens"] += data.get("output_tokens", 0)
            self.model_stats[model]["cache_creation_input_tokens"] += data.get("cache_creation_input_tokens", 0)
            self.model_stats[model]["cache_read_input_tokens"] += data.get("cache_read_input_tokens", 0)
            self.model_stats[model]["bytes_sent"] += data.get("request_size", 0)
            self.model_stats[model]["bytes_received"] += data.get("response_size", 0)

            # Update endpoint stats
            endpoint = data.get("endpoint", "unknown")
            if endpoint not in self.endpoint_stats:
                self.endpoint_stats[endpoint] = {
                    "request_count": 0,
                    "bytes_sent": 0,
                    "bytes_received": 0,
                }
            self.endpoint_stats[endpoint]["request_count"] += 1
            self.endpoint_stats[endpoint]["bytes_sent"] += data.get("request_size", 0)
            self.endpoint_stats[endpoint]["bytes_received"] += data.get("response_size", 0)

            # Per-user accumulators (kept in parallel to the global ones above).
            self._bump_user_stats(
                user_id=user_id,
                model=model,
                endpoint=endpoint,
                input_tokens=data.get("input_tokens", 0),
                output_tokens=data.get("output_tokens", 0),
                cache_creation_input_tokens=data.get("cache_creation_input_tokens", 0),
                cache_read_input_tokens=data.get("cache_read_input_tokens", 0),
                bytes_sent=data.get("request_size", 0),
                bytes_received=data.get("response_size", 0),
            )

            entry_snapshot = dict(self.cache[request_id])

        if entry_snapshot:
            self._append_request_to_daily_file(entry_snapshot)

    def _bump_user_stats(
        self,
        user_id: str,
        model: str,
        endpoint: str,
        input_tokens: int,
        output_tokens: int,
        cache_creation_input_tokens: int,
        cache_read_input_tokens: int,
        bytes_sent: int,
        bytes_received: int,
    ) -> None:
        """Increment the per-user accumulators. Called from within self.lock."""
        models_for_user = self.user_model_stats.setdefault(user_id, {})
        if model not in models_for_user:
            models_for_user[model] = {
                "request_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "bytes_sent": 0,
                "bytes_received": 0,
            }
        m = models_for_user[model]
        m["request_count"] += 1
        m["input_tokens"] += input_tokens
        m["output_tokens"] += output_tokens
        m["cache_creation_input_tokens"] += cache_creation_input_tokens
        m["cache_read_input_tokens"] += cache_read_input_tokens
        m["bytes_sent"] += bytes_sent
        m["bytes_received"] += bytes_received

        endpoints_for_user = self.user_endpoint_stats.setdefault(user_id, {})
        if endpoint not in endpoints_for_user:
            endpoints_for_user[endpoint] = {
                "request_count": 0,
                "bytes_sent": 0,
                "bytes_received": 0,
            }
        e = endpoints_for_user[endpoint]
        e["request_count"] += 1
        e["bytes_sent"] += bytes_sent
        e["bytes_received"] += bytes_received

        totals = self.user_totals.setdefault(user_id, {
            "request_count": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
        })
        totals["request_count"] += 1
        totals["bytes_sent"] += bytes_sent
        totals["bytes_received"] += bytes_received

    def _append_request_to_daily_file(self, request_entry: Dict[str, Any]) -> None:
        try:
            from .state import state
            if not getattr(state, "save_request_to_file", False):
                return

            from .utils import get_config_dir
            requests_dir = os.path.join(get_config_dir(), "requests")
            os.makedirs(requests_dir, exist_ok=True)
            daily_file = os.path.join(requests_dir, f"{datetime.now().strftime('%Y-%m-%d')}.jl")
            with open(daily_file, "a", encoding="utf-8") as f:
                f.write(self.format_request_jsonl_line(request_entry))
        except Exception as e:
            print(f"[Request File Logging] Failed to append request: {e}")

    @staticmethod
    def format_request_jsonl_line(request_entry: Dict[str, Any]) -> str:
        """Format a request entry as one JSON Lines record."""
        return json.dumps(request_entry, ensure_ascii=False) + "\n"

    def add_request(self, request_id: str, data: Dict) -> None:
        """Add a request to the cache (legacy method for backwards compatibility)"""
        # Check if request already exists (started with start_request)
        with self.lock:
            exists = request_id in self.cache

        if exists:
            self.complete_request(request_id, data)
        else:
            # Legacy path: create and complete in one step
            self.start_request(request_id, data)
            self.complete_request(request_id, data)

    def get_request(self, request_id: str) -> Optional[Dict]:
        """Get a specific request by ID"""
        with self.lock:
            return self.cache.get(request_id)

    def get_recent_requests(self, limit: int = 50, offset: int = 0, user_id: Optional[str] = None) -> List[Dict]:
        """Get recent requests with pagination, optionally filtered by user_id."""
        with self.lock:
            items = list(reversed(list(self.cache.values())))
            if user_id is not None:
                items = [it for it in items if _coerce_user_id(it.get("user_id")) == user_id]
            return items[offset:offset + limit]

    def get_total_count(self, user_id: Optional[str] = None) -> int:
        """Get total number of cached requests (optionally for a single user)."""
        with self.lock:
            if user_id is None:
                return len(self.cache)
            return sum(1 for it in self.cache.values() if _coerce_user_id(it.get("user_id")) == user_id)

    def get_stats(self, user_id: Optional[str] = None) -> Dict:
        """Get overall statistics. When user_id is provided, returns that user's
        slice; otherwise returns global stats (the historical behavior)."""
        with self.lock:
            if user_id is None:
                return {
                    "user_id": None,
                    "total_requests": self.request_count,
                    "cached_requests": len(self.cache),
                    "bytes_sent": self.bytes_sent,
                    "bytes_received": self.bytes_received,
                    "model_stats": dict(self.model_stats),
                    "endpoint_stats": dict(self.endpoint_stats),
                }

            totals = self.user_totals.get(user_id, {
                "request_count": 0,
                "bytes_sent": 0,
                "bytes_received": 0,
            })
            cached_for_user = sum(
                1 for it in self.cache.values()
                if _coerce_user_id(it.get("user_id")) == user_id
            )
            return {
                "user_id": user_id,
                "total_requests": totals["request_count"],
                "cached_requests": cached_for_user,
                "bytes_sent": totals["bytes_sent"],
                "bytes_received": totals["bytes_received"],
                "model_stats": dict(self.user_model_stats.get(user_id, {})),
                "endpoint_stats": dict(self.user_endpoint_stats.get(user_id, {})),
            }

    def get_model_token_snapshot(self) -> Dict[str, Dict[str, int]]:
        """Get a thread-safe snapshot of model token counters (global)."""
        with self.lock:
            return {
                model: {
                    "request_count": int(stats.get("request_count", 0)),
                    "input_tokens": int(stats.get("input_tokens", 0)),
                    "output_tokens": int(stats.get("output_tokens", 0)),
                    "cache_creation_input_tokens": int(stats.get("cache_creation_input_tokens", 0)),
                    "cache_read_input_tokens": int(stats.get("cache_read_input_tokens", 0)),
                    "bytes_sent": int(stats.get("bytes_sent", 0)),
                    "bytes_received": int(stats.get("bytes_received", 0)),
                }
                for model, stats in self.model_stats.items()
            }

    def get_user_model_token_snapshot(self) -> Dict[Tuple[str, str], Dict[str, int]]:
        """Get a thread-safe snapshot of token counters keyed by (user_id, model).

        Used by the token-usage reporter to produce per-user JSONL deltas."""
        with self.lock:
            snapshot: Dict[Tuple[str, str], Dict[str, int]] = {}
            for user_id, models in self.user_model_stats.items():
                for model, stats in models.items():
                    snapshot[(user_id, model)] = {
                        "request_count": int(stats.get("request_count", 0)),
                        "input_tokens": int(stats.get("input_tokens", 0)),
                        "output_tokens": int(stats.get("output_tokens", 0)),
                        "cache_creation_input_tokens": int(stats.get("cache_creation_input_tokens", 0)),
                        "cache_read_input_tokens": int(stats.get("cache_read_input_tokens", 0)),
                        "bytes_sent": int(stats.get("bytes_sent", 0)),
                        "bytes_received": int(stats.get("bytes_received", 0)),
                    }
            return snapshot

    def list_user_ids(self) -> List[str]:
        """List all user_ids that have any data in the cache or stats."""
        with self.lock:
            ids = set(self.user_model_stats.keys()) | set(self.user_totals.keys())
            for entry in self.cache.values():
                ids.add(_coerce_user_id(entry.get("user_id")))
            return sorted(ids)

    def search_requests(self, query: str, limit: int = 50, offset: int = 0, user_id: Optional[str] = None) -> List[Dict]:
        """Search requests by model, endpoint, or content (optionally per-user)."""
        with self.lock:
            results = []
            query_lower = query.lower()
            for item in reversed(list(self.cache.values())):
                if user_id is not None and _coerce_user_id(item.get("user_id")) != user_id:
                    continue
                if (query_lower in item.get("model", "").lower() or
                    query_lower in item.get("endpoint", "").lower() or
                    query_lower in json.dumps(item.get("request_body", {})).lower()):
                    results.append(item)
            return results[offset:offset + limit]

    def fulltext_search(self, query: str, limit: int = 50, offset: int = 0, user_id: Optional[str] = None) -> Tuple[List[Dict], int]:
        """Full-text search in request and response bodies (optionally per-user)."""
        with self.lock:
            results = []
            query_lower = query.lower()
            for item in reversed(list(self.cache.values())):
                if user_id is not None and _coerce_user_id(item.get("user_id")) != user_id:
                    continue
                # Search in request body
                request_body_str = json.dumps(item.get("request_body", {})).lower()
                # Search in response body (or raw SSE events for streaming entries)
                if item.get("raw_events") is not None:
                    response_body_str = "\n".join(item.get("raw_events") or []).lower()
                else:
                    response_body_str = json.dumps(item.get("response_body", {})).lower()

                if query_lower in request_body_str or query_lower in response_body_str:
                    results.append(item)

            total = len(results)
            return results[offset:offset + limit], total

    def get_all_requests(self) -> List[Dict]:
        """Get all requests for export"""
        with self.lock:
            return list(self.cache.values())

    def import_request(self, data: Dict) -> None:
        """Import a single request entry"""
        with self.lock:
            request_id = data.get("id", str(uuid.uuid4()))

            if len(self.cache) >= self.max_entries:
                self.cache.popitem(last=False)

            user_id = _coerce_user_id(data.get("user_id"))

            self.cache[request_id] = {
                "id": request_id,
                "timestamp": self._normalize_import_timestamp(data.get("timestamp")),
                "client_ip": data.get("client_ip"),
                "request_headers": data.get("request_headers"),
                "original_request_body": data.get("original_request_body"),
                "request_body": data.get("request_body"),
                "response_body": data.get("response_body"),
                "raw_events": data.get("raw_events"),
                "model": data.get("model", "unknown"),
                "translated_model": data.get("translated_model"),
                "endpoint": data.get("endpoint", "unknown"),
                "status_code": data.get("status_code", 200),
                "request_size": data.get("request_size", 0),
                "response_size": data.get("response_size", 0),
                "input_tokens": data.get("input_tokens", 0),
                "output_tokens": data.get("output_tokens", 0),
                "cache_creation_input_tokens": data.get("cache_creation_input_tokens", 0),
                "cache_read_input_tokens": data.get("cache_read_input_tokens", 0),
                "duration": data.get("duration", 0),
                "state": data.get("state", "completed"),
                "user_id": user_id,
            }
            # Pass through any unrecognized keys so sidecar fields written by
            # subclassed SSE handlers (e.g. recovered_content) round-trip
            # through export/import.
            _IMPORT_KNOWN = {
                "id", "timestamp", "client_ip", "request_headers",
                "original_request_body", "request_body", "response_body",
                "raw_events", "model", "translated_model", "endpoint",
                "status_code", "request_size", "response_size", "input_tokens",
                "output_tokens", "cache_creation_input_tokens",
                "cache_read_input_tokens", "duration", "state", "user_id",
            }
            for key, value in data.items():
                if key not in _IMPORT_KNOWN:
                    self.cache[request_id][key] = value
            self._truncate_oversize_bodies(self.cache[request_id])

            # Update stats
            self.request_count += 1
            self.bytes_sent += data.get("request_size", 0)
            self.bytes_received += data.get("response_size", 0)

            model = data.get("translated_model") or data.get("model", "unknown")
            if model not in self.model_stats:
                self.model_stats[model] = {
                    "request_count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "bytes_sent": 0,
                    "bytes_received": 0,
                }
            self.model_stats[model]["request_count"] += 1
            self.model_stats[model]["input_tokens"] += data.get("input_tokens", 0)
            self.model_stats[model]["output_tokens"] += data.get("output_tokens", 0)
            self.model_stats[model]["cache_creation_input_tokens"] += data.get("cache_creation_input_tokens", 0)
            self.model_stats[model]["cache_read_input_tokens"] += data.get("cache_read_input_tokens", 0)
            self.model_stats[model]["bytes_sent"] += data.get("request_size", 0)
            self.model_stats[model]["bytes_received"] += data.get("response_size", 0)

            endpoint = data.get("endpoint", "unknown")
            if endpoint not in self.endpoint_stats:
                self.endpoint_stats[endpoint] = {
                    "request_count": 0,
                    "bytes_sent": 0,
                    "bytes_received": 0,
                }
            self.endpoint_stats[endpoint]["request_count"] += 1
            self.endpoint_stats[endpoint]["bytes_sent"] += data.get("request_size", 0)
            self.endpoint_stats[endpoint]["bytes_received"] += data.get("response_size", 0)

            # Per-user accumulators.
            self._bump_user_stats(
                user_id=user_id,
                model=model,
                endpoint=endpoint,
                input_tokens=data.get("input_tokens", 0),
                output_tokens=data.get("output_tokens", 0),
                cache_creation_input_tokens=data.get("cache_creation_input_tokens", 0),
                cache_read_input_tokens=data.get("cache_read_input_tokens", 0),
                bytes_sent=data.get("request_size", 0),
                bytes_received=data.get("response_size", 0),
            )


# Global cache instance
cache = RequestCache()
