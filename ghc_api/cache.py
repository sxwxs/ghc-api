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


class RequestCache:
    """Thread-safe cache for storing API requests and responses"""

    # Request states
    STATE_PENDING = "pending"
    STATE_SENDING = "sending"
    STATE_RECEIVING = "receiving"
    STATE_COMPLETED = "completed"
    STATE_ERROR = "error"

    def __init__(self, max_size_mb: int = 200, max_entries: int = 10000):
        self.max_size_bytes = int(max_size_mb) * 1024 * 1024
        self.max_entries = int(max_entries)
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.Lock()
        self.request_count = 0
        self.bytes_sent = 0
        self.bytes_received = 0
        self.model_stats: Dict[str, Dict] = {}
        self.endpoint_stats: Dict[str, Dict] = {}
        self.current_size_bytes: int = 0
        self.entry_sizes: Dict[str, int] = {}

    @staticmethod
    def _current_timestamp() -> int:
        return int(time.time())

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

    @staticmethod
    def _estimate_entry_size(entry: Dict) -> int:
        """Estimate memory size (bytes) of the heavy fields in an entry."""
        total = 0
        for field in ("request_headers", "original_request_body", "request_body", "response_body"):
            value = entry.get(field)
            if value is None:
                continue
            try:
                if isinstance(value, bytes):
                    total += len(value)
                elif isinstance(value, str):
                    total += len(value.encode("utf-8"))
                else:
                    total += len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))
            except Exception:
                total += len(repr(value))
        return total

    def _evict_until_within_limits(self) -> None:
        """Evict oldest entries until cache is within size and count limits.

        Must be called while self.lock is held.
        """
        while self.cache and (
            len(self.cache) > self.max_entries
            or self.current_size_bytes > self.max_size_bytes
        ):
            oldest_id, _ = self.cache.popitem(last=False)
            self.current_size_bytes -= self.entry_sizes.pop(oldest_id, 0)
        if self.current_size_bytes < 0:
            self.current_size_bytes = 0

    def _update_entry_size(self, request_id: str) -> None:
        """Recompute and update the tracked size for an existing entry.

        Must be called while self.lock is held.
        """
        new_size = self._estimate_entry_size(self.cache[request_id])
        old_size = self.entry_sizes.get(request_id, 0)
        self.current_size_bytes += new_size - old_size
        if self.current_size_bytes < 0:
            self.current_size_bytes = 0
        self.entry_sizes[request_id] = new_size

    def start_request(self, request_id: str, data: Dict) -> None:
        """Start tracking a new request (before sending to upstream)"""
        with self.lock:
            self.cache[request_id] = {
                "id": request_id,
                "timestamp": self._current_timestamp(),
                "request_headers": data.get("request_headers"),
                "original_request_body": data.get("original_request_body"),
                "request_body": data.get("request_body"),
                "response_body": None,
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
            }
            entry_size = self._estimate_entry_size(self.cache[request_id])
            self.entry_sizes[request_id] = entry_size
            self.current_size_bytes += entry_size
            self._evict_until_within_limits()

    def update_request_state(self, request_id: str, state: str, **kwargs) -> None:
        """Update the state and optional fields of an existing request"""
        with self.lock:
            if request_id in self.cache:
                self.cache[request_id]["state"] = state
                for key, value in kwargs.items():
                    if key in self.cache[request_id]:
                        self.cache[request_id][key] = value
                self._update_entry_size(request_id)
                self._evict_until_within_limits()

    def complete_request(self, request_id: str, data: Dict) -> None:
        """Complete a request with response data and update statistics"""
        entry_snapshot = None
        with self.lock:
            if request_id in self.cache:
                # Update existing entry
                entry = self.cache[request_id]
                if "request_headers" in data:
                    entry["request_headers"] = data.get("request_headers")
                if "original_request_body" in data:
                    entry["original_request_body"] = data.get("original_request_body")
                if "request_body" in data:
                    entry["request_body"] = data.get("request_body")
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
                self._update_entry_size(request_id)
                self._evict_until_within_limits()
            else:
                # Fallback: create new entry if somehow missing
                self.cache[request_id] = {
                    "id": request_id,
                    "timestamp": self._current_timestamp(),
                    "request_headers": data.get("request_headers"),
                    "original_request_body": data.get("original_request_body"),
                    "request_body": data.get("request_body"),
                    "response_body": data.get("response_body"),
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
                }
                entry_size = self._estimate_entry_size(self.cache[request_id])
                self.entry_sizes[request_id] = entry_size
                self.current_size_bytes += entry_size
                self._evict_until_within_limits()
            self.request_count += 1
            self.bytes_sent += data.get("request_size", 0)
            self.bytes_received += data.get("response_size", 0)

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
            entry_snapshot = dict(self.cache[request_id])

        if entry_snapshot:
            self._append_request_to_daily_file(entry_snapshot)

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

    def get_recent_requests(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        """Get recent requests with pagination"""
        with self.lock:
            items = list(reversed(list(self.cache.values())))
            return items[offset:offset + limit]

    def get_total_count(self) -> int:
        """Get total number of cached requests"""
        with self.lock:
            return len(self.cache)

    def get_stats(self) -> Dict:
        """Get overall statistics"""
        with self.lock:
            return {
                "total_requests": self.request_count,
                "cached_requests": len(self.cache),
                "bytes_sent": self.bytes_sent,
                "bytes_received": self.bytes_received,
                "model_stats": dict(self.model_stats),
                "endpoint_stats": dict(self.endpoint_stats),
                "cache_size_bytes": self.current_size_bytes,
                "cache_size_mb": round(self.current_size_bytes / (1024 * 1024), 2),
                "cache_max_size_mb": self.max_size_bytes // (1024 * 1024),
                "cache_entry_count": len(self.cache),
                "cache_max_entries": self.max_entries,
            }

    def get_model_token_snapshot(self) -> Dict[str, Dict[str, int]]:
        """Get a thread-safe snapshot of model token counters."""
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

    def search_requests(self, query: str, limit: int = 50, offset: int = 0) -> List[Dict]:
        """Search requests by model, endpoint, or content"""
        with self.lock:
            results = []
            query_lower = query.lower()
            for item in reversed(list(self.cache.values())):
                if (query_lower in item.get("model", "").lower() or
                    query_lower in item.get("endpoint", "").lower() or
                    query_lower in json.dumps(item.get("request_body", {})).lower()):
                    results.append(item)
            return results[offset:offset + limit]

    def fulltext_search(self, query: str, limit: int = 50, offset: int = 0) -> Tuple[List[Dict], int]:
        """Full-text search in request and response bodies"""
        with self.lock:
            results = []
            query_lower = query.lower()
            for item in reversed(list(self.cache.values())):
                # Search in request body
                request_body_str = json.dumps(item.get("request_body", {})).lower()
                # Search in response body
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

            self.cache[request_id] = {
                "id": request_id,
                "timestamp": self._normalize_import_timestamp(data.get("timestamp")),
                "request_headers": data.get("request_headers"),
                "original_request_body": data.get("original_request_body"),
                "request_body": data.get("request_body"),
                "response_body": data.get("response_body"),
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
            }
            entry_size = self._estimate_entry_size(self.cache[request_id])
            self.entry_sizes[request_id] = entry_size
            self.current_size_bytes += entry_size
            self._evict_until_within_limits()

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


# Global cache instance
cache = RequestCache()
