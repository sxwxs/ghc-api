"""
Thread-safe in-memory index of recent requests, plus usage statistics.

Request and response bodies are held here only while a request is in flight.
When it finishes, the full record is appended to the request log
(``request_log``) and the cached entry shrinks to metadata plus a locator
pointing at that line. The log is the single source of truth for bodies, so
nothing here ever needs to truncate one.
"""

import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from .perf_stats import perf_stats, tokens_per_second
from .request_log import RequestLogError, metadata_view, request_log


ANONYMOUS_USER_ID = "anonymous"

# States a request passes through before it is finalized. Used by the live
# dashboard to answer "what is running right now" without a second registry.
IN_FLIGHT_STATES = ("pending", "sending", "receiving")


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

    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
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
        # Billing dimension: (user_id, provider, model) -> token counters, kept
        # in parallel to user_model_stats so existing consumers are unaffected.
        # provider comes from the request's upstream_provider sidecar field and
        # defaults to "copilot" (this proxy's primary upstream).
        self.user_provider_model_stats: Dict[Tuple[str, str, str], Dict[str, int]] = {}

    @staticmethod
    def _current_timestamp() -> int:
        return int(time.time())

    def _new_entry(self, request_id: str, data: Dict) -> Dict[str, Any]:
        return {
            "id": request_id,
            "timestamp": self._current_timestamp(),
            # Sub-second start time, the origin for both TTFT and duration.
            # Callers pass the route's start_time: streaming handlers often
            # start tracking only after upstream headers arrived.
            "started_at": data.get("started_at") or time.time(),
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
            # Seconds until the first streamed chunk, and output decode
            # speed. Both stay None until measurable (see _finalize_speed).
            "ttft": None,
            "tokens_per_second": None,
            "state": self.STATE_PENDING,
            "user_id": _coerce_user_id(data.get("user_id")),
        }

    def _insert(self, request_id: str, entry: Dict[str, Any]) -> None:
        """Insert a new entry, evicting the oldest finished ones. Called from within self.lock.

        In-flight entries are never evicted: their bodies exist only here until
        completion writes them to the request log, and the live view lists
        them. With more in-flight requests than max_entries the list grows
        until they finish (the routes hold that data anyway) and is trimmed
        back to the limit on later inserts.
        """
        if request_id not in self.cache:
            for key in [key for key, cached in self.cache.items()
                        if cached.get("state") not in IN_FLIGHT_STATES]:
                if len(self.cache) < self.max_entries:
                    break
                del self.cache[key]
        self.cache[request_id] = entry

    def start_request(self, request_id: str, data: Dict) -> None:
        """Start tracking a new request (before sending to upstream)"""
        with self.lock:
            self._insert(request_id, self._new_entry(request_id, data))

    def update_request_state(self, request_id: str, state: str, **kwargs) -> None:
        """Update the state and optional fields of an existing request"""
        with self.lock:
            if request_id in self.cache:
                entry = self.cache[request_id]
                entry["state"] = state
                for key, value in kwargs.items():
                    if key in entry:
                        entry[key] = value
                # Streaming handlers enter RECEIVING on the first event that
                # carries model output (perf_stats.is_output_event), so that
                # transition is exactly the time to first token.
                if state == self.STATE_RECEIVING and entry.get("ttft") is None:
                    entry["ttft"] = round(max(0.0, time.time() - entry["started_at"]), 3)

    def complete_request(self, request_id: str, data: Dict) -> None:
        """Finalize a request attempt, persist its full record and update statistics.

        The full record (bodies included, exactly as the route supplied them)
        is appended to the request log. Afterwards the cached entry is replaced
        by its metadata view, so finished requests never hold bodies in memory.
        """
        with self.lock:
            record = self.cache.get(request_id)
            if record is None:
                # Fallback: the request was never started (or already evicted).
                record = self._new_entry(request_id, data)
                self._insert(request_id, record)
            self._apply_completion(record, data)
            self._finalize_speed(record)
            model, endpoint = self._count_completion(record, data)

        locator = request_log.append(record)

        with self.lock:
            # Only shrink the entry we wrote; it may have been evicted meanwhile.
            if self.cache.get(request_id) is record:
                self.cache[request_id] = metadata_view(record, locator)

        self._record_perf_sample(record, model, endpoint)

    def _apply_completion(self, entry: Dict[str, Any], data: Dict) -> None:
        """Merge a completion payload into an entry. Called from within self.lock."""
        for key in ("client_ip", "request_headers", "original_request_body", "request_body"):
            if key in data:
                entry[key] = data.get(key)
        if "raw_events" in data:
            entry["raw_events"] = data.get("raw_events")
            # Translation handlers may provide both the authoritative raw
            # upstream events and a client-facing terminal projection. Preserve
            # the latter when explicitly supplied; ordinary passthrough SSE
            # handlers still omit it.
            entry["response_body"] = data.get("response_body") if "response_body" in data else None
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
        for key in ("ttft", "tokens_per_second"):
            if data.get(key) is not None:
                entry[key] = data[key]
        entry["state"] = self.STATE_COMPLETED if data.get("status_code", 200) < 400 else self.STATE_ERROR
        # Defensive: if user_id wasn't set during start_request (legacy path
        # or external caller forgot), fall back to anonymous.
        if not entry.get("user_id"):
            entry["user_id"] = _coerce_user_id(data.get("user_id"))
        # Pass through any unrecognized keys so subclasses can surface sidecar
        # fields (e.g. recovered_content for tool-call recovery).
        for key, value in data.items():
            if key not in _COMPLETION_KNOWN:
                entry[key] = value

    def _count_completion(self, entry: Dict[str, Any], data: Dict) -> Tuple[str, str]:
        """Update the usage counters for one finished request. Called from within self.lock."""
        self.request_count += 1
        self.bytes_sent += data.get("request_size", 0)
        self.bytes_received += data.get("response_size", 0)

        user_id = _coerce_user_id(entry.get("user_id"))

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

        provider = str(data.get("upstream_provider") or entry.get("upstream_provider") or "copilot")
        billing_key = (user_id, provider, model)
        billing = self.user_provider_model_stats.setdefault(billing_key, {
            "request_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
        })
        billing["request_count"] += 1
        billing["input_tokens"] += data.get("input_tokens", 0)
        billing["output_tokens"] += data.get("output_tokens", 0)
        billing["cache_creation_input_tokens"] += data.get("cache_creation_input_tokens", 0)
        billing["cache_read_input_tokens"] += data.get("cache_read_input_tokens", 0)
        billing["bytes_sent"] += data.get("request_size", 0)
        billing["bytes_received"] += data.get("response_size", 0)
        return model, endpoint

    @staticmethod
    def _finalize_speed(entry: Dict[str, Any]) -> None:
        """Fill in the per-request throughput fields, if not already supplied.

        ``tokens_per_second`` is the decode rate: output tokens divided by the
        time spent generating them (total duration minus TTFT), so a slow
        upstream queue shows up as a large TTFT instead of silently dragging
        the measured speed down. Called from within ``self.lock``.
        """
        if entry.get("tokens_per_second") is None:
            entry["tokens_per_second"] = tokens_per_second(
                entry.get("output_tokens") or 0,
                entry.get("duration") or 0.0,
                entry.get("ttft"),
            )

    @staticmethod
    def _record_perf_sample(entry: Dict[str, Any], model: str, endpoint: str) -> None:
        """Feed the live throughput ring. Never allowed to break a response."""
        try:
            perf_stats.record(
                model=model,
                endpoint=endpoint,
                status_code=entry.get("status_code") or 0,
                duration=entry.get("duration") or 0.0,
                output_tokens=entry.get("output_tokens") or 0,
                input_tokens=entry.get("input_tokens") or 0,
                ttft=entry.get("ttft"),
                user_id=entry.get("user_id") or ANONYMOUS_USER_ID,
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[Perf Stats] Failed to record sample: {exc}")

    def cancel_request(self, request_id: str) -> None:
        """Finalize a request as 499 after the client disconnected.

        A no-op when the request already finished, so stream wrappers can call
        it unconditionally even when an inner handler completed the entry.
        """
        with self.lock:
            entry = self.cache.get(request_id)
            if entry is None or entry.get("state") not in IN_FLIGHT_STATES:
                return
            data = {
                key: entry.get(key)
                for key in ("model", "translated_model", "endpoint", "request_size", "user_id")
            }
            data["status_code"] = 499
            data["duration"] = round(time.time() - entry["started_at"], 2)
        self.complete_request(request_id, data)

    def get_in_flight(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Requests that have not been finalized yet, newest first.

        Derived from the existing cache states instead of a second registry, so
        there is nothing extra to keep in sync (and nothing to leak when a
        stream dies in an unusual way).
        """
        now = time.time()
        rows: List[Dict[str, Any]] = []
        with self.lock:
            for entry in self.cache.values():
                if entry.get("state") not in IN_FLIGHT_STATES:
                    continue
                if user_id is not None and _coerce_user_id(entry.get("user_id")) != user_id:
                    continue
                started = entry["started_at"]
                rows.append({
                    "id": entry.get("id"),
                    "model": entry.get("translated_model") or entry.get("model") or "unknown",
                    "endpoint": entry.get("endpoint"),
                    "state": entry.get("state"),
                    "user_id": _coerce_user_id(entry.get("user_id")),
                    "started_at": started,
                    "elapsed": round(max(0.0, now - started), 1),
                    "ttft": entry.get("ttft"),
                    "output_tokens": entry.get("output_tokens") or 0,
                })
        rows.sort(key=lambda row: row["started_at"], reverse=True)
        return rows

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
        """The cached entry: full while in flight, metadata plus ``log`` locator once finished."""
        with self.lock:
            return self.cache.get(request_id)

    def get_record(self, request_id: str) -> Optional[Dict]:
        """The full record of a listed request, bodies included.

        In-flight requests are served from memory; finished ones are read back
        from the request log. Returns None when the request is unknown or its
        record was never written (or has since been deleted by retention).
        """
        entry = self.get_request(request_id)
        if entry is None:
            return None
        if "log" not in entry:
            return entry
        if entry["log"] is None:
            return None
        try:
            record = request_log.read(entry["log"])
        except RequestLogError:
            return None
        return record if record.get("id") == request_id else None

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

    def get_user_provider_model_token_snapshot(self) -> Dict[Tuple[str, str, str], Dict[str, int]]:
        """Get a thread-safe snapshot of token counters keyed by (user_id, provider, model).

        Used by the token-usage reporter so usage deltas carry the upstream
        provider dimension needed for per-provider billing.
        """
        with self.lock:
            return {
                key: {
                    "request_count": int(stats.get("request_count", 0)),
                    "input_tokens": int(stats.get("input_tokens", 0)),
                    "output_tokens": int(stats.get("output_tokens", 0)),
                    "cache_creation_input_tokens": int(stats.get("cache_creation_input_tokens", 0)),
                    "cache_read_input_tokens": int(stats.get("cache_read_input_tokens", 0)),
                    "bytes_sent": int(stats.get("bytes_sent", 0)),
                    "bytes_received": int(stats.get("bytes_received", 0)),
                }
                for key, stats in self.user_provider_model_stats.items()
            }

    def list_user_ids(self) -> List[str]:
        """List all user_ids that have any data in the cache or stats."""
        with self.lock:
            ids = set(self.user_model_stats.keys()) | set(self.user_totals.keys())
            for entry in self.cache.values():
                ids.add(_coerce_user_id(entry.get("user_id")))
            return sorted(ids)

    def search_requests(self, query: str, limit: int = 50, offset: int = 0, user_id: Optional[str] = None) -> Tuple[List[Dict], int]:
        """Search cached requests by model or endpoint (optionally per-user)."""
        with self.lock:
            query_lower = query.lower()
            results = [
                item for item in reversed(list(self.cache.values()))
                if (user_id is None or _coerce_user_id(item.get("user_id")) == user_id)
                and (
                    query_lower in str(item.get("model") or "").lower()
                    or query_lower in str(item.get("translated_model") or "").lower()
                    or query_lower in str(item.get("endpoint") or "").lower()
                )
            ]
            return results[offset:offset + limit], len(results)

    def restore(self, entries: List[Dict[str, Any]]) -> int:
        """Prepend persisted metadata entries (oldest first) after a restart.

        Restored entries only repopulate the request list. They are not
        counted in the usage statistics or the live throughput samples, which
        describe the current process run.
        """
        with self.lock:
            room = self.max_entries - len(self.cache)
            fresh = [entry for entry in entries if entry["id"] not in self.cache]
            if room <= 0 or not fresh:
                return 0
            fresh = fresh[-room:]
            restored = OrderedDict()
            for entry in fresh:
                self._finalize_speed(entry)
                restored[entry["id"]] = entry
            restored.update(self.cache)
            self.cache = restored
            return len(fresh)


_COMPLETION_KNOWN = {
    "client_ip", "request_headers", "original_request_body",
    "request_body", "response_body", "raw_events", "status_code",
    "request_size", "response_size", "input_tokens", "output_tokens",
    "cache_creation_input_tokens", "cache_read_input_tokens",
    "duration", "ttft", "tokens_per_second", "user_id", "model",
    "translated_model", "endpoint",
}


# Global cache instance
cache = RequestCache()
