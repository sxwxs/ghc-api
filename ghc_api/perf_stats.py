"""Live throughput metrics (tokens/second) for the current process run.

The request cache answers "what was sent"; it does not answer "why does this
feel slow right now". This module keeps a tiny, bounded, in-memory ring of
*finished* request samples so the dashboard can show, per model:

* decode speed (output tokens / generation seconds, i.e. excluding TTFT),
* time to first token,
* total duration, error rate and token volume,

over a selectable window (last 5 minutes ... since process start).

Design constraints, in order of importance:

1. **No measurable cost on the request path.** ``record()`` appends one plain
   tuple to a ``collections.deque`` under a lock it holds for microseconds.
   Nothing is computed there: every aggregate is derived on read, i.e. only
   when a dashboard page asks for it.
2. **Bounded memory.** The deque has a ``maxlen`` (default 2000 samples, a few
   hundred KB). Old samples are dropped silently; ``snapshot()`` reports
   ``truncated`` and the oldest retained timestamp so a "24h" view cannot
   quietly lie about a window it no longer covers.
3. **Current run only.** Samples live in memory and vanish on restart, which is
   exactly the question being asked ("what is running right now"). Historical
   analysis already exists in ``request_file_stats``.

Imported requests (``RequestCache.import_request``) deliberately do not feed
this store: replaying a JSONL export would otherwise fabricate throughput that
this process never produced.
"""

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

# Default number of retained samples. 2000 finished requests is far more than
# an interactive session produces in a day, while costing well under 1 MB.
DEFAULT_CAPACITY = 2000

# A request only contributes to tokens/second when it actually generated
# something measurable. Sub-50ms generations are mostly clock/rounding noise
# and would produce absurd outliers (e.g. 1 token / 0.001s = 1000 tok/s).
MIN_TPS_OUTPUT_TOKENS = 1
MIN_TPS_SECONDS = 0.05

# Selectable dashboard windows. ``None`` means "everything retained".
WINDOW_PRESETS: Dict[str, Optional[int]] = {
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "6h": 21600,
    "24h": 86400,
    "all": None,
}
DEFAULT_WINDOW = "1h"

# Sample layout. A plain tuple instead of a dataclass: this is the one object
# allocated per request, and tuples are the cheapest option that still keeps
# the fields ordered and immutable.
#   0 finished_at    unix seconds (float)
#   1 model          translated model name when available
#   2 endpoint       proxy-side endpoint
#   3 status_code    int
#   4 duration       total seconds, request start -> completion
#   5 ttft           seconds to first streamed token, or None (non-streaming)
#   6 output_tokens  int
#   7 input_tokens   int (billed prompt tokens, excluding cache fields)
#   8 user_id        str
Sample = Tuple[float, str, str, int, float, Optional[float], int, int, str]


def parse_window(raw: Any) -> Tuple[str, Optional[int]]:
    """Resolve a window request into ``(label, seconds)``.

    Accepts a preset label (``5m``/``1h``/``all``...) or a raw number of
    seconds. Anything unrecognized falls back to the default window rather
    than erroring: this is a monitoring view, not an API contract.
    """
    if raw is None:
        return DEFAULT_WINDOW, WINDOW_PRESETS[DEFAULT_WINDOW]

    text = str(raw).strip().lower()
    if not text:
        return DEFAULT_WINDOW, WINDOW_PRESETS[DEFAULT_WINDOW]
    if text in WINDOW_PRESETS:
        return text, WINDOW_PRESETS[text]
    if text in ("since-start", "start", "*", "run"):
        return "all", None

    try:
        seconds = int(float(text))
    except (TypeError, ValueError):
        return DEFAULT_WINDOW, WINDOW_PRESETS[DEFAULT_WINDOW]
    if seconds <= 0:
        return "all", None
    return f"{seconds}s", seconds


def _percentile(sorted_values: Sequence[float], fraction: float) -> Optional[float]:
    """Linear-interpolated percentile of an already sorted sequence."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = fraction * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return float(sorted_values[low] * (1 - weight) + sorted_values[high] * weight)


def _round(value: Optional[float], digits: int = 2) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def generation_seconds(duration: float, ttft: Optional[float]) -> float:
    """Seconds spent producing tokens: total duration minus the wait for the
    first one. Falls back to the full duration when TTFT is unknown
    (non-streaming responses), where the two are indistinguishable anyway."""
    if ttft is None:
        return max(0.0, float(duration))
    return max(0.0, float(duration) - float(ttft))


def tokens_per_second(
    output_tokens: int, duration: float, ttft: Optional[float]
) -> Optional[float]:
    """Decode speed for one request, or None when it is not measurable."""
    if not output_tokens or output_tokens < MIN_TPS_OUTPUT_TOKENS:
        return None
    seconds = generation_seconds(duration, ttft)
    if seconds < MIN_TPS_SECONDS:
        return None
    return round(output_tokens / seconds, 2)


class PerfStats:
    """Bounded ring of finished-request samples for the current process run."""

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        self._lock = threading.Lock()
        self._samples: Deque[Sample] = deque(maxlen=max(1, int(capacity)))
        self.started_at: float = time.time()
        # Samples evicted because the ring was full. Surfaced so the UI can say
        # "window truncated" instead of silently under-reporting.
        self.dropped: int = 0
        self.recorded: int = 0

    # ------------------------------------------------------------- lifecycle

    @property
    def capacity(self) -> int:
        return self._samples.maxlen or DEFAULT_CAPACITY

    def set_capacity(self, capacity: int) -> None:
        """Resize the ring, keeping the most recent samples."""
        capacity = max(1, int(capacity))
        with self._lock:
            if capacity == self._samples.maxlen:
                return
            kept = list(self._samples)[-capacity:]
            self.dropped += max(0, len(self._samples) - len(kept))
            self._samples = deque(kept, maxlen=capacity)

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self.started_at = time.time()
            self.dropped = 0
            self.recorded = 0

    # ----------------------------------------------------------- hot path

    def record(
        self,
        model: str,
        endpoint: str,
        status_code: int,
        duration: float,
        output_tokens: int,
        input_tokens: int = 0,
        ttft: Optional[float] = None,
        user_id: str = "anonymous",
        finished_at: Optional[float] = None,
    ) -> None:
        """Append one finished-request sample. O(1), no aggregation."""
        sample: Sample = (
            float(finished_at if finished_at is not None else time.time()),
            model or "unknown",
            endpoint or "unknown",
            int(status_code or 0),
            float(duration or 0.0),
            None if ttft is None else float(ttft),
            int(output_tokens or 0),
            int(input_tokens or 0),
            user_id or "anonymous",
        )
        with self._lock:
            if self._samples.maxlen and len(self._samples) == self._samples.maxlen:
                self.dropped += 1
            self._samples.append(sample)
            self.recorded += 1

    # -------------------------------------------------------------- reading

    def snapshot(
        self,
        window_seconds: Optional[int] = None,
        user_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Aggregate retained samples into overall and per-model metrics."""
        now = time.time() if now is None else now
        cutoff = None if window_seconds is None else now - window_seconds

        with self._lock:
            samples = list(self._samples)
            started_at = self.started_at
            dropped = self.dropped
            recorded = self.recorded
            capacity = self._samples.maxlen or DEFAULT_CAPACITY

        oldest_retained = samples[0][0] if samples else None

        rows = [s for s in samples if cutoff is None or s[0] >= cutoff]
        if user_id is not None:
            rows = [s for s in rows if s[8] == user_id]

        by_model: Dict[str, List[Sample]] = {}
        by_endpoint: Dict[str, List[Sample]] = {}
        for sample in rows:
            by_model.setdefault(sample[1], []).append(sample)
            by_endpoint.setdefault(sample[2], []).append(sample)

        # A window wider than what the ring still holds cannot be answered in
        # full; say so instead of implying the numbers cover it.
        window_start = cutoff if cutoff is not None else started_at
        truncated = bool(
            dropped
            and oldest_retained is not None
            and oldest_retained > window_start + 1
        )

        models = [_aggregate(name, group, now) for name, group in by_model.items()]
        models.sort(key=lambda m: m["output_tokens"], reverse=True)

        endpoints = [
            {
                "endpoint": name,
                "requests": len(group),
                "errors": sum(1 for s in group if s[3] >= 400),
                "output_tokens": sum(s[6] for s in group),
            }
            for name, group in by_endpoint.items()
        ]
        endpoints.sort(key=lambda e: e["requests"], reverse=True)

        return {
            "server_started_at": started_at,
            "uptime_seconds": round(now - started_at, 1),
            "now": now,
            "window_seconds": window_seconds,
            "window_start": window_start,
            "effective_window_seconds": round(
                now - max(window_start, oldest_retained or window_start), 1
            ),
            "sample_capacity": capacity,
            "samples_recorded": recorded,
            "samples_retained": len(samples),
            "samples_in_window": len(rows),
            "samples_dropped": dropped,
            "oldest_sample_at": oldest_retained,
            "truncated": truncated,
            "user_filter": user_id,
            "overall": _aggregate("__all__", rows, now),
            "models": models,
            "endpoints": endpoints,
        }


def _aggregate(name: str, rows: Sequence[Sample], now: float) -> Dict[str, Any]:
    """Compute one model's (or the overall) metrics from its samples."""
    requests = len(rows)
    errors = sum(1 for s in rows if s[3] >= 400)
    input_tokens = sum(s[7] for s in rows)
    output_tokens = sum(s[6] for s in rows)

    durations: List[float] = []
    ttfts: List[float] = []
    tps_values: List[float] = []
    # Token-weighted throughput: total tokens over total generating time. This
    # is the number that explains "the model is slow", while the median of
    # per-request rates explains "a typical reply feels slow".
    tps_token_total = 0
    tps_second_total = 0.0
    first_seen: Optional[float] = None
    last_seen: Optional[float] = None

    for finished_at, _model, _endpoint, status, duration, ttft, out_tokens, _in, _user in rows:
        durations.append(duration)
        if ttft is not None:
            ttfts.append(ttft)
        if first_seen is None or finished_at < first_seen:
            first_seen = finished_at
        if last_seen is None or finished_at > last_seen:
            last_seen = finished_at
        if status >= 400:
            continue
        rate = tokens_per_second(out_tokens, duration, ttft)
        if rate is not None:
            tps_values.append(rate)
            tps_token_total += out_tokens
            tps_second_total += generation_seconds(duration, ttft)

    durations.sort()
    ttfts.sort()
    tps_values.sort()

    return {
        "model": name,
        "requests": requests,
        "errors": errors,
        "error_rate": _round(errors / requests, 4) if requests else 0.0,
        "streaming_requests": len(ttfts),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "measured_requests": len(tps_values),
        # Aggregate decode speed across every measured request in the window.
        "tps": _round(tps_token_total / tps_second_total) if tps_second_total else None,
        "tps_median": _round(_percentile(tps_values, 0.5)),
        "tps_p10": _round(_percentile(tps_values, 0.10)),
        "tps_p90": _round(_percentile(tps_values, 0.90)),
        "tps_min": _round(tps_values[0]) if tps_values else None,
        "tps_max": _round(tps_values[-1]) if tps_values else None,
        "ttft_median": _round(_percentile(ttfts, 0.5), 3),
        "ttft_p90": _round(_percentile(ttfts, 0.90), 3),
        "ttft_max": _round(ttfts[-1], 3) if ttfts else None,
        "duration_median": _round(_percentile(durations, 0.5)),
        "duration_p90": _round(_percentile(durations, 0.90)),
        "duration_total": _round(sum(durations)),
        "avg_output_tokens": _round(output_tokens / requests, 1) if requests else None,
        "first_seen": first_seen,
        "last_seen": last_seen,
        "seconds_since_last": _round(now - last_seen, 1) if last_seen else None,
    }


# Global instance, mirroring the ``state`` / ``cache`` / ``counters`` singletons.
perf_stats = PerfStats()
