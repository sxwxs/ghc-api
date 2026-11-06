"""Streaming historical performance aggregates over scalar request-index rows.

No request bodies or per-request dataset are retained. Memory grows with the
number of groups, plus a bounded reservoir per day/model for historical fits.
Measured timing and end-to-end throughput stay separate from fitted estimates.
"""

from bisect import bisect_left
from collections import defaultdict
from datetime import datetime
import math
import random
from typing import Any, Dict, Iterable


SAMPLES_PER_CELL = 90
CONTEXT_BOUNDS = [8000, 32000, 64000, 128000, 200000]
CONTEXT_LABELS = ["<8k", "8–32k", "32–64k", "64–128k", "128–200k", "≥200k"]


def log_edges(low, high, count=48):
    step = math.log(high / low) / count
    return [low * math.exp(i * step) for i in range(count + 1)]


DURATION_EDGES = log_edges(0.05, 3000)
TPS_EDGES = log_edges(0.5, 400)
TOTAL_FIELDS = (
    "count", "errors", "cancelled", "unknown_status", "duration_count", "duration",
    "output", "input", "cache_read", "cache_write", "gen_count", "gen_duration",
    "gen_output", "context_count", "context_sum", "ttft_count", "ttft_sum",
    "decode_count", "decode_duration", "decode_output",
)


def new_totals():
    return dict.fromkeys(TOTAL_FIELDS, 0)


def new_cell():
    return {
        **new_totals(),
        "duration_hist": [0] * (len(DURATION_EDGES) - 1),
        "tps_hist": [0] * (len(TPS_EDGES) - 1),
        "samples": [],  # [duration seconds, output tokens, context tokens or null]
    }


def add_totals(target, values):
    for key in TOTAL_FIELDS:
        target[key] += values[key]


def record_histogram(histogram, edges, value):
    histogram[max(0, min(len(histogram) - 1, bisect_left(edges, value) - 1))] += 1
    return int(value < edges[0] or value > edges[-1])


def build_performance_cube(rows: Iterable[Dict[str, Any]], *, max_models: int = 500) -> Dict[str, Any]:
    cells = defaultdict(new_cell)
    hourly = defaultdict(new_totals)
    contexts = defaultdict(new_totals)
    upstream = defaultdict(new_totals)
    endpoints = defaultdict(new_totals)
    models = set()
    model_overflow = False
    quality = dict.fromkeys((
        "date_fallback", "missing_duration", "missing_output", "missing_context",
        "invalid_ttft", "duration_hist_clipped", "tps_hist_clipped",
    ), 0)
    rng = random.Random(7)
    total = 0

    for row in rows:
        total += 1
        timestamp = row.get("timestamp") or 0
        try:
            if timestamp <= 0:
                raise ValueError("missing timestamp")
            dt = datetime.fromtimestamp(timestamp)
            date, hour = dt.strftime("%Y-%m-%d"), dt.hour
        except (ValueError, OverflowError, OSError):
            # Keep the request in its daily file's totals, but do not invent an
            # hour for the heatmap when the timestamp is unavailable.
            date, hour = row["source_file"][:-3], None
            quality["date_fallback"] += 1

        model = row.get("effective_model") or "unknown"
        if model not in models and len(models - {"__other__"}) >= max_models:
            model = "__other__"
            model_overflow = True
        models.add(model)
        status = row.get("status_code")
        state = row.get("state") or ""
        cancelled = status == 499 or state in ("cancelled", "canceled")
        failed = not cancelled and ((status is not None and status >= 400) or state == "error")
        successful = (status is not None and 200 <= status < 300
                      and state in ("", "completed"))
        duration_ms = row.get("duration_ms")
        duration = duration_ms / 1000 if duration_ms is not None else None
        output = row.get("output_tokens")
        context = row.get("context_tokens")
        generating = successful and output is not None and output > 0 and duration is not None and duration > 0
        ttft_ms = row.get("ttft_ms")
        valid_ttft = ttft_ms is not None and duration_ms is not None and 0 <= ttft_ms <= duration_ms
        quality["invalid_ttft"] += int(ttft_ms is not None and not valid_ttft)
        quality["missing_duration"] += int(duration is None)
        quality["missing_output"] += int(output is None)
        quality["missing_context"] += int(context is None)

        values = new_totals()
        values.update({
            "count": 1, "errors": int(failed), "cancelled": int(cancelled),
            "unknown_status": int(status is None),
            "duration_count": int(duration is not None), "duration": duration or 0,
            "output": output or 0,
        })
        # Context totals and cache share must use the same validated requests
        # as the context chart, not partial sums from invalid components.
        if context is not None:
            values.update(input=row.get("input_not_cached_tokens") or 0,
                          cache_read=row.get("cache_read_tokens") or 0,
                          cache_write=row.get("cache_write_tokens") or 0)
        if generating:
            values.update(gen_count=1, gen_duration=duration, gen_output=output)
            if context is not None:
                values.update(context_count=1, context_sum=context)
        if successful and valid_ttft:
            values.update(ttft_count=1, ttft_sum=ttft_ms / 1000)
            if generating and duration_ms > ttft_ms:
                values.update(decode_count=1, decode_duration=(duration_ms - ttft_ms) / 1000,
                              decode_output=output)

        cell = cells[(date, model)]
        add_totals(cell, values)
        if duration is not None:
            quality["duration_hist_clipped"] += record_histogram(cell["duration_hist"], DURATION_EDGES, duration)
        if generating:
            quality["tps_hist_clipped"] += record_histogram(cell["tps_hist"], TPS_EDGES, output / duration)
            sample = [duration, output, context]
            if len(cell["samples"]) < SAMPLES_PER_CELL:
                cell["samples"].append(sample)
            else:
                position = rng.randrange(cell["gen_count"])
                if position < SAMPLES_PER_CELL:
                    cell["samples"][position] = sample

        if hour is not None:
            add_totals(hourly[(date, model, hour)], values)
        if context is not None:
            # Bounds are exclusive: 8k belongs to 8–32k, not <8k.
            bucket = sum(context >= bound for bound in CONTEXT_BOUNDS)
            add_totals(contexts[(date, model, bucket)], values)
        label = row.get("upstream_profile") or row.get("upstream_provider") or "(unlogged)"
        add_totals(upstream[(date, model, label)], values)
        add_totals(endpoints[(date, model, row.get("endpoint") or "(unknown)")], values)

    def pack(groups, dimensions):
        return [{**dict(zip(dimensions, key)), **values} for key, values in sorted(groups.items())]

    return {
        "dates": sorted({date for date, _ in cells}),
        "models": sorted(models),
        "total_rows": total,
        "model_overflow": model_overflow,
        "quality": quality,
        "timezone": "Server local time",
        "cells": pack(cells, ("date", "model")),
        "hourly": pack(hourly, ("date", "model", "hour")),
        "contexts": pack(contexts, ("date", "model", "bucket")),
        "upstream": pack(upstream, ("date", "model", "name")),
        "endpoints": pack(endpoints, ("date", "model", "name")),
        "duration_edges": DURATION_EDGES,
        "tps_edges": TPS_EDGES,
        "context_bounds": CONTEXT_BOUNDS,
        "context_labels": CONTEXT_LABELS,
        "total_fields": list(TOTAL_FIELDS),
        "samples_per_cell": SAMPLES_PER_CELL,
    }
