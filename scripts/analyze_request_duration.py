#!/usr/bin/env python3
"""Stream duration statistics from JSON Lines request logs."""

from __future__ import annotations

import argparse
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_NUMBER_PATTERN = br"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_DURATION_RE = re.compile(br'"duration"\s*:\s*(' + _NUMBER_PATTERN + br"|null)")
_TIMESTAMP_RE = re.compile(br'"timestamp"\s*:\s*(' + _NUMBER_PATTERN + br")")
_INTERVAL_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s*([smhd]?)$", re.IGNORECASE)


def parse_interval(value: str) -> float:
    """Parse values such as 300s, 5m, 1h, or 1d; default to minutes."""
    match = _INTERVAL_RE.fullmatch(value.strip())
    if match is None:
        raise argparse.ArgumentTypeError(
            "interval must be formatted as 300s, 5m, 1h, or 1d"
        )

    amount = float(match.group(1))
    unit = match.group(2).lower() or "m"
    seconds = amount * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    if seconds <= 0:
        raise argparse.ArgumentTypeError("interval must be greater than 0")
    return seconds


def percentile(sorted_values: list[float], percent: float) -> float:
    """Calculate a percentile using NumPy-style linear interpolation."""
    if not sorted_values:
        raise ValueError("no data available for percentile calculation")

    position = (len(sorted_values) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]

    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def analyze(
    path: Path, interval_seconds: float | None = None
) -> tuple[list[float], dict[float, list[float]], int, int, int]:
    durations: list[float] = []
    buckets: dict[float, list[float]] = defaultdict(list)
    missing_or_null = 0
    invalid = 0
    missing_timestamp = 0

    # Read line by line because the file may be large. Duration is near the end
    # of each record, so use rfind to avoid parsing a potentially huge body.
    with path.open("rb") as file:
        for line in file:
            index = line.rfind(b'"duration"')
            if index < 0:
                missing_or_null += 1
                continue

            match = _DURATION_RE.match(line, index)
            if match is None:
                invalid += 1
                continue

            raw_value = match.group(1)
            if raw_value == b"null":
                missing_or_null += 1
                continue

            value = float(raw_value)
            if not math.isfinite(value):
                invalid += 1
                continue

            durations.append(value)
            if interval_seconds is not None:
                # The Unix timestamp is near the start of the line, so scan
                # only the first 1 KiB.
                timestamp_match = _TIMESTAMP_RE.search(line, 0, min(len(line), 1024))
                if timestamp_match is None:
                    missing_timestamp += 1
                    continue
                timestamp = float(timestamp_match.group(1))
                if not math.isfinite(timestamp):
                    missing_timestamp += 1
                    continue
                bucket_start = math.floor(timestamp / interval_seconds) * interval_seconds
                buckets[bucket_start].append(value)

    return durations, buckets, missing_or_null, invalid, missing_timestamp


def print_stats(values: list[float]) -> None:
    values.sort()
    print(f"Average duration: {statistics.fmean(values):.3f} seconds")
    print(f"P50: {percentile(values, 50):.3f} seconds")
    print(f"P90: {percentile(values, 90):.3f} seconds")
    print(f"P99: {percentile(values, 99):.3f} seconds")


def format_utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate duration statistics for JSON Lines request logs"
    )
    parser.add_argument("path", nargs="?", default="2026-07-19.jl", type=Path)
    parser.add_argument(
        "--interval",
        type=parse_interval,
        metavar="LENGTH",
        help=(
            "group statistics by time interval, such as 300s, 5m, or 1h; "
            "values without a unit are interpreted as minutes"
        ),
    )
    args = parser.parse_args()

    durations, buckets, missing_or_null, invalid, missing_timestamp = analyze(
        args.path, args.interval
    )
    if not durations:
        raise SystemExit("no valid duration values found")

    print(f"File: {args.path}")
    print(f"Valid requests: {len(durations):,}")
    print(f"Missing or null: {missing_or_null:,}")
    print(f"Invalid values: {invalid:,}")

    if args.interval is None:
        print_stats(durations)
        return

    print(f"Missing or invalid timestamps: {missing_timestamp:,}")
    print(f"Interval: {args.interval:g} seconds (UTC)")
    print()
    print(
        f"{'Period (UTC, start inclusive)':41} {'Requests':>8} "
        f"{'Avg/sec':>10} {'P50/sec':>10} {'P90/sec':>10} {'P99/sec':>10}"
    )
    for start, values in sorted(buckets.items()):
        values.sort()
        period = f"{format_utc(start)} ~ {format_utc(start + args.interval)}"
        print(
            f"{period:41} {len(values):8d} "
            f"{statistics.fmean(values):10.3f} "
            f"{percentile(values, 50):10.3f} "
            f"{percentile(values, 90):10.3f} "
            f"{percentile(values, 99):10.3f}"
        )


if __name__ == "__main__":
    main()
