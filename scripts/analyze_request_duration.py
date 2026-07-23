#!/usr/bin/env python3
"""Stream duration statistics from JSON Lines request logs."""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_NUMBER_PATTERN = br"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_DURATION_RE = re.compile(br'"duration"\s*:\s*(' + _NUMBER_PATTERN + br"|null)")
_TIMESTAMP_RE = re.compile(
    br'"timestamp"\s*:\s*('
    + _NUMBER_PATTERN
    + br'|"(?:\\.|[^"\\])*")'
)
_JSON_STRING_OR_NULL_PATTERN = br'("(?:\\.|[^"\\])*"|null)'
_MODEL_RE = re.compile(br'"model"\s*:\s*' + _JSON_STRING_OR_NULL_PATTERN)
_TRANSLATED_MODEL_RE = re.compile(
    br'"translated_model"\s*:\s*' + _JSON_STRING_OR_NULL_PATTERN
)
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


def parse_timestamp(raw_value: bytes) -> float | None:
    """Normalize a numeric or ISO-8601 JSON timestamp to Unix seconds."""
    try:
        value = json.loads(raw_value)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        timestamp = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            timestamp = float(text)
        except ValueError:
            try:
                timestamp = datetime.fromisoformat(
                    text.replace("Z", "+00:00")
                ).timestamp()
            except (ValueError, OverflowError):
                return None
    else:
        return None

    return timestamp if math.isfinite(timestamp) else None


def parse_model(line: bytes, end: int) -> str:
    """Return the effective top-level model preceding the duration field."""
    for field, pattern in (
        (b'"translated_model"', _TRANSLATED_MODEL_RE),
        (b'"model"', _MODEL_RE),
    ):
        index = line.rfind(field, 0, end)
        if index < 0:
            continue
        match = pattern.match(line, index)
        if match is None or match.group(1) == b"null":
            continue
        try:
            value = json.loads(match.group(1))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


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


def expand_paths(patterns: list[str]) -> list[Path]:
    """Expand file paths and glob patterns, returning unique regular files."""
    paths: list[Path] = []
    seen: set[Path] = set()
    unmatched: list[str] = []

    for pattern in patterns:
        matches = sorted(Path(match) for match in glob.glob(pattern, recursive=True))
        files = [path for path in matches if path.is_file()]
        if not files:
            unmatched.append(pattern)
            continue
        for path in files:
            normalized = path.resolve()
            if normalized not in seen:
                seen.add(normalized)
                paths.append(path)

    if unmatched:
        raise ValueError(f"no files matched: {', '.join(unmatched)}")
    return paths


def analyze(
    path: Path, interval_seconds: float | None = None
) -> tuple[
    list[float], dict[float, list[float]], dict[str, list[float]], int, int, int
]:
    durations: list[float] = []
    buckets: dict[float, list[float]] = defaultdict(list)
    model_durations: dict[str, list[float]] = defaultdict(list)
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
            if interval_seconds is None:
                model_durations[parse_model(line, index)].append(value)
            else:
                # The Unix timestamp is near the start of the line, so scan
                # only the first 1 KiB.
                timestamp_match = _TIMESTAMP_RE.search(line, 0, min(len(line), 1024))
                if timestamp_match is None:
                    missing_timestamp += 1
                    continue
                timestamp = parse_timestamp(timestamp_match.group(1))
                if timestamp is None:
                    missing_timestamp += 1
                    continue
                bucket_start = math.floor(timestamp / interval_seconds) * interval_seconds
                buckets[bucket_start].append(value)

    return (
        durations,
        buckets,
        model_durations,
        missing_or_null,
        invalid,
        missing_timestamp,
    )


def analyze_paths(
    paths: list[Path], interval_seconds: float | None = None
) -> tuple[
    list[float], dict[float, list[float]], dict[str, list[float]], int, int, int
]:
    durations: list[float] = []
    buckets: dict[float, list[float]] = defaultdict(list)
    model_durations: dict[str, list[float]] = defaultdict(list)
    missing_or_null = 0
    invalid = 0
    missing_timestamp = 0

    for path in paths:
        (
            file_durations,
            file_buckets,
            file_model_durations,
            file_missing_or_null,
            file_invalid,
            file_missing_timestamp,
        ) = analyze(path, interval_seconds)
        durations.extend(file_durations)
        for start, values in file_buckets.items():
            buckets[start].extend(values)
        for model, values in file_model_durations.items():
            model_durations[model].extend(values)
        missing_or_null += file_missing_or_null
        invalid += file_invalid
        missing_timestamp += file_missing_timestamp

    return (
        durations,
        buckets,
        model_durations,
        missing_or_null,
        invalid,
        missing_timestamp,
    )


def print_stats(values: list[float]) -> None:
    values.sort()
    print(f"Average duration: {statistics.fmean(values):.3f} seconds")
    print(f"P50: {percentile(values, 50):.3f} seconds")
    print(f"P90: {percentile(values, 90):.3f} seconds")
    print(f"P99: {percentile(values, 99):.3f} seconds")
    print(f"P999: {percentile(values, 99.9):.3f} seconds")


def print_model_stats_tsv(
    model_durations: dict[str, list[float]],
    path: Path | None = None,
    print_header: bool = True,
) -> None:
    if print_header:
        prefix = "File\t" if path is not None else ""
        print(f"{prefix}Model\tRequests\tAvg/sec\tP50/sec\tP90/sec\tP99/sec\tP999/sec")
    for model, values in sorted(model_durations.items()):
        values.sort()
        file_prefix = f"{path}\t" if path is not None else ""
        print(
            f"{file_prefix}{model}\t{len(values)}\t"
            f"{statistics.fmean(values):.3f}\t"
            f"{percentile(values, 50):.3f}\t{percentile(values, 90):.3f}\t"
            f"{percentile(values, 99):.3f}\t{percentile(values, 99.9):.3f}"
        )


def format_utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def print_interval_stats(
    paths: list[Path],
    result: tuple[
        list[float], dict[float, list[float]], dict[str, list[float]], int, int, int
    ],
    interval_seconds: float,
) -> None:
    durations, buckets, _, missing_or_null, invalid, missing_timestamp = result
    if len(paths) == 1:
        print(f"File: {paths[0]}")
    else:
        print(f"Files: {len(paths):,}")
    print(f"Valid requests: {len(durations):,}")
    print(f"Missing or null: {missing_or_null:,}")
    print(f"Invalid values: {invalid:,}")
    print(f"Missing or invalid timestamps: {missing_timestamp:,}")
    print(f"Interval: {interval_seconds:g} seconds (UTC)")
    print()
    print(
        f"{'Period (UTC, start inclusive)':41} {'Requests':>8} "
        f"{'Avg/sec':>10} {'P50/sec':>10} {'P90/sec':>10} "
        f"{'P99/sec':>10} {'P999/sec':>10}"
    )
    for start, values in sorted(buckets.items()):
        values.sort()
        period = f"{format_utc(start)} ~ {format_utc(start + interval_seconds)}"
        print(
            f"{period:41} {len(values):8d} "
            f"{statistics.fmean(values):10.3f} "
            f"{percentile(values, 50):10.3f} "
            f"{percentile(values, 90):10.3f} "
            f"{percentile(values, 99):10.3f} "
            f"{percentile(values, 99.9):10.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate duration statistics for JSON Lines request logs"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["2026-07-19.jl"],
        metavar="PATH",
        help="request log paths or glob patterns, such as 2026-07-*.jl",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="aggregate all matched files; by default each file is calculated separately",
    )
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
    try:
        paths = expand_paths(args.paths)
    except ValueError as exc:
        parser.error(str(exc))

    if args.aggregate:
        result = analyze_paths(paths, args.interval)
        if not result[0]:
            raise SystemExit("no valid duration values found")
        if args.interval is None:
            print_model_stats_tsv(result[2])
        else:
            print_interval_stats(paths, result, args.interval)
        return

    found_valid_duration = False
    for index, path in enumerate(paths):
        result = analyze(path, args.interval)
        found_valid_duration = found_valid_duration or bool(result[0])
        if args.interval is None:
            print_model_stats_tsv(result[2], path, print_header=index == 0)
        else:
            if index:
                print()
            print_interval_stats([path], result, args.interval)
        sys.stdout.flush()

    if not found_valid_duration:
        raise SystemExit("no valid duration values found")


if __name__ == "__main__":
    main()
