"""One-shot rebuild of token_usage.jl from per-request JSONL files.

Why this exists
---------------
Earlier versions of ``TokenUsageReporter`` wrote per-tick rows that omitted
``cache_creation_input_tokens`` and ``cache_read_input_tokens`` entirely. The
dashboard's "Total Token" column therefore underreported any traffic that used
Anthropic prompt caching (``input_tokens`` on the wire only counts the
uncached-new slice; the cached portions are reported separately).

The per-request files at ``<config_dir>/requests/YYYY-MM-DD.jl`` always
carried the full per-request breakdown -- including cache fields -- because
``cache.complete_request`` writes them via ``_append_request_to_daily_file``.
This script rebuilds ``token_usage.jl`` from that ground truth so historical
dashboards become accurate.

Usage
-----
    python rebuild_token_usage.py                # rebuild with defaults
    python rebuild_token_usage.py --dry-run      # show what would change
    python rebuild_token_usage.py --requests-dir <path> --output <path>

The script refuses to overwrite an existing ``token_usage.jl`` unless
``--force`` is given AND the user has confirmed the live server is stopped --
otherwise the next reporter tick double-counts the current in-memory snapshot
on top of the rebuilt history. ``--dry-run`` is always safe.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple


# Match the live reporter's emit cadence so rebuilt rows align with any rows
# that may already have been written correctly. 300s = 5 minutes.
DEFAULT_BUCKET_SECONDS = 300

ANONYMOUS_USER_ID = "anonymous"


def _normalize_timestamp(value) -> int:
    """Coerce timestamp values to Unix seconds. Older request files used ISO 8601
    strings; newer ones use Unix seconds. Returns 0 if unparseable."""
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        try:
            return int(float(text))
        except ValueError:
            try:
                return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())
            except ValueError:
                return 0
    return 0


def _default_requests_dir() -> Path:
    """Mirror ``ghc_api.utils.get_config_dir`` without importing the package
    (the script is meant to be runnable in isolation)."""
    if os.name == "nt":
        return Path(os.path.expandvars("%APPDATA%/ghc-api")) / "requests"
    return Path.home() / ".ghc-api" / "requests"


def _default_output_path() -> Path:
    """Best-effort match of ``TokenUsageReporter._usage_file``. We can't resolve
    the OneDrive agent root without the package config, so fall back to
    ``~/.ghc-api/token_usage.jl`` and let the user override with ``--output``."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from ghc_api.config_sync import get_agent_root  # type: ignore
        agent_root = get_agent_root()
        if agent_root:
            return Path(agent_root) / "token_usage.jl"
    except Exception:
        pass
    return Path.home() / ".ghc-api" / "token_usage.jl"


def _coerce_user_id(value) -> str:
    if value is None:
        return ANONYMOUS_USER_ID
    text = str(value).strip()
    return text or ANONYMOUS_USER_ID


def _iter_request_files(requests_dir: Path) -> Iterable[Path]:
    if not requests_dir.is_dir():
        return []
    return sorted(requests_dir.glob("*.jl"))


def _aggregate(
    requests_dir: Path,
    bucket_seconds: int,
) -> Tuple[Dict[Tuple[int, str], Dict[str, Dict[str, int]]], Dict[str, int]]:
    """Walk every request line and bucket by (rounded_timestamp, user_id).

    Returns (buckets, stats) where:
      buckets[(ts, user_id)][model] = {request_count, input_tokens, ...}
      stats = {"files": N, "lines": N, "skipped": N}
    """
    buckets: Dict[Tuple[int, str], Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {
            "request_count": 0,
            "input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 0,
            "data_sent": 0,
            "data_received": 0,
        })
    )
    stats = {"files": 0, "lines": 0, "skipped": 0}

    for path in _iter_request_files(requests_dir):
        stats["files"] += 1
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        stats["skipped"] += 1
                        continue
                    stats["lines"] += 1

                    ts = _normalize_timestamp(entry.get("timestamp"))
                    if ts <= 0:
                        stats["skipped"] += 1
                        continue
                    bucket_ts = (ts // bucket_seconds) * bucket_seconds

                    user_id = _coerce_user_id(entry.get("user_id"))
                    # The reporter aggregates by translated_model when present,
                    # falling back to model. Same rule here.
                    model = entry.get("translated_model") or entry.get("model") or "unknown"

                    bucket = buckets[(bucket_ts, user_id)][str(model)]
                    bucket["request_count"] += 1
                    bucket["input_tokens"] += int(entry.get("input_tokens") or 0)
                    bucket["cache_creation_input_tokens"] += int(entry.get("cache_creation_input_tokens") or 0)
                    bucket["cache_read_input_tokens"] += int(entry.get("cache_read_input_tokens") or 0)
                    bucket["output_tokens"] += int(entry.get("output_tokens") or 0)
                    bucket["data_sent"] += int(entry.get("request_size") or 0)
                    bucket["data_received"] += int(entry.get("response_size") or 0)
        except OSError as e:
            print(f"[warn] could not read {path}: {e}", file=sys.stderr)

    return buckets, stats


def _emit_lines(
    buckets: Dict[Tuple[int, str], Dict[str, Dict[str, int]]],
) -> Iterable[str]:
    """Render rebuilt rows in the same shape token_usage_reporter writes."""
    for (ts, user_id), models in sorted(buckets.items()):
        payload_models = []
        for model_id, totals in sorted(models.items()):
            input_tokens = totals["input_tokens"]
            cache_create = totals["cache_creation_input_tokens"]
            cache_read = totals["cache_read_input_tokens"]
            output_tokens = totals["output_tokens"]
            data_sent = totals["data_sent"]
            data_received = totals["data_received"]
            payload_models.append({
                "model": model_id,
                "request_count": totals["request_count"],
                "input_tokens": input_tokens,
                "cache_creation_input_tokens": cache_create,
                "cache_read_input_tokens": cache_read,
                "output_tokens": output_tokens,
                # Match the post-fix total_tokens definition: every billed
                # input slice + output. Old reader code that didn't know about
                # cache fields will undercount, but the dashboard reader
                # recomputes from components either way.
                "total_tokens": input_tokens + cache_create + cache_read + output_tokens,
                "data_sent": data_sent,
                "data_received": data_received,
                "total_data": data_sent + data_received,
            })
        yield json.dumps({
            "timestamp": ts,
            "user_id": user_id,
            "models": payload_models,
        }, ensure_ascii=False) + "\n"


def _confirm(prompt: str) -> bool:
    try:
        reply = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError:
        return False
    return reply in ("y", "yes")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--requests-dir", type=Path, default=_default_requests_dir(),
                        help=f"Directory of per-request JSONL files (default: {_default_requests_dir()})")
    parser.add_argument("--output", type=Path, default=_default_output_path(),
                        help=f"Output token_usage.jl path (default: {_default_output_path()})")
    parser.add_argument("--bucket-seconds", type=int, default=DEFAULT_BUCKET_SECONDS,
                        help=f"Aggregation bucket width in seconds (default: {DEFAULT_BUCKET_SECONDS})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print rebuilt summary; do not write the output file")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output without interactive confirmation. "
                             "STOP THE SERVER FIRST -- otherwise the next reporter tick "
                             "double-counts the in-memory snapshot.")
    args = parser.parse_args(argv)

    if not args.requests_dir.is_dir():
        print(f"[error] requests directory not found: {args.requests_dir}", file=sys.stderr)
        return 2

    print(f"[info] scanning request files in: {args.requests_dir}")
    buckets, stats = _aggregate(args.requests_dir, args.bucket_seconds)
    rebuilt_rows = list(_emit_lines(buckets))

    print(f"[info] read {stats['lines']} request(s) across {stats['files']} file(s); "
          f"skipped {stats['skipped']} bad/timestamp-less line(s)")
    print(f"[info] rebuilt {len(rebuilt_rows)} aggregated row(s) "
          f"(bucket = {args.bucket_seconds}s, {len(buckets)} unique (timestamp,user) pair(s))")

    if not rebuilt_rows:
        print("[info] nothing to write")
        return 0

    if args.dry_run:
        preview = min(3, len(rebuilt_rows))
        print(f"[dry-run] first {preview} row(s):")
        for line in rebuilt_rows[:preview]:
            print(f"  {line.rstrip()}")
        print(f"[dry-run] target output: {args.output}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if not args.force:
            print(f"[error] {args.output} exists. Stop the ghc-api server first, then re-run with --force.",
                  file=sys.stderr)
            print("        (running while the server is up will double-count the live snapshot)",
                  file=sys.stderr)
            return 3
        if not _confirm(f"Overwrite {args.output}? Confirm the server is stopped"):
            print("[abort] user did not confirm")
            return 4
        backup = args.output.with_suffix(args.output.suffix + f".bak.{int(time.time())}")
        shutil.copy2(args.output, backup)
        print(f"[info] backed up existing file to: {backup}")

    with args.output.open("w", encoding="utf-8") as f:
        f.writelines(rebuilt_rows)
    print(f"[done] wrote {len(rebuilt_rows)} row(s) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
