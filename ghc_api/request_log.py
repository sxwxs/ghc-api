"""The full-fidelity request log: ``<config dir>/requests/YYYY-MM-DD.jl``.

Every finished request is appended as one JSON line, exactly as the route
recorded it. Nothing is truncated, stripped or redacted on the way here; only
credentials are removed, by ``auth.redact_auth_headers``, before a route hands
its headers over.

Once a request finishes, this file is the only place its request and response
bodies live. ``RequestCache`` keeps per-request metadata plus a locator
(``{"file", "offset", "length"}``) that points at the line written here, and
the dashboard reads bodies back through :meth:`RequestLog.read`.

Daily files are deleted after ``request_log_retention_days`` days. ``0`` keeps
them forever. Cleanup runs at startup and again on the first append of each
new day; it only touches files named ``YYYY-MM-DD.jl`` and their
request-statistics sidecars.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .file_logging import append_jsonl_line
from .utils import get_config_dir


DEFAULT_RETENTION_DAYS = 7
RETENTION_KEY = "request_log_retention_days"
LEGACY_SAVE_KEY = "save_request_to_file"


class RequestLogError(Exception):
    """A request record could not be located or read back."""


# --------------------------------------------------------------- configuration


def resolve_retention_days(config: Mapping[str, Any]) -> Tuple[int, List[str]]:
    """Return the effective retention in days and any deprecation warnings.

    An explicit ``request_log_retention_days`` always wins. Without it, the
    deprecated ``save_request_to_file`` decides: ``true`` used to mean "keep
    every request on disk", so it maps to ``0`` (keep forever); ``false`` or a
    missing key gets the default retention.
    """
    warnings: List[str] = []
    has_legacy = LEGACY_SAVE_KEY in config
    legacy = config.get(LEGACY_SAVE_KEY)

    if RETENTION_KEY in config:
        days = validate_retention_days(config[RETENTION_KEY])
        if has_legacy:
            warnings.append(
                f"'{LEGACY_SAVE_KEY}' is deprecated and ignored because "
                f"'{RETENTION_KEY}' is set. Remove it from config.yaml."
            )
        return days, warnings

    if not has_legacy:
        return DEFAULT_RETENTION_DAYS, warnings

    if bool(legacy):
        warnings.append(
            f"'{LEGACY_SAVE_KEY}: true' is deprecated. It now means "
            f"'{RETENTION_KEY}: 0' (keep request logs forever). "
            f"Replace it with '{RETENTION_KEY}' in config.yaml."
        )
        return 0, warnings

    warnings.append(
        f"'{LEGACY_SAVE_KEY}: false' is deprecated and no longer turns request "
        f"logging off. Every request is now written to {requests_dir()} and "
        f"kept for {DEFAULT_RETENTION_DAYS} days. Set '{RETENTION_KEY}' in "
        "config.yaml to choose how many days to keep (0 keeps them forever), "
        f"and remove '{LEGACY_SAVE_KEY}'."
    )
    return DEFAULT_RETENTION_DAYS, warnings


def validate_retention_days(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"'{RETENTION_KEY}' must be an integer >= 0 (0 keeps request logs forever)"
        )
    return value


# ---------------------------------------------------------------------- paths


def requests_dir() -> Path:
    return Path(get_config_dir()) / "requests"


def file_date(filename: str) -> Optional[date]:
    """The day a daily request file covers, or None for any other file."""
    if len(filename) != len("YYYY-MM-DD.jl") or not filename.endswith(".jl"):
        return None
    try:
        return datetime.strptime(filename[:-3], "%Y-%m-%d").date()
    except ValueError:
        return None


def daily_files(directory: Optional[Path] = None) -> List[str]:
    """Daily request filenames, newest first."""
    directory = directory or requests_dir()
    try:
        names = [entry.name for entry in directory.iterdir() if entry.is_file()]
    except FileNotFoundError:
        return []
    return sorted((name for name in names if file_date(name) is not None), reverse=True)


def _resolve(filename: Any) -> Path:
    if not isinstance(filename, str) or file_date(filename) is None:
        raise RequestLogError(f"Invalid request log filename: {filename!r}")
    directory = requests_dir().resolve()
    path = (directory / filename).resolve()
    if path.parent != directory:
        raise RequestLogError(f"Request log file is outside the requests directory: {filename!r}")
    return path


# ------------------------------------------------------------------ record I/O


def metadata_view(record: Mapping[str, Any], locator: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The bodiless projection of a record that the in-memory cache keeps."""
    view = {key: record.get(key) for key in METADATA_FIELDS if key in record}
    view["log"] = locator
    return view


# Everything the request list, live view and stats need. Bodies, headers and
# protocol sidecars (conversion reports, recovered content, ...) stay on disk.
METADATA_FIELDS = (
    "id", "timestamp", "started_at", "client_ip", "model", "translated_model",
    "endpoint", "status_code", "request_size", "response_size", "input_tokens",
    "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
    "duration", "ttft", "tokens_per_second", "state", "user_id",
    "upstream_provider", "upstream_profile", "upstream_api",
)

# Fields full-text search looks in.
SEARCHABLE_FIELDS = ("original_request_body", "request_body", "response_body", "raw_events")


def _searchable_text(record: Mapping[str, Any]) -> str:
    """Lower-cased text full-text search matches against.

    Every string value is searched as-is (so quotes in a prompt, or a raw SSE
    event's literal ``"content":"hello"``, match), followed by the JSON form
    of the bodies for structural fragments such as ``"role": "user"``.
    """
    values = [record.get(key) for key in SEARCHABLE_FIELDS]
    strings: List[str] = []
    stack: List[Any] = [values]
    while stack:
        value = stack.pop()
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    strings.append(json.dumps(values, ensure_ascii=False))
    return "\n".join(strings).lower()


class RequestLog:
    def __init__(self) -> None:
        self._cleanup_lock = threading.Lock()
        self._last_cleanup_day: Optional[date] = None

    # ---------------------------------------------------------------- write

    def append(self, record: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """Append one record and return its locator, or None if it was not written.

        A failed write is reported and swallowed: logging must never break the
        response the client is already receiving.
        """
        today = datetime.now().date()
        self.cleanup_if_new_day(today)
        filename = f"{today.isoformat()}.jl"
        try:
            # default=str: a stray non-JSON value must not cost the whole record.
            line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
            directory = requests_dir()
            directory.mkdir(parents=True, exist_ok=True)
            offset, length = append_jsonl_line(directory / filename, line)
        except Exception as exc:
            print(f"[Request Log] Failed to append request: {exc}")
            return None
        return {"file": filename, "offset": offset, "length": length}

    # ----------------------------------------------------------------- read

    def read(self, locator: Mapping[str, Any]) -> Dict[str, Any]:
        """Read back the record a locator points at."""
        offset = locator.get("offset")
        length = locator.get("length")
        if (
            isinstance(offset, bool) or not isinstance(offset, int) or offset < 0
            or isinstance(length, bool) or not isinstance(length, int) or length <= 0
        ):
            raise RequestLogError("Invalid request log locator")
        path = _resolve(locator.get("file"))
        try:
            with path.open("rb") as file:
                file.seek(offset)
                raw = file.read(length)
        except FileNotFoundError as exc:
            raise RequestLogError("The request log file no longer exists (retention may have removed it)") from exc
        try:
            record = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RequestLogError("The locator does not point at a request record") from exc
        if not isinstance(record, dict):
            raise RequestLogError("The locator does not point at a request record")
        return record

    # --------------------------------------------------------------- search

    def search(
        self,
        query: str,
        *,
        since: Optional[float],
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Full-text search over persisted records, newest first.

        ``since`` is a Unix timestamp; None searches every retained file.
        Returns metadata views (with locators), not full records.
        """
        needle = query.lower()
        # How the needle looks inside a JSON-encoded line: quotes, backslashes
        # and newlines in string values (prompts, raw SSE events) are escaped.
        escaped_needle = json.dumps(needle, ensure_ascii=False)[1:-1]
        first_day = datetime.fromtimestamp(since).date() if since is not None else None
        matches: List[Dict[str, Any]] = []
        for filename in daily_files():
            day = file_date(filename)
            if first_day is not None and day is not None and day < first_day:
                break
            matches.extend(reversed(list(self._search_file(filename, needle, escaped_needle, since, user_id))))
        return matches

    def _search_file(
        self, filename: str, needle: str, escaped_needle: str,
        since: Optional[float], user_id: Optional[str],
    ) -> Iterable[Dict[str, Any]]:
        from .request_file_stats import _normalize_timestamp

        try:
            file = _resolve(filename).open("rb")
        except (OSError, RequestLogError):
            return
        with file:
            offset = 0
            for raw in file:
                start = offset
                offset += len(raw)
                if not raw.endswith(b"\n"):
                    break  # a record still being written
                text = raw.decode("utf-8", errors="replace")
                # Cheap pre-filter on the raw line before parsing it.
                lowered = text.lower()
                if needle not in lowered and escaped_needle not in lowered:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                # Logs written before 2026-03-23 store ISO-string timestamps.
                record["timestamp"] = _normalize_timestamp(record.get("timestamp"))
                if since is not None and record["timestamp"] < since:
                    continue
                if user_id is not None and (record.get("user_id") or "anonymous") != user_id:
                    continue
                if needle in _searchable_text(record):
                    yield metadata_view(record, {"file": filename, "offset": start, "length": len(raw)})

    # -------------------------------------------------------------- restore

    def recent_entries(self, limit: int) -> List[Dict[str, Any]]:
        """Metadata for the newest ``limit`` persisted requests, oldest first.

        Uses the request-statistics sidecar index, which already holds per-line
        metadata and byte locators, so restoring never parses request bodies
        more than once per line. Index rows follow file order, which is the
        order requests finished, and only the last ``limit`` rows of a file are
        kept while streaming it, so memory stays bounded however large a day is.
        """
        from .request_file_stats import build_or_load_file_index, iter_index_rows

        entries: List[Dict[str, Any]] = []
        for filename in daily_files():
            if len(entries) >= limit:
                break
            try:
                build_or_load_file_index(_resolve(filename))
                newest = deque(
                    filter(None, map(_entry_from_index_row, iter_index_rows(filename))),
                    maxlen=limit - len(entries),
                )
            except Exception as exc:
                print(f"[Request Log] Could not restore requests from {filename}: {exc}")
                continue
            entries = list(newest) + entries
        return entries

    # -------------------------------------------------------------- cleanup

    def cleanup(self, today: Optional[date] = None) -> List[str]:
        """Delete daily files older than the retention window. Returns their names."""
        from .request_file_stats import _index_path_for, _meta_path_for, request_stats_jobs
        from .state import state

        days = state.request_log_retention_days
        if days <= 0:
            return []
        today = today or datetime.now().date()
        # Keep today plus the previous days - 1 days.
        oldest_kept = today - timedelta(days=days - 1)
        busy = set(request_stats_jobs.active_files())
        removed: List[str] = []
        for filename in daily_files():
            day = file_date(filename)
            if day is None or day >= oldest_kept or filename in busy:
                continue
            path = requests_dir() / filename
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                print(f"[Request Log] Could not delete expired {filename}: {exc}")
                continue
            for sidecar in (_index_path_for(path), _meta_path_for(path)):
                try:
                    sidecar.unlink()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    print(f"[Request Log] Could not delete index for {filename}: {exc}")
            removed.append(filename)
        if removed:
            print(f"[Request Log] Removed {len(removed)} expired daily file(s) (retention {days} days)")
        return removed

    def cleanup_if_new_day(self, today: Optional[date] = None) -> None:
        today = today or datetime.now().date()
        with self._cleanup_lock:
            if self._last_cleanup_day == today:
                return
            self._last_cleanup_day = today
        try:
            self.cleanup(today)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[Request Log] Cleanup failed: {exc}")


def _seconds(milliseconds: Any) -> Optional[float]:
    return milliseconds / 1000 if isinstance(milliseconds, int) else None


def _entry_from_index_row(row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Turn a request-statistics sidecar row back into a cache metadata entry."""
    request_id = row.get("id")
    if not request_id:
        return None
    timestamp = row.get("timestamp") or 0
    return {
        "id": request_id,
        "timestamp": timestamp,
        "started_at": float(timestamp),
        "client_ip": row.get("client_ip"),
        "model": row.get("model") or "unknown",
        "translated_model": row.get("translated_model") or None,
        "endpoint": row.get("endpoint") or "unknown",
        "status_code": row.get("status_code"),
        "request_size": row.get("request_size") or 0,
        "response_size": row.get("response_size") or 0,
        "input_tokens": row.get("input_not_cached_tokens") or 0,
        "output_tokens": row.get("output_tokens") or 0,
        "cache_creation_input_tokens": row.get("cache_write_tokens") or 0,
        "cache_read_input_tokens": row.get("cache_read_tokens") or 0,
        "duration": _seconds(row.get("duration_ms")) or 0,
        "ttft": _seconds(row.get("ttft_ms")),
        "tokens_per_second": None,
        "state": row.get("state") or "completed",
        "user_id": row.get("user_id") or "anonymous",
        "upstream_provider": row.get("upstream_provider") or None,
        "upstream_profile": row.get("upstream_profile") or None,
        "upstream_api": row.get("upstream_api") or None,
        "log": {"file": row["source_file"], "offset": row["offset"], "length": row["length"]},
    }


request_log = RequestLog()
