"""Line-level request-file indexes, in-memory datasets, and scan jobs."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode

from .utils import get_config_dir


INDEX_SCHEMA_VERSION = 3
INDEX_DIR_NAME = ".request-stats-index"
REQUEST_FILENAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.jl$")
MAX_SELECTED_FILES = 366
MAX_MODELS_PER_DATASET = 500
MAX_MODEL_NAME_LENGTH = 256
MAX_METRIC_VALUE = (1 << 63) - 1
MAX_LINE_BYTES = 64 * 1024 * 1024
MAX_DETAIL_LINE_BYTES = 4 * 1024 * 1024
DISCARD_CHUNK_BYTES = 1024 * 1024
SIGNATURE_CHUNK_BYTES = 4 * 1024 * 1024
INDEX_REPLACE_RETRIES = 10
INDEX_REPLACE_RETRY_SECONDS = 0.05
PROGRESS_BYTES_INTERVAL = 4 * 1024 * 1024
PROGRESS_SECONDS_INTERVAL = 0.25
MAX_DATASET_ROWS = 1_000_000
MAX_DATASET_ESTIMATED_BYTES = 256 * 1024 * 1024
MAX_DATASETS = 3
DATASET_TTL_SECONDS = 900
MAX_BUCKET_PAGE_SIZE = 200
OTHER_MODEL_KEY = "__other__"
UNKNOWN_MODEL_KEY = "unknown"
UNKNOWN_STATUS_KEY = "unknown"

INPUT_NOT_CACHED = "input_not_cached_tokens"
CACHE_WRITE = "cache_write_tokens"
CACHE_READ = "cache_read_tokens"
OUTPUT = "output_tokens"
TOTAL_BILLED = "total_billed_tokens"

TOKEN_FIELD_BY_METRIC = {
    INPUT_NOT_CACHED: "input_tokens",
    CACHE_WRITE: "cache_creation_input_tokens",
    CACHE_READ: "cache_read_input_tokens",
    OUTPUT: "output_tokens",
}

REQUEST_SIZE_BOUNDS = [0, 511] + [(1 << power) - 1 for power in range(10, 27)]
DURATION_BOUNDS = [100, 250, 500, 1000, 2000, 5000, 10000, 30000, 60000, 120000, 300000, 600000]
TOKEN_BOUNDS = [0, 16] + [1 << power for power in range(5, 18)]

METRIC_DEFINITIONS = {
    "request_size": {"unit": "bytes", "upper_bounds": REQUEST_SIZE_BOUNDS},
    "duration_ms": {"unit": "milliseconds", "upper_bounds": DURATION_BOUNDS},
    INPUT_NOT_CACHED: {"unit": "tokens", "upper_bounds": TOKEN_BOUNDS},
    CACHE_WRITE: {"unit": "tokens", "upper_bounds": TOKEN_BOUNDS},
    CACHE_READ: {"unit": "tokens", "upper_bounds": TOKEN_BOUNDS},
    OUTPUT: {"unit": "tokens", "upper_bounds": TOKEN_BOUNDS},
    TOTAL_BILLED: {"unit": "tokens", "upper_bounds": TOKEN_BOUNDS},
}

METRIC_BUCKET_FIELDS = {
    metric_name: f"{metric_name}_bucket"
    for metric_name in METRIC_DEFINITIONS
}

QUALITY_FIELDS = (
    "lines_seen",
    "records_indexed",
    "blank_lines",
    "invalid_json_lines",
    "non_object_lines",
    "oversize_lines",
    "partial_tail_bytes",
)

ProgressCallback = Callable[[str, int, int], None]


class RequestStatsError(Exception):
    """Base exception for request-file statistics."""


class RequestStatsValidationError(RequestStatsError, ValueError):
    """Raised when a request-stats argument is invalid."""


class RequestStatsBusyError(RequestStatsError):
    """Raised when a different scan job is already active."""

    def __init__(self, active_job_id: str):
        super().__init__("Another request statistics job is already running.")
        self.active_job_id = active_job_id


class RequestStatsJobNotFound(RequestStatsError):
    """Raised when a scan job id is unknown or expired."""


class RequestStatsDatasetNotFound(RequestStatsError):
    """Raised when an in-memory dataset is unknown or expired."""


class RequestStatsCancelled(RequestStatsError):
    """Raised internally when a scan job is cancelled."""


class RequestFileChangedError(RequestStatsError):
    """Raised when a source file changed or a stable detail locator is stale."""


class RequestIndexValidationError(RequestStatsError):
    """Raised when a persisted sidecar index is invalid."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _requests_dir() -> Path:
    return Path(get_config_dir()) / "requests"


def _index_dir(requests_dir: Optional[Path] = None) -> Path:
    return (requests_dir or _requests_dir()) / INDEX_DIR_NAME


def _index_path_for(request_path: Path) -> Path:
    return _index_dir(request_path.parent) / f"{request_path.name}.index.jl"


def _meta_path_for(request_path: Path) -> Path:
    return _index_dir(request_path.parent) / f"{request_path.name}.meta.json"


def _is_valid_request_filename(filename: Any) -> bool:
    if not isinstance(filename, str) or not REQUEST_FILENAME_RE.fullmatch(filename):
        return False
    try:
        datetime.strptime(filename[:-3], "%Y-%m-%d")
    except ValueError:
        return False
    return True


def resolve_request_file(filename: str, *, require_exists: bool = True) -> Path:
    """Resolve one daily request file without allowing directory traversal."""
    if not _is_valid_request_filename(filename):
        raise RequestStatsValidationError(f"Invalid request filename: {filename!r}")

    requests_dir = _requests_dir().resolve()
    candidate = (requests_dir / filename).resolve()
    if candidate.parent != requests_dir:
        raise RequestStatsValidationError(f"Request file is outside the requests directory: {filename!r}")
    if require_exists and not candidate.is_file():
        raise RequestStatsValidationError(f"Request file does not exist: {filename}")
    return candidate


def normalize_selected_files(filenames: Any) -> List[str]:
    """Validate, deduplicate, and preserve order for a request file selection."""
    if not isinstance(filenames, list) or not filenames:
        raise RequestStatsValidationError("'files' must be a non-empty list of request filenames")
    if len(filenames) > MAX_SELECTED_FILES:
        raise RequestStatsValidationError(f"Select at most {MAX_SELECTED_FILES} request files")

    selected: List[str] = []
    seen = set()
    for filename in filenames:
        if not isinstance(filename, str):
            raise RequestStatsValidationError("Every selected request filename must be a string")
        filename = filename.strip()
        if filename in seen:
            continue
        resolve_request_file(filename)
        seen.add(filename)
        selected.append(filename)

    if not selected:
        raise RequestStatsValidationError("Select at least one request file")
    return selected


def _new_quality() -> Dict[str, int]:
    return {field: 0 for field in QUALITY_FIELDS}


def _normalize_timestamp(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        try:
            return max(0, int(value))
        except (TypeError, ValueError, OverflowError):
            return 0
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        try:
            return max(0, int(float(text)))
        except (ValueError, OverflowError):
            try:
                return max(0, int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()))
            except (ValueError, OverflowError):
                return 0
    return 0


def _coerce_nonnegative_int(entry: Dict[str, Any], field_name: str) -> Optional[int]:
    if field_name not in entry or entry.get(field_name) is None:
        return None
    value = entry.get(field_name)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        result = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        result = int(value)
    elif isinstance(value, str):
        try:
            result = int(value.strip(), 10)
        except (TypeError, ValueError):
            return None
    else:
        return None
    if result < 0 or result > MAX_METRIC_VALUE:
        return None
    return result


def _coerce_duration_ms(entry: Dict[str, Any]) -> Optional[int]:
    if "duration" not in entry or entry.get("duration") is None:
        return None
    value = entry.get("duration")
    if isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    milliseconds = int(round(seconds * 1000))
    return milliseconds if milliseconds <= MAX_METRIC_VALUE else None


def _effective_model(entry: Dict[str, Any]) -> str:
    model = entry.get("translated_model") or entry.get("model")
    if not isinstance(model, str) or not model.strip():
        return UNKNOWN_MODEL_KEY
    model = model.strip()
    return model if len(model) <= MAX_MODEL_NAME_LENGTH else OTHER_MODEL_KEY


def _effective_status_code(entry: Dict[str, Any]) -> Optional[int]:
    value = _coerce_nonnegative_int(entry, "status_code")
    return value if value is not None and 100 <= value <= 599 else None


def _bucket_index(metric_name: str, value: int) -> int:
    return bisect.bisect_left(METRIC_DEFINITIONS[metric_name]["upper_bounds"], value)


def _index_number(value: Optional[int], metric_name: str) -> Optional[int]:
    return _bucket_index(metric_name, value) if value is not None else None


def _extract_index_row(
    entry: Dict[str, Any],
    source_file: str,
    offset: int,
    length: int,
    raw_line: bytes,
) -> Dict[str, Any]:
    token_values = {
        metric_name: _coerce_nonnegative_int(entry, field_name)
        for metric_name, field_name in TOKEN_FIELD_BY_METRIC.items()
    }
    total_billed = None
    if any(field_name in entry and entry.get(field_name) is not None for field_name in TOKEN_FIELD_BY_METRIC.values()):
        invalid_present_component = any(
            field_name in entry and entry.get(field_name) is not None and token_values[metric_name] is None
            for metric_name, field_name in TOKEN_FIELD_BY_METRIC.items()
        )
        if not invalid_present_component:
            total = sum(value or 0 for value in token_values.values())
            total_billed = total if total <= MAX_METRIC_VALUE else None

    request_size = _coerce_nonnegative_int(entry, "request_size")
    response_size = _coerce_nonnegative_int(entry, "response_size")
    duration_ms = _coerce_duration_ms(entry)
    metric_values = {
        "request_size": request_size,
        "duration_ms": duration_ms,
        **token_values,
        TOTAL_BILLED: total_billed,
    }
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "source_file": source_file,
        "offset": offset,
        "length": length,
        "line_sha256": hashlib.sha256(raw_line).hexdigest(),
        "id": str(entry.get("id") or ""),
        "timestamp": _normalize_timestamp(entry.get("timestamp")),
        "model": str(entry.get("model") or ""),
        "translated_model": str(entry.get("translated_model") or ""),
        "effective_model": _effective_model(entry),
        "endpoint": str(entry.get("endpoint") or ""),
        "status_code": _effective_status_code(entry),
        "duration_ms": duration_ms,
        "request_size": request_size,
        "response_size": response_size,
        INPUT_NOT_CACHED: token_values[INPUT_NOT_CACHED],
        CACHE_WRITE: token_values[CACHE_WRITE],
        CACHE_READ: token_values[CACHE_READ],
        OUTPUT: token_values[OUTPUT],
        TOTAL_BILLED: total_billed,
        "user_id": str(entry.get("user_id") or "anonymous"),
        "client_ip": str(entry.get("client_ip") or "unknown"),
        **{
            METRIC_BUCKET_FIELDS[metric_name]: _index_number(value, metric_name)
            for metric_name, value in metric_values.items()
        },
    }


def _check_cancelled(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RequestStatsCancelled("Request statistics job was cancelled")


def _read_bounded_line(
    file,
    remaining: int,
    cancel_event: Optional[threading.Event] = None,
) -> Tuple[Optional[bytes], int, bool, bool]:
    first_limit = min(remaining, MAX_LINE_BYTES + 1)
    first = file.readline(first_limit)
    if not first:
        return b"", 0, False, False
    consumed = len(first)
    oversize = consumed > MAX_LINE_BYTES
    complete = first.endswith(b"\n")
    if not oversize:
        return first, consumed, complete, False
    while not complete and consumed < remaining:
        _check_cancelled(cancel_event)
        chunk = file.readline(min(remaining - consumed, DISCARD_CHUNK_BYTES))
        if not chunk:
            break
        consumed += len(chunk)
        complete = chunk.endswith(b"\n")
    return None, consumed, complete, True


def _content_signature(path: Path, length: int) -> str:
    digest = hashlib.sha256()
    remaining = length
    with path.open("rb") as file:
        while remaining > 0:
            chunk = file.read(min(remaining, SIGNATURE_CHUNK_BYTES))
            if not chunk:
                raise RequestFileChangedError(f"Request file changed while hashing: {path.name}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _atomic_replace(source: Path, target: Path) -> None:
    for attempt in range(INDEX_REPLACE_RETRIES):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt + 1 >= INDEX_REPLACE_RETRIES:
                raise
            time.sleep(INDEX_REPLACE_RETRY_SECONDS * (attempt + 1))


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
        _atomic_replace(temporary_path, path)
    except Exception:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise


def _load_meta(request_path: Path) -> Dict[str, Any]:
    meta_path = _meta_path_for(request_path)
    try:
        with meta_path.open("r", encoding="utf-8") as file:
            meta = json.load(file)
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise RequestIndexValidationError(str(exc)) from exc
    if not isinstance(meta, dict) or meta.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise RequestIndexValidationError("Unsupported request index metadata")
    if meta.get("source_file") != request_path.name:
        raise RequestIndexValidationError("Request index metadata source does not match")
    for field in ("size_bytes", "processed_bytes", "row_count", "index_size_bytes"):
        value = meta.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RequestIndexValidationError(f"Invalid request index metadata field: {field}")
    if meta["processed_bytes"] > meta["size_bytes"]:
        raise RequestIndexValidationError("Request index processed beyond source size")
    if not isinstance(meta.get("content_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", meta["content_sha256"]):
        raise RequestIndexValidationError("Invalid request index content hash")
    quality = meta.get("quality")
    if not isinstance(quality, dict) or any(not isinstance(quality.get(field), int) for field in QUALITY_FIELDS):
        raise RequestIndexValidationError("Invalid request index quality counters")
    index_path = _index_path_for(request_path)
    try:
        if index_path.stat().st_size != meta["index_size_bytes"]:
            raise RequestIndexValidationError("Request index sidecar size does not match metadata")
    except FileNotFoundError as exc:
        raise RequestIndexValidationError("Request index sidecar is missing") from exc
    return meta


def _write_index_rows(file, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        count += 1
    return count


def _scan_to_sidecar(
    request_path: Path,
    start_offset: int,
    snapshot_size: int,
    output_file,
    quality: Dict[str, int],
    cancel_event: Optional[threading.Event],
    progress: Optional[ProgressCallback],
    mode: str,
) -> Tuple[int, int]:
    processed_bytes = start_offset
    rows_written = 0
    last_progress_bytes = start_offset
    last_progress_time = time.monotonic()
    with request_path.open("rb") as source:
        source.seek(start_offset)
        while processed_bytes < snapshot_size:
            _check_cancelled(cancel_event)
            line_start = processed_bytes
            raw, consumed, complete, oversize = _read_bounded_line(
                source,
                snapshot_size - line_start,
                cancel_event,
            )
            if consumed == 0:
                raise RequestFileChangedError(f"Request file became shorter while scanning: {request_path.name}")
            if not complete:
                quality["partial_tail_bytes"] = snapshot_size - line_start
                break
            processed_bytes = line_start + consumed
            quality["lines_seen"] += 1
            if oversize:
                quality["oversize_lines"] += 1
            elif raw is not None and not raw.strip():
                quality["blank_lines"] += 1
            else:
                try:
                    entry = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                    quality["invalid_json_lines"] += 1
                else:
                    if not isinstance(entry, dict):
                        quality["non_object_lines"] += 1
                    else:
                        row = _extract_index_row(entry, request_path.name, line_start, consumed, raw)
                        output_file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
                        rows_written += 1
                        quality["records_indexed"] += 1
            now = time.monotonic()
            if progress is not None and (
                processed_bytes - last_progress_bytes >= PROGRESS_BYTES_INTERVAL
                or now - last_progress_time >= PROGRESS_SECONDS_INTERVAL
            ):
                progress(mode, processed_bytes, snapshot_size)
                last_progress_bytes = processed_bytes
                last_progress_time = now
    if progress is not None:
        progress(mode, processed_bytes, snapshot_size)
    return processed_bytes, rows_written


def _new_meta(
    request_path: Path,
    snapshot_size: int,
    source_mtime_ns: int,
    processed_bytes: int,
    row_count: int,
    index_size_bytes: int,
    quality: Dict[str, int],
) -> Dict[str, Any]:
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "source_file": request_path.name,
        "size_bytes": snapshot_size,
        "processed_bytes": processed_bytes,
        "mtime_ns": source_mtime_ns,
        "content_sha256": _content_signature(request_path, snapshot_size),
        "row_count": row_count,
        "index_size_bytes": index_size_bytes,
        "quality": quality,
    }


def build_or_load_file_index(
    request_path: Path,
    *,
    cancel_event: Optional[threading.Event] = None,
    progress: Optional[ProgressCallback] = None,
) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
    """Reuse, append, or rebuild one line-level request sidecar."""
    _check_cancelled(cancel_event)
    stat = request_path.stat()
    snapshot_size = stat.st_size
    warnings: List[Dict[str, Any]] = []
    meta: Optional[Dict[str, Any]] = None
    try:
        meta = _load_meta(request_path)
    except FileNotFoundError:
        pass
    except RequestIndexValidationError as exc:
        warnings.append({"code": "index_rebuilt", "file": request_path.name, "message": str(exc)})

    if meta is not None and meta["size_bytes"] == snapshot_size and meta.get("mtime_ns") == stat.st_mtime_ns:
        if progress is not None:
            progress("cached", snapshot_size, snapshot_size)
        return meta, "cached", warnings

    mode = "rebuild"
    if meta is not None and snapshot_size > meta["size_bytes"]:
        try:
            old_content_matches = _content_signature(request_path, meta["size_bytes"]) == meta["content_sha256"]
        except RequestFileChangedError:
            old_content_matches = False
        if old_content_matches:
            mode = "incremental"
        else:
            warnings.append({
                "code": "source_changed",
                "file": request_path.name,
                "message": "Indexed source bytes changed; the sidecar was rebuilt.",
            })
    elif meta is not None and snapshot_size < meta["size_bytes"]:
        warnings.append({
            "code": "source_shrank",
            "file": request_path.name,
            "message": "The request file became smaller; the sidecar was rebuilt.",
        })
    elif meta is not None and snapshot_size == meta["size_bytes"]:
        try:
            same_content = _content_signature(request_path, snapshot_size) == meta["content_sha256"]
        except RequestFileChangedError:
            same_content = False
        if same_content:
            meta["mtime_ns"] = stat.st_mtime_ns
            if progress is not None:
                progress("cached", snapshot_size, snapshot_size)
            _atomic_write_json(_meta_path_for(request_path), meta)
            return meta, "cached", warnings
        warnings.append({
            "code": "source_changed",
            "file": request_path.name,
            "message": "The request file changed without changing size; the sidecar was rebuilt.",
        })

    index_path = _index_path_for(request_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{index_path.name}.", suffix=".tmp", dir=str(index_path.parent))
    temporary_path = Path(temporary_name)
    try:
        if mode == "incremental" and meta is not None:
            with index_path.open("rb") as existing, os.fdopen(fd, "wb") as output:
                while True:
                    chunk = existing.read(SIGNATURE_CHUNK_BYTES)
                    if not chunk:
                        break
                    output.write(chunk)
                output.flush()
                start_offset = meta["processed_bytes"]
                quality = dict(meta["quality"])
                quality["partial_tail_bytes"] = 0
                with temporary_path.open("a", encoding="utf-8") as output_text:
                    processed_bytes, added_rows = _scan_to_sidecar(
                        request_path,
                        start_offset,
                        snapshot_size,
                        output_text,
                        quality,
                        cancel_event,
                        progress,
                        mode,
                    )
                    output_text.flush()
                    os.fsync(output_text.fileno())
                row_count = meta["row_count"] + added_rows
        else:
            quality = _new_quality()
            with os.fdopen(fd, "w", encoding="utf-8") as output_text:
                processed_bytes, row_count = _scan_to_sidecar(
                    request_path,
                    0,
                    snapshot_size,
                    output_text,
                    quality,
                    cancel_event,
                    progress,
                    mode,
                )
                output_text.flush()
                os.fsync(output_text.fileno())

        _check_cancelled(cancel_event)
        final_stat = request_path.stat()
        if final_stat.st_size < snapshot_size:
            raise RequestFileChangedError(f"Request file was truncated while scanning: {request_path.name}")
        if final_stat.st_size > snapshot_size:
            warnings.append({
                "code": "file_grew",
                "file": request_path.name,
                "message": "The request file grew during indexing; run statistics again for appended data.",
                "indexed_size_bytes": snapshot_size,
                "current_size_bytes": final_stat.st_size,
            })
        temporary_size = temporary_path.stat().st_size
        new_meta = _new_meta(
            request_path,
            snapshot_size,
            stat.st_mtime_ns,
            processed_bytes,
            row_count,
            temporary_size,
            quality,
        )
        _atomic_replace(temporary_path, index_path)
        _atomic_write_json(_meta_path_for(request_path), new_meta)
        return new_meta, mode, warnings
    except Exception:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise


def _index_status(request_path: Path, source_size: Optional[int] = None) -> Tuple[str, Optional[Dict[str, Any]]]:
    try:
        meta = _load_meta(request_path)
    except FileNotFoundError:
        return "missing", None
    except RequestIndexValidationError:
        return "error", None
    if source_size is None:
        try:
            source_size = request_path.stat().st_size
        except OSError:
            return "error", None
    return ("ready" if meta["size_bytes"] == source_size else "stale"), meta


def list_request_files(indexing_files: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    """List selectable request files without scanning their contents."""
    requests_dir = _requests_dir()
    if not requests_dir.is_dir():
        return []
    indexing = set(indexing_files or [])
    files = []
    for entry in requests_dir.iterdir():
        if not _is_valid_request_filename(entry.name):
            continue
        try:
            resolved = resolve_request_file(entry.name)
            stat = resolved.stat()
        except (OSError, RequestStatsValidationError):
            continue
        status, meta = _index_status(resolved, stat.st_size)
        if entry.name in indexing:
            status = "indexing"
        item = {
            "name": entry.name,
            "size_bytes": stat.st_size,
            "modified_ts": int(stat.st_mtime),
            "index_status": status,
        }
        if meta is not None:
            item["indexed_size_bytes"] = meta["size_bytes"]
            item["index_generated_at"] = meta["generated_at"]
            item["indexed_requests"] = meta["row_count"]
        files.append(item)
    files.sort(key=lambda item: item["name"], reverse=True)
    return files


def _new_metric(metric_name: str) -> Dict[str, Any]:
    return {
        "count": 0,
        "missing": 0,
        "invalid": 0,
        "sum": 0,
        "min": None,
        "max": None,
        "bucket_counts": [0] * (len(METRIC_DEFINITIONS[metric_name]["upper_bounds"]) + 1),
    }


def _new_group() -> Dict[str, Any]:
    return {
        "request_count": 0,
        "metrics": {name: _new_metric(name) for name in METRIC_DEFINITIONS},
    }


def _record_group(group: Dict[str, Any], row: Dict[str, Any]) -> None:
    group["request_count"] += 1
    for metric_name in METRIC_DEFINITIONS:
        metric = group["metrics"][metric_name]
        value = row.get(metric_name)
        bucket = row.get(METRIC_BUCKET_FIELDS[metric_name])
        if value is None or bucket is None:
            metric["missing"] += 1
            continue
        metric["count"] += 1
        metric["sum"] += value
        metric["min"] = value if metric["min"] is None else min(metric["min"], value)
        metric["max"] = value if metric["max"] is None else max(metric["max"], value)
        metric["bucket_counts"][bucket] += 1


def _metric_buckets(metric_name: str, bucket_counts: List[int]) -> List[Dict[str, Any]]:
    bounds = METRIC_DEFINITIONS[metric_name]["upper_bounds"]
    return [
        {
            "index": index,
            "lower": 0 if index == 0 else bounds[index - 1] + 1,
            "upper": bounds[index] if index < len(bounds) else None,
            "count": count,
        }
        for index, count in enumerate(bucket_counts)
    ]


def _finalize_metric(metric_name: str, metric: Dict[str, Any]) -> Dict[str, Any]:
    count = metric["count"]
    return {
        "unit": METRIC_DEFINITIONS[metric_name]["unit"],
        "count": count,
        "missing": metric["missing"],
        "invalid": metric["invalid"],
        "sum": metric["sum"],
        "avg": metric["sum"] / count if count else None,
        "min": metric["min"],
        "max": metric["max"],
        "buckets": _metric_buckets(metric_name, metric["bucket_counts"]),
    }


def _finalize_group(group: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "request_count": group["request_count"],
        "metrics": {
            metric_name: _finalize_metric(metric_name, metric)
            for metric_name, metric in group["metrics"].items()
        },
    }


class RequestStatsDataset:
    """One selected set of lightweight request-index rows."""

    def __init__(self, dataset_id: str, files: List[str], rows: List[Dict[str, Any]], estimated_bytes: int):
        self.dataset_id = dataset_id
        self.files = tuple(files)
        self.rows = rows
        self.estimated_bytes = estimated_bytes
        self.created_at = time.time()
        self.last_access = self.created_at
        self.by_model: Dict[str, List[int]] = {}
        self.model_key_by_position: List[str] = []
        self.by_status_code: Dict[str, List[int]] = {}
        self.by_bucket: Dict[Tuple[str, int], List[int]] = {}
        for position, row in enumerate(rows):
            model = row.get("effective_model") or UNKNOWN_MODEL_KEY
            if model not in self.by_model and len(self.by_model) - int(OTHER_MODEL_KEY in self.by_model) >= MAX_MODELS_PER_DATASET:
                model = OTHER_MODEL_KEY
            self.model_key_by_position.append(model)
            self.by_model.setdefault(model, []).append(position)
            status_key = str(row["status_code"]) if row.get("status_code") is not None else UNKNOWN_STATUS_KEY
            self.by_status_code.setdefault(status_key, []).append(position)
            for metric_name, bucket_field in METRIC_BUCKET_FIELDS.items():
                bucket = row.get(bucket_field)
                if bucket is not None:
                    self.by_bucket.setdefault((metric_name, bucket), []).append(position)

    def touch(self) -> None:
        self.last_access = time.time()

    def _positions_for_view(self, view: str, value: Optional[str]) -> Optional[List[int]]:
        if view == "overall":
            return None
        if view == "model":
            return self.by_model.get(value or "", [])
        if view == "status_code":
            return self.by_status_code.get(value or "", [])
        raise RequestStatsValidationError("Invalid view. Use: overall, model, status_code")

    def query_bucket(
        self,
        metric: str,
        bucket: int,
        view: str,
        value: Optional[str],
        page: int,
        per_page: int,
    ) -> Dict[str, Any]:
        if metric not in METRIC_DEFINITIONS:
            raise RequestStatsValidationError("Invalid metric")
        max_bucket = len(METRIC_DEFINITIONS[metric]["upper_bounds"])
        if bucket < 0 or bucket > max_bucket:
            raise RequestStatsValidationError("Invalid bucket")
        view_positions = self._positions_for_view(view, value)
        bucket_positions = self.by_bucket.get((metric, bucket), [])
        if view_positions is None:
            positions = bucket_positions
        else:
            positions = []
            left = right = 0
            while left < len(bucket_positions) and right < len(view_positions):
                bucket_position = bucket_positions[left]
                view_position = view_positions[right]
                if bucket_position == view_position:
                    positions.append(bucket_position); left += 1; right += 1
                elif bucket_position < view_position:
                    left += 1
                else:
                    right += 1
        total = len(positions)
        start = (page - 1) * per_page
        items = [self._summary(self.rows[pos]) for pos in positions[start:start + per_page]]
        self.touch()
        return {
            "items": items,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page if total else 0,
            "filter": {"metric": metric, "bucket": bucket, "view": view, "value": value},
        }

    @staticmethod
    def _summary(row: Dict[str, Any]) -> Dict[str, Any]:
        params = urlencode({
            "file": row["source_file"],
            "offset": row["offset"],
            "length": row["length"],
            "sha256": row["line_sha256"],
        })
        summary = {
            key: row.get(key)
            for key in (
                "id", "timestamp", "model", "translated_model", "effective_model", "endpoint",
                "status_code", "duration_ms", "request_size", "response_size", INPUT_NOT_CACHED,
                CACHE_WRITE, CACHE_READ, OUTPUT, TOTAL_BILLED, "user_id", "client_ip", "source_file",
                "offset", "length", "line_sha256",
            )
        }
        summary["detail_url"] = f"/request-file-detail?{params}"
        return summary

    def build_result(self, file_results: List[Dict[str, Any]], quality: Dict[str, int]) -> Dict[str, Any]:
        overall = _new_group()
        by_model: Dict[str, Dict[str, Any]] = {}
        by_status: Dict[str, Dict[str, Any]] = {}
        model_overflow = OTHER_MODEL_KEY in self.by_model
        for position, row in enumerate(self.rows):
            _record_group(overall, row)
            model = self.model_key_by_position[position]
            _record_group(by_model.setdefault(model, _new_group()), row)
            status = str(row["status_code"]) if row.get("status_code") is not None else UNKNOWN_STATUS_KEY
            _record_group(by_status.setdefault(status, _new_group()), row)

        ordered_models = sorted(by_model, key=lambda model: (model == OTHER_MODEL_KEY, -by_model[model]["request_count"], model.lower()))
        ordered_status = sorted(by_status, key=lambda code: (code == UNKNOWN_STATUS_KEY, int(code) if code.isdigit() else 999))
        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "generated_at": _utc_now(),
            "dataset_id": self.dataset_id,
            "files": file_results,
            "quality": quality,
            "model_overflow": model_overflow,
            "contains_error_requests": any(code.isdigit() and int(code) >= 400 for code in ordered_status),
            "models_ordered": ordered_models,
            "status_codes_ordered": ordered_status,
            "overall": _finalize_group(overall),
            "by_model": {model: _finalize_group(by_model[model]) for model in ordered_models},
            "by_status_code": {code: _finalize_group(by_status[code]) for code in ordered_status},
        }


class RequestStatsDatasetCache:
    """Small TTL/LRU cache for selected in-memory datasets."""

    def __init__(self, max_datasets: int = MAX_DATASETS, ttl_seconds: int = DATASET_TTL_SECONDS):
        self.max_datasets = max_datasets
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._datasets: OrderedDict[str, RequestStatsDataset] = OrderedDict()

    def _cleanup_locked(self) -> None:
        now = time.time()
        expired = [key for key, dataset in self._datasets.items() if now - dataset.last_access > self.ttl_seconds]
        for key in expired:
            self._datasets.pop(key, None)
        while len(self._datasets) > self.max_datasets:
            self._datasets.popitem(last=False)

    def put(self, dataset: RequestStatsDataset) -> None:
        with self._lock:
            self._cleanup_locked()
            self._datasets[dataset.dataset_id] = dataset
            self._datasets.move_to_end(dataset.dataset_id)
            self._cleanup_locked()

    def get(self, dataset_id: str) -> RequestStatsDataset:
        with self._lock:
            self._cleanup_locked()
            dataset = self._datasets.get(dataset_id)
            if dataset is None:
                raise RequestStatsDatasetNotFound(dataset_id)
            dataset.touch()
            self._datasets.move_to_end(dataset_id)
            return dataset


def _load_dataset(files: List[str]) -> RequestStatsDataset:
    rows: List[Dict[str, Any]] = []
    estimated_bytes = 0
    for filename in files:
        request_path = resolve_request_file(filename)
        _load_meta(request_path)
        index_path = _index_path_for(request_path)
        with index_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or row.get("schema_version") != INDEX_SCHEMA_VERSION:
                    raise RequestIndexValidationError(f"Invalid sidecar row in {index_path.name}")
                rows.append(row)
                estimated_bytes += len(line) + 256
                if len(rows) > MAX_DATASET_ROWS or estimated_bytes > MAX_DATASET_ESTIMATED_BYTES:
                    raise RequestStatsValidationError(
                        "Selected indexes exceed the in-memory dataset limit; select fewer request files"
                    )
    rows.sort(key=lambda row: (row.get("timestamp") or 0, row["source_file"], row["offset"]), reverse=True)
    return RequestStatsDataset(uuid.uuid4().hex, files, rows, estimated_bytes)


def load_indexed_rows(files: List[str]) -> List[Dict[str, Any]]:
    """Compatibility helper for tests and offline callers."""
    return _load_dataset(files).rows


def _merge_quality(target: Dict[str, int], source: Dict[str, int]) -> None:
    for field in QUALITY_FIELDS:
        target[field] += source[field]


class RequestStatsJobManager:
    """Run one scan job and publish completed in-memory datasets."""

    TERMINAL_STATUSES = {"completed", "cancelled", "failed"}

    def __init__(
        self,
        *,
        ttl_seconds: int = 900,
        max_completed_jobs: int = 5,
        dataset_cache: Optional[RequestStatsDatasetCache] = None,
    ):
        self.ttl_seconds = ttl_seconds
        self.max_completed_jobs = max_completed_jobs
        self.datasets = dataset_cache or RequestStatsDatasetCache()
        self._lock = threading.RLock()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._active_job_id: Optional[str] = None

    def _cleanup_locked(self) -> None:
        now = time.time()
        for job_id in [
            job_id for job_id, job in self._jobs.items()
            if job["status"] in self.TERMINAL_STATUSES and now - (job.get("finished_at_ts") or now) > self.ttl_seconds
        ]:
            self._jobs.pop(job_id, None)
        completed = sorted(
            ((job.get("finished_at_ts") or 0, job_id) for job_id, job in self._jobs.items() if job["status"] in self.TERMINAL_STATUSES),
            reverse=True,
        )
        for _, job_id in completed[self.max_completed_jobs:]:
            self._jobs.pop(job_id, None)

    @staticmethod
    def _public_job(job: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in job.items() if not key.startswith("_") and not key.endswith("_ts")}

    def active_files(self) -> List[str]:
        with self._lock:
            if not self._active_job_id:
                return []
            job = self._jobs.get(self._active_job_id)
            return [job["current_file"]] if job and job.get("current_file") else []

    def start(self, filenames: Any) -> Tuple[Dict[str, Any], bool]:
        selected = normalize_selected_files(filenames)
        selection_key = tuple(sorted(selected))
        with self._lock:
            self._cleanup_locked()
            if self._active_job_id:
                active = self._jobs.get(self._active_job_id)
                if active and active["status"] in {"queued", "running"}:
                    if active["_selection_key"] == selection_key:
                        return self._public_job(active), True
                    raise RequestStatsBusyError(active["id"])
            job_id = uuid.uuid4().hex
            job = {
                "id": job_id,
                "status": "queued",
                "files": selected,
                "created_at": _utc_now(),
                "started_at": None,
                "finished_at": None,
                "current_file": None,
                "current_mode": None,
                "files_completed": 0,
                "files_total": len(selected),
                "bytes_processed": 0,
                "bytes_total": 0,
                "current_file_bytes_processed": 0,
                "current_file_bytes_total": 0,
                "cache_hits": 0,
                "incremental_scans": 0,
                "rebuild_scans": 0,
                "cancel_requested": False,
                "warnings": [],
                "skipped_files": [],
                "error": None,
                "result": None,
                "_selection_key": selection_key,
                "_cancel_event": threading.Event(),
                "created_at_ts": time.time(),
                "finished_at_ts": None,
            }
            self._jobs[job_id] = job
            self._active_job_id = job_id
            threading.Thread(target=self._run_job, args=(job_id,), name=f"request-stats-{job_id[:8]}", daemon=True).start()
            return self._public_job(job), False

    def get(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            self._cleanup_locked()
            job = self._jobs.get(job_id)
            if job is None:
                raise RequestStatsJobNotFound(job_id)
            return self._public_job(job)

    def cancel(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise RequestStatsJobNotFound(job_id)
            if job["status"] in {"queued", "running"}:
                job["cancel_requested"] = True
                job["_cancel_event"].set()
            return self._public_job(job)

    def query_dataset(self, dataset_id: str, **kwargs: Any) -> Dict[str, Any]:
        return self.datasets.get(dataset_id).query_bucket(**kwargs)

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = "running"
            job["started_at"] = _utc_now()
            filenames = list(job["files"])
            cancel_event = job["_cancel_event"]
        try:
            planned = []
            total_bytes = 0
            for filename in filenames:
                try:
                    path = resolve_request_file(filename)
                    size = path.stat().st_size
                except (OSError, RequestStatsValidationError) as exc:
                    planned.append((filename, None, 0))
                    self._append_skipped(job_id, filename, "not_found", str(exc))
                else:
                    planned.append((filename, path, size))
                    total_bytes += size
            self._set_job(job_id, bytes_total=total_bytes)

            completed_bytes = 0
            file_results = []
            quality = _new_quality()
            indexed_files = []
            for filename, path, planned_size in planned:
                _check_cancelled(cancel_event)
                if path is None:
                    self._increment(job_id, "files_completed")
                    continue
                current_expected = [planned_size]

                def on_progress(mode: str, processed: int, total: int) -> None:
                    if total != current_expected[0]:
                        with self._lock:
                            self._jobs[job_id]["bytes_total"] += total - current_expected[0]
                        current_expected[0] = total
                    self._set_job(
                        job_id,
                        current_mode=mode,
                        current_file_bytes_processed=processed,
                        current_file_bytes_total=total,
                        bytes_processed=completed_bytes + processed,
                    )

                self._set_job(
                    job_id,
                    current_file=filename,
                    current_mode="checking",
                    current_file_bytes_processed=0,
                    current_file_bytes_total=planned_size,
                    bytes_processed=completed_bytes,
                )
                try:
                    meta, mode, warnings = build_or_load_file_index(path, cancel_event=cancel_event, progress=on_progress)
                except RequestStatsCancelled:
                    raise
                except (OSError, RequestFileChangedError, RequestIndexValidationError) as exc:
                    self._append_skipped(job_id, filename, "read_error", str(exc))
                    completed_bytes += current_expected[0]
                    self._increment(job_id, "files_completed")
                    continue
                indexed_files.append(filename)
                _merge_quality(quality, meta["quality"])
                file_results.append({
                    "name": filename,
                    "size_bytes": meta["size_bytes"],
                    "processed_bytes": meta["processed_bytes"],
                    "generated_at": meta["generated_at"],
                    "mode": mode,
                    "quality": dict(meta["quality"]),
                })
                for warning in warnings:
                    self._append_warning(job_id, warning)
                completed_bytes += meta["size_bytes"]
                with self._lock:
                    active = self._jobs[job_id]
                    active["files_completed"] += 1
                    active["bytes_processed"] = completed_bytes
                    active["current_file_bytes_processed"] = meta["size_bytes"]
                    active[{"cached": "cache_hits", "incremental": "incremental_scans"}.get(mode, "rebuild_scans")] += 1

            _check_cancelled(cancel_event)
            dataset = _load_dataset(indexed_files)
            result = dataset.build_result(file_results, quality)
            with self._lock:
                active = self._jobs[job_id]
                result["skipped_files"] = list(active["skipped_files"])
                result["warnings"] = list(active["warnings"])
            self.datasets.put(dataset)
            self._set_job(job_id, result=result, status="completed", current_file=None, current_mode=None)
        except RequestStatsCancelled:
            self._set_job(job_id, status="cancelled", current_file=None, current_mode=None)
        except Exception as exc:
            self._set_job(job_id, status="failed", current_file=None, current_mode=None, error=str(exc))
        finally:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    job["finished_at"] = _utc_now()
                    job["finished_at_ts"] = time.time()
                if self._active_job_id == job_id:
                    self._active_job_id = None

    def _set_job(self, job_id: str, **updates: Any) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(updates)

    def _increment(self, job_id: str, field: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id][field] += 1

    def _append_warning(self, job_id: str, warning: Dict[str, Any]) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["warnings"].append(dict(warning))

    def _append_skipped(self, job_id: str, filename: str, reason: str, message: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["skipped_files"].append({"file": filename, "reason": reason, "message": message})


def _sidecar_contains_locator(filename: str, offset: int, length: int, sha256: str) -> bool:
    request_path = resolve_request_file(filename)
    try:
        _load_meta(request_path)
    except FileNotFoundError as exc:
        raise RequestIndexValidationError("Request file has not been indexed") from exc
    with _index_path_for(request_path).open("r", encoding="utf-8") as file:
        for line in file:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise RequestIndexValidationError("Request sidecar contains invalid JSON")
            if row.get("offset") == offset:
                return row.get("length") == length and row.get("line_sha256") == sha256
            if isinstance(row.get("offset"), int) and row["offset"] > offset:
                return False
    return False


def read_request_detail(filename: str, offset: int, length: int, sha256: str) -> Dict[str, Any]:
    """Read one stable, indexed JSONL line after validating its locator and hash."""
    if offset < 0 or length <= 0:
        raise RequestStatsValidationError("offset and length must be positive integers")
    if length > MAX_DETAIL_LINE_BYTES:
        raise RequestStatsValidationError(f"Request detail exceeds the {MAX_DETAIL_LINE_BYTES}-byte display limit")
    if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise RequestStatsValidationError("Invalid request line sha256")
    if not _sidecar_contains_locator(filename, offset, length, sha256):
        raise RequestStatsValidationError("Request locator is not present in the sidecar index")
    request_path = resolve_request_file(filename)
    with request_path.open("rb") as file:
        file.seek(offset)
        raw = file.read(length)
    if len(raw) != length or hashlib.sha256(raw).hexdigest() != sha256:
        raise RequestFileChangedError("The request file changed; rebuild its statistics index")
    try:
        entry = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RequestFileChangedError("The indexed request line is no longer valid JSON") from exc
    if not isinstance(entry, dict):
        raise RequestFileChangedError("The indexed request line is no longer a JSON object")
    return entry


request_stats_jobs = RequestStatsJobManager()
