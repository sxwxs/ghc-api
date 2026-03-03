"""
Periodic token usage reporter to OneDrive agent folder.
"""

from __future__ import annotations

import atexit
import json
import os
import platform
import signal
import socket
import threading
import time
from pathlib import Path
from typing import Dict

from .cache import cache
from .config_sync import get_agent_root, get_onedrive_path


class TokenUsageReporter:
    def __init__(self, interval_seconds: int = 300):
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_snapshot: Dict[str, Dict[str, int]] = cache.get_model_token_snapshot()
        self._write_lock = threading.Lock()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="token-usage-reporter", daemon=True)
        self._thread.start()
        print("[Token Usage] Reporter started (every 300s).")

    def _usage_file(self) -> Path | None:
        agent_root = get_agent_root()
        if agent_root:
            return agent_root / "token_usage.jl"
        return Path.home() / ".ghc-api" / "token_usage.jl"

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._write_usage_delta()

    def _write_usage_delta(self) -> None:
        with self._write_lock:
            self._write_usage_delta_locked()

    def _write_usage_delta_locked(self) -> None:
        usage_file = self._usage_file()
        if not usage_file:
            return

        current = cache.get_model_token_snapshot()
        delta_models = []
        total_delta_tokens = 0

        model_names = sorted(set(self._last_snapshot.keys()) | set(current.keys()))
        for model_name in model_names:
            prev = self._last_snapshot.get(model_name, {})
            now = current.get(model_name, {})

            delta_request_count = max(0, int(now.get("request_count", 0)) - int(prev.get("request_count", 0)))
            delta_input = max(0, int(now.get("input_tokens", 0)) - int(prev.get("input_tokens", 0)))
            delta_output = max(0, int(now.get("output_tokens", 0)) - int(prev.get("output_tokens", 0)))
            delta_total = delta_input + delta_output
            if delta_total == 0:
                continue

            total_delta_tokens += delta_total
            delta_models.append({
                "model": model_name,
                "request_count": delta_request_count,
                "input_tokens": delta_input,
                "output_tokens": delta_output,
                "total_tokens": delta_total,
            })

        self._last_snapshot = current
        if total_delta_tokens == 0:
            return

        payload = {
            "timestamp": int(time.time()),
            "models": delta_models,
        }

        try:
            usage_file.parent.mkdir(parents=True, exist_ok=True)
            with usage_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[Token Usage] Failed to append usage file {usage_file}: {e}")

    def flush(self) -> None:
        """Flush any unreported token usage delta immediately."""
        self._write_usage_delta()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self.flush()


_reporter: TokenUsageReporter | None = None
_shutdown_lock = threading.Lock()
_shutdown_done = False
_handlers_registered = False
_previous_sig_handlers: dict[int, signal.Handlers] = {}


def _perform_shutdown_flush() -> None:
    global _shutdown_done
    with _shutdown_lock:
        if _shutdown_done:
            return
        _shutdown_done = True
    if _reporter is not None:
        _reporter.stop()
        print("[Token Usage] Reporter flushed on shutdown.")


def _signal_handler(signum: int, frame) -> None:  # noqa: ANN001
    _perform_shutdown_flush()
    previous = _previous_sig_handlers.get(signum)
    if previous in (None, signal.SIG_DFL):
        raise KeyboardInterrupt
    if previous == signal.SIG_IGN:
        return
    if callable(previous):
        previous(signum, frame)


def _register_shutdown_handlers() -> None:
    global _handlers_registered
    if _handlers_registered:
        return
    _handlers_registered = True
    atexit.register(_perform_shutdown_flush)

    for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        _previous_sig_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, _signal_handler)


def start_token_usage_reporter() -> None:
    global _reporter
    if _reporter is None:
        _reporter = TokenUsageReporter(interval_seconds=300)
    _register_shutdown_handlers()
    _reporter.start()


def _local_machine_name() -> str:
    os_label = "WSL" if (os.environ.get("WSL_DISTRO_NAME") or "microsoft" in platform.release().lower()) else (
        "Win" if platform.system() == "Windows" else "Linux"
    )
    return f"{socket.gethostname()}_{os_label}"


def _resolve_usage_files() -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    onedrive = get_onedrive_path()
    if onedrive:
        agents_root = onedrive / ".ghc-api" / "agents"
        if agents_root.exists() and agents_root.is_dir():
            for machine_dir in sorted([d for d in agents_root.iterdir() if d.is_dir()], key=lambda p: p.name.lower()):
                usage_file = machine_dir / "token_usage.jl"
                if usage_file.exists() and usage_file.is_file():
                    files.append((machine_dir.name, usage_file))

    if files:
        return files

    local_file = Path.home() / ".ghc-api" / "token_usage.jl"
    if local_file.exists() and local_file.is_file():
        return [(_local_machine_name(), local_file)]
    return []


def _range_cutoff_ts(range_key: str) -> int | None:
    now_ts = int(time.time())
    mapping = {
        "hour": 60 * 60,
        "day": 24 * 60 * 60,
        "week": 7 * 24 * 60 * 60,
        "month": 30 * 24 * 60 * 60,
    }
    delta = mapping.get((range_key or "").lower())
    return now_ts - delta if delta else None


def get_token_usage_overview(range_key: str = "all") -> Dict[str, object]:
    usage_files = _resolve_usage_files()
    cutoff = _range_cutoff_ts(range_key)
    machines = sorted([machine for machine, _ in usage_files], key=str.lower)

    aggregate: Dict[tuple[str, str], Dict[str, int]] = {}

    for machine, path in usage_files:
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue

                    ts = int(payload.get("timestamp", 0))
                    if cutoff is not None and ts < cutoff:
                        continue

                    models = payload.get("models")
                    if not isinstance(models, list):
                        continue

                    for model_usage in models:
                        if not isinstance(model_usage, dict):
                            continue
                        model_id = str(model_usage.get("model") or model_usage.get("model_id") or "unknown")
                        key = (machine, model_id)
                        if key not in aggregate:
                            aggregate[key] = {
                                "request_count": 0,
                                "input_tokens": 0,
                                "output_tokens": 0,
                                "total_tokens": 0,
                            }
                        req_count = int(model_usage.get("request_count", 0) or 0)
                        input_tokens = int(model_usage.get("input_tokens", 0) or 0)
                        output_tokens = int(model_usage.get("output_tokens", 0) or 0)
                        total_tokens = int(model_usage.get("total_tokens", input_tokens + output_tokens) or 0)

                        aggregate[key]["request_count"] += req_count
                        aggregate[key]["input_tokens"] += input_tokens
                        aggregate[key]["output_tokens"] += output_tokens
                        aggregate[key]["total_tokens"] += total_tokens
        except Exception:
            continue

    rows = []
    totals = {
        "request_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }

    for (machine, model_id), stats in sorted(aggregate.items(), key=lambda item: (item[0][0].lower(), item[0][1].lower())):
        row = {
            "machine": machine,
            "model_id": model_id,
            "request_count": stats["request_count"],
            "input_tokens": stats["input_tokens"],
            "output_tokens": stats["output_tokens"],
            "total_tokens": stats["total_tokens"],
        }
        rows.append(row)
        totals["request_count"] += row["request_count"]
        totals["input_tokens"] += row["input_tokens"]
        totals["output_tokens"] += row["output_tokens"]
        totals["total_tokens"] += row["total_tokens"]

    return {
        "range": range_key,
        "machines": machines,
        "rows": rows,
        "totals": totals,
    }
