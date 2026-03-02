"""
Periodic token usage reporter to OneDrive agent folder.
"""

from __future__ import annotations

import atexit
import json
import signal
import threading
import time
from pathlib import Path
from typing import Dict

from .cache import cache
from .config_sync import get_agent_root


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

            delta_input = max(0, int(now.get("input_tokens", 0)) - int(prev.get("input_tokens", 0)))
            delta_output = max(0, int(now.get("output_tokens", 0)) - int(prev.get("output_tokens", 0)))
            delta_total = delta_input + delta_output
            if delta_total == 0:
                continue

            total_delta_tokens += delta_total
            delta_models.append({
                "model": model_name,
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
