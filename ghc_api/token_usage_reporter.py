"""
Periodic token usage reporter to OneDrive agent folder.
"""

from __future__ import annotations

import json
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

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="token-usage-reporter", daemon=True)
        self._thread.start()
        print("[Token Usage] Reporter started (every 300s).")

    def _usage_file(self) -> Path | None:
        agent_root = get_agent_root()
        if not agent_root:
            return None
        return agent_root / "token_usage.jl"

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._write_usage_delta()

    def _write_usage_delta(self) -> None:
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


_reporter: TokenUsageReporter | None = None


def start_token_usage_reporter() -> None:
    global _reporter
    if _reporter is None:
        _reporter = TokenUsageReporter(interval_seconds=300)
    _reporter.start()
