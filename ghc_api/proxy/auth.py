"""Authentication providers for configured upstream proxies."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Optional

from .config import ProxyAuthConfig


class ProxyAuthError(RuntimeError):
    """Raised when an upstream credential cannot be resolved."""


class ProxyAuthProvider:
    def __init__(self, config: ProxyAuthConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._token: Optional[str] = None
        self._expires_at_monotonic: float = 0.0

    def invalidate(self) -> None:
        with self._lock:
            self._token = None
            self._expires_at_monotonic = 0.0

    def get_token(self) -> Optional[str]:
        if self.config.type == "none":
            return None
        if self.config.type == "bearer_env":
            value = os.environ.get(self.config.env or "", "").strip()
            if not value:
                raise ProxyAuthError("The configured upstream credential environment variable is empty or missing")
            return value

        now = time.monotonic()
        if self._token and now < self._expires_at_monotonic:
            return self._token

        with self._lock:
            now = time.monotonic()
            if self._token and now < self._expires_at_monotonic:
                return self._token

            try:
                completed = subprocess.run(
                    list(self.config.command),
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.config.command_timeout_seconds,
                    shell=False,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise ProxyAuthError(f"The configured upstream credential command could not run: {exc}") from exc

            if completed.returncode != 0:
                raise ProxyAuthError(
                    f"The configured upstream credential command exited with code {completed.returncode}"
                )

            token = (completed.stdout or "").strip()
            if not token:
                raise ProxyAuthError("The configured upstream credential command returned an empty token")

            self._token = token
            self._expires_at_monotonic = time.monotonic() + self.config.cache_ttl_seconds
            return token
