"""Persistent response-header affinity for configured upstream proxies."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from ..utils import get_config_dir
from .config import ProxyApiConfig, ProxyModelApiConfig, ProxyModelConfig, ProxyProfileConfig


AFFINITY_FILE_VERSION = 1


def get_affinity_path() -> Path:
    override = os.environ.get("GHC_API_PROXY_AFFINITY_FILE")
    if override:
        return Path(os.path.abspath(os.path.expanduser(os.path.expandvars(override))))
    return Path(get_config_dir()) / "proxy-affinity.json"


def affinity_key(
    profile: ProxyProfileConfig,
    api: ProxyApiConfig,
    model: ProxyModelConfig,
    model_api: ProxyModelApiConfig,
) -> str:
    model_scope = model.id if profile.affinity.scope == "model" else "*"
    routing_config = {
        "profile": profile.name,
        "api": api.name,
        "model": model_scope,
        "upstream_url": api.upstream_url,
        "request_model": api.request_model,
        "upstream_model": model_api.upstream_model,
        "response_header": profile.affinity.response_header,
        "request_header": profile.affinity.request_header,
        "headers": {
            "profile": profile.headers,
            "api": api.headers,
            "model": model.headers,
            "model_api": model_api.headers,
        },
    }
    raw = json.dumps(routing_config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ProxyAffinityStore:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or get_affinity_path()
        self._lock = threading.RLock()
        self._tokens: Dict[str, str] = {}
        self._persistent_keys = set()
        self._discovery_locks: Dict[str, threading.Lock] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            try:
                with self.path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            except FileNotFoundError:
                return
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[Configured Proxy] Ignoring unreadable affinity state at {self.path}: {exc}")
                return

            if not isinstance(payload, dict) or payload.get("version") != AFFINITY_FILE_VERSION:
                return
            entries = payload.get("entries")
            if not isinstance(entries, dict):
                return
            for key, entry in entries.items():
                if not isinstance(key, str) or not isinstance(entry, dict):
                    continue
                token = entry.get("token")
                if isinstance(token, str) and token:
                    self._tokens[key] = token
                    self._persistent_keys.add(key)

    def get(self, key: str) -> Optional[str]:
        self._ensure_loaded()
        with self._lock:
            return self._tokens.get(key)

    def set(self, key: str, token: str, persist: bool) -> None:
        if not token:
            return
        self._ensure_loaded()
        with self._lock:
            previous_token = self._tokens.get(key)
            was_persistent = key in self._persistent_keys
            self._tokens[key] = token
            if persist:
                self._persistent_keys.add(key)
            else:
                self._persistent_keys.discard(key)
            if previous_token == token and was_persistent == persist:
                return
            if persist or was_persistent:
                try:
                    self._write_locked()
                except OSError as exc:
                    print(f"[Configured Proxy] Failed to persist affinity state: {exc}")

    def clear(self, key: str, persist: bool) -> None:
        self._ensure_loaded()
        with self._lock:
            if key not in self._tokens:
                return
            was_persistent = key in self._persistent_keys
            self._tokens.pop(key, None)
            self._persistent_keys.discard(key)
            if persist or was_persistent:
                try:
                    self._write_locked()
                except OSError as exc:
                    print(f"[Configured Proxy] Failed to persist affinity state: {exc}")

    def discovery_lock(self, key: str) -> threading.Lock:
        with self._lock:
            lock = self._discovery_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._discovery_locks[key] = lock
            return lock

    def _write_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": AFFINITY_FILE_VERSION,
            "entries": {
                key: {"token": self._tokens[key], "updated_at": int(time.time())}
                for key in self._persistent_keys
                if key in self._tokens
            },
        }
        fd, temp_name = tempfile.mkstemp(
            prefix=".proxy-affinity.", suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_name, self.path)
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
