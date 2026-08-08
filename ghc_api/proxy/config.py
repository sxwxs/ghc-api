"""Configuration loading for isolated, config-driven upstream proxies."""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

from ..utils import get_config_dir


SUPPORTED_APIS = frozenset({"responses", "chat_completions"})
MODEL_REQUEST_MODES = frozenset({"preserve", "omit", "upstream"})
MODEL_RESPONSE_MODES = frozenset({"preserve", "public"})
AUTH_TYPES = frozenset({"none", "bearer_env", "bearer_command"})
AFFINITY_SCOPES = frozenset({"proxy", "model"})
PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
ENV_VAR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ProxyConfigError(ValueError):
    """Raised when the private proxy configuration is invalid."""


@dataclass(frozen=True)
class ProxyAuthConfig:
    type: str = "none"
    env: Optional[str] = None
    command: Tuple[str, ...] = ()
    cache_ttl_seconds: int = 300
    command_timeout_seconds: int = 30

    def fingerprint(self) -> Tuple:
        return (
            self.type,
            self.env,
            self.command,
            self.cache_ttl_seconds,
            self.command_timeout_seconds,
        )


@dataclass(frozen=True)
class ProxyAffinityConfig:
    enabled: bool = False
    response_header: str = ""
    request_header: str = ""
    scope: str = "model"
    persist: bool = True


@dataclass(frozen=True)
class ProxyApiConfig:
    name: str
    upstream_url: str
    timeout_seconds: int = 180
    max_connection_retries: int = 0
    request_model: str = "preserve"
    response_model: str = "preserve"
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ProxyModelApiConfig:
    enabled: bool = True
    upstream_model: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ProxyModelConfig:
    id: str
    display_name: str
    headers: Dict[str, str] = field(default_factory=dict)
    apis: Dict[str, ProxyModelApiConfig] = field(default_factory=dict)
    reasoning: bool = False
    input_types: Tuple[str, ...] = ("text",)
    context_window: int = 128000
    max_output_tokens: int = 16384

    def api_config(self, api_name: str) -> Optional[ProxyModelApiConfig]:
        value = self.apis.get(api_name)
        if value is None or not value.enabled:
            return None
        return value


@dataclass(frozen=True)
class ProxyProfileConfig:
    name: str
    auth: ProxyAuthConfig
    headers: Dict[str, str]
    affinity: ProxyAffinityConfig
    apis: Dict[str, ProxyApiConfig]
    models: Dict[str, ProxyModelConfig]

    def resolve(self, api_name: str, model_id: str) -> Optional[Tuple[ProxyApiConfig, ProxyModelConfig, ProxyModelApiConfig]]:
        api_config = self.apis.get(api_name)
        model_config = self.models.get(model_id)
        if api_config is None or model_config is None:
            return None
        model_api = model_config.api_config(api_name)
        if model_api is None:
            return None
        return api_config, model_config, model_api


@dataclass(frozen=True)
class ProxyConfigSnapshot:
    profiles: Dict[str, ProxyProfileConfig] = field(default_factory=dict)


def get_proxy_config_path() -> Path:
    override = os.environ.get("GHC_API_PROXY_CONFIG")
    if override:
        return Path(os.path.abspath(os.path.expanduser(os.path.expandvars(override))))
    return Path(get_config_dir()) / "upstream-proxies.yaml"


def _require_mapping(value, field_name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ProxyConfigError(f"'{field_name}' must be an object")
    return value


def _parse_bool(value, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ProxyConfigError(f"'{field_name}' must be a boolean")
    return value


def _parse_headers(value, field_name: str) -> Dict[str, str]:
    mapping = _require_mapping(value, field_name)
    headers: Dict[str, str] = {}
    for name, header_value in mapping.items():
        if not isinstance(name, str) or not name.strip():
            raise ProxyConfigError(f"'{field_name}' header names must be non-empty strings")
        normalized_name = name.strip()
        if normalized_name in headers:
            raise ProxyConfigError(f"'{field_name}' contains duplicate header '{normalized_name}'")
        if not isinstance(header_value, str):
            raise ProxyConfigError(f"'{field_name}.{normalized_name}' must be a string")
        headers[normalized_name] = header_value
    return headers


def _parse_positive_int(value, default: int, field_name: str, allow_zero: bool = False) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProxyConfigError(f"'{field_name}' must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        comparator = ">= 0" if allow_zero else "> 0"
        raise ProxyConfigError(f"'{field_name}' must be {comparator}")
    return value


def _parse_auth(value, field_name: str) -> ProxyAuthConfig:
    raw = _require_mapping(value, field_name)
    auth_type = raw.get("type", "none")
    if auth_type not in AUTH_TYPES:
        raise ProxyConfigError(f"'{field_name}.type' must be one of: {', '.join(sorted(AUTH_TYPES))}")

    env_name = raw.get("env")
    command_value = raw.get("command", [])
    if command_value is None:
        command_value = []
    if not isinstance(command_value, list) or any(not isinstance(item, str) or not item for item in command_value):
        raise ProxyConfigError(f"'{field_name}.command' must be a list of non-empty strings")

    if auth_type == "bearer_env":
        if not isinstance(env_name, str) or not ENV_VAR_RE.match(env_name):
            raise ProxyConfigError(f"'{field_name}.env' must be a valid environment variable name")
    elif env_name is not None and not isinstance(env_name, str):
        raise ProxyConfigError(f"'{field_name}.env' must be a string")

    if auth_type == "bearer_command" and not command_value:
        raise ProxyConfigError(f"'{field_name}.command' is required for bearer_command auth")

    return ProxyAuthConfig(
        type=auth_type,
        env=env_name,
        command=tuple(command_value),
        cache_ttl_seconds=_parse_positive_int(
            raw.get("cache_ttl_seconds"), 300, f"{field_name}.cache_ttl_seconds"
        ),
        command_timeout_seconds=_parse_positive_int(
            raw.get("command_timeout_seconds"), 30, f"{field_name}.command_timeout_seconds"
        ),
    )


def _parse_affinity(value, field_name: str) -> ProxyAffinityConfig:
    raw = _require_mapping(value, field_name)
    enabled = _parse_bool(raw.get("enabled", False), f"{field_name}.enabled")
    response_header = raw.get("response_header", "")
    request_header = raw.get("request_header", response_header)
    scope = raw.get("scope", "model")
    persist = _parse_bool(raw.get("persist", True), f"{field_name}.persist")

    if scope not in AFFINITY_SCOPES:
        raise ProxyConfigError(f"'{field_name}.scope' must be one of: {', '.join(sorted(AFFINITY_SCOPES))}")
    if enabled:
        if not isinstance(response_header, str) or not response_header.strip():
            raise ProxyConfigError(f"'{field_name}.response_header' is required when affinity is enabled")
        if not isinstance(request_header, str) or not request_header.strip():
            raise ProxyConfigError(f"'{field_name}.request_header' is required when affinity is enabled")

    return ProxyAffinityConfig(
        enabled=enabled,
        response_header=response_header,
        request_header=request_header,
        scope=scope,
        persist=persist,
    )


def _parse_apis(value, field_name: str) -> Dict[str, ProxyApiConfig]:
    raw_apis = _require_mapping(value, field_name)
    apis: Dict[str, ProxyApiConfig] = {}
    for api_name, raw_value in raw_apis.items():
        if api_name not in SUPPORTED_APIS:
            raise ProxyConfigError(
                f"Unsupported API '{api_name}' in '{field_name}'; supported APIs: {', '.join(sorted(SUPPORTED_APIS))}"
            )
        raw = _require_mapping(raw_value, f"{field_name}.{api_name}")
        if not _parse_bool(raw.get("enabled", True), f"{field_name}.{api_name}.enabled"):
            continue
        upstream_url = raw.get("upstream_url")
        if not isinstance(upstream_url, str) or not upstream_url.startswith(("http://", "https://")):
            raise ProxyConfigError(f"'{field_name}.{api_name}.upstream_url' must be an HTTP(S) URL")
        request_model = raw.get("request_model", "preserve")
        response_model = raw.get("response_model", "preserve")
        if request_model not in MODEL_REQUEST_MODES:
            raise ProxyConfigError(
                f"'{field_name}.{api_name}.request_model' must be one of: {', '.join(sorted(MODEL_REQUEST_MODES))}"
            )
        if response_model not in MODEL_RESPONSE_MODES:
            raise ProxyConfigError(
                f"'{field_name}.{api_name}.response_model' must be one of: {', '.join(sorted(MODEL_RESPONSE_MODES))}"
            )
        apis[api_name] = ProxyApiConfig(
            name=api_name,
            upstream_url=upstream_url,
            timeout_seconds=_parse_positive_int(
                raw.get("timeout_seconds"), 180, f"{field_name}.{api_name}.timeout_seconds"
            ),
            max_connection_retries=_parse_positive_int(
                raw.get("max_connection_retries"),
                0,
                f"{field_name}.{api_name}.max_connection_retries",
                allow_zero=True,
            ),
            request_model=request_model,
            response_model=response_model,
            headers=_parse_headers(raw.get("headers"), f"{field_name}.{api_name}.headers"),
        )
    if not apis:
        raise ProxyConfigError(f"'{field_name}' must enable at least one supported API")
    return apis


def _parse_model_apis(value, enabled_apis: Dict[str, ProxyApiConfig], field_name: str) -> Dict[str, ProxyModelApiConfig]:
    if value is None:
        return {name: ProxyModelApiConfig() for name in enabled_apis}

    raw_apis = _require_mapping(value, field_name)
    result: Dict[str, ProxyModelApiConfig] = {
        name: ProxyModelApiConfig() for name in enabled_apis
    }
    for api_name, raw_value in raw_apis.items():
        if api_name not in SUPPORTED_APIS:
            raise ProxyConfigError(f"Unsupported API '{api_name}' in '{field_name}'")
        if api_name not in enabled_apis:
            continue
        raw = _require_mapping(raw_value, f"{field_name}.{api_name}")
        enabled = _parse_bool(raw.get("enabled", True), f"{field_name}.{api_name}.enabled")
        upstream_model = raw.get("upstream_model")
        if upstream_model is not None and (not isinstance(upstream_model, str) or not upstream_model):
            raise ProxyConfigError(f"'{field_name}.{api_name}.upstream_model' must be null or a non-empty string")
        result[api_name] = ProxyModelApiConfig(
            enabled=enabled,
            upstream_model=upstream_model,
            headers=_parse_headers(raw.get("headers"), f"{field_name}.{api_name}.headers"),
        )
    return result


def _parse_models(value, enabled_apis: Dict[str, ProxyApiConfig], field_name: str) -> Dict[str, ProxyModelConfig]:
    raw_models = _require_mapping(value, field_name)
    if not raw_models:
        raise ProxyConfigError(f"'{field_name}' must define at least one model")

    models: Dict[str, ProxyModelConfig] = {}
    for model_id, raw_value in raw_models.items():
        if not isinstance(model_id, str) or not model_id.strip():
            raise ProxyConfigError(f"'{field_name}' model ids must be non-empty strings")
        raw = _require_mapping(raw_value, f"{field_name}.{model_id}")
        display_name = raw.get("display_name", model_id)
        if not isinstance(display_name, str) or not display_name:
            raise ProxyConfigError(f"'{field_name}.{model_id}.display_name' must be a non-empty string")
        input_types = raw.get("input", ["text"])
        if not isinstance(input_types, list) or not input_types or any(item not in ("text", "image") for item in input_types):
            raise ProxyConfigError(f"'{field_name}.{model_id}.input' must contain 'text' and/or 'image'")

        model_apis = _parse_model_apis(raw.get("apis"), enabled_apis, f"{field_name}.{model_id}.apis")
        for api_name, model_api in model_apis.items():
            if model_api.enabled and enabled_apis[api_name].request_model == "upstream" and not model_api.upstream_model:
                raise ProxyConfigError(
                    f"'{field_name}.{model_id}.apis.{api_name}.upstream_model' is required when request_model is upstream"
                )

        models[model_id] = ProxyModelConfig(
            id=model_id,
            display_name=display_name,
            headers=_parse_headers(raw.get("headers"), f"{field_name}.{model_id}.headers"),
            apis=model_apis,
            reasoning=_parse_bool(raw.get("reasoning", False), f"{field_name}.{model_id}.reasoning"),
            input_types=tuple(input_types),
            context_window=_parse_positive_int(
                raw.get("context_window"), 128000, f"{field_name}.{model_id}.context_window"
            ),
            max_output_tokens=_parse_positive_int(
                raw.get("max_output_tokens"), 16384, f"{field_name}.{model_id}.max_output_tokens"
            ),
        )
    return models


def parse_proxy_config(data) -> ProxyConfigSnapshot:
    root = _require_mapping(data, "root")
    raw_profiles = _require_mapping(root.get("proxies"), "proxies")
    profiles: Dict[str, ProxyProfileConfig] = {}

    for profile_name, raw_value in raw_profiles.items():
        if not isinstance(profile_name, str) or not PROFILE_NAME_RE.match(profile_name):
            raise ProxyConfigError("Proxy profile names may contain only letters, digits, '.', '_' and '-'")
        raw = _require_mapping(raw_value, f"proxies.{profile_name}")
        if not _parse_bool(raw.get("enabled", True), f"proxies.{profile_name}.enabled"):
            continue
        apis = _parse_apis(raw.get("apis"), f"proxies.{profile_name}.apis")
        profiles[profile_name] = ProxyProfileConfig(
            name=profile_name,
            auth=_parse_auth(raw.get("auth"), f"proxies.{profile_name}.auth"),
            headers=_parse_headers(raw.get("headers"), f"proxies.{profile_name}.headers"),
            affinity=_parse_affinity(raw.get("affinity"), f"proxies.{profile_name}.affinity"),
            apis=apis,
            models=_parse_models(raw.get("models"), apis, f"proxies.{profile_name}.models"),
        )

    return ProxyConfigSnapshot(profiles=profiles)


class ProxyRegistry:
    """mtime-reloaded private config with last-known-good semantics."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or get_proxy_config_path()
        self._lock = threading.RLock()
        self._snapshot = ProxyConfigSnapshot()
        self._mtime_ns: Optional[int] = None
        self._last_error: Optional[str] = None
        self._loaded_once = False

    @property
    def last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    def _reload_if_needed(self) -> None:
        with self._lock:
            try:
                stat = self.path.stat()
            except FileNotFoundError:
                self._snapshot = ProxyConfigSnapshot()
                self._mtime_ns = None
                self._last_error = None
                self._loaded_once = True
                return
            except OSError as exc:
                self._last_error = f"Unable to stat private proxy configuration: {exc}"
                return

            if self._loaded_once and self._mtime_ns == stat.st_mtime_ns:
                return

            try:
                with self.path.open("r", encoding="utf-8") as f:
                    parsed = yaml.safe_load(f) or {}
                snapshot = parse_proxy_config(parsed)
            except (OSError, yaml.YAMLError, ProxyConfigError) as exc:
                self._last_error = f"Invalid private proxy configuration: {exc}"
                self._loaded_once = True
                self._mtime_ns = stat.st_mtime_ns
                print(f"[Configured Proxy] {self._last_error}")
                return

            self._snapshot = snapshot
            self._last_error = None
            self._loaded_once = True
            self._mtime_ns = stat.st_mtime_ns
            print(f"[Configured Proxy] Loaded {len(snapshot.profiles)} profile(s) from {self.path}")

    def get_profile(self, profile_name: str) -> Optional[ProxyProfileConfig]:
        self._reload_if_needed()
        with self._lock:
            return self._snapshot.profiles.get(profile_name)

    def snapshot(self) -> ProxyConfigSnapshot:
        self._reload_if_needed()
        with self._lock:
            return self._snapshot
