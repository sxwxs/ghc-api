"""HTTP client runtime for isolated configured upstream proxies."""

from __future__ import annotations

import copy
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import requests

from .affinity import ProxyAffinityStore, affinity_key
from .auth import ProxyAuthError, ProxyAuthProvider
from .config import (
    ProxyApiConfig,
    ProxyModelApiConfig,
    ProxyModelConfig,
    ProxyProfileConfig,
    ProxyRegistry,
)
from .kimi_k3 import KIMI_K3_PAPYRUS, fold_chat_messages


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ProxyRequestError(RuntimeError):
    """Raised before an upstream HTTP response can be returned."""


@dataclass
class ProxyUpstreamResult:
    response: requests.Response
    payload: dict


def _resolve_header_value(value: str) -> str:
    def replace(match: re.Match) -> str:
        name = match.group(1)
        env_value = os.environ.get(name)
        if env_value is None:
            raise ProxyRequestError(f"Required environment variable '{name}' is not set")
        return env_value

    return _ENV_PATTERN.sub(replace, value)


def _merge_headers(*header_sets: Dict[str, str]) -> Dict[str, str]:
    merged: Dict[str, str] = {"Content-Type": "application/json"}
    for headers in header_sets:
        for name, value in headers.items():
            merged[name] = _resolve_header_value(value)
    return merged


def transform_payload(
    payload: dict,
    api: ProxyApiConfig,
    model: ProxyModelConfig,
    model_api: ProxyModelApiConfig,
) -> dict:
    upstream_payload = copy.deepcopy(payload)
    if api.request_model == "omit":
        upstream_payload.pop("model", None)
    elif api.request_model == "upstream":
        upstream_payload["model"] = model_api.upstream_model
    else:
        upstream_payload["model"] = model.id

    # Compatibility is selected by validated model API configuration, never by
    # either the public or upstream model name.  Run it after model rewriting so
    # the exact payload returned here is the one sent and cached.
    if model_api.compatibility == KIMI_K3_PAPYRUS:
        upstream_payload = fold_chat_messages(upstream_payload)
    return upstream_payload


class ProxyRuntime:
    def __init__(
        self,
        registry: Optional[ProxyRegistry] = None,
        affinity_store: Optional[ProxyAffinityStore] = None,
    ) -> None:
        self.registry = registry or ProxyRegistry()
        self.affinity_store = affinity_store or ProxyAffinityStore()
        self._auth_lock = threading.Lock()
        self._auth_providers: Dict[str, Tuple[Tuple, ProxyAuthProvider]] = {}

    def _auth_provider(self, profile: ProxyProfileConfig) -> ProxyAuthProvider:
        fingerprint = profile.auth.fingerprint()
        with self._auth_lock:
            current = self._auth_providers.get(profile.name)
            if current is None or current[0] != fingerprint:
                current = (fingerprint, ProxyAuthProvider(profile.auth))
                self._auth_providers[profile.name] = current
            return current[1]

    def _build_headers(
        self,
        profile: ProxyProfileConfig,
        api: ProxyApiConfig,
        model: ProxyModelConfig,
        model_api: ProxyModelApiConfig,
        auth_provider: ProxyAuthProvider,
        affinity_token: Optional[str],
    ) -> Dict[str, str]:
        headers = _merge_headers(profile.headers, api.headers, model.headers, model_api.headers)
        token = auth_provider.get_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if profile.affinity.enabled and affinity_token:
            headers[profile.affinity.request_header] = affinity_token
        return headers

    def _capture_affinity(
        self,
        profile: ProxyProfileConfig,
        key: Optional[str],
        response: requests.Response,
    ) -> None:
        if not profile.affinity.enabled or key is None:
            return
        token = response.headers.get(profile.affinity.response_header)
        if token:
            self.affinity_store.set(key, token, profile.affinity.persist)

    def _post_with_retries(
        self,
        profile: ProxyProfileConfig,
        api: ProxyApiConfig,
        model: ProxyModelConfig,
        model_api: ProxyModelApiConfig,
        payload: dict,
        stream: bool,
        affinity_key_value: Optional[str],
        initial_affinity_token: Optional[str],
    ) -> requests.Response:
        auth_provider = self._auth_provider(profile)
        connection_attempt = 0
        auth_retry_attempted = False
        affinity_token = initial_affinity_token

        while True:
            if profile.affinity.enabled and affinity_key_value is not None:
                affinity_token = self.affinity_store.get(affinity_key_value) or affinity_token
            headers = self._build_headers(
                profile, api, model, model_api, auth_provider, affinity_token
            )
            try:
                response = requests.post(
                    api.upstream_url,
                    headers=headers,
                    json=payload,
                    stream=stream,
                    timeout=api.timeout_seconds,
                )
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
                if connection_attempt >= api.max_connection_retries:
                    raise ProxyRequestError(
                        f"Configured upstream connection failed after {connection_attempt + 1} attempt(s): "
                        f"{type(exc).__name__}"
                    ) from exc
                time.sleep(min(2 ** connection_attempt, 8))
                connection_attempt += 1
                continue

            self._capture_affinity(profile, affinity_key_value, response)

            if response.status_code == 401 and profile.auth.type == "bearer_command" and not auth_retry_attempted:
                auth_retry_attempted = True
                auth_provider.invalidate()
                response.close()
                continue

            return response

    def post(
        self,
        profile: ProxyProfileConfig,
        api: ProxyApiConfig,
        model: ProxyModelConfig,
        model_api: ProxyModelApiConfig,
        payload: dict,
        stream: bool,
    ) -> ProxyUpstreamResult:
        upstream_payload = transform_payload(payload, api, model, model_api)
        key: Optional[str] = None
        affinity_token: Optional[str] = None

        if profile.affinity.enabled:
            key = affinity_key(profile, api, model, model_api)
            affinity_token = self.affinity_store.get(key)

        try:
            if key is not None and affinity_token is None:
                discovery_lock = self.affinity_store.discovery_lock(key)
                with discovery_lock:
                    affinity_token = self.affinity_store.get(key)
                    response = self._post_with_retries(
                        profile,
                        api,
                        model,
                        model_api,
                        upstream_payload,
                        stream,
                        key,
                        affinity_token,
                    )
            else:
                response = self._post_with_retries(
                    profile,
                    api,
                    model,
                    model_api,
                    upstream_payload,
                    stream,
                    key,
                    affinity_token,
                )
        except ProxyAuthError as exc:
            raise ProxyRequestError(str(exc)) from exc

        return ProxyUpstreamResult(response=response, payload=upstream_payload)
