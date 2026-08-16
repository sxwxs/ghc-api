"""
API helper functions for GitHub Copilot API
"""

import time
import uuid
from typing import Dict, List
from urllib.parse import urlsplit, urlunsplit

import requests

from .config import GITHUB_API_BASE_URL, chat_completions_model_support
from .state import state
from .utils import log_upstream_error


CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"
_configured_chat_completions_support_by_models_id: Dict[int, set[str]] = {}


def get_copilot_base_url() -> str:
    """Get the Copilot API base URL based on account type or override."""
    if state.copilot_api_base_url:
        return state.copilot_api_base_url.rstrip("/")
    if state.account_type == "individual":
        return "https://api.githubcopilot.com"
    return f"https://api.{state.account_type}.githubcopilot.com"


def get_github_api_base_url() -> str:
    """Get the GitHub API base URL, allowing an explicit local override."""
    return (state.github_api_base_url or GITHUB_API_BASE_URL).rstrip("/")


def resolve_ghe_endpoints(endpoint: str) -> Dict[str, str]:
    """Normalize a GHE tenant/web/API endpoint into all required origins.

    Accepts ``octocorp.ghe.com``, ``https://octocorp.ghe.com``,
    ``https://api.octocorp.ghe.com``, or
    ``https://copilot-api.octocorp.ghe.com``.
    """
    raw = (endpoint or "").strip()
    if not raw:
        raise ValueError("GHE endpoint is required")
    if "://" not in raw:
        raw = f"https://{raw}"

    parsed = urlsplit(raw)
    hostname = parsed.hostname or ""
    if parsed.scheme != "https":
        raise ValueError("GHE endpoint must use HTTPS")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("GHE endpoint contains unsupported URL components")
    if parsed.path not in ("", "/"):
        raise ValueError("GHE endpoint must not contain a path")
    if parsed.port not in (None, 443):
        raise ValueError("GHE endpoint only supports the default HTTPS port")

    if hostname.startswith("copilot-api."):
        tenant_hostname = hostname[len("copilot-api."):]
    elif hostname.startswith("api."):
        tenant_hostname = hostname[len("api."):]
    else:
        tenant_hostname = hostname

    if tenant_hostname == "ghe.com" or not tenant_hostname.endswith(".ghe.com"):
        raise ValueError("GHE endpoint must identify a tenant under *.ghe.com")

    return {
        "github_web_base_url": f"https://{tenant_hostname}",
        "github_api_base_url": f"https://api.{tenant_hostname}",
        "copilot_api_base_url": f"https://copilot-api.{tenant_hostname}",
    }


def get_github_web_base_url() -> str:
    """Derive the GitHub web/OAuth origin from the configured API base URL.

    GitHub.com uses ``api.github.com`` -> ``github.com``. GitHub Enterprise
    Cloud with data residency uses ``api.<tenant>.ghe.com`` ->
    ``<tenant>.ghe.com``. Arbitrary benchmark/private-gateway overrides remain
    valid for API traffic, but Device Flow is disabled when their OAuth origin
    cannot be derived safely.
    """
    api_base_url = get_github_api_base_url()
    parsed = urlsplit(api_base_url)
    hostname = parsed.hostname or ""

    if parsed.scheme != "https":
        raise ValueError(
            "GitHub Device Flow requires an HTTPS github_api_base_url "
            "for github.com or api.<tenant>.ghe.com"
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("github_api_base_url contains unsupported URL components")
    if parsed.path not in ("", "/"):
        raise ValueError("github_api_base_url must not contain a path for GitHub Device Flow")

    if hostname == "api.github.com":
        web_hostname = "github.com"
    elif hostname.startswith("api.") and hostname.endswith(".ghe.com"):
        web_hostname = hostname[4:]
        if web_hostname == "ghe.com":
            raise ValueError("github_api_base_url must include a GHE tenant subdomain")
    else:
        raise ValueError(
            "Cannot derive the GitHub Device Flow host from github_api_base_url; "
            "expected https://api.github.com or https://api.<tenant>.ghe.com"
        )

    if parsed.port not in (None, 443):
        raise ValueError("GitHub Device Flow only supports the default HTTPS port")
    return urlunsplit(("https", web_hostname, "", "", ""))


def get_github_headers() -> Dict[str, str]:
    """Get headers for GitHub API requests"""
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"token {state.github_token}",
        "Editor-Version": f"vscode/{state.vscode_version}",
        "Editor-Plugin-Version": state.editor_plugin_version,
        "User-Agent": state.user_agent,
        "X-GitHub-Api-Version": state.api_version,
        "X-VSCode-User-Agent-Library-Version": "electron-fetch",
    }


def get_copilot_headers(enable_vision: bool = False) -> Dict[str, str]:
    """Get headers for Copilot API requests"""
    headers = {
        "Authorization": f"Bearer {state.copilot_token}",
        "Content-Type": "application/json",
        "Copilot-Integration-Id": "vscode-chat",
        "Editor-Version": f"vscode/{state.vscode_version}",
        "Editor-Plugin-Version": state.editor_plugin_version,
        "User-Agent": state.user_agent,
        "OpenAI-Intent": "conversation-panel",
        "X-GitHub-Api-Version": state.api_version,
        "X-Request-Id": str(uuid.uuid4()),
        "X-VSCode-User-Agent-Library-Version": "electron-fetch",
    }
    if enable_vision:
        headers["Copilot-Vision-Request"] = "true"
    return headers


def refresh_copilot_token(force: bool = False) -> None:
    """Refresh the Copilot token from GitHub and record the latest outcome."""
    with state.token_lock:
        if not force and state.copilot_token and time.time() < state.token_expires_at - 60:
            return

        state.token_refresh_last_attempt_at = time.time()
        token_endpoint = f"{get_github_api_base_url()}/copilot_internal/v2/token"
        response = None
        print("Refreshing Copilot token...")
        try:
            response = requests.get(
                token_endpoint,
                headers=get_github_headers(),
                timeout=30,
            )
            if not response.ok:
                response_text = (response.text or "").strip()
                if len(response_text) > 500:
                    response_text = response_text[:500] + "... (response truncated)"
                detail = f" {response_text}" if response_text else ""
                raise RuntimeError(
                    f"Failed to get Copilot token: HTTP {response.status_code}.{detail}"
                )

            data = response.json()
            state.copilot_token = data["token"]
            state.token_expires_at = time.time() + data.get("refresh_in", 1800)
            state.token_refresh_last_succeeded = True
            state.token_refresh_last_success_at = time.time()
            state.token_refresh_last_error = None
            print("Copilot token refreshed successfully")
        except Exception as exc:
            state.token_refresh_last_succeeded = False
            state.token_refresh_last_error = str(exc)
            error_logged = log_upstream_error(
                operation="copilot_token_refresh",
                endpoint=token_endpoint,
                status_code=response.status_code if response is not None else None,
                response_body=response.text if response is not None else "",
                error=str(exc),
            )
            print("\nCopilot token refresh failed.")
            if error_logged:
                print("The upstream error response was written to error.log in the ghc-api config directory.")
            print("This may be caused by a temporary GitHub service issue; retrying later may resolve it.")
            print("If the problem persists, clear the locally saved GitHub token and sign in again:")
            print("  ghc-api --delete-github-token")
            print("  ghc-api --github-device-login")
            raise


def fetch_models() -> None:
    """Fetch available models from Copilot API"""
    ensure_copilot_token()
    response = requests.get(
        f"{get_copilot_base_url()}/models",
        headers=get_copilot_headers(),
        timeout=30,
    )

    if response.ok:
        state.models = response.json()
        updated_count = apply_configured_chat_completions_support(state.models, reset_tracking=True)
        print(f"Loaded {len(state.models.get('data', []))} models")
        if updated_count:
            print(f"Added chat completions endpoint support to {updated_count} configured model(s)")
    else:
        print(f"Failed to fetch models: {response.status_code}")


def ensure_copilot_token() -> None:
    """Ensure we have a valid Copilot token"""
    if not state.copilot_token or time.time() >= state.token_expires_at - 60:
        refresh_copilot_token()


def apply_configured_chat_completions_support(models: Dict, reset_tracking: bool = False) -> int:
    """Add chat completions endpoints to configured models in a model listing.

    Removes endpoint support added by previous calls when a model no longer
    matches the current config. Returns the number of model entries changed.
    """
    if not isinstance(models, dict):
        return 0

    models_key = id(models)
    if reset_tracking:
        _configured_chat_completions_support_by_models_id.clear()
    added_model_ids = _configured_chat_completions_support_by_models_id.setdefault(models_key, set())

    data = models.get("data")
    if not isinstance(data, list):
        return 0

    updated_count = 0
    for model in data:
        if not isinstance(model, dict):
            continue

        model_id = model.get("id")
        if not isinstance(model_id, str):
            continue

        supported_endpoints = model.get("supported_endpoints")
        if not isinstance(supported_endpoints, list):
            supported_endpoints = []

        changed = False
        matches_config = chat_completions_model_support.matches(model_id)
        was_added_by_config = model_id in added_model_ids

        if matches_config and CHAT_COMPLETIONS_ENDPOINT not in supported_endpoints:
            supported_endpoints.append(CHAT_COMPLETIONS_ENDPOINT)
            added_model_ids.add(model_id)
            changed = True
        elif not matches_config and was_added_by_config:
            supported_endpoints = [
                endpoint for endpoint in supported_endpoints
                if endpoint != CHAT_COMPLETIONS_ENDPOINT
            ]
            added_model_ids.discard(model_id)
            changed = True

        if changed:
            model["supported_endpoints"] = supported_endpoints
            updated_count += 1

    return updated_count


def is_configured_chat_completions_support_added(models: Dict, model_id: str) -> bool:
    """Return True if this process added chat completions support for a model."""
    if not isinstance(models, dict) or not isinstance(model_id, str):
        return False
    return model_id in _configured_chat_completions_support_by_models_id.get(id(models), set())


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text using tiktoken"""
    try:
        import tiktoken
        # Try to get the encoding for the model
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fall back to cl100k_base for unknown models
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Rough estimation: 4 characters per token
        return len(text) // 4


def count_message_tokens(messages: List[Dict], model: str = "gpt-4") -> int:
    """Count tokens in a list of messages"""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content, model)
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    total += count_tokens(part.get("text", ""), model)
    return total


def advertises_anthropic_messages_api(model_id: str) -> bool:
    """Return whether model metadata advertises the native Messages endpoint."""
    if not state.models or not state.models.get("data"):
        return False
    model = next((m for m in state.models["data"] if m.get("id") == model_id), None)
    if not model:
        return False
    return "/v1/messages" in model.get("supported_endpoints", [])


def supports_direct_anthropic_api(model_id: str) -> bool:
    """Check if a model should use the direct Anthropic API path."""
    if state.redirect_anthropic:
        return False
    return advertises_anthropic_messages_api(model_id)


def supports_responses_api(model_id: str) -> bool:
    """Check if a model supports the OpenAI Responses API (/v1/responses).

    Returns True if the model's supported_endpoints includes "/v1/responses".
    """
    if not state.models or not state.models.get("data"):
        return False

    model = next((m for m in state.models["data"] if m.get("id") == model_id), None)
    if not model:
        return False

    supported_endpoints = model.get("supported_endpoints", [])
    return "/responses" in supported_endpoints or "/v1/responses" in supported_endpoints


def anthropic_responses_wire_profile(model_id: str) -> str:
    """Resolve the configured wire profile for a Responses-backed model."""
    configured = getattr(state, "anthropic_responses_model_profiles", {}) or {}
    if isinstance(configured, dict):
        exact = configured.get(model_id)
        if isinstance(exact, str) and exact:
            return exact
        # A trailing '*' denotes a prefix rule without adding another config
        # structure solely for this small capability map.
        prefix_matches = [
            (pattern[:-1], profile)
            for pattern, profile in configured.items()
            if (
                isinstance(pattern, str)
                and pattern.endswith("*")
                and model_id.startswith(pattern[:-1])
                and isinstance(profile, str)
                and profile
            )
        ]
        if prefix_matches:
            # Most-specific prefix wins regardless of YAML/dict insertion order.
            return max(prefix_matches, key=lambda item: len(item[0]))[1]
    return getattr(state, "anthropic_responses_wire_profile", "copilot_responses_lite")


def supports_embeddings_api(model_id: str) -> bool:
    """Return whether Copilot advertises the model as an embedding model.

    Embedding entries currently do not expose ``supported_endpoints``. Their
    ``capabilities.type`` field is therefore the authoritative discriminator.
    """
    if not state.models or not state.models.get("data"):
        return False

    model = next((m for m in state.models["data"] if m.get("id") == model_id), None)
    if not isinstance(model, dict):
        return False
    return model.get("capabilities", {}).get("type") == "embeddings"


def supported_reasoning_efforts(model_id: str) -> set:
    """Reasoning-effort values Copilot reports for a model.

    Empty set if the model has no reasoning_effort capability, is unknown,
    or models are not loaded yet (callers should then drop output_config).
    """
    if not state.models or not state.models.get("data"):
        return set()
    model = next((m for m in state.models["data"] if m.get("id") == model_id), None)
    if model is None:
        return set()
    efforts = model.get("capabilities", {}).get("supports", {}).get("reasoning_effort")
    return set(efforts) if efforts else set()
