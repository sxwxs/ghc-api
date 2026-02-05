"""
API helper functions for GitHub Copilot API
"""

import time
import uuid
from typing import Dict, List

import requests

from .config import GITHUB_API_BASE_URL
from .state import state


def get_copilot_base_url() -> str:
    """Get the Copilot API base URL based on account type"""
    if state.account_type == "individual":
        return "https://api.githubcopilot.com"
    return f"https://api.{state.account_type}.githubcopilot.com"


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
    if enable_vision:
        headers["Copilot-Vision-Request"] = "true"
    return headers


def refresh_copilot_token() -> None:
    """Refresh the Copilot token from GitHub"""
    with state.token_lock:
        # Check if token is still valid
        if state.copilot_token and time.time() < state.token_expires_at - 60:
            return

        print("Refreshing Copilot token...")
        response = requests.get(
            f"{GITHUB_API_BASE_URL}/copilot_internal/v2/token",
            headers=get_github_headers(),
            timeout=30,
        )

        if not response.ok:
            raise Exception(f"Failed to get Copilot token: {response.status_code} {response.text}")

        data = response.json()
        state.copilot_token = data["token"]
        state.token_expires_at = time.time() + data.get("refresh_in", 1800)
        print("Copilot token refreshed successfully")


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
        print(f"Loaded {len(state.models.get('data', []))} models")
    else:
        print(f"Failed to fetch models: {response.status_code}")


def ensure_copilot_token() -> None:
    """Ensure we have a valid Copilot token"""
    if not state.copilot_token or time.time() >= state.token_expires_at - 60:
        refresh_copilot_token()


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
