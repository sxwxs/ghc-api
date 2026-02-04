#!/usr/bin/env python3
"""
GitHub Copilot API Proxy Server

A Flask application that replicates the functionality of the Node.js copilot-api project,
serving as a proxy server for GitHub Copilot API with caching and monitoring capabilities.
"""

import argparse
import json
import os
import platform
import sys
import threading
import time
import uuid
import yaml
from collections import OrderedDict
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional, Generator, Tuple

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
import requests


# ============================================================================
# Configuration
# ============================================================================

app = Flask(__name__)

# Environment variables
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
PORT = int(os.environ.get("PORT", 8313))
HOST = os.environ.get("HOST", "localhost")
DEBUG = os.environ.get("DEBUG", "true").lower() == "true"

# API Configuration
GITHUB_API_BASE_URL = "https://api.github.com"
COPILOT_API_BASE_URL = "https://api.githubcopilot.com"

COPILOT_VERSION = "0.26.7"
EDITOR_PLUGIN_VERSION = f"copilot-chat/{COPILOT_VERSION}"
USER_AGENT = f"GitHubCopilotChat/{COPILOT_VERSION}"
API_VERSION = "2025-04-01"
VSCODE_VERSION = "1.93.0"

# ============================================================================
# State Management
# ============================================================================

class State:
    """Global application state"""
    def __init__(self):
        self.github_token: str = GITHUB_TOKEN
        self.copilot_token: Optional[str] = None
        self.models: Optional[Dict] = None
        self.account_type: str = "individual"
        self.vscode_version: str = VSCODE_VERSION
        self.token_expires_at: float = 0
        self.token_lock = threading.Lock()

state = State()

# ============================================================================
# Request/Response Cache
# ============================================================================

class RequestCache:
    """Thread-safe cache for storing API requests and responses"""

    # Request states
    STATE_PENDING = "pending"
    STATE_SENDING = "sending"
    STATE_RECEIVING = "receiving"
    STATE_COMPLETED = "completed"
    STATE_ERROR = "error"

    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.Lock()
        self.request_count = 0
        self.bytes_sent = 0
        self.bytes_received = 0
        self.model_stats: Dict[str, Dict] = {}
        self.endpoint_stats: Dict[str, Dict] = {}

    def start_request(self, request_id: str, data: Dict) -> None:
        """Start tracking a new request (before sending to upstream)"""
        with self.lock:
            if len(self.cache) >= self.max_entries:
                self.cache.popitem(last=False)

            self.cache[request_id] = {
                "id": request_id,
                "timestamp": datetime.now().isoformat(),
                "request_body": data.get("request_body"),
                "response_body": None,
                "model": data.get("model", "unknown"),
                "endpoint": data.get("endpoint", "unknown"),
                "status_code": None,
                "request_size": data.get("request_size", 0),
                "response_size": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "duration": 0,
                "state": self.STATE_PENDING,
            }

    def update_request_state(self, request_id: str, state: str, **kwargs) -> None:
        """Update the state and optional fields of an existing request"""
        with self.lock:
            if request_id in self.cache:
                self.cache[request_id]["state"] = state
                for key, value in kwargs.items():
                    if key in self.cache[request_id]:
                        self.cache[request_id][key] = value

    def complete_request(self, request_id: str, data: Dict) -> None:
        """Complete a request with response data and update statistics"""
        with self.lock:
            if request_id in self.cache:
                # Update existing entry
                entry = self.cache[request_id]
                entry["response_body"] = data.get("response_body")
                entry["status_code"] = data.get("status_code", 200)
                entry["response_size"] = data.get("response_size", 0)
                entry["input_tokens"] = data.get("input_tokens", 0)
                entry["output_tokens"] = data.get("output_tokens", 0)
                entry["duration"] = data.get("duration", 0)
                entry["state"] = self.STATE_COMPLETED if data.get("status_code", 200) < 400 else self.STATE_ERROR
            else:
                # Fallback: create new entry if somehow missing
                if len(self.cache) >= self.max_entries:
                    self.cache.popitem(last=False)

                self.cache[request_id] = {
                    "id": request_id,
                    "timestamp": datetime.now().isoformat(),
                    "request_body": data.get("request_body"),
                    "response_body": data.get("response_body"),
                    "model": data.get("model", "unknown"),
                    "endpoint": data.get("endpoint", "unknown"),
                    "status_code": data.get("status_code", 200),
                    "request_size": data.get("request_size", 0),
                    "response_size": data.get("response_size", 0),
                    "input_tokens": data.get("input_tokens", 0),
                    "output_tokens": data.get("output_tokens", 0),
                    "duration": data.get("duration", 0),
                    "state": self.STATE_COMPLETED if data.get("status_code", 200) < 400 else self.STATE_ERROR,
                }

            self.request_count += 1
            self.bytes_sent += data.get("request_size", 0)
            self.bytes_received += data.get("response_size", 0)

            # Update model stats
            model = data.get("model", "unknown")
            if model not in self.model_stats:
                self.model_stats[model] = {
                    "request_count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "bytes_sent": 0,
                    "bytes_received": 0,
                }
            self.model_stats[model]["request_count"] += 1
            self.model_stats[model]["input_tokens"] += data.get("input_tokens", 0)
            self.model_stats[model]["output_tokens"] += data.get("output_tokens", 0)
            self.model_stats[model]["bytes_sent"] += data.get("request_size", 0)
            self.model_stats[model]["bytes_received"] += data.get("response_size", 0)

            # Update endpoint stats
            endpoint = data.get("endpoint", "unknown")
            if endpoint not in self.endpoint_stats:
                self.endpoint_stats[endpoint] = {
                    "request_count": 0,
                    "bytes_sent": 0,
                    "bytes_received": 0,
                }
            self.endpoint_stats[endpoint]["request_count"] += 1
            self.endpoint_stats[endpoint]["bytes_sent"] += data.get("request_size", 0)
            self.endpoint_stats[endpoint]["bytes_received"] += data.get("response_size", 0)

    def add_request(self, request_id: str, data: Dict) -> None:
        """Add a request to the cache (legacy method for backwards compatibility)"""
        # Check if request already exists (started with start_request)
        with self.lock:
            exists = request_id in self.cache

        if exists:
            self.complete_request(request_id, data)
        else:
            # Legacy path: create and complete in one step
            self.start_request(request_id, data)
            self.complete_request(request_id, data)

    def get_request(self, request_id: str) -> Optional[Dict]:
        """Get a specific request by ID"""
        with self.lock:
            return self.cache.get(request_id)

    def get_recent_requests(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        """Get recent requests with pagination"""
        with self.lock:
            items = list(reversed(list(self.cache.values())))
            return items[offset:offset + limit]

    def get_total_count(self) -> int:
        """Get total number of cached requests"""
        with self.lock:
            return len(self.cache)

    def get_stats(self) -> Dict:
        """Get overall statistics"""
        with self.lock:
            return {
                "total_requests": self.request_count,
                "cached_requests": len(self.cache),
                "bytes_sent": self.bytes_sent,
                "bytes_received": self.bytes_received,
                "model_stats": dict(self.model_stats),
                "endpoint_stats": dict(self.endpoint_stats),
            }

    def search_requests(self, query: str, limit: int = 50, offset: int = 0) -> List[Dict]:
        """Search requests by model, endpoint, or content"""
        with self.lock:
            results = []
            query_lower = query.lower()
            for item in reversed(list(self.cache.values())):
                if (query_lower in item.get("model", "").lower() or
                    query_lower in item.get("endpoint", "").lower() or
                    query_lower in json.dumps(item.get("request_body", {})).lower()):
                    results.append(item)
            return results[offset:offset + limit]

    def fulltext_search(self, query: str, limit: int = 50, offset: int = 0) -> Tuple[List[Dict], int]:
        """Full-text search in request and response bodies"""
        with self.lock:
            results = []
            query_lower = query.lower()
            for item in reversed(list(self.cache.values())):
                # Search in request body
                request_body_str = json.dumps(item.get("request_body", {})).lower()
                # Search in response body
                response_body_str = json.dumps(item.get("response_body", {})).lower()

                if query_lower in request_body_str or query_lower in response_body_str:
                    results.append(item)

            total = len(results)
            return results[offset:offset + limit], total

    def get_all_requests(self) -> List[Dict]:
        """Get all requests for export"""
        with self.lock:
            return list(self.cache.values())

    def import_request(self, data: Dict) -> None:
        """Import a single request entry"""
        with self.lock:
            request_id = data.get("id", str(uuid.uuid4()))

            if len(self.cache) >= self.max_entries:
                self.cache.popitem(last=False)

            self.cache[request_id] = {
                "id": request_id,
                "timestamp": data.get("timestamp", datetime.now().isoformat()),
                "request_body": data.get("request_body"),
                "response_body": data.get("response_body"),
                "model": data.get("model", "unknown"),
                "endpoint": data.get("endpoint", "unknown"),
                "status_code": data.get("status_code", 200),
                "request_size": data.get("request_size", 0),
                "response_size": data.get("response_size", 0),
                "input_tokens": data.get("input_tokens", 0),
                "output_tokens": data.get("output_tokens", 0),
                "duration": data.get("duration", 0),
                "state": data.get("state", "completed"),
            }

            # Update stats
            self.request_count += 1
            self.bytes_sent += data.get("request_size", 0)
            self.bytes_received += data.get("response_size", 0)

            model = data.get("model", "unknown")
            if model not in self.model_stats:
                self.model_stats[model] = {
                    "request_count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "bytes_sent": 0,
                    "bytes_received": 0,
                }
            self.model_stats[model]["request_count"] += 1
            self.model_stats[model]["input_tokens"] += data.get("input_tokens", 0)
            self.model_stats[model]["output_tokens"] += data.get("output_tokens", 0)
            self.model_stats[model]["bytes_sent"] += data.get("request_size", 0)
            self.model_stats[model]["bytes_received"] += data.get("response_size", 0)

            endpoint = data.get("endpoint", "unknown")
            if endpoint not in self.endpoint_stats:
                self.endpoint_stats[endpoint] = {
                    "request_count": 0,
                    "bytes_sent": 0,
                    "bytes_received": 0,
                }
            self.endpoint_stats[endpoint]["request_count"] += 1
            self.endpoint_stats[endpoint]["bytes_sent"] += data.get("request_size", 0)
            self.endpoint_stats[endpoint]["bytes_received"] += data.get("response_size", 0)

cache = RequestCache()

# ============================================================================
# GitHub Token Management
# ============================================================================

def get_config_dir():
    """Get the config directory path based on the OS"""
    if platform.system() == "Windows":
        return os.path.expandvars("%APPDATA%/ghc-api")
    else:
        return os.path.expanduser("~/.ghc-api")

def get_token_file_path():
    """Get the path to the token file"""
    config_dir = get_config_dir()
    os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, "github_token.txt")

# GitHub OAuth App for Device Flow (using GitHub CLI's client ID as it's public)
GITHUB_OAUTH_CLIENT_ID = "01ab8ac9400c4e429b23"  # GitHub CLI client ID


def load_github_token_from_file() -> Optional[str]:
    """Load GitHub token from github_token.txt file in config directory"""
    token_file_path = get_token_file_path()
    if os.path.exists(token_file_path):
        try:
            with open(token_file_path, "r", encoding="utf-8") as f:
                token = f.read().strip()
                if token:
                    print(f"Loaded GitHub token from {token_file_path}")
                    return token
        except Exception as e:
            print(f"Failed to read token file: {e}")
    return None


def save_github_token_to_file(token: str) -> bool:
    """Save GitHub token to github_token.txt file in config directory"""
    token_file_path = get_token_file_path()
    try:
        with open(token_file_path, "w", encoding="utf-8") as f:
            f.write(token)
        print(f"Saved GitHub token to {token_file_path}")
        return True
    except Exception as e:
        print(f"Failed to save token file: {e}")
        return False


def authenticate_github_device_flow() -> Optional[str]:
    """
    Authenticate using GitHub Device Flow.
    This opens a browser for the user to authorize the app.
    Returns the access token if successful, None otherwise.
    """
    print("\n" + "=" * 60)
    print("GitHub Device Flow Authentication")
    print("=" * 60)

    # Step 1: Request device and user codes
    device_code_url = "https://github.com/login/device/code"
    response = requests.post(
        device_code_url,
        data={
            "client_id": GITHUB_OAUTH_CLIENT_ID,
            "scope": "read:user copilot",
        },
        headers={"Accept": "application/json"},
        timeout=30,
    )

    if not response.ok:
        print(f"Failed to get device code: {response.status_code} {response.text}")
        return None

    data = response.json()
    device_code = data.get("device_code")
    user_code = data.get("user_code")
    verification_uri = data.get("verification_uri")
    expires_in = data.get("expires_in", 900)
    interval = data.get("interval", 5)

    print(f"\nPlease visit: {verification_uri}")
    print(f"And enter the code: {user_code}")
    print(f"\nWaiting for authorization (expires in {expires_in} seconds)...")

    # Try to open browser automatically
    try:
        import webbrowser
        webbrowser.open(verification_uri)
        print("(Browser opened automatically)")
    except Exception:
        pass

    # Step 2: Poll for authorization
    token_url = "https://github.com/login/oauth/access_token"
    start_time = time.time()

    while time.time() - start_time < expires_in:
        time.sleep(interval)

        response = requests.post(
            token_url,
            data={
                "client_id": GITHUB_OAUTH_CLIENT_ID,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            headers={"Accept": "application/json"},
            timeout=30,
        )

        if not response.ok:
            continue

        token_data = response.json()

        if token_data.get("error") == "authorization_pending":
            print(".", end="", flush=True)
            continue
        elif token_data.get("error") == "slow_down":
            interval += 5
            continue
        elif token_data.get("error") == "expired_token":
            print("\nAuthorization expired. Please try again.")
            return None
        elif token_data.get("error") == "access_denied":
            print("\nAuthorization denied by user.")
            return None
        elif token_data.get("error"):
            print(f"\nError: {token_data.get('error_description', token_data['error'])}")
            return None
        elif token_data.get("access_token"):
            print("\n\nAuthorization successful!")
            return token_data["access_token"]

    print("\nAuthorization timed out. Please try again.")
    return None


def get_github_token() -> Optional[str]:
    """
    Get GitHub token using the following priority:
    1. Environment variable GITHUB_TOKEN
    2. Token file (~/.ghc-api/github_token.txt)
    3. GitHub Device Flow authentication (interactive)
    """
    # Priority 1: Environment variable
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        print("Using GitHub token from GITHUB_TOKEN environment variable")
        return token

    # Priority 2: Token file
    token = load_github_token_from_file()
    if token:
        return token

    # Priority 3: GitHub Device Flow (interactive)
    print("\nNo GitHub token found. Starting GitHub Device Flow authentication...")
    token = authenticate_github_device_flow()
    if token:
        save_github_token_to_file(token)
        return token

    return None


# ============================================================================
# Token Management
# ============================================================================

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
        "Editor-Plugin-Version": EDITOR_PLUGIN_VERSION,
        "User-Agent": USER_AGENT,
        "X-GitHub-Api-Version": API_VERSION,
        "X-VSCode-User-Agent-Library-Version": "electron-fetch",
    }


def get_copilot_headers(enable_vision: bool = False) -> Dict[str, str]:
    """Get headers for Copilot API requests"""
    headers = {
        "Authorization": f"Bearer {state.copilot_token}",
        "Content-Type": "application/json",
        "Copilot-Integration-Id": "vscode-chat",
        "Editor-Version": f"vscode/{state.vscode_version}",
        "Editor-Plugin-Version": EDITOR_PLUGIN_VERSION,
        "User-Agent": USER_AGENT,
        "OpenAI-Intent": "conversation-panel",
        "X-GitHub-Api-Version": API_VERSION,
        "X-Request-Id": str(uuid.uuid4()),
        "X-VSCode-User-Agent-Library-Version": "electron-fetch",
    }
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


# ============================================================================
# Token Counting
# ============================================================================

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


# ============================================================================
# API Format Translation
# ============================================================================

def translate_model_name(model: str) -> str:
    """Translate model names for Copilot API compatibility"""
    if model.startswith("claude-sonnet-4-"):
        return "claude-sonnet-4"
    elif model.startswith("claude-opus-4-"):
        return "claude-opus-4"
    return model


def translate_anthropic_to_openai(payload: Dict) -> Dict:
    """Translate Anthropic API format to OpenAI format"""
    messages = []

    # Handle system prompt
    system = payload.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            # Filter out billing headers which cause errors in GitHub API
            filtered_system = [
                block.get("text", "") for block in system
                if isinstance(block, dict) and
                block.get("type") == "text" and
                "x-anthropic-billing-header" not in block.get("text", "")
            ]
            if filtered_system:
                system_text = "\n\n".join(filtered_system)
                messages.append({"role": "system", "content": system_text})

    # Translate messages
    for msg in payload.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            if isinstance(content, list):
                # Handle tool results and other content blocks
                tool_results = [b for b in content if b.get("type") == "tool_result"]
                other_blocks = [b for b in content if b.get("type") != "tool_result"]

                # Tool results become tool messages
                for tr in tool_results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tr.get("tool_use_id"),
                        "content": tr.get("content", ""),
                    })

                # Other content
                if other_blocks:
                    translated_content = translate_content_blocks(other_blocks)
                    if translated_content:
                        messages.append({"role": "user", "content": translated_content})
            else:
                messages.append({"role": "user", "content": content})

        elif role == "assistant":
            if isinstance(content, list):
                tool_uses = [b for b in content if b.get("type") == "tool_use"]
                text_blocks = [b for b in content if b.get("type") in ("text", "thinking")]

                text_content = "\n\n".join(
                    b.get("text", "") if b.get("type") == "text" else b.get("thinking", "")
                    for b in text_blocks
                )

                if tool_uses:
                    messages.append({
                        "role": "assistant",
                        "content": text_content or None,
                        "tool_calls": [
                            {
                                "id": tu.get("id"),
                                "type": "function",
                                "function": {
                                    "name": tu.get("name"),
                                    "arguments": json.dumps(tu.get("input", {})),
                                },
                            }
                            for tu in tool_uses
                        ],
                    })
                else:
                    messages.append({"role": "assistant", "content": text_content})
            else:
                messages.append({"role": "assistant", "content": content})

    # Build OpenAI payload
    openai_payload = {
        "model": translate_model_name(payload.get("model", "")),
        "messages": messages,
        "max_tokens": payload.get("max_tokens"),
        "stream": payload.get("stream", False),
    }

    if payload.get("temperature") is not None:
        openai_payload["temperature"] = payload["temperature"]
    if payload.get("top_p") is not None:
        openai_payload["top_p"] = payload["top_p"]
    if payload.get("stop_sequences"):
        openai_payload["stop"] = payload["stop_sequences"]

    # Translate tools
    if payload.get("tools"):
        openai_payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description"),
                    "parameters": tool.get("input_schema", {}),
                },
            }
            for tool in payload["tools"]
        ]

    # Translate tool_choice
    tool_choice = payload.get("tool_choice")
    if tool_choice:
        choice_type = tool_choice.get("type")
        if choice_type == "auto":
            openai_payload["tool_choice"] = "auto"
        elif choice_type == "any":
            openai_payload["tool_choice"] = "required"
        elif choice_type == "none":
            openai_payload["tool_choice"] = "none"
        elif choice_type == "tool" and tool_choice.get("name"):
            openai_payload["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice["name"]},
            }

    return openai_payload


def translate_content_blocks(blocks: List[Dict]) -> Any:
    """Translate Anthropic content blocks to OpenAI format"""
    has_image = any(b.get("type") == "image" for b in blocks)

    if not has_image:
        # Just combine text
        texts = []
        for b in blocks:
            if b.get("type") == "text":
                texts.append(b.get("text", ""))
            elif b.get("type") == "thinking":
                texts.append(b.get("thinking", ""))
        return "\n\n".join(texts) if texts else None

    # Handle mixed content with images
    parts = []
    for b in blocks:
        if b.get("type") == "text":
            parts.append({"type": "text", "text": b.get("text", "")})
        elif b.get("type") == "thinking":
            parts.append({"type": "text", "text": b.get("thinking", "")})
        elif b.get("type") == "image":
            source = b.get("source", {})
            parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{source.get('media_type')};base64,{source.get('data')}",
                },
            })
    return parts if parts else None


def map_openai_stop_reason_to_anthropic(reason: Optional[str]) -> Optional[str]:
    """Map OpenAI stop reason to Anthropic format"""
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "refusal",
    }
    return mapping.get(reason) if reason else None


def translate_openai_to_anthropic(response: Dict) -> Dict:
    """Translate OpenAI response to Anthropic format"""
    content = []
    stop_reason = None

    for choice in response.get("choices", []):
        message = choice.get("message", {})

        # Handle text content
        if message.get("content"):
            content.append({"type": "text", "text": message["content"]})

        # Handle tool calls
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                # Safely parse arguments - handle malformed JSON from incomplete streams
                args_str = tc["function"].get("arguments", "{}")
                try:
                    args_input = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    # If arguments are malformed (e.g., incomplete stream), use raw string
                    args_input = {"_raw_arguments": args_str}
                content.append({
                    "type": "tool_use",
                    "id": tc.get("id"),
                    "name": tc["function"]["name"],
                    "input": args_input,
                })

        if choice.get("finish_reason"):
            stop_reason = choice["finish_reason"]

    usage = response.get("usage", {})
    cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)

    return {
        "id": response.get("id", str(uuid.uuid4())),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": response.get("model", ""),
        "stop_reason": map_openai_stop_reason_to_anthropic(stop_reason),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0) - cached_tokens,
            "output_tokens": usage.get("completion_tokens", 0),
            **({"cache_read_input_tokens": cached_tokens} if cached_tokens else {}),
        },
    }


# ============================================================================
# Streaming Translation
# ============================================================================

class AnthropicStreamState:
    """State for translating streaming responses to Anthropic format"""
    def __init__(self):
        self.message_start_sent = False
        self.content_block_index = 0
        self.content_block_open = False
        self.tool_calls: Dict[int, Dict] = {}


def translate_chunk_to_anthropic_events(chunk: Dict, stream_state: AnthropicStreamState) -> List[Dict]:
    """Translate OpenAI streaming chunk to Anthropic events"""
    events = []

    if not chunk.get("choices"):
        return events

    choice = chunk["choices"][0]
    delta = choice.get("delta", {})

    # Send message_start if not sent yet
    if not stream_state.message_start_sent:
        usage = chunk.get("usage", {})
        cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)

        events.append({
            "type": "message_start",
            "message": {
                "id": chunk.get("id", str(uuid.uuid4())),
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": chunk.get("model", ""),
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": usage.get("prompt_tokens", 0) - cached_tokens,
                    "output_tokens": 0,
                    **({"cache_read_input_tokens": cached_tokens} if cached_tokens else {}),
                },
            },
        })
        stream_state.message_start_sent = True

    # Handle text content
    if delta.get("content"):
        # Check if we need to close a tool block
        if stream_state.content_block_open and any(
            tc.get("anthropic_block_index") == stream_state.content_block_index
            for tc in stream_state.tool_calls.values()
        ):
            events.append({
                "type": "content_block_stop",
                "index": stream_state.content_block_index,
            })
            stream_state.content_block_index += 1
            stream_state.content_block_open = False

        if not stream_state.content_block_open:
            events.append({
                "type": "content_block_start",
                "index": stream_state.content_block_index,
                "content_block": {"type": "text", "text": ""},
            })
            stream_state.content_block_open = True

        events.append({
            "type": "content_block_delta",
            "index": stream_state.content_block_index,
            "delta": {"type": "text_delta", "text": delta["content"]},
        })

    # Handle tool calls
    if delta.get("tool_calls"):
        for tc in delta["tool_calls"]:
            tc_index = tc.get("index", 0)

            if tc.get("id") and tc.get("function", {}).get("name"):
                # New tool call starting
                if stream_state.content_block_open:
                    events.append({
                        "type": "content_block_stop",
                        "index": stream_state.content_block_index,
                    })
                    stream_state.content_block_index += 1
                    stream_state.content_block_open = False

                block_index = stream_state.content_block_index
                stream_state.tool_calls[tc_index] = {
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "anthropic_block_index": block_index,
                }

                events.append({
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": {},
                    },
                })
                stream_state.content_block_open = True

            if tc.get("function", {}).get("arguments"):
                tc_info = stream_state.tool_calls.get(tc_index)
                if tc_info:
                    events.append({
                        "type": "content_block_delta",
                        "index": tc_info["anthropic_block_index"],
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": tc["function"]["arguments"],
                        },
                    })

    # Handle finish
    if choice.get("finish_reason"):
        if stream_state.content_block_open:
            events.append({
                "type": "content_block_stop",
                "index": stream_state.content_block_index,
            })
            stream_state.content_block_open = False

        usage = chunk.get("usage", {})
        cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)

        events.append({
            "type": "message_delta",
            "delta": {
                "stop_reason": map_openai_stop_reason_to_anthropic(choice["finish_reason"]),
                "stop_sequence": None,
            },
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0) - cached_tokens,
                "output_tokens": usage.get("completion_tokens", 0),
                **({"cache_read_input_tokens": cached_tokens} if cached_tokens else {}),
            },
        })
        events.append({"type": "message_stop"})

    return events


# ============================================================================
# API Routes - OpenAI Compatible
# ============================================================================

@app.route("/", methods=["GET"])
def index():
    """Serve the dashboard"""
    return render_template("dashboard.html")


@app.route("/requests", methods=["GET"])
def requests_page():
    """Serve the requests browser page"""
    return render_template("requests.html")


@app.route("/v1/models", methods=["GET"])
@app.route("/models", methods=["GET"])
def list_models():
    """List available models"""
    try:
        ensure_copilot_token()
        if not state.models:
            fetch_models()

        models = [
            {
                "id": m["id"],
                "object": "model",
                "type": "model",
                "created": 0,
                "created_at": datetime.utcfromtimestamp(0).isoformat() + "Z",
                "owned_by": m.get("vendor", "unknown"),
                "display_name": m.get("name", m["id"]),
            }
            for m in state.models.get("data", [])
        ]

        return jsonify({
            "object": "list",
            "data": models,
            "has_more": False,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/v1/chat/completions", methods=["POST"])
@app.route("/chat/completions", methods=["POST"])
def chat_completions():
    """Handle chat completions (OpenAI format)"""
    try:
        start_time = time.time()
        ensure_copilot_token()
        payload = request.get_json()
        request_id = str(uuid.uuid4())

        # Check for vision content
        enable_vision = False
        for msg in payload.get("messages", []):
            content = msg.get("content")
            if isinstance(content, list):
                if any(p.get("type") == "image_url" for p in content):
                    enable_vision = True
                    break

        # Detect agent vs user call
        is_agent_call = any(
            msg.get("role") in ("assistant", "tool")
            for msg in payload.get("messages", [])
        )

        headers = get_copilot_headers(enable_vision)
        headers["X-Initiator"] = "agent" if is_agent_call else "user"

        request_body = json.dumps(payload)
        request_size = len(request_body)

        if payload.get("stream"):
            return stream_chat_completions(payload, headers, request_id, request_body, request_size, start_time)

        # Non-streaming request
        response = requests.post(
            f"{get_copilot_base_url()}/chat/completions",
            headers=headers,
            json=payload,
            timeout=1200,
        )

        duration = round(time.time() - start_time, 2)
        response_body = response.text
        response_size = len(response_body)

        if response.ok:
            result = response.json()

            # Cache the request/response
            usage = result.get("usage", {})
            cache.add_request(request_id, {
                "request_body": payload,
                "response_body": result,
                "model": payload.get("model", "unknown"),
                "endpoint": "/v1/chat/completions",
                "status_code": response.status_code,
                "request_size": request_size,
                "response_size": response_size,
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "duration": duration,
            })

            return jsonify(result)
        else:
            return Response(response.text, status=response.status_code, mimetype="application/json")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def reconstruct_openai_response_from_chunks(chunks: List[Dict]) -> Dict:
    """Reconstruct a complete OpenAI response from streaming chunks"""
    if not chunks:
        return {}

    # Get metadata from first chunk
    first_chunk = chunks[0]
    response_id = first_chunk.get("id", "")
    model = first_chunk.get("model", "")
    created = first_chunk.get("created", 0)

    # Accumulate content and tool calls
    content_parts = []
    tool_calls: Dict[int, Dict] = {}
    finish_reason = None
    usage = {}

    for chunk in chunks:
        if chunk.get("usage"):
            usage = chunk["usage"]

        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})

            if delta.get("content"):
                content_parts.append(delta["content"])

            if delta.get("tool_calls"):
                for tc in delta["tool_calls"]:
                    tc_index = tc.get("index", 0)
                    if tc_index not in tool_calls:
                        tool_calls[tc_index] = {
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc.get("id"):
                        tool_calls[tc_index]["id"] = tc["id"]
                    if tc.get("function", {}).get("name"):
                        tool_calls[tc_index]["function"]["name"] = tc["function"]["name"]
                    if tc.get("function", {}).get("arguments"):
                        tool_calls[tc_index]["function"]["arguments"] += tc["function"]["arguments"]

            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    # Build the reconstructed response
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts) if content_parts else None,
    }

    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls.keys())]

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": usage,
    }


def stream_chat_completions(payload: Dict, headers: Dict, request_id: str,
                            request_body: str, request_size: int, start_time: float) -> Response:
    """Handle streaming chat completions"""
    # Start tracking request immediately
    cache.start_request(request_id, {
        "request_body": payload,
        "model": payload.get("model", "unknown"),
        "endpoint": "/v1/chat/completions",
        "request_size": request_size,
    })

    def generate() -> Generator[str, None, None]:
        response_chunks = []
        total_output_tokens = 0
        total_input_tokens = 0
        error_occurred = False
        status_code = 200
        first_chunk_received = False

        try:
            # Update state to sending
            cache.update_request_state(request_id, cache.STATE_SENDING)

            response = requests.post(
                f"{get_copilot_base_url()}/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=1200,
            )
            status_code = response.status_code

            for line in response.iter_lines():
                if not line:
                    continue

                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break

                    try:
                        chunk = json.loads(data)
                        response_chunks.append(chunk)

                        # Update state to receiving on first chunk
                        if not first_chunk_received:
                            first_chunk_received = True
                            cache.update_request_state(request_id, cache.STATE_RECEIVING)

                        # Track tokens from streaming chunks
                        if chunk.get("usage"):
                            total_output_tokens = chunk["usage"].get("completion_tokens", 0)
                            total_input_tokens = chunk["usage"].get("prompt_tokens", 0)

                        yield f"data: {data}\n\n"
                    except json.JSONDecodeError:
                        continue
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            # Timeout or connection error from upstream - log but don't try to yield after client disconnect
            error_occurred = True
            status_code = 504
            print(f"[Stream] Upstream timeout/connection error for request {request_id}: {type(e).__name__}")
        except GeneratorExit:
            # Client disconnected - this clean up
            error_occurred = True
            print(f"[Stream] Client disconnected for request {request_id}")
            # Update state to error since client disconnected
            cache.update_request_state(request_id, cache.STATE_ERROR, status_code=499)
            return
        except Exception as e:
            error_occurred = True
            status_code = 500
            print(f"[Stream] Error for request {request_id}: {type(e).__name__}: {e}")
            try:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            except GeneratorExit:
                # Client already disconnected, can't yield
                cache.update_request_state(request_id, cache.STATE_ERROR, status_code=499)
                return

        duration = round(time.time() - start_time, 2)

        # Reconstruct the full response from chunks
        reconstructed_response = reconstruct_openai_response_from_chunks(response_chunks)
        if error_occurred and not reconstructed_response:
            reconstructed_response = {"error": "Stream interrupted"}

        # Complete the request in cache
        cache.complete_request(request_id, {
            "request_body": payload,
            "response_body": reconstructed_response,
            "model": payload.get("model", "unknown"),
            "endpoint": "/v1/chat/completions",
            "status_code": status_code,
            "request_size": request_size,
            "response_size": sum(len(json.dumps(c)) for c in response_chunks),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "duration": duration,
        })

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================================
# API Routes - Anthropic Compatible
# ============================================================================

@app.route("/v1/messages", methods=["POST"])
def anthropic_messages():
    """Handle Anthropic messages API"""
    if True:
        start_time = time.time()
        ensure_copilot_token()
        anthropic_payload = request.get_json()
        request_id = str(uuid.uuid4())

        # Translate to OpenAI format
        openai_payload = translate_anthropic_to_openai(anthropic_payload)

        # Check for vision content
        enable_vision = any(
            isinstance(msg.get("content"), list) and
            any(p.get("type") == "image" for p in msg.get("content", []))
            for msg in anthropic_payload.get("messages", [])
        )

        is_agent_call = any(
            msg.get("role") in ("assistant", "tool")
            for msg in openai_payload.get("messages", [])
        )

        headers = get_copilot_headers(enable_vision)
        headers["X-Initiator"] = "agent" if is_agent_call else "user"

        request_size = len(json.dumps(anthropic_payload))

        if anthropic_payload.get("stream"):
            return stream_anthropic_messages(openai_payload, headers, request_id,
                                             anthropic_payload, request_size, start_time)

        # Non-streaming request
        response = requests.post(
            f"{get_copilot_base_url()}/chat/completions",
            headers=headers,
            json=openai_payload,
            timeout=1200,
        )

        duration = round(time.time() - start_time, 2)

        if response.ok:
            openai_response = response.json()
            anthropic_response = translate_openai_to_anthropic(openai_response)

            # Cache the request/response
            usage = openai_response.get("usage", {})
            cache.add_request(request_id, {
                "request_body": anthropic_payload,
                "response_body": anthropic_response,
                "model": anthropic_payload.get("model", "unknown"),
                "endpoint": "/v1/messages",
                "status_code": response.status_code,
                "request_size": request_size,
                "response_size": len(json.dumps(anthropic_response)),
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "duration": duration,
            })

            return jsonify(anthropic_response)
        else:
            return Response(response.text, status=response.status_code, mimetype="application/json")

    # except Exception as e:
    #     return jsonify({"error": {"type": "api_error", "message": str(e)}}), 500


def stream_anthropic_messages(openai_payload: Dict, headers: Dict, request_id: str,
                              anthropic_payload: Dict, request_size: int, start_time: float) -> Response:
    """Handle streaming Anthropic messages"""
    # Start tracking request immediately
    cache.start_request(request_id, {
        "request_body": anthropic_payload,
        "model": anthropic_payload.get("model", "unknown"),
        "endpoint": "/v1/messages",
        "request_size": request_size,
    })

    def generate() -> Generator[str, None, None]:
        stream_state = AnthropicStreamState()
        response_chunks = []
        total_output_tokens = 0
        total_input_tokens = 0
        error_occurred = False
        status_code = 200
        first_chunk_received = False

        try:
            # Update state to sending
            cache.update_request_state(request_id, cache.STATE_SENDING)

            response = requests.post(
                f"{get_copilot_base_url()}/chat/completions",
                headers=headers,
                json=openai_payload,
                stream=True,
                timeout=1200,
            )
            status_code = response.status_code

            for line in response.iter_lines():
                if not line:
                    continue

                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        response_chunks.append(chunk)

                        # Update state to receiving on first chunk
                        if not first_chunk_received:
                            first_chunk_received = True
                            cache.update_request_state(request_id, cache.STATE_RECEIVING)

                        if chunk.get("usage"):
                            total_output_tokens = chunk["usage"].get("completion_tokens", 0)
                            total_input_tokens = chunk["usage"].get("prompt_tokens", 0)

                        # Translate to Anthropic events
                        events = translate_chunk_to_anthropic_events(chunk, stream_state)
                        for event in events:
                            yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                    except json.JSONDecodeError:
                        continue
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            # Timeout or connection error from upstream - log but don't try to yield after client disconnect
            error_occurred = True
            status_code = 504
            print(f"[Stream Anthropic] Upstream timeout/connection error for request {request_id}: {type(e).__name__}")
        except GeneratorExit:
            # Client disconnected - this is normal, just clean up
            error_occurred = True
            print(f"[Stream Anthropic] Client disconnected for request {request_id}")
            # Update state to error since client disconnected
            cache.update_request_state(request_id, cache.STATE_ERROR, status_code=499)
            return
        except Exception as e:
            error_occurred = True
            status_code = 500
            print(f"[Stream Anthropic] Error for request {request_id}: {type(e).__name__}: {e}")
            try:
                yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': str(e)}})}\n\n"
            except GeneratorExit:
                # Client already disconnected, can't yield
                cache.update_request_state(request_id, cache.STATE_ERROR, status_code=499)
                return

        duration = round(time.time() - start_time, 2)

        # Reconstruct the OpenAI response then translate to Anthropic format
        reconstructed_openai = reconstruct_openai_response_from_chunks(response_chunks)
        anthropic_response = translate_openai_to_anthropic(reconstructed_openai) if reconstructed_openai else {}
        if error_occurred and not anthropic_response:
            anthropic_response = {"error": {"type": "api_error", "message": "Stream interrupted"}}

        # Complete the request in cache
        cache.complete_request(request_id, {
            "request_body": anthropic_payload,
            "response_body": anthropic_response,
            "model": anthropic_payload.get("model", "unknown"),
            "endpoint": "/v1/messages",
            "status_code": status_code,
            "request_size": request_size,
            "response_size": sum(len(json.dumps(c)) for c in response_chunks),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "duration": duration,
        })

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================================
# Dashboard API Routes
# ============================================================================

@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Get API statistics"""
    return jsonify(cache.get_stats())


@app.route("/api/requests", methods=["GET"])
def api_requests():
    """Get paginated list of requests"""
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)
    search = request.args.get("search", "")

    offset = (page - 1) * per_page

    if search:
        items = cache.search_requests(search, per_page, offset)
        total = len(cache.search_requests(search, 10000, 0))  # Get total count
    else:
        items = cache.get_recent_requests(per_page, offset)
        total = cache.get_total_count()

    # Remove large body content for list view
    items_summary = []
    for item in items:
        summary = dict(item)
        summary["request_body"] = None  # Remove for list view
        summary["response_body"] = None  # Remove for list view
        items_summary.append(summary)

    return jsonify({
        "items": items_summary,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
    })


@app.route("/api/request/<request_id>", methods=["GET"])
def api_request_detail(request_id: str):
    """Get detailed request/response data"""
    item = cache.get_request(request_id)
    if not item:
        return jsonify({"error": "Request not found"}), 404
    return jsonify(item)


@app.route("/api/request/<request_id>/request-body", methods=["GET"])
def api_request_body(request_id: str):
    """Get just the request body"""
    item = cache.get_request(request_id)
    if not item:
        return jsonify({"error": "Request not found"}), 404
    return jsonify(item.get("request_body"))


@app.route("/api/request/<request_id>/response-body", methods=["GET"])
def api_response_body(request_id: str):
    """Get just the response body"""
    item = cache.get_request(request_id)
    if not item:
        return jsonify({"error": "Request not found"}), 404
    return jsonify(item.get("response_body"))


@app.route("/api/requests/search", methods=["GET"])
def api_fulltext_search():
    """Full-text search in request/response bodies"""
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)
    query = request.args.get("q", "")

    if not query:
        return jsonify({
            "items": [],
            "total": 0,
            "page": page,
            "per_page": per_page,
            "total_pages": 0,
        })

    offset = (page - 1) * per_page
    items, total = cache.fulltext_search(query, per_page, offset)

    # Remove large body content for list view
    items_summary = []
    for item in items:
        summary = dict(item)
        summary["request_body"] = None
        summary["response_body"] = None
        items_summary.append(summary)

    return jsonify({
        "items": items_summary,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page if total > 0 else 0,
    })


@app.route("/api/requests/export", methods=["GET"])
def api_export_requests():
    """Export all requests as JSON Lines (.jl) file"""
    def generate():
        for item in cache.get_all_requests():
            yield json.dumps(item, ensure_ascii=False) + "\n"

    return Response(
        generate(),
        mimetype="application/x-jsonlines",
        headers={
            "Content-Disposition": f"attachment; filename=requests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jl",
        },
    )


@app.route("/api/requests/import", methods=["POST"])
def api_import_requests():
    """Import requests from JSON Lines (.jl) file"""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    imported_count = 0
    errors = []

    try:
        for line_num, line in enumerate(file.stream, 1):
            line = line.decode("utf-8").strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                cache.import_request(data)
                imported_count += 1
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {str(e)}")
            except Exception as e:
                errors.append(f"Line {line_num}: {str(e)}")

    except Exception as e:
        return jsonify({"error": f"Failed to read file: {str(e)}"}), 500

    return jsonify({
        "imported": imported_count,
        "errors": errors[:10] if errors else [],  # Limit errors to first 10
        "total_errors": len(errors),
    })


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ============================================================================
# Application Startup
# ============================================================================

def initialize():
    """Initialize the application"""
    # Get GitHub token using the new token management system
    token = get_github_token()
    if not token:
        print("\n" + "=" * 60)
        print("ERROR: No GitHub token available!")
        print("=" * 60)
        print("Options to provide a GitHub token:")
        print("  1. Set GITHUB_TOKEN environment variable")
        print("  2. Create a github_token.txt file in the config directory")
        print("  3. Run the app again to use interactive Device Flow authentication")
        print("=" * 60)
        return

    # Update the state with the token
    state.github_token = token

    try:
        refresh_copilot_token()
        fetch_models()
        print("Application initialized successfully")
    except Exception as e:
        print(f"Failed to initialize: {e}")


def generate_config_file():
    """Generate a YAML config file in the config directory"""
    config_dir = get_config_dir()
    os.makedirs(config_dir, exist_ok=True)
    
    config_path = os.path.join(config_dir, "config.yaml")
    
    config_data = {
        "address": "localhost",
        "port": 8313,
        "account_type": "individual"
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False)
    
    print(f"Configuration file generated at: {config_path}")
    
    # On Windows, try to open with notepad
    if platform.system() == "Windows":
        try:
            os.system(f"notepad {config_path}")
        except Exception:
            pass


def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description='GitHub Copilot API Proxy')
    parser.add_argument('-p', '--port', type=int, help='Port to listen on', default=8313)
    parser.add_argument('-a', '--address', type=str, help='Address to listen on', default='localhost')
    parser.add_argument('-c', '--config', action='store_true', help='Generate a YAML config file')

    args = parser.parse_args()

    if args.config:
        generate_config_file()
        return

    # Load config from file if it exists
    config_dir = get_config_dir()
    config_path = os.path.join(config_dir, "config.yaml")

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                import yaml
                config = yaml.safe_load(f)

                # Override defaults with config values
                if args.address is not None:
                    HOST = args.address
                else:
                    HOST = config.get('address', 'localhost')

                if args.port is not None:
                    PORT = args.port
                else:
                    PORT = config.get('port', 8313)

                # Set account type from config
                state.account_type = config.get('account_type', 'individual')
        except Exception as e:
            print(f"Error loading config file: {e}")
            HOST = args.address or 'localhost'
            PORT = args.port or 8313
    else:
        # Use command line args or defaults
        HOST = args.address or 'localhost'
        PORT = args.port or 8313

    # Initialize the app context
    with app.app_context():
        initialize()

    print(f"Starting GitHub Copilot API Proxy on {HOST}:{PORT}")
    print(f"Dashboard: http://{HOST}:{PORT}/")
    print(f"OpenAI API: http://{HOST}:{PORT}/v1/chat/completions")
    print(f"Anthropic API: http://{HOST}:{PORT}/v1/messages")

    app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True)


if __name__ == "__main__":
    main()