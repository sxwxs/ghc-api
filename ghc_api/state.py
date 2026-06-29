"""
Global application state management
"""

import threading
from typing import Dict, List, Optional

from .config import (
    GITHUB_TOKEN,
    DEFAULT_VSCODE_VERSION,
    DEFAULT_COPILOT_VERSION,
    DEFAULT_API_VERSION,
)


class State:
    """Global application state"""
    def __init__(self):
        self.github_token: str = GITHUB_TOKEN
        self.copilot_token: Optional[str] = None
        self.models: Optional[Dict] = None
        self.account_type: str = "individual"
        self.token_expires_at: float = 0
        self.token_lock = threading.Lock()

        # Configurable version settings (can be overridden by config file)
        self.vscode_version: str = DEFAULT_VSCODE_VERSION
        self.copilot_version: str = DEFAULT_COPILOT_VERSION
        self.api_version: str = DEFAULT_API_VERSION

        # Content filtering settings
        self.system_prompt_remove: List[str] = []
        self.tool_result_suffix_remove: List[str] = []
        self.system_prompt_add: List[str] = []

        # Direct Anthropic API settings
        self.redirect_anthropic: bool = False  # Force Anthropic through OpenAI translation
        # Recover tool calls that Copilot intermittently leaks as plain text on the
        # direct Anthropic streaming path. Disabled by default; when off the stream is
        # forwarded untouched (see ghc_api/tool_call_recovery.py).
        self.enable_tool_call_recovery: bool = False

        # Retry settings
        self.max_connection_retries: int = 3  # Max retries for upstream connection errors

        # Upstream request timeout in seconds, passed to requests as a single
        # value so it applies to both the connect and read phases.
        self.upstream_read_timeout: int = 1800

        # SSE keepalive: when a stream is idle this many seconds, emit a keepalive
        # ping to the client so its read timeout does not fire. 0 disables.
        self.sse_keepalive_interval: int = 30
        self.auto_remove_encrypted_content_on_parse_error: bool = False
        self.save_request_to_file: bool = False
        self.disable_onedrive_access: bool = True

        # Web search proxy settings
        self.enable_web_search_proxy: bool = False
        self.web_search_proxy_endpoint: str = ""

        # User authentication settings
        # When True, /v1/chat/completions, /v1/messages, /v1/responses, /v1/models
        # require an approved token from the user registry (users.json).
        # When False (default), all requests are tagged with user_id="anonymous"
        # and no auth check is performed.
        self.enable_auth: bool = False

        # Session persistence settings
        self.session_flush_interval: int = 5  # seconds between buffered writes

        # Request cache memory limits
        self.cache_max_entries: int = 1000  # Max number of requests kept in memory
        self.cache_max_request_size: int = 1024 * 1024  # Max bytes per cached entry body (0 disables limit)

        # Background worker guards
        self.token_usage_reporter_started: bool = False

    @property
    def editor_plugin_version(self) -> str:
        """Get the editor plugin version string"""
        return f"copilot-chat/{self.copilot_version}"

    @property
    def user_agent(self) -> str:
        """Get the user agent string"""
        return f"GitHubCopilotChat/{self.copilot_version}"


# Global state instance
state = State()
