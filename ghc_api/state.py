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
        self.github_token_source: str = "environment" if GITHUB_TOKEN else "unconfigured"
        self.copilot_token: Optional[str] = None
        self.models: Optional[Dict] = None
        self.account_type: str = "individual"
        # Optional upstream URL overrides. Empty values preserve the normal
        # GitHub/Copilot account-type routing; benchmarks use loopback URLs.
        self.github_api_base_url: str = ""
        self.copilot_api_base_url: str = ""
        self.token_expires_at: float = 0
        self.token_lock = threading.Lock()
        self.token_refresh_last_attempt_at: Optional[float] = None
        self.token_refresh_last_success_at: Optional[float] = None
        self.token_refresh_last_succeeded: Optional[bool] = None
        self.token_refresh_last_error: Optional[str] = None

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

        # Copilot intermittently answers /v1/responses with HTTP 200 whose SSE body is
        # just response.created followed by response.failed, before any model output.
        # When enabled, such a stream is transparently retried (up to
        # max_connection_retries) as long as nothing has been forwarded to the client
        # yet. Enabled by default; it can be disabled to hand the upstream response
        # object to the stream handler untouched (see ghc_api/sse/openai_responses.py).
        self.enable_responses_early_failure_retry: bool = True

        # Retry settings
        self.max_connection_retries: int = 3  # Max retries for upstream connection errors

        # Upstream request timeout in seconds, passed to requests as a single
        # value so it applies to both the connect and read phases.
        self.upstream_read_timeout: int = 1800

        # SSE keepalive: when a stream is idle this many seconds, emit a keepalive
        # ping to the client so its read timeout does not fire. 0 disables.
        self.sse_keepalive_interval: int = 30
        # When /v1/responses rejects a request because encrypted reasoning or tool
        # output content cannot be decrypted, clean the input and retry once instead
        # of surfacing the 400. Lossy by design (see remove_encrypted_content_items),
        # so it is opt-in.
        self.auto_remove_encrypted_content_on_parse_error: bool = False
        self.save_request_to_file: bool = False
        self.disable_onedrive_access: bool = True

        # Web search proxy settings
        self.enable_web_search_proxy: bool = False
        self.web_search_proxy_endpoint: str = ""

        # Microsoft Web IQ search settings. The API key may be loaded directly
        # from config.yaml; it is never exposed to browser clients.
        self.enable_webiq_search: bool = False
        self.webiq_api_key: str = ""
        self.webiq_endpoint: str = "https://api.microsoftol.com/v3/search/web"
        self.webiq_max_results: int = 5
        self.webiq_max_length: int = 3000
        self.webiq_content_format: str = "passage"
        self.webiq_language: str = "en"
        self.webiq_region: str = "US"
        self.webiq_safe_search: str = "strict"
        self.webiq_timeout: int = 30
        # Record every /v1/webiq/search request and response. Entries go to a
        # daily .jl file under <config_dir>/webiq/ (on by default, unlike the
        # much larger LLM request dumps) and to an in-memory ring buffer of the
        # most recent webiq_log_max_entries searches for the dashboard.
        self.log_webiq_requests: bool = True
        self.webiq_log_max_entries: int = 20

        # User authentication settings
        # When True, /v1/chat/completions, /v1/messages, /v1/responses,
        # /v1/embeddings, and /v1/models require an approved user token.
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
