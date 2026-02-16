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

        # Retry settings
        self.max_connection_retries: int = 3  # Max retries for upstream connection errors

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
