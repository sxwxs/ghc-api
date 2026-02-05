"""
Global application state management
"""

import threading
from typing import Dict, Optional

from .config import GITHUB_TOKEN, VSCODE_VERSION


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


# Global state instance
state = State()
