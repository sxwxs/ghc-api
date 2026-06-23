"""
Configuration constants for GitHub Copilot API Proxy
"""

import os
from typing import Dict, List, Any, Optional

# Environment variables
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
PORT = int(os.environ.get("PORT", 8313))
HOST = os.environ.get("HOST", "localhost")
DEBUG = os.environ.get("DEBUG", "true").lower() == "true"

# API Configuration
GITHUB_API_BASE_URL = "https://api.github.com"
# COPILOT_API_BASE_URL = "https://api.githubcopilot.com"

# Default values (can be overridden by config file)
DEFAULT_COPILOT_VERSION = "0.26.7"
DEFAULT_API_VERSION = "2025-04-01"
DEFAULT_VSCODE_VERSION = "1.93.0"

# Default model mappings (will be replaced if config file includes model_mappings)
DEFAULT_MODEL_MAPPINGS = {
    "exact": {
        # Add exact model name mappings here
        'opus': 'claude-opus-4.5',
        'sonnet': 'claude-sonnet-4.5',
        'haiku': 'claude-haiku-4.5'
    },
    "prefix": {
        # Prefix-based mappings: if model name starts with the key, replace with value
        "claude-sonnet-4-": "claude-sonnet-4",
        "claude-opus-4-": "claude-opus-4",
        "claude-opus-4.5-": "claude-opus-4.5",
        "claude-haiku-4.5-": "claude-haiku-4.5",
    }
}

# Models that should be advertised as supporting the OpenAI-compatible
# /v1/chat/completions endpoint even when Copilot's model metadata omits it.
DEFAULT_CHAT_COMPLETIONS_MODEL_SUPPORT = {
    "exact": [],
    "prefix": [
        "gpt-",
        "mai-code-",
    ],
}

# GitHub OAuth App for Device Flow (using GitHub CLI's client ID as it's public)
GITHUB_OAUTH_CLIENT_ID = "01ab8ac9400c4e429b23"


# Model name mappings (loaded from config file)
class ModelMappings:
    """Stores model name translation mappings"""
    def __init__(self):
        self.exact_mappings: Dict[str, str] = {}
        self.prefix_mappings: Dict[str, str] = {}

    def translate(self, model: str) -> str:
        """Translate model name using exact match first, then prefix match"""
        # Try exact match first
        if model in self.exact_mappings:
            return self.exact_mappings[model]

        # Try prefix match
        for prefix, target in self.prefix_mappings.items():
            if model.startswith(prefix):
                return target

        # No match, return original
        return model

    def load_from_config(self, config: Dict[str, Any]) -> None:
        """Load mappings from config dictionary"""
        model_mappings = config.get("model_mappings", {})
        self.exact_mappings = model_mappings.get("exact", {})
        if self.exact_mappings is None:
            self.exact_mappings = {}
        self.prefix_mappings = model_mappings.get("prefix", {})
        if self.prefix_mappings is None:
            self.prefix_mappings = {}


# Global model mappings instance
model_mappings = ModelMappings()


class ChatCompletionsModelSupport:
    """Stores model names that should advertise chat completions support."""

    def __init__(self):
        self.exact_model_names: List[str] = []
        self.prefix_model_names: List[str] = []

    @staticmethod
    def _string_list(value: Any) -> List[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str) and item]

    def matches(self, model: str) -> bool:
        """Return True when a model id matches exact or prefix config."""
        if not isinstance(model, str):
            return False

        normalized_model = model.lower()
        exact_model_names = {name.lower() for name in self.exact_model_names}
        prefix_model_names = [prefix.lower() for prefix in self.prefix_model_names]

        if normalized_model in exact_model_names:
            return True

        return any(normalized_model.startswith(prefix) for prefix in prefix_model_names)

    def load_from_config(self, config: Dict[str, Any]) -> None:
        """Load endpoint support overrides from config dictionary."""
        endpoint_support = config.get("chat_completions_model_support", {})
        if not isinstance(endpoint_support, dict):
            endpoint_support = {}
        self.exact_model_names = self._string_list(endpoint_support.get("exact"))
        self.prefix_model_names = self._string_list(endpoint_support.get("prefix"))


# Global chat completions endpoint support instance
chat_completions_model_support = ChatCompletionsModelSupport()
