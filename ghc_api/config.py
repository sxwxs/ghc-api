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
COPILOT_API_BASE_URL = "https://api.githubcopilot.com"

COPILOT_VERSION = "0.26.7"
EDITOR_PLUGIN_VERSION = f"copilot-chat/{COPILOT_VERSION}"
USER_AGENT = f"GitHubCopilotChat/{COPILOT_VERSION}"
API_VERSION = "2025-04-01"
VSCODE_VERSION = "1.93.0"

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
        self.prefix_mappings = model_mappings.get("prefix", {})


# Global model mappings instance
model_mappings = ModelMappings()
