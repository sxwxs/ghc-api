import os
import platform

from .utils import get_config_dir
from .config import DEFAULT_COPILOT_VERSION, DEFAULT_API_VERSION, DEFAULT_VSCODE_VERSION

def generate_config_file():
    """Generate a YAML config file in the config directory with comments"""
    config_dir = get_config_dir()
    os.makedirs(config_dir, exist_ok=True)

    config_path = os.path.join(config_dir, "config.yaml")

    # Generate config file with comments using manual string construction
    config_content = """# GitHub Copilot API Proxy Configuration
# ========================================

# Server Settings
# ---------------

address: localhost
port: 8313
debug: false

# GitHub Copilot Account Type
# Options:
#   - "individual": For personal GitHub Copilot subscriptions
#                   Uses: https://api.githubcopilot.com
#   - "business": For GitHub Copilot Business accounts (organization-managed)
#                 Uses: https://api.business.githubcopilot.com
#   - "enterprise": For GitHub Copilot Enterprise accounts (enterprise-managed)
#                   Uses: https://api.enterprise.githubcopilot.com
account_type: individual

# VSCode Version (only used to build request headers)
vscode_version: "{vscode_version}"
# GitHub API version to use (only used to build request headers)
api_version: "{api_version}"
# Copilot extension version to emulate (only used to build request headers)
copilot_version: "{copilot_version}"
# Model Name Mappings
# -------------------
# Translate incoming model names to different names for the Copilot API.
# Useful for compatibility with different client libraries or to map
# specific model versions to generic model names.
#
# Two types of mappings are supported:
#
# 1. exact: Full model name must match exactly
#    Example: "gpt-4-turbo-preview" -> "gpt-4-turbo"
#
# 2. prefix: Model name starts with the specified prefix
#    Example: "claude-sonnet-4-" matches "claude-sonnet-4-20250514"
#
model_mappings:
  # Exact match mappings (full model name -> target model name)
  exact:
    # Example:
    opus: claude-opus-4.5
    sonnet: claude-sonnet-4.5
    haiku: claude-haiku-4.5

  # Prefix match mappings (model name prefix -> target model name)
  # If a model name starts with the key, it will be replaced with the value
  prefix:
    claude-sonnet-4-: claude-sonnet-4
    claude-haiku-4.5-: claude-haiku-4.5
    claude-opus-4.5-: claude-opus-4.5
    claude-haiku-4-5-: claude-haiku-4.5
    claude-opus-4-5-: claude-opus-4.5
    
""".format(
        vscode_version=DEFAULT_VSCODE_VERSION,
        api_version=DEFAULT_API_VERSION,
        copilot_version=DEFAULT_COPILOT_VERSION
    )
    if os.path.exists(config_path):
        print(f"Configuration file already exists at: {config_path}")
    else:
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)

    print(f"Configuration file generated at: {config_path}")

    # On Windows, try to open with notepad
    if platform.system() == "Windows":
        try:
            os.system(f"notepad {config_path}")
        except Exception:
            pass
