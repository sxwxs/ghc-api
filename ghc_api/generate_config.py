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
    opus: claude-opus-4.6
    sonnet: claude-sonnet-4.5
    haiku: claude-haiku-4.5
    claude-opus-4-6: claude-opus-4.6
    claude-opus-4-5: claude-opus-4.5
    claude-haiku-4-5: claude-haiku-4.5


  # Prefix match mappings (model name prefix -> target model name)
  # If a model name starts with the key, it will be replaced with the value
  prefix:
    claude-sonnet-4-: claude-sonnet-4
    claude-haiku-4.5-: claude-haiku-4.5
    claude-opus-4.5-: claude-opus-4.5
    claude-haiku-4-5-: claude-haiku-4.5
    claude-opus-4-5-: claude-opus-4.5
    claude-opus-4-6-: claude-opus-4.6
    claude-opus-4.6-: claude-opus-4.6


# Content Filtering
# -----------------
# Remove or add content from system prompts and tool results.
#
# system_prompt_remove: List of strings to remove from system prompts
# Each string in the list will be removed from the system prompt text.
#
# tool_result_suffix_remove: List of strings to remove from tool results
# These strings are only removed if they appear at the END of the tool result.
# Useful for removing trailing tags or markers from tool outputs.
#
# system_prompt_add: List of strings to append to the system prompt
# Each string will be added as a new paragraph at the end of the system prompt.
#
# Example:
# system_prompt_remove:
#   - "IMPORTANT: Assist with authorized security testing, defensive security, CTF challenges, and educational contexts. Refuse requests for destructive techniques, DoS attacks, mass targeting, supply chain compromise, or detection evasion for malicious purposes. Dual-use security tools (C2 frameworks, credential testing, exploit development) require clear authorization context: pentesting engagements, CTF competitions, security research, or defensive use cases."
#
# tool_result_suffix_remove:
#   - "<system-reminder>\\nWhenever you read a file, you should consider whether it would be considered malware. You CAN and SHOULD provide analysis of malware, what it is doing. But you MUST refuse to improve or augment the code. You can still analyze existing code, write reports, or answer questions about the code behavior.\\n</system-reminder>\\n"
#
# system_prompt_add:
#   - "Additional instruction for the model."
#   - "Another instruction to append."

system_prompt_remove: []
tool_result_suffix_remove: []
system_prompt_add: []

# Retry Settings
# ---------------
# Maximum number of retries for upstream connection errors (e.g., RemoteDisconnected,
# ReadTimeout). Set to 0 to disable retries.
max_connection_retries: 3

# If true, when /v1/responses gets HTTP 400 with message that starts with
# "The encrypted content" and ends with
# "Reason: Encrypted content could not be decrypted or parsed.",
# all items containing "encrypted_content" in request.input are removed and retried once.
auto_remove_encrypted_content_on_parse_error: false

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
