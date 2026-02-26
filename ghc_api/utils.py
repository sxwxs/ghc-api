from datetime import datetime
import json
import os
import platform
from typing import Dict, List


from .config import model_mappings
from .state import state


def log_error_request(endpoint: str, request_body: dict, response_body: str, status_code: int):
    """Log failed requests to error log file"""
    log_dir = get_config_dir()
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "error.log")

    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "endpoint": endpoint,
        "status_code": status_code,
        "request": request_body,
        "response": response_body,
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


def get_config_dir():
    """Get the config directory path based on the OS"""
    if platform.system() == "Windows":
        return os.path.expandvars("%APPDATA%/ghc-api")
    else:
        return os.path.expanduser("~/.ghc-api")


def print_model_mappings():
    """Print the loaded model mappings"""
    print("\n" + "=" * 60)
    print("Model Name Mappings")
    print("=" * 60)

    if model_mappings.exact_mappings:
        print("\nExact Mappings:")
        for source, target in model_mappings.exact_mappings.items():
            print(f"  {source} -> {target}")
    else:
        print("\nExact Mappings: (none)")

    if model_mappings.prefix_mappings:
        print("\nPrefix Mappings:")
        for prefix, target in model_mappings.prefix_mappings.items():
            print(f"  {prefix}* -> {target}")
    else:
        print("\nPrefix Mappings: (none)")

    print("=" * 60 + "\n")


def print_available_models():
    """Print all available models with their info"""
    if not state.models or not state.models.get("data"):
        print("No models available yet.")
        return

    print("\n" + "=" * 60)
    print("Available Models")
    print("=" * 60)

    models_data = state.models.get("data", [])
    for model in models_data:
        model_id = model.get("id", "unknown")
        # model_name = model.get("name", model_id)
        capabilities = model.get("capabilities", {})
        preview = model.get("preview", False)
        vendor = model.get("vendor", "unknown")
        supported_endpoints = model.get("supported_endpoints", [])

        # Extract model info
        max_input_tokens = capabilities.get("limits", {}).get("max_prompt_tokens", 0)
        if max_input_tokens >= 1000:
            max_input_tokens = f"{max_input_tokens // 1000}K"
        max_output_tokens = capabilities.get("limits", {}).get("max_output_tokens", 0)
        if max_output_tokens >= 1000:
            max_output_tokens = f"{max_output_tokens // 1000}K"
        max_context_window_tokens = capabilities.get("limits", {}).get("max_context_window_tokens", 0)
        if max_context_window_tokens >= 1000:
            max_context_window_tokens = f"{max_context_window_tokens // 1000}K"

        supports_vision = capabilities.get("supports", {}).get("vision", False)
        supports_tool_calls = capabilities.get("supports", {}).get("tool_calls", False)
        supports_anthropic_api = "/v1/messages" in supported_endpoints

        flags = []
        if supports_vision:
            flags.append("Vision")
        if supports_tool_calls:
            flags.append("Tool")
        if supports_anthropic_api:
            flags.append("Anthropic")
        if preview:
            flags.append("Preview")

        flags_str = ",".join(flags) if flags else ""
        print(f"{model_id:30}\tctx: {max_context_window_tokens} in: {max_input_tokens or 'N/A'}\t out: {max_output_tokens or 'N/A'}\t[{vendor}] ({flags_str})")
    print("\n" + "=" * 60 + "\n")


# ============================================================================
# Orphaned Tool Result Handling
# ============================================================================

# Log file for orphaned tool_result cleanup events
TOOL_RESULT_CLEANUP_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tool_result_cleanup.jl")

def log_tool_result_cleanup(log_entry: Dict) -> None:
    """
    Write a cleanup event to the JSON Lines log file.

    Log entry contains:
    - timestamp: when the cleanup occurred
    - original_request: the original request payload
    - error_response: the error response from backend
    - orphaned_ids: list of orphaned tool_use_ids found
    - modified_request: the cleaned request payload
    - final_status_code: status code after retry
    - final_response: response after retry (success or error)
    """
    try:
        log_entry["timestamp"] = datetime.now().isoformat()
        with open(TOOL_RESULT_CLEANUP_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[Tool Result Cleanup] Failed to write log: {e}")



# Log file for connection retry events
CONNECTION_RETRY_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "connection_retry.jl")

def log_connection_retry(request_id: str, endpoint: str, attempt: int, max_retries: int, error: Exception) -> None:
    """
    Write a connection retry event to the JSON Lines log file.
    """
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "endpoint": endpoint,
            "attempt": attempt + 1,
            "max_attempts": max_retries + 1,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        with open(CONNECTION_RETRY_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[Connection Retry] Failed to write log: {e}")


def extract_orphaned_tool_use_ids(error_response: str) -> List[str]:
    """
    Extract orphaned tool_use_id(s) from an Anthropic error response.

    Error format example:
    {"error":{"message":"{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",
    \"message\":\"messages.0.content.0: unexpected `tool_use_id` found in `tool_result` blocks: toolu_xxx.
    Each `tool_result` block must have a corresponding `tool_use` block in the previous message.\"}}"}}
    """
    orphaned_ids = []

    # Look for the specific error pattern without regex
    # Pattern: "unexpected `tool_use_id` found in `tool_result` blocks: <id>"
    marker = "unexpected `tool_use_id` found in `tool_result` blocks: "
    start_idx = error_response.find(marker)
    if start_idx != -1:
        start_idx += len(marker)
        # Find the end of the ID (ends with period, space, quote, or backslash)
        end_idx = start_idx
        while end_idx < len(error_response):
            char = error_response[end_idx]
            if char in ".  \"\\'\\n":
                break
            end_idx += 1
        tool_id = error_response[start_idx:end_idx].strip()
        if tool_id:
            orphaned_ids.append(tool_id)

    # Fallback: find all toolu_ prefixed IDs in the error
    if not orphaned_ids:
        search_str = error_response
        prefix = "toolu_"
        while prefix in search_str:
            idx = search_str.find(prefix)
            # Extract the ID (alphanumeric, underscore, hyphen)
            end_idx = idx + len(prefix)
            while end_idx < len(search_str):
                char = search_str[end_idx]
                if char.isalnum() or char in "_-":
                    end_idx += 1
                else:
                    break
            tool_id = search_str[idx:end_idx]
            if tool_id and tool_id not in orphaned_ids:
                orphaned_ids.append(tool_id)
            search_str = search_str[end_idx:]

    return orphaned_ids


def remove_orphaned_tool_results(messages: List[Dict], orphaned_ids: List[str]) -> List[Dict]:
    """
    Remove tool_result blocks with orphaned tool_use_ids from messages.

    This modifies the messages to remove tool_result blocks that don't have
    a corresponding tool_use block in a previous assistant message.
    """
    if not orphaned_ids:
        return messages

    orphaned_set = set(orphaned_ids)
    cleaned_messages = []

    for msg in messages:
        if msg.get("role") != "user":
            cleaned_messages.append(msg)
            continue

        content = msg.get("content")
        if not isinstance(content, list):
            cleaned_messages.append(msg)
            continue

        # Filter out orphaned tool_result blocks
        cleaned_content = []
        removed_count = 0
        for block in content:
            if block.get("type") == "tool_result":
                tool_use_id = block.get("tool_use_id", "")
                if tool_use_id in orphaned_set:
                    print(f"[Tool Result Cleanup] Removing orphaned tool_result with id: {tool_use_id}")
                    removed_count += 1
                    continue
            cleaned_content.append(block)

        if removed_count > 0:
            if cleaned_content:
                # Keep the message with remaining content
                cleaned_msg = dict(msg)
                cleaned_msg["content"] = cleaned_content
                cleaned_messages.append(cleaned_msg)
            # If no content left, skip the message entirely
        else:
            cleaned_messages.append(msg)

    return cleaned_messages


def is_orphaned_tool_result_error(status_code: int, response_text: str) -> bool:
    """Check if the error is about orphaned tool_result blocks"""
    if status_code != 400:
        return False
    return "tool_use_id" in response_text and "tool_result" in response_text


def is_encrypted_content_parse_error(status_code: int, response_text: str) -> bool:
    """Check if the error indicates encrypted content cannot be decrypted or parsed"""
    if status_code != 400:
        return False

    prefix = "The encrypted content"
    suffix = "Reason: Encrypted content could not be decrypted or parsed."
    stripped = response_text.strip()
    return stripped.startswith(prefix) and stripped.endswith(suffix)
