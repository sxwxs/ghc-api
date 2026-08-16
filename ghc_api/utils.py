from datetime import datetime
import json
import os
import platform
import time
from typing import Any, Dict, List, Optional, Tuple


from .config import model_mappings
from .state import state


def get_client_ip(request) -> str:
    """Extract the client IP from a Flask request, honoring common proxy headers.

    X-Forwarded-For is preferred, but a second proxy layer can overwrite it with a
    loopback address (e.g. nginx using $remote_addr). When XFF is loopback, fall
    back to X-Real-IP, which carries the genuine client address in that setup.
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client = forwarded_for.split(",")[0].strip()
        if client and not client.startswith("127."):
            return client
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    return request.remote_addr or "unknown"


def log_error_request(endpoint: str, request_body: dict, response_body: str, status_code: int, client_ip: str = None):
    """Log failed requests to error log file"""
    log_dir = get_config_dir()
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "error.log")

    log_entry = {
        "timestamp": int(time.time()),
        "client_ip": client_ip,
        "endpoint": endpoint,
        "status_code": status_code,
        "request": request_body,
        "response": response_body,
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


def log_upstream_error(
    operation: str,
    endpoint: str,
    status_code: Optional[int] = None,
    response_body: str = "",
    error: str = "",
    max_response_chars: int = 65536,
) -> bool:
    """Append a structured upstream failure to error.log without logging auth headers."""
    original_body = response_body or ""
    logged_body = original_body[:max_response_chars]
    entry = {
        "timestamp": int(time.time()),
        "event_type": "upstream_error",
        "operation": operation,
        "endpoint": endpoint,
        "status_code": status_code,
        "error": error,
        "response": logged_body,
        "response_length": len(original_body),
        "response_truncated": len(logged_body) < len(original_body),
    }
    try:
        log_dir = get_config_dir()
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "error.log"), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return True
    except Exception as exc:
        print(f"Failed to write upstream error log: {exc}")
        return False


def get_config_dir():
    """Get the config directory path based on an override or the OS default."""
    override = os.environ.get("GHC_API_CONFIG_DIR")
    if override:
        return os.path.abspath(os.path.expanduser(os.path.expandvars(override)))
    if platform.system() == "Windows":
        return os.path.expandvars("%APPDATA%/ghc-api")
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
def diagnostic_log_path(filename: str) -> str:
    """Resolve a runtime JSON Lines log under the ghc-api config directory.

    Runtime output must not land in the package directory: that is read-only in
    most installs, it puts request ids and upstream endpoints from a real
    deployment one stray ``git add`` away from the repository, and it leaves the
    working tree dirty after every local run. Resolved on each call so that a
    ``GHC_API_CONFIG_DIR`` set after import is still honoured.
    """
    directory = get_config_dir()
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, filename)


TOOL_RESULT_CLEANUP_LOG_NAME = "tool_result_cleanup.jl"

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
        with open(diagnostic_log_path(TOOL_RESULT_CLEANUP_LOG_NAME), "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[Tool Result Cleanup] Failed to write log: {e}")



# Log file for connection retry events
CONNECTION_RETRY_LOG_NAME = "connection_retry.jl"

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
        with open(diagnostic_log_path(CONNECTION_RETRY_LOG_NAME), "a", encoding="utf-8") as f:
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

    from .counters import counters
    counters.incr("mod.orphaned_tool_cleanup")
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


# Guard against pathological nesting while scanning arbitrary client payloads.
_MAX_ENCRYPTED_SCAN_DEPTH = 32

_TOOL_CALL_ITEM_TYPES = frozenset({
    "function_call",
    "custom_tool_call",
    "computer_call",
    "local_shell_call",
})

_TOOL_OUTPUT_ITEM_TYPES = frozenset({
    "function_call_output",
    "custom_tool_call_output",
    "computer_call_output",
    "local_shell_call_output",
})

_REMOVED_TOOL_OUTPUT_PLACEHOLDER = (
    "[ghc-api] tool output omitted: encrypted content could not be decrypted upstream"
)


def is_encrypted_content_parse_error(status_code: int, response_text: str) -> bool:
    """Check if the error indicates encrypted content cannot be decrypted or parsed"""
    if status_code != 400:
        return False

    try:
        error = json.loads(response_text).get("error", {})
    except (AttributeError, json.JSONDecodeError, TypeError):
        return False

    if not isinstance(error, dict) or error.get("code") != "invalid_request_body":
        return False

    message = error.get("message", "")
    if not isinstance(message, str):
        return False

    normalized = message.lower()
    return (
        (
            normalized.startswith("the encrypted content ")
            and normalized.endswith(
                " could not be verified. reason: encrypted content could not be decrypted or parsed."
            )
        )
        or normalized == "encrypted function output content could not be decrypted or decoded."
    )


def _contains_encrypted_content(value: Any, depth: int = 0) -> bool:
    if depth > _MAX_ENCRYPTED_SCAN_DEPTH:
        return False
    if isinstance(value, dict):
        if "encrypted_content" in value:
            return True
        return any(_contains_encrypted_content(child, depth + 1) for child in value.values())
    if isinstance(value, list):
        return any(_contains_encrypted_content(child, depth + 1) for child in value)
    return False


def _strip_encrypted_content(value: Any, depth: int = 0) -> Tuple[Any, bool]:
    """Remove encrypted payloads while keeping the surrounding item structure intact."""
    if depth > _MAX_ENCRYPTED_SCAN_DEPTH:
        return value, False

    if isinstance(value, dict):
        changed = "encrypted_content" in value
        result = {}
        for key, child in value.items():
            if key == "encrypted_content":
                continue
            new_child, child_changed = _strip_encrypted_content(child, depth + 1)
            changed = changed or child_changed
            result[key] = new_child
        return result, changed

    if isinstance(value, list):
        changed = False
        result = []
        for element in value:
            # A content block that carries encrypted data is meaningless once the
            # payload is gone, so drop the block rather than leaving a husk behind.
            if isinstance(element, dict) and _contains_encrypted_content(element, depth + 1):
                changed = True
                continue
            new_element, element_changed = _strip_encrypted_content(element, depth + 1)
            changed = changed or element_changed
            result.append(new_element)
        return result, changed

    return value, False


def _ensure_tool_output_present(item: Dict[str, Any]) -> Dict[str, Any]:
    """Keep a tool output item structurally valid after its content was stripped."""
    output = item.get("output")
    if isinstance(output, list):
        if not output:
            item["output"] = [{"type": "output_text", "text": _REMOVED_TOOL_OUTPUT_PLACEHOLDER}]
    elif isinstance(output, str):
        if not output.strip():
            item["output"] = _REMOVED_TOOL_OUTPUT_PLACEHOLDER
    elif output is None:
        item["output"] = _REMOVED_TOOL_OUTPUT_PLACEHOLDER
    return item


def remove_encrypted_content_items(request_input: Any) -> Tuple[Any, int]:
    """Strip encrypted content from Responses input, preserving tool-call pairing.

    Encrypted data appears either directly on a reasoning item or nested inside tool
    output content blocks. Dropping a whole ``function_call_output`` would orphan its
    ``function_call`` and make the retry fail with "No tool output found for function
    call", so tool outputs are sanitized in place (with a placeholder body) instead of
    being deleted. Any other item carrying encrypted content is removed outright, and
    if a tool *call* itself has to go, its paired output is removed with it.

    Returns the cleaned input and the number of items that were removed or rewritten.
    """
    if not isinstance(request_input, list):
        return request_input, 0

    cleaned_input: List[Any] = []
    changed_count = 0
    dropped_call_ids = set()

    for item in request_input:
        if not isinstance(item, dict) or not _contains_encrypted_content(item):
            cleaned_input.append(item)
            continue

        item_type = item.get("type")
        if item_type in _TOOL_OUTPUT_ITEM_TYPES:
            sanitized, _ = _strip_encrypted_content(item)
            cleaned_input.append(_ensure_tool_output_present(sanitized))
            changed_count += 1
            continue

        changed_count += 1
        if item_type in _TOOL_CALL_ITEM_TYPES:
            call_id = item.get("call_id") or item.get("id")
            if isinstance(call_id, str):
                dropped_call_ids.add(call_id)

    if dropped_call_ids:
        paired: List[Any] = []
        for item in cleaned_input:
            if (
                isinstance(item, dict)
                and item.get("type") in _TOOL_OUTPUT_ITEM_TYPES
                and item.get("call_id") in dropped_call_ids
            ):
                changed_count += 1
                continue
            paired.append(item)
        cleaned_input = paired

    return cleaned_input, changed_count
