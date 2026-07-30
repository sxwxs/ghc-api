from datetime import datetime
import json
import os
import platform
import threading
import time
from typing import Dict, List, Optional


from .config import model_mappings
from .state import state


# Serializes rotation + append so concurrent request threads cannot interleave
# a rotation with another thread's write. This is a process-local lock: with a
# multi-process server (e.g. gunicorn workers) a rotation in one process can
# race another process's append, which may drop a few lines but cannot corrupt
# the file, since every write is a single small append.
_log_write_lock = threading.Lock()


def _rotate_log_if_needed(log_file: str, pending_bytes: int = 0) -> None:
    """Rotate `log_file` when appending `pending_bytes` would exceed the cap.

    The cap is state.error_log_max_bytes, which bounds every diagnostic log
    written through append_log_line() (error.log, connection_retry.jl,
    tool_result_cleanup.jl). With error_log_backup_count == 0 (default) the
    oversized file is simply removed, so total retention equals
    error_log_max_bytes. With N > 0 the usual `.1 .. .N` suffix chain is kept
    instead. Callers must hold `_log_write_lock`.
    """
    max_bytes = getattr(state, "error_log_max_bytes", 0) or 0
    if max_bytes <= 0:
        return
    try:
        if os.path.getsize(log_file) + pending_bytes <= max_bytes:
            return
    except OSError:
        return

    backup_count = max(0, getattr(state, "error_log_backup_count", 0) or 0)
    try:
        if backup_count == 0:
            os.remove(log_file)
            return
        oldest = f"{log_file}.{backup_count}"
        if os.path.exists(oldest):
            os.remove(oldest)
        for index in range(backup_count - 1, 0, -1):
            source = f"{log_file}.{index}"
            if os.path.exists(source):
                os.replace(source, f"{log_file}.{index + 1}")
        os.replace(log_file, f"{log_file}.1")
    except OSError as exc:
        print(f"Failed to rotate log file {log_file}: {exc}")


def append_log_line(log_file: str, line: str) -> None:
    """Append one line to `log_file`, rotating it first when it would overflow."""
    with _log_write_lock:
        _rotate_log_if_needed(log_file, len(line.encode("utf-8")))
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line)


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

    try:
        append_log_line(log_file, json.dumps(log_entry) + "\n")
    except Exception as exc:
        print(f"Failed to write error log: {exc}")


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
        append_log_line(os.path.join(log_dir, "error.log"), json.dumps(entry) + "\n")
        return True
    except Exception as exc:
        print(f"Failed to write upstream error log: {exc}")
        return False


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

# Log file for orphaned tool_result cleanup events. Kept in the user config
# directory -- the package install directory may be read-only (pipx, system
# packages, containers) and must not accumulate runtime data.
def _tool_result_cleanup_log() -> str:
    return os.path.join(get_config_dir(), "tool_result_cleanup.jl")

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
        os.makedirs(get_config_dir(), exist_ok=True)
        append_log_line(_tool_result_cleanup_log(), json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[Tool Result Cleanup] Failed to write log: {e}")



# Log file for connection retry events (see note above about the config dir).
def _connection_retry_log() -> str:
    return os.path.join(get_config_dir(), "connection_retry.jl")

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
        os.makedirs(get_config_dir(), exist_ok=True)
        append_log_line(_connection_retry_log(), json.dumps(log_entry, ensure_ascii=False) + "\n")
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
        # Consume the ID itself. Tool-use IDs are `toolu_` + [A-Za-z0-9_-], so
        # stop at the first character outside that set (period, space, quote,
        # backslash, newline, ...). Do NOT use a literal stop-character set
        # here: ".  \"\\'\\n" also contains 'n' and '\\', which truncated every
        # ID containing the letter 'n' (e.g. `toolu_orphan` -> `toolu_orpha`).
        end_idx = start_idx
        while end_idx < len(error_response):
            char = error_response[end_idx]
            if not (char.isalnum() or char in "_-"):
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


def is_encrypted_content_parse_error(status_code: int, response_text: str) -> bool:
    """Check if the error indicates encrypted content cannot be decrypted or parsed"""
    if status_code != 400:
        return False

    prefix = '{"error":{"message":"The encrypted content '
    suffix = ' could not be verified. Reason: Encrypted content could not be decrypted or parsed.","code":"invalid_request_body"}}'
    stripped = response_text.strip()
    return stripped.startswith(prefix) and stripped.endswith(suffix)
