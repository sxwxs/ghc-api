import json
import os
import platform
from datetime import datetime


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

        # Extract model info
        max_input_tokens = capabilities.get("limits", {}).get("max_prompt_tokens", 0)
        if max_input_tokens >= 1024:
            max_input_tokens = f"{max_input_tokens // 1024}K"
        max_output_tokens = capabilities.get("limits", {}).get("max_output_tokens", 0)
        if max_output_tokens >= 1024:
            max_output_tokens = f"{max_output_tokens // 1024}K"
        max_context_window_tokens = capabilities.get("limits", {}).get("max_context_window_tokens", 0)
        if max_context_window_tokens >= 1024:
            max_context_window_tokens = f"{max_context_window_tokens // 1024}K"

        supports_vision = capabilities.get("supports", {}).get("vision", False)
        supports_tool_calls = capabilities.get("supports", {}).get("tool_calls", False)
        print(f"{model_id:30}\tctx: {max_context_window_tokens} in: {max_input_tokens or 'N/A'}\t out: {max_output_tokens or 'N/A'}\t({'Vision,' if supports_vision else ''}{'Tool,' if supports_tool_calls else ''}{'Preview' if preview else ''})")
    print("\n" + "=" * 60 + "\n")
