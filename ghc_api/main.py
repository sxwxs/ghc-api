#!/usr/bin/env python3
"""
GitHub Copilot API Proxy Server

A Flask application that replicates the functionality of the Node.js copilot-api project,
serving as a proxy server for GitHub Copilot API with caching and monitoring capabilities.
"""

import argparse
from multiprocessing.util import DEBUG
import os
import yaml

from .app import create_app, initialize_app
from .state import state
from .token_manager import get_config_dir
from .generate_config import generate_config_file
from .config import DEBUG, model_mappings


# Default model mappings (same as the original hardcoded logic)
DEFAULT_MODEL_MAPPINGS = {
    "exact": {
        # Add exact model name mappings here
        # Example: "gpt-4-turbo-preview": "gpt-4-turbo"
    },
    "prefix": {
        # Prefix-based mappings: if model name starts with the key, replace with value
        "claude-sonnet-4-": "claude-sonnet-4",
        "claude-opus-4-": "claude-opus-4",
    }
}


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


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
        model_name = model.get("name", model_id)
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
        print(f"{model_id:25}\tctx: {max_context_window_tokens} in: {max_input_tokens or 'N/A'}\t out: {max_output_tokens or 'N/A'}\t({'Vision,' if supports_vision else ''}{'Tool' if supports_tool_calls else ''}{'Preview' if preview else ''})")
    print("\n" + "=" * 60 + "\n")


def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description='GitHub Copilot API Proxy')
    parser.add_argument('-p', '--port', type=int, help='Port to listen on')
    parser.add_argument('-a', '--address', type=str, help='Address to listen on')
    parser.add_argument('-c', '--config', action='store_true', help='Generate a YAML config file')

    args = parser.parse_args()

    if args.config:
        generate_config_file()
        return

    # Default values
    host = 'localhost'
    port = 8313
    debug = DEBUG

    # Load config from file if it exists
    config_dir = get_config_dir()
    config_path = os.path.join(config_dir, "config.yaml")

    if os.path.exists(config_path):
        try:
            config = load_config(config_path)

            # Load server settings from config (can be overridden by command line)
            host = config.get('address', 'localhost')
            port = config.get('port', 8313)
            debug = config.get('debug', DEBUG)

            # Set account type from config
            state.account_type = config.get('account_type', 'individual')

            # Set vscode version from config
            if 'vscode_version' in config:
                state.vscode_version = config['vscode_version']

            # Load model mappings from config
            if 'model_mappings' in config:
                model_mappings.load_from_config(config)
            else:
                # Use default mappings if not specified
                model_mappings.load_from_config({"model_mappings": DEFAULT_MODEL_MAPPINGS})

            print(f"Loaded configuration from: {config_path}")

        except Exception as e:
            print(f"Error loading config file: {e}")
            # Use default mappings on error
            model_mappings.load_from_config({"model_mappings": DEFAULT_MODEL_MAPPINGS})
    else:
        # Use default mappings if no config file
        model_mappings.load_from_config({"model_mappings": DEFAULT_MODEL_MAPPINGS})

    # Command line args override config file
    if args.address is not None:
        host = args.address
    if args.port is not None:
        port = args.port

    # Print model mappings at startup
    print_model_mappings()

    # Create the Flask app
    app = create_app()

    # Initialize the app context
    with app.app_context():
        initialize_app()

    # Print available models after initialization
    print_available_models()

    print(f"Starting GitHub Copilot API Proxy on {host}:{port}")
    print(f"Dashboard: http://{host}:{port}/")
    print(f"OpenAI API: http://{host}:{port}/v1/chat/completions")
    print(f"Anthropic API: http://{host}:{port}/v1/messages")

    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    main()
