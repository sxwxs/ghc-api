#!/usr/bin/env python3
"""
GitHub Copilot API Proxy Server

A Flask application that replicates the functionality of the Node.js copilot-api project,
serving as a proxy server for GitHub Copilot API with caching and monitoring capabilities.
"""

import argparse
import os
import yaml

from .app import create_app, initialize_app
from .config import DEBUG, model_mappings, DEFAULT_MODEL_MAPPINGS
from .generate_config import generate_config_file
from .state import state
from .token_manager import get_config_dir
from .utils import print_model_mappings, print_available_models


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


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

    # Load config from file if it exists
    config_dir = get_config_dir()
    config_path = os.path.join(config_dir, "config.yaml")

    if not os.path.exists(config_path):
        print(f"No config file found at {config_path}, will generate one.")
        generate_config_file()
    try:
        config = load_config(config_path)
        print(config)
        # Load server settings from config (can be overridden by command line)
        host = config.get('address', 'localhost')
        port = config.get('port', 8313)
        debug = config.get('debug', DEBUG)

        # Set account type from config
        state.account_type = config.get('account_type', 'individual')

        # Set vscode version from config
        if 'vscode_version' in config:
            state.vscode_version = config['vscode_version']

        # Set api_version from config
        if 'api_version' in config:
            state.api_version = config['api_version']

        # Set copilot_version from config
        if 'copilot_version' in config:
            state.copilot_version = config['copilot_version']

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

    # Command line args override config file
    if args.address is not None:
        print(f"Overriding host to: {args.address} (was {host})")
        host = args.address
    if args.port is not None:
        print(f"Overriding port to: {args.port} (was {port})")
        port = args.port

    # Create the Flask app
    app = create_app()

    # Initialize the app context
    with app.app_context():
        initialize_app()

    # Print available models after initialization
    print_available_models()

    # Print model mappings at startup
    print_model_mappings()

    print(f"Starting GitHub Copilot API Proxy on {host}:{port}")
    print(f"Dashboard: http://{host}:{port}/")
    print(f"OpenAI API: http://{host}:{port}/v1/chat/completions")
    print(f"Anthropic API: http://{host}:{port}/v1/messages")

    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    main()
