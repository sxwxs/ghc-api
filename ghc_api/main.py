#!/usr/bin/env python3
"""
GitHub Copilot API Proxy Server

A Flask application that replicates the functionality of the Node.js copilot-api project,
serving as a proxy server for GitHub Copilot API with caching and monitoring capabilities.
"""

import argparse
import json
import os
import re
import tempfile
import yaml

from . import __version__
from .anthropic_responses import VALID_MODES, WIRE_PROFILES
from .api_helpers import resolve_ghe_endpoints
from .app import create_app, initialize_app
from .config import (
    DEBUG,
    model_mappings,
    DEFAULT_MODEL_MAPPINGS,
    chat_completions_model_support,
    DEFAULT_CHAT_COMPLETIONS_MODEL_SUPPORT,
)
from .config_sync import print_sync_diff_status
from .generate_config import generate_config_file
from .state import state
from .token_manager import (
    authenticate_github_device_flow,
    delete_github_token_file,
    get_config_dir,
    save_github_token_to_file,
)
from .utils import print_model_mappings, print_available_models


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _config_bool(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _config_positive_int(value, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be an integer > 0")
    return value


def apply_anthropic_responses_config(config: dict) -> None:
    """Validate and apply the Responses-backed Messages settings."""
    proposed_total = _config_positive_int(
        config.get(
            'anthropic_responses_replay_max_bytes',
            state.anthropic_responses_replay_max_bytes,
        ),
        'anthropic_responses_replay_max_bytes',
    )
    proposed_tenant = _config_positive_int(
        config.get(
            'anthropic_responses_replay_max_tenant_bytes',
            state.anthropic_responses_replay_max_tenant_bytes,
        ),
        'anthropic_responses_replay_max_tenant_bytes',
    )
    proposed_record = _config_positive_int(
        config.get(
            'anthropic_responses_replay_max_record_bytes',
            state.anthropic_responses_replay_max_record_bytes,
        ),
        'anthropic_responses_replay_max_record_bytes',
    )
    if proposed_record > proposed_tenant:
        raise ValueError(
            "anthropic_responses_replay_max_record_bytes must not exceed "
            "anthropic_responses_replay_max_tenant_bytes"
        )
    if proposed_tenant > proposed_total:
        raise ValueError(
            "anthropic_responses_replay_max_tenant_bytes must not exceed "
            "anthropic_responses_replay_max_bytes"
        )
    if 'anthropic_responses_compat_enabled' in config:
        state.anthropic_responses_compat_enabled = _config_bool(
            config['anthropic_responses_compat_enabled'],
            'anthropic_responses_compat_enabled',
        )
    if 'anthropic_responses_compat_mode' in config:
        mode = config['anthropic_responses_compat_mode']
        if not isinstance(mode, str) or mode not in VALID_MODES:
            raise ValueError(
                "anthropic_responses_compat_mode must be compatibility or lossless_required"
            )
        state.anthropic_responses_compat_mode = mode
    if 'anthropic_responses_wire_profile' in config:
        profile = config['anthropic_responses_wire_profile']
        if not isinstance(profile, str) or profile not in WIRE_PROFILES:
            raise ValueError(
                "anthropic_responses_wire_profile must be a registered wire profile"
            )
        state.anthropic_responses_wire_profile = profile
    if 'anthropic_responses_model_profiles' in config:
        profiles = config['anthropic_responses_model_profiles']
        if not isinstance(profiles, dict):
            raise ValueError(
                "anthropic_responses_model_profiles must be a string-to-string mapping"
            )
        for model, profile in profiles.items():
            if (
                not isinstance(model, str) or not model.strip()
                or not isinstance(profile, str) or profile not in WIRE_PROFILES
            ):
                raise ValueError(
                    "anthropic_responses_model_profiles contains an invalid model/profile"
                )
        state.anthropic_responses_model_profiles = dict(profiles)
    if 'anthropic_responses_replay_path' in config:
        value = config['anthropic_responses_replay_path']
        if not isinstance(value, str):
            raise ValueError("anthropic_responses_replay_path must be a string")
        state.anthropic_responses_replay_path = value
    if 'anthropic_responses_replay_ttl_seconds' in config:
        state.anthropic_responses_replay_ttl_seconds = _config_positive_int(
            config['anthropic_responses_replay_ttl_seconds'],
            'anthropic_responses_replay_ttl_seconds',
        )
    if 'anthropic_responses_replay_max_bytes' in config:
        state.anthropic_responses_replay_max_bytes = _config_positive_int(
            config['anthropic_responses_replay_max_bytes'],
            'anthropic_responses_replay_max_bytes',
        )
    if 'anthropic_responses_replay_max_tenant_bytes' in config:
        state.anthropic_responses_replay_max_tenant_bytes = _config_positive_int(
            config['anthropic_responses_replay_max_tenant_bytes'],
            'anthropic_responses_replay_max_tenant_bytes',
        )
    if 'anthropic_responses_replay_max_record_bytes' in config:
        state.anthropic_responses_replay_max_record_bytes = _config_positive_int(
            config['anthropic_responses_replay_max_record_bytes'],
            'anthropic_responses_replay_max_record_bytes',
        )
    if 'anthropic_responses_replay_encryption_key_env' in config:
        value = config['anthropic_responses_replay_encryption_key_env']
        if not isinstance(value, str):
            raise ValueError(
                "anthropic_responses_replay_encryption_key_env must be a string"
            )
        state.anthropic_responses_replay_encryption_key_env = value
    for key in (
        'anthropic_responses_replay_require_trusted_tenant',
        'anthropic_responses_replay_trusted_single_user',
    ):
        if key in config:
            setattr(state, key, _config_bool(config[key], key))


def apply_upstream_config(config: dict) -> None:
    """Apply account and upstream endpoint settings to global state."""
    state.account_type = config.get('account_type', 'individual')
    state.github_api_base_url = str(config.get('github_api_base_url', '') or '')
    state.copilot_api_base_url = str(config.get('copilot_api_base_url', '') or '')


def update_top_level_config_values(config_path: str, updates: dict[str, str]) -> None:
    """Update selected top-level YAML scalar values while preserving comments."""
    lines = []
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    found = set()
    patterns = {
        key: re.compile(rf"^{re.escape(key)}\s*:")
        for key in updates
    }
    for index, line in enumerate(lines):
        if line[:1].isspace():
            continue
        for key, pattern in patterns.items():
            if pattern.match(line):
                newline = "\n" if line.endswith("\n") else ""
                lines[index] = f"{key}: {json.dumps(updates[key])}{newline}"
                found.add(key)
                break

    remaining = {
        key: value for key, value in updates.items()
        if key not in found
    }
    if remaining:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        if lines and lines[-1].strip():
            lines.append("\n")
        for key, value in remaining.items():
            lines.append(f"{key}: {json.dumps(value)}\n")

    config_dir = os.path.dirname(config_path) or "."
    os.makedirs(config_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=".config.", suffix=".yaml.tmp", dir=config_dir
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.writelines(lines)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, config_path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def configure_ghe_endpoint(config_path: str, endpoint: str) -> dict[str, str]:
    """Resolve a GHE tenant and persist both upstream endpoint overrides."""
    resolved = resolve_ghe_endpoints(endpoint)
    updates = {
        "github_api_base_url": resolved["github_api_base_url"],
        "copilot_api_base_url": resolved["copilot_api_base_url"],
    }
    update_top_level_config_values(config_path, updates)
    return resolved


def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description='GitHub Copilot API Proxy')
    parser.add_argument('-p', '--port', type=int, help='Port to listen on')
    parser.add_argument('-a', '--address', type=str, help='Address to listen on')
    parser.add_argument('-c', '--config', action='store_true', help='Generate a YAML config file')
    parser.add_argument(
        '--ghe-endpoint', metavar='HOST',
        help='Configure GitHub Enterprise Cloud data-residency endpoints and exit',
    )
    token_actions = parser.add_mutually_exclusive_group()
    token_actions.add_argument(
        '--delete-github-token', action='store_true',
        help='Delete the locally saved GitHub token file and exit',
    )
    token_actions.add_argument(
        '--github-device-login', action='store_true',
        help='Run GitHub Device Flow, replace the locally saved token, and exit',
    )
    parser.add_argument('--enable-auth', dest='enable_auth', action='store_true', default=None,
                        help='Require an approved user token on LLM API endpoints (overrides config)')
    parser.add_argument('--no-enable-auth', dest='enable_auth', action='store_false',
                        help='Disable user-token auth even if enabled in config')
    parser.add_argument('-v', '--version', action='version', version=f'ghc-api {__version__}')

    args = parser.parse_args()

    if args.config:
        generate_config_file()
        return
    if args.delete_github_token:
        delete_github_token_file()
        return

    config_dir = get_config_dir()
    config_path = os.path.join(config_dir, "config.yaml")

    if args.ghe_endpoint:
        try:
            resolved = configure_ghe_endpoint(config_path, args.ghe_endpoint)
        except Exception as exc:
            print(f"Failed to configure GHE endpoint: {exc}")
            return
        print(f"Configured GHE endpoints in {config_path}:")
        print(f"  GitHub OAuth: {resolved['github_web_base_url']}")
        print(f"  GitHub API:   {resolved['github_api_base_url']}")
        print(f"  Copilot API:  {resolved['copilot_api_base_url']}")
        print("Restart ghc-api to use the new endpoints.")
        print("Run 'ghc-api --github-device-login' if this tenant requires a new token.")
        return

    if args.github_device_login:
        if os.path.exists(config_path):
            try:
                apply_upstream_config(load_config(config_path))
            except Exception as exc:
                print(f"Failed to load upstream settings from {config_path}: {exc}")
                return
        token = authenticate_github_device_flow()
        if token:
            save_github_token_to_file(token)
        else:
            print("GitHub Device Flow login failed; the existing token file was not changed.")
        return

    # Load config from file if it exists
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

        # Set account type and optional upstream URL overrides. The overrides
        # are useful for GHE data residency, private gateways, and offline E2E benchmarks.
        apply_upstream_config(config)

        # Set vscode version from config
        if 'vscode_version' in config:
            state.vscode_version = config['vscode_version']

        # Set api_version from config
        if 'api_version' in config:
            state.api_version = config['api_version']

        # Set copilot_version from config
        if 'copilot_version' in config:
            state.copilot_version = config['copilot_version']

        # Load content filtering settings
        if 'system_prompt_remove' in config:
            state.system_prompt_remove = config['system_prompt_remove'] or []
        if 'tool_result_suffix_remove' in config:
            state.tool_result_suffix_remove = config['tool_result_suffix_remove'] or []
        if 'system_prompt_add' in config:
            state.system_prompt_add = config['system_prompt_add'] or []

        # Load retry settings
        if 'max_connection_retries' in config:
            state.max_connection_retries = config['max_connection_retries']
        if 'upstream_read_timeout' in config:
            state.upstream_read_timeout = int(config['upstream_read_timeout'])
        if 'sse_keepalive_interval' in config:
            state.sse_keepalive_interval = int(config['sse_keepalive_interval'])
        if 'auto_remove_encrypted_content_on_parse_error' in config:
            state.auto_remove_encrypted_content_on_parse_error = bool(config['auto_remove_encrypted_content_on_parse_error'])
        if 'save_request_to_file' in config:
            state.save_request_to_file = bool(config['save_request_to_file'])
        if 'disable_onedrive_access' in config:
            state.disable_onedrive_access = bool(config['disable_onedrive_access'])
        if 'enable_tool_call_recovery' in config:
            state.enable_tool_call_recovery = bool(config['enable_tool_call_recovery'])
        apply_anthropic_responses_config(config)
        if 'enable_responses_early_failure_retry' in config:
            state.enable_responses_early_failure_retry = bool(config['enable_responses_early_failure_retry'])
        if 'session_flush_interval' in config:
            state.session_flush_interval = int(config['session_flush_interval'])

        # Load request cache memory limits
        if 'cache_max_entries' in config:
            state.cache_max_entries = int(config['cache_max_entries'])
        if 'cache_max_request_size' in config:
            state.cache_max_request_size = int(config['cache_max_request_size'])
        from .cache import cache as _request_cache
        _request_cache.max_entries = state.cache_max_entries
        _request_cache.max_request_size = state.cache_max_request_size

        # Load web search proxy settings
        if 'enable_web_search_proxy' in config:
            state.enable_web_search_proxy = bool(config['enable_web_search_proxy'])
        if 'web_search_proxy_endpoint' in config:
            state.web_search_proxy_endpoint = config['web_search_proxy_endpoint']

        # Load user-token auth setting
        if 'enable_auth' in config:
            state.enable_auth = bool(config['enable_auth'])

        # Load model mappings from config
        if 'model_mappings' in config:
            model_mappings.load_from_config(config)
        else:
            # Use default mappings if not specified
            model_mappings.load_from_config({"model_mappings": DEFAULT_MODEL_MAPPINGS})

        # Load configured chat completions endpoint support
        if 'chat_completions_model_support' in config:
            chat_completions_model_support.load_from_config(config)
        else:
            chat_completions_model_support.load_from_config({
                "chat_completions_model_support": DEFAULT_CHAT_COMPLETIONS_MODEL_SUPPORT,
            })

        print(f"Loaded configuration from: {config_path}")

    except Exception as e:
        print(f"Error loading config file: {e}")
        # Never fall back from a requested lossless policy to the more
        # permissive in-memory defaults after a validation error.
        state.anthropic_responses_compat_enabled = False
        print("[AnthropicResponsesCompat] Disabled because configuration validation failed")
        # Use default mappings on error
        model_mappings.load_from_config({"model_mappings": DEFAULT_MODEL_MAPPINGS})
        chat_completions_model_support.load_from_config({
            "chat_completions_model_support": DEFAULT_CHAT_COMPLETIONS_MODEL_SUPPORT,
        })

    # Command line args override config file
    if args.address is not None:
        print(f"Overriding host to: {args.address} (was {host})")
        host = args.address
    if args.port is not None:
        print(f"Overriding port to: {args.port} (was {port})")
        port = args.port
    if args.enable_auth is not None:
        print(f"Overriding enable_auth to: {args.enable_auth} (was {state.enable_auth})")
        state.enable_auth = args.enable_auth

    if state.enable_auth:
        print("\n" + "=" * 60)
        print("[Auth] enable_auth=True — LLM API endpoints require an approved")
        print("       user token from the registry (users.json).")
        print("")
        print("       WARNING: dashboard pages (/, /requests, /code-agent-manager)")
        print("       and admin APIs (/api/users/*, /api/runtime-config, /api/config-manager/*)")
        print("       are NOT auth-protected by ghc-api itself. In production, put")
        print("       a reverse proxy (e.g. nginx basic-auth) in front to gate them.")
        print("       See README \"Production deployment\" for a sample nginx config.")
        print("")
        print(f"       For local-only use, bind to 127.0.0.1: ghc-api --enable-auth -a 127.0.0.1")
        print("=" * 60 + "\n")

    # Create the Flask app
    app = create_app()

    # Check config sync diff status on startup.
    print_sync_diff_status()

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
    print(f"Responses API: http://{host}:{port}/v1/responses")
    print(f"Embeddings API: http://{host}:{port}/v1/embeddings")
    print(f"Anthropic API: http://{host}:{port}/v1/messages")

    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    main()
