"""
Flask application factory and initialization
"""

from flask import Flask, g, jsonify, request

from .auth import ANONYMOUS_USER_ID, require_auth
from .json_guard import MAX_JSON_NESTING_DEPTH, exceeds_max_nesting
from .routes.agent import agent_bp
from .routes.anthropic import anthropic_bp
from .routes.auth import auth_bp
from .routes.dashboard import dashboard_bp
from .routes.openai import openai_bp
from .routes.proxy import proxy_bp
from .routes.webiq import webiq_bp
from .state import state


# Paths that require an approved user token when state.enable_auth is True.
# Everything not in this set bypasses auth at the Flask layer (dashboard,
# /signup, /api/users/*, /agent, static files). Production deployments are
# expected to put nginx (or equivalent) in front to gate admin paths.
PROTECTED_PATHS = frozenset({
    "/v1/chat/completions",
    "/chat/completions",
    "/v1/responses",
    "/responses",
    "/v1/embeddings",
    "/embeddings",
    "/v1/messages",
    "/v1/messages/count_tokens",
    "/v1/models",
    "/models",
    "/v1/models/full/",
    "/models/full/",
    # Spend the server-held Web IQ quota, so they are gated like LLM paths.
    "/v3/search/web",
    "/v3/search/videos",
    "/v3/browse",
    "/v3/search/news",
    "/v3/search/images",
    "/v3/search/classic",
    "/v3/mcp",
})


def create_app() -> Flask:
    """Create and configure the Flask application"""
    app = Flask(__name__)

    @app.before_request
    def _auth_gate():
        """Set g.user_id for every request. When auth is enabled, validate the
        token on protected LLM-API paths and reject (401/403) if invalid."""
        if not state.enable_auth:
            g.user_id = ANONYMOUS_USER_ID
            return None

        if request.path not in PROTECTED_PATHS:
            g.user_id = ANONYMOUS_USER_ID
            return None

        result = require_auth(request)
        if result.user_id is None:
            return jsonify({
                "error": result.error_code,
                "message": result.error_message,
            }), result.http_status

        g.user_id = result.user_id
        return None

    @app.before_request
    def _reject_deeply_nested_json():
        """Reject a JSON body that nests deeper than the shared guard allows.

        Runs for every endpoint because the danger is not endpoint-specific:
        ``request.get_json()`` raises ``RecursionError`` on a deep enough body
        (``silent=True`` does not catch it, it is not a parse error), and the
        routes that do parse it hand the value to ``copy.deepcopy`` and
        ``json.dumps``, which give out even earlier. Either way a few KB of
        ``[[[[...`` became a 500. Checking once here, before routing work
        begins, keeps every endpoint's answer a deterministic 400.

        It runs after the auth gate so an unauthenticated client cannot spend
        the scan on a protected path.
        """
        if not request.is_json:
            return None
        if not exceeds_max_nesting(request.get_data(cache=True)):
            return None

        message = (
            "JSON nesting is too deep: exceeds the maximum of "
            f"{MAX_JSON_NESTING_DEPTH} levels"
        )
        if request.blueprint == "anthropic":
            return jsonify({
                "type": "error",
                "error": {"type": "invalid_request_error", "message": message},
            }), 400
        return jsonify({
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": "invalid_json",
            }
        }), 400

    # Register blueprints
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(openai_bp)
    app.register_blueprint(proxy_bp)
    app.register_blueprint(anthropic_bp)
    app.register_blueprint(agent_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(webiq_bp)

    # Error handlers
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "Internal server error"}), 500

    return app


def initialize_app() -> None:
    """Initialize the application (token, models, etc.)"""
    from .api_helpers import fetch_models, refresh_copilot_token
    from .state import state
    from .token_usage_reporter import start_token_usage_reporter
    from .token_manager import get_github_token

    # Get GitHub token using the token management system
    token = get_github_token()
    if not token:
        print("\n" + "=" * 60)
        print("ERROR: No GitHub token available!")
        print("=" * 60)
        print("Options to provide a GitHub token:")
        print("  1. Set GITHUB_TOKEN environment variable")
        print("  2. Create a github_token.txt file in the config directory")
        print("  3. Run the app again to use interactive Device Flow authentication")
        print("=" * 60)
        return

    # Update the state with the token and expose its origin in the manager UI.
    state.github_token = token
    if state.github_token_source == "unconfigured":
        state.github_token_source = "file"

    token_initialized = False
    try:
        refresh_copilot_token()
        fetch_models()
        token_initialized = True
    except Exception as exc:
        # Keep the dashboard available so an operator can inspect refresh state
        # or replace the GitHub token through Device Flow.
        print(f"Application token initialization failed: {exc}")

    if not state.token_usage_reporter_started:
        start_token_usage_reporter()
        state.token_usage_reporter_started = True

    if token_initialized:
        print("Application initialized successfully")
    else:
        print("Application started, but token initialization is incomplete. Use the manager UI or the commands above to recover.")
