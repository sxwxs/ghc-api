"""
Flask application factory and initialization
"""

from flask import Flask, g, jsonify, request

from .auth import ANONYMOUS_USER_ID, require_auth
from .routes.agent import agent_bp
from .routes.anthropic import anthropic_bp
from .routes.auth import auth_bp
from .routes.dashboard import dashboard_bp
from .routes.openai import openai_bp
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
    "/v1/messages",
    "/v1/messages/count_tokens",
    "/v1/models",
    "/models",
    "/v1/models/full/",
    "/models/full/",
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

    # Register blueprints
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(openai_bp)
    app.register_blueprint(anthropic_bp)
    app.register_blueprint(agent_bp)
    app.register_blueprint(auth_bp)

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

    try:
        refresh_copilot_token()
        fetch_models()
    except Exception as exc:
        # Keep the dashboard available so an operator can inspect refresh state
        # or replace the GitHub token through Device Flow.
        print(f"Application token initialization failed: {exc}")

    if not state.token_usage_reporter_started:
        start_token_usage_reporter()
        state.token_usage_reporter_started = True

    print("Application initialized successfully")
