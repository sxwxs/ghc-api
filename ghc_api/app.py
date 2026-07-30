"""
Flask application factory and initialization
"""

from flask import Flask, g, jsonify, redirect, request, url_for

from .auth import ANONYMOUS_USER_ID, require_auth
from .email_auth import get_email_auth, init_email_auth
from .routes.agent import agent_bp
from .routes.anthropic import anthropic_bp
from .routes.auth import auth_bp
from .routes.dashboard import dashboard_bp
from .routes.openai import openai_bp
from .state import state


# Paths that require an approved API token when state.enable_auth is True.
# Legacy top-level enable_auth protects only these paths. The separate nested
# email-auth mode additionally protects dashboard and management routes.
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
    init_email_auth(app)

    @app.before_request
    def _auth_gate():
        """Set g.user_id for every request. When auth is enabled, validate the
        token on protected LLM-API paths and reject (401/403) if invalid."""
        if not state.enable_auth:
            g.user_id = ANONYMOUS_USER_ID
            return None

        if request.path in PROTECTED_PATHS:
            result = require_auth(request)
            if result.user_id is None:
                return jsonify({
                    "error": result.error_code,
                    "message": result.error_message,
                }), result.http_status
            g.user_id = result.user_id
            return None

        # Legacy top-level enable_auth protects only LLM API endpoints. Email
        # dashboard/management gating is opt-in through nested auth.enabled.
        if not state.enable_email_auth:
            g.user_id = ANONYMOUS_USER_ID
            return None

        g.user_id = ANONYMOUS_USER_ID
        public_path = (
            request.path in {"/login", "/signup", "/favicon.ico"}
            or request.path.startswith("/static/")
            or request.path.startswith("/api/auth/")
            or request.path.startswith("/api/register/")
        )
        if public_path:
            return None

        auth = get_email_auth(app)
        if auth is None:
            return jsonify({"error": "auth_not_configured"}), 503

        user = auth.current_user()
        if request.path == "/account":
            if user is not None:
                g.user_id = user.get("id") or ANONYMOUS_USER_ID
                return None
            return redirect(url_for("auth.login_page", next=request.path))

        if user is not None and user.get("is_admin"):
            g.user_id = user.get("id") or ANONYMOUS_USER_ID
            return None

        wants_html = request.method == "GET" and not request.path.startswith("/api/")
        if wants_html:
            return redirect(url_for("auth.login_page", next=request.path))
        return jsonify({"ok": False, "error": "admin_required"}), 403

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
