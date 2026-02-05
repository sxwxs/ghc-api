"""
Flask application factory and initialization
"""

from flask import Flask, jsonify

from .routes.anthropic import anthropic_bp
from .routes.dashboard import dashboard_bp
from .routes.openai import openai_bp


def create_app() -> Flask:
    """Create and configure the Flask application"""
    app = Flask(__name__)

    # Register blueprints
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(openai_bp)
    app.register_blueprint(anthropic_bp)

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

    # Update the state with the token
    state.github_token = token


    refresh_copilot_token()
    fetch_models()
    print("Application initialized successfully")

