"""GitHub Copilot API Proxy - ghc-api package"""

__version__ = "1.0.0"

from .app import create_app, initialize_app
from .main import main

__all__ = ["create_app", "initialize_app", "main", "__version__"]
