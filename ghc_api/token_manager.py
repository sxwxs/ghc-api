"""
GitHub token management including Device Flow authentication
"""

import os
import platform
import time
from typing import Optional

import requests

from .config import GITHUB_OAUTH_CLIENT_ID
from .utils import get_config_dir


def get_token_file_path():
    """Get the path to the token file"""
    config_dir = get_config_dir()
    os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, "github_token.txt")


def load_github_token_from_file() -> Optional[str]:
    """Load GitHub token from github_token.txt file in config directory"""
    token_file_path = get_token_file_path()
    if os.path.exists(token_file_path):
        try:
            with open(token_file_path, "r", encoding="utf-8") as f:
                token = f.read().strip()
                if token:
                    print(f"Loaded GitHub token from {token_file_path}")
                    return token
        except Exception as e:
            print(f"Failed to read token file: {e}")
    return None


def save_github_token_to_file(token: str) -> bool:
    """Save GitHub token to github_token.txt file in config directory"""
    token_file_path = get_token_file_path()
    try:
        with open(token_file_path, "w", encoding="utf-8") as f:
            f.write(token)
        print(f"Saved GitHub token to {token_file_path}")
        return True
    except Exception as e:
        print(f"Failed to save token file: {e}")
        return False


def authenticate_github_device_flow() -> Optional[str]:
    """
    Authenticate using GitHub Device Flow.
    This opens a browser for the user to authorize the app.
    Returns the access token if successful, None otherwise.
    """
    print("\n" + "=" * 60)
    print("GitHub Device Flow Authentication")
    print("=" * 60)

    # Step 1: Request device and user codes
    device_code_url = "https://github.com/login/device/code"
    response = requests.post(
        device_code_url,
        data={
            "client_id": GITHUB_OAUTH_CLIENT_ID,
            "scope": "read:user copilot",
        },
        headers={"Accept": "application/json"},
        timeout=30,
    )

    if not response.ok:
        print(f"Failed to get device code: {response.status_code} {response.text}")
        return None

    data = response.json()
    device_code = data.get("device_code")
    user_code = data.get("user_code")
    verification_uri = data.get("verification_uri")
    expires_in = data.get("expires_in", 900)
    interval = data.get("interval", 5)

    print(f"\nPlease visit: {verification_uri}")
    print(f"And enter the code: {user_code}")
    print(f"\nWaiting for authorization (expires in {expires_in} seconds)...")

    # Try to open browser automatically
    try:
        import webbrowser
        webbrowser.open(verification_uri)
        print("(Browser opened automatically)")
    except Exception:
        pass

    # Step 2: Poll for authorization
    token_url = "https://github.com/login/oauth/access_token"
    start_time = time.time()

    while time.time() - start_time < expires_in:
        time.sleep(interval)

        response = requests.post(
            token_url,
            data={
                "client_id": GITHUB_OAUTH_CLIENT_ID,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            headers={"Accept": "application/json"},
            timeout=30,
        )

        if not response.ok:
            continue

        token_data = response.json()

        if token_data.get("error") == "authorization_pending":
            print(".", end="", flush=True)
            continue
        elif token_data.get("error") == "slow_down":
            interval += 5
            continue
        elif token_data.get("error") == "expired_token":
            print("\nAuthorization expired. Please try again.")
            return None
        elif token_data.get("error") == "access_denied":
            print("\nAuthorization denied by user.")
            return None
        elif token_data.get("error"):
            print(f"\nError: {token_data.get('error_description', token_data['error'])}")
            return None
        elif token_data.get("access_token"):
            print("\n\nAuthorization successful!")
            return token_data["access_token"]

    print("\nAuthorization timed out. Please try again.")
    return None


def get_github_token() -> Optional[str]:
    """
    Get GitHub token using the following priority:
    1. Environment variable GITHUB_TOKEN
    2. Token file (~/.ghc-api/github_token.txt)
    3. GitHub Device Flow authentication (interactive)
    """
    # Priority 1: Environment variable
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        print("Using GitHub token from GITHUB_TOKEN environment variable")
        return token

    # Priority 2: Token file
    token = load_github_token_from_file()
    if token:
        return token

    # Priority 3: GitHub Device Flow (interactive)
    print("\nNo GitHub token found. Starting GitHub Device Flow authentication...")
    token = authenticate_github_device_flow()
    if token:
        save_github_token_to_file(token)
        return token

    return None
