"""
GitHub token management including Device Flow authentication.
"""

import os
import threading
import time
from typing import Any, Dict, Optional

import requests

from .config import GITHUB_OAUTH_CLIENT_ID
from .utils import get_config_dir

_DEVICE_CODE_URL = "https://github.com/login/device/code"
_TOKEN_URL = "https://github.com/login/oauth/access_token"


def get_token_file_path() -> str:
    """Get the path to the token file without modifying the filesystem."""
    return os.path.join(get_config_dir(), "github_token.txt")


def load_github_token_from_file() -> Optional[str]:
    """Load GitHub token from github_token.txt in the config directory."""
    token_file_path = get_token_file_path()
    if os.path.exists(token_file_path):
        try:
            with open(token_file_path, "r", encoding="utf-8") as f:
                token = f.read().strip()
                if token:
                    print(f"Loaded GitHub token from {token_file_path}")
                    return token
        except Exception as exc:
            print(f"Failed to read token file: {exc}")
    return None


def save_github_token_to_file(token: str) -> bool:
    """Save a GitHub token to github_token.txt in the config directory."""
    token_file_path = get_token_file_path()
    try:
        os.makedirs(os.path.dirname(token_file_path), exist_ok=True)
        with open(token_file_path, "w", encoding="utf-8") as f:
            f.write(token)
        print(f"Saved GitHub token to {token_file_path}")
        return True
    except Exception as exc:
        print(f"Failed to save token file: {exc}")
        return False


def delete_github_token_file() -> bool:
    """Delete the locally saved GitHub token file. Missing is considered success."""
    token_file_path = get_token_file_path()
    try:
        if os.path.exists(token_file_path):
            os.remove(token_file_path)
            print(f"Deleted GitHub token file: {token_file_path}")
        else:
            print(f"GitHub token file does not exist: {token_file_path}")
        return True
    except OSError as exc:
        print(f"Failed to delete GitHub token file: {exc}")
        return False


def request_github_device_code() -> Dict[str, Any]:
    """Start Device Flow and return internal details, including secret device_code."""
    response = requests.post(
        _DEVICE_CODE_URL,
        data={
            "client_id": GITHUB_OAUTH_CLIENT_ID,
            "scope": "read:user copilot",
        },
        headers={"Accept": "application/json"},
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(f"Failed to get device code: {response.status_code} {response.text}")

    data = response.json()
    required = ("device_code", "user_code", "verification_uri")
    if any(not data.get(key) for key in required):
        raise RuntimeError("GitHub Device Flow returned an incomplete response")
    return {
        "device_code": data["device_code"],
        "user_code": data["user_code"],
        "verification_uri": data["verification_uri"],
        "expires_in": int(data.get("expires_in", 900)),
        "interval": int(data.get("interval", 5)),
    }


def poll_github_device_flow(device: Dict[str, Any], progress=None) -> str:
    """Poll GitHub until Device Flow succeeds, fails, or expires."""
    interval = int(device.get("interval", 5))
    expires_in = int(device.get("expires_in", 900))
    started_at = time.time()

    while time.time() - started_at < expires_in:
        time.sleep(interval)
        try:
            response = requests.post(
                _TOKEN_URL,
                data={
                    "client_id": GITHUB_OAUTH_CLIENT_ID,
                    "device_code": device["device_code"],
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
                headers={"Accept": "application/json"},
                timeout=30,
            )
        except requests.RequestException as exc:
            if progress:
                progress("retrying", str(exc))
            continue

        if not response.ok:
            if progress:
                progress("retrying", f"HTTP {response.status_code}")
            continue

        token_data = response.json()
        error = token_data.get("error")
        if error == "authorization_pending":
            if progress:
                progress("pending", "")
            continue
        if error == "slow_down":
            interval += 5
            continue
        if error == "expired_token":
            raise RuntimeError("Authorization expired. Please try again.")
        if error == "access_denied":
            raise RuntimeError("Authorization denied by user.")
        if error:
            raise RuntimeError(token_data.get("error_description", error))
        if token_data.get("access_token"):
            return token_data["access_token"]

    raise RuntimeError("Authorization timed out. Please try again.")


def authenticate_github_device_flow() -> Optional[str]:
    """Run interactive GitHub Device Flow and return the access token."""
    print("\n" + "=" * 60)
    print("GitHub Device Flow Authentication")
    print("=" * 60)
    try:
        device = request_github_device_code()
    except Exception as exc:
        print(str(exc))
        return None

    print(f"\nPlease visit: {device['verification_uri']}")
    print(f"And enter the code: {device['user_code']}")
    print(f"\nWaiting for authorization (expires in {device['expires_in']} seconds)...")
    try:
        import webbrowser
        webbrowser.open(device["verification_uri"])
        print("(Browser opened automatically)")
    except Exception:
        pass

    def progress(status: str, _message: str) -> None:
        if status == "pending":
            print(".", end="", flush=True)

    try:
        token = poll_github_device_flow(device, progress=progress)
    except Exception as exc:
        print(f"\n{exc}")
        return None
    print("\n\nAuthorization successful!")
    return token


class GitHubDeviceFlowManager:
    """Manage one non-blocking Device Flow session for the web UI."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session: Dict[str, Any] = {"status": "idle"}

    def _public_status_locked(self) -> Dict[str, Any]:
        return {
            key: value for key, value in self._session.items()
            if key != "device_code"
        }

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return self._public_status_locked()

    def start(self) -> Dict[str, Any]:
        with self._lock:
            status = self._session.get("status")
            if status == "starting":
                return self._public_status_locked()
            if status == "pending" and time.time() < self._session.get("expires_at", 0):
                return self._public_status_locked()

            starting_session: Dict[str, Any] = {
                "status": "starting",
                "started_at": time.time(),
                "error": None,
            }
            self._session = starting_session

        try:
            device = request_github_device_code()
        except Exception as exc:
            with self._lock:
                if self._session is starting_session:
                    self._session.update({
                        "status": "error",
                        "completed_at": time.time(),
                        "error": str(exc),
                    })
            raise

        now = time.time()
        session = {
            **device,
            "status": "pending",
            "started_at": now,
            "expires_at": now + device["expires_in"],
            "error": None,
        }
        with self._lock:
            if self._session is not starting_session:
                return self._public_status_locked()
            self._session = session
            public = self._public_status_locked()
        threading.Thread(target=self._complete, args=(session,), daemon=True).start()
        return public

    def _complete(self, session: Dict[str, Any]) -> None:
        try:
            token = poll_github_device_flow(session)
            if not token or not save_github_token_to_file(token):
                raise RuntimeError("Authorization succeeded, but the token file could not be saved")

            from .api_helpers import fetch_models, refresh_copilot_token
            from .state import state

            with state.token_lock:
                state.github_token = token
                state.github_token_source = "device_flow"
                state.copilot_token = None
                state.token_expires_at = 0
            refresh_error = None
            try:
                refresh_copilot_token(force=True)
                fetch_models()
            except Exception as exc:
                refresh_error = str(exc)

            with self._lock:
                if self._session is session:
                    self._session.update({
                        "status": "success",
                        "completed_at": time.time(),
                        "error": None,
                        "message": (
                            "GitHub login completed and Copilot token refreshed."
                            if not refresh_error else
                            "GitHub login completed, but Copilot token refresh failed: " + refresh_error
                        ),
                    })
        except Exception as exc:
            with self._lock:
                if self._session is session:
                    self._session.update({
                        "status": "error",
                        "completed_at": time.time(),
                        "error": str(exc),
                    })


github_device_flow_manager = GitHubDeviceFlowManager()


def get_github_token() -> Optional[str]:
    """Get GitHub token from environment, local file, or interactive Device Flow."""
    from .state import state

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        print("Using GitHub token from GITHUB_TOKEN environment variable")
        state.github_token_source = "environment"
        return token

    token = load_github_token_from_file()
    if token:
        state.github_token_source = "file"
        return token

    print("\nNo GitHub token found. Starting GitHub Device Flow authentication...")
    token = authenticate_github_device_flow()
    if token:
        save_github_token_to_file(token)
        state.github_token_source = "device_flow"
        return token
    return None
