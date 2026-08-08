"""
User authentication and token registry for ghc-api.

Provides UserRegistry: a small JSON-backed registry of users, each with a
generated token and a status (pending/approved/revoked). Re-reads the
underlying file when its mtime changes (5-second poll), so changes synced
in via OneDrive on another machine pick up without a restart.

Token presentation precedence (handled in extract_token):
  1. Authorization: Bearer <token>
  2. x-api-key: <token>
  3. ?api_key=<token>

When state.enable_auth is False, callers should bypass require_auth entirely
and tag requests with user_id="anonymous". The middleware in app.py does this.
"""

from __future__ import annotations

import json
import os
import secrets
import tempfile
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ANONYMOUS_USER_ID = "anonymous"
TOKEN_PREFIX = "gha_"
RELOAD_INTERVAL_SECONDS = 5.0
REGISTRY_SCHEMA_VERSION = 1

STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REVOKED = "revoked"
VALID_STATUSES = {STATUS_PENDING, STATUS_APPROVED, STATUS_REVOKED}


@dataclass
class UserRecord:
    user_id: str
    display_name: str
    token: str
    status: str
    created_at: int
    approved_at: Optional[int] = None

    def to_public_dict(self) -> Dict[str, Any]:
        """Dict shape returned by /api/users (includes the token; this endpoint
        is admin-only by virtue of nginx/reverse-proxy auth in production)."""
        return asdict(self)

    def to_safe_dict(self) -> Dict[str, Any]:
        """Dict shape returned by /api/users-list (no token, safe to expose
        to non-admin views like the filter-by-user dropdown)."""
        d = asdict(self)
        d.pop("token", None)
        return d


def generate_token() -> str:
    return f"{TOKEN_PREFIX}{secrets.token_urlsafe(32)}"


def _registry_path() -> Path:
    """Resolve the users.json path. Prefer OneDrive (shared across machines)
    when available; otherwise fall back to the local config directory."""
    # Local import to avoid circular imports at module load time.
    from .config_sync import get_sync_root
    from .utils import get_config_dir

    sync_root = get_sync_root()
    if sync_root is not None:
        return sync_root / "users.json"
    return Path(get_config_dir()) / "users.json"


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON to `path` atomically (tmp file + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".users.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


class UserRegistry:
    """In-memory cache over users.json with mtime-based reload."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._by_token: Dict[str, UserRecord] = {}
        self._by_user_id: Dict[str, UserRecord] = {}
        self._path: Optional[Path] = None
        self._mtime: float = 0.0
        self._last_check: float = 0.0

    # ------------------------------------------------------------------
    # File <-> memory
    # ------------------------------------------------------------------

    def _ensure_fresh(self) -> None:
        """Reload from disk if the file's mtime has changed since last load.
        Cheap: stats once per RELOAD_INTERVAL_SECONDS at most."""
        now = time.monotonic()
        if (now - self._last_check) < RELOAD_INTERVAL_SECONDS and self._by_token:
            return
        self._last_check = now

        path = _registry_path()
        self._path = path

        try:
            stat = path.stat()
        except FileNotFoundError:
            # File doesn't exist yet — treat as empty registry, but don't
            # overwrite the in-memory state if we already had records loaded
            # from a previous run (defensive: avoids wiping registry when
            # OneDrive briefly hides the file mid-sync).
            if not self._by_token:
                self._by_token = {}
                self._by_user_id = {}
                self._mtime = 0.0
            return

        if stat.st_mtime == self._mtime and self._by_token:
            return

        self._load_from_disk(path)
        self._mtime = stat.st_mtime

    def _load_from_disk(self, path: Path) -> None:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[Auth] Failed to read user registry at {path}: {e}")
            return

        users = data.get("users") if isinstance(data, dict) else None
        if not isinstance(users, list):
            return

        by_token: Dict[str, UserRecord] = {}
        by_user_id: Dict[str, UserRecord] = {}
        for raw in users:
            if not isinstance(raw, dict):
                continue
            user_id = raw.get("user_id")
            token = raw.get("token")
            status = raw.get("status")
            if not user_id or not token or status not in VALID_STATUSES:
                continue
            record = UserRecord(
                user_id=str(user_id),
                display_name=str(raw.get("display_name") or user_id),
                token=str(token),
                status=str(status),
                created_at=int(raw.get("created_at") or 0),
                approved_at=int(raw["approved_at"]) if raw.get("approved_at") else None,
            )
            by_token[record.token] = record
            by_user_id[record.user_id] = record

        self._by_token = by_token
        self._by_user_id = by_user_id

    def _flush_to_disk(self) -> None:
        if self._path is None:
            self._path = _registry_path()

        # Iteration order = insertion order = creation order.
        users = [r.to_public_dict() for r in self._by_user_id.values()]
        payload = {"version": REGISTRY_SCHEMA_VERSION, "users": users}

        _atomic_write_json(self._path, payload)

        try:
            self._mtime = self._path.stat().st_mtime
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def lookup_by_token(self, token: str) -> Optional[UserRecord]:
        with self._lock:
            self._ensure_fresh()
            return self._by_token.get(token)

    def lookup_by_user_id(self, user_id: str) -> Optional[UserRecord]:
        with self._lock:
            self._ensure_fresh()
            return self._by_user_id.get(user_id)

    def list_all(self) -> List[UserRecord]:
        with self._lock:
            self._ensure_fresh()
            return list(self._by_user_id.values())

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def create_pending(self, user_id: str, display_name: str) -> Tuple[Optional[UserRecord], Optional[str]]:
        """Create a new pending user. Returns (record, error_message)."""
        user_id = (user_id or "").strip()
        display_name = (display_name or "").strip() or user_id

        if not user_id:
            return None, "user_id is required"
        if len(user_id) > 64:
            return None, "user_id is too long (max 64 chars)"
        if user_id == ANONYMOUS_USER_ID:
            return None, f"'{ANONYMOUS_USER_ID}' is reserved"
        # Keep IDs URL-safe and easy to type in the dashboard.
        for ch in user_id:
            if not (ch.isalnum() or ch in "_-."):
                return None, "user_id may only contain letters, digits, '_', '-', '.'"

        with self._lock:
            self._ensure_fresh()
            if user_id in self._by_user_id:
                return None, f"user_id '{user_id}' already exists"

            record = UserRecord(
                user_id=user_id,
                display_name=display_name[:128],
                token=generate_token(),
                status=STATUS_PENDING,
                created_at=int(time.time()),
                approved_at=None,
            )
            self._by_token[record.token] = record
            self._by_user_id[record.user_id] = record
            self._flush_to_disk()
            return record, None

    def set_status(self, user_id: str, new_status: str) -> Tuple[Optional[UserRecord], Optional[str]]:
        if new_status not in VALID_STATUSES:
            return None, f"invalid status: {new_status}"
        with self._lock:
            self._ensure_fresh()
            record = self._by_user_id.get(user_id)
            if record is None:
                return None, f"user_id '{user_id}' not found"
            record.status = new_status
            if new_status == STATUS_APPROVED and record.approved_at is None:
                record.approved_at = int(time.time())
            self._flush_to_disk()
            return record, None

    def delete(self, user_id: str) -> Tuple[bool, Optional[str]]:
        with self._lock:
            self._ensure_fresh()
            record = self._by_user_id.pop(user_id, None)
            if record is None:
                return False, f"user_id '{user_id}' not found"
            self._by_token.pop(record.token, None)
            self._flush_to_disk()
            return True, None


_registry: Optional[UserRegistry] = None
_registry_lock = threading.Lock()


def get_user_registry() -> UserRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = UserRegistry()
    return _registry


# ----------------------------------------------------------------------
# Token extraction from incoming requests
# ----------------------------------------------------------------------


def extract_token(request) -> Optional[str]:
    """Pull the auth token from an incoming Flask request, trying in order:
    Authorization: Bearer, x-api-key, ?api_key="""
    auth_header = request.headers.get("Authorization") or ""
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        if token:
            return token

    x_api_key = request.headers.get("x-api-key") or request.headers.get("X-Api-Key")
    if x_api_key:
        token = x_api_key.strip()
        if token:
            return token

    api_key_arg = request.args.get("api_key")
    if api_key_arg:
        return api_key_arg.strip()

    return None


def redact_auth_headers(headers: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of `headers` with auth values replaced by '***REDACTED***'.
    Used before persisting request headers to the cache, so the dashboard
    request-detail view can't expose other users' tokens."""
    if not isinstance(headers, dict):
        return headers
    redacted = dict(headers)
    sensitive_names = {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "api-key",
        "cookie",
        "set-cookie",
        "x-auth-token",
        "x-access-token",
        "x-github-token",
    }
    for key in list(redacted.keys()):
        lower = key.lower().strip()
        if (
            lower in sensitive_names
            or lower.endswith("-access-token")
            or lower.endswith("-auth-token")
            or lower.endswith("-api-key")
            or lower.endswith("-secret")
        ):
            redacted[key] = "***REDACTED***"
    return redacted


# ----------------------------------------------------------------------
# Middleware helper
# ----------------------------------------------------------------------


@dataclass
class AuthResult:
    user_id: Optional[str]
    error_code: Optional[str] = None  # e.g. "missing_token", "invalid_token", "token_pending"
    error_message: Optional[str] = None
    http_status: int = 401


def require_auth(request) -> AuthResult:
    """Validate the request's token against the registry. Returns an AuthResult
    with user_id set on success, or error_code/message/http_status set on failure.

    Callers (the Flask before_request hook) translate failures into JSON 401/403."""
    token = extract_token(request)
    if not token:
        return AuthResult(
            user_id=None,
            error_code="missing_token",
            error_message="Provide an API token via 'Authorization: Bearer <token>', 'x-api-key: <token>', or '?api_key=<token>'.",
            http_status=401,
        )

    registry = get_user_registry()
    record = registry.lookup_by_token(token)
    if record is None:
        return AuthResult(
            user_id=None,
            error_code="invalid_token",
            error_message="The provided token is not recognized.",
            http_status=401,
        )

    if record.status == STATUS_PENDING:
        return AuthResult(
            user_id=None,
            error_code="token_pending",
            error_message="Token has been registered but is pending administrator approval.",
            http_status=403,
        )

    if record.status == STATUS_REVOKED:
        return AuthResult(
            user_id=None,
            error_code="token_revoked",
            error_message="Token has been revoked.",
            http_status=403,
        )

    if record.status != STATUS_APPROVED:
        return AuthResult(
            user_id=None,
            error_code=f"token_{record.status}",
            error_message=f"Token is in unexpected state: {record.status}",
            http_status=403,
        )

    return AuthResult(user_id=record.user_id)
