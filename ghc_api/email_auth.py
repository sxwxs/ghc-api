"""maglink email-session authentication integration for ghc-api."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from flask import Flask
from maglink import AuthCore, EmailVerificationCore, HttpMailer, Identity, SqliteStore
from maglink.flask import EmailAuth, EmailVerifier

from .auth import STATUS_APPROVED, get_user_registry
from .state import state
from .utils import get_config_dir


class RegistryIdentityProvider:
    """Resolve browser-login identities from users.json and configured admins."""

    def __init__(self, admin_emails: list[str]) -> None:
        self.admin_emails = {
            value.strip().lower() for value in admin_emails if value and value.strip()
        }

    def get_identity(self, email: str) -> Optional[Identity]:
        normalized = (email or "").strip().lower()
        if not normalized:
            return None

        record = get_user_registry().lookup_by_email(normalized)
        is_admin = normalized in self.admin_emails
        if not is_admin:
            if (
                record is None
                or not record.email_verified
                or record.status != STATUS_APPROVED
            ):
                return None

        roles = ("admin",) if is_admin else ("user",)
        return Identity(
            id=record.user_id if record is not None else normalized,
            email=normalized,
            roles=roles,
            claims={
                "display_name": (
                    record.display_name if record is not None else normalized
                ),
                "user_type": record.user_type if record is not None else "configured_admin",
            },
        )

    def can_login(self, email: str) -> bool:
        return self.get_identity(email) is not None


def _public_base_url() -> str:
    hostname = state.auth_hostname.strip().lower()
    if not hostname:
        raise RuntimeError("auth.hostname is required when auth is enabled")
    if "://" in hostname or "/" in hostname or any(ch.isspace() for ch in hostname):
        raise RuntimeError("auth.hostname must be a hostname, not a URL or path")
    return f"https://{hostname}"


def registration_email_allowed(email: str) -> bool:
    if state.auth_allow_public_registration:
        return True
    record = get_user_registry().lookup_by_email(email)
    return bool(record is not None and not record.email_verified)


def init_email_auth(app: Flask) -> None:
    """Configure maglink only when the opt-in public-deployment auth is enabled."""
    if not state.enable_email_auth:
        return

    if not state.auth_secret_key:
        raise RuntimeError("auth.secret_key is required when auth is enabled")
    if not state.auth_maildispatch_endpoint:
        raise RuntimeError("auth.maildispatch.endpoint is required when auth is enabled")
    if not state.auth_maildispatch_api_key:
        raise RuntimeError("auth.maildispatch.api_key is required when auth is enabled")

    base_url = _public_base_url()
    app.secret_key = state.auth_secret_key
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=True,
    )

    store_path = state.auth_store_path.strip()
    if not store_path:
        store_path = str(Path(get_config_dir()) / "maglink.db")
    Path(store_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    store_path = str(Path(store_path).expanduser())

    mailer = HttpMailer(
        endpoint=state.auth_maildispatch_endpoint,
        api_key=state.auth_maildispatch_api_key,
        sender_id=state.auth_maildispatch_sender_id,
        timeout=state.auth_maildispatch_timeout,
    )
    login_core = AuthCore(
        store=SqliteStore(store_path),
        mailer=mailer,
        verify_url_base=f"{base_url}/api/auth/verify",
        identity_provider=RegistryIdentityProvider(state.auth_admin_emails),
        login_sender_id=state.auth_maildispatch_sender_id,
        code_ttl=state.auth_code_ttl,
        rate_max=state.auth_rate_max,
        rate_window=state.auth_rate_window,
        confirm_max_attempts=state.auth_confirm_max_attempts,
    )
    verification_core = EmailVerificationCore(
        store=SqliteStore(store_path),
        mailer=mailer,
        verify_url_base=f"{base_url}/api/register/verify",
        email_allowed=registration_email_allowed,
        login_sender_id=state.auth_maildispatch_sender_id,
        code_ttl=state.auth_code_ttl,
        rate_max=state.auth_rate_max,
        rate_window=state.auth_rate_window,
        confirm_max_attempts=state.auth_confirm_max_attempts,
    )
    auth = EmailAuth(
        login_core,
        require_captcha=True,
        trust_proxy_headers=state.auth_trust_proxy_headers,
    )
    verifier = EmailVerifier(
        verification_core,
        require_captcha=True,
        trust_proxy_headers=state.auth_trust_proxy_headers,
    )
    app.register_blueprint(auth.blueprint(url_prefix="/api/auth"))
    app.register_blueprint(verifier.blueprint(url_prefix="/api/register"))
    app.extensions["email_auth"] = auth
    app.extensions["email_verifier"] = verifier


def get_email_auth(app: Flask) -> Optional[EmailAuth]:
    return app.extensions.get("email_auth")


def get_email_verifier(app: Flask) -> Optional[EmailVerifier]:
    return app.extensions.get("email_verifier")
