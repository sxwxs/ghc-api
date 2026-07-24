import base64
import re

import pytest

from ghc_api import auth as auth_module
from ghc_api.app import create_app
from ghc_api.auth import STATUS_APPROVED, UserRegistry, require_auth
from ghc_api.main import _load_auth_config
from ghc_api.state import state


HTTPS = "https://ghc.example.test"


def _captcha_answer(client, prefix):
    response = client.get(f"{prefix}/captcha", base_url=HTTPS)
    assert response.status_code == 200
    svg_uri = response.get_json()["image"]
    svg = base64.b64decode(svg_uri.split(",", 1)[1]).decode()
    return "".join(re.findall(r">([A-Z2-9])</text>", svg))


def _email_token(mailer):
    body = mailer.last["body"]
    return [line for line in body.splitlines() if "token=" in line][0].split("token=")[1]


@pytest.fixture
def configured_auth(monkeypatch, tmp_path):
    from maglink import ConsoleMailer

    tracked = [
        "enable_auth",
        "enable_email_auth",
        "auth_hostname",
        "auth_secret_key",
        "auth_allow_public_registration",
        "auth_admin_emails",
        "auth_maildispatch_endpoint",
        "auth_maildispatch_api_key",
        "auth_maildispatch_sender_id",
        "auth_maildispatch_timeout",
        "auth_store_path",
        "auth_code_ttl",
        "auth_rate_max",
        "auth_rate_window",
        "auth_confirm_max_attempts",
        "auth_trust_proxy_headers",
    ]
    original = {name: getattr(state, name) for name in tracked}
    state.enable_auth = True
    state.enable_email_auth = True
    state.auth_hostname = "ghc.example.test"
    state.auth_secret_key = "test-session-secret"
    state.auth_allow_public_registration = True
    state.auth_admin_emails = ["admin@example.test"]
    state.auth_maildispatch_endpoint = "https://mail.example.test/api/v1/messages"
    state.auth_maildispatch_api_key = "md_test_secret"
    state.auth_maildispatch_sender_id = "system"
    state.auth_maildispatch_timeout = 10
    state.auth_store_path = str(tmp_path / "maglink.db")
    state.auth_code_ttl = 900
    state.auth_rate_max = 20
    state.auth_rate_window = 900
    state.auth_confirm_max_attempts = 8
    state.auth_trust_proxy_headers = False

    registry_path = tmp_path / "users.json"
    monkeypatch.setattr(auth_module, "_registry_path", lambda: registry_path)
    auth_module._registry = UserRegistry()

    mailer = ConsoleMailer()
    monkeypatch.setattr("ghc_api.email_auth.HttpMailer", lambda **kwargs: mailer)
    app = create_app()
    app.config.update(TESTING=True)
    yield app, mailer, auth_module._registry

    auth_module._registry = None
    for name, value in original.items():
        setattr(state, name, value)


def test_auth_disabled_keeps_local_dashboard_open(monkeypatch):
    original_auth = state.enable_auth
    original_email_auth = state.enable_email_auth
    state.enable_auth = False
    state.enable_email_auth = False
    try:
        app = create_app()
        assert app.test_client().get("/").status_code == 200
    finally:
        state.enable_auth = original_auth
        state.enable_email_auth = original_email_auth


def test_legacy_enable_auth_config_stays_token_only():
    tracked = [
        "enable_auth",
        "enable_email_auth",
        "auth_hostname",
        "auth_secret_key",
        "auth_allow_public_registration",
        "auth_admin_emails",
        "auth_maildispatch_endpoint",
        "auth_maildispatch_api_key",
        "auth_maildispatch_sender_id",
        "auth_maildispatch_timeout",
        "auth_store_path",
        "auth_code_ttl",
        "auth_rate_max",
        "auth_rate_window",
        "auth_confirm_max_attempts",
        "auth_trust_proxy_headers",
    ]
    original = {name: getattr(state, name) for name in tracked}
    try:
        _load_auth_config({"enable_auth": True})
        assert state.enable_auth is True
        assert state.enable_email_auth is False
        assert state.auth_hostname == ""
        assert state.auth_secret_key == ""
        app = create_app()
        assert app.test_client().get("/").status_code == 200
    finally:
        for name, value in original.items():
            setattr(state, name, value)


def test_legacy_token_auth_preserves_signup_and_open_management(monkeypatch, tmp_path):
    original_auth = state.enable_auth
    original_email_auth = state.enable_email_auth
    original_registry = auth_module._registry
    state.enable_auth = True
    state.enable_email_auth = False
    monkeypatch.setattr(auth_module, "_registry_path", lambda: tmp_path / "users.json")
    auth_module._registry = UserRegistry()
    try:
        app = create_app()
        client = app.test_client()
        assert client.get("/").status_code == 200
        assert client.get("/api/users").status_code == 200
        assert client.get("/v1/models").status_code == 401
        assert client.get("/signup").status_code == 200
        created = client.post(
            "/signup", json={"user_id": "legacy-signup", "display_name": "Legacy"}
        )
        assert created.status_code == 201
        assert created.get_json()["token"].startswith("gha_")
    finally:
        state.enable_auth = original_auth
        state.enable_email_auth = original_email_auth
        auth_module._registry = original_registry


def test_dashboard_requires_admin_but_configured_admin_can_login(configured_auth):
    app, mailer, _ = configured_auth
    waiting = app.test_client()
    inbox = app.test_client()

    blocked = waiting.get("/", base_url=HTTPS)
    assert blocked.status_code == 302
    assert "/login" in blocked.headers["Location"]

    captcha = _captcha_answer(waiting, "/api/auth")
    started = waiting.post(
        "/api/auth/request",
        base_url=HTTPS,
        json={"email": "admin@example.test", "captcha": captcha},
    )
    code = started.get_json()["user_code"]
    token = _email_token(mailer)

    assert waiting.get("/api/auth/status", base_url=HTTPS).get_json()["status"] == "pending"
    confirmed = inbox.post(
        "/api/auth/verify/confirm",
        base_url=HTTPS,
        json={"token": token, "user_code": code},
    )
    assert confirmed.status_code == 200
    assert waiting.get("/api/auth/status", base_url=HTTPS).get_json()["status"] == "approved"
    assert waiting.get("/", base_url=HTTPS).status_code == 200


def test_verified_registration_is_pending_until_admin_approval(configured_auth):
    app, mailer, registry = configured_auth
    waiting = app.test_client()
    inbox = app.test_client()

    captcha = _captcha_answer(waiting, "/api/register")
    started = waiting.post(
        "/api/register/request",
        base_url=HTTPS,
        json={"email": "alice@example.test", "captcha": captcha},
    )
    code = started.get_json()["user_code"]
    token = _email_token(mailer)
    inbox.post(
        "/api/register/verify/confirm",
        base_url=HTTPS,
        json={"token": token, "user_code": code},
    )
    assert waiting.get("/api/register/status", base_url=HTTPS).get_json()["status"] == "verified"

    completed = waiting.post(
        "/signup",
        base_url=HTTPS,
        json={"user_id": "alice", "display_name": "Alice"},
    )
    assert completed.status_code == 201
    record = registry.lookup_by_email("alice@example.test")
    assert record.email_verified is True
    assert record.status == "pending"

    with app.test_request_context(
        "/v1/models", headers={"Authorization": f"Bearer {record.token}"}
    ):
        assert require_auth(__import__("flask").request).error_code == "token_pending"

    registry.set_status("alice", STATUS_APPROVED)
    with app.test_request_context(
        "/v1/models", headers={"Authorization": f"Bearer {record.token}"}
    ):
        assert require_auth(__import__("flask").request).user_id == "alice"


def test_registration_validation_error_does_not_consume_verified_email(configured_auth):
    app, mailer, registry = configured_auth
    waiting = app.test_client()
    inbox = app.test_client()
    captcha = _captcha_answer(waiting, "/api/register")
    started = waiting.post(
        "/api/register/request",
        base_url=HTTPS,
        json={"email": "retry@example.test", "captcha": captcha},
    )
    code = started.get_json()["user_code"]
    token = _email_token(mailer)
    inbox.post(
        "/api/register/verify/confirm",
        base_url=HTTPS,
        json={"token": token, "user_code": code},
    )
    assert waiting.get("/api/register/status", base_url=HTTPS).get_json()["status"] == "verified"

    invalid = waiting.post(
        "/signup",
        base_url=HTTPS,
        json={"user_id": "bad user id", "display_name": "Retry"},
    )
    assert invalid.status_code == 400
    assert registry.lookup_by_email("retry@example.test") is None

    corrected = waiting.post(
        "/signup",
        base_url=HTTPS,
        json={"user_id": "retry", "display_name": "Retry"},
    )
    assert corrected.status_code == 201
    assert registry.lookup_by_email("retry@example.test").user_id == "retry"


def test_registration_policy_is_rechecked_at_final_user_creation(configured_auth):
    app, mailer, registry = configured_auth
    waiting = app.test_client()
    inbox = app.test_client()
    captcha = _captcha_answer(waiting, "/api/register")
    started = waiting.post(
        "/api/register/request",
        base_url=HTTPS,
        json={"email": "late@example.test", "captcha": captcha},
    )
    code = started.get_json()["user_code"]
    token = _email_token(mailer)
    inbox.post(
        "/api/register/verify/confirm",
        base_url=HTTPS,
        json={"token": token, "user_code": code},
    )
    assert waiting.get("/api/register/status", base_url=HTTPS).get_json()["status"] == "verified"

    state.auth_allow_public_registration = False
    completed = waiting.post(
        "/signup",
        base_url=HTTPS,
        json={"user_id": "late", "display_name": "Late"},
    )
    assert completed.status_code == 403
    assert completed.get_json()["error"] == "registration_no_longer_allowed"
    assert registry.lookup_by_email("late@example.test") is None


def test_approved_email_user_gets_account_but_not_admin_dashboard(configured_auth):
    app, mailer, registry = configured_auth
    record, error = registry.complete_email_registration(
        "bob", "bob@example.test", "Bob"
    )
    assert error is None
    registry.set_status(record.user_id, STATUS_APPROVED)

    waiting = app.test_client()
    inbox = app.test_client()
    captcha = _captcha_answer(waiting, "/api/auth")
    started = waiting.post(
        "/api/auth/request",
        base_url=HTTPS,
        json={"email": record.email, "captcha": captcha},
    )
    code = started.get_json()["user_code"]
    token = _email_token(mailer)
    inbox.post(
        "/api/auth/verify/confirm",
        base_url=HTTPS,
        json={"token": token, "user_code": code},
    )
    assert waiting.get("/api/auth/status", base_url=HTTPS).get_json()["status"] == "approved"
    assert waiting.get("/account", base_url=HTTPS).status_code == 200
    assert waiting.get("/", base_url=HTTPS).status_code == 302
    assert waiting.get("/api/users", base_url=HTTPS).status_code == 403

    registry.set_status(record.user_id, "revoked")
    assert waiting.get("/account", base_url=HTTPS).status_code == 302


def test_invited_email_token_stays_blocked_until_verification(configured_auth):
    app, _, registry = configured_auth
    record, error = registry.create_email_invitation(
        "invited", "invited@example.test", "Invited"
    )
    assert error is None
    registry.set_status("invited", STATUS_APPROVED)
    with app.test_request_context(
        "/v1/models", headers={"Authorization": f"Bearer {record.token}"}
    ):
        result = require_auth(__import__("flask").request)
    assert result.error_code == "email_unverified"


def test_legacy_user_token_remains_valid(configured_auth):
    app, _, registry = configured_auth
    record, error = registry.create_pending("legacy", "Legacy")
    assert error is None and record.email is None
    registry.set_status("legacy", STATUS_APPROVED)

    with app.test_request_context(
        "/v1/models", headers={"Authorization": f"Bearer {record.token}"}
    ):
        result = require_auth(__import__("flask").request)
    assert result.user_id == "legacy"
