"""Email registration, browser login pages, and user administration."""

from __future__ import annotations

from urllib.parse import urlparse

from flask import (
    Blueprint,
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

from ..auth import (
    STATUS_APPROVED,
    STATUS_REVOKED,
    get_user_registry,
)
from ..email_auth import (
    get_email_auth,
    get_email_verifier,
    registration_email_allowed,
)
from ..state import state


auth_bp = Blueprint("auth", __name__)


def _payload():
    if request.is_json:
        return request.get_json(silent=True) or {}
    return request.form


def _safe_next(value: str) -> str:
    value = (value or "").strip()
    parsed = urlparse(value)
    if value.startswith("/") and not value.startswith("//") and not parsed.netloc:
        return value
    return url_for("dashboard.index")


@auth_bp.get("/login")
def login_page():
    if not state.enable_email_auth:
        return redirect(url_for("dashboard.index"))
    auth = get_email_auth(current_app)
    user = auth.current_user() if auth is not None else None
    next_url = _safe_next(request.args.get("next", ""))
    if user is not None:
        return redirect(next_url if user.get("is_admin") else url_for("auth.account_page"))
    return render_template("login.html", next_url=next_url)


@auth_bp.get("/account")
def account_page():
    if not state.enable_email_auth:
        return redirect(url_for("dashboard.index"))
    auth = get_email_auth(current_app)
    user = auth.current_user() if auth is not None else None
    if user is None:
        return redirect(url_for("auth.login_page", next=request.path))
    record = get_user_registry().lookup_by_email(user["email"])
    return render_template("account.html", user=user, record=record)


@auth_bp.route("/signup", methods=["GET"])
def signup_page():
    if not state.enable_email_auth:
        return render_template("signup_legacy.html")
    return render_template(
        "signup.html",
        allow_public_registration=state.auth_allow_public_registration,
    )


@auth_bp.route("/signup", methods=["POST"])
def signup_submit():
    if not state.enable_email_auth:
        payload = _payload()
        record, err = get_user_registry().create_pending(
            payload.get("user_id") or "",
            payload.get("display_name") or "",
        )
        if err is not None:
            return jsonify({"error": "signup_failed", "message": err}), 400
        return jsonify({
            "user_id": record.user_id,
            "display_name": record.display_name,
            "token": record.token,
            "status": record.status,
            "message": "Account registered. Your token will be usable once an administrator approves it.",
        }), 201

    verifier = get_email_verifier(current_app)
    if verifier is None:
        return jsonify({"error": "auth_not_configured"}), 503
    email = verifier.verified_email()
    if email is None:
        return jsonify({
            "error": "email_verification_required",
            "message": "Complete email verification in this browser first.",
        }), 403

    # Re-check the current registration policy at the final mutation boundary.
    # The invitation/public-registration setting may have changed after mail
    # delivery or after the waiting browser observed a verified status.
    if not registration_email_allowed(email):
        return jsonify({
            "error": "registration_no_longer_allowed",
            "message": "Registration eligibility changed. Request a new verification if invited again.",
        }), 403

    payload = _payload()
    registry = get_user_registry()
    invited = registry.lookup_by_email(email)
    record, err = registry.complete_email_registration(
        payload.get("user_id") or (invited.user_id if invited else ""),
        email,
        payload.get("display_name") or (invited.display_name if invited else ""),
    )
    if err is not None:
        # Keep the verified email in the signed waiting-browser session so the
        # user can correct user_id/display-name conflicts without another mail.
        return jsonify({"error": "signup_failed", "message": err}), 400

    verifier.consume_verified_email()
    return jsonify({
        "user_id": record.user_id,
        "email": record.email,
        "display_name": record.display_name,
        "token": record.token,
        "status": record.status,
        "email_verified": record.email_verified,
        "message": "Email verified. An administrator must approve the account before login or API use.",
    }), 201


@auth_bp.route("/api/users", methods=["GET"])
def list_users():
    records = get_user_registry().list_all()
    return jsonify({"users": [record.to_public_dict() for record in records]})


@auth_bp.route("/api/users", methods=["POST"])
def invite_user():
    if not state.enable_email_auth:
        return jsonify({"error": "email_auth_disabled"}), 404
    payload = _payload()
    record, err = get_user_registry().create_email_invitation(
        payload.get("user_id") or "",
        payload.get("email") or "",
        payload.get("display_name") or "",
    )
    if err is not None:
        return jsonify({"error": "invite_failed", "message": err}), 400
    return jsonify({"user": record.to_public_dict()}), 201


@auth_bp.route("/api/users/<user_id>/approve", methods=["POST"])
def approve_user(user_id: str):
    record, err = get_user_registry().set_status(user_id, STATUS_APPROVED)
    if err is not None:
        return jsonify({"error": "approve_failed", "message": err}), 404 if "not found" in err else 400
    return jsonify({"user": record.to_public_dict()})


@auth_bp.route("/api/users/<user_id>/revoke", methods=["POST"])
def revoke_user(user_id: str):
    record, err = get_user_registry().set_status(user_id, STATUS_REVOKED)
    if err is not None:
        return jsonify({"error": "revoke_failed", "message": err}), 404 if "not found" in err else 400
    return jsonify({"user": record.to_public_dict()})


@auth_bp.route("/api/users/<user_id>", methods=["DELETE"])
def delete_user(user_id: str):
    ok, err = get_user_registry().delete(user_id)
    if not ok:
        return jsonify({"error": "delete_failed", "message": err}), 404
    return jsonify({"deleted": user_id})
