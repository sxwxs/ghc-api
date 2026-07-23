"""
Dashboard and API monitoring routes
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List

from flask import Blueprint, Response, jsonify, render_template, request

from ..anthropic_responses import VALID_MODES, WIRE_PROFILES
from ..cache import cache
from ..counters import counters
from ..config import chat_completions_model_support, model_mappings
from ..config_sync import (
    get_software_versions,
    get_config_hash_overview,
    get_sync_status,
    install_code_agents,
    sync_local_to_onedrive,
    sync_onedrive_to_local,
)
from ..auth import ANONYMOUS_USER_ID, get_user_registry
from ..state import state
from ..token_usage_reporter import get_token_usage_overview
from ..token_manager import (
    get_token_file_path,
    github_device_flow_manager,
)
from ..request_file_stats import (
    MAX_BUCKET_PAGE_SIZE,
    MAX_DETAIL_LINE_BYTES,
    RequestFileChangedError,
    RequestStatsBusyError,
    RequestIndexValidationError,
    RequestStatsDatasetNotFound,
    RequestStatsJobNotFound,
    RequestStatsValidationError,
    list_request_files,
    read_request_detail,
    request_stats_jobs,
)

dashboard_bp = Blueprint('dashboard', __name__)

ALLOWED_ACCOUNT_TYPES = {"individual", "business", "enterprise"}


def _runtime_config() -> Dict[str, Any]:
    return {
        "account_type": state.account_type,
        "vscode_version": state.vscode_version,
        "api_version": state.api_version,
        "copilot_version": state.copilot_version,
        "system_prompt_remove": state.system_prompt_remove,
        "tool_result_suffix_remove": state.tool_result_suffix_remove,
        "system_prompt_add": state.system_prompt_add,
        "max_connection_retries": state.max_connection_retries,
        "upstream_read_timeout": state.upstream_read_timeout,
        "sse_keepalive_interval": state.sse_keepalive_interval,
        "auto_remove_encrypted_content_on_parse_error": state.auto_remove_encrypted_content_on_parse_error,
        "save_request_to_file": state.save_request_to_file,
        "disable_onedrive_access": state.disable_onedrive_access,
        "enable_auth": state.enable_auth,
        "anthropic_responses_compat_enabled": state.anthropic_responses_compat_enabled,
        "anthropic_responses_compat_mode": state.anthropic_responses_compat_mode,
        "anthropic_responses_wire_profile": state.anthropic_responses_wire_profile,
        "anthropic_responses_model_profiles": state.anthropic_responses_model_profiles,
        "anthropic_responses_replay_path": state.anthropic_responses_replay_path,
        "anthropic_responses_replay_ttl_seconds": state.anthropic_responses_replay_ttl_seconds,
        "anthropic_responses_replay_max_bytes": state.anthropic_responses_replay_max_bytes,
        "anthropic_responses_replay_max_tenant_bytes": state.anthropic_responses_replay_max_tenant_bytes,
        "anthropic_responses_replay_max_record_bytes": state.anthropic_responses_replay_max_record_bytes,
        "anthropic_responses_replay_encryption_key_env": state.anthropic_responses_replay_encryption_key_env,
        "anthropic_responses_replay_require_trusted_tenant": state.anthropic_responses_replay_require_trusted_tenant,
        "anthropic_responses_replay_trusted_single_user": state.anthropic_responses_replay_trusted_single_user,
        "model_mappings": {
            "exact": model_mappings.exact_mappings,
            "prefix": model_mappings.prefix_mappings,
        },
        "chat_completions_model_support": {
            "exact": chat_completions_model_support.exact_model_names,
            "prefix": chat_completions_model_support.prefix_model_names,
        },
    }


def _user_filter_from_request() -> str | None:
    """Read the optional ?user=<id> query parameter. Empty / 'all' / 'any' = no filter."""
    raw = request.args.get("user")
    if raw is None:
        return None
    raw = raw.strip()
    if not raw or raw.lower() in ("all", "any", "*"):
        return None
    return raw


def _validate_string_list(value: Any, field_name: str) -> List[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"'{field_name}' must be a list of strings")
    return value


def _validate_mapping(value: Any, field_name: str) -> Dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"'{field_name}' must be an object")
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in value.items()):
        raise ValueError(f"'{field_name}' values must be string-to-string pairs")
    return value


def _validate_endpoint_support(value: Any, field_name: str) -> tuple[List[str], List[str]]:
    if not isinstance(value, dict):
        raise ValueError(f"'{field_name}' must be an object")
    exact = _validate_string_list(value.get("exact", []), f"{field_name}.exact")
    prefix = _validate_string_list(value.get("prefix", []), f"{field_name}.prefix")
    return exact, prefix


def _validate_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"'{field_name}' must be a boolean")
    return value


def _validate_string(value: Any, field_name: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f"'{field_name}' must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"'{field_name}' must not be empty")
    return value


def _validate_positive_integer(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"'{field_name}' must be an integer > 0")
    return value


def _validate_wire_profile(value: Any, field_name: str) -> str:
    profile = _validate_string(value, field_name, allow_empty=False)
    if profile not in WIRE_PROFILES:
        choices = ", ".join(sorted(WIRE_PROFILES))
        raise ValueError(f"'{field_name}' must be one of: {choices}")
    return profile


def _validate_model_profiles(value: Any) -> Dict[str, str]:
    profiles = _validate_mapping(value, "anthropic_responses_model_profiles")
    for model, profile in profiles.items():
        if not model.strip():
            raise ValueError("'anthropic_responses_model_profiles' model names must not be empty")
        _validate_wire_profile(profile, f"anthropic_responses_model_profiles.{model}")
    return dict(profiles)


@dashboard_bp.route("/", methods=["GET"])
def index():
    """Serve the dashboard"""
    return render_template("dashboard.html")


@dashboard_bp.route("/requests", methods=["GET"])
def requests_page():
    """Serve the requests browser page"""
    return render_template("requests.html")


@dashboard_bp.route("/chat", methods=["GET"])
def chat_page():
    """Serve the chat page"""
    return render_template("chat.html")


@dashboard_bp.route("/code-agent-manager", methods=["GET"])
def code_agent_manager_page():
    """Serve the code agent manager page."""
    return render_template("code_agent_manager.html")


@dashboard_bp.route("/code-agent-manager/config", methods=["GET"])
def code_agent_manager_config_page():
    """Serve code-agent installation and config-sync details."""
    return render_template("code_agent_manager_config.html")


@dashboard_bp.route("/request-stats", methods=["GET"])
def request_stats_page():
    """Serve the persisted request-file statistics page."""
    return render_template("request_stats.html")


@dashboard_bp.route("/request-file-detail", methods=["GET"])
def request_file_detail_page():
    """Serve one stable request-file detail view."""
    return render_template("request_file_detail.html")


@dashboard_bp.route("/api/runtime-config", methods=["GET"])
def api_runtime_config():
    """Get current in-memory runtime configuration"""
    return jsonify(_runtime_config())


@dashboard_bp.route("/api/runtime-config", methods=["POST"])
def api_runtime_config_update():
    """Update in-memory runtime configuration only"""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid JSON body"}), 400

    allowed_keys = {
        "account_type",
        "vscode_version",
        "api_version",
        "copilot_version",
        "system_prompt_remove",
        "tool_result_suffix_remove",
        "system_prompt_add",
        "max_connection_retries",
        "upstream_read_timeout",
        "sse_keepalive_interval",
        "auto_remove_encrypted_content_on_parse_error",
        "save_request_to_file",
        "disable_onedrive_access",
        "anthropic_responses_compat_enabled",
        "anthropic_responses_compat_mode",
        "anthropic_responses_wire_profile",
        "anthropic_responses_model_profiles",
        "anthropic_responses_replay_path",
        "anthropic_responses_replay_ttl_seconds",
        "anthropic_responses_replay_max_bytes",
        "anthropic_responses_replay_max_tenant_bytes",
        "anthropic_responses_replay_max_record_bytes",
        "anthropic_responses_replay_encryption_key_env",
        "anthropic_responses_replay_require_trusted_tenant",
        "anthropic_responses_replay_trusted_single_user",
        "model_mappings",
        "chat_completions_model_support",
    }
    unknown_keys = sorted(set(payload.keys()) - allowed_keys)
    if unknown_keys:
        return jsonify({"error": f"Unknown config key(s): {', '.join(unknown_keys)}"}), 400

    try:
        proposed_total = _validate_positive_integer(
            payload.get(
                "anthropic_responses_replay_max_bytes",
                state.anthropic_responses_replay_max_bytes,
            ),
            "anthropic_responses_replay_max_bytes",
        )
        proposed_tenant = _validate_positive_integer(
            payload.get(
                "anthropic_responses_replay_max_tenant_bytes",
                state.anthropic_responses_replay_max_tenant_bytes,
            ),
            "anthropic_responses_replay_max_tenant_bytes",
        )
        proposed_record = _validate_positive_integer(
            payload.get(
                "anthropic_responses_replay_max_record_bytes",
                state.anthropic_responses_replay_max_record_bytes,
            ),
            "anthropic_responses_replay_max_record_bytes",
        )
        if proposed_record > proposed_tenant:
            raise ValueError(
                "'anthropic_responses_replay_max_record_bytes' must not exceed "
                "'anthropic_responses_replay_max_tenant_bytes'"
            )
        if proposed_tenant > proposed_total:
            raise ValueError(
                "'anthropic_responses_replay_max_tenant_bytes' must not exceed "
                "'anthropic_responses_replay_max_bytes'"
            )

        if "account_type" in payload:
            account_type = payload["account_type"]
            if not isinstance(account_type, str) or account_type not in ALLOWED_ACCOUNT_TYPES:
                raise ValueError("'account_type' must be one of: individual, business, enterprise")
            state.account_type = account_type

        if "vscode_version" in payload:
            if not isinstance(payload["vscode_version"], str):
                raise ValueError("'vscode_version' must be a string")
            state.vscode_version = payload["vscode_version"]

        if "api_version" in payload:
            if not isinstance(payload["api_version"], str):
                raise ValueError("'api_version' must be a string")
            state.api_version = payload["api_version"]

        if "copilot_version" in payload:
            if not isinstance(payload["copilot_version"], str):
                raise ValueError("'copilot_version' must be a string")
            state.copilot_version = payload["copilot_version"]

        if "system_prompt_remove" in payload:
            state.system_prompt_remove = _validate_string_list(payload["system_prompt_remove"], "system_prompt_remove")

        if "tool_result_suffix_remove" in payload:
            state.tool_result_suffix_remove = _validate_string_list(payload["tool_result_suffix_remove"], "tool_result_suffix_remove")

        if "system_prompt_add" in payload:
            state.system_prompt_add = _validate_string_list(payload["system_prompt_add"], "system_prompt_add")

        if "max_connection_retries" in payload:
            retries = payload["max_connection_retries"]
            if not isinstance(retries, int) or retries < 0:
                raise ValueError("'max_connection_retries' must be an integer >= 0")
            state.max_connection_retries = retries

        if "upstream_read_timeout" in payload:
            timeout = payload["upstream_read_timeout"]
            if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout < 0:
                raise ValueError("'upstream_read_timeout' must be an integer >= 0")
            state.upstream_read_timeout = timeout

        if "sse_keepalive_interval" in payload:
            interval = payload["sse_keepalive_interval"]
            if not isinstance(interval, int) or isinstance(interval, bool) or interval < 0:
                raise ValueError("'sse_keepalive_interval' must be an integer >= 0")
            state.sse_keepalive_interval = interval

        if "auto_remove_encrypted_content_on_parse_error" in payload:
            flag = payload["auto_remove_encrypted_content_on_parse_error"]
            if not isinstance(flag, bool):
                raise ValueError("'auto_remove_encrypted_content_on_parse_error' must be a boolean")
            state.auto_remove_encrypted_content_on_parse_error = flag

        if "save_request_to_file" in payload:
            save_req = payload["save_request_to_file"]
            if not isinstance(save_req, bool):
                raise ValueError("'save_request_to_file' must be a boolean")
            state.save_request_to_file = save_req

        if "disable_onedrive_access" in payload:
            disable_onedrive = payload["disable_onedrive_access"]
            if not isinstance(disable_onedrive, bool):
                raise ValueError("'disable_onedrive_access' must be a boolean")
            state.disable_onedrive_access = disable_onedrive

        if "anthropic_responses_compat_enabled" in payload:
            state.anthropic_responses_compat_enabled = _validate_bool(
                payload["anthropic_responses_compat_enabled"],
                "anthropic_responses_compat_enabled",
            )

        if "anthropic_responses_compat_mode" in payload:
            mode = _validate_string(
                payload["anthropic_responses_compat_mode"],
                "anthropic_responses_compat_mode",
                allow_empty=False,
            )
            if mode not in VALID_MODES:
                choices = ", ".join(sorted(VALID_MODES))
                raise ValueError(f"'anthropic_responses_compat_mode' must be one of: {choices}")
            state.anthropic_responses_compat_mode = mode

        if "anthropic_responses_wire_profile" in payload:
            state.anthropic_responses_wire_profile = _validate_wire_profile(
                payload["anthropic_responses_wire_profile"],
                "anthropic_responses_wire_profile",
            )

        if "anthropic_responses_model_profiles" in payload:
            state.anthropic_responses_model_profiles = _validate_model_profiles(
                payload["anthropic_responses_model_profiles"]
            )

        if "anthropic_responses_replay_path" in payload:
            state.anthropic_responses_replay_path = _validate_string(
                payload["anthropic_responses_replay_path"],
                "anthropic_responses_replay_path",
            )

        if "anthropic_responses_replay_ttl_seconds" in payload:
            state.anthropic_responses_replay_ttl_seconds = _validate_positive_integer(
                payload["anthropic_responses_replay_ttl_seconds"],
                "anthropic_responses_replay_ttl_seconds",
            )

        if "anthropic_responses_replay_max_bytes" in payload:
            state.anthropic_responses_replay_max_bytes = _validate_positive_integer(
                payload["anthropic_responses_replay_max_bytes"],
                "anthropic_responses_replay_max_bytes",
            )

        if "anthropic_responses_replay_max_tenant_bytes" in payload:
            state.anthropic_responses_replay_max_tenant_bytes = _validate_positive_integer(
                payload["anthropic_responses_replay_max_tenant_bytes"],
                "anthropic_responses_replay_max_tenant_bytes",
            )

        if "anthropic_responses_replay_max_record_bytes" in payload:
            state.anthropic_responses_replay_max_record_bytes = _validate_positive_integer(
                payload["anthropic_responses_replay_max_record_bytes"],
                "anthropic_responses_replay_max_record_bytes",
            )

        if "anthropic_responses_replay_encryption_key_env" in payload:
            state.anthropic_responses_replay_encryption_key_env = _validate_string(
                payload["anthropic_responses_replay_encryption_key_env"],
                "anthropic_responses_replay_encryption_key_env",
            )

        if "anthropic_responses_replay_require_trusted_tenant" in payload:
            state.anthropic_responses_replay_require_trusted_tenant = _validate_bool(
                payload["anthropic_responses_replay_require_trusted_tenant"],
                "anthropic_responses_replay_require_trusted_tenant",
            )

        if "anthropic_responses_replay_trusted_single_user" in payload:
            state.anthropic_responses_replay_trusted_single_user = _validate_bool(
                payload["anthropic_responses_replay_trusted_single_user"],
                "anthropic_responses_replay_trusted_single_user",
            )

        if "model_mappings" in payload:
            mappings = payload["model_mappings"]
            if not isinstance(mappings, dict):
                raise ValueError("'model_mappings' must be an object")
            exact = _validate_mapping(mappings.get("exact", {}), "model_mappings.exact")
            prefix = _validate_mapping(mappings.get("prefix", {}), "model_mappings.prefix")
            model_mappings.exact_mappings = exact
            model_mappings.prefix_mappings = prefix

        if "chat_completions_model_support" in payload:
            exact, prefix = _validate_endpoint_support(
                payload["chat_completions_model_support"],
                "chat_completions_model_support",
            )
            chat_completions_model_support.exact_model_names = exact
            chat_completions_model_support.prefix_model_names = prefix

            if state.models:
                from ..api_helpers import apply_configured_chat_completions_support

                apply_configured_chat_completions_support(state.models)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({
        "message": "Runtime configuration updated. This does not modify config.yaml.",
        "config": _runtime_config(),
    })


@dashboard_bp.route("/api/config-manager/token-status", methods=["GET"])
def api_config_manager_token_status():
    """Return Copilot refresh state and the active web Device Flow session."""
    return jsonify({
        "github_token_configured": bool(state.github_token),
        "github_token_source": state.github_token_source,
        "local_token_file_exists": os.path.exists(get_token_file_path()),
        "copilot_token_configured": bool(state.copilot_token),
        "copilot_token_expires_at": state.token_expires_at or None,
        "last_refresh_attempt_at": state.token_refresh_last_attempt_at,
        "last_refresh_success_at": state.token_refresh_last_success_at,
        "last_refresh_succeeded": state.token_refresh_last_succeeded,
        "last_refresh_error": state.token_refresh_last_error,
        "device_flow": github_device_flow_manager.status(),
    })


@dashboard_bp.route("/api/config-manager/github-device-login", methods=["POST"])
def api_config_manager_github_device_login():
    """Start a non-blocking GitHub Device Flow session for the manager UI."""
    try:
        return jsonify(github_device_flow_manager.start()), 202
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502


@dashboard_bp.route("/api/config-manager/status", methods=["GET"])
def api_config_manager_status():
    """Get OneDrive sync and local config diff status."""
    return jsonify(get_sync_status())


@dashboard_bp.route("/api/config-manager/install-tools", methods=["POST"])
def api_config_manager_install_tools():
    """Install Codex, Claude Code, and Copilot CLI via npm."""
    result = install_code_agents()
    status_code = 200 if result.get("ok") else 500
    return jsonify(result), status_code


@dashboard_bp.route("/api/config-manager/sync-to-onedrive", methods=["POST"])
def api_config_manager_sync_to_onedrive():
    """Copy local config files to OneDrive sync folder."""
    result = sync_local_to_onedrive()
    status_code = 200 if result.get("ok") else 400
    return jsonify(result), status_code


@dashboard_bp.route("/api/config-manager/sync-from-onedrive", methods=["POST"])
def api_config_manager_sync_from_onedrive():
    """Copy synced config files from OneDrive to local machine with backups."""
    result = sync_onedrive_to_local()
    status_code = 200 if result.get("ok") else 400
    return jsonify(result), status_code


@dashboard_bp.route("/api/config-manager/token-usage", methods=["GET"])
def api_config_manager_token_usage():
    """Get token usage overview grouped by machine, model, and (optionally) user."""
    range_key = request.args.get("range", "week")
    if range_key not in {"all", "hour", "day", "week", "month"}:
        return jsonify({"error": "Invalid range. Use: all, hour, day, week, month"}), 400
    user_filter = _user_filter_from_request()
    return jsonify(get_token_usage_overview(range_key, user_filter=user_filter))


@dashboard_bp.route("/api/config-manager/config-hashes", methods=["GET"])
def api_config_manager_config_hashes():
    """Get config hash values and creation times across machines."""
    return jsonify(get_config_hash_overview())


@dashboard_bp.route("/api/config-manager/software-versions", methods=["GET"])
def api_config_manager_software_versions():
    """Get software install/version status for common code agent tools."""
    return jsonify(get_software_versions())


@dashboard_bp.route("/api/stats", methods=["GET"])
def api_stats():
    """Get API statistics, optionally filtered to a single user via ?user=<id>."""
    stats = cache.get_stats(user_id=_user_filter_from_request())
    stats["counters"] = counters.snapshot()
    return jsonify(stats)


@dashboard_bp.route("/api/request-stats/files", methods=["GET"])
def api_request_stats_files():
    """List persisted request files and their request-statistics index state."""
    return jsonify({"files": list_request_files(request_stats_jobs.active_files())})


@dashboard_bp.route("/api/request-stats/jobs", methods=["POST"])
def api_request_stats_jobs_create():
    """Start an asynchronous statistics job for selected request files."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid JSON body"}), 400
    try:
        job, reused = request_stats_jobs.start(payload.get("files"))
    except RequestStatsValidationError as exc:
        return jsonify({"error": str(exc)}), 400
    except RequestStatsBusyError as exc:
        return jsonify({
            "error": "busy",
            "message": str(exc),
            "active_job_id": exc.active_job_id,
        }), 409
    return jsonify({"job": job, "reused": reused}), 200 if reused else 202


@dashboard_bp.route("/api/request-stats/jobs/<job_id>", methods=["GET"])
def api_request_stats_job(job_id: str):
    """Get progress or the completed result for one statistics job."""
    try:
        return jsonify({"job": request_stats_jobs.get(job_id)})
    except RequestStatsJobNotFound:
        return jsonify({"error": "Request statistics job not found"}), 404


@dashboard_bp.route("/api/request-stats/jobs/<job_id>/cancel", methods=["POST"])
def api_request_stats_job_cancel(job_id: str):
    """Request cooperative cancellation of a running statistics job."""
    try:
        return jsonify({"job": request_stats_jobs.cancel(job_id)})
    except RequestStatsJobNotFound:
        return jsonify({"error": "Request statistics job not found"}), 404


@dashboard_bp.route("/api/request-stats/datasets/<dataset_id>/requests", methods=["GET"])
def api_request_stats_dataset_requests(dataset_id: str):
    """List requests that contributed to one histogram bucket."""
    metric = request.args.get("metric", "")
    view = request.args.get("view", "overall")
    value = request.args.get("value")
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)
    bucket = request.args.get("bucket", type=int)
    if page < 1 or per_page < 1 or per_page > MAX_BUCKET_PAGE_SIZE or bucket is None:
        return jsonify({"error": f"Invalid pagination or bucket; per_page must be 1-{MAX_BUCKET_PAGE_SIZE}"}), 400
    try:
        result = request_stats_jobs.query_dataset(
            dataset_id,
            metric=metric,
            bucket=bucket,
            view=view,
            value=value,
            page=page,
            per_page=per_page,
        )
    except RequestStatsValidationError as exc:
        return jsonify({"error": str(exc)}), 400
    except RequestStatsDatasetNotFound:
        return jsonify({"error": "Request statistics dataset expired; generate statistics again"}), 410
    return jsonify(result)


@dashboard_bp.route("/api/request-stats/request-detail", methods=["GET"])
def api_request_stats_request_detail():
    """Read one indexed request line from its source JSONL file."""
    filename = request.args.get("file", "")
    sha256 = request.args.get("sha256", "")
    offset = request.args.get("offset", type=int)
    length = request.args.get("length", type=int)
    if offset is None or length is None:
        return jsonify({"error": "offset and length are required integers"}), 400
    try:
        return jsonify(read_request_detail(filename, offset, length, sha256))
    except RequestStatsValidationError as exc:
        status_code = 413 if "display limit" in str(exc) else 400
        return jsonify({"error": str(exc), "max_detail_bytes": MAX_DETAIL_LINE_BYTES}), status_code
    except RequestIndexValidationError as exc:
        return jsonify({"error": str(exc)}), 409
    except RequestFileChangedError as exc:
        return jsonify({"error": str(exc)}), 409


@dashboard_bp.route("/api/users-list", methods=["GET"])
def api_users_list():
    """List user_ids known to this instance. Combines registered users with
    user_ids that appear in the request cache (e.g. 'anonymous'). Used by the
    dashboard's filter-by-user dropdown — no tokens returned."""
    seen: Dict[str, Dict[str, Any]] = {}

    for record in get_user_registry().list_all():
        seen[record.user_id] = record.to_safe_dict()

    for user_id in cache.list_user_ids():
        if user_id not in seen:
            seen[user_id] = {
                "user_id": user_id,
                "display_name": user_id,
                "status": "transient" if user_id == ANONYMOUS_USER_ID else "unregistered",
                "created_at": 0,
                "approved_at": None,
            }

    users = sorted(seen.values(), key=lambda u: (u["user_id"] != ANONYMOUS_USER_ID, u["user_id"].lower()))
    return jsonify({"users": users})


@dashboard_bp.route("/api/requests", methods=["GET"])
def api_requests():
    """Get paginated list of requests, optionally filtered by ?user=<id>."""
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)
    search = request.args.get("search", "")
    user_filter = _user_filter_from_request()

    offset = (page - 1) * per_page

    if search:
        items = cache.search_requests(search, per_page, offset, user_id=user_filter)
        total = len(cache.search_requests(search, 10000, 0, user_id=user_filter))  # Get total count
    else:
        items = cache.get_recent_requests(per_page, offset, user_id=user_filter)
        total = cache.get_total_count(user_id=user_filter)

    # Remove large body content for list view
    items_summary = []
    for item in items:
        summary = dict(item)
        summary.pop("original_request_body", None)
        summary.pop("request_body", None)
        summary.pop("response_body", None)
        summary.pop("raw_events", None)
        summary.pop("request_headers", None)
        items_summary.append(summary)

    return jsonify({
        "items": items_summary,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
        "user_filter": user_filter,
    })


@dashboard_bp.route("/api/request/<request_id>", methods=["GET"])
def api_request_detail(request_id: str):
    """Get detailed request/response data"""
    item = cache.get_request(request_id)
    if not item:
        return jsonify({"error": "Request not found"}), 404
    return jsonify(item)


@dashboard_bp.route("/api/request/<request_id>/request-body", methods=["GET"])
def api_request_body(request_id: str):
    """Get just the request body"""
    item = cache.get_request(request_id)
    if not item:
        return jsonify({"error": "Request not found"}), 404
    return jsonify(item.get("request_body"))


@dashboard_bp.route("/api/request/<request_id>/response-body", methods=["GET"])
def api_response_body(request_id: str):
    """Get just the response body.

    For streaming entries written by the new SSE handlers, the cache stores the
    raw upstream SSE ``data:`` payloads under ``raw_events`` instead of a
    reconstructed ``response_body`` dict. Return either, with a discriminator
    so the dashboard knows which view to render.
    """
    item = cache.get_request(request_id)
    if not item:
        return jsonify({"error": "Request not found"}), 404
    if item.get("raw_events") is not None:
        return jsonify({"raw_events": item.get("raw_events")})
    return jsonify(item.get("response_body"))


@dashboard_bp.route("/api/requests/search", methods=["GET"])
def api_fulltext_search():
    """Full-text search in request/response bodies, optionally filtered by ?user=<id>."""
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)
    query = request.args.get("q", "")
    user_filter = _user_filter_from_request()

    if not query:
        return jsonify({
            "items": [],
            "total": 0,
            "page": page,
            "per_page": per_page,
            "total_pages": 0,
            "user_filter": user_filter,
        })

    offset = (page - 1) * per_page
    items, total = cache.fulltext_search(query, per_page, offset, user_id=user_filter)

    # Remove large body content for list view
    items_summary = []
    for item in items:
        summary = dict(item)
        summary.pop("original_request_body", None)
        summary.pop("request_body", None)
        summary.pop("response_body", None)
        summary.pop("raw_events", None)
        summary.pop("request_headers", None)
        items_summary.append(summary)

    return jsonify({
        "items": items_summary,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page if total > 0 else 0,
        "user_filter": user_filter,
    })


@dashboard_bp.route("/api/requests/export", methods=["GET"])
def api_export_requests():
    """Export all requests as JSON Lines (.jl) file"""
    def generate():
        for item in cache.get_all_requests():
            yield cache.format_request_jsonl_line(item)

    return Response(
        generate(),
        mimetype="application/x-jsonlines",
        headers={
            "Content-Disposition": f"attachment; filename=requests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jl",
        },
    )


@dashboard_bp.route("/api/requests/import", methods=["POST"])
def api_import_requests():
    """Import requests from JSON Lines (.jl) file"""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    imported_count = 0
    errors = []

    try:
        for line_num, line in enumerate(file.stream, 1):
            line = line.decode("utf-8").strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                cache.import_request(data)
                imported_count += 1
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {str(e)}")
            except Exception as e:
                errors.append(f"Line {line_num}: {str(e)}")

    except Exception as e:
        return jsonify({"error": f"Failed to read file: {str(e)}"}), 500

    return jsonify({
        "imported": imported_count,
        "errors": errors[:10] if errors else [],  # Limit errors to first 10
        "total_errors": len(errors),
    })
