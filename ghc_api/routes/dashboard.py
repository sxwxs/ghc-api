"""
Dashboard and API monitoring routes
"""

import os
import time
from typing import Any, Dict, List

from flask import Blueprint, jsonify, render_template, request

from ..anthropic_responses import WIRE_PROFILES
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
from ..api_helpers import (
    get_copilot_base_url,
    get_github_api_base_url,
    get_github_web_base_url,
)
from ..auth import ANONYMOUS_USER_ID, get_user_registry
from ..perf_stats import WINDOW_PRESETS, parse_window, perf_stats
from ..request_log import RequestLogError, request_log, validate_retention_days
from ..state import state
from ..webiq import is_configured as webiq_is_configured, tool_definition as webiq_tool_definition
from ..anysearch import is_configured as anysearch_is_configured, tool_definitions as anysearch_tool_definitions
from ..token_usage_reporter import get_token_usage_overview
from .. import billing
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
        "request_log_retention_days": state.request_log_retention_days,
        "log_webiq_requests": state.log_webiq_requests,
        "log_anysearch_requests": state.log_anysearch_requests,
        "disable_onedrive_access": state.disable_onedrive_access,
        "enable_auth": state.enable_auth,
        "anthropic_responses_compat_enabled": state.anthropic_responses_compat_enabled,
        "anthropic_responses_wire_profile": state.anthropic_responses_wire_profile,
        "anthropic_responses_model_profiles": state.anthropic_responses_model_profiles,
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
    """Serve the chat page."""
    return render_template(
        "chat.html",
        webiq_available=webiq_is_configured(state),
        webiq_tool_chat=webiq_tool_definition("chat"),
        anysearch_available=anysearch_is_configured(state),
        anysearch_tools_chat=anysearch_tool_definitions("chat"),
    )


@dashboard_bp.route("/code-agent-manager", methods=["GET"])
def code_agent_manager_page():
    """Serve the code agent manager page."""
    return render_template("code_agent_manager.html")


@dashboard_bp.route("/code-agent-manager/config", methods=["GET"])
def code_agent_manager_config_page():
    """Serve code-agent installation and config-sync details."""
    return render_template("code_agent_manager_config.html")


@dashboard_bp.route("/live", methods=["GET"])
def live_stats_page():
    """Serve the live throughput (tokens/second) dashboard."""
    return render_template("live_stats.html")


@dashboard_bp.route("/request-stats", methods=["GET"])
def request_stats_page():
    """Serve the persisted request-file statistics page."""
    return render_template("request_stats.html")


@dashboard_bp.route("/request-perf", methods=["GET"])
def request_perf_page():
    """Serve historical performance analysis over the request-file indexes."""
    return render_template("request_perf.html")


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
        "request_log_retention_days",
        "log_webiq_requests",
        "log_anysearch_requests",
        "disable_onedrive_access",
        "anthropic_responses_compat_enabled",
        "anthropic_responses_wire_profile",
        "anthropic_responses_model_profiles",
        "model_mappings",
        "chat_completions_model_support",
    }
    unknown_keys = sorted(set(payload.keys()) - allowed_keys)
    if unknown_keys:
        return jsonify({"error": f"Unknown config key(s): {', '.join(unknown_keys)}"}), 400

    try:
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

        if "request_log_retention_days" in payload:
            state.request_log_retention_days = validate_retention_days(
                payload["request_log_retention_days"])

        if "log_webiq_requests" in payload:
            log_webiq = payload["log_webiq_requests"]
            if not isinstance(log_webiq, bool):
                raise ValueError("'log_webiq_requests' must be a boolean")
            state.log_webiq_requests = log_webiq
        if "log_anysearch_requests" in payload:
            state.log_anysearch_requests = _validate_bool(
                payload["log_anysearch_requests"], "log_anysearch_requests")

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

        if "anthropic_responses_wire_profile" in payload:
            state.anthropic_responses_wire_profile = _validate_wire_profile(
                payload["anthropic_responses_wire_profile"],
                "anthropic_responses_wire_profile",
            )

        if "anthropic_responses_model_profiles" in payload:
            state.anthropic_responses_model_profiles = _validate_model_profiles(
                payload["anthropic_responses_model_profiles"]
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
    github_oauth_base_url = None
    github_device_flow_error = None
    try:
        github_oauth_base_url = get_github_web_base_url()
    except ValueError as exc:
        github_device_flow_error = str(exc)

    return jsonify({
        "github_api_base_url": get_github_api_base_url(),
        "github_oauth_base_url": github_oauth_base_url,
        "github_device_flow_error": github_device_flow_error,
        "copilot_api_base_url": get_copilot_base_url(),
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


@dashboard_bp.route("/api/config-manager/billing", methods=["GET"])
def api_config_manager_billing():
    """Get the Copilot API billing overview for the selected range/user."""
    range_key = request.args.get("range", "week")
    if range_key not in {"all", "hour", "day", "week", "month"}:
        return jsonify({"error": "Invalid range. Use: all, hour, day, week, month"}), 400
    user_filter = _user_filter_from_request()
    provider = request.args.get("provider", "copilot")
    try:
        overview = billing.get_billing_overview(range_key, user_filter=user_filter, provider=provider)
    except Exception as e:
        return jsonify({"error": f"Failed to compute billing overview: {e}"}), 500
    return jsonify(overview)


@dashboard_bp.route("/api/config-manager/billing/prices", methods=["GET"])
def api_config_manager_billing_prices():
    """Get the price catalog status and current manual overrides."""
    catalog = billing.get_price_catalog()
    return jsonify({
        "catalog": catalog.catalog_status(),
        "overrides": catalog.list_overrides(),
    })


@dashboard_bp.route("/api/config-manager/billing/prices", methods=["POST"])
def api_config_manager_billing_prices_upsert():
    """Create or update a manual price override for one model."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        return jsonify({"error": "model is required."}), 400
    try:
        saved = billing.get_price_catalog().set_override(model, payload)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True, "model": model.strip(), "prices": saved})


@dashboard_bp.route("/api/config-manager/billing/prices/<path:model>", methods=["DELETE"])
def api_config_manager_billing_prices_delete(model: str):
    """Remove a manual price override."""
    removed = billing.get_price_catalog().delete_override(model)
    if not removed:
        return jsonify({"error": f"No override found for model '{model}'."}), 404
    return jsonify({"ok": True})


@dashboard_bp.route("/api/config-manager/billing/prices/refresh", methods=["POST"])
def api_config_manager_billing_prices_refresh():
    """Force-refresh the auto price catalog from the upstream source."""
    result = billing.get_price_catalog().refresh(force=True)
    status = 200 if result.get("ok") else 502
    return jsonify(result), status


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


@dashboard_bp.route("/api/live-stats", methods=["GET"])
def api_live_stats():
    """Live per-model throughput for the current process run.

    ``?window=`` takes a preset (5m/15m/1h/6h/24h/all); ``?user=`` filters to
    one user. Everything is computed on read from a bounded in-memory sample
    ring, so this endpoint costs nothing when nobody is looking at it.
    """
    user_filter = _user_filter_from_request()
    window_label, window_seconds = parse_window(request.args.get("window"))

    payload = perf_stats.snapshot(window_seconds=window_seconds, user_id=user_filter)
    payload["window"] = window_label
    payload["window_presets"] = list(WINDOW_PRESETS.keys())
    payload["in_flight"] = cache.get_in_flight(user_id=user_filter)
    return jsonify(payload)


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
        job, reused = request_stats_jobs.start(payload.get("files"), kind=payload.get("kind", "statistics"))
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


DEFAULT_SEARCH_HOURS = 24


def _page_args() -> tuple:
    page = max(1, request.args.get("page", 1, type=int) or 1)
    per_page = max(1, request.args.get("per_page", 50, type=int) or 50)
    return page, per_page


def _page_payload(items: List[Dict[str, Any]], total: int, page: int, per_page: int, **extra: Any) -> Dict[str, Any]:
    return {
        "items": items,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page if total > 0 else 0,
        **extra,
    }


def _summary(item: Dict[str, Any]) -> Dict[str, Any]:
    """List-view projection. In-flight entries still carry their bodies."""
    summary = dict(item)
    for key in ("original_request_body", "request_body", "response_body", "raw_events", "request_headers"):
        summary.pop(key, None)
    return summary


@dashboard_bp.route("/api/requests", methods=["GET"])
def api_requests():
    """Recent requests from memory, optionally filtered by ?user= and ?search= (model/endpoint)."""
    page, per_page = _page_args()
    search = request.args.get("search", "")
    user_filter = _user_filter_from_request()
    offset = (page - 1) * per_page

    if search:
        items, total = cache.search_requests(search, per_page, offset, user_id=user_filter)
    else:
        items = cache.get_recent_requests(per_page, offset, user_id=user_filter)
        total = cache.get_total_count(user_id=user_filter)

    return jsonify(_page_payload([_summary(item) for item in items], total, page, per_page,
                                 user_filter=user_filter))


def _load_request_record(request_id: str):
    """Full record for one request: in memory while in flight, otherwise from the request log.

    Requests that are no longer in memory (full-text search hits from older
    days) are located through the ?file=&offset=&length= the search returned.
    """
    item = cache.get_request(request_id)
    if item is not None and "log" not in item:
        return item, None  # in flight: bodies are still in memory
    locator = item.get("log") if item is not None else None
    if locator is None and request.args.get("file"):
        locator = {
            "file": request.args.get("file"),
            "offset": request.args.get("offset", type=int),
            "length": request.args.get("length", type=int),
        }
    if locator is None:
        if item is not None:
            return None, (jsonify({**item, "error": "This request was not written to the request log"}), 410)
        return None, (jsonify({"error": "Request not found"}), 404)
    try:
        record = request_log.read(locator)
    except RequestLogError as exc:
        return None, (jsonify({"error": str(exc)}), 410)
    if record.get("id") != request_id:
        return None, (jsonify({"error": "Request not found"}), 404)
    return record, None


@dashboard_bp.route("/api/request/<request_id>", methods=["GET"])
def api_request_detail(request_id: str):
    """Full request/response record."""
    record, error = _load_request_record(request_id)
    return error if error else jsonify(record)


@dashboard_bp.route("/api/request/<request_id>/request-body", methods=["GET"])
def api_request_body(request_id: str):
    """Just the upstream request body."""
    record, error = _load_request_record(request_id)
    return error if error else jsonify(record.get("request_body"))


@dashboard_bp.route("/api/request/<request_id>/response-body", methods=["GET"])
def api_response_body(request_id: str):
    """Just the response body.

    Streaming entries store the raw upstream SSE ``data:`` payloads under
    ``raw_events`` instead of a reconstructed ``response_body`` dict. Return
    either, with a discriminator so the dashboard knows which view to render.
    """
    record, error = _load_request_record(request_id)
    if error:
        return error
    if record.get("raw_events") is not None:
        return jsonify({"raw_events": record.get("raw_events")})
    return jsonify(record.get("response_body"))


@dashboard_bp.route("/api/requests/search", methods=["GET"])
def api_fulltext_search():
    """Full-text search in persisted request/response bodies.

    ``?hours=`` limits the search to the last N hours (default 24; 0 searches
    every retained day). ``?user=`` filters to one user.
    """
    page, per_page = _page_args()
    query = request.args.get("q", "")
    user_filter = _user_filter_from_request()
    hours = request.args.get("hours", DEFAULT_SEARCH_HOURS, type=int)
    if hours is None or hours < 0:
        return jsonify({"error": "'hours' must be an integer >= 0"}), 400

    if not query:
        return jsonify(_page_payload([], 0, page, per_page, user_filter=user_filter, hours=hours))

    since = time.time() - hours * 3600 if hours else None
    matches = request_log.search(query, since=since, user_id=user_filter)
    offset = (page - 1) * per_page
    return jsonify(_page_payload(matches[offset:offset + per_page], len(matches), page, per_page,
                                 user_filter=user_filter, hours=hours))
