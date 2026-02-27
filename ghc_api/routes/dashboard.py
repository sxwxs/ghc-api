"""
Dashboard and API monitoring routes
"""

import json
from datetime import datetime
from typing import Any, Dict, List

from flask import Blueprint, Response, jsonify, render_template, request

from ..cache import cache
from ..config import model_mappings
from ..state import state

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
        "auto_remove_encrypted_content_on_parse_error": state.auto_remove_encrypted_content_on_parse_error,
        "model_mappings": {
            "exact": model_mappings.exact_mappings,
            "prefix": model_mappings.prefix_mappings,
        },
    }


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


@dashboard_bp.route("/", methods=["GET"])
def index():
    """Serve the dashboard"""
    return render_template("dashboard.html")


@dashboard_bp.route("/requests", methods=["GET"])
def requests_page():
    """Serve the requests browser page"""
    return render_template("requests.html")


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
        "auto_remove_encrypted_content_on_parse_error",
        "model_mappings",
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

        if "auto_remove_encrypted_content_on_parse_error" in payload:
            flag = payload["auto_remove_encrypted_content_on_parse_error"]
            if not isinstance(flag, bool):
                raise ValueError("'auto_remove_encrypted_content_on_parse_error' must be a boolean")
            state.auto_remove_encrypted_content_on_parse_error = flag

        if "model_mappings" in payload:
            mappings = payload["model_mappings"]
            if not isinstance(mappings, dict):
                raise ValueError("'model_mappings' must be an object")
            exact = _validate_mapping(mappings.get("exact", {}), "model_mappings.exact")
            prefix = _validate_mapping(mappings.get("prefix", {}), "model_mappings.prefix")
            model_mappings.exact_mappings = exact
            model_mappings.prefix_mappings = prefix

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({
        "message": "Runtime configuration updated. This does not modify config.yaml.",
        "config": _runtime_config(),
    })


@dashboard_bp.route("/api/stats", methods=["GET"])
def api_stats():
    """Get API statistics"""
    return jsonify(cache.get_stats())


@dashboard_bp.route("/api/requests", methods=["GET"])
def api_requests():
    """Get paginated list of requests"""
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)
    search = request.args.get("search", "")

    offset = (page - 1) * per_page

    if search:
        items = cache.search_requests(search, per_page, offset)
        total = len(cache.search_requests(search, 10000, 0))  # Get total count
    else:
        items = cache.get_recent_requests(per_page, offset)
        total = cache.get_total_count()

    # Remove large body content for list view
    items_summary = []
    for item in items:
        summary = dict(item)
        summary["request_body"] = None  # Remove for list view
        summary["response_body"] = None  # Remove for list view
        items_summary.append(summary)

    return jsonify({
        "items": items_summary,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
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
    """Get just the response body"""
    item = cache.get_request(request_id)
    if not item:
        return jsonify({"error": "Request not found"}), 404
    return jsonify(item.get("response_body"))


@dashboard_bp.route("/api/requests/search", methods=["GET"])
def api_fulltext_search():
    """Full-text search in request/response bodies"""
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)
    query = request.args.get("q", "")

    if not query:
        return jsonify({
            "items": [],
            "total": 0,
            "page": page,
            "per_page": per_page,
            "total_pages": 0,
        })

    offset = (page - 1) * per_page
    items, total = cache.fulltext_search(query, per_page, offset)

    # Remove large body content for list view
    items_summary = []
    for item in items:
        summary = dict(item)
        summary["request_body"] = None
        summary["response_body"] = None
        items_summary.append(summary)

    return jsonify({
        "items": items_summary,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page if total > 0 else 0,
    })


@dashboard_bp.route("/api/requests/export", methods=["GET"])
def api_export_requests():
    """Export all requests as JSON Lines (.jl) file"""
    def generate():
        for item in cache.get_all_requests():
            yield json.dumps(item, ensure_ascii=False) + "\n"

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
