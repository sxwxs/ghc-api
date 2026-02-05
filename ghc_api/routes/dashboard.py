"""
Dashboard and API monitoring routes
"""

import json
from datetime import datetime

from flask import Blueprint, Response, jsonify, render_template, request

from ..cache import cache

dashboard_bp = Blueprint('dashboard', __name__)


@dashboard_bp.route("/", methods=["GET"])
def index():
    """Serve the dashboard"""
    return render_template("dashboard.html")


@dashboard_bp.route("/requests", methods=["GET"])
def requests_page():
    """Serve the requests browser page"""
    return render_template("requests.html")


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
