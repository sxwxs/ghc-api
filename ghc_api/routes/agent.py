"""
Code Agent interaction routes — ACP session management and prompt streaming.
"""

import json
import logging
from typing import Optional

from flask import Blueprint, Response, jsonify, render_template, request, stream_with_context

logger = logging.getLogger(__name__)

agent_bp = Blueprint("agent", __name__)

# Lazy-initialized session manager (singleton)
_session_manager = None


def _get_session_manager():
    global _session_manager
    if _session_manager is None:
        from ..acp.session_manager import SessionManager
        _session_manager = SessionManager()
    return _session_manager


# ---------------------------------------------------------------------------
# Page route
# ---------------------------------------------------------------------------

@agent_bp.route("/agent", methods=["GET"])
def agent_page():
    return render_template("agent.html")


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

@agent_bp.route("/api/agent/sessions", methods=["POST"])
def api_create_session():
    """Create a new agent session."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid JSON body"}), 400

    agent_kind = payload.get("agent_kind", "")
    working_directory = payload.get("working_directory", "")
    bypass_permissions = payload.get("bypass_permissions", True)

    if not agent_kind:
        return jsonify({"error": "agent_kind is required"}), 400
    if not working_directory:
        return jsonify({"error": "working_directory is required"}), 400

    valid_kinds = ["claude-code", "codex", "copilot-cli"]
    if agent_kind not in valid_kinds:
        return jsonify({"error": f"agent_kind must be one of: {valid_kinds}"}), 400

    mgr = _get_session_manager()
    try:
        result = mgr.create_session(agent_kind, working_directory, bypass_permissions)
        return jsonify(result), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.exception("Failed to create session")
        return jsonify({"error": f"Failed to create session: {e}"}), 500


@agent_bp.route("/api/agent/sessions", methods=["GET"])
def api_list_sessions():
    """List sessions, paginated, optionally filtered by machine."""
    machine = request.args.get("machine", None)
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)

    mgr = _get_session_manager()
    result = mgr.list_sessions(machine=machine, page=page, per_page=per_page)
    return jsonify(result)


@agent_bp.route("/api/agent/sessions/<session_id>", methods=["GET"])
def api_session_detail(session_id: str):
    """Get full session detail including message history."""
    machine = request.args.get("machine", None)
    mgr = _get_session_manager()
    detail = mgr.get_session_detail(session_id, machine=machine)
    if detail is None:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(detail)


@agent_bp.route("/api/agent/sessions/<session_id>", methods=["DELETE"])
def api_terminate_session(session_id: str):
    """Terminate a session and kill the agent process."""
    mgr = _get_session_manager()
    try:
        mgr.terminate_session(session_id)
        return jsonify({"ok": True})
    except Exception as e:
        logger.exception("Failed to terminate session %s", session_id)
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Prompt (SSE streaming)
# ---------------------------------------------------------------------------

@agent_bp.route("/api/agent/sessions/<session_id>/prompt", methods=["POST"])
def api_send_prompt(session_id: str):
    """Send a prompt to an active session. Returns SSE stream of updates."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid JSON body"}), 400

    text = payload.get("text", "")
    if not text:
        return jsonify({"error": "text is required"}), 400

    mgr = _get_session_manager()
    from ..acp.session_manager import SENTINEL

    try:
        update_queue = mgr.send_prompt(session_id, text)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400

    def generate():
        collected_updates = []
        stop_reason = "endTurn"
        event_count = 0
        try:
            while True:
                try:
                    # No hard timeout — agent tool calls can take arbitrarily long
                    item = update_queue.get(block=True)
                except Exception as e:
                    print(f"[ACP SSE] queue.get exception: {e}")
                    break

                if item is SENTINEL:
                    print(f"[ACP SSE] got SENTINEL after {event_count} events")
                    break

                if isinstance(item, dict):
                    if "_error" in item:
                        print(f"[ACP SSE] got error item: {item['_error']}")
                        yield _sse_event("error", {"message": item["_error"]})
                        stop_reason = "error"
                        continue

                    # Extract the update payload
                    event_count += 1
                    update = item.get("update", item)
                    collected_updates.append(update)
                    yield _sse_event("update", update)

            yield _sse_event("done", {"stop_reason": stop_reason})

        except GeneratorExit:
            print(f"[ACP SSE] GeneratorExit after {event_count} events (client disconnected)")
            stop_reason = "cancelled"
        finally:
            # Persist agent response
            try:
                mgr.append_agent_response(session_id, collected_updates, stop_reason)
            except Exception:
                logger.warning("Failed to persist agent response for %s", session_id)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@agent_bp.route("/api/agent/sessions/<session_id>/cancel", methods=["POST"])
def api_cancel_prompt(session_id: str):
    """Cancel the current prompt for a session."""
    mgr = _get_session_manager()
    try:
        mgr.cancel_prompt(session_id)
        return jsonify({"ok": True})
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Failed to cancel prompt for %s", session_id)
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@agent_bp.route("/api/agent/machines", methods=["GET"])
def api_list_machines():
    """List available machine names."""
    mgr = _get_session_manager()
    machines = mgr.list_machines()
    return jsonify({"machines": machines})


@agent_bp.route("/api/agent/workdirs", methods=["GET"])
def api_list_workdirs():
    """List recent working directories."""
    mgr = _get_session_manager()
    dirs = mgr.list_recent_workdirs()
    return jsonify({"dirs": dirs})


@agent_bp.route("/api/agent/sessions/<session_id>/storage-path", methods=["GET"])
def api_session_storage_path(session_id: str):
    """Get the storage path for a session."""
    machine = request.args.get("machine", None)
    mgr = _get_session_manager()
    path = mgr.get_session_storage_path(session_id, machine=machine)
    return jsonify({"path": path})


def _sse_event(event_type: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
