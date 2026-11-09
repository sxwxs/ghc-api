"""
Code Agent: the /agent page and its bridge to pyagent (see agent_bridge.py).

The page speaks pyagent's JSON-RPC protocol (docs/pyagent/protocol.md) through
POST /api/agent/rpc and follows a session's events over SSE. The agent runs
shell commands on this machine, so these routes are admin-only: keep them behind
the reverse-proxy auth, like /api/users/*.
"""

import os

from flask import Blueprint, Response, jsonify, render_template, request, stream_with_context

from ..agent_bridge import get_bridge

agent_bp = Blueprint("agent", __name__)

# Keepalive interval while a run is active; an idle session's stream closes after this long without events.
KEEPALIVE_SECONDS = 15.0


@agent_bp.route("/agent", methods=["GET"])
def agent_page():
    return render_template("agent.html")


@agent_bp.route("/api/agent/rpc", methods=["POST"])
def agent_rpc():
    """Forward one protocol request: {"method", "params"} -> {"result"} or {"error": {"code", "message"}}."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get("method"), str):
        return jsonify({"error": {"code": -32600, "message": "Expected {\"method\": str, \"params\": object}"}}), 400
    method, params = body["method"], body.get("params") or {}
    if not isinstance(params, dict):
        return jsonify({"error": {"code": -32602, "message": "params must be an object"}}), 400
    if method == "initialize":
        return jsonify({"error": {"code": -32601, "message": "ghc-api initializes the agent itself"}}), 400
    if method == "session/new" and params.get("cwd"):
        cwd = os.path.abspath(os.path.expanduser(str(params["cwd"])))
        if not os.path.isdir(cwd):
            return jsonify({"error": {"code": -32602, "message": f"Not a directory: {cwd}"}}), 400
        params = {**params, "cwd": cwd}
    try:
        reply = get_bridge().call(method, params)
    except TimeoutError:
        return jsonify({"error": {"code": -32603, "message": f"pyagent did not answer {method} in time"}}), 504
    return jsonify({key: reply[key] for key in ("result", "error") if key in reply})


@agent_bp.route("/api/agent/sessions/<session_id>/events", methods=["GET"])
def agent_events(session_id: str):
    """SSE: the session's events after `since` (or Last-Event-ID).

    Every open stream holds a server worker thread, so a stream lasts only while the session runs: once it is
    idle and quiet, `idle` ends it and the page subscribes again when it starts a run or comes back into view.
    `resync` tells the page to reload the session, because it is not open or the events it missed are gone.
    """
    last_id = request.headers.get("Last-Event-ID") or request.args.get("since") or "0"
    try:
        since = int(last_id)
    except ValueError:
        since = 0
    bridge = get_bridge()

    def generate():
        position = since
        while True:
            events = bridge.wait_events(session_id, position, KEEPALIVE_SECONDS)
            if events is None:
                yield "event: resync\ndata: {}\n\n"
                return
            if events:
                for seq, text in events:
                    yield f"id: {seq}\ndata: {text}\n\n"
                position = events[-1][0]
            elif bridge.is_running(session_id):
                yield ": keepalive\n\n"
            else:
                yield "event: idle\ndata: {}\n\n"
                return

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
