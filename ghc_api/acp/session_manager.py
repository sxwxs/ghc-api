"""Session management: in-memory active sessions + disk persistence.

Session files use an append-only JSON Lines format (.jl.b64gz):
  - Line 1: gzip-compressed, base64-encoded JSON metadata (header)
  - Lines 2+: gzip-compressed, base64-encoded JSON update objects

Updates are buffered in memory and flushed to disk periodically
(controlled by state.session_flush_interval). Same-type non-message
updates are merged (only the latest is kept) before flushing.
"""

import atexit
import base64
import gzip
import json
import logging
import os
import queue
import signal
import socket
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .types import AgentKind, SessionState
from .connection import AgentConnection, resolve_agent_binary, _SENTINEL

logger = logging.getLogger(__name__)

_SESSION_EXT = ".jl.b64gz"


def _encode_update(obj):
    """Encode a dict as gzip-compressed, base64-encoded JSON string."""
    raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    compressed = gzip.compress(raw)
    return base64.b64encode(compressed).decode("ascii")


def _decode_update(line):
    """Decode a base64+gzip line back to a dict."""
    compressed = base64.b64decode(line.encode("ascii"))
    raw = gzip.decompress(compressed)
    return json.loads(raw)


def _local_machine_name() -> str:
    """Get current machine name in the same format as config_sync."""
    from ..config_sync import _os_label
    return f"{socket.gethostname()}_{_os_label()}"


def _get_sessions_root(machine: Optional[str] = None) -> Path:
    """Get the sessions storage directory."""
    from ..config_sync import get_agent_root, get_onedrive_path, onedrive_access_disabled
    from ..utils import get_config_dir

    if not onedrive_access_disabled():
        onedrive = get_onedrive_path()
        if onedrive:
            m = machine or _local_machine_name()
            return onedrive / ".ghc-api" / "agents" / m / "sessions"

    return Path(get_config_dir()) / "sessions"


def _get_workdirs_path() -> Path:
    """Get the workdirs.json file path."""
    from ..config_sync import get_agent_root
    from ..utils import get_config_dir

    agent_root = get_agent_root()
    if agent_root:
        return agent_root / "workdirs.json"
    return Path(get_config_dir()) / "workdirs.json"


def _merge_updates(updates):
    """Merge a list of buffered updates: keep all messages in order,
    keep only the latest of each non-message type."""
    messages = []
    latest_by_type = {}  # type -> update

    for u in updates:
        if u.get("type") == "message":
            messages.append(u)
        else:
            latest_by_type[u.get("type")] = u

    return messages + list(latest_by_type.values())


def _apply_updates_to_session(data, updates):
    """Apply a list of decoded updates to a session metadata dict in place."""
    for u in updates:
        utype = u.get("type")
        if utype == "message":
            data.setdefault("messages", []).append(u["data"])
        elif utype == "status":
            data["status"] = u["status"]
        if "updated_at" in u:
            data["updated_at"] = u["updated_at"]


class SessionManager:
    """Manages ACP agent sessions (in-memory + persistent)."""

    MAX_WORKDIRS = 20

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions = {}  # type: Dict[str, AgentConnection]
        self._session_locks = {}  # type: Dict[str, threading.Lock]

        # Buffered writes
        self._buffers = {}  # type: Dict[str, List[Dict]]
        self._buffer_lock = threading.Lock()
        self._flush_thread = None  # type: Optional[threading.Thread]
        self._stop_event = threading.Event()
        self._flush_started = False

    # --- Flush loop lifecycle ---

    def _start_flush_loop(self):
        """Start the periodic flush thread (called lazily on first session create)."""
        if self._flush_started:
            return
        self._flush_started = True
        self._flush_thread = threading.Thread(
            target=self._flush_loop, name="session-flush", daemon=True
        )
        self._flush_thread.start()
        atexit.register(self._shutdown_flush)
        logger.info("Session flush loop started.")

    def _flush_loop(self):
        from ..state import state
        interval = state.session_flush_interval
        while not self._stop_event.wait(interval):
            try:
                self._flush_all_buffers()
            except Exception:
                logger.warning("Session flush error", exc_info=True)

    def _shutdown_flush(self):
        """Stop flush thread and do a final flush."""
        self._stop_event.set()
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=2)
        try:
            self._flush_all_buffers()
        except Exception:
            logger.warning("Final session flush error", exc_info=True)

    # --- Buffer management ---

    def _buffer_update(self, session_id, update):
        """Add an update to the in-memory buffer for a session."""
        with self._buffer_lock:
            self._buffers.setdefault(session_id, []).append(update)

    def _flush_all_buffers(self):
        """Snapshot all buffers, then flush each to disk."""
        with self._buffer_lock:
            snapshot = dict(self._buffers)
            self._buffers.clear()

        for session_id, updates in snapshot.items():
            if updates:
                self._flush_session(session_id, updates)

    def _flush_session(self, session_id, updates):
        """Merge and write buffered updates to the session file."""
        merged = _merge_updates(updates)
        if not merged:
            return

        lock = self._get_file_lock(session_id)
        with lock:
            root = _get_sessions_root()
            path = root / f"{session_id}{_SESSION_EXT}"
            if not path.exists():
                return
            try:
                lines = []
                for u in merged:
                    lines.append(_encode_update(u) + "\n")
                with path.open("a", encoding="utf-8") as f:
                    f.writelines(lines)
            except Exception:
                logger.warning("Failed to flush session %s", session_id, exc_info=True)

    def _flush_session_id(self, session_id):
        """Flush a single session's buffer immediately."""
        with self._buffer_lock:
            updates = self._buffers.pop(session_id, [])
        if updates:
            self._flush_session(session_id, updates)

    # --- Public API ---

    def create_session(
        self,
        agent_kind_str: str,
        working_directory: str,
        bypass_permissions: bool = True,
    ) -> Dict[str, Any]:
        """Create a new agent session. Blocks until ready."""
        from . import run_async

        kind = AgentKind(agent_kind_str)
        config = resolve_agent_binary(kind)
        workspace = Path(working_directory)

        if not workspace.is_dir():
            raise ValueError(f"Working directory does not exist: {working_directory}")

        conn = AgentConnection(
            kind=kind,
            config=config,
            workspace_root=workspace,
            auto_approve=bypass_permissions,
        )

        session_state = run_async(conn.start(), timeout=120)
        session_id = session_state.session_id

        with self._lock:
            self._sessions[session_id] = conn
            self._session_locks[session_id] = threading.Lock()

        # Persist metadata header
        metadata = {
            "session_id": session_id,
            "agent_kind": agent_kind_str,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "active",
            "working_directory": working_directory,
            "machine": _local_machine_name(),
            "agent_info": session_state.agent_info or {},
            "modes": session_state.modes or {},
            "models": session_state.models or {},
        }
        self._write_session_header(session_id, metadata)
        self._add_recent_workdir(working_directory)
        self._start_flush_loop()

        # Return with empty messages for API response compatibility
        result = dict(metadata)
        result["messages"] = []
        return result

    def list_sessions(
        self,
        machine: Optional[str] = None,
        page: int = 1,
        per_page: int = 50,
    ) -> Dict[str, Any]:
        """List sessions from disk, paginated."""
        sessions_root = _get_sessions_root(machine)
        items = []

        if sessions_root.exists():
            # Collect both old .json and new .jl.b64gz files
            files = list(sessions_root.glob("*.json")) + list(sessions_root.glob("*" + _SESSION_EXT))
            for f in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    summary, message_count = self._read_session_summary(f)
                    if summary is None:
                        continue
                    summary["message_count"] = message_count
                    # Check if session is alive in memory
                    sid = summary.get("session_id", "")
                    with self._lock:
                        conn = self._sessions.get(sid)
                    if conn and conn.is_alive:
                        summary["status"] = "active"
                    elif summary.get("status") == "active":
                        summary["status"] = "disconnected"
                    items.append(summary)
                except Exception:
                    logger.debug("Failed to read session file %s", f, exc_info=True)

        total = len(items)
        offset = (page - 1) * per_page
        page_items = items[offset:offset + per_page]

        return {
            "items": page_items,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": max(1, (total + per_page - 1) // per_page),
        }

    def get_session_detail(self, session_id: str, machine: Optional[str] = None) -> Optional[Dict]:
        """Get full session detail including messages."""
        path = self._resolve_session_path(session_id, machine)
        if not path:
            return None
        try:
            data = self._read_session_file(path, session_id)
            # Update status if we have a live connection
            with self._lock:
                conn = self._sessions.get(session_id)
            if conn and conn.is_alive:
                data["status"] = "active"
            elif data.get("status") == "active":
                data["status"] = "disconnected"
            return data
        except Exception:
            logger.exception("Failed to read session %s", session_id)
            return None

    def send_prompt(self, session_id: str, text: str) -> queue.Queue:
        """Send a prompt to a live session. Returns a queue that receives updates.
        The queue will receive dicts for each update, and _SENTINEL when done."""
        with self._lock:
            conn = self._sessions.get(session_id)
        if not conn or not conn.is_alive:
            raise RuntimeError("Session is not active")

        from . import run_async, get_event_loop
        import asyncio

        update_queue = queue.Queue()  # type: queue.Queue

        # Append user message to session file
        self._append_message(session_id, {
            "role": "user",
            "content": [{"type": "text", "text": text}],
            "timestamp": datetime.now().isoformat(),
        })

        # Run prompt in background on the event loop
        loop = get_event_loop()

        async def _do_prompt():
            try:
                result = await conn.prompt(text, update_queue)
                return result
            except Exception as e:
                logger.exception("Prompt failed for session %s", session_id)
                update_queue.put({"_error": str(e)})
                update_queue.put(_SENTINEL)
                return {"stop_reason": "error", "error": str(e)}

        future = asyncio.run_coroutine_threadsafe(_do_prompt(), loop)

        # Store future for later retrieval of result (for persistence)
        def _on_done(f):
            try:
                result = f.result(timeout=0)
                stop_reason = result.get("stop_reason", "endTurn")
            except Exception as e:
                stop_reason = "error"

            # Collect all updates that were queued
            updates = []
            # We can't drain the queue here since the SSE consumer needs them.
            # Instead, we persist after the SSE stream closes in the route handler.
            self._update_session_status(session_id, "active", datetime.now().isoformat())

        future.add_done_callback(_on_done)

        return update_queue

    def cancel_prompt(self, session_id: str):
        """Cancel the current prompt for a session."""
        with self._lock:
            conn = self._sessions.get(session_id)
        if not conn or not conn.is_alive:
            raise RuntimeError("Session is not active")

        from . import run_async
        run_async(conn.cancel(), timeout=10)

    def terminate_session(self, session_id: str):
        """Terminate a session and kill the agent process."""
        with self._lock:
            conn = self._sessions.pop(session_id, None)
            self._session_locks.pop(session_id, None)

        if conn:
            from . import run_async
            try:
                run_async(conn.close(), timeout=10)
            except Exception:
                logger.warning("Failed to close session %s", session_id, exc_info=True)

        self._update_session_status(session_id, "terminated")
        self._flush_session_id(session_id)

    def append_agent_response(self, session_id: str, updates: List[Dict], stop_reason: str):
        """Persist agent response updates to the session file."""
        self._append_message(session_id, {
            "role": "agent",
            "updates": updates,
            "stop_reason": stop_reason,
            "timestamp": datetime.now().isoformat(),
        })

    def list_machines(self) -> List[str]:
        """List available machine names from OneDrive agents directory."""
        from ..config_sync import get_machines_list
        return get_machines_list()

    def list_recent_workdirs(self) -> List[str]:
        """List recent working directories."""
        path = _get_workdirs_path()
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("dirs", [])
        except Exception:
            return []

    def get_session_storage_path(self, session_id: str, machine: Optional[str] = None) -> str:
        """Get the file system path for a session's storage directory."""
        return str(_get_sessions_root(machine))

    # --- Internal helpers ---

    def _get_file_lock(self, session_id: str) -> threading.Lock:
        """Get or create a per-session lock for file I/O."""
        with self._lock:
            if session_id not in self._session_locks:
                self._session_locks[session_id] = threading.Lock()
            return self._session_locks[session_id]

    def _write_session_header(self, session_id: str, metadata: Dict):
        """Write the session header (line 1) to a new .jl.b64gz file."""
        root = _get_sessions_root()
        root.mkdir(parents=True, exist_ok=True)
        target = root / f"{session_id}{_SESSION_EXT}"
        header = {k: v for k, v in metadata.items() if k != "messages"}
        target.write_text(_encode_update(header) + "\n", encoding="utf-8")

    def _append_message(self, session_id: str, message: Dict):
        """Buffer a message update for later flush."""
        update = {
            "type": "message",
            "data": message,
            "updated_at": datetime.now().isoformat(),
        }
        self._buffer_update(session_id, update)

    def _update_session_status(self, session_id: str, status: str, updated_at: Optional[str] = None):
        """Buffer a status update for later flush."""
        update = {
            "type": "status",
            "status": status,
            "updated_at": updated_at or datetime.now().isoformat(),
        }
        self._buffer_update(session_id, update)

    def _resolve_session_path(self, session_id: str, machine: Optional[str] = None) -> Optional[Path]:
        """Find the session file path, trying new format first, then old."""
        root = _get_sessions_root(machine)
        # Try new format first
        new_path = root / f"{session_id}{_SESSION_EXT}"
        if new_path.exists():
            return new_path
        # Try old format
        old_path = root / f"{session_id}.json"
        if old_path.exists():
            return old_path
        # Fall back to local machine path if a remote machine was requested
        if machine:
            local_root = _get_sessions_root()
            for ext in (_SESSION_EXT, ".json"):
                p = local_root / f"{session_id}{ext}"
                if p.exists():
                    return p
        return None

    def _read_session_file(self, path: Path, session_id: Optional[str] = None) -> Dict:
        """Read a session file (either .json or .jl.b64gz) and return full session dict."""
        lock = self._get_file_lock(session_id) if session_id else threading.Lock()

        if str(path).endswith(".json"):
            with lock:
                data = json.loads(path.read_text(encoding="utf-8"))
            return data

        # .jl.b64gz format — all lines are encoded
        with lock:
            lines = path.read_text(encoding="utf-8").splitlines()

        if not lines:
            return {}

        # Line 1: encoded header
        data = _decode_update(lines[0].strip())
        data.setdefault("messages", [])

        # Lines 2+: encoded updates
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            try:
                update = _decode_update(line)
                _apply_updates_to_session(data, [update])
            except Exception:
                logger.debug("Failed to decode session update line", exc_info=True)

        # Merge any unflushed buffered updates
        if session_id:
            with self._buffer_lock:
                pending = list(self._buffers.get(session_id, []))
            _apply_updates_to_session(data, pending)

        return data

    def _read_session_summary(self, path: Path) -> Tuple[Optional[Dict], int]:
        """Read session summary for list_sessions. Returns (summary_dict, message_count).
        For .jl.b64gz, reads only what's needed (header + line count).
        For .json, reads the full file."""
        if str(path).endswith(".json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            summary = {k: v for k, v in data.items() if k != "messages"}
            return summary, len(data.get("messages", []))

        # .jl.b64gz: all lines are encoded, header is line 1
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if not lines:
            return None, 0

        header = _decode_update(lines[0].strip())

        # Count message-type updates (decode each line to check type)
        # Also apply status updates to get current status
        message_count = 0
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            try:
                update = _decode_update(line)
                if update.get("type") == "message":
                    message_count += 1
                elif update.get("type") == "status":
                    header["status"] = update["status"]
                if "updated_at" in update:
                    header["updated_at"] = update["updated_at"]
            except Exception:
                pass

        # Also count buffered updates
        sid = header.get("session_id", "")
        if sid:
            with self._buffer_lock:
                pending = list(self._buffers.get(sid, []))
            for u in pending:
                if u.get("type") == "message":
                    message_count += 1
                elif u.get("type") == "status":
                    header["status"] = u["status"]
                if "updated_at" in u:
                    header["updated_at"] = u["updated_at"]

        return header, message_count

    def _add_recent_workdir(self, directory: str):
        """Add a working directory to the recents list."""
        path = _get_workdirs_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        dirs = []  # type: List[str]
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                dirs = data.get("dirs", [])
            except Exception:
                pass

        # Move to front if already present, or prepend
        normalized = os.path.normpath(directory)
        dirs = [d for d in dirs if os.path.normpath(d) != normalized]
        dirs.insert(0, directory)
        dirs = dirs[:self.MAX_WORKDIRS]

        path.write_text(
            json.dumps({"dirs": dirs}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# Module-level sentinel for queue consumers
SENTINEL = _SENTINEL
