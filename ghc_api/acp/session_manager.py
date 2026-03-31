"""Session management: in-memory active sessions + disk persistence."""

import json
import logging
import os
import queue
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


class SessionManager:
    """Manages ACP agent sessions (in-memory + persistent)."""

    MAX_WORKDIRS = 20

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions = {}  # type: Dict[str, AgentConnection]
        self._session_locks = {}  # type: Dict[str, threading.Lock]

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

        # Persist metadata
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
            "messages": [],
        }
        self._save_session(session_id, metadata)
        self._add_recent_workdir(working_directory)

        return metadata

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
            for f in sorted(sessions_root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    # Return summary only (no messages)
                    summary = {k: v for k, v in data.items() if k != "messages"}
                    summary["message_count"] = len(data.get("messages", []))
                    # Check if session is alive in memory
                    sid = data.get("session_id", "")
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
        # Try the requested machine path first
        path = _get_sessions_root(machine) / f"{session_id}.json"
        if not path.exists() and machine:
            # Fall back to local machine path (session may have been created locally)
            local_path = _get_sessions_root() / f"{session_id}.json"
            if local_path.exists():
                path = local_path
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
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

    def _save_session(self, session_id: str, data: Dict):
        """Write session data directly to disk (caller must hold file lock)."""
        root = _get_sessions_root()
        root.mkdir(parents=True, exist_ok=True)
        target = root / f"{session_id}.json"
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _append_message(self, session_id: str, message: Dict):
        """Append a message to the session file."""
        lock = self._get_file_lock(session_id)
        with lock:
            root = _get_sessions_root()
            path = root / f"{session_id}.json"
            if not path.exists():
                return
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                data.setdefault("messages", []).append(message)
                data["updated_at"] = datetime.now().isoformat()
                self._save_session(session_id, data)
            except Exception:
                logger.warning("Failed to append message to session %s", session_id, exc_info=True)

    def _update_session_status(self, session_id: str, status: str, updated_at: Optional[str] = None):
        """Update just the status field of a session file."""
        lock = self._get_file_lock(session_id)
        with lock:
            root = _get_sessions_root()
            path = root / f"{session_id}.json"
            if not path.exists():
                return
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                data["status"] = status
                if updated_at:
                    data["updated_at"] = updated_at
                self._save_session(session_id, data)
            except Exception:
                logger.warning("Failed to update session status %s", session_id, exc_info=True)

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
