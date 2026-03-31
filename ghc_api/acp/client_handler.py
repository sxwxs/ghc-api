"""Handles agent -> client callbacks (file I/O, permissions)."""

import asyncio
import logging
from pathlib import Path
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class AcpClientHandler:
    """Responds to agent requests for file operations and permissions."""

    def __init__(
        self,
        workspace_root: Path,
        auto_approve: bool = True,
        on_permission_request: Optional[Callable] = None,
    ):
        self.workspace_root = workspace_root
        self.auto_approve = auto_approve
        self.on_permission_request = on_permission_request

    async def handle(self, method: str, params: Dict) -> Dict:
        """Dispatch an agent -> client request to the right handler."""
        handlers = {
            "fs/readTextFile": self._read_file,
            "fs/writeTextFile": self._write_file,
            "requestPermission": self._request_permission,
            "terminal/create": self._terminal_create,
            "terminal/output": self._terminal_output,
            "terminal/kill": self._terminal_kill,
            "terminal/release": self._terminal_release,
        }
        handler = handlers.get(method)
        if handler is None:
            raise NotImplementedError(f"Unsupported client method: {method}")
        return await handler(params)

    async def _read_file(self, params: Dict) -> Dict:
        """Read a file and return its contents (non-blocking)."""
        path = self._resolve_path(params["path"])
        logger.info("Agent reading file: %s", path)

        def _do_read():
            try:
                return path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return path.read_text(encoding="utf-8", errors="replace")

        content = await asyncio.get_event_loop().run_in_executor(None, _do_read)
        return {"content": content}

    async def _write_file(self, params: Dict) -> Dict:
        """Write content to a file (non-blocking)."""
        path = self._resolve_path(params["path"])
        content = params["content"]
        logger.info("Agent writing file: %s (%d chars)", path, len(content))

        def _do_write():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        await asyncio.get_event_loop().run_in_executor(None, _do_write)
        return {}

    async def _request_permission(self, params: Dict) -> Dict:
        """Handle a permission request from the agent."""
        options = params.get("options", [])

        if self.auto_approve and options:
            for preferred in ["allow", "allow_once"]:
                for opt in options:
                    if opt.get("optionId") == preferred:
                        logger.debug("Auto-approving permission: %s", preferred)
                        return {"outcome": {"kind": "selected", "optionId": preferred}}
            # Fallback to first option
            first_id = options[0]["optionId"]
            logger.debug("Auto-approving permission (fallback): %s", first_id)
            return {"outcome": {"kind": "selected", "optionId": first_id}}

        # When not auto-approving, notify via callback or deny
        if self.on_permission_request:
            result = self.on_permission_request(params)
            if result:
                return result

        # Default: deny
        logger.warning("Permission denied (no auto-approve, no callback)")
        return {"outcome": {"kind": "dismissed"}}

    async def _terminal_create(self, params: Dict) -> Dict:
        """Handle terminal creation request - stub implementation."""
        logger.info("Agent requested terminal creation (not fully supported)")
        return {"terminalId": "stub-terminal-1"}

    async def _terminal_output(self, params: Dict) -> Dict:
        """Handle terminal output request - stub."""
        return {"output": ""}

    async def _terminal_kill(self, params: Dict) -> Dict:
        """Handle terminal kill request - stub."""
        return {}

    async def _terminal_release(self, params: Dict) -> Dict:
        """Handle terminal release request - stub."""
        return {}

    def _resolve_path(self, path_str: str) -> Path:
        """Resolve a path, making relative paths relative to workspace."""
        p = Path(path_str)
        if p.is_absolute():
            return p
        return self.workspace_root / p
