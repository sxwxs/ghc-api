"""Agent process spawning and connection management."""

import asyncio
import json
import logging
import os
import platform
import queue
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import AgentBinaryConfig, AgentKind, SessionState
from .protocol import JsonRpcProtocol, JsonRpcError
from .client_handler import AcpClientHandler

logger = logging.getLogger(__name__)

# Default permission modes per agent
DEFAULT_MODES = {
    AgentKind.CLAUDE_CODE: "bypassPermissions",
    AgentKind.CODEX: "full-access",
    AgentKind.COPILOT_CLI: "https://agentclientprotocol.com/protocol/session-modes#autopilot",
}

_SENTINEL = object()


def resolve_agent_binary(kind: AgentKind) -> AgentBinaryConfig:
    """Resolve agent binary based on agent kind."""
    resolvers = {
        AgentKind.CLAUDE_CODE: _resolve_claude,
        AgentKind.CODEX: _resolve_codex,
        AgentKind.COPILOT_CLI: _resolve_copilot,
    }
    resolver = resolvers.get(kind)
    if resolver is None:
        raise ValueError(f"Unknown agent kind: {kind}")
    return resolver()


def _resolve_claude() -> AgentBinaryConfig:
    """Resolve Claude Code ACP agent binary."""
    # Check env var first
    env_binary = os.environ.get("CLAUDE_ACP_BINARY")
    if env_binary and Path(env_binary).exists():
        node = shutil.which("node") or "node"
        return AgentBinaryConfig(
            name="claude-code-env",
            command=node,
            args=[env_binary],
        )

    # Check if claude CLI is available (which would mean the npm package is installed)
    claude_bin = shutil.which("claude")
    if claude_bin:
        # The ACP agent is part of the @anthropic-ai/claude-code package
        # Try to find the ACP entry point relative to the claude binary
        claude_path = Path(claude_bin).resolve()
        # npm global: <prefix>/node_modules/@anthropic-ai/claude-code/...
        # Try common patterns
        for candidate in _find_acp_entry_from_cli(claude_path, "claude-code"):
            if candidate.exists():
                node = shutil.which("node") or "node"
                return AgentBinaryConfig(
                    name="claude-code-local",
                    command=node,
                    args=[str(candidate)],
                )

    # Check @zed-industries/claude-agent-acp (managed install)
    zed_entry = _find_npm_global_package("@zed-industries/claude-agent-acp", "dist/index.js")
    if zed_entry:
        node = shutil.which("node") or "node"
        return AgentBinaryConfig(name="claude-code-zed", command=node, args=[str(zed_entry)])

    raise RuntimeError(
        "Claude Code ACP agent not found. Install options:\n"
        "  1. Set CLAUDE_ACP_BINARY environment variable\n"
        "  2. npm install -g @anthropic-ai/claude-code\n"
        "  3. npm install -g @zed-industries/claude-agent-acp"
    )


def _resolve_codex() -> AgentBinaryConfig:
    """Resolve Codex ACP agent binary."""
    env_binary = os.environ.get("CODEX_ACP_BINARY")
    print('env_binary:', env_binary)
    if env_binary and Path(env_binary).exists():
        return AgentBinaryConfig(name="codex-env", command=env_binary)

    # codex-acp is a separate binary from zed-industries/codex-acp (GitHub release)
    # The regular @openai/codex CLI does NOT support ACP mode
    codex_acp_bin = shutil.which("codex-acp")
    if codex_acp_bin:
        return AgentBinaryConfig(name="codex-acp-local", command=codex_acp_bin)

    raise RuntimeError(
        "Codex ACP agent not found. The regular 'codex' CLI does not support ACP.\n"
        "Install options:\n"
        "  1. Set CODEX_ACP_BINARY environment variable pointing to the codex-acp binary\n"
        "  2. Download codex-acp from https://github.com/zed-industries/codex-acp/releases"
    )


def _resolve_copilot() -> AgentBinaryConfig:
    """Resolve Copilot CLI ACP agent binary."""
    env_binary = os.environ.get("COPILOT_CLI_BINARY")
    if env_binary and Path(env_binary).exists():
        node = shutil.which("node") or "node"
        return AgentBinaryConfig(name="copilot-env", command=node, args=[env_binary, "--acp"])

    copilot_bin = shutil.which("copilot")
    if copilot_bin:
        return AgentBinaryConfig(name="copilot-local", command=copilot_bin, args=["--acp"])

    # Check npm global package
    entry = _find_npm_global_package("@github/copilot", "index.js")
    if entry:
        node = shutil.which("node") or "node"
        return AgentBinaryConfig(name="copilot-npm", command=node, args=[str(entry), "--acp"])

    raise RuntimeError(
        "Copilot CLI ACP agent not found. Install options:\n"
        "  1. Set COPILOT_CLI_BINARY environment variable\n"
        "  2. npm install -g @github/copilot"
    )


def _find_acp_entry_from_cli(cli_path: Path, pkg_name: str) -> List[Path]:
    """Try to find ACP entry point files relative to a CLI binary."""
    candidates = []
    # Walk up to find node_modules
    for parent in cli_path.parents:
        nm = parent / "node_modules"
        if nm.is_dir():
            # Check various possible entry points
            for sub in [
                nm / "@anthropic-ai" / "claude-code" / "dist" / "acp.js",
                nm / "@anthropic-ai" / "claude-code" / "dist" / "index.js",
                nm / "@zed-industries" / "claude-agent-acp" / "dist" / "index.js",
            ]:
                candidates.append(sub)
            break
    return candidates


def _find_npm_global_package(package_name: str, entry_file: str) -> Optional[Path]:
    """Try to find an npm global package entry point."""
    try:
        npm = shutil.which("npm")
        if not npm:
            return None
        result = subprocess.run(
            [npm, "root", "-g"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            global_root = Path(result.stdout.strip())
            parts = package_name.split("/")
            entry = global_root.joinpath(*parts) / entry_file
            if entry.exists():
                return entry
    except Exception:
        pass
    return None


class AgentConnection:
    """Manages a connection to an ACP agent process."""

    def __init__(
        self,
        kind: AgentKind,
        config: AgentBinaryConfig,
        workspace_root: Path,
        auto_approve: bool = True,
    ):
        self.kind = kind
        self.config = config
        self.workspace_root = workspace_root
        self.auto_approve = auto_approve
        self.protocol = None  # type: Optional[JsonRpcProtocol]
        self.process = None  # type: Optional[asyncio.subprocess.Process]
        self.session = None  # type: Optional[SessionState]
        self._update_queue = None  # type: Optional[queue.Queue]
        self._stderr_task = None  # type: Optional[asyncio.Task]
        self._is_prompting = False

    @property
    def is_alive(self) -> bool:
        """Check if the agent process is still running."""
        return self.process is not None and self.process.returncode is None

    async def start(self) -> SessionState:
        """Spawn agent, initialize, create session. Returns session state."""
        # 1. Spawn the agent process
        env = {**os.environ, **self.config.env}
        cmd = [self.config.command] + self.config.args
        logger.info("Spawning agent: %s", " ".join(cmd))

        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.config.cwd or str(self.workspace_root),
                env=env,
                # Default StreamReader limit is 64KB — agent messages can be much
                # larger (e.g., file contents in JSON-RPC responses). 100MB limit.
                limit=100 * 1024 * 1024,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to spawn agent process ({' '.join(cmd)}): {e}")

        logger.info("Agent process pid=%s", self.process.pid)

        # Brief wait to check if process exits immediately
        await asyncio.sleep(0.5)
        if self.process.returncode is not None:
            stderr_bytes = b""
            if self.process.stderr:
                try:
                    stderr_bytes = await asyncio.wait_for(self.process.stderr.read(), timeout=2)
                except Exception:
                    pass
            stderr_text = stderr_bytes.decode(errors="replace").strip()
            raise RuntimeError(
                f"Agent process exited immediately (code {self.process.returncode}).\n"
                f"Command: {' '.join(cmd)}\n"
                f"Stderr: {stderr_text[:2000] if stderr_text else '(empty)'}"
            )

        # Start stderr reader for logging
        self._stderr_task = asyncio.ensure_future(self._read_stderr())

        # 2. Setup protocol handler
        client_handler = AcpClientHandler(
            workspace_root=self.workspace_root,
            auto_approve=self.auto_approve,
        )
        self.protocol = JsonRpcProtocol(self.process, client_handler)
        self.protocol.on_notification("session/update", self._on_session_update)
        self.protocol.start()

        # 3. Initialize connection
        init_result = await self.protocol.request("initialize", {
            "protocolVersion": 1,
            "clientCapabilities": {
                "fs": {"readTextFile": True, "writeTextFile": True},
                "terminal": False,
            },
            "clientInfo": {
                "name": "ghc-api-acp-client",
                "version": "0.1.0",
                "title": "GHC-API ACP Client",
            },
        })
        agent_info = init_result.get("agentInfo", {})
        logger.info("Initialized agent: %s", agent_info)

        # 4. Create new session
        session_result = await self.protocol.request("session/new", {
            "cwd": str(self.workspace_root),
            "mcpServers": [],
        })

        self.session = SessionState(
            session_id=session_result["sessionId"],
            modes=session_result.get("modes"),
            models=session_result.get("models"),
            config_options=session_result.get("configOptions"),
            agent_info=agent_info,
        )

        # 5. Apply default mode
        await self._apply_default_mode()

        return self.session

    async def prompt(self, text: str, update_queue: queue.Queue) -> Dict[str, Any]:
        """Send a text prompt. Pushes session/update dicts to update_queue.
        Pushes _SENTINEL when done. Returns the prompt result."""
        return await self.prompt_blocks(
            [{"type": "text", "text": text}],
            update_queue,
        )

    async def prompt_blocks(
        self,
        content_blocks: List[Dict],
        update_queue: queue.Queue,
    ) -> Dict[str, Any]:
        """Send content blocks as a prompt. Streams updates to queue."""
        assert self.protocol and self.session

        self._update_queue = update_queue
        self._is_prompting = True

        try:
            result = await self.protocol.request("session/prompt", {
                "sessionId": self.session.session_id,
                "prompt": content_blocks,
            }, timeout=None)
            return {
                "stop_reason": result.get("stopReason", "endTurn"),
            }
        except Exception as e:
            logger.exception("Prompt error")
            raise
        finally:
            self._is_prompting = False
            if self._update_queue:
                self._update_queue.put(_SENTINEL)
            self._update_queue = None

    async def cancel(self):
        """Cancel the current prompt."""
        if self.protocol and self.session:
            await self.protocol.notify("session/cancel", {
                "sessionId": self.session.session_id,
            })

    async def set_mode(self, mode_id: str):
        """Switch agent mode."""
        if self.protocol and self.session:
            result = await self.protocol.request("session/set_mode", {
                "sessionId": self.session.session_id,
                "modeId": mode_id,
            })
            if self.session.modes:
                self.session.modes["currentModeId"] = mode_id
            return result

    async def set_model(self, model_id: str):
        """Switch LLM model."""
        if self.protocol and self.session:
            result = await self.protocol.request("session/set_model", {
                "sessionId": self.session.session_id,
                "modelId": model_id,
            })
            if self.session.models:
                self.session.models["currentModelId"] = model_id
            return result

    async def close(self):
        """Shut down the agent."""
        if self.protocol:
            await self.protocol.close()
        if self._stderr_task:
            self._stderr_task.cancel()
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
            except Exception:
                pass

    async def _apply_default_mode(self):
        """Apply the agent-specific default mode if bypass is enabled."""
        if not self.auto_approve:
            return
        mode = DEFAULT_MODES.get(self.kind)
        if mode and self.session and self.session.modes:
            available = [m["id"] for m in self.session.modes.get("availableModes", [])]
            if mode in available:
                try:
                    await self.set_mode(mode)
                except Exception:
                    logger.warning("Failed to set default mode %s", mode, exc_info=True)

    async def _on_session_update(self, params: Dict):
        """Handle streaming session updates — push to queue if active."""
        if self._update_queue:
            self._update_queue.put(params)

    async def _read_stderr(self):
        """Read and log agent stderr."""
        assert self.process and self.process.stderr
        while True:
            try:
                line = await self.process.stderr.readline()
            except Exception as e:
                print(f"[ACP] stderr: read error: {e}")
                break
            if not line:
                print(f"[ACP] stderr: closed (process returncode={self.process.returncode})")
                break
            text = line.decode(errors="replace").rstrip()
            if text:
                print(f"[ACP] stderr: {text[:300]}")
