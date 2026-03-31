"""ACP Protocol data types."""

from enum import Enum
from typing import Any, Dict, List, Optional


class AgentKind(str, Enum):
    CLAUDE_CODE = "claude-code"
    CODEX = "codex"
    COPILOT_CLI = "copilot-cli"


class StopReason(str, Enum):
    END_TURN = "endTurn"
    CANCELLED = "cancelled"
    MAX_TOKENS = "maxTokens"
    TOOL_USE = "toolUse"


class AgentBinaryConfig:
    """How to launch an agent process."""

    def __init__(
        self,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.cwd = cwd


class SessionState:
    """Represents an ACP session."""

    def __init__(
        self,
        session_id: str,
        modes: Optional[Dict] = None,
        models: Optional[Dict] = None,
        config_options: Optional[List] = None,
        agent_info: Optional[Dict] = None,
    ):
        self.session_id = session_id
        self.modes = modes
        self.models = models
        self.config_options = config_options
        self.agent_info = agent_info

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "modes": self.modes,
            "models": self.models,
            "config_options": self.config_options,
            "agent_info": self.agent_info,
        }
