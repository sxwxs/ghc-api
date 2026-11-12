"""Agent events: the run, turn, message and tool lifecycle a UI renders."""

from typing import Literal

from ..ai.events import BlockDelta, BlockEnd, BlockStart, Event, MessageEnd, MessageStart
from .tool import ToolKind, ToolResult

EndReason = Literal["completed", "aborted", "error"]


class AgentStart(Event, tag="agent_start"):
    pass


class AgentEnd(Event, tag="agent_end"):
    reason: EndReason
    error_message: str | None = None


class TurnStart(Event, tag="turn_start"):
    turn: int


class TurnEnd(Event, tag="turn_end"):
    turn: int


class ToolStart(Event, tag="tool_start"):
    """A tool call is being handled. Its arguments are in the assistant message; `title` summarizes them."""

    tool_call_id: str
    name: str
    kind: ToolKind
    title: str


class ToolUpdate(Event, tag="tool_update"):
    tool_call_id: str
    partial: ToolResult


class ToolEnd(Event, tag="tool_end"):
    """A tool call finished. `duration_ms` is absent when the tool did not run."""

    tool_call_id: str
    result: ToolResult
    duration_ms: int | None = None


class QueueChanged(Event, tag="queue_changed"):
    """The complete pending queues, as message texts."""

    steering: list[str]
    follow_up: list[str]


AgentEvent = (
    AgentStart
    | AgentEnd
    | TurnStart
    | TurnEnd
    | MessageStart
    | BlockStart
    | BlockDelta
    | BlockEnd
    | MessageEnd
    | ToolStart
    | ToolUpdate
    | ToolEnd
    | QueueChanged
)
