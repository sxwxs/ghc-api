"""What clients see of a session: its state snapshot, run events, and the numbered event envelope."""

from ..agent.events import EndReason, QueueChanged, ToolEnd, ToolStart, ToolUpdate, TurnEnd, TurnStart
from ..ai.events import BlockDelta, BlockEnd, BlockStart, Event, MessageEnd, MessageStart
from ..ai.types import ThinkingLevel, Usage, Wire


class ModelRef(Wire):
    provider: str
    id: str


class QueueState(Wire):
    steering: list[str]
    follow_up: list[str]


class SessionState(Wire):
    """A full snapshot. `usage` sums all replies; `context_tokens` is the size of the last reply's context."""

    session_id: str
    cwd: str
    model: ModelRef
    thinking: ThinkingLevel
    running: bool
    queue: QueueState
    usage: Usage
    context_tokens: int
    context_window: int
    last_seq: int
    name: str | None = None
    run_id: str | None = None
    file: str | None = None


class RunStart(Event, tag="run_start"):
    """A run started; its id is in the envelope."""


class RunEnd(Event, tag="run_end"):
    """The run ended and nothing more will happen without a new prompt."""

    reason: EndReason
    error_message: str | None = None


class StateChanged(Event, tag="state_changed"):
    """Settings changed or a run ended; carries the full state."""

    state: SessionState


SessionEvent = (
    RunStart
    | RunEnd
    | StateChanged
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


class EventEnvelope(Wire):
    """One event of one session. `seq` increases by one per event; `run_id` is set while a run is active."""

    session_id: str
    seq: int
    event: SessionEvent
    run_id: str | None = None
