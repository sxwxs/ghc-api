"""Protocol methods: the params and result type of every method, and the method table both sides share."""

from ..ai.types import Message, ThinkingLevel, UserContent, Wire
from ..coding.agent_session import Delivery, Disposition
from ..coding.session_events import EventEnvelope as EventEnvelope  # re-exported: the event notification's params
from ..coding.session_events import SessionState
from ..coding.session_store import SessionSummary

# The notification that carries session events.
EVENT_METHOD = "session/event"


class Empty(Wire):
    """No params, or an empty result."""


class ClientInfo(Wire):
    name: str
    version: str = ""


class InitializeParams(Wire):
    protocol_version: int
    client_info: ClientInfo | None = None


class AgentInfo(Wire):
    name: str
    version: str


class AgentCapabilities(Wire):
    multi_session: bool
    fork: bool
    images: bool


class InitializeResult(Wire):
    protocol_version: int
    agent_info: AgentInfo
    capabilities: AgentCapabilities


class NewSessionParams(Wire):
    """Unset fields use the server's working directory and the configured defaults."""

    cwd: str | None = None
    provider: str | None = None
    model: str | None = None
    thinking: ThinkingLevel | None = None


class SessionParams(Wire):
    session_id: str


class ListSessionsParams(Wire):
    cwd: str | None = None  # default: the server's working directory
    all: bool = False  # every working directory's sessions; `cwd` is ignored


class ListSessionsResult(Wire):
    sessions: list[SessionSummary]


class MessagesResult(Wire):
    messages: list[Message]


class ForkParams(Wire):
    """Fork through `entry_id` (a message id): the new session ends with that message."""

    session_id: str
    entry_id: str


class SetNameParams(Wire):
    session_id: str
    name: str


class PromptParams(Wire):
    session_id: str
    content: list[UserContent]
    delivery: Delivery = "auto"


class PromptResult(Wire):
    run_id: str
    disposition: Disposition


class DroppedMessages(Wire):
    """Texts of queued messages that were dropped, so a UI can restore them to its editor."""

    texts: list[str]


class ModelInfo(Wire):
    provider: str
    id: str
    context_window: int
    reasoning: bool
    images: bool
    available: bool  # credentials are present or the provider needs none


class ModelsResult(Wire):
    models: list[ModelInfo]


class SetModelParams(Wire):
    session_id: str
    provider: str
    model: str | None = None  # default: the provider's default model


class SetThinkingParams(Wire):
    session_id: str
    level: ThinkingLevel


# Method name -> (params type, result type). The server dispatches by it; the client decodes with it.
METHODS: dict[str, tuple[type[Wire], type[Wire]]] = {
    "initialize": (InitializeParams, InitializeResult),
    "session/new": (NewSessionParams, SessionState),
    "session/load": (SessionParams, SessionState),
    "session/list": (ListSessionsParams, ListSessionsResult),
    "session/close": (SessionParams, Empty),
    "session/state": (SessionParams, SessionState),
    "session/messages": (SessionParams, MessagesResult),
    "session/fork": (ForkParams, SessionState),
    "session/set_name": (SetNameParams, SessionState),
    "run/prompt": (PromptParams, PromptResult),
    "run/abort": (SessionParams, DroppedMessages),
    "run/clear_queue": (SessionParams, DroppedMessages),
    "config/models": (Empty, ModelsResult),
    "config/set_model": (SetModelParams, SessionState),
    "config/set_thinking": (SetThinkingParams, SessionState),
}
