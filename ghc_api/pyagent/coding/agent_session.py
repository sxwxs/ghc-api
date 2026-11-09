"""AgentSession: a conversation backed by a session file; it runs prompts, numbers events, and records entries."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..agent.agent import Agent
from ..agent.events import AgentEnd, AgentEvent, AgentStart
from ..ai.events import MessageEnd
from ..ai.models import Model, ModelError, ProviderConfig, resolve_model
from ..ai.provider import ProviderFn
from ..ai.types import AssistantMessage, Message, ThinkingLevel, Usage, UserContent, UserMessage, new_id
from .config import Config, select_model
from .prompt import build_system_prompt, load_context_files
from .session_events import (
    EventEnvelope,
    ModelRef,
    QueueState,
    RunEnd,
    RunStart,
    SessionEvent,
    SessionState,
    StateChanged,
)
from .session_store import SessionStore, sessions_dir
from .tools import default_tools

Delivery = Literal["auto", "steer", "follow_up"]
Disposition = Literal["started", "queued"]
EnvelopeListener = Callable[[EventEnvelope], None]


@dataclass
class SessionSetup:
    """What all sessions of one host share. `provider` replaces the HTTP adapters (tests, demos)."""

    home: Path
    config: Config
    providers: dict[str, ProviderConfig]
    persist: bool = True
    provider: ProviderFn | None = None


class AgentSession:
    """Runs prompts through an Agent, records messages and settings in the store, and emits numbered events."""

    def __init__(self, store: SessionStore, agent: Agent) -> None:
        self.store = store
        self.agent = agent
        self.seq = 0
        # Set from prompt to run_end. The agent's task outlives run_end by one synchronous step, so this, not
        # Agent.running, is what "running" means for clients.
        self.run_id: str | None = None
        self._listeners: list[EnvelopeListener] = []
        agent.subscribe(self._on_agent_event)

    @property
    def id(self) -> str:
        return self.store.id

    def subscribe(self, listener: EnvelopeListener) -> Callable[[], None]:
        """Receive every event; returns an unsubscribe function."""
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    def prompt(self, content: list[UserContent], delivery: Delivery) -> tuple[str, Disposition]:
        """Start a run when idle, whatever the delivery; queue the message when a run is active ("auto" steers)."""
        message = UserMessage(content=content)
        if self.run_id is not None:
            if delivery == "follow_up":
                self.agent.follow_up(message)
            else:
                self.agent.steer(message)
            return self.run_id, "queued"
        self.run_id = new_id()
        self.agent.start([message])
        return self.run_id, "started"

    def abort(self) -> list[str]:
        """Abort the active run; returns the texts of dropped queued messages."""
        return self.agent.abort()

    def clear_queue(self) -> list[str]:
        """Drop queued messages; returns their texts."""
        return self.agent.clear_queue()

    def set_model(self, model: Model) -> None:
        """Use `model` from the next run on."""
        self.agent.model = model
        self.store.append_model_change(model.provider, model.id)
        self._emit(StateChanged(state=self.state()))

    def set_thinking(self, level: ThinkingLevel) -> None:
        """Use `level` from the next run on."""
        self.agent.thinking = level
        self.store.append_thinking_change(level)
        self._emit(StateChanged(state=self.state()))

    def set_name(self, name: str) -> None:
        """Set the display name."""
        self.store.append_session_info(name)
        self._emit(StateChanged(state=self.state()))

    def messages(self) -> list[Message]:
        """The active branch's messages."""
        return self.agent.messages

    def state(self) -> SessionState:
        """A full snapshot."""
        replies = [m for m in self.agent.messages if isinstance(m, AssistantMessage)]
        completed = [m for m in replies if m.stop_reason not in ("error", "aborted")]
        steering, follow_up = self.agent.queued()
        return SessionState(
            session_id=self.id,
            cwd=self.store.cwd,
            model=ModelRef(provider=self.agent.model.provider, id=self.agent.model.id),
            thinking=self.agent.thinking,
            running=self.run_id is not None,
            queue=QueueState(steering=steering, follow_up=follow_up),
            usage=_total(replies),
            context_tokens=completed[-1].usage.context_tokens if completed else 0,
            context_window=self.agent.model.context_window,
            last_seq=self.seq,
            name=self.store.context().name,
            run_id=self.run_id,
            file=str(self.store.path) if self.store.path else None,
        )

    async def close(self) -> None:
        """Abort any run, wait for it to settle, and close the file."""
        self.agent.abort()
        await self.agent.wait_idle()
        self.store.close()

    def _on_agent_event(self, event: AgentEvent) -> None:
        match event:
            case AgentStart():
                self._emit(RunStart())
            case AgentEnd():
                self._emit(RunEnd(reason=event.reason, error_message=event.error_message))
                self.run_id = None
                self._emit(StateChanged(state=self.state()))
            case MessageEnd():
                self.store.append_message(event.message)
                self._emit(event)
            case _:
                self._emit(event)

    def _emit(self, event: SessionEvent) -> None:
        self.seq += 1
        envelope = EventEnvelope(session_id=self.id, seq=self.seq, run_id=self.run_id, event=event)
        for listener in list(self._listeners):
            listener(envelope)


def new_session(setup: SessionSetup, cwd: str, model: Model, thinking: ThinkingLevel) -> AgentSession:
    """A new session; its first entries record the model and thinking level."""
    store = SessionStore.create(cwd, _directory(setup, cwd))
    store.append_model_change(model.provider, model.id)
    store.append_thinking_change(thinking)
    return _build(setup, store, model, thinking)


def open_session(setup: SessionSetup, path: Path) -> AgentSession:
    """Load a session file; an unavailable recorded model falls back to the configured default."""
    store = SessionStore.open(path)
    context = store.context()
    try:
        model = resolve_model(setup.providers, context.provider or "", context.model_id)
    except ModelError:
        model = select_model(setup.config, setup.providers, None, None)
    thinking = context.thinking or setup.config.default_thinking or "off"
    return _build(setup, store, model, thinking, context.messages)


def fork_session(setup: SessionSetup, source: AgentSession, entry_id: str) -> AgentSession:
    """A new session with the source's active branch from the root through `entry_id`."""
    store = source.store.fork(entry_id, _directory(setup, source.store.cwd))
    return _build(setup, store, source.agent.model, source.agent.thinking, store.context().messages)


def _build(
    setup: SessionSetup,
    store: SessionStore,
    model: Model,
    thinking: ThinkingLevel,
    messages: list[Message] | None = None,
) -> AgentSession:
    tools = default_tools()
    agent = Agent(
        model=model,
        system_prompt=build_system_prompt(store.cwd, tools, load_context_files(store.cwd, setup.home)),
        tools=tools,
        cwd=store.cwd,
        thinking=thinking,
        messages=messages,
        provider=setup.provider,
    )
    return AgentSession(store, agent)


def _directory(setup: SessionSetup, cwd: str) -> Path | None:
    return sessions_dir(setup.home, cwd) if setup.persist else None


def _total(replies: list[AssistantMessage]) -> Usage:
    total = Usage()
    for reply in replies:
        usage, cost = reply.usage, reply.usage.cost
        total.input += usage.input
        total.output += usage.output
        total.cache_read += usage.cache_read
        total.cache_write += usage.cache_write
        total.cost.input += cost.input
        total.cost.output += cost.output
        total.cost.cache_read += cost.cache_read
        total.cost.cache_write += cost.cache_write
        total.cost.total += cost.total
    return total
