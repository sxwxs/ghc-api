"""Agent: owns the transcript and settings, runs the loop as a task, and queues steering and follow-up messages."""

import asyncio
from collections.abc import Callable
from typing import Any

from ..ai.models import Model, api_key_for
from ..ai.provider import ProviderFn, StreamOptions
from ..ai.types import Message, TextContent, ThinkingLevel, UserMessage, user_message
from .events import AgentEnd, AgentEvent, QueueChanged
from .loop import BeforeToolCall, RunState, run
from .tool import Tool

Listener = Callable[[AgentEvent], None]


class AgentBusyError(RuntimeError):
    """A run is already active."""


class AgentIdleError(RuntimeError):
    """The operation needs an active run."""


class Agent:
    """Runs prompts against a model with tools. Settings changed during a run apply to the next run."""

    def __init__(
        self,
        *,
        model: Model,
        system_prompt: str,
        tools: list[Tool[Any]],
        cwd: str,
        thinking: ThinkingLevel = "off",
        messages: list[Message] | None = None,
        get_api_key: Callable[[Model], str | None] = api_key_for,
        provider: ProviderFn | None = None,
        before_tool_call: BeforeToolCall | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools
        self.cwd = cwd
        self.thinking: ThinkingLevel = thinking
        self.messages: list[Message] = messages if messages is not None else []
        self.get_api_key = get_api_key
        self.provider = provider
        self.before_tool_call = before_tool_call
        self._listeners: list[Listener] = []
        self._steering: list[Message] = []
        self._follow_up: list[Message] = []
        self._task: asyncio.Task[AgentEnd] | None = None
        self._state: RunState | None = None

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        """Receive every event; returns an unsubscribe function."""
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    @property
    def running(self) -> bool:
        """Whether a run is active."""
        return self._task is not None and not self._task.done()

    def start(self, prompt: str | list[Message]) -> "asyncio.Task[AgentEnd]":
        """Start a run and return its task. Raises AgentBusyError while a run is active."""
        if self.running:
            raise AgentBusyError("A run is already active; use steer() or follow_up()")
        state = RunState(
            model=self.model,
            system_prompt=self.system_prompt,
            tools={tool.name: tool for tool in self.tools},
            messages=self.messages,
            options=StreamOptions(api_key=self.get_api_key(self.model), thinking=self.thinking),
            cwd=self.cwd,
            emit=self._emit,
            take_steering=self._take_steering,
            take_follow_up=self._take_follow_up,
            provider=self.provider,
            before_tool_call=self.before_tool_call,
        )
        self._state = state
        self._task = asyncio.create_task(run(state, [user_message(prompt)] if isinstance(prompt, str) else prompt))
        return self._task

    async def prompt(self, prompt: str | list[Message]) -> AgentEnd:
        """Run a prompt to completion."""
        return await self.start(prompt)

    def steer(self, message: UserMessage) -> None:
        """Queue a message the model sees after the current turn's tool calls. Requires an active run."""
        self._require_running()
        self._steering.append(message)
        self._emit_queue()

    def follow_up(self, message: UserMessage) -> None:
        """Queue a message the model sees once it would otherwise stop. Requires an active run."""
        self._require_running()
        self._follow_up.append(message)
        self._emit_queue()

    def queued(self) -> tuple[list[str], list[str]]:
        """The texts of the queued steering and follow-up messages."""
        return [_text(m) for m in self._steering], [_text(m) for m in self._follow_up]

    def clear_queue(self) -> list[str]:
        """Drop all queued messages and return their texts, so a UI can restore them to the editor."""
        dropped = [_text(m) for m in self._steering + self._follow_up]
        if dropped:
            self._steering, self._follow_up = [], []
            self._emit_queue()
        return dropped

    def abort(self) -> list[str]:
        """Abort the active run; queued messages are dropped and their texts returned."""
        dropped = self.clear_queue()
        if self._task is not None and self._state is not None and self.running:
            self._state.abort_requested = True
            if self._state.started:
                self._task.cancel()
        return dropped

    async def wait_idle(self) -> None:
        """Wait until no run is active."""
        if self._task is not None:
            await asyncio.wait({self._task})

    def _emit(self, event: AgentEvent) -> None:
        for listener in list(self._listeners):
            listener(event)

    def _require_running(self) -> None:
        if not self.running:
            raise AgentIdleError("No active run; use start()")

    def _take_steering(self) -> list[Message]:
        taken, self._steering = self._steering, []
        if taken:
            self._emit_queue()
        return taken

    def _take_follow_up(self) -> list[Message]:
        taken, self._follow_up = self._follow_up, []
        if taken:
            self._emit_queue()
        return taken

    def _emit_queue(self) -> None:
        steering, follow_up = self.queued()
        self._emit(QueueChanged(steering=steering, follow_up=follow_up))


def _text(message: Message) -> str:
    if isinstance(message, UserMessage):
        return "".join(block.text for block in message.content if isinstance(block, TextContent))
    return ""
