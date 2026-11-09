"""Python client for the pyagent protocol: request correlation, typed methods, and event delivery over any transport."""

import asyncio
from collections.abc import Callable
from typing import TypeVar

import msgspec

from ..ai.types import Message, TextContent, ThinkingLevel, UserContent
from ..coding.agent_session import Delivery
from ..coding.session_events import EventEnvelope, SessionState
from ..coding.session_store import SessionSummary
from ..protocol.jsonrpc import INCOMING, PROTOCOL_VERSION, Request, encode_line
from ..protocol.types import (
    EVENT_METHOD,
    ClientInfo,
    DroppedMessages,
    Empty,
    ForkParams,
    InitializeParams,
    InitializeResult,
    ListSessionsParams,
    ListSessionsResult,
    MessagesResult,
    ModelInfo,
    ModelsResult,
    NewSessionParams,
    PromptParams,
    PromptResult,
    SessionParams,
    SetModelParams,
    SetNameParams,
    SetThinkingParams,
)

T = TypeVar("T", bound=msgspec.Struct)
EventListener = Callable[[EventEnvelope], None]
ENVELOPE = msgspec.json.Decoder(EventEnvelope)


class RemoteError(Exception):
    """The server answered a request with an error."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"{message} ({code})")
        self.code = code
        self.message = message


class Client:
    """Talks to one server. `send` writes one encoded line; the transport passes every incoming line to `receive`."""

    def __init__(self, send: Callable[[bytes], None]) -> None:
        self._send = send
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[msgspec.Raw]] = {}
        self._listeners: list[EventListener] = []

    def receive(self, line: bytes) -> None:
        """Handle one line from the server: a response or an event."""
        message = INCOMING.decode(line)
        if message.method == EVENT_METHOD:
            try:
                envelope = ENVELOPE.decode(message.params)
            except msgspec.ValidationError:
                return  # an event type this client does not know; the protocol says to ignore it
            for listener in list(self._listeners):
                listener(envelope)
            return
        future = self._pending.pop(message.id, None) if isinstance(message.id, int) else None
        if future is None or future.done():
            return
        if message.error is not None:
            future.set_exception(RemoteError(message.error.code, message.error.message))
        else:
            future.set_result(message.result)

    def close(self) -> None:
        """Fail all pending requests; the connection is gone."""
        for future in self._pending.values():
            if not future.done():
                future.set_exception(ConnectionError("Connection to the pyagent server closed"))
        self._pending.clear()

    def on_event(self, listener: EventListener) -> Callable[[], None]:
        """Receive every session event; returns an unsubscribe function."""
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    async def call(self, method: str, params: msgspec.Struct, result_type: type[T]) -> T:
        """Send a request and decode its result."""
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future[msgspec.Raw] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        self._send(encode_line(Request(jsonrpc="2.0", id=request_id, method=method, params=params)))
        return msgspec.json.decode(await future, type=result_type)

    async def initialize(self, client_name: str = "pyagent-client") -> InitializeResult:
        """Negotiate the protocol version; must be the first call."""
        params = InitializeParams(protocol_version=PROTOCOL_VERSION, client_info=ClientInfo(name=client_name))
        return await self.call("initialize", params, InitializeResult)

    async def new_session(
        self,
        cwd: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        thinking: ThinkingLevel | None = None,
    ) -> SessionState:
        """Create a session; unset arguments use the server's defaults."""
        params = NewSessionParams(cwd=cwd, provider=provider, model=model, thinking=thinking)
        return await self.call("session/new", params, SessionState)

    async def load_session(self, session_id: str) -> SessionState:
        """Open a stored session."""
        return await self.call("session/load", SessionParams(session_id=session_id), SessionState)

    async def list_sessions(self, cwd: str | None = None) -> list[SessionSummary]:
        """Stored sessions of a working directory, most recent first."""
        return (await self.call("session/list", ListSessionsParams(cwd=cwd), ListSessionsResult)).sessions

    async def close_session(self, session_id: str) -> None:
        """Abort its run and release it on the server."""
        await self.call("session/close", SessionParams(session_id=session_id), Empty)

    async def state(self, session_id: str) -> SessionState:
        """The session's state snapshot."""
        return await self.call("session/state", SessionParams(session_id=session_id), SessionState)

    async def messages(self, session_id: str) -> list[Message]:
        """The active branch's messages."""
        return (await self.call("session/messages", SessionParams(session_id=session_id), MessagesResult)).messages

    async def fork(self, session_id: str, entry_id: str) -> SessionState:
        """A new session ending with message `entry_id`."""
        return await self.call("session/fork", ForkParams(session_id=session_id, entry_id=entry_id), SessionState)

    async def set_name(self, session_id: str, name: str) -> SessionState:
        """Set the session's display name."""
        return await self.call("session/set_name", SetNameParams(session_id=session_id, name=name), SessionState)

    async def prompt(
        self, session_id: str, content: str | list[UserContent], delivery: Delivery = "auto"
    ) -> PromptResult:
        """Send a user message: starts a run when idle, else queues it."""
        blocks: list[UserContent] = [TextContent(text=content)] if isinstance(content, str) else content
        params = PromptParams(session_id=session_id, content=blocks, delivery=delivery)
        return await self.call("run/prompt", params, PromptResult)

    async def abort(self, session_id: str) -> list[str]:
        """Abort the active run; returns the texts of dropped queued messages."""
        return (await self.call("run/abort", SessionParams(session_id=session_id), DroppedMessages)).texts

    async def clear_queue(self, session_id: str) -> list[str]:
        """Drop queued messages; returns their texts."""
        return (await self.call("run/clear_queue", SessionParams(session_id=session_id), DroppedMessages)).texts

    async def models(self) -> list[ModelInfo]:
        """Models of all configured providers."""
        return (await self.call("config/models", Empty(), ModelsResult)).models

    async def set_model(self, session_id: str, provider: str, model: str | None = None) -> SessionState:
        """Switch the session's model from the next run on."""
        params = SetModelParams(session_id=session_id, provider=provider, model=model)
        return await self.call("config/set_model", params, SessionState)

    async def set_thinking(self, session_id: str, level: ThinkingLevel) -> SessionState:
        """Switch the session's thinking level from the next run on."""
        return await self.call(
            "config/set_thinking", SetThinkingParams(session_id=session_id, level=level), SessionState
        )
