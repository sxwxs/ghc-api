"""Protocol server for one connection: handles JSON-RPC requests in arrival order and forwards session events."""

import asyncio
import sys
import traceback
from collections.abc import Awaitable, Callable
from typing import Any

import msgspec

from .. import __version__
from ..ai.models import ModelError, has_credentials, resolve_model
from ..coding.agent_session import AgentSession, SessionSetup, fork_session, new_session, open_session
from ..coding.config import select_model
from ..coding.session_events import EventEnvelope, SessionState
from ..coding.session_store import SessionError, find_session, list_sessions, sessions_dir
from ..protocol.jsonrpc import (
    ENTRY_NOT_FOUND,
    INCOMING,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    MODEL_ERROR,
    NOT_INITIALIZED,
    PARSE_ERROR,
    PROTOCOL_VERSION,
    SESSION_NOT_FOUND,
    UNSUPPORTED_VERSION,
    ErrorObject,
    Failure,
    Notification,
    ProtocolError,
    Success,
    encode_line,
)
from ..protocol.types import (
    EVENT_METHOD,
    METHODS,
    AgentCapabilities,
    AgentInfo,
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

Handler = Callable[[Any], Awaitable[msgspec.Struct]]


class Server:
    """Owns the sessions of one connection. `send` writes one encoded line to the client."""

    def __init__(self, send: Callable[[bytes], None], setup: SessionSetup, cwd: str) -> None:
        self.send = send
        self.setup = setup
        self.cwd = cwd
        self.sessions: dict[str, AgentSession] = {}
        self.initialized = False
        self._inbox: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._handlers: dict[str, Handler] = {
            "initialize": self._initialize,
            "session/new": self._new_session,
            "session/load": self._load_session,
            "session/list": self._list_sessions,
            "session/close": self._close_session,
            "session/state": self._state,
            "session/messages": self._messages,
            "session/fork": self._fork,
            "session/set_name": self._set_name,
            "run/prompt": self._prompt,
            "run/abort": self._abort,
            "run/clear_queue": self._clear_queue,
            "config/models": self._models,
            "config/set_model": self._set_model,
            "config/set_thinking": self._set_thinking,
        }

    def receive(self, line: bytes) -> None:
        """Queue one line from the client. Lines are handled one at a time, in order."""
        if self._worker is None:
            self._worker = asyncio.create_task(self._work())
        self._inbox.put_nowait(line)

    async def close(self) -> None:
        """Handle the lines already received, then abort active runs and close every session."""
        if self._worker is not None:
            self._inbox.put_nowait(None)
            await self._worker
        for session in list(self.sessions.values()):
            await session.close()
        self.sessions.clear()

    async def _work(self) -> None:
        while (line := await self._inbox.get()) is not None:
            await self._handle(line)

    async def _handle(self, line: bytes) -> None:
        if not line.strip():
            return
        try:
            message = INCOMING.decode(line)
        except msgspec.DecodeError as error:
            self._fail(None, PARSE_ERROR, f"Parse error: {error}")
            return
        if message.method is None or message.id is None:
            return  # client responses and notifications: none are defined in protocol version 1
        if message.jsonrpc != "2.0":
            self._fail(message.id, INVALID_REQUEST, 'jsonrpc must be "2.0"')
            return
        try:
            result = await self._dispatch(message.method, message.params)
        except ProtocolError as error:
            self._fail(message.id, error.code, error.message)
        except ModelError as error:
            self._fail(message.id, MODEL_ERROR, str(error))
        except Exception as error:
            # The wire is a trust boundary: every request gets an answer, even when a handler fails.
            traceback.print_exc(file=sys.stderr)
            self._fail(message.id, INTERNAL_ERROR, str(error) or type(error).__name__)
        else:
            self.send(encode_line(Success(jsonrpc="2.0", id=message.id, result=result)))

    async def _dispatch(self, method: str, raw: msgspec.Raw) -> msgspec.Struct:
        handler = self._handlers.get(method)
        if handler is None:
            raise ProtocolError(METHOD_NOT_FOUND, f"Unknown method {method}")
        if not self.initialized and method != "initialize":
            raise ProtocolError(NOT_INITIALIZED, "Call initialize first")
        try:
            params = msgspec.json.decode(raw or b"{}", type=METHODS[method][0])
        except msgspec.DecodeError as error:
            raise ProtocolError(INVALID_PARAMS, f"Invalid params for {method}: {error}") from error
        return await handler(params)

    async def _initialize(self, params: InitializeParams) -> InitializeResult:
        if params.protocol_version != PROTOCOL_VERSION:
            raise ProtocolError(
                UNSUPPORTED_VERSION,
                f"Protocol version {params.protocol_version} is not supported; use {PROTOCOL_VERSION}",
            )
        self.initialized = True
        return InitializeResult(
            protocol_version=PROTOCOL_VERSION,
            agent_info=AgentInfo(name="pyagent", version=__version__),
            capabilities=AgentCapabilities(multi_session=True, fork=True, images=True),
        )

    async def _new_session(self, params: NewSessionParams) -> SessionState:
        model = select_model(self.setup.config, self.setup.providers, params.provider, params.model)
        thinking = params.thinking or self.setup.config.default_thinking or "off"
        return self._add(new_session(self.setup, params.cwd or self.cwd, model, thinking))

    async def _load_session(self, params: SessionParams) -> SessionState:
        session = self.sessions.get(params.session_id)
        if session is not None:
            return session.state()
        try:
            path = find_session(self.setup.home, params.session_id)
        except SessionError as error:
            raise ProtocolError(SESSION_NOT_FOUND, str(error)) from error
        return self._add(open_session(self.setup, path))

    async def _list_sessions(self, params: ListSessionsParams) -> ListSessionsResult:
        return ListSessionsResult(sessions=list_sessions(sessions_dir(self.setup.home, params.cwd or self.cwd)))

    async def _close_session(self, params: SessionParams) -> Empty:
        session = self._session(params.session_id)
        del self.sessions[session.id]
        await session.close()
        return Empty()

    async def _state(self, params: SessionParams) -> SessionState:
        return self._session(params.session_id).state()

    async def _messages(self, params: SessionParams) -> MessagesResult:
        return MessagesResult(messages=self._session(params.session_id).messages())

    async def _fork(self, params: ForkParams) -> SessionState:
        source = self._session(params.session_id)
        try:
            return self._add(fork_session(self.setup, source, params.entry_id))
        except SessionError as error:
            raise ProtocolError(ENTRY_NOT_FOUND, str(error)) from error

    async def _set_name(self, params: SetNameParams) -> SessionState:
        session = self._session(params.session_id)
        session.set_name(params.name)
        return session.state()

    async def _prompt(self, params: PromptParams) -> PromptResult:
        if not params.content:
            raise ProtocolError(INVALID_PARAMS, "content must not be empty")
        run_id, disposition = self._session(params.session_id).prompt(params.content, params.delivery)
        return PromptResult(run_id=run_id, disposition=disposition)

    async def _abort(self, params: SessionParams) -> DroppedMessages:
        return DroppedMessages(texts=self._session(params.session_id).abort())

    async def _clear_queue(self, params: SessionParams) -> DroppedMessages:
        return DroppedMessages(texts=self._session(params.session_id).clear_queue())

    async def _models(self, params: Empty) -> ModelsResult:
        models: list[ModelInfo] = []
        for name, provider in self.setup.providers.items():
            ids = list(provider.models)
            if provider.default_model is not None and provider.default_model not in provider.models:
                ids.insert(0, provider.default_model)
            available = provider.api_key_env is None or has_credentials(provider)
            for model_id in ids:
                model = resolve_model(self.setup.providers, name, model_id)
                models.append(
                    ModelInfo(
                        provider=name,
                        id=model_id,
                        context_window=model.context_window,
                        reasoning=model.reasoning,
                        images=model.images,
                        available=available,
                    )
                )
        return ModelsResult(models=models)

    async def _set_model(self, params: SetModelParams) -> SessionState:
        session = self._session(params.session_id)
        session.set_model(resolve_model(self.setup.providers, params.provider, params.model))
        return session.state()

    async def _set_thinking(self, params: SetThinkingParams) -> SessionState:
        session = self._session(params.session_id)
        session.set_thinking(params.level)
        return session.state()

    def _session(self, session_id: str) -> AgentSession:
        session = self.sessions.get(session_id)
        if session is None:
            raise ProtocolError(SESSION_NOT_FOUND, f"Session {session_id} is not open; load it first")
        return session

    def _add(self, session: AgentSession) -> SessionState:
        self.sessions[session.id] = session
        session.subscribe(self._forward)
        return session.state()

    def _forward(self, envelope: EventEnvelope) -> None:
        self.send(encode_line(Notification(jsonrpc="2.0", method=EVENT_METHOD, params=envelope)))

    def _fail(self, request_id: int | str | None, code: int, message: str) -> None:
        self.send(encode_line(Failure(jsonrpc="2.0", id=request_id, error=ErrorObject(code=code, message=message))))
