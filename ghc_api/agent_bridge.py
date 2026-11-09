"""Runs pyagent (ghc_api/pyagent) inside ghc-api for the /agent page.

pyagent's protocol server runs on a background asyncio loop. Flask threads send it
the same JSON-RPC lines a stdio client would (docs/pyagent/protocol.md), and the
events of each session are kept for the current run so a browser can follow them
over SSE and catch up after a reconnect. Apart from the protocol, this module
only touches pyagent's setup objects.

The agent calls models through ghc-api itself: one provider, "ghc-api", points at
this server with the internal token, and lists the Copilot catalog's chat models.
Each model uses the endpoint it supports best: Anthropic Messages, then
Responses, then Chat Completions.
"""

import asyncio
import concurrent.futures
import itertools
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .auth import INTERNAL_TOKEN
from .pyagent.ai.models import ModelSpec, ProviderOverride, merge_providers
from .pyagent.coding.agent_session import SessionSetup
from .pyagent.coding.config import Config
from .pyagent.protocol.jsonrpc import PROTOCOL_VERSION
from .pyagent.server.server import Server
from .state import state
from .utils import get_config_dir

PROVIDER = "ghc-api"
CALL_TIMEOUT = 120.0
# A run's events are dropped when the next run starts; this caps one very long run.
MAX_RUN_EVENTS = 50_000


# ---------------------------------------------------------------------------
# Setup: the ghc-api provider
# ---------------------------------------------------------------------------

def _api_for(endpoints: List[str]) -> Optional[str]:
    if "/v1/messages" in endpoints or "/messages" in endpoints:
        return "anthropic"
    if "/responses" in endpoints or "/v1/responses" in endpoints:
        return "openai_responses"
    if not endpoints or "/chat/completions" in endpoints or "/v1/chat/completions" in endpoints:
        return "openai_chat"
    return None


def catalog_models(catalog: List[Dict[str, Any]]) -> Dict[str, ModelSpec]:
    """pyagent model specs for the chat models of a Copilot model catalog, in catalog order."""
    models: Dict[str, ModelSpec] = {}
    for entry in catalog:
        capabilities = entry.get("capabilities") or {}
        supports = capabilities.get("supports") or {}
        limits = capabilities.get("limits") or {}
        model_id = entry.get("id")
        api = _api_for(entry.get("supported_endpoints") or [])
        if (
            not isinstance(model_id, str) or model_id in models or api is None
            or capabilities.get("type", "chat") != "chat" or supports.get("tool_calls") is False
        ):
            continue
        efforts = bool(supports.get("reasoning_effort"))
        reasoning = efforts or bool(supports.get("adaptive_thinking") or supports.get("max_thinking_budget"))
        # ghc-api's /v1/messages turns budget thinking into adaptive thinking for models that need it.
        thinking_format = {"anthropic": "anthropic_budget" if reasoning else "none",
                           "openai_chat": "reasoning_effort" if efforts else "none"}.get(api)
        models[model_id] = ModelSpec(
            api=api,
            context_window=limits.get("max_context_window_tokens") or 200_000,
            max_tokens=limits.get("max_output_tokens") or 32_000,
            reasoning=reasoning,
            images=bool(supports.get("vision")),
            thinking_format=thinking_format,
        )
    return models


def self_url() -> str:
    """This server's address as seen from inside the process."""
    host = state.server_host
    if host in ("", "0.0.0.0"):
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{state.server_port}"


def ghc_setup() -> SessionSetup:
    """pyagent's setup for the /agent page: its home in the ghc-api config dir and the ghc-api provider."""
    models = catalog_models((state.models or {}).get("data") or [])
    default = next((m for m, spec in models.items() if spec.api != "openai_chat"), next(iter(models), None))
    provider = ProviderOverride(
        api="openai_responses",  # each model sets its own
        base_url=self_url(),
        api_key=INTERNAL_TOKEN,
        default_model=default,
        models=models,
    )
    return SessionSetup(
        home=Path(get_config_dir()) / "pyagent",
        config=Config(default_provider=PROVIDER, default_model=default, default_thinking="medium"),
        providers=merge_providers({PROVIDER: provider}),
    )


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class _RunEvents:
    """The current run's event envelopes of one session, as JSON text, with consecutive seqs."""

    def __init__(self, last_seq: int) -> None:
        self.events: List[Tuple[int, str]] = []
        self.last_seq = last_seq

    def add(self, seq: int, envelope: Dict[str, Any]) -> None:
        if envelope["event"].get("type") == "run_start" or len(self.events) >= MAX_RUN_EVENTS:
            self.events = []
        self.events.append((seq, json.dumps(envelope, ensure_ascii=False)))
        self.last_seq = seq

    def since(self, seq: int) -> Optional[List[Tuple[int, str]]]:
        """Events after `seq`, or None when some of them are gone and the client must reload."""
        if seq > self.last_seq:
            return None  # the session was reopened and counts from zero again
        first = self.events[0][0] if self.events else self.last_seq + 1
        if seq + 1 < first:
            return None
        return self.events[seq + 1 - first:]


class AgentBridge:
    """One pyagent protocol server for the whole process, started on first use."""

    def __init__(self, make_setup: Callable[[], SessionSetup] = ghc_setup) -> None:
        self._make_setup = make_setup
        self._start_lock = threading.Lock()
        self._lock = threading.Condition()
        self._ids = itertools.count(1)
        self._pending: Dict[int, Tuple[str, Dict[str, Any], concurrent.futures.Future]] = {}
        self._runs: Dict[str, _RunEvents] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server: Optional[Server] = None
        self._setup: Optional[SessionSetup] = None

    def call(self, method: str, params: Dict[str, Any], timeout: float = CALL_TIMEOUT) -> Dict[str, Any]:
        """Send one request; returns the JSON-RPC response object ({"result": ...} or {"error": ...})."""
        self._start()
        if method in ("session/new", "config/models"):
            self._refresh_models()
        return self._send(method, params).result(timeout)

    def wait_events(self, session_id: str, since: int, timeout: float) -> Optional[List[Tuple[int, str]]]:
        """Events of an open session after `since`, waiting up to `timeout` for one. An empty list means none
        arrived; None means the client must reload the session (unknown session, or events already dropped)."""
        deadline = time.monotonic() + timeout
        with self._lock:
            while True:
                run = self._runs.get(session_id)
                events = run.since(since) if run is not None else None
                remaining = deadline - time.monotonic()
                if events is None or events or remaining <= 0:
                    return events
                self._lock.wait(remaining)

    def close(self) -> None:
        """Abort runs, close the sessions and stop the loop. The process-wide bridge lives as long as ghc-api."""
        with self._start_lock:
            if self._server is None:
                return
            asyncio.run_coroutine_threadsafe(self._server.close(), self._loop).result(CALL_TIMEOUT)
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._server = None

    # --- internals ---

    def _start(self) -> None:
        with self._start_lock:
            if self._server is not None:
                return
            setup = self._make_setup()
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, name="pyagent", daemon=True).start()

            async def make_server() -> Server:
                return Server(self._on_line, setup, os.path.expanduser("~"))

            self._server = asyncio.run_coroutine_threadsafe(make_server(), loop).result()
            self._loop, self._setup = loop, setup
            reply = self._send("initialize", {
                "protocolVersion": PROTOCOL_VERSION, "clientInfo": {"name": "ghc-api"},
            }).result(CALL_TIMEOUT)
            if "error" in reply:
                raise RuntimeError(f"pyagent initialize failed: {reply['error'].get('message')}")

    def _refresh_models(self) -> None:
        """Follow the current model catalog; applied on the loop thread before the next request."""
        fresh = self._make_setup()

        def apply() -> None:
            self._setup.config, self._setup.providers = fresh.config, fresh.providers

        self._loop.call_soon_threadsafe(apply)

    def _send(self, method: str, params: Dict[str, Any]) -> concurrent.futures.Future:
        future: concurrent.futures.Future = concurrent.futures.Future()
        with self._lock:
            request_id = next(self._ids)
            self._pending[request_id] = (method, params, future)
        line = json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        self._loop.call_soon_threadsafe(self._server.receive, line.encode("utf-8") + b"\n")
        return future

    def _on_line(self, line: bytes) -> None:
        """Every line the server sends, on the loop thread."""
        message = json.loads(line)
        with self._lock:
            if message.get("method") == "session/event":
                envelope = message["params"]
                run = self._runs.get(envelope["sessionId"])
                if run is not None:
                    run.add(envelope["seq"], envelope)
                    self._lock.notify_all()
                return
            method, params, future = self._pending.pop(message.get("id"), (None, None, None))
            result = message.get("result")
            if isinstance(result, dict) and "sessionId" in result and "lastSeq" in result:
                self._runs.setdefault(result["sessionId"], _RunEvents(result["lastSeq"]))
            elif method == "session/close" and "error" not in message:
                self._runs.pop(params.get("sessionId"), None)
                self._lock.notify_all()
        if future is not None:
            future.set_result(message)


_bridge: Optional[AgentBridge] = None
_bridge_lock = threading.Lock()


def get_bridge() -> AgentBridge:
    global _bridge
    with _bridge_lock:
        if _bridge is None:
            _bridge = AgentBridge()
        return _bridge
