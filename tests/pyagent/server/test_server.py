"""The protocol end to end through an in-process client: lifecycle, events, queueing, persistence, errors."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import msgspec
import pytest

from ghc_api.pyagent.ai.models import ProviderOverride, merge_providers
from ghc_api.pyagent.ai.providers.faux import FauxProvider, Reply, text_reply, tool_reply
from ghc_api.pyagent.client.client import Client, RemoteError
from ghc_api.pyagent.client.transports import connect_in_process
from ghc_api.pyagent.coding.agent_session import SessionSetup
from ghc_api.pyagent.coding.config import Config
from ghc_api.pyagent.coding.session_events import EventEnvelope, RunEnd, StateChanged
from ghc_api.pyagent.protocol import jsonrpc
from ghc_api.pyagent.protocol.types import METHODS
from ghc_api.pyagent.server.server import Server

pytestmark = pytest.mark.anyio


class Harness:
    """A client and server over one scripted provider, with every event recorded."""

    def __init__(self, tmp_path: Path, replies: list[Reply]) -> None:
        config = Config(
            default_provider="test",
            default_model="m1",
            providers={"test": ProviderOverride(api="openai_chat", base_url="http://unused")},
        )
        self.provider = FauxProvider(list(replies))
        self.setup = SessionSetup(
            home=tmp_path, config=config, providers=merge_providers(config.providers), provider=self.provider
        )
        self.cwd = str(tmp_path / "work")
        self.client, self.server = connect_in_process(self.setup, self.cwd)
        self.events: list[EventEnvelope] = []
        self.client.on_event(self.events.append)

    async def run(self, session_id: str, text: str) -> RunEnd:
        """Prompt and wait for the run to end."""
        ended: asyncio.Future[RunEnd] = asyncio.get_running_loop().create_future()

        def on_event(envelope: EventEnvelope) -> None:
            if isinstance(envelope.event, RunEnd) and not ended.done():
                ended.set_result(envelope.event)

        unsubscribe = self.client.on_event(on_event)
        await self.client.prompt(session_id, text)
        try:
            return await asyncio.wait_for(ended, 5)
        finally:
            unsubscribe()


@pytest.fixture
async def harness(tmp_path: Path) -> AsyncIterator[Harness]:
    harness = Harness(tmp_path, [])
    yield harness
    await harness.server.close()


def types(events: list[EventEnvelope]) -> list[str]:
    return [e.event.type for e in events if e.event.type not in ("block_start", "block_delta", "block_end")]


async def test_requests_before_initialize_are_rejected(harness: Harness) -> None:
    with pytest.raises(RemoteError) as error:
        await harness.client.new_session()
    assert error.value.code == jsonrpc.NOT_INITIALIZED


async def test_initialize_negotiates_the_version(harness: Harness) -> None:
    result = await harness.client.initialize("test")
    assert (result.protocol_version, result.agent_info.name, result.capabilities.multi_session) == (1, "pyagent", True)
    with pytest.raises(RemoteError) as error:
        await harness.client.call(
            "initialize", msgspec.json.decode(b'{"protocolVersion": 9}', type=METHODS["initialize"][0]), type(result)
        )
    assert error.value.code == jsonrpc.UNSUPPORTED_VERSION


async def test_malformed_requests_get_standard_errors(harness: Harness) -> None:
    replies: list[bytes] = []
    server = Server(replies.append, harness.setup, harness.cwd)
    server.receive(b"{not json\n")
    server.receive(b'{"jsonrpc":"2.0","id":1,"method":"nope"}\n')
    server.receive(b'{"jsonrpc":"2.0","id":2,"method":"initialize","params":{"protocolVersion":"one"}}\n')
    await server.close()
    errors = [msgspec.json.decode(r)["error"] for r in replies]
    assert [e["code"] for e in errors] == [jsonrpc.PARSE_ERROR, jsonrpc.METHOD_NOT_FOUND, jsonrpc.INVALID_PARAMS]
    assert "Expected `int`, got `str` - at `$.protocolVersion`" in errors[2]["message"]


async def test_a_prompt_streams_numbered_events_and_ends_the_run(harness: Harness) -> None:
    harness.provider.replies += [tool_reply(("c1", "read", {"path": "missing.txt"})), text_reply("done")]
    await harness.client.initialize()
    state = await harness.client.new_session()
    end = await harness.run(state.session_id, "hi")

    assert end.reason == "completed"
    assert types(harness.events) == [
        "run_start",
        "message_start",
        "message_end",  # user
        "turn_start",
        "message_start",
        "message_end",  # tool call
        "tool_start",
        "tool_end",
        "message_start",
        "message_end",  # tool result
        "turn_end",
        "turn_start",
        "message_start",
        "message_end",  # answer
        "turn_end",
        "run_end",
        "state_changed",
    ]
    assert [e.seq for e in harness.events] == list(range(1, len(harness.events) + 1))
    run_ids = {e.run_id for e in harness.events[:-1]}
    assert len(run_ids) == 1 and None not in run_ids
    assert harness.events[-1].run_id is None
    final = harness.events[-1].event
    assert isinstance(final, StateChanged) and not final.state.running
    assert final.state.last_seq == harness.events[-1].seq - 1


async def test_prompts_during_a_run_are_queued(harness: Harness) -> None:
    harness.provider.replies += [text_reply("x" * 40, delay=0.01), text_reply("second")]
    await harness.client.initialize()
    state = await harness.client.new_session()
    first = await harness.client.prompt(state.session_id, "start")
    second = await harness.client.prompt(state.session_id, "and this", delivery="follow_up")
    assert (first.disposition, second.disposition, second.run_id) == ("started", "queued", first.run_id)
    while not any(isinstance(e.event, RunEnd) for e in harness.events):
        await asyncio.sleep(0.01)
    assert len(harness.provider.contexts) == 2


async def test_abort_ends_the_run_as_aborted(harness: Harness) -> None:
    harness.provider.replies.append(text_reply("x" * 400, delay=0.01))
    await harness.client.initialize()
    state = await harness.client.new_session()
    await harness.client.prompt(state.session_id, "long")
    await asyncio.sleep(0.05)
    assert await harness.client.abort(state.session_id) == []
    while not any(isinstance(e.event, RunEnd) for e in harness.events):
        await asyncio.sleep(0.01)
    ends = [e.event for e in harness.events if isinstance(e.event, RunEnd)]
    assert ends[0].reason == "aborted"


async def test_sessions_persist_and_reload_with_their_settings(harness: Harness) -> None:
    harness.provider.replies.append(text_reply("hello"))
    await harness.client.initialize()
    state = await harness.client.new_session(thinking="low")
    await harness.run(state.session_id, "first question")
    await harness.client.set_name(state.session_id, "demo")
    await harness.client.set_thinking(state.session_id, "high")
    before = await harness.client.messages(state.session_id)
    await harness.client.close_session(state.session_id)

    [summary] = await harness.client.list_sessions()
    assert (summary.session_id, summary.name, summary.first_message, summary.message_count) == (
        state.session_id,
        "demo",
        "first question",
        2,
    )
    reloaded = await harness.client.load_session(state.session_id)
    assert (reloaded.thinking, reloaded.name, reloaded.model.id) == ("high", "demo", "m1")
    assert await harness.client.messages(state.session_id) == before


async def test_fork_starts_a_new_session_from_a_message(harness: Harness) -> None:
    harness.provider.replies += [text_reply("one"), text_reply("two")]
    await harness.client.initialize()
    state = await harness.client.new_session()
    await harness.run(state.session_id, "q1")
    await harness.run(state.session_id, "q2")
    messages = await harness.client.messages(state.session_id)
    fork = await harness.client.fork(state.session_id, messages[1].id)
    assert fork.session_id != state.session_id
    assert [m.id for m in await harness.client.messages(fork.session_id)] == [m.id for m in messages[:2]]
    with pytest.raises(RemoteError) as error:
        await harness.client.fork(state.session_id, "nope")
    assert error.value.code == jsonrpc.ENTRY_NOT_FOUND


async def test_settings_and_unknown_sessions(harness: Harness) -> None:
    await harness.client.initialize()
    state = await harness.client.new_session()
    changed = await harness.client.set_model(state.session_id, "test", "m2")
    assert changed.model.id == "m2"
    assert isinstance(harness.events[-1].event, StateChanged)
    with pytest.raises(RemoteError) as error:
        await harness.client.set_model(state.session_id, "nope")
    assert error.value.code == jsonrpc.MODEL_ERROR
    with pytest.raises(RemoteError) as error:
        await harness.client.state("missing")
    assert error.value.code == jsonrpc.SESSION_NOT_FOUND
    models = await harness.client.models()
    assert any(m.provider == "deepseek" and m.id == "deepseek-v4-pro" for m in models)


def test_every_method_has_a_handler(tmp_path: Path) -> None:
    server = Server(lambda _: None, Harness(tmp_path, []).setup, str(tmp_path))
    assert set(server._handlers) == set(METHODS)  # pyright: ignore[reportPrivateUsage]


def test_client_ignores_unknown_event_types() -> None:
    received: list[EventEnvelope] = []
    client = Client(lambda _: None)
    client.on_event(received.append)
    client.receive(
        b'{"jsonrpc":"2.0","method":"session/event","params":{"sessionId":"s","seq":1,"event":{"type":"future_event"}}}'
    )
    client.receive(
        b'{"jsonrpc":"2.0","method":"session/event","params":{"sessionId":"s","seq":2,"event":{"type":"run_start"}}}'
    )
    assert [e.seq for e in received] == [2]
