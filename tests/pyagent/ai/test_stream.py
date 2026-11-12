"""stream_assistant: one ending path for success, provider exceptions and cancellation."""

import asyncio

import pytest

from ghc_api.pyagent.ai.events import MessageEvent
from ghc_api.pyagent.ai.models import Model
from ghc_api.pyagent.ai.provider import StreamOptions
from ghc_api.pyagent.ai.providers.faux import FauxProvider, Reply, text_reply
from ghc_api.pyagent.ai.stream import empty_reply, stream_assistant
from ghc_api.pyagent.ai.types import Context

pytestmark = pytest.mark.anyio

CONTEXT = Context(system_prompt="s", messages=[], tools=[])


async def test_provider_exceptions_become_error_replies(model: Model) -> None:
    events: list[MessageEvent] = []
    reply = empty_reply(model)
    await stream_assistant(model, CONTEXT, StreamOptions(), reply, events.append, FauxProvider([Reply(error="boom")]))
    assert (reply.stop_reason, reply.error_message) == ("error", "boom")
    assert [e.type for e in events] == ["message_start", "message_end"]


async def test_cancellation_marks_the_reply_aborted_and_propagates(model: Model) -> None:
    events: list[MessageEvent] = []
    reply = empty_reply(model)
    provider = FauxProvider([text_reply("x" * 100, delay=0.05)], chunk_size=1)
    task = asyncio.create_task(stream_assistant(model, CONTEXT, StreamOptions(), reply, events.append, provider))
    await asyncio.sleep(0.12)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert reply.stop_reason == "aborted"
    assert events[-1].type == "message_end"
    assert 0 < len(reply.text()) < 100
