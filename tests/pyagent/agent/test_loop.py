"""Agent loop through the Agent class with scripted replies: turns, tools, hooks, queues, errors and aborts."""

import asyncio
from typing import Any

import msgspec
import pytest

from ghc_api.pyagent.agent.agent import Agent, AgentBusyError, AgentIdleError
from ghc_api.pyagent.agent.events import AgentEvent, ToolEnd, ToolStart
from ghc_api.pyagent.agent.tool import Tool, ToolContext, ToolResult, text_result
from ghc_api.pyagent.ai.models import Model
from ghc_api.pyagent.ai.providers.faux import FauxProvider, Reply, text_reply, tool_reply
from ghc_api.pyagent.ai.types import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    Wire,
    user_message,
)

pytestmark = pytest.mark.anyio


class EchoParams(Wire):
    text: str


class EchoTool(Tool[EchoParams]):
    """Returns its text after an optional wait, reporting one partial update."""

    name = "echo"
    description = "Echo text"
    params = EchoParams
    kind = "other"

    def __init__(self, parallel_safe: bool = False, wait: asyncio.Event | None = None) -> None:
        self.parallel_safe = parallel_safe
        self.wait = wait
        self.running = 0
        self.max_running = 0

    def title(self, params: EchoParams) -> str:
        return f"echo {params.text}"

    async def execute(self, params: EchoParams, ctx: ToolContext) -> ToolResult:
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        try:
            ctx.on_update(text_result("working"))
            if self.wait is not None:
                await self.wait.wait()
            else:
                await asyncio.sleep(0.01)
            return text_result(params.text)
        finally:
            self.running -= 1


def make_agent(
    model: Model, replies: list[Any], tools: list[Tool[Any]] | None = None
) -> tuple[Agent, FauxProvider, list[AgentEvent]]:
    provider = FauxProvider(replies)
    agent = Agent(model=model, system_prompt="sys", tools=tools or [EchoTool()], cwd=".", provider=provider)
    events: list[AgentEvent] = []
    agent.subscribe(events.append)
    return agent, provider, events


def types(events: list[AgentEvent]) -> list[str]:
    return [e.type for e in events if e.type not in ("block_start", "block_delta", "block_end")]


async def test_a_text_reply_is_one_turn(model: Model) -> None:
    agent, _, events = make_agent(model, [text_reply("hello")])
    end = await agent.prompt("hi")
    assert end.reason == "completed"
    assert types(events) == [
        "agent_start",
        "message_start",  # user
        "message_end",
        "turn_start",
        "message_start",  # assistant
        "message_end",
        "turn_end",
        "agent_end",
    ]
    assert [type(m) for m in agent.messages] == [UserMessage, AssistantMessage]
    assert agent.messages[-1].content == [TextContent(text="hello")]


async def test_tool_calls_run_and_their_results_reach_the_next_request(model: Model) -> None:
    agent, provider, events = make_agent(model, [tool_reply(("c1", "echo", {"text": "A"})), text_reply("done")])
    await agent.prompt("go")
    assert types(events)[4:12] == [
        "message_start",
        "message_end",
        "tool_start",
        "tool_update",
        "tool_end",
        "message_start",  # tool result
        "message_end",
        "turn_end",
    ]
    start = next(e for e in events if isinstance(e, ToolStart))
    assert (start.tool_call_id, start.title, start.kind) == ("c1", "echo A", "other")
    result = provider.contexts[1].messages[-1]
    assert isinstance(result, ToolResultMessage)
    assert (result.tool_call_id, result.content[0], result.is_error) == ("c1", text_result("A").content[0], False)
    assert agent.messages[-1].content == text_reply("done").blocks


async def test_a_failing_listener_ends_the_run_with_an_error(model: Model) -> None:
    agent, _, events = make_agent(model, [tool_reply(("c1", "echo", {"text": "A"})), text_reply("later")])

    def fail_on_tool_end(event: AgentEvent) -> None:  # like a session store that cannot write
        if isinstance(event, ToolEnd):
            raise OSError("disk full")

    unsubscribe = agent.subscribe(fail_on_tool_end)
    end = await agent.prompt("go")
    assert (end.reason, end.error_message) == ("error", "OSError: disk full")
    assert types(events)[-1] == "agent_end" and not agent.running
    unsubscribe()
    assert (await agent.prompt("again")).reason == "completed"


async def test_unknown_tools_and_invalid_arguments_become_error_results(model: Model) -> None:
    agent, provider, _ = make_agent(
        model, [tool_reply(("c1", "nope", {}), ("c2", "echo", {"text": 5})), text_reply("ok")]
    )
    await agent.prompt("go")
    results = [m for m in provider.contexts[1].messages if isinstance(m, ToolResultMessage)]
    assert [r.is_error for r in results] == [True, True]
    assert results[0].content == [TextContent(text="Tool 'nope' not found")]
    assert results[1].content == [
        TextContent(text="Invalid arguments for echo: Expected `str`, got `int` - at `$.text`")
    ]


async def test_parallel_safe_tools_run_concurrently(model: Model) -> None:
    gate = asyncio.Event()
    tool = EchoTool(parallel_safe=True, wait=gate)
    agent, _, _ = make_agent(
        model, [tool_reply(("c1", "echo", {"text": "A"}), ("c2", "echo", {"text": "B"})), text_reply("ok")], [tool]
    )
    task = agent.start("go")
    while tool.running < 2:
        await asyncio.sleep(0.001)
    gate.set()
    await task
    assert tool.max_running == 2
    results = [m.tool_call_id for m in agent.messages if isinstance(m, ToolResultMessage)]
    assert results == ["c1", "c2"]  # source order, whatever the completion order


async def test_other_tools_run_one_at_a_time(model: Model) -> None:
    tool = EchoTool(parallel_safe=False)
    agent, _, _ = make_agent(
        model, [tool_reply(("c1", "echo", {"text": "A"}), ("c2", "echo", {"text": "B"})), text_reply("ok")], [tool]
    )
    await agent.prompt("go")
    assert tool.max_running == 1


async def test_before_tool_call_can_block_a_call(model: Model) -> None:
    agent, provider, events = make_agent(model, [tool_reply(("c1", "echo", {"text": "A"})), text_reply("ok")])
    seen: list[tuple[str, Any]] = []

    async def deny(call: ToolCall, tool: Tool[Any], params: Any) -> str | None:
        seen.append((call.id, params))
        return "Denied by user"

    agent.before_tool_call = deny
    await agent.prompt("go")
    assert seen == [("c1", EchoParams(text="A"))]
    end = next(e for e in events if isinstance(e, ToolEnd))
    assert end.result.is_error and end.duration_ms is None
    result = provider.contexts[1].messages[-1]
    assert isinstance(result, ToolResultMessage) and result.content[0] == text_result("Denied by user").content[0]


async def test_provider_errors_end_the_run_with_the_error(model: Model) -> None:
    agent, _, _ = make_agent(model, [Reply(error="429: slow down")])
    end = await agent.prompt("go")
    assert (end.reason, end.error_message) == ("error", "429: slow down")


async def test_abort_while_streaming_keeps_the_partial_reply_out_of_later_requests(model: Model) -> None:
    agent, provider, events = make_agent(model, [text_reply("x" * 200, delay=0.01), text_reply("again")])
    task = agent.start("go")
    await asyncio.sleep(0.05)
    agent.abort()
    end = await task
    assert end.reason == "aborted"
    reply = agent.messages[-1]
    assert isinstance(reply, AssistantMessage) and reply.stop_reason == "aborted"
    assert types(events)[-3:] == ["message_end", "turn_end", "agent_end"]

    await agent.prompt("next")
    sent = provider.contexts[1].messages
    assert [type(m) for m in sent] == [
        UserMessage,
        AssistantMessage,
        UserMessage,
    ]  # the provider drops aborted replies itself
    assert len(agent.messages) == 4


async def test_abort_during_tools_gives_every_call_a_result(model: Model) -> None:
    gate = asyncio.Event()  # never set: the tools hang until aborted
    tool = EchoTool(parallel_safe=True, wait=gate)
    agent, _, events = make_agent(
        model, [tool_reply(("c1", "echo", {"text": "A"}), ("c2", "echo", {"text": "B"}))], [tool]
    )
    task = agent.start("go")
    while tool.running < 2:
        await asyncio.sleep(0.001)
    agent.abort()
    end = await task
    assert end.reason == "aborted"
    assert tool.running == 0  # cancellation reached the tools
    results = [m for m in agent.messages if isinstance(m, ToolResultMessage)]
    assert [(r.tool_call_id, r.is_error) for r in results] == [("c1", True), ("c2", True)]
    assert sum(isinstance(e, ToolEnd) for e in events) == 2


async def test_abort_before_the_run_starts_ends_it_immediately(model: Model) -> None:
    agent, provider, events = make_agent(model, [text_reply("unused")])
    task = agent.start("go")
    agent.abort()
    end = await task
    assert end.reason == "aborted"
    assert types(events) == ["agent_start", "agent_end"]
    assert agent.messages == [] and provider.contexts == []


async def test_steering_enters_after_the_tool_batch(model: Model) -> None:
    gate = asyncio.Event()
    tool = EchoTool(wait=gate)
    agent, provider, events = make_agent(model, [tool_reply(("c1", "echo", {"text": "A"})), text_reply("ok")], [tool])
    task = agent.start("go")
    while tool.running < 1:
        await asyncio.sleep(0.001)
    agent.steer(user_message("also do B"))
    gate.set()
    await task
    second = provider.contexts[1].messages
    assert isinstance(second[-2], ToolResultMessage)
    assert isinstance(second[-1], UserMessage) and second[-1].content == [TextContent(text="also do B")]
    queue_events = [e for e in events if e.type == "queue_changed"]
    assert [msgspec.structs.astuple(e) for e in queue_events] == [(["also do B"], []), ([], [])]


async def test_follow_up_starts_another_turn_when_the_model_would_stop(model: Model) -> None:
    agent, provider, _ = make_agent(model, [text_reply("first", delay=0.01), text_reply("second")])
    task = agent.start("go")
    await asyncio.sleep(0.005)
    agent.follow_up(user_message("and then?"))
    await task
    assert len(provider.contexts) == 2
    assert [type(m) for m in provider.contexts[1].messages] == [UserMessage, AssistantMessage, UserMessage]


async def test_one_run_at_a_time_and_queues_need_a_run(model: Model) -> None:
    agent, _, _ = make_agent(model, [text_reply("a", delay=0.01)])
    task = agent.start("go")
    with pytest.raises(AgentBusyError):
        agent.start("again")
    await task
    with pytest.raises(AgentIdleError):
        agent.steer(user_message("late"))
