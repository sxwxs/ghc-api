"""The agent loop: stream a reply, run its tool calls, repeat until the model stops; aborts keep transcripts valid."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import msgspec

from ..ai.events import MessageEnd, MessageStart
from ..ai.models import Model
from ..ai.provider import ProviderFn, StreamOptions
from ..ai.stream import empty_reply, stream_assistant
from ..ai.types import AssistantMessage, Context, Message, ToolCall, ToolResultMessage, ToolSpec
from .events import AgentEnd, AgentEvent, AgentStart, ToolEnd, ToolStart, ToolUpdate, TurnEnd, TurnStart
from .tool import Tool, ToolContext, ToolKind, ToolResult, text_result

# Returns a reason to block the call, or None to allow it. Permission checks hook in here.
BeforeToolCall = Callable[[ToolCall, Tool[Any], Any], Awaitable[str | None]]


@dataclass
class ToolBatch:
    """The tool calls of one reply and their results so far."""

    calls: list[ToolCall]
    started: set[str] = field(default_factory=set[str])
    results: dict[str, ToolResult] = field(default_factory=dict[str, ToolResult])


@dataclass(kw_only=True)
class RunState:
    """One run's configuration and progress. The Agent builds it; the loop owns it while running."""

    model: Model
    system_prompt: str
    tools: dict[str, Tool[Any]]
    messages: list[Message]  # the agent's transcript, appended in place
    options: StreamOptions
    cwd: str
    emit: Callable[[AgentEvent], None]
    take_steering: Callable[[], list[Message]]
    take_follow_up: Callable[[], list[Message]]
    provider: ProviderFn | None = None
    before_tool_call: BeforeToolCall | None = None
    # A task cancelled before its first step never enters run(), so Agent.abort cancels only a started run
    # and otherwise just sets abort_requested, which run() checks on entry.
    started: bool = False
    # Set by Agent.abort; any other cancellation propagates.
    abort_requested: bool = False
    # Work an abort must finish: the reply being streamed and the tool batch being run.
    reply: AssistantMessage | None = None
    batch: ToolBatch | None = None


async def run(state: RunState, prompts: list[Message]) -> AgentEnd:
    """Run until the model stops, fails, or the run is aborted. Emits agent_start first and agent_end last.

    Invariant: every tool call in a completed reply gets exactly one tool result, also on abort.
    """
    state.started = True
    state.emit(AgentStart())
    if state.abort_requested:
        end = AgentEnd(reason="aborted")
        state.emit(end)
        return end
    end = AgentEnd(reason="completed")
    specs = [tool.spec() for tool in state.tools.values()]
    pending = prompts
    turn = 0
    try:
        while True:
            for message in pending:
                _append(state, message)
            turn += 1
            state.emit(TurnStart(turn=turn))
            reply = await _stream_reply(state, specs)
            if reply.stop_reason in ("error", "aborted"):
                state.emit(TurnEnd(turn=turn))
                end = AgentEnd(reason="error", error_message=reply.error_message)
                break
            calls = reply.tool_calls()
            if calls:
                await _run_tools(state, calls)
            state.emit(TurnEnd(turn=turn))
            pending = state.take_steering()
            if calls or pending:
                continue
            pending = state.take_follow_up()
            if not pending:
                break
    except asyncio.CancelledError:
        if not state.abort_requested:
            raise
        task = asyncio.current_task()
        assert task is not None  # run() always executes inside the Agent's task
        task.uncancel()
        _finish_aborted(state)
        # Every await sits inside a turn, so an abort always interrupts an open turn.
        state.emit(TurnEnd(turn=turn))
        end = AgentEnd(reason="aborted")
    state.emit(end)
    return end


def _append(state: RunState, message: Message) -> None:
    state.emit(MessageStart(message=message))
    state.messages.append(message)
    state.emit(MessageEnd(message=message))


async def _stream_reply(state: RunState, specs: list[ToolSpec]) -> AssistantMessage:
    reply = empty_reply(state.model)
    state.reply = reply
    context = Context(system_prompt=state.system_prompt, messages=state.messages, tools=specs)
    await stream_assistant(state.model, context, state.options, reply, state.emit, state.provider)
    state.reply = None
    state.messages.append(reply)
    return reply


async def _run_tools(state: RunState, calls: list[ToolCall]) -> None:
    """Run one reply's tool calls: concurrently when all are parallel-safe, else one by one."""
    batch = ToolBatch(calls)
    state.batch = batch
    tools = [state.tools.get(call.name) for call in calls]
    if len(calls) > 1 and all(tool is not None and tool.parallel_safe for tool in tools):
        ready = [prepared for call in calls if (prepared := await _prepare(state, batch, call)) is not None]
        async with asyncio.TaskGroup() as group:
            for call, tool, params in ready:
                group.create_task(_execute(state, batch, call, tool, params))
    else:
        for call in calls:
            prepared = await _prepare(state, batch, call)
            if prepared is not None:
                await _execute(state, batch, *prepared)
    _commit(state, batch)


async def _prepare(state: RunState, batch: ToolBatch, call: ToolCall) -> tuple[ToolCall, Tool[Any], Any] | None:
    """Validate a call and ask the hook. Returns what to execute, or None after recording a failed result."""
    tool = state.tools.get(call.name)
    if tool is None:
        _start(state, batch, call, "other", call.name)
        _finish(state, batch, call.id, text_result(f"Tool '{call.name}' not found", is_error=True), None)
        return None
    try:
        params = msgspec.convert(tool.prepare_arguments(call.arguments), tool.params)
    except msgspec.ValidationError as error:
        _start(state, batch, call, tool.kind, call.name)
        _finish(state, batch, call.id, text_result(f"Invalid arguments for {call.name}: {error}", is_error=True), None)
        return None
    _start(state, batch, call, tool.kind, tool.title(params))
    if state.before_tool_call is not None:
        reason = await state.before_tool_call(call, tool, params)
        if reason is not None:
            _finish(state, batch, call.id, text_result(reason, is_error=True), None)
            return None
    return call, tool, params


async def _execute(state: RunState, batch: ToolBatch, call: ToolCall, tool: Tool[Any], params: Any) -> None:
    def on_update(partial: ToolResult) -> None:
        state.emit(ToolUpdate(tool_call_id=call.id, partial=partial))

    started = time.perf_counter()
    try:
        result = await tool.execute(params, ToolContext(cwd=state.cwd, on_update=on_update))
    except Exception as error:
        result = text_result(str(error) or type(error).__name__, is_error=True)
    _finish(state, batch, call.id, result, round((time.perf_counter() - started) * 1000))


def _start(state: RunState, batch: ToolBatch, call: ToolCall, kind: ToolKind, title: str) -> None:
    batch.started.add(call.id)
    state.emit(ToolStart(tool_call_id=call.id, name=call.name, kind=kind, title=title))


def _finish(state: RunState, batch: ToolBatch, call_id: str, result: ToolResult, duration_ms: int | None) -> None:
    batch.results[call_id] = result
    state.emit(ToolEnd(tool_call_id=call_id, result=result, duration_ms=duration_ms))


def _commit(state: RunState, batch: ToolBatch) -> None:
    """Append one tool result message per call, in the order the model made the calls."""
    state.batch = None
    for call in batch.calls:
        result = batch.results[call.id]
        _append(
            state,
            ToolResultMessage(
                tool_call_id=call.id,
                tool_name=call.name,
                content=result.content,
                details=result.details,
                is_error=result.is_error,
            ),
        )


def _finish_aborted(state: RunState) -> None:
    if state.reply is not None:
        # stream_assistant already marked it aborted and emitted message_end.
        state.messages.append(state.reply)
        state.reply = None
    batch = state.batch
    if batch is None:
        return
    for call in batch.calls:
        if call.id in batch.results:
            continue
        if call.id not in batch.started:
            tool = state.tools.get(call.name)
            _start(state, batch, call, tool.kind if tool else "other", call.name)
        _finish(state, batch, call.id, text_result("Aborted", is_error=True), None)
    _commit(state, batch)
