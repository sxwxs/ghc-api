"""Terminal rendering of session events, and JSONL framing."""

import msgspec
import pytest

from ghc_api.pyagent.agent.events import ToolEnd, ToolStart
from ghc_api.pyagent.agent.tool import text_result
from ghc_api.pyagent.ai.events import BlockDelta, BlockStart
from ghc_api.pyagent.ai.types import user_message
from ghc_api.pyagent.cli.printers import JsonPrinter, TextPrinter
from ghc_api.pyagent.coding.session_events import EventEnvelope, RunEnd, SessionEvent
from ghc_api.pyagent.protocol.jsonrpc import encode_line


def envelope(event: SessionEvent, seq: int = 1) -> EventEnvelope:
    return EventEnvelope(session_id="s", seq=seq, run_id="r", event=event)


def test_encode_line_is_one_line_with_unicode_separators_escaped() -> None:
    line = encode_line(user_message("a\u2028b\u2029c\nd"))
    assert line.endswith(b"\n") and line.count(b"\n") == 1
    assert b"\\u2028" in line and b"\\u2029" in line


def test_text_printer_streams_text_and_reports_tools(capsys: pytest.CaptureFixture[str]) -> None:
    printer = TextPrinter()
    for event in [
        BlockStart(message_id="m", index=0, kind="thinking"),
        BlockDelta(message_id="m", index=0, delta="hidden"),
        BlockStart(message_id="m", index=1, kind="text"),
        BlockDelta(message_id="m", index=1, delta="Hello"),
        ToolStart(tool_call_id="c", name="bash", kind="execute", title="ls"),
        ToolEnd(tool_call_id="c", result=text_result("out\nCommand exited with code 2", is_error=True)),
        RunEnd(reason="aborted"),
    ]:
        printer(envelope(event))
    out = capsys.readouterr()
    assert out.out == "Hello\n"
    assert out.err == "[bash] ls\n  ! Command exited with code 2\n(aborted)\n"


def test_json_printer_writes_envelopes(capsysbinary: pytest.CaptureFixture[bytes]) -> None:
    JsonPrinter()(envelope(RunEnd(reason="completed"), seq=7))
    assert msgspec.json.decode(capsysbinary.readouterr().out) == {
        "sessionId": "s",
        "seq": 7,
        "event": {"type": "run_end", "reason": "completed"},
        "runId": "r",
    }
