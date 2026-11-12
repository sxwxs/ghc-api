"""Renders session events for a terminal: streamed text, or one JSON line per event."""

import sys

from ..agent.events import ToolEnd, ToolStart
from ..ai.events import BlockDelta, BlockStart, MessageEnd
from ..ai.types import AssistantMessage, TextContent
from ..coding.session_events import EventEnvelope, RunEnd
from ..protocol.jsonrpc import encode_line


class TextPrinter:
    """Prints reply text to stdout as it streams, and tool activity and errors to stderr."""

    def __init__(self) -> None:
        self.text_blocks: set[tuple[str, int]] = set()
        self.open_line = False

    def __call__(self, envelope: EventEnvelope) -> None:
        event = envelope.event
        match event:
            case BlockStart(kind="text"):
                self.text_blocks.add((event.message_id, event.index))
            case BlockDelta() if (event.message_id, event.index) in self.text_blocks:
                sys.stdout.write(event.delta)
                sys.stdout.flush()
                self.open_line = not event.delta.endswith("\n")
            case ToolStart():
                self._end_line()
                print(f"[{event.name}] {event.title}", file=sys.stderr)
            case ToolEnd() if event.result.is_error:
                text = "".join(b.text for b in event.result.content if isinstance(b, TextContent))
                lines = text.strip().splitlines()
                print(f"  ! {lines[-1] if lines else 'failed'}", file=sys.stderr)
            case MessageEnd(message=AssistantMessage(stop_reason="error") as message):
                self._end_line()
                print(f"error: {message.error_message}", file=sys.stderr)
            case RunEnd():
                self._end_line()
                if event.reason == "aborted":
                    print("(aborted)", file=sys.stderr)
            case _:
                pass

    def _end_line(self) -> None:
        if self.open_line:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.open_line = False


class JsonPrinter:
    """Writes every event envelope to stdout as one JSON line: the params of the protocol's session/event."""

    def __call__(self, envelope: EventEnvelope) -> None:
        sys.stdout.buffer.write(encode_line(envelope))
        sys.stdout.buffer.flush()
