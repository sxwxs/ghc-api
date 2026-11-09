"""Message stream events: the incremental form of a message that providers emit and UIs render."""

from typing import Literal

from .types import Message, Wire


class Event(Wire, tag_field="type"):
    """Base of all events, here and in the agent layer."""

    @property
    def type(self) -> str:
        """The wire tag, e.g. "block_delta"."""
        return str(self.__struct_config__.tag)


class MessageStart(Event, tag="message_start"):
    message: Message


class MessageEnd(Event, tag="message_end"):
    """The authoritative final message; replaces whatever deltas built."""

    message: Message


BlockKind = Literal["text", "thinking", "tool_call"]


class BlockStart(Event, tag="block_start"):
    message_id: str
    index: int
    kind: BlockKind
    tool_call_id: str | None = None
    tool_name: str | None = None


class BlockDelta(Event, tag="block_delta"):
    """Appends text, thinking text, or raw tool-argument JSON to block `index`."""

    message_id: str
    index: int
    delta: str


class BlockEnd(Event, tag="block_end"):
    message_id: str
    index: int


BlockEvent = BlockStart | BlockDelta | BlockEnd
MessageEvent = MessageStart | BlockStart | BlockDelta | BlockEnd | MessageEnd
