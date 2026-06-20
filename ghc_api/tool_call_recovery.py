"""
Recovery of tool calls that Copilot leaks as plain text on the direct Anthropic
streaming passthrough path.

Copilot's Claude ``/v1/messages`` endpoint normally emits structured ``tool_use``
content blocks. Intermittently its server-side tool-call parser fails mid-parse: it
partially strips the ``<function_calls>`` wrapper (leaving a bare ``call`` token on its
own line) and the ``antml:`` namespace prefixes, then gives up and streams the rest of
the tool call as ``text_delta`` with ``stop_reason: "end_turn"``. Clients that rely on
structured ``tool_use`` blocks (e.g. Claude Code) then render the tool call as text and
never execute it, stalling the agent loop.

``LeakedToolCallTransformer`` is a stateful, per-response SSE transformer used only in
the direct passthrough path. It watches text content blocks for the leaked
``<invoke name="...">...</invoke>`` markup, swallows it (stripping the ``call`` /
``<function_calls>`` wrapper residue from the surrounding prose), and re-emits proper
structured ``tool_use`` events plus a ``stop_reason: "tool_use"`` rewrite -- i.e.
exactly what the non-leaked path already produces.

Safety properties:
- Normal streams (no leak) pass through unchanged; only ``text_delta`` chunk boundaries
  may shift, which is invisible to clients that concatenate them.
- The leak is disambiguated from prose that merely *mentions* ``<invoke>`` by requiring
  the full ``<invoke name="...">`` structure (a ``name="..."`` attribute). Inline-code
  mentions where the tag is immediately preceded by a backtick are treated as prose.
"""

import json
import re
import uuid
from typing import Dict, List, Optional, Tuple

# A complete leaked invoke open tag, e.g. ``<invoke name="Bash">``.
_INVOKE_OPEN = re.compile(r'<invoke\s+name="([^"]*)"\s*>')
# A single leaked parameter, e.g. ``<parameter name="command">ls -la</parameter>``.
# DOTALL because values routinely contain newlines (multi-line commands, code, ...).
_PARAM = re.compile(r'<parameter\s+name="([^"]*)"\s*>(.*?)</parameter>', re.DOTALL)
_INVOKE_CLOSE = "</invoke>"
_FUNCTION_CALLS_OPEN = "<function_calls>"
_FUNCTION_CALLS_CLOSE = "</function_calls>"

# Emitted/yielded events are (sse_event_type, json_data_string) pairs.
Emission = Tuple[str, str]


_WHITESPACE = " \t\r\n"
_INVOKE_PREFIX = '<invoke name="'
_CALL_TOKEN = "call"


def _common_prefix_len(s: str, literal: str) -> int:
    """Length of the longest common prefix of ``s`` and ``literal``."""
    n = min(len(s), len(literal))
    i = 0
    while i < n and s[i] == literal[i]:
        i += 1
    return i


def _is_viable_danger_prefix(s: str) -> bool:
    """True if ``s`` could be the beginning of a leaked tool-call construct.

    The leaked construct (what we must never emit as prose) is::

        [whitespace] [bare ``call`` token] [<function_calls>] <invoke name="...">

    ``s`` is "viable" if it is a prefix of some such string -- i.e. it might still grow
    into a real ``<invoke name="...">`` opener once more text streams in. This is the
    core of the streaming holdback: any trailing buffer region that is viable is kept
    back rather than emitted as text.
    """
    n = len(s)
    i = 0
    # Leading whitespace.
    while i < n and s[i] in _WHITESPACE:
        i += 1
    if i == n:
        return True

    # Optional bare ``call`` wrapper token.
    if s[i] == _CALL_TOKEN[0]:
        k = _common_prefix_len(s[i:], _CALL_TOKEN)
        if i + k == n and k < len(_CALL_TOKEN):
            return True  # a partial ``call`` token that consumed the whole tail
        if k < len(_CALL_TOKEN):
            return False  # mismatch inside ``call`` (e.g. ``callback``)
        i += len(_CALL_TOKEN)
        if i == n:
            return True
        if s[i] in _WHITESPACE:
            while i < n and s[i] in _WHITESPACE:
                i += 1
            if i == n:
                return True
        elif s[i] != "<":
            return False  # ``call`` not followed by a boundary -> not the wrapper token

    rest = s[i:]
    if rest == "":
        return True

    # Optional ``<function_calls>`` wrapper open.
    fk = _common_prefix_len(rest, _FUNCTION_CALLS_OPEN)
    if fk == len(rest) and fk < len(_FUNCTION_CALLS_OPEN):
        return True  # a partial ``<function_calls>`` that consumed the whole tail
    if fk == len(_FUNCTION_CALLS_OPEN):
        rest = rest[fk:]
        while rest and rest[0] in _WHITESPACE:
            rest = rest[1:]
        if rest == "":
            return True

    # The ``<invoke name="...">`` opener.
    ik = _common_prefix_len(rest, _INVOKE_PREFIX)
    if ik == len(rest) and ik <= len(_INVOKE_PREFIX):
        return True  # a partial ``<invoke name="`` that consumed the whole tail
    if ik == len(_INVOKE_PREFIX):
        # Full ``<invoke name="`` reached; the rest is the (unquoted) name, which may be
        # any run of characters until the closing ``>``. Always a viable prefix.
        return True
    return False


class LeakedToolCallTransformer:
    """Stateful per-response transformer for the direct Anthropic SSE passthrough.

    Feed it parsed SSE events in order via :meth:`process`; it returns the list of
    events to forward to the client. Call :meth:`finalize` once the upstream stream
    ends to flush any pending state.
    """

    def __init__(self, enabled: bool = False) -> None:
        # When False (the default) the transformer never recovers tool calls: events are
        # forwarded untouched and only plain text is accumulated for the cached response
        # body. Recovery must be explicitly opted into (see state.enable_tool_call_recovery).
        self._enabled = enabled
        # Index bookkeeping so injected tool_use blocks get unique, increasing indices.
        self._max_index = -1
        # The upstream text block currently being watched.
        self._original_text_index: Optional[int] = None
        # The text block index we currently have open on our side (original or a block
        # we re-opened for trailing prose after a recovered tool call); None if closed.
        self._open_text_index: Optional[int] = None
        # Whether any tool-call recovery happened for the current text block (so we know
        # to swallow the upstream content_block_stop for it).
        self._block_had_recovery = False
        # Held-back text for the current text block (text that might be part of a
        # construct and is therefore not yet safe to emit).
        self._buffer = ""
        # Last character we actually emitted as text for the current block (used for the
        # backtick-mention disambiguation when an invoke sits at the buffer start).
        self._last_emitted = ""
        # Invoke accumulation state.
        self._in_invoke = False
        self._invoke_name: Optional[str] = None
        self._invoke_body = ""
        # Whether we recovered at least one tool call anywhere in this response.
        self._recovered_any = False
        # Accumulators for the cache response body.
        self._emitted_text_parts: List[str] = []
        self._recovered_tool_calls: List[Dict] = []

    # -- public API --------------------------------------------------------------

    def process(self, event_type: str, event: Dict, raw_data: str) -> List[Emission]:
        """Transform a single parsed SSE event into the events to forward."""
        if not self._enabled:
            return self._passthrough(event_type, event, raw_data)
        if event_type == "content_block_start":
            return self._on_content_block_start(event, raw_data)
        if event_type == "content_block_delta":
            return self._on_content_block_delta(event, raw_data)
        if event_type == "content_block_stop":
            return self._on_content_block_stop(event, raw_data)
        if event_type == "message_delta":
            return self._on_message_delta(event, raw_data)
        # message_start, message_stop, ping, error, ... pass through untouched.
        return [(event_type, raw_data)]

    def _passthrough(self, event_type: str, event: Dict, raw_data: str) -> List[Emission]:
        """Forward an event unchanged, accumulating plain text for the cache body.

        Used when recovery is disabled: no leak detection happens, so the stream is
        identical to having no transformer at all, but we still record text deltas so
        :meth:`build_response_content` reflects what the client received.
        """
        if event_type == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                self._emitted_text_parts.append(delta.get("text", ""))
        return [(event_type, raw_data)]

    def finalize(self) -> List[Emission]:
        """Flush any pending text/invoke if the stream ended without a clean stop."""
        if not self._enabled:
            return []
        events: List[Emission] = []
        if self._original_text_index is not None:
            events += self._end_text_block(None)
        return events

    @property
    def recovered_any(self) -> bool:
        return self._recovered_any

    def build_response_content(self) -> List[Dict]:
        """Build the ``content`` array for the cached response body."""
        content: List[Dict] = []
        text = "".join(self._emitted_text_parts)
        if text:
            content.append({"type": "text", "text": text})
        content.extend(self._recovered_tool_calls)
        return content

    # -- event handlers ----------------------------------------------------------

    def _on_content_block_start(self, event: Dict, raw_data: str) -> List[Emission]:
        idx = event.get("index", 0)
        self._max_index = max(self._max_index, idx)
        block = event.get("content_block", {})
        if block.get("type") == "text":
            self._reset_block_state(idx)
        return [("content_block_start", raw_data)]

    def _on_content_block_delta(self, event: Dict, raw_data: str) -> List[Emission]:
        idx = event.get("index", 0)
        delta = event.get("delta", {})
        if (
            self._original_text_index is not None
            and idx == self._original_text_index
            and delta.get("type") == "text_delta"
        ):
            return self._consume_text(delta.get("text", ""))
        # Not the watched text block (e.g. input_json_delta for a real tool_use block).
        return [("content_block_delta", raw_data)]

    def _on_content_block_stop(self, event: Dict, raw_data: str) -> List[Emission]:
        idx = event.get("index", 0)
        if self._original_text_index is not None and idx == self._original_text_index:
            return self._end_text_block(raw_data)
        return [("content_block_stop", raw_data)]

    def _on_message_delta(self, event: Dict, raw_data: str) -> List[Emission]:
        if not self._recovered_any:
            return [("message_delta", raw_data)]
        delta = event.get("delta", {})
        stop_reason = delta.get("stop_reason")
        if stop_reason in (None, "end_turn", "stop_sequence"):
            new_event = json.loads(json.dumps(event))
            new_event.setdefault("delta", {})["stop_reason"] = "tool_use"
            return [("message_delta", json.dumps(new_event))]
        return [("message_delta", raw_data)]

    # -- text engine -------------------------------------------------------------

    def _consume_text(self, text: str) -> List[Emission]:
        events: List[Emission] = []
        if self._in_invoke:
            self._invoke_body += text
            events += self._process_invoke()
            if not self._in_invoke:
                events += self._process_scan()
        else:
            self._buffer += text
            events += self._process_scan()
        return events

    def _process_scan(self) -> List[Emission]:
        """Emit safe prose from ``self._buffer``; detect/enter leaked invokes."""
        events: List[Emission] = []
        while True:
            match = _INVOKE_OPEN.search(self._buffer)
            if match and self._is_backtick_mention(match.start()):
                # Inline-code mention, not a real leak: emit through the open tag as
                # plain text and keep scanning the remainder.
                events += self._emit_text(self._buffer[: match.end()])
                self._buffer = self._buffer[match.end():]
                continue
            if match:
                prose = self._strip_trailing_residue(self._buffer[: match.start()])
                if prose:
                    events += self._emit_text(prose)
                events += self._close_text_block_for_tool()
                self._in_invoke = True
                self._invoke_name = match.group(1)
                self._invoke_body = self._buffer[match.end():]
                self._buffer = ""
                events += self._process_invoke()
                if self._in_invoke:
                    break
                continue
            # No complete invoke: emit everything except a held-back danger tail.
            keep = self._danger_holdback(self._buffer)
            if keep:
                emit_text = self._buffer[: len(self._buffer) - keep]
                self._buffer = self._buffer[len(self._buffer) - keep:]
            else:
                emit_text = self._buffer
                self._buffer = ""
            if emit_text:
                events += self._emit_text(emit_text)
            break
        return events

    def _process_invoke(self) -> List[Emission]:
        """Consume ``self._invoke_body`` until ``</invoke>``, emitting a tool_use."""
        close_idx = self._invoke_body.find(_INVOKE_CLOSE)
        if close_idx == -1:
            return []
        inner = self._invoke_body[:close_idx]
        remainder = self._invoke_body[close_idx + len(_INVOKE_CLOSE):]
        events = self._emit_tool_use(self._invoke_name, self._parse_params(inner))
        self._in_invoke = False
        self._invoke_name = None
        self._invoke_body = ""
        self._buffer = self._strip_leading_close_residue(remainder)
        return events

    def _finalize_incomplete_invoke(self) -> List[Emission]:
        """Emit a tool_use for an invoke that never received its ``</invoke>``."""
        events = self._emit_tool_use(self._invoke_name, self._parse_params(self._invoke_body))
        self._in_invoke = False
        self._invoke_name = None
        self._invoke_body = ""
        return events

    def _end_text_block(self, raw_stop: Optional[str]) -> List[Emission]:
        events: List[Emission] = []
        if self._in_invoke:
            events += self._finalize_incomplete_invoke()
        elif self._buffer:
            # Held-back tail that never became an invoke: emit it verbatim as prose.
            events += self._emit_text(self._buffer)
            self._buffer = ""

        if self._open_text_index is not None:
            if (
                not self._block_had_recovery
                and self._open_text_index == self._original_text_index
                and raw_stop is not None
            ):
                # Pure passthrough: forward the upstream stop unchanged.
                events.append(("content_block_stop", raw_stop))
            else:
                events.append((
                    "content_block_stop",
                    json.dumps({"type": "content_block_stop", "index": self._open_text_index}),
                ))
            self._open_text_index = None
        # else: we already closed the text block on our side; swallow the upstream stop.

        self._original_text_index = None
        self._block_had_recovery = False
        self._last_emitted = ""
        return events

    # -- emission helpers --------------------------------------------------------

    def _emit_text(self, text: str) -> List[Emission]:
        if not text:
            return []
        events: List[Emission] = []
        if self._open_text_index is None:
            # The previous text block was closed to emit a tool_use; re-open one so any
            # trailing prose lands in a proper text content block.
            new_idx = self._next_index()
            events.append((
                "content_block_start",
                json.dumps({
                    "type": "content_block_start",
                    "index": new_idx,
                    "content_block": {"type": "text", "text": ""},
                }),
            ))
            self._open_text_index = new_idx
        self._last_emitted = text[-1]
        self._emitted_text_parts.append(text)
        events.append((
            "content_block_delta",
            json.dumps({
                "type": "content_block_delta",
                "index": self._open_text_index,
                "delta": {"type": "text_delta", "text": text},
            }),
        ))
        return events

    def _close_text_block_for_tool(self) -> List[Emission]:
        events: List[Emission] = []
        if self._open_text_index is not None:
            events.append((
                "content_block_stop",
                json.dumps({"type": "content_block_stop", "index": self._open_text_index}),
            ))
            self._open_text_index = None
        self._block_had_recovery = True
        return events

    def _emit_tool_use(self, name: Optional[str], input_obj: Dict) -> List[Emission]:
        tool_index = self._next_index()
        tool_id = "toolu_" + uuid.uuid4().hex
        self._block_had_recovery = True
        self._recovered_any = True
        self._recovered_tool_calls.append({
            "type": "tool_use",
            "id": tool_id,
            "name": name or "",
            "input": input_obj,
        })
        return [
            ("content_block_start", json.dumps({
                "type": "content_block_start",
                "index": tool_index,
                "content_block": {"type": "tool_use", "id": tool_id, "name": name or "", "input": {}},
            })),
            ("content_block_delta", json.dumps({
                "type": "content_block_delta",
                "index": tool_index,
                "delta": {"type": "input_json_delta", "partial_json": json.dumps(input_obj)},
            })),
            ("content_block_stop", json.dumps({
                "type": "content_block_stop",
                "index": tool_index,
            })),
        ]

    # -- parsing/holdback helpers ------------------------------------------------

    @staticmethod
    def _parse_params(inner: str) -> Dict:
        # Leaked XML parameter values are raw text; preserve them as strings, matching
        # what the model emitted. Type coercion is intentionally avoided since the tool
        # schema is not available here.
        return {m.group(1): m.group(2) for m in _PARAM.finditer(inner)}

    def _is_backtick_mention(self, start: int) -> bool:
        """True when the invoke open tag is immediately preceded by a backtick."""
        if start > 0:
            return self._buffer[start - 1] == "`"
        return self._last_emitted == "`"

    @staticmethod
    def _danger_holdback(buf: str) -> int:
        """Number of trailing chars of ``buf`` to hold back as a possible construct.

        Returns the largest holdback: the length of the longest suffix that is still a
        viable prefix of a leaked tool-call construct.
        """
        for i in range(len(buf) + 1):
            if _is_viable_danger_prefix(buf[i:]):
                return len(buf) - i
        return 0

    @staticmethod
    def _strip_trailing_residue(prose: str) -> str:
        """Strip the ``call`` / ``<function_calls>`` wrapper residue that the leak
        leaves between the real prose and the ``<invoke>`` opener."""
        previous = None
        while previous != prose:
            previous = prose
            prose = re.sub(r"\s+$", "", prose)
            prose = re.sub(re.escape(_FUNCTION_CALLS_OPEN) + r"$", "", prose)
            prose = re.sub(re.escape(_FUNCTION_CALLS_CLOSE) + r"$", "", prose)
            # A bare ``call`` token (preceded by start-of-text or whitespace).
            prose = re.sub(r"(?:^|(?<=\s))call$", "", prose)
        return prose

    @staticmethod
    def _strip_leading_close_residue(remainder: str) -> str:
        """Strip a leading ``</function_calls>`` wrapper close after an ``</invoke>``."""
        match = re.match(r"\s*" + re.escape(_FUNCTION_CALLS_CLOSE), remainder)
        if match:
            return remainder[match.end():]
        return remainder

    def _next_index(self) -> int:
        self._max_index += 1
        return self._max_index

    def _reset_block_state(self, idx: int) -> None:
        self._original_text_index = idx
        self._open_text_index = idx
        self._block_had_recovery = False
        self._buffer = ""
        self._last_emitted = ""
        self._in_invoke = False
        self._invoke_name = None
        self._invoke_body = ""
