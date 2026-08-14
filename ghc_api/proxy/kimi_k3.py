"""Explicit Kimi K3/Papyrus Chat Completions compatibility.

This module is deliberately independent of model names.  It is used only when a
model API opts into ``compatibility: kimi_k3_papyrus``.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


KIMI_K3_PAPYRUS = "kimi_k3_papyrus"


class ProxyPayloadError(ValueError):
    """A client payload cannot be represented by the selected compatibility adapter."""


def _content_text(value: Any, field: str, *, allow_none: bool = False) -> str:
    if value is None and allow_none:
        return ""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise ProxyPayloadError(f"'{field}' must be a string or an array of text blocks")

    parts: List[str] = []
    for index, block in enumerate(value):
        block_field = f"{field}[{index}]"
        if not isinstance(block, dict):
            raise ProxyPayloadError(f"'{block_field}' must be a text content block")
        if block.get("type") != "text" or not isinstance(block.get("text"), str):
            block_type = block.get("type", "unknown")
            raise ProxyPayloadError(
                f"'{block_field}' has unsupported content type '{block_type}'; "
                "this model is configured as text-only"
            )
        parts.append(block["text"])
    return "".join(parts)


def _bounded(label: str, text: str, suffix: str = "") -> str:
    attrs = f" length={len(text)}"
    if suffix:
        attrs += f" {suffix}"
    return f"[{label}{attrs}]\n{text}\n[/{label}]"


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def fold_chat_messages(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize text blocks and fold a complete conversation into one user turn."""

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ProxyPayloadError("'messages' must be a non-empty array")

    instructions: List[str] = []
    transcript: List[str] = []
    tool_names_by_id: Dict[str, str] = {}

    for index, raw_message in enumerate(messages):
        field = f"messages[{index}]"
        if not isinstance(raw_message, dict):
            raise ProxyPayloadError(f"'{field}' must be an object")
        role = raw_message.get("role")
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            raise ProxyPayloadError(f"'{field}.role' is not supported by this adapter")

        content = _content_text(
            raw_message.get("content"), f"{field}.content", allow_none=(role == "assistant")
        )

        if role in {"system", "developer"}:
            instructions.append(_bounded(role.upper(), content))
            continue

        if role == "user":
            transcript.append(_bounded("USER", content))
            continue

        if role == "assistant":
            assistant_parts: List[str] = []
            reasoning = raw_message.get("reasoning_content")
            if reasoning is not None:
                reasoning_text = _content_text(
                    reasoning, f"{field}.reasoning_content", allow_none=True
                )
                if reasoning_text:
                    assistant_parts.append(_bounded("REASONING", reasoning_text))
            if content:
                assistant_parts.append(_bounded("TEXT", content))

            tool_calls = raw_message.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                raise ProxyPayloadError(f"'{field}.tool_calls' must be an array")
            for call_index, call in enumerate(tool_calls):
                call_field = f"{field}.tool_calls[{call_index}]"
                if not isinstance(call, dict) or call.get("type", "function") != "function":
                    raise ProxyPayloadError(f"'{call_field}' must be a function tool call")
                function = call.get("function")
                call_id = call.get("id")
                if (
                    not isinstance(function, dict)
                    or not isinstance(function.get("name"), str)
                    or not function["name"]
                    or not isinstance(call_id, str)
                    or not call_id
                ):
                    raise ProxyPayloadError(f"'{call_field}' has an invalid id or function name")
                arguments = _json_text(function.get("arguments", "{}"))
                tool_names_by_id[call_id] = function["name"]
                assistant_parts.append(
                    _bounded(
                        "TOOL CALL",
                        arguments,
                        f"id={json.dumps(call_id)} name={json.dumps(function['name'])}",
                    )
                )
            transcript.append(_bounded("ASSISTANT", "\n\n".join(assistant_parts)))
            continue

        call_id = raw_message.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id:
            raise ProxyPayloadError(f"'{field}.tool_call_id' is required")
        name = raw_message.get("name") or tool_names_by_id.get(call_id) or "unknown"
        if not isinstance(name, str):
            raise ProxyPayloadError(f"'{field}.name' must be a string")
        transcript.append(
            _bounded(
                "TOOL RESULT",
                content,
                f"id={json.dumps(call_id)} name={json.dumps(name)}",
            )
        )

    instruction_text = "\n\n".join(instructions) if instructions else "(none)"
    transcript_text = "\n\n".join(transcript) if transcript else "(empty)"
    folded = (
        "[SYSTEM INSTRUCTIONS]\n"
        f"{instruction_text}\n"
        "[/SYSTEM INSTRUCTIONS]\n\n"
        "[CONVERSATION TRANSCRIPT]\n"
        f"{transcript_text}\n"
        "[/CONVERSATION TRANSCRIPT]\n\n"
        "[INSTRUCTION]\n"
        "Continue from the final transcript entry. Follow the system and developer "
        "instructions and use the declared tools when needed.\n"
        "[/INSTRUCTION]"
    )

    result = dict(payload)
    result["messages"] = [{"role": "user", "content": folded}]
    return result


def declared_tool_names(tools: Any) -> Tuple[str, ...]:
    if not isinstance(tools, list):
        return ()
    names: List[str] = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.append(function["name"])
    return tuple(names)


def split_thinking(text: str) -> Tuple[Optional[str], str]:
    """Split the exact Kimi thinking envelope without exposing its tags."""

    stripped = text.lstrip()
    if not stripped.startswith("<think>"):
        return None, text
    body = stripped[len("<think>") :]
    end = body.find("</think>")
    if end < 0:
        # A truncated thinking envelope is reasoning, never a client-visible tag.
        return body, ""
    return body[:end], body[end + len("</think>") :]


def _quoted_attribute(text: str, position: int, name: str) -> Tuple[Optional[str], int]:
    prefix = name + "="
    if not text.startswith(prefix, position):
        return None, position
    position += len(prefix)
    try:
        value, consumed = json.JSONDecoder().raw_decode(text[position:])
    except (ValueError, TypeError):
        return None, position
    if not isinstance(value, str):
        return None, position
    return value, position + consumed


def _spaces(text: str, position: int, *, required: bool) -> int:
    start = position
    while position < len(text) and text[position] in " \t":
        position += 1
    return -1 if required and position == start else position


def _parse_header(line: str, kind: str) -> Optional[Tuple[str, str]]:
    """Parse one exact Papyrus grammar header without searching ordinary prose."""

    text = line.strip()
    first, position = _quoted_attribute(text, 0, kind)
    if first is None:
        return None
    position = _spaces(text, position, required=True)
    if position < 0:
        return None
    second_name = "index" if kind == "tool" else "type"
    second, position = _quoted_attribute(text, position, second_name)
    if second is None or text[position:] != "<|sep|>":
        return None
    return first, second


def _typed_value(type_name: str, raw: str) -> Tuple[bool, Any]:
    if type_name == "string":
        # Papyrus string values are separator-delimited raw text.  Also accept a
        # canonical JSON string so quotes and escapes can be represented exactly.
        if raw.startswith('"'):
            try:
                value = json.loads(raw)
            except ValueError:
                return False, None
            if isinstance(value, str):
                return True, value
        return True, raw

    if type_name not in {"number", "integer", "boolean", "null", "object", "array"}:
        return False, None
    try:
        value = json.loads(raw)
    except ValueError:
        return False, None
    if type_name == "number":
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        valid = valid and not (isinstance(value, float) and not math.isfinite(value))
    elif type_name == "integer":
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif type_name == "boolean":
        valid = isinstance(value, bool)
    elif type_name == "null":
        valid = value is None
    elif type_name == "object":
        valid = isinstance(value, dict)
    else:
        valid = isinstance(value, list)
    return valid, value


def _stable_call_id(name: str, arguments: Dict[str, Any], ordinal: int) -> str:
    canonical = json.dumps(
        {"name": name, "arguments": arguments, "ordinal": ordinal},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    return "call_" + digest


def parse_native_tool_calls(text: str, allowed_names: Iterable[str]) -> Optional[List[Dict[str, Any]]]:
    """Parse the strict line-oriented Papyrus tool grammar.

    Parsing is fail-closed: the whole candidate must consist of tool and key
    productions, every type must decode exactly, and every tool must have been
    declared by the client.  ``None`` means the text must remain ordinary text.
    """

    allowed = set(allowed_names)
    if not allowed or not text.strip():
        return None
    lines = text.strip().splitlines()
    calls: List[Tuple[str, str, Dict[str, Any]]] = []
    current: Optional[Tuple[str, str, Dict[str, Any]]] = None

    for line in lines:
        if not line.strip():
            return None
        tool_header = _parse_header(line, "tool")
        if tool_header is not None:
            name, native_index = tool_header
            if name not in allowed or not native_index.isdigit():
                return None
            if current is not None:
                calls.append(current)
            current = (name, native_index, {})
            continue

        if current is None:
            return None
        marker = "<|sep|>"
        marker_at = line.find(marker)
        if marker_at < 0 or line.find(marker, marker_at + len(marker)) >= 0:
            return None
        key_header = _parse_header(line[: marker_at + len(marker)], "key")
        if key_header is None:
            return None
        key, type_name = key_header
        raw_value = line[marker_at + len(marker) :]
        valid, value = _typed_value(type_name, raw_value)
        if not valid or not key or key in current[2]:
            return None
        current[2][key] = value

    if current is not None:
        calls.append(current)
    if not calls or len({native_index for _, native_index, _ in calls}) != len(calls):
        return None

    result: List[Dict[str, Any]] = []
    for ordinal, (name, _native_index, arguments) in enumerate(calls):
        result.append({
            "index": ordinal,
            "id": _stable_call_id(name, arguments, ordinal),
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(
                    arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
            },
        })
    return result


def convert_non_stream_response(
    result: Dict[str, Any], allowed_names: Sequence[str]
) -> Dict[str, Any]:
    converted = dict(result)
    choices = result.get("choices")
    if not isinstance(choices, list):
        return converted
    converted_choices: List[Any] = []
    for raw_choice in choices:
        if not isinstance(raw_choice, dict) or not isinstance(raw_choice.get("message"), dict):
            converted_choices.append(raw_choice)
            continue
        choice = dict(raw_choice)
        message = dict(raw_choice["message"])
        content = message.get("content")
        if isinstance(content, str):
            reasoning, final = split_thinking(content)
            calls = parse_native_tool_calls(final, allowed_names)
            message["content"] = "" if calls is not None else final
            if reasoning is not None:
                message["reasoning_content"] = reasoning
            if calls is not None:
                message["tool_calls"] = calls
        has_calls = isinstance(message.get("tool_calls"), list) and bool(message["tool_calls"])
        choice["message"] = message
        choice["finish_reason"] = "tool_calls" if has_calls else "stop"
        converted_choices.append(choice)
    converted["choices"] = converted_choices
    return converted


class KimiTextStream:
    """Incremental tag splitter that buffers only a possible native tool candidate."""

    OPEN = "<think>"
    CLOSE = "</think>"
    TOOL_PREFIX = 'tool="'

    def __init__(self, allowed_names: Sequence[str]) -> None:
        self.allowed_names = tuple(allowed_names)
        self.state = "prefix"
        self.buffer = ""
        self.final_mode = "candidate"

    def _feed_final(self, text: str) -> List[Tuple[str, str]]:
        if not text:
            return []
        if self.final_mode == "normal":
            return [("content", text)]
        self.buffer += text
        stripped = self.buffer.lstrip()
        if not stripped or self.TOOL_PREFIX.startswith(stripped):
            return []
        if stripped.startswith(self.TOOL_PREFIX):
            return []
        self.final_mode = "normal"
        output = self.buffer
        self.buffer = ""
        return [("content", output)]

    def _feed_thinking(self, text: str) -> List[Tuple[str, str]]:
        self.buffer += text
        end = self.buffer.find(self.CLOSE)
        if end >= 0:
            reasoning = self.buffer[:end]
            remainder = self.buffer[end + len(self.CLOSE) :]
            self.buffer = ""
            self.state = "final"
            output = [("reasoning_content", reasoning)] if reasoning else []
            output.extend(self._feed_final(remainder))
            return output
        retained = min(len(self.buffer), len(self.CLOSE) - 1)
        ready = self.buffer[:-retained] if retained else self.buffer
        self.buffer = self.buffer[-retained:] if retained else ""
        return [("reasoning_content", ready)] if ready else []

    def feed(self, text: str) -> List[Tuple[str, str]]:
        if not isinstance(text, str) or not text:
            return []
        if self.state == "thinking":
            return self._feed_thinking(text)
        if self.state == "final":
            return self._feed_final(text)

        self.buffer += text
        stripped = self.buffer.lstrip()
        if not stripped or self.OPEN.startswith(stripped):
            return []
        if stripped.startswith(self.OPEN):
            remainder = stripped[len(self.OPEN) :]
            self.buffer = ""
            self.state = "thinking"
            return self._feed_thinking(remainder)
        self.state = "final"
        pending = self.buffer
        self.buffer = ""
        return self._feed_final(pending)

    def finish(self) -> Tuple[List[Tuple[str, str]], Optional[List[Dict[str, Any]]]]:
        output: List[Tuple[str, str]] = []
        if self.state == "prefix":
            # Incomplete/non-tag prefix is ordinary final text.
            pending = self.buffer
            self.buffer = ""
            self.state = "final"
            output.extend(self._feed_final(pending))
        elif self.state == "thinking":
            # Do not expose a truncated closing tag as ordinary content.
            if self.buffer:
                output.append(("reasoning_content", self.buffer))
            self.buffer = ""
            return output, None

        if self.final_mode == "candidate" and self.buffer:
            candidate = self.buffer
            self.buffer = ""
            calls = parse_native_tool_calls(candidate, self.allowed_names)
            if calls is not None:
                return output, calls
            output.append(("content", candidate))
        return output, None
