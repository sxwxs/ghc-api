"""Explicit GLM 5.2 NVFP4 Chat Completions compatibility.

This adapter is enabled only by ``compatibility: glm_5_2_nvfp4``.  The
upstream accepts OpenAI-shaped requests but requires message content to be a
plain string and emits thinking/tool syntax inside ``message.content``.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .kimi_k3 import KimiTextStream, fold_chat_messages, split_thinking


GLM_5_2_NVFP4 = "glm_5_2_nvfp4"

_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"
_ARG_KEY_OPEN = "<arg_key>"
_ARG_KEY_CLOSE = "</arg_key>"
_ARG_VALUE_OPEN = "<arg_value>"
_ARG_VALUE_CLOSE = "</arg_value>"
_OBSERVATION = "<|observation|>"


def fold_glm_chat_messages(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Fold text-only OpenAI messages into the string form this endpoint accepts."""

    return fold_chat_messages(payload)


def declared_tool_schemas(tools: Any) -> Dict[str, Dict[str, Any]]:
    """Return validated-enough function schemas keyed by declared tool name."""

    if not isinstance(tools, list):
        return {}
    schemas: Dict[str, Dict[str, Any]] = {}
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        parameters = function.get("parameters")
        schemas[name] = parameters if isinstance(parameters, dict) else {}
    return schemas


def _stable_call_id(name: str, arguments: Dict[str, Any], ordinal: int) -> str:
    canonical = json.dumps(
        {"name": name, "arguments": arguments, "ordinal": ordinal},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "call_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def _schema_type(schema: Any) -> Optional[str]:
    if not isinstance(schema, dict):
        return None
    value = schema.get("type")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        non_null = [item for item in value if isinstance(item, str) and item != "null"]
        if len(non_null) == 1:
            return non_null[0]
    return None


def _decode_argument(raw: str, schema: Any) -> Tuple[bool, Any]:
    expected = _schema_type(schema)
    if expected == "string":
        if raw.startswith('"'):
            try:
                value = json.loads(raw)
            except ValueError:
                return False, None
            return (True, value) if isinstance(value, str) else (False, None)
        return True, raw

    if expected in {"number", "integer", "boolean", "null", "object", "array"}:
        try:
            value = json.loads(raw)
        except ValueError:
            return False, None
        if expected == "number":
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
            valid = valid and not (isinstance(value, float) and not math.isfinite(value))
        elif expected == "integer":
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif expected == "boolean":
            valid = isinstance(value, bool)
        elif expected == "null":
            valid = value is None
        elif expected == "object":
            valid = isinstance(value, dict)
        else:
            valid = isinstance(value, list)
        return valid, value

    # Unknown or omitted schemas are common on lightweight OpenAI-compatible
    # servers. Preserve plain text, but retain unambiguous JSON values.
    try:
        value = json.loads(raw)
    except ValueError:
        return True, raw
    if isinstance(value, float) and not math.isfinite(value):
        return False, None
    return True, value


def _take_between(text: str, position: int, opening: str, closing: str) -> Tuple[Optional[str], int]:
    if not text.startswith(opening, position):
        return None, position
    start = position + len(opening)
    end = text.find(closing, start)
    if end < 0:
        return None, position
    return text[start:end], end + len(closing)


def parse_native_tool_calls(
    text: str,
    tool_schemas: Mapping[str, Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    """Parse the GLM native tool grammar, failing closed on ambiguous text.

    The endpoint may continue after a completed tool call with a hallucinated
    observation.  Content from the first observation marker onward is ignored,
    but only after the entire preceding candidate parses as declared tools.
    """

    if not tool_schemas or not isinstance(text, str):
        return None
    candidate = text.strip()
    observation_at = candidate.find(_OBSERVATION)
    if observation_at >= 0:
        candidate = candidate[:observation_at].rstrip()
    if not candidate.startswith(_TOOL_OPEN):
        return None

    parsed: List[Tuple[str, Dict[str, Any]]] = []
    position = 0
    while position < len(candidate):
        while position < len(candidate) and candidate[position].isspace():
            position += 1
        if not candidate.startswith(_TOOL_OPEN, position):
            return None
        position += len(_TOOL_OPEN)

        key_at = candidate.find(_ARG_KEY_OPEN, position)
        close_at = candidate.find(_TOOL_CLOSE, position)
        if close_at < 0:
            return None
        name_end = close_at if key_at < 0 or close_at < key_at else key_at
        name = candidate[position:name_end].strip()
        parameters = tool_schemas.get(name)
        if parameters is None:
            return None
        properties = parameters.get("properties") if isinstance(parameters, dict) else None
        properties = properties if isinstance(properties, dict) else {}
        required = parameters.get("required") if isinstance(parameters, dict) else None
        required_names = {item for item in required if isinstance(item, str)} if isinstance(required, list) else set()

        arguments: Dict[str, Any] = {}
        position = name_end
        while candidate.startswith(_ARG_KEY_OPEN, position):
            key, position = _take_between(
                candidate, position, _ARG_KEY_OPEN, _ARG_KEY_CLOSE
            )
            if key is None or not key or key in arguments:
                return None
            raw_value, position = _take_between(
                candidate, position, _ARG_VALUE_OPEN, _ARG_VALUE_CLOSE
            )
            if raw_value is None:
                return None
            if properties and key not in properties:
                return None
            valid, value = _decode_argument(raw_value, properties.get(key))
            if not valid:
                return None
            arguments[key] = value

        if not candidate.startswith(_TOOL_CLOSE, position):
            return None
        position += len(_TOOL_CLOSE)
        if not required_names.issubset(arguments):
            return None
        parsed.append((name, arguments))

    if not parsed:
        return None
    return [
        {
            "index": ordinal,
            "id": _stable_call_id(name, arguments, ordinal),
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(
                    arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        }
        for ordinal, (name, arguments) in enumerate(parsed)
    ]


def convert_non_stream_response(
    result: Dict[str, Any], tool_schemas: Mapping[str, Dict[str, Any]]
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
            calls = parse_native_tool_calls(final, tool_schemas)
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


class GlmTextStream(KimiTextStream):
    """Reuse the incremental thinking splitter with the GLM tool prefix/parser."""

    TOOL_PREFIX = _TOOL_OPEN

    def __init__(self, tool_schemas: Mapping[str, Dict[str, Any]]) -> None:
        super().__init__(tuple(tool_schemas))
        self.tool_schemas = dict(tool_schemas)

    def finish(self) -> Tuple[List[Tuple[str, str]], Optional[List[Dict[str, Any]]]]:
        output: List[Tuple[str, str]] = []
        if self.state == "prefix":
            pending = self.buffer
            self.buffer = ""
            self.state = "final"
            output.extend(self._feed_final(pending))
        elif self.state == "thinking":
            if self.buffer:
                output.append(("reasoning_content", self.buffer))
            self.buffer = ""
            return output, None

        if self.final_mode == "candidate" and self.buffer:
            candidate = self.buffer
            self.buffer = ""
            calls = parse_native_tool_calls(candidate, self.tool_schemas)
            if calls is not None:
                return output, calls
            output.append(("content", candidate))
        return output, None
