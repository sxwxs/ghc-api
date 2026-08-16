"""Translate OpenAI Responses SSE into Anthropic Messages SSE.

The pure :class:`ResponsesAnthropicEventTranslator` is independently testable.
The Flask-facing :class:`AnthropicResponsesStreamHandler` only connects that
state machine to the shared SSE/cache transport.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from ..anthropic_responses import (
    MODE_COMPATIBILITY,
    AnthropicToResponsesResult,
    ConversionReport,
    IdentifierCodec,
    PRESERVATION_APPROXIMATION,
    PRESERVATION_SEMANTIC,
    PRESERVATION_SIDECAR,
    ResponsesToAnthropicResult,
    anthropic_error_from_responses,
    convert_responses_to_anthropic,
    parse_strict_json_bytes,
)
from ..compat_profiles import audit_responses_event
from ..reasoning_carrier import (
    build_reasoning_carrier,
    redact_reasoning_carriers_for_cache,
)
from ..compat_redaction import (
    redact_responses_event_for_cache,
    redacted_value,
)
from .base import SSEStreamHandler


def _json(value: Dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class StopSequenceScanner:
    """Streaming stop matcher that never leaks a cross-chunk prefix."""

    def __init__(self, stop_sequences: Optional[Sequence[str]] = None) -> None:
        self.stop_sequences = [item for item in (stop_sequences or []) if isinstance(item, str) and item]
        self.max_length = max((len(item) for item in self.stop_sequences), default=0)
        self.pending = ""
        self.matched: Optional[str] = None

    def push(self, text: str) -> str:
        if self.matched is not None or not text:
            return ""
        if not self.stop_sequences:
            return text
        self.pending += text
        best_index: Optional[int] = None
        best_stop: Optional[str] = None
        for stop in self.stop_sequences:
            index = self.pending.find(stop)
            if index >= 0 and (
                best_index is None
                or index < best_index
                or (index == best_index and len(stop) > len(best_stop or ""))
            ):
                best_index = index
                best_stop = stop
        if best_index is not None:
            safe = self.pending[:best_index]
            self.pending = ""
            self.matched = best_stop
            return safe
        keep = max(0, self.max_length - 1)
        if len(self.pending) <= keep:
            return ""
        safe_length = len(self.pending) - keep
        safe = self.pending[:safe_length]
        self.pending = self.pending[safe_length:]
        return safe

    def finish(self) -> str:
        if self.matched is not None:
            self.pending = ""
            return ""
        result = self.pending
        self.pending = ""
        return result


@dataclass
class _OutputState:
    output_index: int
    item_type: str = ""
    item: Dict[str, Any] = field(default_factory=dict)
    done: bool = False
    started: bool = False
    closed: bool = False
    block_index: Optional[int] = None
    call_id: str = ""
    name: str = ""
    text: str = ""
    text_parts: Dict[int, str] = field(default_factory=dict)
    text_forwarded: int = 0
    arguments: str = ""
    arguments_forwarded: int = 0
    arguments_complete: bool = False
    web_search_stage: int = 0
    annotation_indices: set = field(default_factory=set)
    scanner: Optional[StopSequenceScanner] = None


class ResponsesAnthropicEventTranslator:
    """Ordered Responses-event to Anthropic-event state machine."""

    _NO_CONTENT_EVENTS = {
        "response.queued",
        "response.in_progress",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_part.done",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
        "response.reasoning_text.delta",
        "response.reasoning_text.done",
        "response.output_text.annotation.added",
        "response.web_search_call.in_progress",
        "response.web_search_call.searching",
        "response.web_search_call.completed",
        "keepalive",
    }

    def __init__(
        self,
        *,
        original_model: str,
        reasoning_model: str,
        wire_profile: str,
        name_codec: Optional[IdentifierCodec] = None,
        call_id_codec: Optional[IdentifierCodec] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        mode: str = MODE_COMPATIBILITY,
    ) -> None:
        self.original_model = original_model
        self.reasoning_model = reasoning_model
        self.name_codec = name_codec or IdentifierCodec()
        self.call_id_codec = call_id_codec or IdentifierCodec()
        self.stop_sequences = list(stop_sequences or [])
        self.mode = mode
        self.wire_profile = wire_profile
        self.stable_item_ids = wire_profile != "copilot_responses_lite"

        self.message_started = False
        self.message_stopped = False
        self.response_created_seen = False
        self.created_response_model = ""
        self.response_id = ""
        self.response_model = original_model
        self.next_block_index = 0
        self.next_output_index = 0
        self.states: Dict[int, _OutputState] = {}
        self.open_output_index: Optional[int] = None
        self.local_stop_sequence: Optional[str] = None
        self.non_reasoning_output_committed = False
        self.report = ConversionReport("responses_sse_to_anthropic")
        self.compatibility_warnings: List[Dict[str, Any]] = []
        self.terminal_result: Optional[ResponsesToAnthropicResult] = None
        self.terminal_response: Optional[Dict[str, Any]] = None
        self.protocol_failed = False
        self.error_status_code = 502

    # --------------------------------------------------------------- events

    def _message_id(self) -> str:
        if self.response_id.startswith("msg_"):
            return self.response_id
        return "msg_" + hashlib.sha256(self.response_id.encode("utf-8")).hexdigest()[:24]

    def _start_message(self) -> List[Tuple[str, Dict[str, Any]]]:
        if self.message_started:
            return []
        self.message_started = True
        event = {
            "type": "message_start",
            "message": {
                "id": self._message_id(),
                "type": "message",
                "role": "assistant",
                "model": self.original_model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            },
        }
        return [("message_start", event)]

    def _state(self, output_index: Any) -> _OutputState:
        try:
            index = int(output_index)
        except (TypeError, ValueError):
            index = self.next_output_index
        if index not in self.states:
            self.states[index] = _OutputState(
                output_index=index,
                scanner=StopSequenceScanner(self.stop_sequences),
            )
        return self.states[index]

    @staticmethod
    def _item_type(item: Any) -> str:
        return str(item.get("type") or "") if isinstance(item, dict) else ""

    @staticmethod
    def _content_index(event: Dict[str, Any]) -> Optional[int]:
        value = event.get("content_index", 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return None
        return value

    @staticmethod
    def _combined_text_parts(state: _OutputState) -> str:
        return "".join(state.text_parts[index] for index in sorted(state.text_parts))

    def _append_text_part(
        self,
        state: _OutputState,
        content_index: int,
        delta: str,
    ) -> bool:
        if not state.text_parts and state.text:
            state.text_parts[0] = state.text
        previous = state.text
        state.text_parts[content_index] = (
            state.text_parts.get(content_index, "") + delta
        )
        combined = self._combined_text_parts(state)
        if previous and not combined.startswith(previous):
            return False
        state.text = combined
        return True

    def _set_text_part(
        self,
        state: _OutputState,
        content_index: int,
        full: str,
    ) -> bool:
        if not state.text_parts and state.text:
            state.text_parts[0] = state.text
        current = state.text_parts.get(content_index, "")
        if not full.startswith(current) and full != current:
            return False
        previous = state.text
        state.text_parts[content_index] = full
        combined = self._combined_text_parts(state)
        if previous and not combined.startswith(previous):
            return False
        state.text = combined
        return True

    def _merge_item(
        self,
        state: _OutputState,
        item: Dict[str, Any],
    ) -> Optional[Tuple[str, str]]:
        """Merge an added/done/terminal item or return fatal drift metadata."""

        incoming_type = item.get("type")
        if incoming_type is not None:
            incoming_type = str(incoming_type)
            if state.item_type and incoming_type != state.item_type:
                return (
                    "responses.item_type_mutation",
                    f"/output/{state.output_index}/type",
                )

        effective_type = incoming_type or state.item_type
        if effective_type == "web_search_call" and self.stable_item_ids:
            incoming_id = item.get("id")
            existing_id = state.item.get("id") if isinstance(state.item, dict) else None
            if incoming_id is not None and existing_id is not None and str(incoming_id) != str(existing_id):
                return (
                    "responses.web_search_id_mutation",
                    f"/output/{state.output_index}/id",
                )
        if effective_type in ("function_call", "custom_tool_call"):
            incoming_call_id = item.get("call_id")
            if incoming_call_id is not None:
                incoming_call_id = str(incoming_call_id)
                if state.call_id and incoming_call_id != state.call_id:
                    return (
                        "responses.call_id_mutation",
                        f"/output/{state.output_index}/call_id",
                    )
            incoming_name = item.get("name")
            if incoming_name is not None:
                incoming_name = str(incoming_name)
                if state.name and incoming_name != state.name:
                    return (
                        "responses.tool_name_mutation",
                        f"/output/{state.output_index}/name",
                    )

        if effective_type == "function_call":
            terminal = item.get("arguments")
            if "arguments" in item and not isinstance(terminal, str):
                return (
                    "responses.invalid_function_arguments",
                    f"/output/{state.output_index}/arguments",
                )
            if isinstance(terminal, str):
                if terminal.startswith(state.arguments):
                    state.arguments = terminal
                    state.arguments_complete = True
                elif not state.arguments:
                    state.arguments = terminal
                    state.arguments_complete = True
                else:
                    return (
                        "responses.arguments_mismatch",
                        f"/output/{state.output_index}/arguments",
                    )
        elif effective_type == "custom_tool_call":
            terminal = item.get("input")
            if "input" in item and not isinstance(terminal, str):
                return (
                    "responses.invalid_custom_tool_input",
                    f"/output/{state.output_index}/input",
                )
            if isinstance(terminal, str):
                if terminal.startswith(state.arguments):
                    state.arguments = terminal
                    state.arguments_complete = True
                elif not state.arguments:
                    state.arguments = terminal
                    state.arguments_complete = True
                else:
                    return (
                        "responses.custom_input_mismatch",
                        f"/output/{state.output_index}/input",
                    )
        elif effective_type == "message":
            full_text = ""
            parts = item.get("content")
            if isinstance(parts, list):
                full_parts = {
                    index: str(part.get("text") or part.get("refusal") or "")
                    for index, part in enumerate(parts)
                    if isinstance(part, dict)
                }
                full_text = "".join(full_parts[index] for index in sorted(full_parts))
                if not state.text_parts and not state.text:
                    state.text_parts = full_parts
            if full_text.startswith(state.text):
                state.text = full_text
            elif not state.text:
                state.text = full_text
            elif full_text != state.text:
                return (
                    "responses.text_mismatch",
                    f"/output/{state.output_index}/content",
                )

        state.item = copy.deepcopy(item)
        state.item_type = effective_type
        if effective_type in ("function_call", "custom_tool_call"):
            if item.get("call_id") is not None:
                state.call_id = str(item["call_id"])
            if item.get("name") is not None:
                state.name = str(item["name"])
        return None

    def _warn(self, code: str, path: str, action: str = "error") -> None:
        warning = {"code": code, "path": path, "action": action}
        if warning not in self.compatibility_warnings:
            self.compatibility_warnings.append(warning)

    def _start_block(self, state: _OutputState) -> List[Tuple[str, Dict[str, Any]]]:
        if state.started:
            return []
        if self.open_output_index is not None and self.open_output_index != state.output_index:
            # The drain algorithm should make this impossible. Treat it as a
            # protocol error rather than reopening a closed Anthropic index.
            return self._protocol_error("responses.interleaved_open_block", f"/output/{state.output_index}")
        state.block_index = self.next_block_index
        self.next_block_index += 1
        state.started = True
        self.open_output_index = state.output_index
        if state.item_type == "message":
            block = {"type": "text", "text": ""}
        elif state.item_type == "reasoning":
            block = {"type": "thinking", "thinking": "", "signature": ""}
        else:
            block = {
                "type": "tool_use",
                "id": self.call_id_codec.decode(state.call_id),
                "name": self.name_codec.decode(state.name),
                "input": {},
            }
        return [("content_block_start", {
            "type": "content_block_start",
            "index": state.block_index,
            "content_block": block,
        })]

    def _close_block(self, state: _OutputState) -> List[Tuple[str, Dict[str, Any]]]:
        if not state.started or state.closed:
            return []
        state.closed = True
        self.open_output_index = None
        return [("content_block_stop", {"type": "content_block_stop", "index": state.block_index})]

    def _text_delta(self, state: _OutputState, text: str, scan: bool = True) -> List[Tuple[str, Dict[str, Any]]]:
        if not text or self.local_stop_sequence is not None:
            return []
        out = state.scanner.push(text) if scan and state.scanner else text
        if state.scanner and state.scanner.matched:
            self.local_stop_sequence = state.scanner.matched
        if not out:
            return []
        return [("content_block_delta", {
            "type": "content_block_delta",
            "index": state.block_index,
            "delta": {"type": "text_delta", "text": out},
        })]

    def _argument_delta(self, state: _OutputState, value: str) -> List[Tuple[str, Dict[str, Any]]]:
        if not value:
            return []
        return [("content_block_delta", {
            "type": "content_block_delta",
            "index": state.block_index,
            "delta": {"type": "input_json_delta", "partial_json": value},
        })]

    def _state_ready(self, state: _OutputState) -> bool:
        if state.item_type in ("reasoning", "web_search_call"):
            return True
        if state.item_type == "message":
            return bool(state.text) or state.done
        if state.item_type == "function_call":
            # Buffer the complete argument string so malformed/non-object JSON
            # can never be partially committed to an Anthropic tool block.
            return bool(state.done and state.name and state.call_id)
        if state.item_type == "custom_tool_call":
            # Custom input is not necessarily JSON; wait for the terminal item
            # so it can be wrapped as one valid JSON object.
            return bool(state.done and state.name and state.call_id)
        return False

    def _drain(self) -> List[Tuple[str, Dict[str, Any]]]:
        events: List[Tuple[str, Dict[str, Any]]] = []
        while self.next_output_index in self.states:
            state = self.states[self.next_output_index]
            if not self._state_ready(state):
                break
            if state.item_type == "web_search_call":
                if not state.done:
                    break
                self.non_reasoning_output_committed = True
                self.report.mark(
                    f"/output/{state.output_index}",
                    PRESERVATION_SIDECAR,
                    detail="Native web search execution retained for diagnostics",
                    subtree=True,
                )
                state.closed = True
                self.next_output_index += 1
                continue
            if state.item_type == "reasoning":
                if not state.done:
                    break
                if self.non_reasoning_output_committed:
                    return events + self._protocol_error(
                        "responses.late_reasoning_item",
                        f"/output/{state.output_index}",
                    )
                if not self.message_started:
                    self._warn(
                        "responses.missing_created_event",
                        f"/output/{state.output_index}",
                        "approximation",
                    )
                    events.extend(self._start_message())
                events.extend(self._start_block(state))
                summary = state.item.get("summary")
                summary_text = ""
                if isinstance(summary, list):
                    summary_text = "".join(
                        str(part.get("text") or "")
                        for part in summary
                        if isinstance(part, dict)
                        and part.get("type") in ("summary_text", "reasoning_text")
                    )
                if summary_text:
                    events.append(("content_block_delta", {
                        "type": "content_block_delta",
                        "index": state.block_index,
                        "delta": {"type": "thinking_delta", "thinking": summary_text},
                    }))
                encrypted_content = state.item.get("encrypted_content")
                if not isinstance(encrypted_content, str):
                    encrypted_content = None
                item_id = (
                    state.item.get("id")
                    if self.wire_profile == "public_responses"
                    else None
                )
                if not isinstance(item_id, str):
                    item_id = None
                try:
                    signature = build_reasoning_carrier(
                        model=self.reasoning_model,
                        wire_profile=self.wire_profile,
                        item_id=item_id,
                        encrypted_content=encrypted_content,
                    )
                except ValueError:
                    signature = build_reasoning_carrier(
                        model=self.reasoning_model,
                        wire_profile=self.wire_profile,
                        item_id=item_id,
                        encrypted_content=None,
                    )
                    self.report.mark(
                        f"/output/{state.output_index}/encrypted_content",
                        PRESERVATION_APPROXIMATION,
                        detail="Encrypted reasoning exceeded the carrier size limit and was dropped",
                    )
                events.append(("content_block_delta", {
                    "type": "content_block_delta",
                    "index": state.block_index,
                    "delta": {
                        "type": "signature_delta",
                        "signature": signature,
                    },
                }))
                self.report.mark(
                    f"/output/{state.output_index}",
                    PRESERVATION_SEMANTIC,
                    detail="Responses reasoning carried in an Anthropic thinking block",
                    subtree=True,
                )
                events.extend(self._close_block(state))
                self.next_output_index += 1
                continue
            if state.item_type not in ("message", "function_call", "custom_tool_call"):
                return events + self._protocol_error("responses.unknown_output_item", f"/output/{state.output_index}")
            self.non_reasoning_output_committed = True
            if self.local_stop_sequence is not None:
                # The proxy has already committed an Anthropic stop boundary.
                # Balance a block that began before the matching delta, but do
                # not expose any later message/tool structure from upstream.
                if not state.done:
                    break
                if state.started:
                    events.extend(self._close_block(state))
                else:
                    state.closed = True
                self.next_output_index += 1
                continue
            if not self.message_started:
                self._warn(
                    "responses.missing_created_event",
                    f"/output/{state.output_index}",
                    "approximation",
                )
                events.extend(self._start_message())
            if state.item_type == "function_call":
                try:
                    parsed_arguments = parse_strict_json_bytes(
                        state.arguments.encode("utf-8", errors="strict")
                    )
                except (ValueError, UnicodeEncodeError):
                    return events + self._protocol_error(
                        "responses.invalid_function_arguments",
                        f"/output/{state.output_index}/arguments",
                    )
                if not isinstance(parsed_arguments, dict):
                    return events + self._protocol_error(
                        "responses.invalid_function_arguments",
                        f"/output/{state.output_index}/arguments",
                    )
            elif state.item_type == "custom_tool_call" and not state.arguments_complete:
                return events + self._protocol_error(
                    "responses.invalid_custom_tool_input",
                    f"/output/{state.output_index}/input",
                )
            events.extend(self._start_block(state))
            if self.message_stopped:
                return events
            if state.item_type == "message":
                pending = state.text[state.text_forwarded:]
                state.text_forwarded = len(state.text)
                events.extend(self._text_delta(state, pending))
                if state.done:
                    tail = state.scanner.finish() if state.scanner else ""
                    events.extend(self._text_delta(state, tail, scan=False))
            elif state.item_type == "function_call":
                pending = state.arguments[state.arguments_forwarded:]
                state.arguments_forwarded = len(state.arguments)
                events.extend(self._argument_delta(state, pending))
            else:
                if state.arguments_forwarded == 0:
                    wrapped = _json({"input": state.arguments})
                    state.arguments_forwarded = len(state.arguments)
                    events.extend(self._argument_delta(state, wrapped))
                    self._warn("responses.custom_tool_semantic_wrapper", f"/output/{state.output_index}/input", "approximation")
            if not state.done:
                break
            events.extend(self._close_block(state))
            self.next_output_index += 1
        return events

    def _protocol_error(self, code: str, path: str) -> List[Tuple[str, Dict[str, Any]]]:
        self._warn(code, path)
        self.protocol_failed = True
        self.error_status_code = 502
        if self.message_stopped:
            return []
        self.message_stopped = True
        error = {"type": "error", "error": {"type": "api_error", "message": f"Unsupported Responses stream shape ({code})"}}
        return [("error", error)]

    def _hydrate_terminal(
        self,
        response: Dict[str, Any],
    ) -> Optional[Tuple[str, str]]:
        output = response.get("output")
        if not isinstance(output, list):
            return None
        for index, item in enumerate(output):
            if not isinstance(item, dict):
                continue
            state = self._state(index)
            drift = self._merge_item(state, item)
            if drift is not None:
                return drift
            state.done = True
        return None

    def _terminal_events(
        self,
        response: Dict[str, Any],
        terminal_event_type: str,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        if self.message_stopped:
            return []

        terminal_id = str(response.get("id") or "")
        terminal_model = str(response.get("model") or "")
        if self.response_id and terminal_id != self.response_id:
            return self._protocol_error(
                "responses.terminal_response_id_mismatch", "/response/id"
            )
        if (
            self.response_created_seen
            and self.created_response_model
            and terminal_model
            and terminal_model != self.created_response_model
        ):
            return self._protocol_error(
                "responses.terminal_response_model_mismatch", "/response/model"
            )
        expected_status = {
            "response.completed": "completed",
            "response.incomplete": "incomplete",
        }[terminal_event_type]
        terminal_status = response.get("status")
        if terminal_status is not None and terminal_status != expected_status:
            return self._protocol_error(
                "responses.terminal_status_mismatch", "/response/status"
            )
        if response.get("error") is not None:
            return self._protocol_error(
                "responses.error_on_success_terminal", "/response/error"
            )
        output = response.get("output")
        if isinstance(output, list):
            missing_indices = sorted(set(self.states) - set(range(len(output))))
            if missing_indices:
                return self._protocol_error(
                    "responses.terminal_output_missing_item",
                    f"/output/{missing_indices[0]}",
                )

        self.terminal_response = copy.deepcopy(response)
        self.response_id = terminal_id or self.response_id
        self.response_model = terminal_model or self.response_model
        events = self._start_message()
        terminal_drift = self._hydrate_terminal(response)
        if terminal_drift is not None:
            events.extend(self._protocol_error(*terminal_drift))
            return events
        events.extend(self._drain())
        if self.message_stopped:
            return events
        # Any remaining unknown/missing index is a fatal lifecycle drift.
        remaining = [
            index for index, state in self.states.items()
            if not state.closed
        ]
        if remaining:
            events.extend(self._protocol_error("responses.unclosed_output_item", f"/output/{min(remaining)}"))
            return events
        try:
            self.terminal_result = convert_responses_to_anthropic(
                {"type": terminal_event_type, "response": response},
                original_model=self.original_model,
                reasoning_model=self.reasoning_model,
                wire_profile=self.wire_profile,
                name_codec=self.name_codec,
                call_id_codec=self.call_id_codec,
                stop_sequences=self.stop_sequences,
                mode=self.mode,
            )
        except Exception as exc:
            events.extend(self._protocol_error("responses.terminal_conversion_failed", "/response"))
            self._warn(type(exc).__name__, "/response")
            return events
        if self.local_stop_sequence:
            stop_reason = "stop_sequence"
            stop_sequence = self.local_stop_sequence
        else:
            stop_reason = self.terminal_result.response.get("stop_reason")
            stop_sequence = self.terminal_result.response.get("stop_sequence")
        events.append(("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
            "usage": self.terminal_result.response.get("usage", {}),
        }))
        events.append(("message_stop", {"type": "message_stop"}))
        self.message_stopped = True
        return events

    def process(self, event_type: str, event: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        if self.message_stopped:
            return []
        event_type = event_type or str(event.get("type") or "")
        if event_type == "response.created":
            response = event.get("response") if isinstance(event.get("response"), dict) else {}
            response_id = str(response.get("id") or event.get("response_id") or "")
            response_model = str(response.get("model") or "")
            if not response_id:
                return self._protocol_error(
                    "responses.missing_created_response_id", "/response/id"
                )
            if self.response_created_seen:
                if (
                    response_id != self.response_id
                    or response_model != self.created_response_model
                ):
                    return self._protocol_error(
                        "responses.created_response_mutation", "/response"
                    )
                return []
            self.response_created_seen = True
            self.response_id = response_id
            self.created_response_model = response_model
            self.response_model = response_model or self.original_model
            return self._start_message()
        if event_type in self._NO_CONTENT_EVENTS:
            state = None
            if "output_index" in event:
                try:
                    output_index = int(event["output_index"])
                except (TypeError, ValueError):
                    output_index = None
                state = self.states.get(output_index) if output_index is not None else None
                if state is not None and state.closed:
                    return self._protocol_error(
                        "responses.event_after_closed_output_item",
                        f"/output/{state.output_index}",
                    )
            if event_type == "response.output_text.annotation.added":
                if state is None or state.item_type != "message":
                    return self._protocol_error(
                        "responses.annotation_without_message",
                        f"/events/{event_type}",
                    )
                if self.stable_item_ids:
                    expected_id = str(state.item.get("id") or "")
                    item_id = str(event.get("item_id") or "")
                    if expected_id and item_id != expected_id:
                        return self._protocol_error(
                            "responses.annotation_item_id_mismatch",
                            f"/output/{state.output_index}/id",
                        )
                content_index = event.get("content_index")
                annotation_index = event.get("annotation_index")
                if (
                    not isinstance(content_index, int)
                    or isinstance(content_index, bool)
                    or content_index < 0
                    or not isinstance(annotation_index, int)
                    or isinstance(annotation_index, bool)
                    or annotation_index < 0
                ):
                    return self._protocol_error(
                        "responses.invalid_annotation_index",
                        f"/output/{state.output_index}/content",
                    )
                annotation_key = (content_index, annotation_index)
                if annotation_key in state.annotation_indices:
                    return self._protocol_error(
                        "responses.duplicate_annotation",
                        f"/output/{state.output_index}/content/{content_index}/annotations/{annotation_index}",
                    )
                state.annotation_indices.add(annotation_key)
            if event_type.startswith("response.web_search_call."):
                if state is None or state.item_type != "web_search_call":
                    return self._protocol_error(
                        "responses.web_search_lifecycle_without_item",
                        f"/events/{event_type}",
                    )
                if self.stable_item_ids:
                    expected_id = str(state.item.get("id") or "")
                    item_id = str(event.get("item_id") or "")
                    if expected_id and item_id != expected_id:
                        return self._protocol_error(
                            "responses.web_search_id_mismatch",
                            f"/output/{state.output_index}/id",
                        )
                stage = {
                    "response.web_search_call.in_progress": 1,
                    "response.web_search_call.searching": 2,
                    "response.web_search_call.completed": 3,
                }[event_type]
                if stage <= state.web_search_stage:
                    return self._protocol_error(
                        "responses.web_search_status_regression",
                        f"/output/{state.output_index}/status",
                    )
                state.web_search_stage = stage
            self.report.mark("/events", PRESERVATION_SIDECAR, detail=f"Known lifecycle event {event_type}")
            return []
        if event_type == "response.output_item.added":
            state = self._state(event.get("output_index"))
            if state.closed:
                return self._protocol_error(
                    "responses.event_after_closed_output_item",
                    f"/output/{state.output_index}",
                )
            item = event.get("item") if isinstance(event.get("item"), dict) else {}
            drift = self._merge_item(state, item)
            if drift is not None:
                return self._protocol_error(*drift)
            if state.item_type not in ("reasoning", "message", "function_call", "custom_tool_call", "web_search_call"):
                return self._protocol_error("responses.unknown_output_item", f"/output/{state.output_index}")
            return self._drain()
        if event_type == "response.output_item.done":
            state = self._state(event.get("output_index"))
            if state.closed:
                return self._protocol_error(
                    "responses.event_after_closed_output_item",
                    f"/output/{state.output_index}",
                )
            item = event.get("item") if isinstance(event.get("item"), dict) else {}
            drift = self._merge_item(state, item)
            if drift is not None:
                return self._protocol_error(*drift)
            state.done = True
            return self._drain()
        if event_type in ("response.content_part.added", "response.content_part.done"):
            state = self._state(event.get("output_index"))
            if state.closed:
                return self._protocol_error(
                    "responses.event_after_closed_output_item",
                    f"/output/{state.output_index}",
                )
            if state.item_type and state.item_type != "message":
                return self._protocol_error(
                    "responses.item_type_mutation",
                    f"/output/{state.output_index}/type",
                )
            part = event.get("part") if isinstance(event.get("part"), dict) else {}
            if not state.item_type:
                state.item_type = "message"
            content_index = self._content_index(event)
            if content_index is None:
                return self._protocol_error(
                    "responses.invalid_content_index",
                    f"/output/{state.output_index}/content",
                )
            part_type = part.get("type")
            key = "refusal" if part_type == "refusal" else "text"
            full = part.get(key)
            if part_type in ("output_text", "refusal") and isinstance(full, str):
                if not self._set_text_part(state, content_index, full):
                    return self._protocol_error(
                        "responses.text_done_mismatch",
                        f"/output/{state.output_index}/content/{content_index}",
                    )
            return self._drain()
        if event_type == "response.output_text.delta":
            state = self._state(event.get("output_index"))
            if state.closed:
                return self._protocol_error(
                    "responses.event_after_closed_output_item",
                    f"/output/{state.output_index}",
                )
            if state.item_type and state.item_type != "message":
                return self._protocol_error(
                    "responses.item_type_mutation",
                    f"/output/{state.output_index}/type",
                )
            state.item_type = state.item_type or "message"
            content_index = self._content_index(event)
            if content_index is None:
                return self._protocol_error(
                    "responses.invalid_content_index",
                    f"/output/{state.output_index}/content",
                )
            if not self._append_text_part(
                state, content_index, str(event.get("delta") or "")
            ):
                return self._protocol_error(
                    "responses.content_part_reordering",
                    f"/output/{state.output_index}/content/{content_index}",
                )
            return self._drain()
        if event_type == "response.output_text.done":
            state = self._state(event.get("output_index"))
            if state.closed:
                return self._protocol_error(
                    "responses.event_after_closed_output_item",
                    f"/output/{state.output_index}",
                )
            if state.item_type and state.item_type != "message":
                return self._protocol_error(
                    "responses.item_type_mutation",
                    f"/output/{state.output_index}/type",
                )
            content_index = self._content_index(event)
            if content_index is None:
                return self._protocol_error(
                    "responses.invalid_content_index",
                    f"/output/{state.output_index}/content",
                )
            full = event.get("text")
            if isinstance(full, str) and not self._set_text_part(
                state, content_index, full
            ):
                return self._protocol_error(
                    "responses.text_done_mismatch",
                    f"/output/{state.output_index}/content/{content_index}",
                )
            return self._drain()
        if event_type in ("response.function_call_arguments.delta", "response.custom_tool_call_input.delta"):
            state = self._state(event.get("output_index"))
            if state.closed:
                return self._protocol_error(
                    "responses.event_after_closed_output_item",
                    f"/output/{state.output_index}",
                )
            expected_type = "custom_tool_call" if "custom_tool" in event_type else "function_call"
            if state.item_type and state.item_type != expected_type:
                return self._protocol_error(
                    "responses.item_type_mutation",
                    f"/output/{state.output_index}/type",
                )
            state.item_type = state.item_type or expected_type
            state.arguments += str(event.get("delta") or "")
            return self._drain()
        if event_type in ("response.function_call_arguments.done", "response.custom_tool_call_input.done"):
            state = self._state(event.get("output_index"))
            if state.closed:
                return self._protocol_error(
                    "responses.event_after_closed_output_item",
                    f"/output/{state.output_index}",
                )
            expected_type = "custom_tool_call" if "custom_tool" in event_type else "function_call"
            if state.item_type and state.item_type != expected_type:
                return self._protocol_error(
                    "responses.item_type_mutation",
                    f"/output/{state.output_index}/type",
                )
            state.item_type = state.item_type or expected_type
            key = "input" if "custom_tool" in event_type else "arguments"
            full = event.get(key)
            if isinstance(full, str):
                if full.startswith(state.arguments):
                    state.arguments = full
                    state.arguments_complete = True
                elif full != state.arguments:
                    return self._protocol_error(
                        "responses.arguments_done_mismatch",
                        f"/output/{state.output_index}/{key}",
                    )
            else:
                return self._protocol_error(
                    "responses.invalid_custom_tool_input"
                    if expected_type == "custom_tool_call"
                    else "responses.invalid_function_arguments",
                    f"/output/{state.output_index}/{key}",
                )
            return self._drain()
        if event_type == "response.refusal.delta":
            state = self._state(event.get("output_index"))
            if state.closed:
                return self._protocol_error(
                    "responses.event_after_closed_output_item",
                    f"/output/{state.output_index}",
                )
            if state.item_type and state.item_type != "message":
                return self._protocol_error(
                    "responses.item_type_mutation",
                    f"/output/{state.output_index}/type",
                )
            state.item_type = state.item_type or "message"
            content_index = self._content_index(event)
            if content_index is None:
                return self._protocol_error(
                    "responses.invalid_content_index",
                    f"/output/{state.output_index}/content",
                )
            if not self._append_text_part(
                state, content_index, str(event.get("delta") or "")
            ):
                return self._protocol_error(
                    "responses.content_part_reordering",
                    f"/output/{state.output_index}/content/{content_index}",
                )
            self._warn("responses.refusal_projected_as_text", f"/output/{state.output_index}", "approximation")
            return self._drain()
        if event_type == "response.refusal.done":
            state = self._state(event.get("output_index"))
            if state.closed:
                return self._protocol_error(
                    "responses.event_after_closed_output_item",
                    f"/output/{state.output_index}",
                )
            if state.item_type and state.item_type != "message":
                return self._protocol_error(
                    "responses.item_type_mutation",
                    f"/output/{state.output_index}/type",
                )
            state.item_type = state.item_type or "message"
            content_index = self._content_index(event)
            if content_index is None:
                return self._protocol_error(
                    "responses.invalid_content_index",
                    f"/output/{state.output_index}/content",
                )
            full = event.get("refusal")
            if not isinstance(full, str):
                return self._protocol_error(
                    "responses.invalid_refusal",
                    f"/output/{state.output_index}/content/{content_index}",
                )
            if not self._set_text_part(state, content_index, full):
                return self._protocol_error(
                    "responses.text_done_mismatch",
                    f"/output/{state.output_index}/content/{content_index}",
                )
            self._warn("responses.refusal_projected_as_text", f"/output/{state.output_index}", "approximation")
            return self._drain()
        if event_type in ("response.completed", "response.incomplete"):
            response = event.get("response") if isinstance(event.get("response"), dict) else event
            return self._terminal_events(response, event_type)
        if event_type in ("response.failed", "error"):
            error_value = event.get("response") if event_type == "response.failed" else event
            error = anthropic_error_from_responses(error_value, 500)
            self.protocol_failed = True
            self.error_status_code = 502
            self.message_stopped = True
            return [("error", error)]
        # An event type this build does not know. Skipping it is not the same
        # as losing its payload: the terminal response.completed/.incomplete
        # carries the full output array and _hydrate_terminal merges it, so any
        # model output the skipped event previewed is still delivered -- at the
        # end of the stream rather than incrementally. Anything that would
        # actually corrupt or drop output still fails closed elsewhere: an
        # unknown item type or content part is rejected by the auditor and by
        # convert_responses_to_anthropic, a conflicting item index trips
        # item_type_mutation, and a stream with no terminal event trips
        # stream_ended_without_terminal. lossless_required rejects via the
        # auditor before reaching here.
        self._warn(
            "responses.unknown_event_skipped",
            f"/events/{event_type or 'missing'}",
            "approximation",
        )
        return self._drain()

    def finalize_interrupted(self) -> List[Tuple[str, Dict[str, Any]]]:
        if self.message_stopped:
            return []
        return self._protocol_error("responses.stream_ended_without_terminal", "/events")


class AnthropicResponsesStreamHandler(SSEStreamHandler):
    """Shared-transport adapter for the Responses->Anthropic state machine."""

    endpoint = "/v1/messages"
    log_prefix = "[Stream Anthropic/Responses]"
    emit_event_header = True
    emit_done_sentinel = False
    capture_raw_sse_lines = False

    def __init__(
        self,
        *args: Any,
        conversion: AnthropicToResponsesResult,
        compatibility_warnings: Optional[List[Dict[str, Any]]] = None,
        compatibility_audit: Optional[Dict[str, Any]] = None,
        mode: str = MODE_COMPATIBILITY,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.conversion = conversion
        self.translator = ResponsesAnthropicEventTranslator(
            original_model=self.original_model,
            reasoning_model=self.translated_model,
            wire_profile=conversion.wire_profile,
            name_codec=conversion.name_codec,
            call_id_codec=conversion.call_id_codec,
            stop_sequences=conversion.stop_sequences,
            mode=mode,
        )
        self._compatibility_warnings = list(compatibility_warnings or [])
        self._compatibility_audit = copy.deepcopy(compatibility_audit)
        self._mode = mode

    def keepalive_event(self) -> str:
        return 'event: ping\ndata: {"type":"ping"}\n\n'

    def parse_event_data(self, data: str) -> Dict[str, Any]:
        value = parse_strict_json_bytes(data.encode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Responses SSE data must be a JSON object")
        return value

    def forward_malformed_data(self, data: str) -> Iterator[str]:
        # Never leak a foreign/malformed Responses payload into an Anthropic
        # stream. The raw bytes are still retained by the base cache handler.
        self.translator._warn("responses.malformed_json_event", "/events")
        self.translator.protocol_failed = True
        self.translator.error_status_code = 502
        self.translator.message_stopped = True
        self.error_occurred = True
        self.status_code = 502
        event = {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "Upstream Responses stream contained malformed JSON",
            },
        }
        yield f"event: error\ndata: {_json(event)}\n\n"

    def raw_events_for_cache(self) -> List[str]:
        result: List[str] = []
        for raw in self.raw_events:
            try:
                event = parse_strict_json_bytes(raw.encode("utf-8"))
                safe = redact_responses_event_for_cache(event)
            except Exception:
                safe = redacted_value(raw, "malformed Responses event")
            result.append(json.dumps(
                safe,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ))
        return result

    def forward_event(self, event_type: str, event: Dict[str, Any], raw_data: str) -> Iterator[Tuple[str, str]]:
        payload_type = str(event.get("type") or "")
        if event_type and payload_type and event_type != payload_type:
            for out_type, out_event in self.translator._protocol_error(
                "responses.event_type_mismatch", "/events/type"
            ):
                yield out_type, _json(out_event)
            return
        audit_value = event
        if event_type and not payload_type:
            audit_value = {**event, "type": event_type}
        audit = audit_responses_event(audit_value, mode=self._mode)
        for warning in audit.warnings:
            if warning not in self.translator.compatibility_warnings:
                self.translator.compatibility_warnings.append(warning)
        if audit.should_fail:
            for out_type, out_event in self.translator._protocol_error(
                "responses.profile_drift", "/events"
            ):
                yield out_type, _json(out_event)
            return
        translated = self.translator.process(event_type, event)
        if self.translator.protocol_failed:
            self.error_occurred = True
            self.status_code = self.translator.error_status_code
        for out_type, out_event in translated:
            yield out_type, _json(out_event)
        terminal = self.translator.terminal_result
        if terminal is not None:
            usage = terminal.response.get("usage", {})
            self.input_tokens = int(usage.get("input_tokens") or 0)
            self.output_tokens = int(usage.get("output_tokens") or 0)
            self.cache_creation_input_tokens = int(usage.get("cache_creation_input_tokens") or 0)
            self.cache_read_input_tokens = int(usage.get("cache_read_input_tokens") or 0)

    def finalize_stream(self) -> Iterator[Tuple[str, str]]:
        translated = self.translator.finalize_interrupted()
        if self.translator.protocol_failed:
            self.error_occurred = True
            self.status_code = self.translator.error_status_code
        for out_type, out_event in translated:
            yield out_type, _json(out_event)

    def _format_generic_error(self, exc: Exception) -> str:
        event = {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "Internal Responses stream translation failed",
            },
        }
        return f"event: error\ndata: {_json(event)}\n\n"

    def _format_transport_error(self, exc: Exception) -> str:
        event = {
            "type": "error",
            "error": {
                "type": "timeout_error",
                "message": "Upstream Responses stream timed out",
            },
        }
        return f"event: error\ndata: {_json(event)}\n\n"

    def extra_cache_fields(self) -> Dict[str, Any]:
        warnings = self._compatibility_warnings + self.conversion.report.warnings + self.translator.compatibility_warnings
        result: Dict[str, Any] = {
            "compatibility_profile": self.conversion.wire_profile,
            "compatibility_warnings": warnings,
            "conversion_report": {
                "request": self.conversion.report.to_dict(),
                "stream": self.translator.report.to_dict(),
            },
            "raw_events_redacted": True,
            "raw_capture_truncated": self.raw_capture_truncated,
        }
        if self._compatibility_audit is not None:
            result["compatibility_audit"] = copy.deepcopy(self._compatibility_audit)
        if self.translator.terminal_result is not None:
            result["conversion_report"]["response"] = self.translator.terminal_result.report.to_dict()
            result["response_body"] = redact_reasoning_carriers_for_cache(
                self.translator.terminal_result.response
            )
        return result
