"""Kimi K3/Papyrus to OpenAI Chat Completions streaming adapter."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Sequence, Tuple

from ..proxy.kimi_k3 import KimiTextStream
from .proxy_chat_completions import ProxyChatCompletionsStreamHandler


class KimiK3ChatCompletionsStreamHandler(ProxyChatCompletionsStreamHandler):
    """Translate thinking/native tools and synthesize a valid terminal chunk."""

    # The base sees a conforming data sentinel before finalize_stream().  Kimi
    # needs to emit its synthesized finish chunk first, so finalize_stream owns
    # both the terminal chunk and sentinel for this adapter only.
    emit_done_sentinel = False
    capture_raw_sse_lines = True
    passthrough_event_headers = False

    def __init__(self, *args, declared_tools: Sequence[str] = (), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.declared_tools = tuple(declared_tools)
        self._text_streams: Dict[int, KimiTextStream] = {}
        self._finish_reasons: Dict[int, str] = {}
        self._saw_tool_calls: Dict[int, bool] = {}
        self._metadata: Dict[str, Any] = {}
        self.public_events: List[str] = []

    def _stream_for(self, index: int) -> KimiTextStream:
        stream = self._text_streams.get(index)
        if stream is None:
            stream = KimiTextStream(self.declared_tools)
            self._text_streams[index] = stream
        return stream

    def _remember_metadata(self, event: Dict[str, Any]) -> None:
        for key, value in event.items():
            if key not in {"choices", "usage"}:
                self._metadata[key] = value

    def _event_for_choice(self, index: int, delta: Dict, finish_reason=None) -> Dict:
        event = dict(self._metadata)
        event.setdefault("object", "chat.completion.chunk")
        if self.rewrite_model and "model" in event:
            event["model"] = self.public_model
        event["choices"] = [{
            "index": index,
            "delta": delta,
            "finish_reason": finish_reason,
        }]
        return event

    def _emit(self, event: Dict) -> Tuple[str, str]:
        data = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        self.public_events.append(data)
        return "", data

    def _text_events(self, index: int, values) -> Iterator[Tuple[str, str]]:
        for field, text in values:
            if not text:
                continue
            yield self._emit(self._event_for_choice(index, {field: text}))

    def forward_event(
        self, event_type: str, event: Dict, raw_data: str
    ) -> Iterator[tuple]:
        if not isinstance(event, dict):
            return
        self._remember_metadata(event)
        choices = event.get("choices")
        if not isinstance(choices, list):
            # Usage-only chunks remain standard and are useful to clients that
            # requested stream_options.include_usage.
            rewritten = dict(event)
            if self.rewrite_model and "model" in rewritten:
                rewritten["model"] = self.public_model
            yield self._emit(rewritten)
            return
        if not choices:
            rewritten = dict(event)
            if self.rewrite_model and "model" in rewritten:
                rewritten["model"] = self.public_model
            yield self._emit(rewritten)
            return

        for raw_choice in choices:
            if not isinstance(raw_choice, dict):
                continue
            index = raw_choice.get("index", 0)
            if not isinstance(index, int):
                index = 0
            finish_reason = raw_choice.get("finish_reason")
            if isinstance(finish_reason, str) and finish_reason:
                self._finish_reasons[index] = finish_reason

            raw_delta = raw_choice.get("delta")
            if not isinstance(raw_delta, dict):
                continue
            delta = dict(raw_delta)
            content = delta.pop("content", None)
            # Null content carries no information and confuses some strict SDKs.
            if delta.get("reasoning_content") is None:
                delta.pop("reasoning_content", None)
            if isinstance(delta.get("tool_calls"), list) and delta["tool_calls"]:
                self._saw_tool_calls[index] = True

            if delta:
                yield self._emit(self._event_for_choice(index, delta))
            if isinstance(content, str):
                yield from self._text_events(index, self._stream_for(index).feed(content))
            # Upstream terminal chunks are intentionally suppressed.  A single,
            # valid terminal chunk is synthesized in finalize_stream().

    def forward_raw_line(self, line: str) -> Iterator[str]:
        if line.strip() == "[DONE]":
            return
        yield from super().forward_raw_line(line)

    def forward_malformed_data(self, data: str) -> Iterator[str]:
        # Never forward malformed upstream bytes as an apparent Chat Completion
        # event.  The raw value is already retained by SSEStreamHandler.
        self.error_occurred = True
        self.status_code = 502
        return iter(())

    def finalize_stream(self) -> Iterator[tuple]:
        indexes = sorted(set(self._text_streams) | set(self._finish_reasons) | set(self._saw_tool_calls))
        if not indexes:
            indexes = [0]
        for index in indexes:
            stream = self._text_streams.get(index)
            calls = None
            if stream is not None:
                text_values, calls = stream.finish()
                yield from self._text_events(index, text_values)
            if calls:
                self._saw_tool_calls[index] = True
                yield self._emit(self._event_for_choice(index, {"tool_calls": calls}))

            if self._saw_tool_calls.get(index):
                finish = "tool_calls"
            else:
                upstream_finish = self._finish_reasons.get(index)
                finish = upstream_finish if upstream_finish in {"stop", "length", "content_filter"} else "stop"
            yield self._emit(self._event_for_choice(index, {}, finish))

        # An empty event type is ignored because this handler emits data-only SSE.
        yield "", "[DONE]"

    def extra_cache_fields(self) -> Dict:
        fields = super().extra_cache_fields()
        fields.update({
            "compatibility": "kimi_k3_papyrus",
            "public_events": list(self.public_events),
            "raw_sse_lines": list(self.raw_sse_lines),
            "response_body": list(self.public_events),
        })
        return fields
