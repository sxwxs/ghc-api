"""Responses clients must see one ID for each streamed output item."""

import copy
import json
from unittest import mock

import pytest

from ghc_api.cache import RequestCache
from ghc_api.sse import OpenAIResponsesStreamHandler
from ghc_api.sse import base as base_module


class FakeResponse:
    status_code = 200

    def __init__(self, events):
        self.events = events

    def iter_lines(self):
        for event in self.events:
            yield ("event: " + event["type"]).encode()
            yield ("data: " + json.dumps(event, ensure_ascii=False)).encode()
            yield b""
        yield b"data: [DONE]"

    def close(self):
        pass


def handler(events=()):
    return OpenAIResponsesStreamHandler(
        response=FakeResponse(events),
        request_id="item-id-test",
        request_size=0,
        start_time=0,
        original_model="test-model",
        translated_model="test-model",
        request_body_for_cache={},
    )


def forward(stream, event):
    original = copy.deepcopy(event)
    raw = json.dumps(event, ensure_ascii=False)
    result = list(stream.forward_event(event["type"], event, raw))
    assert event == original  # The upstream/cache view must remain unmodified.
    assert len(result) == 1
    assert result[0][0] == event["type"]
    return json.loads(result[0][1])


def item_event(kind, index, item_id, item_type="message", **fields):
    return {
        "type": "response.output_item." + kind,
        "output_index": index,
        "item": {"type": item_type, "id": item_id, **fields},
    }


def test_rotating_ids_render_one_message_and_keep_raw_cache():
    events = [
        {"type": "response.created", "response": {"id": "resp-1", "output": []}},
        item_event("added", 0, "opaque-start", content=[]),
        {"type": "response.content_part.added", "output_index": 0,
         "item_id": "opaque-part", "content_index": 0,
         "part": {"type": "output_text", "text": ""}},
        {"type": "response.output_text.delta", "output_index": 0,
         "item_id": "opaque-delta-1", "content_index": 0, "delta": "h"},
        {"type": "response.output_text.delta", "output_index": 0,
         "item_id": "opaque-delta-2", "content_index": 0, "delta": "i"},
        {"type": "response.output_text.done", "output_index": 0,
         "item_id": "opaque-text-done", "content_index": 0, "text": "hi"},
        {"type": "response.content_part.done", "output_index": 0,
         "item_id": "opaque-part-done", "content_index": 0,
         "part": {"type": "output_text", "text": "hi"}},
        item_event("done", 0, "opaque-done", content=[{"type": "output_text", "text": "hi"}]),
        {"type": "response.completed", "response": {
            "id": "resp-final", "output": [{"type": "message", "id": "opaque-final",
                "content": [{"type": "output_text", "text": "hi"}]}],
            "usage": {"input_tokens": 3, "output_tokens": 1}}},
    ]
    cache = RequestCache()
    with mock.patch.object(base_module, "cache", cache):
        chunks = list(handler(events)._generate())
    output = [json.loads(line[6:]) for line in "".join(chunks).splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]

    # A client replaces completed items by ID and appends delta text by ID.
    messages = {}
    for event in output:
        if event["type"] in ("response.output_item.added", "response.output_item.done"):
            item = event["item"]
            messages[item["id"]] = "".join(part["text"] for part in item["content"])
        elif event["type"] == "response.output_text.delta":
            key = event["item_id"]
            messages[key] = messages.get(key, "") + event["delta"]
        if "item_id" in event:
            assert event["item_id"] == "opaque-start"

    assert messages == {"opaque-start": "hi"}
    assert output[-1]["response"]["output"][0]["id"] == "opaque-start"
    assert output[-1]["response"]["id"] == "resp-final"
    assert "data: [DONE]\n\n" in chunks
    entry = cache.get_request("item-id-test")
    assert [json.loads(raw) for raw in entry["raw_events"]] == events
    assert entry["output_tokens"] == 1


def test_healthy_stream_is_byte_for_byte_passthrough_and_stays_incremental():
    stream = handler()
    events = [
        item_event("added", 0, "msg-1"),
        {"type": "response.output_text.delta", "output_index": 0, "item_id": "msg-1", "delta": "你好"},
        item_event("done", 0, "msg-1"),
        {"type": "response.completed", "response": {"output": [{"id": "msg-1"}]}},
    ]
    for event in events:
        raw = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        assert list(stream.forward_event(event["type"], event, raw)) == [(event["type"], raw)]


@pytest.mark.parametrize("terminal", ["response.completed", "response.incomplete", "response.failed"])
def test_interleaved_outputs_preserve_tool_identity_reasoning_and_terminal_items(terminal):
    stream = handler()
    reasoning = {"type": "reasoning", "id": "r-start", "summary": []}
    tool = {"type": "function_call", "id": "f-start", "call_id": "call-1", "name": "greet", "arguments": ""}
    message = {"type": "message", "id": "m-start", "content": []}
    for index, item in enumerate([reasoning, tool, message]):
        forward(stream, {"type": "response.output_item.added", "output_index": index, "item": item})
    for index, event_type in [(2, "response.output_text.delta"), (0, "response.reasoning_summary_text.delta"),
                              (1, "response.function_call_arguments.delta")]:
        event = forward(stream, {"type": event_type, "output_index": index,
                                 "item_id": "rotated", "delta": "x"})
        assert event["item_id"] == ["r-start", "f-start", "m-start"][index]

    final_items = [
        {**reasoning, "id": "r-final", "encrypted_content": "opaque-reasoning", "summary": []},
        {**tool, "id": "f-final", "arguments": '{"language":"zh"}'},
        {**message, "id": "m-final", "content": [{"type": "output_text", "text": "hi"}]},
    ]
    for index, item in enumerate(final_items):
        result = forward(stream, {"type": "response.output_item.done", "output_index": index, "item": item})
        assert result["item"] == {**item, "id": ["r-start", "f-start", "m-start"][index]}
    result = forward(stream, {"type": terminal, "response": {"output": final_items}})
    assert result["response"]["output"] == [
        {**item, "id": canonical} for item, canonical in zip(final_items, ["r-start", "f-start", "m-start"])
    ]


def test_first_reference_can_arrive_before_added_and_handlers_are_isolated():
    first = handler()
    delta = {"type": "response.output_text.delta", "output_index": 0, "item_id": "first", "delta": "hi"}
    forward(first, delta)
    assert forward(first, item_event("done", 0, "later"))["item"]["id"] == "first"
    assert forward(handler(), item_event("done", 0, "other-request"))["item"]["id"] == "other-request"


@pytest.mark.parametrize("index", [None, -1, True, "0", 0.5])
def test_unusable_output_index_is_not_guessed(index):
    stream = handler()
    forward(stream, item_event("added", 0, "known"))
    event = item_event("done", index, "unchanged")
    assert forward(stream, event) == event


def test_unrelated_fields_and_missing_identifiers_are_not_rewritten():
    stream = handler()
    forward(stream, item_event("added", 0, "known"))
    for event in [
        {"type": "error", "id": "error-id", "message": "failure"},
        {"type": "response.output_text.delta", "item_id": "without-index", "delta": "hi"},
        {"type": "response.output_text.delta", "output_index": 0, "delta": "hi"},
        item_event("done", 0, None),
        {"type": "response.completed", "response": {"output": []}},
    ]:
        assert forward(stream, event) == event
