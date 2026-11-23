import json
from unittest import mock

import pytest
import requests

from ghc_api.app import create_app
from ghc_api.cache import RequestCache
from ghc_api.routes import openai
from ghc_api.state import state


MODEL = "gemini-4-hydrogen"
USAGE = {"prompt_tokens": 100, "completion_tokens": 30,
         "prompt_tokens_details": {"cached_tokens": 80}}
EMPTY = {"id": "empty", "model": MODEL, "choices": [
    {"index": 0, "delta": {"content": None, "role": "assistant"}}
], "usage": USAGE}


def chunk(delta, finish=None):
    return {"id": "answer", "model": MODEL, "choices": [
        {"index": 0, "delta": delta, "finish_reason": finish}
    ]}


class Upstream:
    status_code = 200

    def __init__(self, chunks, done=True):
        self.chunks = chunks
        self.done = done
        self.closed = False

    def iter_lines(self):
        for item in self.chunks:
            yield ("data: " + json.dumps(item)).encode()
            yield b""
        if self.done:
            yield b"data: [DONE]"

    def close(self):
        self.closed = True


@pytest.fixture
def stream(monkeypatch):
    cache = RequestCache()
    monkeypatch.setattr(openai, "cache", cache)
    monkeypatch.setattr(state, "sse_keepalive_interval", 0)
    monkeypatch.setattr(state, "max_connection_retries", 3)
    app = create_app()

    def run(responses, model=MODEL, retries=None):
        if retries is not None:
            monkeypatch.setattr(state, "max_connection_retries", retries)
        payload = {"model": model, "messages": [{"role": "user", "content": "Hi"}], "stream": True}
        with app.test_request_context("/v1/chat/completions"):
            with mock.patch.object(openai.requests, "post", side_effect=responses) as post:
                response = openai.stream_chat_completions(
                    payload, {}, "test-request", "{}", 2, 0, model, model,
                )
                body = "".join(response.response)
        events = [json.loads(line[6:]) for line in body.splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
        return body, events, post, cache.get_record("test-request"), payload

    return run


def test_usage_only_empty_stream_retries_same_request_and_counts_both_attempts(stream):
    first = Upstream([EMPTY])
    final_usage = {"prompt_tokens": 50, "completion_tokens": 4,
                   "prompt_tokens_details": {"cached_tokens": 40}}
    second = Upstream([chunk({"role": "assistant", "content": "Hi"}),
                       {**chunk({}, "stop"), "usage": final_usage}])
    body, events, post, record, payload = stream([first, second])
    assert post.call_count == 2
    assert all(call.kwargs["json"] == payload for call in post.call_args_list)
    assert [event["id"] for event in events] == ["answer", "answer"]
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert "data: [DONE]" in body
    assert record["status_code"] == 200
    assert (record["input_tokens"], record["cache_read_input_tokens"], record["output_tokens"]) == (30, 120, 34)
    assert first.closed and second.closed


def test_exhausted_empty_streams_report_error_and_cost(stream):
    body, events, post, record, _ = stream([Upstream([EMPTY]), Upstream([EMPTY])])
    assert post.call_count == 2
    assert events == [{"error": {"message": "Upstream Gemini stream ended without output or a finish reason.",
                                 "type": "upstream_error", "code": "empty_stream"}}]
    assert "[DONE]" not in body
    assert record["status_code"] == 502
    assert record["response_body"] == events[0]
    assert (record["input_tokens"], record["cache_read_input_tokens"], record["output_tokens"]) == (40, 160, 60)


def test_retry_can_be_disabled(stream):
    _, events, post, record, _ = stream([Upstream([EMPTY])], retries=0)
    assert post.call_count == 1
    assert events[0]["error"]["code"] == "empty_stream"
    assert record["status_code"] == 502


@pytest.mark.parametrize("delta", [
    {"content": "partial"},
    {"reasoning_text": "thinking"},
    {"reasoning_opaque": "signature"},
    {"tool_calls": [{"index": 0, "id": "call-1", "type": "function",
                     "function": {"name": "lookup", "arguments": "{}"}}]},
])
def test_never_replay_after_any_visible_or_tool_output(stream, delta):
    body, events, post, _, _ = stream([Upstream([chunk(delta), EMPTY])])
    assert post.call_count == 1
    outgoing = events[0]["choices"][0]["delta"]
    for key, value in delta.items():
        assert outgoing[key] == value
    assert "[DONE]" in body


def test_empty_but_finished_response_is_not_replayed(stream):
    body, events, post, record, _ = stream([Upstream([chunk({"content": None}, "stop")])])
    assert post.call_count == 1
    assert events[0]["choices"][0]["finish_reason"] == "stop"
    assert "[DONE]" in body
    assert record["status_code"] == 200


@pytest.mark.parametrize("response", [Upstream([EMPTY], done=False), Upstream([chunk({})])])
def test_incomplete_or_usage_free_stream_is_not_replayed(stream, response):
    _, events, post, _, _ = stream([response])
    assert post.call_count == 1
    assert len(events) == 1


def test_malformed_data_prevents_replay(stream):
    class Malformed(Upstream):
        def iter_lines(self):
            yield b"data: {not-json}"
            yield from super().iter_lines()

    _, events, post, _, _ = stream([Malformed([EMPTY])])
    assert post.call_count == 1
    assert events == [EMPTY]


def test_invalid_delta_is_not_treated_as_an_empty_chunk():
    assert not openai._is_empty_gemini_chat_chunk(chunk([]))


def test_non_gemini_stream_is_unchanged(stream):
    body, events, post, record, _ = stream([Upstream([EMPTY])], model="claude-sonnet")
    assert post.call_count == 1
    assert events == [EMPTY]
    assert "[DONE]" in body
    assert record["status_code"] == 200


def test_failed_retry_keeps_first_attempt_usage(stream):
    _, _, post, record, _ = stream([Upstream([EMPTY]), requests.exceptions.ConnectionError()])
    assert post.call_count == 2
    assert record["status_code"] == 504
    assert (record["input_tokens"], record["cache_read_input_tokens"], record["output_tokens"]) == (20, 80, 30)
