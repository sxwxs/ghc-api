import json
import threading
import unittest
from contextlib import ExitStack
from unittest import mock

import requests

from ghc_api.api_helpers import supports_responses_api, supports_responses_via_chat_completions
from ghc_api.app import create_app
from ghc_api.cache import RequestCache
from ghc_api.routes import openai
from ghc_api.sse import base
from ghc_api.state import state


MODEL = "gemini-3.8-flash"


class _UpstreamResponse:
    def __init__(self, body=None, chunks=None, status_code=200):
        self.body = body
        self.chunks = chunks
        self.status_code = status_code
        self.ok = status_code < 400
        self.closed = False

    @property
    def text(self):
        if self.chunks is not None:
            raise AssertionError("A streaming body must not be read through .text")
        return json.dumps(self.body)

    def json(self):
        return self.body

    def iter_lines(self):
        for chunk in self.chunks:
            yield ("data: " + json.dumps(chunk)).encode()
            yield b""
        yield b"data: [DONE]"

    def close(self):
        self.closed = True


def _events(response):
    return [json.loads(line[6:]) for line in response.get_data(as_text=True).splitlines() if line.startswith("data: ")]


class GeminiResponsesTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for field, value in {
            "models": {"data": [{"id": MODEL, "supported_endpoints": ["/chat/completions"]}]},
            "enable_auth": False,
            "sse_keepalive_interval": 0,
            "max_connection_retries": 0,
            "enable_responses_early_failure_retry": False,
            "auto_remove_encrypted_content_on_parse_error": False,
            "responses_pre_header_grace": 0,
        }.items():
            self.stack.enter_context(mock.patch.object(state, field, value))
        self.cache = RequestCache()
        self.stack.enter_context(mock.patch.object(openai, "cache", self.cache))
        self.stack.enter_context(mock.patch.object(base, "cache", self.cache))
        self.stack.enter_context(mock.patch.object(openai, "ensure_copilot_token"))
        self.headers = self.stack.enter_context(mock.patch.object(openai, "get_copilot_headers", return_value={}))
        self.stack.enter_context(mock.patch.object(openai, "get_copilot_base_url", return_value="https://copilot.invalid"))
        self.post = self.stack.enter_context(mock.patch.object(openai.requests, "post"))
        self.client = create_app().test_client()

    def completion(self, content="Hello", finish_reason="stop", **message):
        return _UpstreamResponse({
            "id": "chatcmpl-upstream",
            "choices": [{"message": {"role": "assistant", "content": content, **message}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 2}},
        })

    def test_native_and_bridged_capabilities_are_separate(self):
        self.assertFalse(supports_responses_api(MODEL))
        self.assertTrue(supports_responses_via_chat_completions(MODEL))
        self.assertFalse(supports_responses_via_chat_completions("gemini-unknown"))
        state.models["data"][0]["supported_endpoints"] = ["/v1/chat/completions"]
        self.assertTrue(supports_responses_via_chat_completions(MODEL))
        with mock.patch("ghc_api.api_helpers.is_configured_chat_completions_support_added", return_value=True):
            self.assertFalse(supports_responses_via_chat_completions(MODEL))

    def test_nonstreaming_request_returns_responses_json_and_caches_usage(self):
        self.post.return_value = self.completion()
        response = self.client.post("/v1/responses", json={
            "model": MODEL, "input": "Hi", "instructions": "Be concise", "max_output_tokens": 30,
        })
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["object"], "response")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["output"][0]["content"][0]["text"], "Hello")
        self.assertEqual(result["usage"]["input_tokens"], 10)
        self.assertEqual(result["usage"]["input_tokens_details"]["cached_tokens"], 2)
        self.assertEqual(self.post.call_args.args[0], "https://copilot.invalid/chat/completions")
        sent = self.post.call_args.kwargs["json"]
        self.assertEqual(sent["messages"], [{"role": "system", "content": "Be concise"}, {"role": "user", "content": "Hi"}])
        self.assertEqual(sent["max_tokens"], 30)
        self.assertEqual(self.post.call_args.kwargs["headers"]["X-Initiator"], "user")
        entry = next(iter(self.cache.cache.values()))
        self.assertEqual(entry["endpoint"], "/v1/responses")
        self.assertEqual(entry["output_tokens"], 3)

    def test_model_alias_and_responses_alias(self):
        self.post.return_value = self.completion()
        with mock.patch.object(openai, "translate_model_name", return_value=MODEL):
            response = self.client.post("/responses", json={"model": "my-gemini", "input": "Hi"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.post.call_args.kwargs["json"]["model"], MODEL)
        entry = next(iter(self.cache.cache.values()))
        self.assertEqual(entry["model"], "my-gemini")
        self.assertEqual(entry["translated_model"], MODEL)

    def test_vision_function_history_and_structured_output(self):
        self.post.return_value = self.completion(None, "tool_calls", tool_calls=[{
            "id": "call-next", "type": "function", "function": {"name": "lookup", "arguments": '{"q":"next"}'},
        }])
        response = self.client.post("/v1/responses", json={
            "model": MODEL,
            "input": [
                {"role": "developer", "content": [{"type": "input_text", "text": "Use lookup"}]},
                {"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "low"}]},
                {"type": "reasoning", "encrypted_content": "opaque"},
                {"type": "function_call", "call_id": "call-prev", "name": "lookup", "arguments": '{"q":"prev"}'},
                {"type": "function_call_output", "call_id": "call-prev", "output": "found"},
            ],
            "tools": [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}],
            "tool_choice": {"type": "function", "name": "lookup"},
            "text": {"format": {"type": "json_schema", "name": "result", "schema": {"type": "object"}}},
        })
        self.assertEqual(response.status_code, 200)
        sent = self.post.call_args.kwargs["json"]
        self.assertEqual(sent["messages"][0]["role"], "system")
        self.assertEqual(sent["messages"][1]["content"][0]["image_url"]["detail"], "low")
        self.assertEqual(sent["messages"][2]["tool_calls"][0]["id"], "call-prev")
        self.assertEqual(sent["messages"][3], {"role": "tool", "tool_call_id": "call-prev", "content": "found"})
        self.assertEqual(sent["tools"][0]["function"]["name"], "lookup")
        self.assertEqual(sent["tool_choice"], {"type": "function", "function": {"name": "lookup"}})
        self.assertEqual(sent["response_format"]["json_schema"]["name"], "result")
        self.assertEqual(self.post.call_args.kwargs["headers"]["X-Initiator"], "agent")
        self.headers.assert_called_with(True)
        self.assertEqual(response.get_json()["output"][0]["call_id"], "call-next")

    def test_streaming_text_and_parallel_tools_have_stable_ids_and_usage(self):
        upstream = _UpstreamResponse(chunks=[
            {"choices": [{"delta": {"role": "assistant", "content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 4, "id": "call-a", "function": {"name": "lookup", "arguments": '{"q":'}},
                {"index": 8, "id": "call-b", "function": {"name": "lookup", "arguments": "{}"}},
            ]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 4, "function": {"arguments": '"a"}'}}]}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 5}},
        ])
        self.post.return_value = upstream
        response = self.client.post("/v1/responses", json={"model": MODEL, "input": "Hi", "stream": True})
        events = _events(response)
        self.assertEqual([event["sequence_number"] for event in events], list(range(len(events))))
        self.assertEqual(events[0]["type"], "response.created")
        self.assertEqual(events[0]["response"]["output"], [])
        self.assertEqual(events[-1]["type"], "response.completed")
        result = events[-1]["response"]
        self.assertEqual(result["id"], events[0]["response"]["id"])
        self.assertEqual(result["usage"]["output_tokens"], 5)
        self.assertEqual(result["output"][0]["content"][0]["text"], "Hello")
        self.assertEqual(result["output"][1]["arguments"], '{"q":"a"}')
        self.assertEqual(result["output"][2]["call_id"], "call-b")
        for event in events:
            if "item" in event:
                self.assertEqual(event["item"]["id"], result["output"][event["output_index"]]["id"])
            if "item_id" in event:
                self.assertEqual(event["item_id"], result["output"][event["output_index"]]["id"])
        self.assertTrue(upstream.closed)
        self.assertEqual(self.post.call_args.kwargs["json"]["stream_options"], {"include_usage": True})
        self.assertEqual(next(iter(self.cache.cache.values()))["input_tokens"], 12)

    def test_truncation_returns_incomplete_in_json_and_streaming(self):
        self.post.return_value = self.completion("Partial", "length")
        response = self.client.post("/v1/responses", json={"model": MODEL, "input": "Hi"})
        self.assertEqual(response.get_json()["incomplete_details"], {"reason": "max_output_tokens"})
        self.post.return_value = _UpstreamResponse(chunks=[{"choices": [{"delta": {"content": "Partial"}, "finish_reason": "length"}]}])
        events = _events(self.client.post("/v1/responses", json={"model": MODEL, "input": "Hi", "stream": True}))
        self.assertEqual(events[-1]["type"], "response.incomplete")

    def test_unsupported_requests_fail_before_upstream_or_sse_headers(self):
        for extra, param in [
            ({"previous_response_id": "resp-old"}, "previous_response_id"),
            ({"background": True}, "background"),
            ({"conversation": "conv-old"}, "conversation"),
            ({"tools": [{"type": "custom", "name": "patch"}]}, "tools"),
            ({"tools": [{"type": "web_search"}]}, "tools"),
            ({"tools": [{"type": "image_generation"}]}, "tools"),
            ({"input": [{"role": "user", "content": [{"type": "input_file", "file_id": "file-1"}]}]}, "input"),
        ]:
            with self.subTest(param=param):
                response = self.client.post("/v1/responses", json={"model": MODEL, "input": "Hi", "stream": True, **extra})
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.get_json()["error"]["param"], param)
        self.post.assert_not_called()

    def test_unknown_and_unrelated_chat_models_remain_rejected(self):
        state.models["data"].append({"id": "other-chat", "supported_endpoints": ["/chat/completions"]})
        for model in ("gemini-unknown", "other-chat"):
            response = self.client.post("/v1/responses", json={"model": model, "input": "Hi"})
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.get_json()["error"]["code"], "unsupported_model")
        self.post.assert_not_called()

    def test_native_gemini_responses_is_preferred(self):
        state.models["data"][0]["supported_endpoints"].append("/responses")
        self.post.return_value = _UpstreamResponse({"id": "resp-native", "output": [], "usage": {}})
        response = self.client.post("/v1/responses", json={"model": MODEL, "input": "Hi"})
        self.assertEqual(response.get_json()["id"], "resp-native")
        self.assertEqual(self.post.call_args.args[0], "https://copilot.invalid/v1/responses")
        self.assertEqual(self.post.call_args.kwargs["json"]["input"], "Hi")

    def test_upstream_http_error_is_preserved(self):
        body = {"error": {"code": "quota_exceeded", "message": "No quota"}}
        for streaming in (False, True):
            self.post.return_value = _UpstreamResponse(body, status_code=429)
            response = self.client.post("/v1/responses", json={"model": MODEL, "input": "Hi", "stream": streaming})
            self.assertEqual(response.status_code, 429)
            self.assertEqual(response.get_json(), body)

    def test_early_failure_retry_stays_on_chat_backend(self):
        state.enable_responses_early_failure_retry = True
        state.max_connection_retries = 1
        self.post.side_effect = [
            _UpstreamResponse(chunks=[{"error": {"code": "upstream_error", "message": "Try again"}}]),
            _UpstreamResponse(chunks=[{"choices": [{"delta": {"content": "OK"}, "finish_reason": "stop"}]}]),
        ]
        events = _events(self.client.post("/v1/responses", json={"model": MODEL, "input": "Hi", "stream": True}))
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(sum(event["type"] == "response.created" for event in events), 1)
        self.assertEqual(self.post.call_count, 2)
        self.assertTrue(all(call.args[0].endswith("/chat/completions") for call in self.post.call_args_list))

    def test_stream_failure_after_output_is_not_retried_or_completed(self):
        state.enable_responses_early_failure_retry = True
        state.max_connection_retries = 1
        self.post.return_value = _UpstreamResponse(chunks=[
            {"choices": [{"delta": {"content": "Partial"}}]},
            {"error": {"message": "Failed"}},
        ])
        events = _events(self.client.post("/v1/responses", json={"model": MODEL, "input": "Hi", "stream": True}))
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertEqual(self.post.call_count, 1)
        self.assertEqual(next(iter(self.cache.cache.values()))["status_code"], 502)

    def test_invalid_request_stream_error_is_not_retried(self):
        state.enable_responses_early_failure_retry = True
        state.max_connection_retries = 1
        self.post.return_value = _UpstreamResponse(chunks=[{
            "error": {"type": "invalid_request_error", "message": "Invalid input"},
        }])
        events = _events(self.client.post("/v1/responses", json={"model": MODEL, "input": "Hi", "stream": True}))
        self.assertEqual(events[-1]["response"]["error"]["code"], "invalid_request_error")
        self.assertEqual(self.post.call_count, 1)

    def test_stream_ending_without_finish_reason_is_reported_as_failed(self):
        self.post.return_value = _UpstreamResponse(chunks=[{"choices": [{"delta": {"content": "Partial"}}]}])
        events = _events(self.client.post("/v1/responses", json={"model": MODEL, "input": "Hi", "stream": True}))
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertNotIn("response.completed", [event["type"] for event in events])

    def test_connection_retry_uses_chat_backend(self):
        state.max_connection_retries = 1
        self.post.side_effect = [requests.exceptions.ConnectionError("connection reset"), self.completion()]
        with mock.patch.object(openai.time, "sleep"):
            response = self.client.post("/v1/responses", json={"model": MODEL, "input": "Hi"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.post.call_count, 2)
        self.assertTrue(all(call.args[0].endswith("/chat/completions") for call in self.post.call_args_list))

    def test_pre_header_keepalive_works_with_bridge(self):
        state.sse_keepalive_interval = 0.01
        released = threading.Event()
        upstream = _UpstreamResponse(chunks=[{"choices": [{"delta": {"content": "OK"}, "finish_reason": "stop"}]}])

        def post(*args, **kwargs):
            if not released.wait(5):
                raise requests.exceptions.ReadTimeout()
            return upstream

        self.post.side_effect = post
        response = self.client.post("/v1/responses", json={"model": MODEL, "input": "Hi", "stream": True}, buffered=False)
        try:
            self.assertIn(b": keepalive", next(response.response))
            released.set()
            self.assertIn("response.completed", response.get_data(as_text=True))
        finally:
            released.set()
            response.close()
        self.assertTrue(upstream.closed)


if __name__ == "__main__":
    unittest.main()
