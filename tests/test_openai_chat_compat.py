import unittest
from unittest import mock

from flask import Flask

from ghc_api.api_helpers import supports_chat_completions_api, supports_responses_api
from ghc_api.routes import openai as openai_routes
from ghc_api.state import state


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = "" if payload is None else __import__("json").dumps(payload)

    def json(self):
        return self._payload


class FakeStreamResponse(FakeResponse):
    def __init__(self, lines, status_code=200):
        super().__init__(None, status_code)
        self._lines = lines

    def iter_lines(self):
        for line in self._lines:
            yield line


class OpenAIChatResponsesCompatTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.register_blueprint(openai_routes.openai_bp)
        self.client = self.app.test_client()
        self.original_models = state.models
        self.original_flag = state.enable_gpt_chat_completions_responses_compat
        self.original_token = state.copilot_token
        self.original_expires = state.token_expires_at
        state.models = {
            "data": [
                {"id": "gpt-5.5", "supported_endpoints": ["/responses", "ws:/responses"]},
                {"id": "gpt-5.4", "supported_endpoints": ["/chat/completions", "/responses"]},
                {"id": "claude-sonnet-4.5", "supported_endpoints": ["/v1/messages"]},
            ]
        }
        state.copilot_token = "copilot-token"
        state.token_expires_at = 9999999999

    def tearDown(self):
        state.models = self.original_models
        state.enable_gpt_chat_completions_responses_compat = self.original_flag
        state.copilot_token = self.original_token
        state.token_expires_at = self.original_expires

    def test_endpoint_capability_helpers_normalize_common_http_paths(self):
        self.assertTrue(supports_responses_api("gpt-5.5"))
        self.assertFalse(supports_chat_completions_api("gpt-5.5"))
        self.assertTrue(supports_responses_api("gpt-5.4"))
        self.assertTrue(supports_chat_completions_api("gpt-5.4"))

    def test_chat_tools_are_converted_to_responses_tool_shape(self):
        payload = openai_routes.chat_payload_to_responses_payload({
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hello"}],
            "tool_choice": {"type": "function", "function": {"name": "calculator"}},
            "tools": [{
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate arithmetic",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"},
                        },
                        "required": ["expression"],
                    },
                },
            }],
        })

        self.assertEqual(payload["tools"], [{
            "type": "function",
            "name": "calculator",
            "description": "Evaluate arithmetic",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            },
        }])
        self.assertEqual(payload["tool_choice"], {"type": "function", "name": "calculator"})

    @mock.patch.object(openai_routes.requests, "post")
    def test_default_disabled_keeps_chat_completions_upstream(self, post):
        state.enable_gpt_chat_completions_responses_compat = False
        post.return_value = FakeResponse({
            "id": "chatcmpl-upstream",
            "model": "gpt-5.5",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })

        response = self.client.post("/v1/chat/completions", json={
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hello"}],
        })

        self.assertEqual(response.status_code, 200)
        self.assertTrue(post.call_args.args[0].endswith("/chat/completions"))

    @mock.patch.object(openai_routes.requests, "post")
    def test_enabled_gpt_responses_only_model_uses_responses_compat(self, post):
        state.enable_gpt_chat_completions_responses_compat = True
        post.return_value = FakeResponse({
            "id": "resp-1",
            "created_at": 1711111111,
            "model": "gpt-5.5",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "converted"}],
            }],
            "usage": {
                "input_tokens": 4,
                "output_tokens": 2,
                "total_tokens": 6,
                "input_tokens_details": {"cached_tokens": 1},
            },
        })

        response = self.client.post("/v1/chat/completions", json={
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 123,
        })

        self.assertEqual(response.status_code, 200)
        self.assertTrue(post.call_args.args[0].endswith("/v1/responses"))
        upstream_payload = post.call_args.kwargs["json"]
        self.assertEqual(upstream_payload["model"], "gpt-5.5")
        self.assertEqual(upstream_payload["max_output_tokens"], 123)
        self.assertEqual(upstream_payload["input"], [{"role": "user", "content": "hello"}])

        data = response.get_json()
        self.assertEqual(data["object"], "chat.completion")
        self.assertEqual(data["choices"][0]["message"]["content"], "converted")
        self.assertEqual(data["usage"]["prompt_tokens"], 4)
        self.assertEqual(data["usage"]["completion_tokens"], 2)
        self.assertEqual(data["usage"]["prompt_tokens_details"]["cached_tokens"], 1)

    @mock.patch.object(openai_routes.requests, "post")
    def test_enabled_does_not_shim_non_gpt_models(self, post):
        state.enable_gpt_chat_completions_responses_compat = True
        post.return_value = FakeResponse({
            "id": "chatcmpl-upstream",
            "model": "claude-sonnet-4.5",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })

        response = self.client.post("/v1/chat/completions", json={
            "model": "claude-sonnet-4.5",
            "messages": [{"role": "user", "content": "hello"}],
        })

        self.assertEqual(response.status_code, 200)
        self.assertTrue(post.call_args.args[0].endswith("/chat/completions"))

    @mock.patch.object(openai_routes.requests, "post")
    def test_streaming_gpt_responses_only_model_uses_responses_compat(self, post):
        state.enable_gpt_chat_completions_responses_compat = True
        post.return_value = FakeStreamResponse([
            b'event: response.created',
            b'data: {"type":"response.created","response":{"id":"resp-stream","created_at":1711111111,"model":"gpt-5.5"}}',
            b'event: response.output_text.delta',
            b'data: {"type":"response.output_text.delta","delta":"hello"}',
            b'event: response.completed',
            b'data: {"type":"response.completed","response":{"id":"resp-stream","created_at":1711111111,"model":"gpt-5.5","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}',
            b'data: [DONE]',
        ])

        response = self.client.post("/v1/chat/completions", json={
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        })

        self.assertEqual(response.status_code, 200)
        self.assertTrue(post.call_args.args[0].endswith("/v1/responses"))
        self.assertTrue(post.call_args.kwargs["json"]["stream"])
        body = response.get_data(as_text=True)
        self.assertIn('"object": "chat.completion.chunk"', body)
        self.assertIn('"content": "hello"', body)
        self.assertIn("data: [DONE]", body)


if __name__ == "__main__":
    unittest.main()
