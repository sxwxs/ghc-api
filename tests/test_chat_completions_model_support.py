import json
import unittest
from unittest import mock

import ghc_api.state
from ghc_api.api_helpers import (
    _configured_chat_completions_support_by_models_id,
    apply_configured_chat_completions_support,
)
from ghc_api.app import create_app
from ghc_api.cache import cache
from ghc_api.config import chat_completions_model_support
from ghc_api.routes.openai import (
    _chat_completions_to_responses_payload,
    _responses_to_chat_completion,
    _should_route_chat_completions_via_responses,
    chat_completions_via_responses,
    stream_chat_completions,
    stream_chat_completions_via_responses,
)


class ChatCompletionsModelSupportTest(unittest.TestCase):
    def setUp(self):
        self._saved_exact = list(chat_completions_model_support.exact_model_names)
        self._saved_prefix = list(chat_completions_model_support.prefix_model_names)
        self._saved_models = ghc_api.state.state.models

    def tearDown(self):
        chat_completions_model_support.exact_model_names = self._saved_exact
        chat_completions_model_support.prefix_model_names = self._saved_prefix
        ghc_api.state.state.models = self._saved_models
        cache.cache.clear()
        _configured_chat_completions_support_by_models_id.clear()

    def test_adds_chat_completions_endpoints_for_exact_and_prefix_matches(self):
        chat_completions_model_support.load_from_config({
            "chat_completions_model_support": {
                "exact": ["claude-custom"],
                "prefix": ["gpt-", "mai-code-"],
            }
        })
        models = {"data": [
            {"id": "gpt-5.4", "supported_endpoints": ["/responses"]},
            {"id": "mai-code-1-flash-internal", "supported_endpoints": []},
            {"id": "claude-custom"},
            {"id": "gemini-3.1-pro", "supported_endpoints": ["/responses"]},
        ]}

        updated_count = apply_configured_chat_completions_support(models)

        self.assertEqual(updated_count, 3)
        self.assertEqual(
            models["data"][0]["supported_endpoints"],
            ["/responses", "/v1/chat/completions"],
        )
        self.assertEqual(
            models["data"][1]["supported_endpoints"],
            ["/v1/chat/completions"],
        )
        self.assertEqual(
            models["data"][2]["supported_endpoints"],
            ["/v1/chat/completions"],
        )
        self.assertEqual(models["data"][3]["supported_endpoints"], ["/responses"])

    def test_does_not_duplicate_existing_endpoints(self):
        chat_completions_model_support.load_from_config({
            "chat_completions_model_support": {
                "exact": ["gpt-5.4"],
                "prefix": [],
            }
        })
        models = {"data": [
            {
                "id": "gpt-5.4",
                "supported_endpoints": ["/v1/chat/completions"],
            },
        ]}

        updated_count = apply_configured_chat_completions_support(models)

        self.assertEqual(updated_count, 0)
        self.assertEqual(
            models["data"][0]["supported_endpoints"],
            ["/v1/chat/completions"],
        )

    def test_runtime_config_accepts_chat_completions_model_support(self):
        ghc_api.state.state.models = {"data": [
            {"id": "mai-code-1-flash-internal", "supported_endpoints": ["/responses"]},
        ]}
        app = create_app()

        with app.test_client() as client:
            response = client.post("/api/runtime-config", json={
                "chat_completions_model_support": {
                    "exact": [],
                    "prefix": ["mai-code-"],
                }
            })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(chat_completions_model_support.prefix_model_names, ["mai-code-"])
        self.assertEqual(
            ghc_api.state.state.models["data"][0]["supported_endpoints"],
            ["/responses", "/v1/chat/completions"],
        )

    def test_runtime_config_removes_previous_configured_support(self):
        ghc_api.state.state.models = {"data": [
            {"id": "mai-code-1-flash-internal", "supported_endpoints": ["/responses"]},
        ]}
        app = create_app()

        with app.test_client() as client:
            response = client.post("/api/runtime-config", json={
                "chat_completions_model_support": {
                    "exact": [],
                    "prefix": ["mai-code-"],
                }
            })
            self.assertEqual(response.status_code, 200)
            response = client.post("/api/runtime-config", json={
                "chat_completions_model_support": {
                    "exact": [],
                    "prefix": [],
                }
            })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            ghc_api.state.state.models["data"][0]["supported_endpoints"],
            ["/responses"],
        )

    def test_runtime_config_rejects_invalid_support_config(self):
        app = create_app()

        with app.test_client() as client:
            response = client.post("/api/runtime-config", json={
                "chat_completions_model_support": {
                    "exact": "gpt-",
                    "prefix": [],
                }
            })

        self.assertEqual(response.status_code, 400)
        self.assertIn("chat_completions_model_support.exact", response.get_json()["error"])

    def test_routes_configured_responses_only_model_through_compat_layer(self):
        chat_completions_model_support.load_from_config({
            "chat_completions_model_support": {
                "exact": [],
                "prefix": ["mai-code-"],
            }
        })
        ghc_api.state.state.models = {"data": [
            {"id": "mai-code-1-flash-internal", "supported_endpoints": ["/responses"]},
            {"id": "gpt-4o", "supported_endpoints": ["/chat/completions", "/v1/chat/completions"]},
        ]}
        apply_configured_chat_completions_support(ghc_api.state.state.models)

        self.assertTrue(_should_route_chat_completions_via_responses("mai-code-1-flash-internal"))
        self.assertFalse(_should_route_chat_completions_via_responses("gpt-4o"))

    def test_chat_completion_payload_translates_to_responses_payload(self):
        payload = {
            "model": "mai-code-1-flash-internal",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "who are you"},
                {"role": "assistant", "content": "An assistant."},
                {"role": "user", "content": "again"},
            ],
            "stream": True,
            "temperature": 0.2,
            "max_tokens": 128,
            "tool_choice": {"type": "function", "function": {"name": "bash"}},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "Run a command",
                        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                    },
                },
            ],
        }

        result = _chat_completions_to_responses_payload(payload)

        self.assertEqual(result["model"], "mai-code-1-flash-internal")
        self.assertEqual(result["instructions"], "Be concise.")
        self.assertTrue(result["stream"])
        self.assertEqual(result["temperature"], 0.2)
        self.assertEqual(result["max_output_tokens"], 128)
        self.assertEqual(result["tool_choice"], {"type": "function", "name": "bash"})
        self.assertEqual(result["tools"], [
            {
                "type": "function",
                "name": "bash",
                "description": "Run a command",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
            },
        ])
        self.assertEqual(result["input"], [
            {"role": "user", "content": "who are you"},
            {"role": "assistant", "content": "An assistant."},
            {"role": "user", "content": "again"},
        ])

    def test_chat_completion_payload_maps_max_completion_tokens(self):
        payload = {
            "model": "mai-code-1-flash-internal",
            "messages": [{"role": "user", "content": "hello"}],
            "max_completion_tokens": 64,
        }

        result = _chat_completions_to_responses_payload(payload)

        self.assertEqual(result["max_output_tokens"], 64)

    def test_chat_completion_payload_prefers_max_completion_tokens_over_max_tokens(self):
        payload = {
            "model": "mai-code-1-flash-internal",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 128,
            "max_completion_tokens": 64,
        }

        result = _chat_completions_to_responses_payload(payload)

        self.assertEqual(result["max_output_tokens"], 64)

    def test_chat_completion_payload_preserves_tool_turns(self):
        payload = {
            "model": "mai-code-1-flash-internal",
            "messages": [
                {"role": "user", "content": "Use a tool."},
                {
                    "role": "assistant",
                    "content": "Checking.",
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": "{\"query\":\"weather\"}",
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": "{\"result\":\"sunny\"}",
                },
            ],
        }

        result = _chat_completions_to_responses_payload(payload)

        self.assertEqual(result["input"], [
            {"role": "user", "content": "Use a tool."},
            {"role": "assistant", "content": "Checking."},
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "lookup",
                "arguments": "{\"query\":\"weather\"}",
            },
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "{\"result\":\"sunny\"}",
            },
        ])

    def test_chat_completion_payload_preserves_image_blocks(self):
        payload = {
            "model": "mai-code-1-flash-internal",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,abc",
                            "detail": "high",
                        },
                    },
                ],
            }],
        }

        result = _chat_completions_to_responses_payload(payload)

        self.assertEqual(result["input"], [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe this."},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,abc",
                    "detail": "high",
                },
            ],
        }])

    def test_responses_body_translates_to_chat_completion(self):
        responses_body = {
            "id": "resp-1",
            "output": [
                {"type": "message", "content": [
                    {"type": "output_text", "text": "Hello"},
                    {"type": "output_text", "text": " there"},
                ]},
            ],
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        }

        result = _responses_to_chat_completion(responses_body, "mai-code-1-flash-internal")

        self.assertEqual(result["id"], "resp-1")
        self.assertEqual(result["object"], "chat.completion")
        self.assertEqual(result["model"], "mai-code-1-flash-internal")
        self.assertEqual(result["choices"][0]["message"]["content"], "Hello there")
        self.assertEqual(result["usage"], {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        })

    def test_responses_function_call_translates_to_chat_tool_call(self):
        responses_body = {
            "id": "resp-1",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "skill",
                    "arguments": "{\"name\":\"customize-opencode\"}",
                },
            ],
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        }

        result = _responses_to_chat_completion(responses_body, "mai-code-1-flash-internal")

        self.assertIsNone(result["choices"][0]["message"]["content"])
        self.assertEqual(result["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(result["choices"][0]["message"]["tool_calls"], [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "skill",
                    "arguments": "{\"name\":\"customize-opencode\"}",
                },
            },
        ])

    def test_streaming_upstream_error_returns_error_status(self):
        class FakeResponse:
            ok = False
            status_code = 400
            text = "{\"error\":{\"message\":\"bad request\"}}"

            def json(self):
                return {"error": {"message": "bad request"}}

        app = create_app()
        payload = {
            "model": "mai-code-1-flash-internal",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        }

        with app.test_request_context("/v1/chat/completions"):
            with mock.patch("ghc_api.routes.openai.requests.post", return_value=FakeResponse()):
                response = chat_completions_via_responses(
                    payload=payload,
                    headers={},
                    request_id="req-1",
                    request_body="{}",
                    request_size=2,
                    start_time=0,
                    original_model="mai-code-1-flash-internal",
                    translated_model="mai-code-1-flash-internal",
                )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.mimetype, "application/json")
        self.assertIn("bad request", response.get_data(as_text=True))

    def test_streaming_response_failed_emits_error_without_done(self):
        class FakeResponse:
            ok = True
            status_code = 200

            def iter_lines(self):
                yield b'data: {"type":"response.failed","response":{"error":{"message":"model failed"}}}'

            def close(self):
                pass

        app = create_app()

        with app.test_request_context("/v1/chat/completions"):
            response = stream_chat_completions_via_responses(
                response=FakeResponse(),
                request_id="req-1",
                request_body="{}",
                request_size=2,
                start_time=0,
                original_model="mai-code-1-flash-internal",
                translated_model="mai-code-1-flash-internal",
                chat_payload={"model": "mai-code-1-flash-internal"},
                responses_payload={"model": "mai-code-1-flash-internal"},
            )
            body = "".join(response.response)

        self.assertIn('"error": {"message": "model failed"}', body)
        self.assertNotIn("data: [DONE]", body)
        self.assertEqual(cache.cache["req-1"]["state"], cache.STATE_ERROR)
        self.assertEqual(cache.cache["req-1"]["status_code"], 500)

    def test_streaming_responses_tool_call_matches_chat_chunk_shape(self):
        class FakeResponse:
            ok = True
            status_code = 200

            def iter_lines(self):
                yield b'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call-1","name":"get_weather"}}'
                yield b'data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\\""}'
                yield b'data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"location"}'
                yield b'data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"\\":\\"SF\\"}"}'
                yield b'data: {"type":"response.completed","response":{"id":"resp-1","output":[{"type":"function_call","call_id":"call-1","name":"get_weather","arguments":"{\\"location\\":\\"SF\\"}"}],"usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}'

            def close(self):
                pass

        app = create_app()

        with app.test_request_context("/v1/chat/completions"):
            response = stream_chat_completions_via_responses(
                response=FakeResponse(),
                request_id="req-1",
                request_body="{}",
                request_size=2,
                start_time=0,
                original_model="gpt-5.5",
                translated_model="gpt-5.5",
                chat_payload={"model": "gpt-5.5"},
                responses_payload={"model": "gpt-5.5"},
            )
            body = "".join(response.response)

        data_lines = [
            line.removeprefix("data: ")
            for line in body.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        chunks = [json.loads(line) for line in data_lines]

        first_delta = chunks[0]["choices"][0]["delta"]
        self.assertEqual(first_delta["role"], "assistant")
        self.assertIsNone(first_delta["content"])
        self.assertEqual(first_delta["tool_calls"][0]["id"], "call-1")
        self.assertEqual(first_delta["tool_calls"][0]["function"]["name"], "get_weather")

        argument_deltas = [
            chunk["choices"][0]["delta"]
            for chunk in chunks[1:-1]
        ]
        self.assertTrue(argument_deltas)
        self.assertTrue(all(delta.get("content") is None for delta in argument_deltas))

        finish_choice = chunks[-1]["choices"][0]
        self.assertIsNone(finish_choice["delta"]["content"])
        self.assertEqual(finish_choice["finish_reason"], "tool_calls")
        self.assertIn("data: [DONE]", body)
        self.assertEqual(cache.cache["req-1"]["state"], cache.STATE_COMPLETED)

    def test_streaming_responses_text_chunks_include_native_like_null_content_boundaries(self):
        class FakeResponse:
            ok = True
            status_code = 200

            def iter_lines(self):
                yield b'data: {"type":"response.output_text.delta","delta":"Hello"}'
                yield b'data: {"type":"response.completed","response":{"id":"resp-1","output":[{"type":"message","content":[{"type":"output_text","text":"Hello"}]}],"usage":{"input_tokens":5,"output_tokens":1,"total_tokens":6}}}'

            def close(self):
                pass

        app = create_app()

        with app.test_request_context("/v1/chat/completions"):
            response = stream_chat_completions_via_responses(
                response=FakeResponse(),
                request_id="req-1",
                request_body="{}",
                request_size=2,
                start_time=0,
                original_model="gpt-5.5",
                translated_model="gpt-5.5",
                chat_payload={"model": "gpt-5.5"},
                responses_payload={"model": "gpt-5.5"},
            )
            body = "".join(response.response)

        data_lines = [
            line.removeprefix("data: ")
            for line in body.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        chunks = [json.loads(line) for line in data_lines]

        self.assertEqual(chunks[0]["choices"][0]["delta"], {"content": None, "role": "assistant"})
        self.assertEqual(chunks[1]["choices"][0]["delta"], {"content": "Hello"})
        self.assertEqual(chunks[-1]["choices"][0]["delta"], {"content": None})
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        self.assertIn("data: [DONE]", body)

    def test_native_streaming_tool_call_indexes_are_compacted(self):
        class FakeResponse:
            status_code = 200

            def iter_lines(self):
                yield b'data: {"choices":[{"index":0,"delta":{"content":"Checking","role":"assistant"}}],"created":1,"id":"chat-1","model":"claude-sonnet-4.6"}'
                yield b'data: {"choices":[{"index":0,"delta":{"content":null,"tool_calls":[{"function":{"name":"get_weather"},"id":"tooluse_1","index":2,"type":"function"}]}}],"created":1,"id":"chat-1","model":"claude-sonnet-4.6"}'
                yield b'data: {"choices":[{"index":0,"delta":{"content":null,"tool_calls":[{"function":{"arguments":"{\\"location\\":\\"SF\\"}"},"index":2,"type":"function"}]}}],"created":1,"id":"chat-1","model":"claude-sonnet-4.6"}'
                yield b'data: {"choices":[{"finish_reason":"tool_calls","index":0,"delta":{"content":null}}],"created":1,"id":"chat-1","usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8},"model":"claude-sonnet-4.6"}'
                yield b"data: [DONE]"

            def close(self):
                pass

        app = create_app()

        with app.test_request_context("/v1/chat/completions"):
            with mock.patch("ghc_api.routes.openai.requests.post", return_value=FakeResponse()):
                response = stream_chat_completions(
                    payload={"model": "claude-sonnet-4.6", "stream": True},
                    headers={},
                    request_id="req-1",
                    request_body="{}",
                    request_size=2,
                    start_time=0,
                    original_model="claude-sonnet-4.6",
                    translated_model="claude-sonnet-4.6",
                )
                body = "".join(response.response)

        data_lines = [
            line.removeprefix("data: ")
            for line in body.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        chunks = [json.loads(line) for line in data_lines]

        tool_start = chunks[1]["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(tool_start["index"], 0)
        self.assertEqual(tool_start["function"]["arguments"], "")

        tool_args = chunks[2]["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(tool_args["index"], 0)

        response_body = cache.cache["req-1"]["response_body"]
        self.assertEqual(response_body["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(response_body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"], "{\"location\":\"SF\"}")


if __name__ == "__main__":
    unittest.main()
