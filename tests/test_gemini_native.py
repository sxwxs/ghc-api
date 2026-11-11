import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from unittest import mock

from flask import Flask, Response, g

from ghc_api.app import create_app
from ghc_api.gemini_native import from_chat_stream, to_chat
from ghc_api.routes import gemini, openai
from ghc_api.state import state


MODEL = "gemini-4-argon"
BODY = {"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]}


class GeminiNativeTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.stack.enter_context(mock.patch.dict(os.environ, {"GHC_API_CONFIG_DIR": directory}))
        self.stack.enter_context(mock.patch.object(state, "enable_auth", False))
        self.handler = self.stack.enter_context(mock.patch.object(openai, "handle_chat_completions"))
        self.client = create_app().test_client()

    def completion(self, content="Hello", **message):
        return Response(json.dumps({
            "choices": [{"message": {"content": content, **message}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }), mimetype="application/json")

    def test_native_and_cloud_share_transport_and_preserve_usage(self):
        for path, body in [
            (f"/v1beta/models/{MODEL}:generateContent", BODY),
            ("/v1internal:generateContent", {"model": "MODEL_PLACEHOLDER_M460", "request": BODY}),
            ("/v1internal:generateContent", {"model": "MODEL_PLACEHOLDER_M461", "request": BODY}),
        ]:
            with self.subTest(path=path, body=body):
                self.handler.return_value = self.completion()
                response = self.client.post(path, json=body)
                self.assertEqual(response.status_code, 200)
                result = response.get_json()
                result = result.get("response", result)
                self.assertEqual(result["candidates"][0]["content"]["parts"], [{"text": "Hello"}])
                self.assertEqual(result["usageMetadata"]["totalTokenCount"], 3)
                payload = self.handler.call_args.args[0]
                expected = "gemini-3.8-flash" if body.get("model") == "MODEL_PLACEHOLDER_M461" else MODEL
                self.assertEqual(payload["model"], expected)

    def test_tools_and_results_roundtrip_without_native_call_ids(self):
        body = {"contents": [
            {"role": "user", "parts": [{"text": "Add"}]},
            {"role": "model", "parts": [{"functionCall": {"name": "add", "args": {"a": 2}}}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "add", "response": {"sum": 2}}}]},
        ], "tools": [{"functionDeclarations": [{
            "name": "add", "parameters": {"type": "OBJECT", "properties": {"a": {"type": "INTEGER"}}},
        }]}], "generationConfig": {"thinkingConfig": {"thinkingLevel": "HIGH"}}}
        payload = to_chat(body, MODEL, False)
        self.assertEqual(payload["messages"][1]["tool_calls"][0]["id"], payload["messages"][2]["tool_call_id"])
        self.assertEqual(payload["tools"][0]["function"]["parameters"]["properties"]["a"]["type"], "integer")
        self.assertEqual(payload["reasoning_effort"], "high")
        self.handler.return_value = self.completion(None, tool_calls=[{
            "id": "call_real", "function": {"name": "add", "arguments": '{"a": 2}'},
        }])
        response = self.client.post(f"/v1beta/models/{MODEL}:generateContent", json=BODY)
        call = response.get_json()["candidates"][0]["content"]["parts"][0]["functionCall"]
        self.assertEqual(call, {"id": "call_real", "name": "add", "args": {"a": 2}})

    def test_stream_fragmented_tool_arguments_unicode_and_crlf(self):
        events = [
            {"choices": [{"delta": {"content": "你好"}, "finish_reason": None}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "a", "function": {
                "name": "add", "arguments": '{"a":',
            }}]}, "finish_reason": None}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {
                "arguments": "2}",
            }}]}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}},
        ]
        encoded = "".join("data: " + json.dumps(event, ensure_ascii=False) + "\r\n\r\n" for event in events)
        encoded += "data: [DONE]\r\n\r\n"
        data = encoded.encode()
        chunks = [data[index:index + 1] for index in range(len(data))]
        self.handler.return_value = Response(iter(chunks), mimetype="text/event-stream")
        response = self.client.post("/v1internal:streamGenerateContent", json={"model": MODEL, "request": BODY})
        converted = [json.loads(line[6:])["response"] for line in response.get_data(as_text=True).splitlines()
                     if line.startswith("data: ")]
        self.assertEqual(converted[0]["candidates"][0]["content"]["parts"][0]["text"], "你好")
        call = converted[1]["candidates"][0]["content"]["parts"][0]["functionCall"]
        self.assertEqual(call["args"], {"a": 2})
        self.assertEqual(converted[-1]["usageMetadata"]["totalTokenCount"], 5)

    def test_cli_include_thoughts_controls_json_and_stream_output(self):
        for include in (True, False):
            for streaming in (False, True):
                with self.subTest(include=include, streaming=streaming):
                    body = {**BODY, "generationConfig": {"thinkingConfig": {
                        "thinkingLevel": "HIGH", "includeThoughts": include,
                    }}}
                    if streaming:
                        chunk = {"choices": [{"delta": {"reasoning_content": "Synthetic reasoning",
                                                       "content": "Answer"}, "finish_reason": "stop"}]}
                        self.handler.return_value = Response(
                            "data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
                            mimetype="text/event-stream",
                        )
                    else:
                        self.handler.return_value = self.completion("Answer", reasoning_content="Synthetic reasoning")
                    action = "streamGenerateContent" if streaming else "generateContent"
                    response = self.client.post(f"/v1beta/models/{MODEL}:{action}", json=body)
                    self.assertEqual(response.status_code, 200, response.data)
                    result = (json.loads(response.get_data(as_text=True).splitlines()[0][6:])
                              if streaming else response.get_json())
                    parts = result["candidates"][0]["content"]["parts"]
                    expected = ([{"text": "Synthetic reasoning", "thought": True}] if include else [])
                    self.assertEqual(parts, expected + [{"text": "Answer"}])
                    self.assertEqual(self.handler.call_args.args[0]["reasoning_effort"], "high")
        self.assertNotIn("reasoning_effort", to_chat(
            {**BODY, "generationConfig": {"thinkingConfig": {"includeThoughts": True}}}, MODEL, False,
        ))
        with self.assertRaisesRegex(ValueError, "boolean"):
            to_chat({**BODY, "generationConfig": {"thinkingConfig": {"includeThoughts": "true"}}}, MODEL, False)

    def test_invalid_or_unsupported_requests_never_call_upstream(self):
        invalid = [
            [], {}, {"contents": "bad"},
            {**BODY, "safetySettings": []},
            {**BODY, "tools": [{"googleSearch": {}}]},
            {**BODY, "generationConfig": {"thinkingConfig": {"thinkingBudget": 500}}},
            {"contents": [{"role": "user", "parts": [{"fileData": {"fileUri": "file:///secret"}}]}]},
            {"contents": [{"role": "model", "parts": [{"text": "x", "thoughtSignature": "opaque"}]}]},
            {"contents": [{"role": "user", "parts": [{"functionResponse": {"name": "missing", "response": {}}}]}]},
        ]
        for body in invalid:
            with self.subTest(body=body):
                response = self.client.post(f"/v1beta/models/{MODEL}:generateContent", json=body)
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.get_json())
        self.handler.assert_not_called()

    def test_flash_lite_substitution_is_opt_in_and_scoped_to_helper(self):
        helper = "gemini-3.1-flash-lite-preview"
        body = {**BODY, "generationConfig": {"thinkingConfig": {"thinkingLevel": "MINIMAL"}}}
        with mock.patch.dict(os.environ, {"GHC_API_GEMINI_FLASH_LITE_FALLBACK": "0"}):
            response = self.client.post(f"/v1beta/models/{helper}:generateContent", json=body)
        self.assertEqual(response.status_code, 400)
        self.handler.assert_not_called()
        with mock.patch.dict(os.environ, {"GHC_API_GEMINI_FLASH_LITE_FALLBACK": "1"}):
            for model in (MODEL, "gemini-3.8-flash"):
                with self.subTest(model=model):
                    self.handler.return_value = self.completion()
                    high = {**BODY, "generationConfig": {"thinkingConfig": {"thinkingLevel": "HIGH"}}}
                    response = self.client.post(f"/v1beta/models/{model}:generateContent", json=high)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(self.handler.call_args.args[0]["model"], model)
                    self.assertEqual(self.handler.call_args.args[0]["reasoning_effort"], "high")
                    response = self.client.post(f"/v1beta/models/{model}:generateContent", json=body)
                    self.assertEqual(response.status_code, 400)

    def test_flash_lite_substitution_preserves_instructions_and_response_formats(self):
        helper = "gemini-3.1-flash-lite-preview"
        target = "gemini-3.8-flash"
        catalog = {"data": [{
            "id": target, "supported_endpoints": ["/chat/completions"],
            "capabilities": {"supports": {"reasoning_effort": ["low", "medium", "high"]}},
        }]}
        body = {**BODY, "systemInstruction": {"parts": [{"text": "Synthetic safety rubric"}]},
                "generationConfig": {"thinkingConfig": {"thinkingLevel": "MINIMAL", "includeThoughts": True}}}
        original = json.loads(json.dumps(body))
        with mock.patch.dict(os.environ, {"GHC_API_GEMINI_FLASH_LITE_FALLBACK": "1"}), mock.patch.object(
            state, "models", catalog,
        ):
            for cloud in (False, True):
                for streaming in (False, True):
                    with self.subTest(cloud=cloud, streaming=streaming):
                        action = "streamGenerateContent" if streaming else "generateContent"
                        if streaming:
                            chunk = {"choices": [{"delta": {"content": "Checked"}, "finish_reason": "stop"}]}
                            self.handler.return_value = Response(
                                "data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
                                mimetype="text/event-stream",
                            )
                        else:
                            self.handler.return_value = self.completion("Checked")
                        path = f"/v1internal:{action}" if cloud else f"/v1beta/models/{helper}:{action}"
                        data = {"model": "models/" + helper, "request": body} if cloud else body
                        with self.assertLogs("ghc_api.app", level="WARNING") as logs:
                            response = self.client.post(path, json=data)
                        self.assertEqual(response.status_code, 200, response.data)
                        self.assertIn("not verified safety-equivalent", " ".join(logs.output))
                        payload = self.handler.call_args.args[0]
                        self.assertEqual(payload["model"], target)
                        self.assertEqual(payload["reasoning_effort"], "low")
                        self.assertEqual(payload["messages"], [
                            {"role": "system", "content": "Synthetic safety rubric"},
                            {"role": "user", "content": "Hi"},
                        ])
                        result = (json.loads(response.get_data(as_text=True).splitlines()[0][6:])
                                  if streaming else response.get_json())
                        result = result["response"] if cloud else result
                        self.assertEqual(result["candidates"][0]["content"]["parts"], [{"text": "Checked"}])
            self.handler.reset_mock()
            for invalid in (
                {**body, "safetySettings": []},
                {**body, "generationConfig": {"thinkingConfig": {"thinkingLevel": "UNKNOWN"}}},
            ):
                response = self.client.post(f"/v1beta/models/{helper}:generateContent", json=invalid)
                self.assertEqual(response.status_code, 400)
            self.handler.assert_not_called()
            self.handler.return_value = Response('{"error":{"message":"quota"}}', status=429,
                                                mimetype="application/json")
            response = self.client.post(f"/v1beta/models/{helper}:generateContent", json=body)
            self.assertEqual(response.status_code, 429)
        self.assertEqual(body, original)

    def test_flash_lite_substitution_requires_available_capable_target(self):
        helper = "gemini-3.1-flash-lite-preview"
        target = {"id": "gemini-3.8-flash", "supported_endpoints": ["/chat/completions"],
                  "capabilities": {"supports": {"reasoning_effort": ["low"]}}}
        catalogs = [{"data": [{"id": "unrelated"}]},
                    {"data": [{**target, "supported_endpoints": ["/responses"]}]},
                    {"data": [{**target, "capabilities": {}}]}]
        with mock.patch.dict(os.environ, {"GHC_API_GEMINI_FLASH_LITE_FALLBACK": "1"}):
            for catalog in catalogs:
                with self.subTest(catalog=catalog), mock.patch.object(state, "models", catalog):
                    response = self.client.post(f"/v1beta/models/{helper}:generateContent", json=BODY)
                    self.assertEqual(response.status_code, 503)
                    self.assertIn("lacks Chat Completions/LOW support", response.get_json()["error"]["message"])
        self.handler.assert_not_called()

    def test_errors_and_incomplete_stream_are_not_success_responses(self):
        self.handler.return_value = Response('{"error":{"message":"quota"}}', status=429, mimetype="application/json")
        response = self.client.post(f"/v1beta/models/{MODEL}:generateContent", json=BODY)
        self.assertEqual(response.status_code, 429)
        self.handler.return_value = Response(iter(['data: {"choices":']), mimetype="text/event-stream")
        response = self.client.post(f"/v1beta/models/{MODEL}:streamGenerateContent", json=BODY)
        self.assertIn('"code": 502', response.get_data(as_text=True))
        with self.assertRaises(ValueError):
            list(from_chat_stream([b'data: {"choices":']))
        with self.assertRaisesRegex(ValueError, "finish reason"):
            list(from_chat_stream([b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\ndata: [DONE]\n\n']))

    def test_auth_gate_covers_native_catalog_and_cloud_routes(self):
        with mock.patch.object(state, "enable_auth", True):
            for path, method in [
                ("/v1beta/models", "get"),
                (f"/v1beta/models/{MODEL}:generateContent", "post"),
                ("/v1internal:fetchAvailableModels", "post"),
            ]:
                response = getattr(self.client, method)(path, json=BODY)
                self.assertEqual(response.status_code, 401)
        self.handler.assert_not_called()

    def test_safe_stream_retry_delivers_tool_call_only_once(self):
        complete = 'data: ' + json.dumps({"choices": [{
            "delta": {"tool_calls": [{"index": 0, "id": "safe-call", "function": {
                "name": "lookup", "arguments": '{"key": 2}',
            }}]}, "finish_reason": "tool_calls",
        }]}) + '\n\ndata: [DONE]\n\n'
        for first in ('data: [DONE]\n\n', 'data: {"error":{"code":503}}\n\n'):
            with self.subTest(first=first), mock.patch.dict(
                os.environ, {"GHC_API_GEMINI_SAFE_STREAM_RETRY": "1"},
            ), mock.patch.object(gemini.time, "sleep") as sleep:
                self.handler.reset_mock()
                self.handler.side_effect = [
                    Response(first, mimetype="text/event-stream"),
                    Response(complete, mimetype="text/event-stream"),
                ]
                response = self.client.post(f"/v1beta/models/{MODEL}:streamGenerateContent", json=BODY)
                events = [json.loads(line[6:]) for line in response.get_data(as_text=True).splitlines()
                          if line.startswith("data: ")]
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0]["candidates"][0]["content"]["parts"], [{
                    "functionCall": {"id": "safe-call", "name": "lookup", "args": {"key": 2}},
                }])
                self.assertEqual(self.handler.call_count, 2)
                self.assertEqual(self.handler.call_args_list[0], self.handler.call_args_list[1])
                sleep.assert_called_once_with(1)

    def test_partial_or_malformed_stream_is_never_retried(self):
        partial = 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
        for first in (partial + 'data: [DONE]\n\n',
                      partial + 'data: {"error":{"code":503}}\n\n',
                      'data: {"choices":broken}\n\n',
                      'data: {"error":{"code":400}}\n\n'):
            with self.subTest(first=first), mock.patch.dict(
                os.environ, {"GHC_API_GEMINI_SAFE_STREAM_RETRY": "1"},
            ), mock.patch.object(gemini.time, "sleep") as sleep:
                self.handler.reset_mock()
                self.handler.return_value = Response(first, mimetype="text/event-stream")
                response = self.client.post(f"/v1beta/models/{MODEL}:streamGenerateContent", json=BODY)
                output = response.get_data(as_text=True)
                self.assertIn('"error"', output)
                self.assertEqual(self.handler.call_count, 1)
                sleep.assert_not_called()
                if first.startswith(partial):
                    self.assertEqual(output.count('"text": "partial"'), 1)
                if first.endswith('"code":400}}\n\n'):
                    self.assertIn('"code": 400', output)

    def test_empty_stream_retries_are_opt_in_bounded_and_diagnostic(self):
        for enabled, attempts in (("0", 1), ("1", 2)):
            with self.subTest(enabled=enabled), mock.patch.dict(
                os.environ, {"GHC_API_GEMINI_SAFE_STREAM_RETRY": enabled},
            ), mock.patch.object(gemini.time, "sleep"), self.assertLogs("ghc_api.app", level="WARNING") as logs:
                self.handler.reset_mock()
                self.handler.side_effect = [
                    Response("data: [DONE]\n\n", mimetype="text/event-stream") for _ in range(attempts)
                ]
                response = self.client.post(f"/v1beta/models/{MODEL}:streamGenerateContent", json=BODY)
                output = response.get_data(as_text=True)
                self.assertIn('"code": 502', output)
                self.assertNotIn('"finishReason"', output)
                self.assertEqual(self.handler.call_count, attempts)
                diagnostic = " ".join(logs.output)
                self.assertIn("emitted_events=0", diagnostic)
                self.assertIn("IncompleteChatStream", diagnostic)
                self.assertIn("without a finish reason", diagnostic)

    def test_authenticated_user_context_is_preserved(self):
        self.handler.side_effect = lambda payload: self.completion(g.user_id)
        with mock.patch.object(state, "enable_auth", True), mock.patch(
            "ghc_api.app.require_auth", return_value=mock.Mock(user_id="approved-user"),
        ):
            response = self.client.post(f"/v1beta/models/{MODEL}:generateContent", json=BODY)
        self.assertEqual(response.get_json()["candidates"][0]["content"]["parts"][0]["text"], "approved-user")

    def test_parallel_call_results_are_matched_by_id(self):
        body = {"contents": [
            {"role": "model", "parts": [
                {"functionCall": {"id": "a", "name": "lookup", "args": {"key": 1}}},
                {"functionCall": {"id": "b", "name": "lookup", "args": {"key": 2}}},
            ]},
            {"role": "user", "parts": [
                {"functionResponse": {"id": "b", "name": "lookup", "response": {"value": 20}}},
                {"functionResponse": {"id": "a", "name": "lookup", "response": {"value": 10}}},
            ]},
        ]}
        messages = to_chat(body, MODEL, False)["messages"]
        self.assertEqual([message["tool_call_id"] for message in messages[1:]], ["b", "a"])

    def test_cli_model_role_tool_results_are_not_assistant_messages(self):
        body = {"contents": [
            {"role": "model", "parts": [{"functionCall": {
                "id": "read_1", "name": "read_file", "args": {},
            }}]},
            {"role": "model", "parts": [{"functionResponse": {
                "id": "read_1", "name": "read_file", "response": {"text": "synthetic"},
            }}]},
        ]}
        messages = to_chat(body, MODEL, False)["messages"]
        self.assertEqual(messages[1], {
            "role": "tool", "tool_call_id": "read_1", "content": '{"text": "synthetic"}',
        })
        body["contents"][1]["parts"].append({"text": "Ambiguous assistant text"})
        with self.assertRaisesRegex(ValueError, "only functionResponse"):
            to_chat(body, MODEL, False)

    def test_catalog_lists_both_models_with_stable_ide_ids(self):
        self.stack.enter_context(mock.patch.object(state, "models", {"data": [
            {"id": ident} for ident in (MODEL, "gemini-3.8-flash")
        ]}))
        response = self.client.post("/v1internal:fetchAvailableModels", json={})
        self.assertEqual(response.status_code, 200)
        models = response.get_json()["models"]
        self.assertEqual(models[MODEL]["model"], "MODEL_PLACEHOLDER_M460")
        self.assertEqual(models["gemini-3.8-flash"]["model"], "MODEL_PLACEHOLDER_M461")
        self.assertNotIn("countTokens", str(response.get_json()))

    def test_google_onboarding_is_not_faked(self):
        response = self.client.post("/v1internal:loadCodeAssist", json={})
        self.assertEqual(response.status_code, 501)


class GeminiForwardingTest(unittest.TestCase):
    def test_streaming_http_error_keeps_status_and_closes_upstream(self):
        app = Flask(__name__)
        for status in (400, 429, 503):
            with self.subTest(status=status), app.test_request_context(), mock.patch.object(
                openai, "cache",
            ), mock.patch.object(openai, "log_upstream_error") as log, mock.patch.object(
                openai.requests, "post",
            ) as post, mock.patch.object(openai, "get_copilot_base_url", return_value="https://synthetic.invalid"):
                upstream = post.return_value
                upstream.status_code = status
                upstream.raise_for_status.side_effect = openai.requests.exceptions.HTTPError(response=upstream)
                response = openai.stream_chat_completions(
                    {"stream": True}, {}, "synthetic-http-error", "{}", 2, 0, MODEL, MODEL,
                )
                output = response.get_data(as_text=True)
                self.assertIn(f'"code": {status}', output)
                self.assertNotIn("[DONE]", output)
                upstream.iter_lines.assert_not_called()
                upstream.close.assert_called_once()
                self.assertEqual(log.call_args.kwargs["status_code"], status)

    def test_retry_closing_http_error_finalizes_request_accounting(self):
        app = Flask(__name__)
        with app.test_request_context(), mock.patch.object(openai, "cache") as cache, mock.patch.object(
            openai, "log_upstream_error",
        ), mock.patch.object(openai.requests, "post") as post, mock.patch.object(
            openai, "get_copilot_base_url", return_value="https://synthetic.invalid",
        ):
            post.return_value.status_code = 503
            post.return_value.raise_for_status.side_effect = openai.requests.exceptions.HTTPError()
            response = openai.stream_chat_completions(
                {"stream": True}, {}, "synthetic-closed-error", "{}", 2, 0, MODEL, MODEL,
            )
            self.assertIn('"code": 503', next(iter(response.response)))
            response.close()
            post.return_value.close.assert_called_once()
            cache.complete_request.assert_called_once()
            self.assertEqual(cache.complete_request.call_args.args[1]["status_code"], 503)

    def test_instructions_history_and_tools_reach_http_boundary_unchanged(self):
        body = {
            "systemInstruction": {"parts": [{"text": "SYNTHETIC_CLIENT_RULES"},
                                            {"text": "Use only declared tools."}]},
            "contents": [
                {"role": "user", "parts": [{"text": "Read the synthetic file."}]},
                {"role": "model", "parts": [{"functionCall": {
                    "id": "read_1", "name": "read_file", "args": {"path": "demo.py"},
                }}]},
                {"role": "user", "parts": [{"functionResponse": {
                    "id": "read_1", "name": "read_file", "response": {"text": "return 1"},
                }}]},
                {"role": "user", "parts": [{"text": "Now explain it."}]},
            ],
            "tools": [{"functionDeclarations": [{
                "name": "read_file", "description": "Read a synthetic file.",
                "parameters": {"type": "OBJECT", "properties": {
                    "path": {"type": "STRING"},
                }, "required": ["path"]},
            }]}],
            "generationConfig": {"thinkingConfig": {"thinkingLevel": "HIGH"}},
        }
        expected_messages = [
            {"role": "system", "content": "SYNTHETIC_CLIENT_RULES\nUse only declared tools."},
            {"role": "user", "content": "Read the synthetic file."},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "read_1", "type": "function", "function": {
                    "name": "read_file", "arguments": '{"path": "demo.py"}',
                },
            }]},
            {"role": "tool", "tool_call_id": "read_1", "content": '{"text": "return 1"}'},
            {"role": "user", "content": "Now explain it."},
        ]
        app = Flask(__name__)
        app.register_blueprint(gemini.gemini_bp)
        upstream_body = {"choices": [{"message": {"content": "Synthetic response"},
                                     "finish_reason": "stop"}], "usage": {}}
        upstream = mock.Mock(status_code=200, ok=True, text=json.dumps(upstream_body))
        upstream.json.return_value = upstream_body
        with ExitStack() as stack:
            for name, value in {
                "ensure_copilot_token": mock.Mock(),
                "get_copilot_headers": mock.Mock(return_value={}),
                "get_copilot_base_url": lambda: "https://upstream.test",
                "translate_model_name": lambda model: model,
                "_should_route_chat_completions_via_responses": lambda model: False,
            }.items():
                stack.enter_context(mock.patch.object(openai, name, value))
            stack.enter_context(mock.patch.object(openai.cache, "add_request"))
            post = stack.enter_context(mock.patch.object(openai.requests, "post", return_value=upstream))
            for model, placeholder in gemini.CLOUD_MODELS.items():
                for cloud in (False, True):
                    with self.subTest(model=placeholder, cloud=cloud):
                        path = ("/v1internal:generateContent" if cloud else
                                f"/v1beta/models/{placeholder}:generateContent")
                        request_body = {"model": model, "request": body} if cloud else body
                        response = app.test_client().post(path, json=request_body)
                        self.assertEqual(response.status_code, 200, response.data)
                        post.assert_called_once()
                        self.assertEqual(post.call_args.args[0], "https://upstream.test/chat/completions")
                        self.assertEqual(post.call_args.kwargs["json"], {
                            "model": placeholder, "stream": False, "reasoning_effort": "high",
                            "messages": expected_messages,
                            "tools": [{"type": "function", "function": {
                                "name": "read_file", "description": "Read a synthetic file.",
                                "parameters": {"type": "object", "properties": {
                                    "path": {"type": "string"},
                                }, "required": ["path"]},
                            }}],
                        })
                        self.assertEqual(post.call_args.kwargs["headers"]["X-Initiator"], "agent")
                        post.reset_mock()


if __name__ == "__main__":
    unittest.main()
