import copy
import json
import threading
import time
import unittest
from contextlib import ExitStack
from unittest import mock

from flask import Response

from ghc_api.app import create_app
from ghc_api.cache import RequestCache
from ghc_api.reasoning_carrier import build_reasoning_carrier
from ghc_api.routes import anthropic as anthropic_module
from ghc_api.sse import base as sse_base_module


class _FakeResponse:
    def __init__(self, body, status_code=200, lines=None):
        self._body = body
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = body if isinstance(body, str) else json.dumps(body)
        self._lines = list(lines or [])
        self.closed = False

    def json(self):
        if isinstance(self._body, str):
            return json.loads(self._body)
        return copy.deepcopy(self._body)

    def iter_lines(self):
        yield from self._lines

    def close(self):
        self.closed = True


class _RaisingStreamResponse(_FakeResponse):
    def __init__(self, exc):
        super().__init__({})
        self._exc = exc

    def iter_lines(self):
        raise self._exc
        yield  # pragma: no cover


def _call_argument(call, position, keyword):
    """Read an argument without coupling tests to positional-vs-keyword style."""
    args, kwargs = call
    return kwargs[keyword] if keyword in kwargs else args[position]


class AnthropicResponsesRouteSelectionTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config.update(TESTING=True)
        self.client = self.app.test_client()
        self._state_values = {
            "anthropic_responses_compat_enabled": getattr(
                anthropic_module.state, "anthropic_responses_compat_enabled", True
            ),
            "enable_auth": anthropic_module.state.enable_auth,
        }
        anthropic_module.state.anthropic_responses_compat_enabled = True
        anthropic_module.state.enable_auth = False

    def tearDown(self):
        self.client = None
        for name, value in self._state_values.items():
            setattr(anthropic_module.state, name, value)

    @staticmethod
    def _response(label):
        return Response(
            json.dumps({"path": label}),
            status=200,
            mimetype="application/json",
        )

    def _selection_patches(
        self,
        *,
        direct,
        responses,
        translated_model="gpt-5.6-sol",
        advertises_messages=False,
    ):
        stack = ExitStack()
        patches = {
            "ensure": stack.enter_context(
                mock.patch.object(anthropic_module, "ensure_copilot_token")
            ),
            "translate_model": stack.enter_context(
                mock.patch.object(
                    anthropic_module,
                    "translate_model_name",
                    return_value=translated_model,
                )
            ),
            "supports_direct": stack.enter_context(
                mock.patch.object(
                    anthropic_module,
                    "supports_direct_anthropic_api",
                    return_value=direct,
                )
            ),
            "advertises_messages": stack.enter_context(
                mock.patch.object(
                    anthropic_module,
                    "advertises_anthropic_messages_api",
                    return_value=advertises_messages,
                )
            ),
            # create=True keeps this test importable while the route branch is
            # being implemented; it is still required to be called by the test.
            "supports_responses": stack.enter_context(
                mock.patch.object(
                    anthropic_module,
                    "supports_responses_api",
                    return_value=responses,
                    create=True,
                )
            ),
            "direct": stack.enter_context(
                mock.patch.object(
                    anthropic_module,
                    "handle_direct_anthropic_request",
                    return_value=self._response("direct"),
                )
            ),
            "responses": stack.enter_context(
                mock.patch.object(
                    anthropic_module,
                    "handle_responses_anthropic_request",
                    return_value=self._response("responses"),
                    create=True,
                )
            ),
            "fallback": stack.enter_context(
                mock.patch.object(
                    anthropic_module,
                    "handle_translated_request",
                    return_value=self._response("fallback"),
                )
            ),
        }
        return stack, patches

    def test_native_messages_endpoint_has_priority_over_responses(self):
        payload = {
            "model": "client-model-alias",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 32,
        }
        stack, patched = self._selection_patches(direct=True, responses=True)
        with stack:
            response = self.client.post("/v1/messages", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"path": "direct"})
        patched["direct"].assert_called_once()
        patched["responses"].assert_not_called()
        patched["fallback"].assert_not_called()

    def test_native_messages_path_strips_synthetic_responses_reasoning_carriers(self):
        signature = build_reasoning_carrier(
            model="gpt-5.6-sol",
            wire_profile="copilot_responses_lite",
            encrypted_content="opaque",
        )
        payload = {
            "model": "claude-opus",
            "messages": [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "synthetic", "signature": signature},
                        {"type": "text", "text": "answer"},
                    ],
                },
                {"role": "user", "content": "continue"},
            ],
            "max_tokens": 32,
        }
        stack, patched = self._selection_patches(
            direct=True, responses=True, translated_model="claude-opus"
        )
        with stack:
            response = self.client.post("/v1/messages", json=payload)

        self.assertEqual(response.status_code, 200)
        forwarded = patched["direct"].call_args.args[0]
        self.assertEqual(
            forwarded["messages"][1]["content"],
            [{"type": "text", "text": "answer"}],
        )
        cached_original = patched["direct"].call_args.args[5]
        self.assertEqual(
            [block["type"] for block in cached_original["messages"][1]["content"]],
            ["thinking", "text"],
        )
        self.assertIn(
            "Responses reasoning carrier",
            cached_original["messages"][1]["content"][0]["signature"],
        )

    def test_legacy_fallback_strips_and_redacts_responses_reasoning_carriers(self):
        signature = build_reasoning_carrier(
            model="gpt-5.6-sol",
            wire_profile="copilot_responses_lite",
            encrypted_content="opaque-secret",
        )
        payload = {
            "model": "gpt-5.6-sol",
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "summary", "signature": signature},
                    {"type": "text", "text": "answer"},
                ]},
                {"role": "user", "content": "continue"},
            ],
            "max_tokens": 32,
        }
        stack, patched = self._selection_patches(direct=False, responses=False)
        with stack:
            response = self.client.post("/v1/messages", json=payload)

        self.assertEqual(response.status_code, 200)
        forwarded = patched["fallback"].call_args.args[0]
        self.assertEqual(
            forwarded["messages"][0]["content"],
            [{"type": "text", "text": "answer"}],
        )
        cached_original = patched["fallback"].call_args.args[5]
        cached_signature = cached_original["messages"][0]["content"][0]["signature"]
        self.assertIn("Responses reasoning carrier", cached_signature)
        self.assertNotIn("opaque-secret", json.dumps(cached_original))

    def test_responses_branch_applies_policy_filters_without_legacy_thinking_rewrites(self):
        payload = {
            "model": "client-model-alias",
            "system": [{
                "type": "text",
                "text": "keep this exact system block",
                "cache_control": {"type": "ephemeral", "scope": "request"},
            }],
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "keep suffix verbatim <result-tail>",
                    "is_error": False,
                }],
            }],
            "thinking": {"type": "enabled", "budget_tokens": 8192},
            "output_config": {"effort": "high"},
            "max_tokens": 16384,
            "stream": False,
        }
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        stack, patched = self._selection_patches(direct=False, responses=True)
        with stack:
            filters = {
                "system": stack.enter_context(mock.patch.object(
                    anthropic_module,
                    "apply_system_prompt_filters_to_payload",
                    side_effect=lambda value: value,
                )),
                "tool_result": stack.enter_context(mock.patch.object(
                    anthropic_module,
                    "apply_tool_result_suffix_filter_to_payload",
                    side_effect=lambda value: value,
                )),
                "thinking": stack.enter_context(mock.patch.object(
                    anthropic_module,
                    "translate_thinking_enabled_to_adaptive",
                    side_effect=AssertionError("Responses path rewrote thinking"),
                )),
                "effort": stack.enter_context(mock.patch.object(
                    anthropic_module,
                    "apply_effort_policy",
                    side_effect=AssertionError("Responses path filtered output_config"),
                )),
            }
            response = self.client.post(
                "/v1/messages",
                data=raw,
                content_type="application/json",
                headers={"x-session-id": "session-redacted"},
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json(), {"path": "responses"})
            patched["responses"].assert_called_once()
            call = patched["responses"].call_args
            forwarded = _call_argument(call, 0, "anthropic_payload")
            expected = copy.deepcopy(payload)
            expected["model"] = "gpt-5.6-sol"
            self.assertEqual(forwarded, expected)
            self.assertEqual(_call_argument(call, 3, "original_model"), "client-model-alias")
            self.assertEqual(_call_argument(call, 4, "translated_model"), "gpt-5.6-sol")
            self.assertEqual(_call_argument(call, 5, "original_request_body"), payload)
            self.assertEqual(_call_argument(call, 6, "original_request_raw"), raw)
            filters["system"].assert_called_once()
            filters["tool_result"].assert_called_once()
            filters["thinking"].assert_not_called()
            filters["effort"].assert_not_called()

        patched["direct"].assert_not_called()
        patched["fallback"].assert_not_called()

    def test_redirected_model_that_still_advertises_messages_uses_legacy_fallback(self):
        stack, patched = self._selection_patches(
            direct=False,
            responses=True,
            advertises_messages=True,
        )
        with stack:
            response = self.client.post("/v1/messages", json={
                "model": "client-model-alias",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
            })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"path": "fallback"})
        patched["fallback"].assert_called_once()
        patched["responses"].assert_not_called()

    def test_disabled_responses_compatibility_uses_legacy_fallback(self):
        anthropic_module.state.anthropic_responses_compat_enabled = False
        stack, patched = self._selection_patches(direct=False, responses=True)
        with stack:
            response = self.client.post("/v1/messages", json={
                "model": "client-model-alias",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
            })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"path": "fallback"})
        patched["fallback"].assert_called_once()
        patched["responses"].assert_not_called()

    def test_model_without_responses_endpoint_uses_legacy_fallback(self):
        stack, patched = self._selection_patches(direct=False, responses=False)
        with stack:
            response = self.client.post("/v1/messages", json={
                "model": "client-model-alias",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
            })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"path": "fallback"})
        patched["fallback"].assert_called_once()
        patched["responses"].assert_not_called()

    def test_deeply_nested_json_is_rejected_before_any_upstream_path(self):
        depth = 2000
        raw = (
            b'{"model":"gpt-5.6-sol","messages":[],"max_tokens":16,"nested":'
            + b"[" * depth
            + b"]" * depth
            + b"}"
        )
        stack, patched = self._selection_patches(direct=False, responses=True)
        with stack:
            response = self.client.post(
                "/v1/messages", data=raw, content_type="application/json"
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["type"], "error")
        self.assertIn("nesting", response.get_json()["error"]["message"].lower())
        patched["direct"].assert_not_called()
        patched["responses"].assert_not_called()
        patched["fallback"].assert_not_called()

    def test_duplicate_json_key_is_rejected_before_any_upstream_path(self):
        raw = (
            b'{"model":"gpt-5.6-sol","model":"shadowed-model",'
            b'"messages":[],"max_tokens":16}'
        )
        stack, patched = self._selection_patches(direct=False, responses=True)
        with stack:
            response = self.client.post(
                "/v1/messages", data=raw, content_type="application/json"
            )

        self.assertEqual(response.status_code, 400)
        body = response.get_json()
        self.assertEqual(body["type"], "error")
        self.assertEqual(body["error"]["type"], "invalid_request_error")
        self.assertIn("Duplicate JSON object key", body["error"]["message"])
        patched["direct"].assert_not_called()
        patched["responses"].assert_not_called()
        patched["fallback"].assert_not_called()


class AnthropicResponsesRouteTransportTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config.update(TESTING=True)
        self.client = self.app.test_client()
        self.cache = RequestCache()
        state = anthropic_module.state
        self._state_values = {
            name: getattr(state, name)
            for name in (
                "anthropic_responses_compat_enabled",
                "anthropic_responses_wire_profile",
                "enable_auth",
                "max_connection_retries",
                "sse_keepalive_interval",
                "upstream_read_timeout",
                "enable_responses_early_failure_retry",
            )
        }
        state.anthropic_responses_compat_enabled = True
        state.anthropic_responses_wire_profile = "copilot_responses_lite"
        state.enable_auth = False
        state.max_connection_retries = 0
        state.sse_keepalive_interval = 0
        state.upstream_read_timeout = 123
        state.enable_responses_early_failure_retry = True

        self._patches = [
            mock.patch.object(anthropic_module, "cache", self.cache),
            mock.patch.object(sse_base_module, "cache", self.cache),
            mock.patch.object(anthropic_module, "ensure_copilot_token"),
            mock.patch.object(
                anthropic_module, "translate_model_name", side_effect=lambda value: value
            ),
            mock.patch.object(
                anthropic_module, "supports_direct_anthropic_api", return_value=False
            ),
            mock.patch.object(
                anthropic_module,
                "supports_responses_api",
                return_value=True,
                create=True,
            ),
            mock.patch.object(
                anthropic_module, "get_copilot_base_url", return_value="https://copilot.invalid"
            ),
            mock.patch.object(
                anthropic_module,
                "get_copilot_headers",
                return_value={"Authorization": "Bearer redacted"},
            ),
        ]
        for patcher in self._patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self._patches):
            patcher.stop()
        for name, value in self._state_values.items():
            setattr(anthropic_module.state, name, value)
        self.client = None

    @staticmethod
    def _request_payload(stream=False):
        return {
            "model": "gpt-5.6-sol",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 64,
            "stream": stream,
        }

    @staticmethod
    def _terminal_response():
        return {
            "id": "resp_fixture_1",
            "model": "gpt-5.6-sol",
            "status": "completed",
            "output": [{
                "type": "message",
                "role": "assistant",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": "hello back"}],
            }],
            "usage": {
                "input_tokens": 7,
                "input_tokens_details": {"cached_tokens": 2},
                "output_tokens": 3,
            },
        }

    def test_nonstream_posts_responses_payload_and_returns_anthropic_message(self):
        upstream = _FakeResponse(self._terminal_response())
        with mock.patch.object(anthropic_module.requests, "post", return_value=upstream) as post:
            response = self.client.post(
                "/v1/messages",
                json=self._request_payload(stream=False),
                headers={"user-agent": "claude-cli/2.1.207"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["type"], "message")
        self.assertEqual(body["role"], "assistant")
        self.assertEqual(body["model"], "gpt-5.6-sol")
        self.assertEqual(body["content"], [{"type": "text", "text": "hello back"}])
        self.assertEqual(body["usage"]["input_tokens"], 5)
        self.assertEqual(body["usage"]["cache_read_input_tokens"], 2)

        post.assert_called_once()
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://copilot.invalid/v1/responses")
        self.assertFalse(kwargs["stream"])
        self.assertEqual(kwargs["timeout"], 123)
        self.assertEqual(kwargs["json"], {
            "model": "gpt-5.6-sol",
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }],
            "store": False,
            "stream": False,
            "include": ["reasoning.encrypted_content"],
            "max_output_tokens": 64,
            "text": {"verbosity": "low"},
        })

        self.assertEqual(len(self.cache.cache), 1)
        cached = next(iter(self.cache.cache.values()))
        self.assertEqual(cached["endpoint"], "/v1/messages")
        self.assertEqual(cached["request_body"], kwargs["json"])
        self.assertEqual(cached["response_body"], body)
        self.assertEqual(cached["compatibility_profile"], "copilot_responses_lite")
        self.assertIsInstance(cached["compatibility_warnings"], list)
        self.assertIn("request", cached["conversion_report"])
        self.assertIn("response", cached["conversion_report"])

    def test_session_header_restores_prompt_cache_metadata(self):
        upstream = _FakeResponse(self._terminal_response())
        with mock.patch.object(anthropic_module.requests, "post", return_value=upstream) as post:
            response = self.client.post(
                "/v1/messages",
                json=self._request_payload(stream=False),
                headers={"X-Claude-Code-Session-Id": "session-fixture"},
            )
        self.assertEqual(response.status_code, 200)
        forwarded = post.call_args.kwargs["json"]
        self.assertRegex(forwarded["prompt_cache_key"], r"^[0-9a-f]{64}$")
        self.assertRegex(
            forwarded["client_metadata"]["session_id"], r"^[0-9a-f]{64}$"
        )
        self.assertNotEqual(
            forwarded["client_metadata"]["session_id"], "session-fixture"
        )

    def test_nonstream_converts_web_search_billing_and_structured_output(self):
        upstream_body = self._terminal_response()
        upstream_body["output"] = [
            {
                "type": "web_search_call",
                "id": "search_1",
                "status": "completed",
                "action": {"type": "search", "query": "private query"},
            },
            *upstream_body["output"],
        ]
        upstream_body["tool_usage"] = {"web_search": {"num_requests": 1}}
        payload = {
            "model": "gpt-5.6-sol",
            "system": [
                {"type": "text", "text": "x-anthropic-billing-header: cc_version=test;"},
                {"type": "text", "text": "keep this system prompt"},
            ],
            "messages": [{"role": "user", "content": "search"}],
            "max_tokens": 64,
            "tools": [{
                "type": "web_search_20250305",
                "name": "web_search",
                "allowed_domains": ["python.org"],
            }],
            "output_config": {"format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                    "additionalProperties": False,
                },
            }},
            "stream": False,
        }
        with mock.patch.object(
            anthropic_module.requests,
            "post",
            return_value=_FakeResponse(upstream_body),
        ) as post:
            response = self.client.post(
                "/v1/messages",
                json=payload,
                headers={"user-agent": "claude-cli/2.1.207"},
            )

        self.assertEqual(response.status_code, 200)
        forwarded = post.call_args.kwargs["json"]
        self.assertEqual(forwarded["tools"], [{
            "type": "web_search",
            "filters": {"allowed_domains": ["python.org"]},
        }])
        self.assertEqual(
            forwarded["input"][0]["content"],
            [{"type": "input_text", "text": "keep this system prompt"}],
        )
        self.assertNotIn("x-anthropic-billing-header", json.dumps(forwarded))
        self.assertRegex(
            forwarded["text"]["format"]["name"],
            r"^ghc_schema_[0-9a-f]{16}$",
        )
        cached = next(iter(self.cache.cache.values()))
        self.assertNotIn("private query", json.dumps(cached["upstream_response_body"]))

    def test_description_less_tool_is_not_forwarded_with_an_empty_description(self):
        payload = self._request_payload(stream=False)
        payload["tools"] = [
            {"name": "mcp__srv__run", "input_schema": {"type": "object"}},
            {"name": "blank", "description": " ", "input_schema": {"type": "object"}},
        ]
        with mock.patch.object(
            anthropic_module.requests,
            "post",
            return_value=_FakeResponse(self._terminal_response()),
        ) as post:
            response = self.client.post(
                "/v1/messages",
                json=payload,
                headers={"user-agent": "claude-cli/2.1.207"},
            )

        self.assertEqual(response.status_code, 200)
        forwarded = post.call_args.kwargs["json"]
        tools = next(
            item for item in forwarded["input"]
            if item.get("type") == "additional_tools"
        )["tools"]
        self.assertNotIn("description", tools[0])
        self.assertEqual(tools[1]["description"], "Tool: blank.")
        self.assertNotIn('"description": ""', json.dumps(forwarded))

    def test_orphaned_tool_result_never_reaches_upstream(self):
        payload = self._request_payload(stream=False)
        payload["messages"] = [
            {"role": "user", "content": "run it"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_live", "name": "Bash", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_live", "content": "ok"},
                {"type": "tool_result", "tool_use_id": "toolu_truncated", "content": "stale"},
            ]},
        ]
        with mock.patch.object(
            anthropic_module.requests,
            "post",
            return_value=_FakeResponse(self._terminal_response()),
        ) as post:
            response = self.client.post(
                "/v1/messages",
                json=payload,
                headers={"user-agent": "claude-cli/2.1.207"},
            )

        self.assertEqual(response.status_code, 200)
        forwarded = post.call_args.kwargs["json"]
        self.assertEqual(
            [
                item["call_id"] for item in forwarded["input"]
                if item.get("type") == "function_call_output"
            ],
            ["toolu_live"],
        )
        self.assertIn(
            "conversion.approximation",
            response.headers.get("X-GHC-Compatibility-Warnings", ""),
        )

    def test_malformed_json_schema_is_rejected_before_upstream(self):
        payload = self._request_payload(stream=False)
        payload["output_config"] = {
            "format": {"type": "json_schema", "schema": "not-an-object"}
        }
        with mock.patch.object(anthropic_module.requests, "post") as post:
            response = self.client.post(
                "/v1/messages",
                json=payload,
                headers={"user-agent": "claude-cli/2.1.207"},
            )
        self.assertEqual(response.status_code, 400)
        post.assert_not_called()
        self.assertIn("object schema", response.get_json()["error"]["message"])

    def test_response_metadata_without_a_client_field_is_not_an_error(self):
        """A native web_search_call has no Anthropic representation, and a
        Responses envelope (status/created_at/item ids) has no client field at
        all.  Neither may fail the answer: the message is delivered and the
        approximation is reported in the warning header.
        """
        upstream_body = self._terminal_response()
        upstream_body["output"] = [
            {
                "type": "web_search_call",
                "id": "search_1",
                "status": "completed",
                "action": {"type": "search", "query": "fixture"},
            },
            *upstream_body["output"],
        ]
        with mock.patch.object(
            anthropic_module.requests,
            "post",
            return_value=_FakeResponse(upstream_body),
        ) as post:
            response = self.client.post(
                "/v1/messages",
                json=self._request_payload(stream=False),
                headers={
                    "User-Agent": "claude-cli/2.1.207 (fixture)",
                    "Anthropic-Version": "2023-06-01",
                    "Anthropic-Beta": ",".join(sorted((
                        "claude-code-20250219",
                        "context-1m-2025-08-07",
                        "context-management-2025-06-27",
                        "effort-2025-11-24",
                        "interleaved-thinking-2025-05-14",
                        "mid-conversation-system-2026-04-07",
                        "prompt-caching-scope-2026-01-05",
                        "redact-thinking-2026-02-12",
                        "thinking-token-count-2026-05-13",
                    ))),
                },
            )
        post.assert_called_once()
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(
            body["content"], [{"type": "text", "text": "hello back"}]
        )
        # The search is not invented as an Anthropic block, but it is on record.
        cached = next(iter(self.cache.cache.values()))
        self.assertTrue(any(
            record["source_path"] == "/output/0"
            and record["disposition"] == "sidecar"
            for record in cached["conversion_report"]["response"]["records"]
        ))

    def test_clean_response_round_trips_without_warnings(self):
        with mock.patch.object(
            anthropic_module.requests,
            "post",
            return_value=_FakeResponse(self._terminal_response()),
        ):
            response = self.client.post(
                "/v1/messages",
                json=self._request_payload(stream=False),
                headers={
                    "User-Agent": "claude-cli/2.1.207 (fixture)",
                    "Anthropic-Version": "2023-06-01",
                    "Anthropic-Beta": ",".join(sorted((
                        "claude-code-20250219",
                        "context-1m-2025-08-07",
                        "context-management-2025-06-27",
                        "effort-2025-11-24",
                        "interleaved-thinking-2025-05-14",
                        "mid-conversation-system-2026-04-07",
                        "prompt-caching-scope-2026-01-05",
                        "redact-thinking-2026-02-12",
                        "thinking-token-count-2026-05-13",
                    ))),
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.headers.get("X-GHC-Compatibility-Warnings"))

    def test_nonstream_unknown_status_fails_closed(self):
        """Unlike an unknown stream event, an unknown status has no terminal
        event left to reconcile against: it is what says whether the output is
        complete, truncated, or failed. It must stay a hard 502.
        """
        upstream_body = {**self._terminal_response(), "status": "future-status-private"}
        with mock.patch.object(
            anthropic_module.requests,
            "post",
            return_value=_FakeResponse(upstream_body),
        ):
            response = self.client.post(
                "/v1/messages", json=self._request_payload(stream=False)
            )
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.get_json()["type"], "error")
        self.assertNotIn("future-status-private", response.get_data(as_text=True))
        cached = next(iter(self.cache.cache.values()))
        self.assertEqual(cached["status_code"], 502)
        warning_codes = {item["code"] for item in cached["compatibility_warnings"]}
        self.assertIn("responses.unknown_response_status", warning_codes)
        self.assertNotIn(
            "future-status-private",
            json.dumps(cached["compatibility_warnings"], ensure_ascii=False),
        )

    def test_stream_uses_anthropic_sse_and_keeps_raw_responses_events_in_cache(self):
        terminal = self._terminal_response()
        events = [
            {"type": "response.created", "response": {"id": "resp_fixture_1", "model": "gpt-5.6-sol"}},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "reasoning", "encrypted_content": "opaque-fixture"},
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "reasoning", "summary": [], "encrypted_content": "opaque-fixture"},
            },
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {"type": "message", "role": "assistant", "phase": "final_answer", "content": []},
            },
            {
                "type": "response.output_text.delta",
                "output_index": 1,
                "content_index": 0,
                "item_id": "msg_fixture_1",
                "delta": "hello back",
                "logprobs": [],
            },
            {
                "type": "response.output_item.done",
                "output_index": 1,
                "item": terminal["output"][0],
            },
            {
                "type": "response.completed",
                "response": {
                    **terminal,
                    "output": [
                        {"type": "reasoning", "summary": [], "encrypted_content": "opaque-fixture"},
                        terminal["output"][0],
                    ],
                },
            },
        ]
        for sequence_number, event in enumerate(events):
            event["sequence_number"] = sequence_number
        events[1]["item"]["summary"] = []
        lines = [
            ("data: " + json.dumps(event, separators=(",", ":"))).encode()
            for event in events
        ]
        upstream = _FakeResponse({}, lines=lines)

        with mock.patch.object(anthropic_module.requests, "post", return_value=upstream) as post:
            response = self.client.post(
                "/v1/messages", json=self._request_payload(stream=True)
            )
            body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/event-stream")
        self.assertIn("event: message_start\n", body)
        self.assertIn("event: content_block_delta\n", body)
        self.assertIn('"text":"hello back"', body)
        self.assertIn("event: message_stop\n", body)
        self.assertNotIn("response.output_text.delta", body)
        self.assertNotIn("opaque-fixture", body)
        self.assertNotIn("data: [DONE]", body)
        self.assertTrue(post.call_args.kwargs["stream"])

        cached = next(iter(self.cache.cache.values()))
        self.assertEqual(cached["state"], self.cache.STATE_COMPLETED)
        self.assertFalse(any("opaque-fixture" in item for item in cached["raw_events"]))
        self.assertTrue(cached["raw_events_redacted"])
        self.assertTrue(any("opaque reasoning state" in item for item in cached["raw_events"]))
        self.assertEqual(
            [block["type"] for block in cached["response_body"]["content"]],
            ["thinking", "text"],
        )
        self.assertEqual(cached["response_body"]["content"][1], {
            "type": "text", "text": "hello back"
        })
        self.assertIn("stream", cached["conversion_report"])
        self.assertIn("response", cached["conversion_report"])

    def test_stream_retries_early_response_failed_before_anthropic_output(self):
        failed_events = [
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {"id": "resp_failed", "model": "gpt-5.6-sol"},
            },
            {
                "type": "response.failed",
                "sequence_number": 1,
                "response": {
                    "id": "resp_failed",
                    "model": "gpt-5.6-sol",
                    "status": "failed",
                    "error": {"message": "transient failure"},
                    "output": [],
                    "usage": {},
                },
            },
        ]
        first = _FakeResponse({}, lines=[
            ("data: " + json.dumps(event, separators=(",", ":"))).encode()
            for event in failed_events
        ])
        completed = {
            "type": "response.completed",
            "sequence_number": 0,
            "response": self._terminal_response(),
        }
        second = _FakeResponse({}, lines=[
            ("data: " + json.dumps(completed, separators=(",", ":"))).encode()
        ])

        with mock.patch.object(
            anthropic_module.state, "max_connection_retries", 1
        ), mock.patch.object(
            anthropic_module.requests,
            "post",
            side_effect=[first, second],
        ) as post:
            response = self.client.post(
                "/v1/messages", json=self._request_payload(stream=True)
            )
            body = response.get_data(as_text=True)

        self.assertEqual(post.call_count, 2)
        self.assertIn("event: message_stop\n", body)
        self.assertIn('"text":"hello back"', body)
        self.assertNotIn("event: error\n", body)
        self.assertNotIn("transient failure", body)

    def test_connection_retry_rebuilds_headers_after_token_refresh(self):
        """A retry must not re-send the token the refresh just replaced.

        get_copilot_headers() bakes in state.copilot_token and a fresh
        X-Request-Id, so reusing one dict across attempts would 401 on exactly
        the case the retry exists for, and would repeat one request id upstream.
        """
        completed = {
            "type": "response.completed",
            "sequence_number": 0,
            "response": self._terminal_response(),
        }
        second = _FakeResponse(self._terminal_response())
        tokens = iter(("stale-token", "refreshed-token"))
        request_ids = iter(("request-id-1", "request-id-2"))

        def fresh_headers(*args, **kwargs):
            return {
                "Authorization": f"Bearer {next(tokens)}",
                "X-Request-Id": next(request_ids),
            }

        posts = [
            anthropic_module.requests.exceptions.ConnectionError("transport reset"),
            second,
        ]
        with mock.patch.object(
            anthropic_module.state, "max_connection_retries", 1
        ), mock.patch.object(
            anthropic_module, "get_copilot_headers", side_effect=fresh_headers
        ), mock.patch.object(
            anthropic_module.requests, "post", side_effect=posts
        ) as post:
            response = self.client.post(
                "/v1/messages", json=self._request_payload(stream=False)
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(post.call_count, 2)
        sent = [call.kwargs["headers"] for call in post.call_args_list]
        self.assertEqual(
            [header["Authorization"] for header in sent],
            ["Bearer stale-token", "Bearer refreshed-token"],
        )
        self.assertEqual(
            [header["X-Request-Id"] for header in sent],
            ["request-id-1", "request-id-2"],
        )
        self.assertEqual({header["X-Initiator"] for header in sent}, {"user"})

    def test_connection_retry_after_keepalive_commit_rebuilds_headers_too(self):
        """Same requirement on the retry loop that runs after the response has
        already been committed to streaming."""
        completed = {
            "type": "response.completed",
            "sequence_number": 0,
            "response": self._terminal_response(),
        }
        second = _FakeResponse({}, lines=[
            ("data: " + json.dumps(completed, separators=(",", ":"))).encode()
        ])
        tokens = iter(("stale-token", "refreshed-token"))

        def fresh_headers(*args, **kwargs):
            return {"Authorization": f"Bearer {next(tokens)}"}

        def slow_then_fail(*args, **kwargs):
            if post.call_count == 1:
                time.sleep(0.15)
                raise anthropic_module.requests.exceptions.ConnectionError(
                    "transport reset"
                )
            return second

        with mock.patch.object(
            anthropic_module.state, "max_connection_retries", 1
        ), mock.patch.object(
            anthropic_module.state, "sse_keepalive_interval", 0.05
        ), mock.patch.object(
            anthropic_module.state, "responses_pre_header_grace", 0.05
        ), mock.patch.object(
            anthropic_module, "get_copilot_headers", side_effect=fresh_headers
        ), mock.patch.object(
            anthropic_module.requests, "post", side_effect=slow_then_fail
        ) as post:
            response = self.client.post(
                "/v1/messages", json=self._request_payload(stream=True)
            )
            body = response.get_data(as_text=True)

        self.assertEqual(post.call_count, 2)
        self.assertIn("event: message_stop\n", body)
        sent = [call.kwargs["headers"] for call in post.call_args_list]
        self.assertEqual(
            [header["Authorization"] for header in sent],
            ["Bearer stale-token", "Bearer refreshed-token"],
        )

    def test_stream_slow_upstream_headers_emit_anthropic_ping_before_completion(self):
        completed = {
            "type": "response.completed",
            "response": self._terminal_response(),
        }
        upstream = _FakeResponse({}, lines=[
            ("data: " + json.dumps(completed, separators=(",", ":"))).encode()
        ])

        def slow_post(*args, **kwargs):
            time.sleep(0.12)
            return upstream

        with mock.patch.object(
            anthropic_module.state, "sse_keepalive_interval", 0.05
        ), mock.patch.object(
            anthropic_module.requests, "post", side_effect=slow_post
        ) as post:
            response = self.client.post(
                "/v1/messages", json=self._request_payload(stream=True)
            )
            body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(body.startswith(
            'event: ping\ndata: {"type": "ping"}\n\n'
        ))
        self.assertIn("event: message_stop\n", body)
        post.assert_called_once()
        self.assertTrue(post.call_args.kwargs["stream"])

    def test_pre_header_connection_failure_uses_configured_retry_budget(self):
        completed = {
            "type": "response.completed",
            "response": self._terminal_response(),
        }
        upstream = _FakeResponse({}, lines=[
            ("data: " + json.dumps(completed, separators=(",", ":"))).encode()
        ])
        attempts = []

        def post_with_slow_first_failure(*args, **kwargs):
            attempts.append(kwargs)
            if len(attempts) == 1:
                threading.Event().wait(0.05)
                raise anthropic_module.requests.exceptions.ConnectionError(
                    "first attempt failed"
                )
            return upstream

        with mock.patch.object(
            anthropic_module.state, "sse_keepalive_interval", 0.01
        ), mock.patch.object(
            anthropic_module.state, "max_connection_retries", 1
        ), mock.patch.object(
            anthropic_module.time, "sleep", return_value=None
        ), mock.patch.object(
            anthropic_module, "log_connection_retry"
        ) as retry_log, mock.patch.object(
            anthropic_module.requests,
            "post",
            side_effect=post_with_slow_first_failure,
        ):
            response = self.client.post(
                "/v1/messages", json=self._request_payload(stream=True)
            )
            body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: ping\n", body)
        self.assertIn("event: message_stop\n", body)
        self.assertNotIn("event: error\n", body)
        self.assertEqual(len(attempts), 2)
        retry_log.assert_called_once()
        cached = next(iter(self.cache.cache.values()))
        self.assertEqual(cached["state"], self.cache.STATE_COMPLETED)

    def test_pre_header_grace_commits_stream_before_keepalive_interval(self):
        """The pre-header wait is bounded by responses_pre_header_grace.

        With a realistic 30s keepalive interval the client must not wait 30s for
        the first byte (and the worker must not sit in an uninterruptible wait
        where a client disconnect is invisible because nothing was written).
        """
        completed = {
            "type": "response.completed",
            "response": self._terminal_response(),
        }
        upstream = _FakeResponse({}, lines=[
            ("data: " + json.dumps(completed, separators=(",", ":"))).encode()
        ])
        release_response = threading.Event()

        def slow_post(*args, **kwargs):
            release_response.wait(2)
            return upstream

        started = time.monotonic()
        with mock.patch.object(
            anthropic_module.state, "sse_keepalive_interval", 30
        ), mock.patch.object(
            anthropic_module.state, "responses_pre_header_grace", 0.02
        ), mock.patch.object(
            anthropic_module.requests, "post", side_effect=slow_post
        ):
            response = self.client.post(
                "/v1/messages",
                json=self._request_payload(stream=True),
                buffered=False,
            )
            chunks = iter(response.response)
            first_chunk = next(chunks).decode("utf-8")
            elapsed = time.monotonic() - started
            release_response.set()
            body = first_chunk + "".join(
                chunk.decode("utf-8") for chunk in chunks
            )

        self.assertEqual(
            first_chunk, 'event: ping\ndata: {"type": "ping"}\n\n'
        )
        self.assertLess(elapsed, 5)
        self.assertIn("event: message_stop\n", body)

    def test_pre_header_disconnect_marks_499_and_closes_late_response(self):
        upstream = _FakeResponse({}, lines=[])
        post_started = threading.Event()
        release_response = threading.Event()

        def delayed_post(*args, **kwargs):
            post_started.set()
            release_response.wait(2)
            return upstream

        with mock.patch.object(
            anthropic_module.state, "sse_keepalive_interval", 0.02
        ), mock.patch.object(
            anthropic_module.requests, "post", side_effect=delayed_post
        ) as post:
            response = self.client.post(
                "/v1/messages",
                json=self._request_payload(stream=True),
                buffered=False,
            )
            try:
                first_chunk = next(iter(response.response)).decode("utf-8")
                self.assertEqual(
                    first_chunk,
                    'event: ping\ndata: {"type": "ping"}\n\n',
                )
                self.assertTrue(post_started.wait(1))
                response.close()

                cached = next(iter(self.cache.cache.values()))
                self.assertEqual(cached["status_code"], 499)
                self.assertEqual(cached["state"], self.cache.STATE_ERROR)
                self.assertIn(
                    "responses.client_disconnected",
                    {item["code"] for item in cached["compatibility_warnings"]},
                )
            finally:
                release_response.set()

            deadline = time.monotonic() + 2
            while not upstream.closed and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(upstream.closed)
            post.assert_called_once()

    def test_stream_unknown_responses_event_is_skipped_and_output_still_arrives(self):
        """An additive upstream event must not take the whole path down.

        The terminal event carries the full output array, so a skipped event
        costs incremental delivery, not model output. The approximation is still
        surfaced as a compatibility warning, and no upstream value leaks.
        """
        secret = "DO-NOT-LEAK-unknown-stream-fixture"
        upstream_event = {
            "type": "response.future_private_event",
            "private_value": secret,
        }
        terminal = {
            "type": "response.completed",
            "sequence_number": 1,
            "response": {
                "id": "resp_unknown_event",
                "model": "gpt-5.6-sol",
                "status": "completed",
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "still delivered"}],
                }],
                "usage": {"input_tokens": 1, "output_tokens": 2},
            },
        }
        upstream = _FakeResponse({}, lines=[
            ("data: " + json.dumps(event, separators=(",", ":"))).encode()
            for event in (upstream_event, terminal)
        ])
        with mock.patch.object(anthropic_module.requests, "post", return_value=upstream):
            response = self.client.post(
                "/v1/messages", json=self._request_payload(stream=True)
            )
            body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("event: error\n", body)
        self.assertIn("still delivered", body)
        self.assertIn("event: message_stop\n", body)
        self.assertNotIn("response.future_private_event", body)
        self.assertNotIn(secret, body)
        cached = next(iter(self.cache.cache.values()))
        self.assertEqual(cached["status_code"], 200)
        self.assertFalse(any(secret in item for item in cached["raw_events"]))
        warning_codes = {
            item["code"] for item in cached["compatibility_warnings"]
        }
        self.assertIn("responses.unknown_event", warning_codes)
        self.assertIn("responses.unknown_event_skipped", warning_codes)
        self.assertNotIn(
            secret,
            json.dumps(cached["compatibility_warnings"], ensure_ascii=False),
        )

    def test_stream_unknown_responses_item_type_still_fails_closed(self):
        """An unknown *item* has no second chance: it reaches the client as
        content or not at all, so it must remain a hard protocol error."""
        secret = "DO-NOT-LEAK-unknown-item-fixture"
        upstream_event = {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": {"type": "private_future_item", "private_value": secret},
        }
        upstream = _FakeResponse({}, lines=[
            ("data: " + json.dumps(upstream_event, separators=(",", ":"))).encode()
        ])
        with mock.patch.object(anthropic_module.requests, "post", return_value=upstream):
            response = self.client.post(
                "/v1/messages", json=self._request_payload(stream=True)
            )
            body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: error\n", body)
        self.assertNotIn(secret, body)
        cached = next(iter(self.cache.cache.values()))
        self.assertEqual(cached["status_code"], 502)
        self.assertEqual(cached["state"], self.cache.STATE_ERROR)
        warning_codes = {
            item["code"] for item in cached["compatibility_warnings"]
        }
        self.assertIn("responses.unknown_item", warning_codes)
        self.assertNotIn(
            secret,
            json.dumps(cached["compatibility_warnings"], ensure_ascii=False),
        )

    def test_stream_timeout_emits_safe_anthropic_error_and_marks_cache_error(self):
        upstream = _RaisingStreamResponse(
            anthropic_module.requests.exceptions.ReadTimeout("PRIVATE-UPSTREAM-DETAIL")
        )
        with mock.patch.object(anthropic_module.requests, "post", return_value=upstream):
            response = self.client.post(
                "/v1/messages", json=self._request_payload(stream=True)
            )
            body = response.get_data(as_text=True)

        self.assertIn("event: error\n", body)
        self.assertIn('"type":"timeout_error"', body)
        self.assertNotIn("PRIVATE-UPSTREAM-DETAIL", body)
        cached = next(iter(self.cache.cache.values()))
        self.assertEqual(cached["status_code"], 504)
        self.assertEqual(cached["state"], self.cache.STATE_ERROR)

    def test_upstream_error_is_returned_in_anthropic_error_envelope(self):
        upstream = _FakeResponse(
            {"error": {"message": "fixture rate limit", "type": "rate_limit"}},
            status_code=429,
        )
        with mock.patch.object(anthropic_module.requests, "post", return_value=upstream):
            response = self.client.post(
                "/v1/messages", json=self._request_payload(stream=False)
            )

        self.assertEqual(response.status_code, 429)
        body = response.get_json()
        self.assertEqual(body, {
            "type": "error",
            "error": {
                "type": "rate_limit_error",
                "message": "fixture rate limit",
            },
        })
        cached = next(iter(self.cache.cache.values()))
        self.assertEqual(cached["state"], self.cache.STATE_ERROR)
        self.assertEqual(cached["status_code"], 429)
        self.assertEqual(cached["response_body"], body)

    def test_non_boolean_stream_is_rejected_before_upstream(self):
        payload = self._request_payload(stream=False)
        payload["stream"] = "false"
        with mock.patch.object(anthropic_module.requests, "post") as post:
            response = self.client.post("/v1/messages", json=payload)

        self.assertEqual(response.status_code, 400)
        self.assertIn("stream", response.get_json()["error"]["message"])
        post.assert_not_called()

    def test_null_messages_is_rejected_before_upstream_in_compatibility_mode(self):
        payload = self._request_payload(stream=False)
        payload["messages"] = None
        with mock.patch.object(anthropic_module.requests, "post") as post:
            response = self.client.post("/v1/messages", json=payload)

        self.assertEqual(response.status_code, 400)
        body = response.get_json()
        self.assertEqual(body["type"], "error")
        self.assertEqual(body["error"]["type"], "invalid_request_error")
        self.assertIn("messages", body["error"]["message"])
        post.assert_not_called()
        cached = next(iter(self.cache.cache.values()))
        self.assertEqual(cached["status_code"], 400)
        self.assertEqual(cached["state"], self.cache.STATE_ERROR)
        self.assertIn(
            "conversion.unsupported",
            response.headers.get("X-GHC-Compatibility-Warnings", ""),
        )

    def test_unknown_terminal_responses_item_fails_closed_without_leaking_value(self):
        secret = "DO-NOT-LEAK-unknown-output-fixture"
        terminal = self._terminal_response()
        terminal["output"] = [{
            "type": "future_private_output_item",
            "private_value": secret,
        }]
        with mock.patch.object(
            anthropic_module.requests,
            "post",
            return_value=_FakeResponse(terminal),
        ):
            response = self.client.post(
                "/v1/messages", json=self._request_payload(stream=False)
            )

        self.assertEqual(response.status_code, 502)
        body = response.get_json()
        self.assertEqual(body["type"], "error")
        self.assertEqual(body["error"]["type"], "api_error")
        warning_header = response.headers.get("X-GHC-Compatibility-Warnings", "")
        self.assertIn("responses.unknown_item", warning_header)
        self.assertNotIn(secret, response.get_data(as_text=True))
        self.assertNotIn(secret, warning_header)
        cached = next(iter(self.cache.cache.values()))
        self.assertTrue(cached["upstream_response_body"]["_redacted"])
        self.assertNotIn(
            secret,
            json.dumps(cached["upstream_response_body"], ensure_ascii=False),
        )
        self.assertNotIn(
            secret,
            json.dumps(cached["compatibility_warnings"], ensure_ascii=False),
        )

    def test_unknown_claude_cli_version_sets_warning_header_and_safe_cache_warning(self):
        secret = "DO-NOT-LEAK-fixture-prompt-or-metadata"
        payload = self._request_payload(stream=False)
        payload["system"] = secret
        payload["metadata"] = {"user_id": secret}
        upstream = _FakeResponse(self._terminal_response())
        with mock.patch.object(anthropic_module.requests, "post", return_value=upstream):
            response = self.client.post(
                "/v1/messages",
                json=payload,
                headers={
                    "User-Agent": "claude-cli/99.1.2 (fixture)",
                    "Anthropic-Version": "2023-06-01",
                },
            )

        self.assertEqual(response.status_code, 200)
        warning_header = response.headers.get("X-GHC-Compatibility-Warnings", "")
        self.assertIn("claude_cli.unknown_version", warning_header)
        self.assertNotIn(secret, warning_header)
        cached = next(iter(self.cache.cache.values()))
        warning_codes = {
            item["code"] for item in cached["compatibility_warnings"]
        }
        self.assertIn("claude_cli.unknown_version", warning_codes)
        self.assertNotIn(
            secret,
            json.dumps(cached["compatibility_warnings"], ensure_ascii=False),
        )

    def test_unknown_beta_warns_but_still_serves_the_request(self):
        unknown_beta = "future-private-beta-fixture-value"
        with mock.patch.object(
            anthropic_module.requests,
            "post",
            return_value=_FakeResponse(self._terminal_response()),
        ) as post:
            response = self.client.post(
                "/v1/messages",
                json=self._request_payload(stream=False),
                headers={
                    "User-Agent": "claude-cli/2.1.207 (fixture)",
                    "Anthropic-Version": "2023-06-01",
                    "Anthropic-Beta": unknown_beta,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            "anthropic.beta_unknown",
            response.headers.get("X-GHC-Compatibility-Warnings", ""),
        )
        self.assertNotIn(unknown_beta, response.get_data(as_text=True))
        self.assertNotIn(
            unknown_beta,
            response.headers.get("X-GHC-Compatibility-Warnings", ""),
        )
        post.assert_called_once()

    def test_completed_reasoning_is_carried_by_client_and_restored_next_turn(self):
        opaque_reasoning = "opaque-encrypted-fixture"
        first_terminal = self._terminal_response()
        first_terminal["output"] = [
            {
                "id": "rs_fixture_1",
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": "brief reasoning"}],
                "encrypted_content": opaque_reasoning,
            },
            {
                "id": "msg_fixture_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": "hello back"}],
            },
        ]
        second_terminal = {
            **self._terminal_response(),
            "id": "resp_fixture_2",
            "output": [{
                "type": "message",
                "role": "assistant",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": "continued"}],
            }],
        }

        with mock.patch.object(
            anthropic_module.requests,
            "post",
            side_effect=[_FakeResponse(first_terminal), _FakeResponse(second_terminal)],
        ) as post:
            first_response = self.client.post(
                "/v1/messages", json=self._request_payload(stream=False)
            )
            first_body = first_response.get_json()
            continuation = self._request_payload(stream=False)
            continuation["messages"] = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": first_body["content"]},
                {"role": "user", "content": "continue"},
            ]
            second_response = self.client.post("/v1/messages", json=continuation)

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 200)
        self.assertEqual(first_body["content"][0]["type"], "thinking")
        self.assertEqual(first_body["content"][0]["thinking"], "brief reasoning")
        self.assertNotIn(opaque_reasoning, first_body["content"][0]["signature"])
        self.assertEqual(post.call_count, 2)
        replayed_input = post.call_args_list[1].kwargs["json"]["input"]
        self.assertEqual(replayed_input[1], {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "brief reasoning"}],
            "encrypted_content": opaque_reasoning,
        })
        self.assertEqual(replayed_input[2], {
            "type": "message",
            "role": "assistant",
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": "hello back"}],
        })
        self.assertEqual(replayed_input[3], {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "continue"}],
        })


if __name__ == "__main__":
    unittest.main()
