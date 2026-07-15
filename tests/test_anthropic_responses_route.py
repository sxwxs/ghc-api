import base64
import copy
import json
import os
import tempfile
import threading
import time
import unittest
from contextlib import ExitStack
from unittest import mock

from flask import Response

from ghc_api.app import create_app
from ghc_api.cache import RequestCache
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
            "enable_web_search_proxy": anthropic_module.state.enable_web_search_proxy,
            "web_search_proxy_endpoint": anthropic_module.state.web_search_proxy_endpoint,
        }
        anthropic_module.state.anthropic_responses_compat_enabled = True
        anthropic_module.state.enable_auth = False
        anthropic_module.state.enable_web_search_proxy = False
        anthropic_module.state.web_search_proxy_endpoint = "http://127.0.0.1:5002"

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

    def _selection_patches(self, *, direct, responses, translated_model="gpt-5.6-sol"):
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

    def test_responses_branch_receives_unfiltered_request_with_only_model_translated(self):
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
                    side_effect=AssertionError("Responses path ran legacy system filters"),
                )),
                "tool_result": stack.enter_context(mock.patch.object(
                    anthropic_module,
                    "apply_tool_result_suffix_filter_to_payload",
                    side_effect=AssertionError("Responses path ran legacy tool-result filters"),
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
            for filter_mock in filters.values():
                filter_mock.assert_not_called()

        patched["direct"].assert_not_called()
        patched["fallback"].assert_not_called()

    def test_enabled_web_search_proxy_preprocesses_before_responses_routing(self):
        payload = {
            "model": "client-model-alias",
            "messages": [{"role": "user", "content": "长鑫存储"}],
            "tools": [{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 8,
            }],
            "max_tokens": 32,
        }
        proxied = {
            **payload,
            "model": "gpt-5.6-sol",
            "system": [{"type": "text", "text": "[Web Search Results]\nresult"}],
        }
        proxied.pop("tools")
        anthropic_module.state.enable_web_search_proxy = True
        stack, patched = self._selection_patches(direct=False, responses=True)
        with stack, mock.patch.object(
            anthropic_module,
            "apply_web_search_fallback",
            return_value=proxied,
        ) as apply_proxy:
            response = self.client.post("/v1/messages", json=payload)

        self.assertEqual(response.status_code, 200)
        apply_proxy.assert_called_once_with(
            {**payload, "model": "gpt-5.6-sol"},
            "http://127.0.0.1:5002",
        )
        call = patched["responses"].call_args
        self.assertEqual(_call_argument(call, 0, "anthropic_payload"), proxied)
        self.assertEqual(_call_argument(call, 5, "original_request_body"), payload)

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
                "anthropic_responses_compat_mode",
                "anthropic_responses_wire_profile",
                "anthropic_responses_replay_trusted_single_user",
                "anthropic_responses_replay_path",
                "anthropic_responses_replay_ttl_seconds",
                "anthropic_responses_replay_max_bytes",
                "anthropic_responses_replay_max_tenant_bytes",
                "anthropic_responses_replay_max_record_bytes",
                "anthropic_responses_replay_encryption_key_env",
                "anthropic_responses_replay_require_trusted_tenant",
                "enable_auth",
                "enable_web_search_proxy",
                "web_search_proxy_endpoint",
                "max_connection_retries",
                "sse_keepalive_interval",
                "upstream_read_timeout",
            )
        }
        state.anthropic_responses_compat_enabled = True
        state.anthropic_responses_compat_mode = "compatibility"
        state.anthropic_responses_wire_profile = "copilot_responses_lite"
        state.anthropic_responses_replay_trusted_single_user = False
        state.anthropic_responses_replay_encryption_key_env = "GHC_TEST_REPLAY_KEY"
        state.anthropic_responses_replay_max_bytes = 16 * 1024 * 1024
        state.anthropic_responses_replay_max_tenant_bytes = 8 * 1024 * 1024
        state.anthropic_responses_replay_max_record_bytes = 4 * 1024 * 1024
        state.enable_auth = False
        state.enable_web_search_proxy = False
        state.web_search_proxy_endpoint = "http://127.0.0.1:5002"
        state.max_connection_retries = 0
        state.sse_keepalive_interval = 0
        state.upstream_read_timeout = 123

        self._patches = [
            mock.patch.dict(
                os.environ,
                {"GHC_TEST_REPLAY_KEY": base64.urlsafe_b64encode(b"R" * 32).decode("ascii")},
            ),
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

    def test_compatibility_never_creates_plaintext_replay_without_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            replay_path = os.path.join(temp_dir, "must-not-exist.sqlite3")
            anthropic_module.state.anthropic_responses_replay_path = replay_path
            anthropic_module.state.anthropic_responses_replay_trusted_single_user = True
            anthropic_module.state.anthropic_responses_replay_encryption_key_env = (
                "GHC_INTENTIONALLY_MISSING_REPLAY_KEY"
            )
            anthropic_module._reset_anthropic_responses_replay_store()
            try:
                with mock.patch.object(
                    anthropic_module.requests,
                    "post",
                    return_value=_FakeResponse(self._terminal_response()),
                ):
                    response = self.client.post(
                        "/v1/messages",
                        json=self._request_payload(stream=False),
                        headers={
                            "User-Agent": "claude-cli/2.1.207",
                            "Anthropic-Version": "2023-06-01",
                            "X-Claude-Code-Session-Id": "encrypted-only-fixture",
                        },
                    )
                self.assertEqual(response.status_code, 200)
                self.assertIn(
                    "replay.encryption_required",
                    response.headers.get("X-GHC-Compatibility-Warnings", ""),
                )
                self.assertFalse(os.path.exists(replay_path))
            finally:
                anthropic_module._reset_anthropic_responses_replay_store()

    def test_lossless_with_encrypted_sidecar_succeeds(self):
        session_id = "lossless-encrypted-fixture"
        anthropic_module.state.anthropic_responses_compat_mode = "lossless_required"
        anthropic_module.state.anthropic_responses_replay_trusted_single_user = True
        with tempfile.TemporaryDirectory() as temp_dir:
            anthropic_module.state.anthropic_responses_replay_path = os.path.join(
                temp_dir, "lossless.sqlite3"
            )
            anthropic_module._reset_anthropic_responses_replay_store()
            try:
                with mock.patch.object(
                    anthropic_module.requests,
                    "post",
                    return_value=_FakeResponse(self._terminal_response()),
                ) as post:
                    response = self.client.post(
                        "/v1/messages",
                        json=self._request_payload(stream=False),
                        headers={
                            "User-Agent": "claude-cli/2.1.207",
                            "Anthropic-Version": "2023-06-01",
                            "X-Claude-Code-Session-Id": session_id,
                        },
                    )
                self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
                post.assert_called_once()
                store, _ = anthropic_module._get_anthropic_responses_replay_store()
                lookup = store.get(
                    tenant_id="trusted-single-user",
                    session_id=session_id,
                    model="gpt-5.6-sol",
                    assistant_visible_blocks=[
                        {"type": "text", "text": "hello back"}
                    ],
                )
                self.assertEqual(len(lookup.records), 1)
                self.assertTrue(lookup.records[0].encrypted)
            finally:
                anthropic_module._reset_anthropic_responses_replay_store()

    def test_nonstream_unknown_status_fails_closed(self):
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

    def test_nonstream_web_search_proxy_injects_context_before_responses_conversion(self):
        upstream = _FakeResponse(self._terminal_response())
        payload = self._request_payload(stream=False)
        payload["messages"][0]["content"] = "长鑫存储"
        payload["tools"] = [{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 8,
        }]
        anthropic_module.state.enable_web_search_proxy = True

        with mock.patch.object(
            anthropic_module.requests,
            "get",
            return_value=mock.Mock(
                raise_for_status=mock.Mock(),
                json=mock.Mock(return_value={
                    "results": [{
                        "title": "CXMT",
                        "link": "https://example.com/cxmt",
                        "description": "Latest update",
                    }],
                }),
            ),
        ) as get, mock.patch.object(
            anthropic_module.requests,
            "post",
            return_value=upstream,
        ) as post:
            response = self.client.post("/v1/messages", json=payload)

        self.assertEqual(response.status_code, 200)
        get.assert_called_once_with(
            "http://127.0.0.1:5002/search",
            params={"keyword": "长鑫存储", "limit": 3},
            timeout=30,
        )
        outgoing = post.call_args.kwargs["json"]
        self.assertNotIn("tools", outgoing)
        self.assertEqual(outgoing["input"][0]["role"], "developer")
        search_context = outgoing["input"][0]["content"][0]["text"]
        self.assertIn("CXMT", search_context)
        self.assertIn("https://example.com/cxmt", search_context)

        cached = next(iter(self.cache.cache.values()))
        self.assertEqual(cached["original_request_body"], payload)
        self.assertEqual(cached["request_body"], outgoing)

    def test_stream_web_search_proxy_runs_once_before_upstream_stream(self):
        completed = {
            "type": "response.completed",
            "response": self._terminal_response(),
            "sequence_number": 0,
        }
        upstream = _FakeResponse({}, lines=[
            ("data: " + json.dumps(completed, separators=(",", ":"))).encode()
        ])
        payload = self._request_payload(stream=True)
        payload["messages"][0]["content"] = "长鑫存储"
        payload["tools"] = [{
            "type": "web_search_20250305",
            "name": "web_search",
        }]
        anthropic_module.state.enable_web_search_proxy = True

        search_response = mock.Mock(
            raise_for_status=mock.Mock(),
            json=mock.Mock(return_value={
                "results": [{
                    "title": "CXMT",
                    "link": "https://example.com/cxmt",
                    "description": "Latest update",
                }],
            }),
        )
        with mock.patch.object(
            anthropic_module.requests, "get", return_value=search_response
        ) as get, mock.patch.object(
            anthropic_module.requests, "post", return_value=upstream
        ) as post:
            response = self.client.post("/v1/messages", json=payload)
            body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: message_stop\n", body)
        get.assert_called_once()
        post.assert_called_once()
        outgoing = post.call_args.kwargs["json"]
        self.assertTrue(outgoing["stream"])
        self.assertNotIn("tools", outgoing)
        self.assertIn("CXMT", outgoing["input"][0]["content"][0]["text"])

    def test_lossless_mode_accepts_web_search_after_proxy_preprocessing(self):
        upstream = _FakeResponse(self._terminal_response())
        payload = self._request_payload(stream=False)
        payload["tools"] = [{
            "type": "web_search_20250305",
            "name": "web_search",
        }]
        anthropic_module.state.enable_web_search_proxy = True
        anthropic_module.state.anthropic_responses_compat_mode = "lossless_required"
        anthropic_module.state.anthropic_responses_replay_trusted_single_user = True

        with mock.patch.object(
            anthropic_module.requests,
            "get",
            return_value=mock.Mock(
                raise_for_status=mock.Mock(),
                json=mock.Mock(return_value={"results": []}),
            ),
        ), mock.patch.object(
            anthropic_module.requests,
            "post",
            return_value=upstream,
        ) as post:
            response = self.client.post(
                "/v1/messages",
                json=payload,
                headers={
                    "x-session-id": "web-search-lossless-test",
                    "user-agent": "claude-cli/2.1.207 (external, cli)",
                    "anthropic-version": "2023-06-01",
                },
            )

        self.assertEqual(response.status_code, 200)
        post.assert_called_once()
        self.assertNotIn("tools", post.call_args.kwargs["json"])

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
        self.assertEqual(cached["response_body"]["content"], [
            {"type": "text", "text": "hello back"}
        ])
        self.assertIn("stream", cached["conversion_report"])
        self.assertIn("response", cached["conversion_report"])

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

    def test_stream_unknown_responses_event_emits_safe_anthropic_error_and_is_audited(self):
        secret = "DO-NOT-LEAK-unknown-stream-fixture"
        upstream_event = {
            "type": "response.future_private_event",
            "private_value": secret,
        }
        upstream = _FakeResponse({}, lines=[
            ("data: " + json.dumps(upstream_event, separators=(",", ":"))).encode()
        ])
        with mock.patch.object(anthropic_module.requests, "post", return_value=upstream):
            response = self.client.post(
                "/v1/messages", json=self._request_payload(stream=True)
            )
            body = response.get_data(as_text=True)

        # Headers are already committed for an SSE stream, so protocol drift is
        # reported as an Anthropic error event and retained in the cache audit.
        self.assertEqual(response.status_code, 200)
        self.assertIn("event: error\n", body)
        self.assertIn('"type":"error"', body)
        self.assertNotIn("response.future_private_event", body)
        self.assertNotIn(secret, body)
        cached = next(iter(self.cache.cache.values()))
        self.assertEqual(cached["status_code"], 502)
        self.assertEqual(cached["state"], self.cache.STATE_ERROR)
        self.assertFalse(any(secret in item for item in cached["raw_events"]))
        self.assertTrue(any("drifted Responses event" in item for item in cached["raw_events"]))
        warning_codes = {
            item["code"] for item in cached["compatibility_warnings"]
        }
        self.assertIn("responses.unknown_event", warning_codes)
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

    def test_unknown_stream_is_saved_as_non_replayable_full_audit_snapshot(self):
        secret = "OPAQUE-UNKNOWN-EVENT-FIXTURE"
        session_id = "audit-session-fixture"
        event = {
            "type": "response.future_private_event",
            "private_value": secret,
        }
        raw_event = json.dumps(event, separators=(",", ":"))
        upstream = _FakeResponse({}, lines=[("data: " + raw_event).encode()])

        with tempfile.TemporaryDirectory() as temp_dir:
            anthropic_module.state.anthropic_responses_replay_path = os.path.join(
                temp_dir, "audit-replay.sqlite3"
            )
            anthropic_module.state.anthropic_responses_replay_trusted_single_user = True
            anthropic_module._reset_anthropic_responses_replay_store()
            try:
                with mock.patch.object(
                    anthropic_module.requests, "post", return_value=upstream
                ):
                    response = self.client.post(
                        "/v1/messages",
                        json=self._request_payload(stream=True),
                        headers={"X-Claude-Code-Session-Id": session_id},
                    )
                    response.get_data(as_text=True)

                cached = next(iter(self.cache.cache.values()))
                store, _ = anthropic_module._get_anthropic_responses_replay_store()
                lookup = store.get(
                    tenant_id="trusted-single-user",
                    session_id=session_id,
                    model="gpt-5.6-sol",
                    assistant_visible_blocks=[{
                        "type": "ghc_compatibility_audit_record",
                        "request_id": cached["id"],
                    }],
                )
                self.assertEqual(len(lookup.records), 1)
                record = lookup.records[0]
                self.assertTrue(record.profile["audit_only"])
                self.assertEqual(
                    record.profile["audit_snapshot"]["raw_response_events"],
                    [raw_event],
                )
                self.assertEqual(
                    record.profile["audit_snapshot"]["raw_response_sse_lines"],
                    ["data: " + raw_event],
                )
                self.assertIn(
                    secret,
                    record.profile["audit_snapshot"]["raw_response_events"][0],
                )
            finally:
                anthropic_module._reset_anthropic_responses_replay_store()

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

    def test_lossless_mode_rejects_unknown_beta_before_upstream_request(self):
        unknown_beta = "future-private-beta-fixture-value"
        anthropic_module.state.anthropic_responses_compat_mode = "lossless_required"
        replay_context = anthropic_module._ReplayContext(
            mode="lossless_required",
            wire_profile="copilot_responses_lite",
            model="gpt-5.6-sol",
            tenant_id=None,
            session_id=None,
        )
        with mock.patch.object(
            anthropic_module,
            "_create_replay_context",
            return_value=replay_context,
        ), mock.patch.object(
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

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"]["type"], "invalid_request_error")
        self.assertIn(
            "anthropic.beta_unknown",
            response.headers.get("X-GHC-Compatibility-Warnings", ""),
        )
        self.assertNotIn(unknown_beta, response.get_data(as_text=True))
        self.assertNotIn(
            unknown_beta,
            response.headers.get("X-GHC-Compatibility-Warnings", ""),
        )
        post.assert_not_called()

    def test_completed_reasoning_is_written_then_replayed_on_the_next_turn(self):
        session_id = "session-fixture-not-a-real-identity"
        opaque_reasoning = "opaque-encrypted-fixture"
        first_terminal = self._terminal_response()
        first_terminal["output"] = [
            {
                "id": "rs_fixture_1",
                "type": "reasoning",
                "status": "completed",
                "summary": [],
                "content": None,
                "encrypted_content": opaque_reasoning,
            },
            {
                "id": "msg_fixture_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "phase": "final_answer",
                "content": [{
                    "type": "output_text",
                    "text": "hello back",
                    "annotations": [],
                }],
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

        with tempfile.TemporaryDirectory() as temp_dir:
            anthropic_module.state.anthropic_responses_replay_path = os.path.join(
                temp_dir, "reasoning-replay.sqlite3"
            )
            anthropic_module.state.anthropic_responses_replay_trusted_single_user = True
            reset_store = getattr(
                anthropic_module, "_reset_anthropic_responses_replay_store"
            )
            reset_store()
            try:
                with mock.patch.object(
                    anthropic_module.requests,
                    "post",
                    side_effect=[
                        _FakeResponse(first_terminal),
                        _FakeResponse(second_terminal),
                    ],
                ) as post:
                    first_response = self.client.post(
                        "/v1/messages",
                        json=self._request_payload(stream=False),
                        headers={"X-Claude-Code-Session-Id": session_id},
                    )
                    replay_store, _ = anthropic_module._get_anthropic_responses_replay_store()
                    stored = replay_store.get(
                        tenant_id="trusted-single-user",
                        session_id=session_id,
                        model="gpt-5.6-sol",
                        assistant_visible_blocks=[
                            {"type": "text", "text": "hello back"}
                        ],
                    )
                    self.assertEqual(len(stored.records), 1)
                    audit_snapshot = stored.records[0].profile["audit_snapshot"]
                    raw_request = base64.b64decode(
                        audit_snapshot["request_raw_base64"]
                    )
                    self.assertEqual(json.loads(raw_request), self._request_payload(False))
                    self.assertEqual(
                        audit_snapshot["parsed_request"], self._request_payload(False)
                    )
                    self.assertEqual(
                        json.loads(base64.b64decode(
                            audit_snapshot["upstream_response_raw_base64"]
                        )),
                        first_terminal,
                    )
                    continuation = self._request_payload(stream=False)
                    continuation["messages"] = [
                        {"role": "user", "content": "hello"},
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "hello back"}],
                        },
                        {"role": "user", "content": "continue"},
                    ]
                    second_response = self.client.post(
                        "/v1/messages",
                        json=continuation,
                        headers={"X-Claude-Code-Session-Id": session_id},
                    )
            finally:
                reset_store()

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 200)
        self.assertNotIn(opaque_reasoning, first_response.get_data(as_text=True))
        self.assertEqual(post.call_count, 2)
        replayed_input = post.call_args_list[1].kwargs["json"]["input"]
        self.assertEqual(replayed_input[1], {
            "type": "reasoning",
            "summary": [],
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
