import contextlib
import json
import queue
import unittest
from unittest import mock

import requests

from ghc_api.app import create_app
from ghc_api.cache import cache
from ghc_api.counters import counters
from ghc_api.state import State, state
from ghc_api.utils import (
    is_encrypted_content_parse_error,
    remove_encrypted_content_items,
)


LEGACY_ERROR = {
    "error": {
        "message": (
            "The encrypted content abc could not be verified. Reason: "
            "Encrypted content could not be decrypted or parsed."
        ),
        "code": "invalid_request_body",
    }
}

FUNCTION_OUTPUT_ERROR = {
    "error": {
        "message": "Encrypted function output content could not be decrypted or decoded.",
        "code": "invalid_request_body",
    }
}

# Same failure with a different code/wording. The matcher is intentionally strict, so
# these are documented as *not* recovered.
FUNCTION_OUTPUT_ERROR_VARIANT = {
    "error": {
        "message": (
            "Encrypted function output content for item fc_123 could not be "
            "decrypted or decoded."
        ),
        "code": "invalid_request_error",
    }
}


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.body = body
        self.text = json.dumps(body)
        self.ok = 200 <= status_code < 300
        self.closed = False

    def json(self):
        return self.body

    def close(self):
        self.closed = True


class FakeStreamResponse(FakeResponse):
    def __init__(self, events):
        super().__init__(200, {})
        self.events = events

    def iter_lines(self):
        for event in self.events:
            yield f"event: {event['type']}".encode()
            yield f"data: {json.dumps(event)}".encode()
            yield b""
        yield b"data: [DONE]"


class EncryptedContentHelpersTest(unittest.TestCase):
    def test_recovery_is_disabled_by_default(self):
        self.assertFalse(State().auto_remove_encrypted_content_on_parse_error)

    def test_matches_legacy_and_function_output_errors(self):
        self.assertTrue(is_encrypted_content_parse_error(400, json.dumps(LEGACY_ERROR)))
        self.assertTrue(
            is_encrypted_content_parse_error(400, json.dumps(FUNCTION_OUTPUT_ERROR))
        )

    def test_message_and_code_variations_are_not_matched(self):
        # The matcher requires the exact upstream wording and code; anything else is
        # returned to the client untouched.
        self.assertFalse(
            is_encrypted_content_parse_error(
                400, json.dumps(FUNCTION_OUTPUT_ERROR_VARIANT)
            )
        )
        self.assertFalse(
            is_encrypted_content_parse_error(
                400,
                json.dumps(
                    {"message": "Encrypted content could not be decrypted or parsed."}
                ),
            )
        )
        self.assertFalse(
            is_encrypted_content_parse_error(
                400, "Encrypted content could not be decrypted or parsed."
            )
        )

    def test_rejects_unrelated_errors(self):
        self.assertFalse(
            is_encrypted_content_parse_error(500, json.dumps(FUNCTION_OUTPUT_ERROR))
        )
        self.assertFalse(
            is_encrypted_content_parse_error(
                400,
                json.dumps({"error": {"message": "bad input", "code": "invalid_request_body"}}),
            )
        )
        self.assertFalse(is_encrypted_content_parse_error(400, "not json"))
        # Mentions encryption but is a different failure mode.
        self.assertFalse(
            is_encrypted_content_parse_error(
                400,
                json.dumps(
                    {
                        "error": {
                            "message": "Encrypted content is too large.",
                            "code": "invalid_request_body",
                        }
                    }
                ),
            )
        )

    def test_removes_items_with_direct_or_nested_encrypted_content(self):
        request_input = [
            {"type": "message", "content": [{"type": "input_text", "text": "keep"}]},
            {"type": "reasoning", "encrypted_content": "bad", "summary": []},
            {
                "type": "agent_message",
                "content": [
                    {"type": "input_text", "text": "nested"},
                    {"type": "encrypted_content", "encrypted_content": "bad"},
                ],
            },
        ]

        cleaned, removed_count = remove_encrypted_content_items(request_input)

        self.assertEqual(cleaned, [request_input[0]])
        self.assertEqual(removed_count, 2)

    def test_function_call_output_is_sanitized_not_dropped(self):
        request_input = [
            {"type": "function_call", "call_id": "call_1", "name": "ls", "arguments": "{}"},
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [
                    {"type": "output_text", "text": "keep me"},
                    {"type": "encrypted_content", "encrypted_content": "bad"},
                ],
            },
        ]

        cleaned, changed_count = remove_encrypted_content_items(request_input)

        self.assertEqual(changed_count, 1)
        self.assertEqual(len(cleaned), 2, "function_call must keep its paired output")
        self.assertEqual(cleaned[0], request_input[0])
        self.assertEqual(
            cleaned[1],
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [{"type": "output_text", "text": "keep me"}],
            },
        )

    def test_emptied_function_call_output_gets_placeholder(self):
        request_input = [
            {"type": "function_call", "call_id": "call_1", "name": "ls", "arguments": "{}"},
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "",
                "encrypted_content": "bad",
            },
        ]

        cleaned, changed_count = remove_encrypted_content_items(request_input)

        self.assertEqual(changed_count, 1)
        self.assertEqual(len(cleaned), 2)
        self.assertNotIn("encrypted_content", cleaned[1])
        self.assertTrue(cleaned[1]["output"].startswith("[ghc-api]"))

    def test_dropping_a_tool_call_also_drops_its_output(self):
        request_input = [
            {"type": "message", "role": "user", "content": "hi"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "ls",
                "arguments": "{}",
                "encrypted_content": "bad",
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
            {"type": "function_call", "call_id": "call_2", "name": "ls", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_2", "output": "ok"},
        ]

        cleaned, changed_count = remove_encrypted_content_items(request_input)

        self.assertEqual(changed_count, 2)
        self.assertEqual(
            cleaned, [request_input[0], request_input[3], request_input[4]]
        )

    def test_deeply_nested_payload_does_not_recurse_forever(self):
        item = {"type": "message"}
        cursor = item
        for _ in range(200):
            child = {}
            cursor["content"] = [child]
            cursor = child

        cleaned, changed_count = remove_encrypted_content_items([item])

        self.assertEqual(changed_count, 0)
        self.assertEqual(cleaned, [item])

    def test_non_list_input_is_returned_untouched(self):
        self.assertEqual(remove_encrypted_content_items("hello"), ("hello", 0))


class EncryptedContentTestCase(unittest.TestCase):
    def setUp(self):
        self.saved_models = state.models
        self.saved_auto_remove = state.auto_remove_encrypted_content_on_parse_error
        self.saved_connection_retries = state.max_connection_retries
        self.saved_enable_auth = state.enable_auth
        self.saved_early_retry = state.enable_responses_early_failure_retry
        self.saved_keepalive = state.sse_keepalive_interval
        self.saved_grace = state.responses_pre_header_grace
        state.models = {
            "data": [{"id": "gpt-test", "supported_endpoints": ["/responses"]}]
        }
        state.auto_remove_encrypted_content_on_parse_error = True
        state.max_connection_retries = 0
        state.enable_auth = False
        state.enable_responses_early_failure_retry = False
        state.sse_keepalive_interval = 0
        cache.cache.clear()
        counters.reset()

    def tearDown(self):
        state.models = self.saved_models
        state.auto_remove_encrypted_content_on_parse_error = self.saved_auto_remove
        state.max_connection_retries = self.saved_connection_retries
        state.enable_auth = self.saved_enable_auth
        state.enable_responses_early_failure_retry = self.saved_early_retry
        state.sse_keepalive_interval = self.saved_keepalive
        state.responses_pre_header_grace = self.saved_grace
        cache.cache.clear()
        counters.reset()

    @staticmethod
    def _patched_upstream(upstream_responses):
        """Patch the upstream call stack (3.8-compatible, no parenthesized `with`)."""
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch("ghc_api.routes.openai.ensure_copilot_token"))
        stack.enter_context(
            mock.patch("ghc_api.routes.openai.get_copilot_headers", return_value={})
        )
        stack.enter_context(mock.patch("ghc_api.routes.openai.log_error_request"))
        post = stack.enter_context(
            mock.patch(
                "ghc_api.routes.openai.requests.post", side_effect=upstream_responses
            )
        )
        return stack, post


class EncryptedContentRouteRetryTest(EncryptedContentTestCase):
    def test_retries_nested_function_output_error_when_connection_retries_disabled(self):
        error_response = FakeResponse(400, FUNCTION_OUTPUT_ERROR)
        upstream_responses = [
            error_response,
            FakeResponse(
                200,
                {
                    "id": "resp-1",
                    "output": [],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            ),
        ]
        payload = {
            "model": "gpt-test",
            "input": [
                {"type": "message", "role": "user", "content": "keep"},
                {
                    "type": "agent_message",
                    "content": [
                        {"type": "input_text", "text": "tool output"},
                        {"type": "encrypted_content", "encrypted_content": "bad"},
                    ],
                },
            ],
        }

        app = create_app()
        stack, post = self._patched_upstream(upstream_responses)
        with stack:
            response = app.test_client().post("/v1/responses", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args_list[1][1]["json"]["input"], [payload["input"][0]])
        self.assertEqual(counters.snapshot()["mod.encrypted_content_removal"], 1)
        self.assertTrue(error_response.closed, "failed upstream response must be closed")

    def test_retry_keeps_tool_call_pairing(self):
        upstream_responses = [
            FakeResponse(400, FUNCTION_OUTPUT_ERROR),
            FakeResponse(
                200,
                {
                    "id": "resp-1",
                    "output": [],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            ),
        ]
        payload = {
            "model": "gpt-test",
            "input": [
                {"type": "message", "role": "user", "content": "keep"},
                {"type": "function_call", "call_id": "call_1", "name": "ls", "arguments": "{}"},
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [{"type": "encrypted_content", "encrypted_content": "bad"}],
                },
            ],
        }

        app = create_app()
        stack, post = self._patched_upstream(upstream_responses)
        with stack:
            response = app.test_client().post("/v1/responses", json=payload)

        self.assertEqual(response.status_code, 200)
        retried_input = post.call_args_list[1][1]["json"]["input"]
        self.assertEqual([item["type"] for item in retried_input],
                         ["message", "function_call", "function_call_output"])
        self.assertEqual(retried_input[2]["call_id"], "call_1")
        self.assertNotIn("encrypted_content", json.dumps(retried_input))

    def test_retry_happens_at_most_once(self):
        upstream_responses = [
            FakeResponse(400, FUNCTION_OUTPUT_ERROR),
            FakeResponse(400, FUNCTION_OUTPUT_ERROR),
        ]
        payload = {
            "model": "gpt-test",
            "input": [
                {"type": "message", "role": "user", "content": "keep"},
                {"type": "reasoning", "encrypted_content": "bad", "summary": []},
            ],
        }

        app = create_app()
        stack, post = self._patched_upstream(upstream_responses)
        with stack:
            response = app.test_client().post("/v1/responses", json=payload)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(post.call_count, 2)

    def test_no_retry_when_disabled(self):
        state.auto_remove_encrypted_content_on_parse_error = False
        upstream_responses = [FakeResponse(400, FUNCTION_OUTPUT_ERROR)]
        payload = {
            "model": "gpt-test",
            "input": [{"type": "reasoning", "encrypted_content": "bad", "summary": []}],
        }

        app = create_app()
        stack, post = self._patched_upstream(upstream_responses)
        with stack:
            response = app.test_client().post("/v1/responses", json=payload)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(post.call_count, 1)


class EncryptedContentStreamRetryTest(EncryptedContentTestCase):
    """Reproduce an HTTP 200 rejection with the observed trailing diagnostic."""

    @staticmethod
    def _payload():
        return {
            "model": "gpt-test", "stream": True,
            "input": [
                {"type": "message", "role": "user", "content": "keep"},
                {"type": "function_call", "call_id": "call_1", "name": "ls", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": [
                    {"type": "output_text", "text": "visible tool result"},
                    {"type": "encrypted_content", "encrypted_content": "bad"},
                ]},
            ],
        }

    @staticmethod
    def _failure(error=None, shape="trailing"):
        error = FUNCTION_OUTPUT_ERROR["error"] if error is None else error
        events = [{"type": "response.created", "response": {"status": "in_progress"}}]
        if shape != "standalone":
            events.append({"type": "response.failed", "response": {
                "status": "failed", "output": [], "usage": None,
                "error": error if shape == "nested" else None,
            }})
        if shape != "nested":
            events.append({"type": "error", **error})
        return FakeStreamResponse(events)

    @staticmethod
    def _success():
        return FakeStreamResponse([
            {"type": "response.output_text.delta", "delta": "recovered"},
            {"type": "response.completed", "response": {
                "status": "completed", "usage": {"input_tokens": 10, "output_tokens": 2},
            }},
        ])

    def _run_stream(self, upstream, payload=None, pending=False):
        stack, post = self._patched_upstream(upstream)
        if pending:
            state.sse_keepalive_interval = 1
            state.responses_pre_header_grace = 0

            def delayed_headers(headers, payload):
                response = post(json=payload, headers=headers, stream=True)
                result = mock.Mock()
                result.get.side_effect = [queue.Empty(), response]
                return result

            stack.enter_context(mock.patch(
                "ghc_api.routes.openai._start_responses_post", side_effect=delayed_headers,
            ))
        with stack:
            response = create_app().test_client().post(
                "/v1/responses", json=payload or self._payload(), buffered=True,
            )
        return response, post

    def test_recovers_stream_error_on_direct_immediate_and_pending_paths(self):
        for path in ("direct", "immediate", "pending"):
            for shape in ("trailing", "nested", "standalone"):
                with self.subTest(path=path, shape=shape):
                    cache.cache.clear()
                    counters.reset()
                    state.sse_keepalive_interval = 0 if path == "direct" else 1
                    state.responses_pre_header_grace = 1
                    failure, success = self._failure(shape=shape), self._success()
                    response, post = self._run_stream([failure, success], pending=path == "pending")

                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(post.call_count, 2)
                    self.assertIn(b"recovered", response.data)
                    self.assertNotIn(b"response.failed", response.data)
                    self.assertNotIn(b"invalid_request_body", response.data)
                    sent = post.call_args_list[1].kwargs["json"]
                    self.assertEqual(sent["input"][1], self._payload()["input"][1])
                    self.assertEqual(sent["input"][2], {
                        "type": "function_call_output", "call_id": "call_1",
                        "output": [{"type": "output_text", "text": "visible tool result"}],
                    })
                    self.assertIn("encrypted_content", json.dumps(post.call_args_list[0].kwargs["json"]))
                    self.assertTrue(failure.closed)
                    self.assertTrue(success.closed)
                    record = cache.get_recent_requests(1)[0]
                    self.assertEqual(record["status_code"], 200)
                    self.assertEqual(record["input_tokens"], 10)
                    self.assertEqual(record["request_body"], sent)
                    self.assertEqual(record["request_size"], len(json.dumps(sent)))
                    self.assertEqual(record["original_request_body"], self._payload())
                    self.assertEqual(counters.snapshot()["mod.encrypted_content_removal"], 1)

    def test_invalid_input_is_not_replayed_with_recovery_disabled_or_inapplicable(self):
        state.enable_responses_early_failure_retry = True
        state.max_connection_retries = 3
        for case in ("disabled", "no-encrypted-input", "unrelated-error"):
            with self.subTest(case=case):
                state.auto_remove_encrypted_content_on_parse_error = case != "disabled"
                payload = self._payload()
                error = None
                if case == "no-encrypted-input":
                    payload["input"] = payload["input"][:1]
                if case == "unrelated-error":
                    error = {"code": "invalid_request_body", "message": "bad tool schema"}
                response, post = self._run_stream([self._failure(error)], payload)
                self.assertEqual(post.call_count, 1)
                self.assertIn(b"invalid_request_body", response.data)
                self.assertEqual(cache.get_recent_requests(1)[0]["status_code"], 502)

    def test_recovery_happens_only_once_even_with_generic_retries_enabled(self):
        state.enable_responses_early_failure_retry = True
        state.max_connection_retries = 3
        response, post = self._run_stream([self._failure(), self._failure()])
        self.assertEqual(post.call_count, 2)
        self.assertIn(b"Encrypted function output", response.data)
        self.assertEqual(cache.get_recent_requests(1)[0]["status_code"], 502)

    def test_encrypted_recovery_and_transient_retries_have_independent_budgets(self):
        state.enable_responses_early_failure_retry = True
        state.max_connection_retries = 1
        for recovery_first in (False, True):
            with self.subTest(recovery_first=recovery_first):
                transient = FakeStreamResponse([
                    {"type": "response.failed", "response": {"error": None}},
                ])
                failures = [self._failure(), transient] if recovery_first else [transient, self._failure()]
                response, post = self._run_stream([*failures, self._success()])
                self.assertEqual(post.call_count, 3)
                self.assertIn(b"recovered", response.data)
                self.assertNotIn("encrypted_content", json.dumps(post.call_args_list[-1].kwargs["json"]))

    def test_http_recovery_cannot_be_repeated_by_stream_recovery(self):
        for pending in (False, True):
            with self.subTest(pending=pending):
                counters.reset()
                response, post = self._run_stream(
                    [FakeResponse(400, FUNCTION_OUTPUT_ERROR), self._failure()], pending=pending,
                )
                self.assertEqual(post.call_count, 2)
                self.assertIn(b"Encrypted function output", response.data)
                self.assertEqual(counters.snapshot()["mod.encrypted_content_removal"], 1)

    def test_recovery_never_replays_partial_output(self):
        for output in (
            {"type": "response.output_text.delta", "delta": "partial"},
            {"type": "response.output_item.added", "output_index": 0,
             "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1"}},
        ):
            with self.subTest(output=output["type"]):
                failure = self._failure()
                failure.events.insert(1, output)
                response, post = self._run_stream([failure])
                self.assertEqual(post.call_count, 1)
                self.assertIn(b"Encrypted function output", response.data)

    def test_failed_recovery_preserves_the_original_diagnostic(self):
        for retry in (FakeResponse(503, {"error": "unavailable"}), requests.ConnectionError("offline")):
            with self.subTest(retry=type(retry).__name__):
                failure = self._failure()
                response, post = self._run_stream([failure, retry])
                self.assertEqual(post.call_count, 2)
                self.assertIn(b"Encrypted function output", response.data)
                self.assertEqual(cache.get_recent_requests(1)[0]["status_code"], 502)
                self.assertTrue(failure.closed)
                if isinstance(retry, FakeResponse):
                    self.assertTrue(retry.closed)


if __name__ == "__main__":
    unittest.main()
