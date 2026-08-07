import contextlib
import json
import unittest
from unittest import mock

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


class EncryptedContentRouteRetryTest(unittest.TestCase):
    def setUp(self):
        self.saved_models = state.models
        self.saved_auto_remove = state.auto_remove_encrypted_content_on_parse_error
        self.saved_connection_retries = state.max_connection_retries
        self.saved_enable_auth = state.enable_auth
        state.models = {
            "data": [{"id": "gpt-test", "supported_endpoints": ["/responses"]}]
        }
        state.auto_remove_encrypted_content_on_parse_error = True
        state.max_connection_retries = 0
        state.enable_auth = False
        cache.cache.clear()
        counters.reset()

    def tearDown(self):
        state.models = self.saved_models
        state.auto_remove_encrypted_content_on_parse_error = self.saved_auto_remove
        state.max_connection_retries = self.saved_connection_retries
        state.enable_auth = self.saved_enable_auth
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


if __name__ == "__main__":
    unittest.main()
