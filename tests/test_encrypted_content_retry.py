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


class EncryptedContentHelpersTest(unittest.TestCase):
    def test_recovery_is_enabled_by_default(self):
        self.assertTrue(State().auto_remove_encrypted_content_on_parse_error)

    def test_matches_legacy_and_function_output_errors(self):
        self.assertTrue(is_encrypted_content_parse_error(400, json.dumps(LEGACY_ERROR)))
        self.assertTrue(
            is_encrypted_content_parse_error(400, json.dumps(FUNCTION_OUTPUT_ERROR))
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

    def test_retries_nested_function_output_error_when_connection_retries_disabled(self):
        class FakeResponse:
            def __init__(self, status_code, body):
                self.status_code = status_code
                self.body = body
                self.text = json.dumps(body)
                self.ok = 200 <= status_code < 300

            def json(self):
                return self.body

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
        with (
            mock.patch("ghc_api.routes.openai.ensure_copilot_token"),
            mock.patch("ghc_api.routes.openai.get_copilot_headers", return_value={}),
            mock.patch("ghc_api.routes.openai.log_error_request"),
            mock.patch(
                "ghc_api.routes.openai.requests.post", side_effect=upstream_responses
            ) as post,
        ):
            response = app.test_client().post("/v1/responses", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args_list[1].kwargs["json"]["input"], [payload["input"][0]])
        self.assertEqual(counters.snapshot()["mod.encrypted_content_removal"], 1)


if __name__ == "__main__":
    unittest.main()
