"""The early-stream-failure retry is enabled by default but remains configurable.

When ``state.enable_responses_early_failure_retry`` is off, the retry wrapper
is never constructed, so the upstream ``requests.Response`` reaches the stream
handler untouched and the feature is a true no-op.
"""

import unittest
from unittest import mock

import ghc_api.state
from ghc_api.app import create_app
from ghc_api.routes import openai as openai_routes
from ghc_api.sse import RetryingResponsesResponse


class _FakeUpstreamResponse:
    status_code = 200
    ok = True
    text = ""

    def iter_lines(self):
        return iter(())

    def close(self):
        pass


class ResponsesEarlyFailureRetryToggleTest(unittest.TestCase):
    def setUp(self):
        self.state = ghc_api.state.state
        self._saved_flag = self.state.enable_responses_early_failure_retry
        self._saved_models = self.state.models
        self.state.models = {"data": [
            {"id": "gpt-5", "supported_endpoints": ["/responses"]},
        ]}
        self.app = create_app()

    def tearDown(self):
        self.state.enable_responses_early_failure_retry = self._saved_flag
        self.state.models = self._saved_models

    def _post_streaming_request(self):
        """Drive /v1/responses and return the ``response`` the handler received."""
        captured = {}

        def fake_handler(**kwargs):
            captured["response"] = kwargs["response"]
            handler = mock.Mock()
            handler.stream.return_value = ("", 200)
            return handler

        with mock.patch.object(openai_routes, "ensure_copilot_token"), \
                mock.patch.object(openai_routes, "get_copilot_headers", return_value={}), \
                mock.patch.object(openai_routes, "get_copilot_base_url", return_value="https://upstream.test"), \
                mock.patch.object(openai_routes.requests, "post", return_value=_FakeUpstreamResponse()), \
                mock.patch.object(openai_routes, "OpenAIResponsesStreamHandler", side_effect=fake_handler):
            with self.app.test_client() as client:
                client.post("/v1/responses", json={
                    "model": "gpt-5",
                    "stream": True,
                    "input": [],
                })

        return captured.get("response")

    def test_enabled_by_default(self):
        self.assertTrue(ghc_api.state.State().enable_responses_early_failure_retry)

    def test_upstream_response_is_untouched_when_disabled(self):
        self.state.enable_responses_early_failure_retry = False

        forwarded = self._post_streaming_request()

        self.assertIsInstance(forwarded, _FakeUpstreamResponse)
        self.assertNotIsInstance(forwarded, RetryingResponsesResponse)

    def test_upstream_response_is_wrapped_when_enabled(self):
        self.state.enable_responses_early_failure_retry = True

        forwarded = self._post_streaming_request()

        self.assertIsInstance(forwarded, RetryingResponsesResponse)


if __name__ == "__main__":
    unittest.main()
