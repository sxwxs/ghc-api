import json
import time
import unittest
from unittest import mock

import requests

from ghc_api.app import create_app
from ghc_api.cache import RequestCache
from ghc_api.routes import openai as openai_routes
from ghc_api.sse import base as base_module
from ghc_api.state import state


class _FakeStreamResponse:
    status_code = 200
    ok = True
    text = ""

    def __init__(self, lines):
        self._lines = lines
        self.closed = False

    def iter_lines(self):
        yield from self._lines

    def close(self):
        self.closed = True


class ResponsesPreResponseKeepaliveTest(unittest.TestCase):
    def setUp(self):
        self.saved_models = state.models
        self.saved_retries = state.enable_responses_early_failure_retry
        self.saved_connection_retries = state.max_connection_retries
        state.models = {"data": [
            {"id": "gpt-5", "supported_endpoints": ["/responses"]},
        ]}
        state.enable_responses_early_failure_retry = False
        self.cache = RequestCache()
        self.app = create_app()

    def tearDown(self):
        state.models = self.saved_models
        state.enable_responses_early_failure_retry = self.saved_retries
        state.max_connection_retries = self.saved_connection_retries

    def test_stream_sends_headers_and_keepalive_before_upstream_headers(self):
        completed = json.dumps({
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {"input_tokens": 2, "output_tokens": 1},
            },
        })

        def slow_post(*args, **kwargs):
            time.sleep(0.3)
            return _FakeStreamResponse([
                b"event: response.completed",
                f"data: {completed}".encode(),
                b"data: [DONE]",
            ])

        patches = [
            mock.patch.object(openai_routes, "cache", self.cache),
            mock.patch.object(base_module, "cache", self.cache),
            mock.patch.object(openai_routes, "ensure_copilot_token"),
            mock.patch.object(openai_routes, "get_copilot_headers", return_value={}),
            mock.patch.object(openai_routes, "get_copilot_base_url", return_value="https://upstream.test"),
            mock.patch.object(openai_routes.requests, "post", side_effect=slow_post),
        ]

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            with self.app.test_client() as client:
                started = time.monotonic()
                response = client.post(
                    "/v1/responses",
                    json={"model": "gpt-5", "stream": True, "input": []},
                    buffered=False,
                )
                first_chunk = next(response.response)
                first_chunk = (
                    first_chunk.decode("utf-8")
                    if isinstance(first_chunk, bytes)
                    else first_chunk
                )
                first_chunk_elapsed = time.monotonic() - started
                remaining = "".join(
                    chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
                    for chunk in response.response
                )

        self.assertEqual(first_chunk, ": keepalive\n\n")
        self.assertLess(first_chunk_elapsed, 0.2)
        self.assertIn("event: response.completed\n", remaining)
        self.assertIn("data: [DONE]\n\n", remaining)
        entry = next(iter(self.cache.cache.values()))
        self.assertEqual(entry["status_code"], 200)
        self.assertEqual(entry["state"], RequestCache.STATE_COMPLETED)

    def test_fast_connection_failure_uses_streaming_504_path(self):
        state.max_connection_retries = 0
        connection_error = requests.exceptions.ConnectionError("upstream unavailable")

        patches = [
            mock.patch.object(openai_routes, "cache", self.cache),
            mock.patch.object(base_module, "cache", self.cache),
            mock.patch.object(openai_routes, "ensure_copilot_token"),
            mock.patch.object(openai_routes, "get_copilot_headers", return_value={}),
            mock.patch.object(openai_routes, "get_copilot_base_url", return_value="https://upstream.test"),
            mock.patch.object(openai_routes, "log_connection_retry"),
            mock.patch.object(openai_routes.requests, "post", side_effect=connection_error),
        ]

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            with self.app.test_client() as client:
                response = client.post(
                    "/v1/responses",
                    json={"model": "gpt-5", "stream": True, "input": []},
                    buffered=True,
                )

        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(body.startswith(": keepalive\n\n"))
        self.assertIn("event: response.failed\n", body)
        self.assertIn('"code": "upstream_connection_error"', body)
        entry = next(iter(self.cache.cache.values()))
        self.assertEqual(entry["status_code"], 504)
        self.assertEqual(entry["state"], RequestCache.STATE_ERROR)


if __name__ == "__main__":
    unittest.main()
