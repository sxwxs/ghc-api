import json
import threading
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
    def __init__(self, lines=(), status_code=200, text="", close_event=None):
        self._lines = lines
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = text
        self._close_event = close_event
        self.closed = False

    def iter_lines(self):
        yield from self._lines

    def close(self):
        self.closed = True
        if self._close_event is not None:
            self._close_event.set()

    def json(self):
        return json.loads(self.text)


class _CompletedResult:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def get(self, timeout=None):
        if self.error is not None:
            raise self.error
        return self.result

    def cancel(self):
        if self.result is not None:
            self.result.close()


class ResponsesPreResponseKeepaliveTest(unittest.TestCase):
    def setUp(self):
        self.saved_models = state.models
        self.saved_retries = state.enable_responses_early_failure_retry
        self.saved_connection_retries = state.max_connection_retries
        self.saved_auto_remove = state.auto_remove_encrypted_content_on_parse_error
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
        state.auto_remove_encrypted_content_on_parse_error = self.saved_auto_remove

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
            mock.patch.object(
                openai_routes,
                "_start_responses_post",
                return_value=_CompletedResult(error=connection_error),
            ),
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

    def test_fast_unexpected_failure_uses_streaming_500_path(self):
        patches = [
            mock.patch.object(openai_routes, "cache", self.cache),
            mock.patch.object(openai_routes, "ensure_copilot_token"),
            mock.patch.object(openai_routes, "get_copilot_headers", return_value={}),
            mock.patch.object(openai_routes, "get_copilot_base_url", return_value="https://upstream.test"),
            mock.patch.object(
                openai_routes,
                "_start_responses_post",
                return_value=_CompletedResult(error=ValueError("unexpected")),
            ),
        ]

        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            with self.app.test_client() as client:
                response = client.post(
                    "/v1/responses",
                    json={"model": "gpt-5", "stream": True, "input": []},
                    buffered=True,
                )

        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('"code": "proxy_error"', body)
        entry = next(iter(self.cache.cache.values()))
        self.assertEqual(entry["status_code"], 500)

    def test_fast_upstream_error_preserves_http_status_and_body(self):
        error_body = json.dumps({"error": {"message": "rate limited"}})
        upstream = _FakeStreamResponse(status_code=429, text=error_body)

        patches = [
            mock.patch.object(openai_routes, "cache", self.cache),
            mock.patch.object(openai_routes, "ensure_copilot_token"),
            mock.patch.object(openai_routes, "get_copilot_headers", return_value={}),
            mock.patch.object(openai_routes, "get_copilot_base_url", return_value="https://upstream.test"),
            mock.patch.object(openai_routes, "log_error_request"),
            mock.patch.object(
                openai_routes,
                "_start_responses_post",
                return_value=_CompletedResult(result=upstream),
            ),
        ]

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            with self.app.test_client() as client:
                response = client.post(
                    "/v1/responses",
                    json={"model": "gpt-5", "stream": True, "input": []},
                )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.get_data(as_text=True), error_body)
        self.assertNotIn(": keepalive", response.get_data(as_text=True))
        entry = next(iter(self.cache.cache.values()))
        self.assertEqual(entry["status_code"], 429)

    def test_fast_encrypted_content_error_retries_into_stream(self):
        state.auto_remove_encrypted_content_on_parse_error = True
        error_body = json.dumps({
            "error": {
                "message": "Encrypted function output content could not be decrypted or decoded.",
                "code": "invalid_request_body",
            },
        })
        failed = _FakeStreamResponse(status_code=400, text=error_body)
        completed = json.dumps({
            "type": "response.completed",
            "response": {"status": "completed", "usage": {}},
        })
        succeeded = _FakeStreamResponse(lines=[
            b"event: response.completed",
            f"data: {completed}".encode(),
            b"data: [DONE]",
        ])
        payload = {
            "model": "gpt-5",
            "stream": True,
            "input": [
                {"type": "message", "role": "user", "content": "keep"},
                {"type": "encrypted_content", "encrypted_content": "bad"},
            ],
        }

        patches = [
            mock.patch.object(openai_routes, "cache", self.cache),
            mock.patch.object(base_module, "cache", self.cache),
            mock.patch.object(openai_routes, "ensure_copilot_token"),
            mock.patch.object(openai_routes, "get_copilot_headers", return_value={}),
            mock.patch.object(openai_routes, "get_copilot_base_url", return_value="https://upstream.test"),
            mock.patch.object(openai_routes, "log_error_request"),
            mock.patch.object(
                openai_routes,
                "_start_responses_post",
                return_value=_CompletedResult(result=failed),
            ),
            mock.patch.object(openai_routes.requests, "post", return_value=succeeded),
        ]

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7] as post:
            with self.app.test_client() as client:
                response = client.post("/v1/responses", json=payload)
                body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: response.completed\n", body)
        self.assertTrue(failed.closed)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(post.call_args.kwargs["json"]["input"], [payload["input"][0]])
        entry = next(iter(self.cache.cache.values()))
        self.assertEqual(entry["status_code"], 200)

    def test_fast_connection_failure_retries_successfully(self):
        state.max_connection_retries = 1
        completed = json.dumps({
            "type": "response.completed",
            "response": {"status": "completed", "usage": {}},
        })
        succeeded = _FakeStreamResponse(lines=[
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
            mock.patch.object(openai_routes, "log_connection_retry"),
            mock.patch.object(openai_routes.time, "sleep"),
            mock.patch.object(
                openai_routes,
                "_start_responses_post",
                side_effect=[
                    _CompletedResult(error=requests.exceptions.ConnectionError("first")),
                    _CompletedResult(result=succeeded),
                ],
            ),
        ]

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7] as start_post:
            with self.app.test_client() as client:
                response = client.post(
                    "/v1/responses",
                    json={"model": "gpt-5", "stream": True, "input": []},
                )
                body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: response.completed\n", body)
        self.assertEqual(start_post.call_count, 2)
        entry = next(iter(self.cache.cache.values()))
        self.assertEqual(entry["status_code"], 200)

    def test_disconnect_closes_eventual_upstream_response(self):
        release_upstream = threading.Event()
        response_closed = threading.Event()
        upstream = _FakeStreamResponse(close_event=response_closed)

        def delayed_post(*args, **kwargs):
            release_upstream.wait(2)
            return upstream

        patches = [
            mock.patch.object(openai_routes, "cache", self.cache),
            mock.patch.object(base_module, "cache", self.cache),
            mock.patch.object(openai_routes, "ensure_copilot_token"),
            mock.patch.object(openai_routes, "get_copilot_headers", return_value={}),
            mock.patch.object(openai_routes, "get_copilot_base_url", return_value="https://upstream.test"),
            mock.patch.object(openai_routes.requests, "post", side_effect=delayed_post),
        ]

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            with self.app.test_client() as client:
                response = client.post(
                    "/v1/responses",
                    json={"model": "gpt-5", "stream": True, "input": []},
                    buffered=False,
                )
                first_chunk = next(response.response)
                if isinstance(first_chunk, bytes):
                    first_chunk = first_chunk.decode("utf-8")
                self.assertEqual(first_chunk, ": keepalive\n\n")
                response.close()
                release_upstream.set()
                self.assertTrue(response_closed.wait(1))

        entry = next(iter(self.cache.cache.values()))
        self.assertEqual(entry["status_code"], 499)
        self.assertEqual(entry["state"], RequestCache.STATE_ERROR)


if __name__ == "__main__":
    unittest.main()
