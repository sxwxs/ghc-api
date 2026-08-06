import unittest
from unittest import mock

from ghc_api.app import PROTECTED_PATHS, create_app
from ghc_api.state import state


class FakeResponse:
    def __init__(self, body, status_code=200, text=None, content_type="application/json"):
        self._body = body
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = text if text is not None else ""
        if text is None and isinstance(body, dict):
            import json
            self.text = json.dumps(body)
        self.content = self.text.encode("utf-8")
        self.headers = {"Content-Type": content_type}

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class OpenAIEmbeddingsTests(unittest.TestCase):
    def setUp(self):
        self.saved_models = state.models
        self.saved_retries = state.max_connection_retries
        state.models = {
            "data": [
                {
                    "id": "text-embedding-3-small",
                    "capabilities": {"type": "embeddings"},
                },
                {
                    "id": "gpt-chat",
                    "capabilities": {"type": "chat"},
                },
            ]
        }
        state.max_connection_retries = 0
        self.client = create_app().test_client()

    def tearDown(self):
        state.models = self.saved_models
        state.max_connection_retries = self.saved_retries

    @mock.patch("ghc_api.routes.openai.cache.add_request")
    @mock.patch("ghc_api.routes.openai.requests.post")
    @mock.patch("ghc_api.routes.openai.get_copilot_headers", return_value={})
    @mock.patch("ghc_api.routes.openai.ensure_copilot_token")
    def test_string_input_is_normalized_and_response_is_openai_compatible(
        self, _ensure_token, _headers, post, add_request
    ):
        post.return_value = FakeResponse({
            "data": [{"embedding": [0.1, 0.2], "index": 0}],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        })

        response = self.client.post("/v1/embeddings", json={
            "model": "text-embedding-3-small",
            "input": "hello",
            "dimensions": 2,
        })

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["object"], "list")
        self.assertEqual(body["model"], "text-embedding-3-small")
        self.assertEqual(body["data"][0]["object"], "embedding")

        args, kwargs = post.call_args
        self.assertTrue(args[0].endswith("/embeddings"))
        self.assertEqual(kwargs["json"]["input"], ["hello"])
        self.assertEqual(kwargs["json"]["dimensions"], 2)
        self.assertEqual(kwargs["headers"]["X-Initiator"], "user")

        cached = add_request.call_args.args[1]
        self.assertEqual(cached["endpoint"], "/v1/embeddings")
        self.assertEqual(cached["input_tokens"], 1)
        self.assertEqual(cached["output_tokens"], 0)

    @mock.patch("ghc_api.routes.openai.requests.post")
    @mock.patch("ghc_api.routes.openai.ensure_copilot_token")
    def test_rejects_non_embedding_model(self, _ensure_token, post):
        response = self.client.post("/embeddings", json={
            "model": "gpt-chat",
            "input": ["hello"],
        })

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"]["code"], "unsupported_model")
        post.assert_not_called()

    @mock.patch("ghc_api.routes.openai.requests.post")
    @mock.patch("ghc_api.routes.openai.get_copilot_headers", return_value={})
    @mock.patch("ghc_api.routes.openai.ensure_copilot_token")
    def test_successful_non_json_response_is_passed_through_unchanged(
        self, _ensure_token, _headers, post
    ):
        post.return_value = FakeResponse(
            ValueError("not json"),
            text="upstream-body",
            content_type="text/plain; charset=utf-8",
        )

        response = self.client.post("/v1/embeddings", json={
            "model": "text-embedding-3-small",
            "input": ["hello"],
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_data(), b"upstream-body")
        self.assertEqual(response.content_type, "text/plain; charset=utf-8")

    @mock.patch("ghc_api.routes.openai.requests.post")
    @mock.patch("ghc_api.routes.openai.get_copilot_headers", return_value={})
    @mock.patch("ghc_api.routes.openai.ensure_copilot_token")
    def test_flat_token_array_is_wrapped_as_one_input(self, _ensure_token, _headers, post):
        post.return_value = FakeResponse({
            "data": [{"embedding": [0.1], "index": 0}],
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        })

        response = self.client.post("/v1/embeddings", json={
            "model": "text-embedding-3-small",
            "input": [123, 456],
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(post.call_args.kwargs["json"]["input"], [[123, 456]])

    def test_embedding_paths_are_auth_protected(self):
        self.assertIn("/v1/embeddings", PROTECTED_PATHS)
        self.assertIn("/embeddings", PROTECTED_PATHS)


if __name__ == "__main__":
    unittest.main()
