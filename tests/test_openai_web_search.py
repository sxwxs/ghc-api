import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ghc_api.routes.openai import _filter_responses_web_search_tools
from ghc_api.web_search import (
    apply_webiq_search,
    extract_webiq_query,
    inject_webiq_grounding,
)


class ResponsesWebSearchToolTest(unittest.TestCase):
    def test_keeps_web_search_for_gpt_responses_models(self):
        payload = {
            "tools": [
                {"type": "web_search"},
                {"type": "image_generation"},
                {"type": "function", "name": "shell"},
            ]
        }

        _filter_responses_web_search_tools(payload, "gpt-5.4-mini", "req-1")

        self.assertEqual(
            payload["tools"],
            [
                {"type": "web_search"},
                {"type": "function", "name": "shell"},
            ],
        )

    def test_removes_web_search_for_non_gpt_responses_models(self):
        payload = {
            "tools": [
                {"type": "web_search"},
                {"type": "function", "name": "shell"},
            ]
        }

        _filter_responses_web_search_tools(payload, "gemini-3.1-pro-preview", "req-1")

        self.assertEqual(payload["tools"], [{"type": "function", "name": "shell"}])


class WebIQSearchTest(unittest.TestCase):
    def test_extracts_query_from_chat_and_responses(self):
        self.assertEqual(
            extract_webiq_query({"messages": [{"role": "user", "content": "latest Python"}]}, "chat"),
            "latest Python",
        )
        self.assertEqual(
            extract_webiq_query({
                "input": [{"role": "user", "content": [{"type": "input_text", "text": "latest Rust"}]}]
            }, "responses"),
            "latest Rust",
        )

    def test_injects_grounding_and_removes_proxy_option(self):
        chat = inject_webiq_grounding({
            "webiq_search_options": {"enabled": True},
            "messages": [{"role": "system", "content": "Be concise"}],
        }, "grounding", "chat")
        self.assertNotIn("webiq_search_options", chat)
        self.assertEqual(chat["messages"][0]["content"], "Be concise\n\ngrounding")

        responses = inject_webiq_grounding({
            "webiq_search_options": True,
            "instructions": "Be concise",
        }, "grounding", "responses")
        self.assertNotIn("webiq_search_options", responses)
        self.assertEqual(responses["instructions"], "Be concise\n\ngrounding")

    @patch("ghc_api.web_search.requests.post")
    def test_apply_calls_webiq_v3_and_injects_results(self, post):
        response = Mock(ok=True)
        response.json.return_value = {
            "webResults": [{"title": "Python", "url": "https://python.org", "content": "Latest release"}]
        }
        post.return_value = response
        settings = SimpleNamespace(
            enable_webiq_search=True,
            webiq_api_key="secret",
            webiq_endpoint="https://api.microsoftol.com/v3/search/web",
            webiq_max_results=5,
            webiq_language="en",
            webiq_region="US",
            webiq_max_length=3000,
            webiq_content_format="passage",
            webiq_safe_search="strict",
            webiq_timeout=30,
        )

        result = apply_webiq_search({
            "input": "latest Python",
            "webiq_search_options": {"enabled": True},
        }, "responses", settings)

        self.assertIn("https://python.org", result["instructions"])
        headers = post.call_args.kwargs["headers"]
        request_body = post.call_args.kwargs["json"]
        self.assertEqual(headers["x-apikey"], "secret")
        self.assertEqual(request_body["contentFormat"], "passage")
        self.assertEqual(request_body["query"], "latest Python")


if __name__ == "__main__":
    unittest.main()
