import unittest

from ghc_api.routes.openai import _filter_responses_web_search_tools


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


if __name__ == "__main__":
    unittest.main()
