import unittest
from unittest import mock

from ghc_api.app import create_app
from ghc_api.cache import cache
from ghc_api.routes.openai import _normalize_responses_tool_descriptions


class ResponsesToolDescriptionNormalizationTests(unittest.TestCase):
    def tearDown(self):
        cache.cache.clear()

    def test_fills_blank_description_in_additional_tools_namespace(self):
        payload = {
            "input": [
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [
                        {
                            "type": "namespace",
                            "name": "functions",
                            "description": "",
                            "tools": [
                                {
                                    "type": "function",
                                    "name": "exec",
                                    "description": "Run commands",
                                }
                            ],
                        }
                    ],
                }
            ]
        }

        count = _normalize_responses_tool_descriptions(payload)

        self.assertEqual(count, 1)
        namespace = payload["input"][0]["tools"][0]
        self.assertEqual(namespace["description"], "Tools in the functions namespace.")
        self.assertEqual(namespace["tools"][0]["description"], "Run commands")

    def test_normalizes_nested_and_top_level_tools_but_preserves_missing_description(self):
        payload = {
            "tools": [
                {"type": "function", "name": "empty", "description": "  "},
                {"type": "function", "name": "missing"},
                {"type": "function", "name": "valid", "description": "Useful"},
            ],
            "input": [
                {
                    "type": "additional_tools",
                    "tools": [
                        {
                            "type": "namespace",
                            "name": "outer",
                            "description": "Namespace tools",
                            "tools": [
                                {"type": "function", "name": "inner", "description": ""}
                            ],
                        }
                    ],
                }
            ],
        }

        count = _normalize_responses_tool_descriptions(payload)

        self.assertEqual(count, 2)
        self.assertEqual(payload["tools"][0]["description"], "Tool: empty.")
        self.assertNotIn("description", payload["tools"][1])
        self.assertEqual(payload["tools"][2]["description"], "Useful")
        self.assertEqual(
            payload["input"][0]["tools"][0]["tools"][0]["description"],
            "Tool: inner.",
        )

    def test_responses_route_normalizes_before_forwarding_and_keeps_original_for_cache(self):
        class FakeResponse:
            ok = True
            status_code = 200
            text = '{"id":"resp-1","usage":{}}'

            def json(self):
                return {"id": "resp-1", "usage": {}}

        payload = {
            "model": "gpt-test",
            "input": [
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [
                        {
                            "type": "namespace",
                            "name": "functions",
                            "description": "",
                            "tools": [],
                        }
                    ],
                }
            ],
        }
        app = create_app()

        with mock.patch("ghc_api.routes.openai.ensure_copilot_token"), \
             mock.patch("ghc_api.routes.openai.supports_responses_api", return_value=True), \
             mock.patch("ghc_api.routes.openai.get_copilot_headers", return_value={}), \
             mock.patch("ghc_api.routes.openai.requests.post", return_value=FakeResponse()) as post:
            with app.test_client() as client:
                response = client.post("/v1/responses", json=payload)

        self.assertEqual(response.status_code, 200)
        forwarded = post.call_args.kwargs["json"]
        self.assertEqual(
            forwarded["input"][0]["tools"][0]["description"],
            "Tools in the functions namespace.",
        )
        cached = next(iter(cache.cache.values()))
        self.assertEqual(
            cached["original_request_body"]["input"][0]["tools"][0]["description"],
            "",
        )
        self.assertEqual(
            cached["request_body"]["input"][0]["tools"][0]["description"],
            "Tools in the functions namespace.",
        )


if __name__ == "__main__":
    unittest.main()
