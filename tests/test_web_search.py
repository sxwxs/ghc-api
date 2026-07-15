import copy
import unittest
from unittest import mock

from ghc_api import web_search


class WebSearchTest(unittest.TestCase):
    def test_detects_and_removes_versioned_web_search_tools(self):
        payload = {
            "tools": [
                {"type": "web_search_20250305", "name": "web_search"},
                {"name": "get_weather", "input_schema": {"type": "object"}},
            ],
            "tool_choice": {"type": "auto"},
        }

        self.assertTrue(web_search.has_web_search_tool(payload))
        result = web_search.remove_web_search_tools(payload)

        self.assertEqual(result["tools"], [payload["tools"][1]])
        self.assertEqual(result["tool_choice"], {"type": "auto"})
        self.assertEqual(len(payload["tools"]), 2)

    def test_removes_tool_choice_when_web_search_is_the_only_tool(self):
        payload = {
            "tools": [{"type": "web_search_20260209", "name": "web_search"}],
            "tool_choice": {"type": "tool", "name": "web_search"},
        }

        result = web_search.remove_web_search_tools(payload)

        self.assertNotIn("tools", result)
        self.assertNotIn("tool_choice", result)

    def test_extracts_query_from_last_user_message(self):
        payload = {
            "messages": [
                {"role": "user", "content": "old query"},
                {"role": "assistant", "content": "answer"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "长鑫存储"},
                        {"type": "image", "source": {}},
                        {"type": "text", "text": "最新进展"},
                    ],
                },
            ]
        }

        self.assertEqual(web_search.extract_search_query(payload), "长鑫存储 最新进展")

    def test_strips_claude_code_web_search_wrapper(self):
        query = "北京 今日天气 2026年7月15日 当前天气 预报"
        payloads = (
            {
                "messages": [{
                    "role": "user",
                    "content": f"Perform a web search for the query: {query}",
                }]
            },
            {
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": f"perform a WEB search FOR the QUERY:   {query}",
                    }],
                }]
            },
        )

        for payload in payloads:
            with self.subTest(payload=payload):
                self.assertEqual(web_search.extract_search_query(payload), query)

    def test_does_not_strip_wrapper_text_embedded_in_normal_prompt(self):
        text = (
            "Explain this literal example: "
            "Perform a web search for the query: 北京天气"
        )
        payload = {"messages": [{"role": "user", "content": text}]}

        self.assertEqual(web_search.extract_search_query(payload), text)

    def test_calls_search_endpoint_with_limit_three(self):
        response = mock.Mock()
        response.json.return_value = {
            "keyword": "长鑫存储",
            "results": [{"title": "result", "link": "https://example.com"}],
        }

        with mock.patch.object(web_search.requests, "get", return_value=response) as get:
            results = web_search.call_search_proxy("长鑫存储", "http://127.0.0.1:5002/")

        self.assertEqual(results, response.json.return_value["results"])
        get.assert_called_once_with(
            "http://127.0.0.1:5002/search",
            params={"keyword": "长鑫存储", "limit": 3},
            timeout=30,
        )
        response.raise_for_status.assert_called_once_with()

    def test_accepts_compatible_result_envelopes_and_ignores_invalid_items(self):
        for body in (
            [{"title": "list"}, "invalid"],
            {"items": [{"title": "items"}]},
            {"data": [{"title": "data"}]},
        ):
            with self.subTest(body=body), mock.patch.object(
                web_search.requests,
                "get",
                return_value=mock.Mock(
                    json=mock.Mock(return_value=body),
                    raise_for_status=mock.Mock(),
                ),
            ):
                results = web_search.call_search_proxy("query", "http://search")
                self.assertTrue(results)
                self.assertTrue(all(isinstance(item, dict) for item in results))

    def test_search_errors_and_malformed_data_return_empty_results(self):
        responses = (
            mock.Mock(raise_for_status=mock.Mock(side_effect=RuntimeError("offline"))),
            mock.Mock(
                raise_for_status=mock.Mock(),
                json=mock.Mock(side_effect=ValueError("invalid json")),
            ),
            mock.Mock(
                raise_for_status=mock.Mock(),
                json=mock.Mock(return_value={"results": "invalid"}),
            ),
        )
        for response in responses:
            with self.subTest(response=response), mock.patch.object(
                web_search.requests, "get", return_value=response
            ):
                self.assertEqual(
                    web_search.call_search_proxy("query", "http://search"),
                    [],
                )

    def test_formats_field_aliases(self):
        text = web_search.format_search_results(
            "query",
            [{
                "title": "Alias result",
                "url": "https://example.com/alias",
                "snippet": "Snippet text",
            }],
        )

        self.assertIn("Alias result", text)
        self.assertIn("https://example.com/alias", text)
        self.assertIn("Snippet text", text)

    def test_injects_system_context_immutably(self):
        payloads = (
            ({"messages": []}, [{"type": "text", "text": "results"}]),
            ({"system": "base", "messages": []}, "base\n\nresults"),
            (
                {"system": [{"type": "text", "text": "base"}], "messages": []},
                [
                    {"type": "text", "text": "base"},
                    {"type": "text", "text": "results"},
                ],
            ),
        )
        for payload, expected in payloads:
            with self.subTest(payload=payload):
                original = copy.deepcopy(payload)
                result = web_search.inject_search_results_into_payload(payload, "results")
                self.assertEqual(result["system"], expected)
                self.assertEqual(payload, original)

    def test_apply_fallback_passes_unwrapped_query_to_search_proxy(self):
        query = "北京 今日天气 2026年7月15日 当前天气 预报"
        payload = {
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": f"Perform a web search for the query: {query}",
                }],
            }],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        }

        with mock.patch.object(
            web_search, "call_search_proxy", return_value=[]
        ) as call_search:
            result = web_search.apply_web_search_fallback(payload, "http://search")

        call_search.assert_called_once_with(query, "http://search")
        self.assertIn(query, result["system"][0]["text"])
        self.assertNotIn(web_search.SEARCH_QUERY_PREFIX, result["system"][0]["text"])

    def test_apply_fallback_injects_no_results_and_removes_tool(self):
        payload = {
            "messages": [{"role": "user", "content": "query"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        }

        with mock.patch.object(web_search, "call_search_proxy", return_value=[]):
            result = web_search.apply_web_search_fallback(payload, "http://search")

        self.assertNotIn("tools", result)
        self.assertIn("No results found", result["system"][0]["text"])
        self.assertIn("tools", payload)


if __name__ == "__main__":
    unittest.main()
