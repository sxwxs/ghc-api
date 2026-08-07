import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ghc_api.app import PROTECTED_PATHS, create_app
from ghc_api.webiq import (
    WebIQError,
    normalize_max_results,
    normalize_query,
    pop_legacy_option,
    search,
    tool_definition,
)


def make_settings(**overrides):
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
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


class ToolDefinitionTest(unittest.TestCase):
    def test_responses_shape_is_flat(self):
        tool = tool_definition("responses")
        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["name"], "webiq_search")
        self.assertIn("query", tool["parameters"]["properties"])

    def test_chat_shape_is_nested(self):
        tool = tool_definition("chat")
        self.assertEqual(tool["function"]["name"], "webiq_search")


class NormalizationTest(unittest.TestCase):
    def test_query_is_collapsed_and_capped(self):
        self.assertEqual(normalize_query("  latest   Python  "), "latest Python")
        self.assertEqual(len(normalize_query("x" * 5000)), 400)

    def test_bad_queries_are_rejected(self):
        for bad in ("", "   ", None, 42):
            with self.assertRaises(WebIQError) as ctx:
                normalize_query(bad)
            self.assertEqual(ctx.exception.status_code, 400)

    def test_max_results_is_clamped(self):
        self.assertEqual(normalize_max_results(None, 5), 5)
        self.assertEqual(normalize_max_results(999, 5), 10)
        self.assertEqual(normalize_max_results(0, 5), 1)
        with self.assertRaises(WebIQError):
            normalize_max_results("many", 5)


class LegacyOptionTest(unittest.TestCase):
    def test_option_is_always_stripped(self):
        """It must never survive into a payload forwarded upstream."""
        for value in (True, {"enabled": True}, False, {"enabled": False}, {}, 0):
            payload = {"model": "m", "webiq_search_options": value}
            pop_legacy_option(payload)
            self.assertNotIn("webiq_search_options", payload)

    def test_reports_whether_search_was_requested(self):
        self.assertTrue(pop_legacy_option({"webiq_search_options": True}))
        self.assertTrue(pop_legacy_option({"webiq_search_options": {"enabled": True}}))
        self.assertTrue(pop_legacy_option({"webiq_search_options": {}}))
        self.assertFalse(pop_legacy_option({"webiq_search_options": False}))
        self.assertFalse(pop_legacy_option({"webiq_search_options": {"enabled": False}}))
        self.assertFalse(pop_legacy_option({"webiq_search_options": {"enabled": 0}}))
        self.assertFalse(pop_legacy_option({"model": "m"}))


class SearchTest(unittest.TestCase):
    @patch("ghc_api.webiq.requests.post")
    def test_sends_configured_parameters(self, post):
        post.return_value = Mock(ok=True, **{"json.return_value": {
            "webResults": [{"title": "Python", "url": "https://python.org", "content": "3.13"}]
        }})

        results = search("latest python", make_settings(), max_results=3)

        self.assertEqual(results, [
            {"title": "Python", "url": "https://python.org", "content": "3.13"},
        ])
        self.assertEqual(post.call_args.kwargs["headers"]["x-apikey"], "secret")
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["query"], "latest python")
        self.assertEqual(body["maxResults"], 3)
        self.assertEqual(body["contentFormat"], "passage")

    @patch("ghc_api.webiq.requests.post")
    def test_content_is_capped_locally(self, post):
        post.return_value = Mock(ok=True, **{"json.return_value": {
            "webResults": [{"title": "t", "url": "u", "content": "x" * 9000}]
        }})

        results = search("q", make_settings(webiq_max_length=100))

        self.assertEqual(len(results[0]["content"]), 100)

    @patch("ghc_api.webiq.requests.post")
    def test_malformed_results_are_dropped(self, post):
        post.return_value = Mock(ok=True, **{"json.return_value": {
            "webResults": ["nope", {"url": "u"}]
        }})

        results = search("q", make_settings())

        self.assertEqual(results, [{"title": "Untitled", "url": "u", "content": ""}])

    @patch("ghc_api.webiq.requests.post")
    def test_rate_limit_is_surfaced_as_429(self, post):
        post.return_value = Mock(ok=False, status_code=429)

        with self.assertRaises(WebIQError) as ctx:
            search("q", make_settings())

        self.assertEqual(ctx.exception.status_code, 429)

    def test_unconfigured_server_reports_503(self):
        with self.assertRaises(WebIQError) as ctx:
            search("q", make_settings(webiq_api_key=""))

        self.assertEqual(ctx.exception.status_code, 503)


class RouteTest(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_search_endpoint_is_auth_gated(self):
        self.assertIn("/v1/webiq/search", PROTECTED_PATHS)

    @patch("ghc_api.routes.webiq.search")
    def test_returns_normalized_results(self, search_mock):
        search_mock.return_value = [{"title": "t", "url": "u", "content": "c"}]

        res = self.client.post("/v1/webiq/search", json={"query": "python"})

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["results"], [{"title": "t", "url": "u", "content": "c"}])

    @patch("ghc_api.routes.webiq.search", side_effect=WebIQError("nope", 503))
    def test_propagates_status_code(self, _search):
        res = self.client.post("/v1/webiq/search", json={"query": "python"})

        self.assertEqual(res.status_code, 503)
        self.assertEqual(res.get_json()["error"]["type"], "webiq_search_error")

    def test_rejects_non_object_body(self):
        res = self.client.post("/v1/webiq/search", json=["python"])

        self.assertEqual(res.status_code, 400)


class LegacyRouteRejectionTest(unittest.TestCase):
    """The retired option must fail loudly rather than silently do nothing."""

    def setUp(self):
        self.client = create_app().test_client()

    @patch("ghc_api.routes.openai.ensure_copilot_token")
    def test_chat_completions_rejects_legacy_option(self, _token):
        res = self.client.post("/v1/chat/completions", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "webiq_search_options": {"enabled": True},
        })

        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.get_json()["error"]["code"], "webiq_search_options_removed")

    @patch("ghc_api.routes.openai.ensure_copilot_token")
    def test_responses_rejects_legacy_option(self, _token):
        res = self.client.post("/v1/responses", json={
            "model": "gpt-4o",
            "input": "hi",
            "webiq_search_options": True,
        })

        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.get_json()["error"]["code"], "webiq_search_options_removed")


if __name__ == "__main__":
    unittest.main()
