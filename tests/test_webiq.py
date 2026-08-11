import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ghc_api.app import PROTECTED_PATHS, create_app
from ghc_api.webiq import (
    ENDPOINT,
    WebIQError,
    build_upstream_request,
    normalize_query,
    pop_legacy_option,
    search,
    tool_definition,
)


def make_settings(**overrides):
    settings = SimpleNamespace(
        enable_webiq_search=True,
        webiq_api_key="secret",
        webiq_endpoint="",
        webiq_max_results=5,
        webiq_language="en",
        webiq_region="US",
        webiq_max_length=3000,
        webiq_content_format="passage",
        webiq_safe_search="strict",
        webiq_max_results_cap=50,
        webiq_max_length_cap=500000,
        webiq_timeout=30,
    )
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


class ToolDefinitionTest(unittest.TestCase):
    """The model-facing tool is a prompt surface, not the HTTP contract.

    It stays deliberately narrow: a model handed every Web Search v3 knob fills
    them in badly.
    """

    def test_responses_shape_is_flat(self):
        tool = tool_definition("responses")
        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["name"], "webiq_search")
        self.assertIn("query", tool["parameters"]["properties"])

    def test_chat_shape_is_nested(self):
        tool = tool_definition("chat")
        self.assertEqual(tool["function"]["name"], "webiq_search")

    def test_it_exposes_only_query_and_a_result_count(self):
        properties = tool_definition("responses")["parameters"]["properties"]
        self.assertEqual(set(properties), {"query", "max_results"})


class NormalizationTest(unittest.TestCase):
    def test_query_whitespace_is_collapsed(self):
        self.assertEqual(normalize_query("  latest   Python  "), "latest Python")

    def test_over_long_query_is_rejected_not_truncated(self):
        """Truncating would turn a trailing site: operator into a different search."""
        with self.assertRaises(WebIQError) as ctx:
            normalize_query("x" * 1001)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(len(normalize_query("x" * 1000)), 1000)

    def test_bad_queries_are_rejected(self):
        for bad in ("", "   ", None, 42):
            with self.assertRaises(WebIQError) as ctx:
                normalize_query(bad)
            self.assertEqual(ctx.exception.status_code, 400)


class LegacyOptionTest(unittest.TestCase):
    def test_option_is_always_stripped(self):
        for value in (True, False, {"enabled": True}, {"enabled": False}, {}):
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


class RequestBuildingTest(unittest.TestCase):
    """The official contract in, an official upstream body out."""

    def test_config_supplies_defaults(self):
        body = build_upstream_request({"query": "q"}, make_settings())
        self.assertEqual(body, {
            "query": "q",
            "maxResults": 5,
            "language": "en",
            "region": "US",
            "maxLength": 3000,
            "contentFormat": "passage",
            "safeSearch": "strict",
        })

    def test_every_parameter_can_be_overridden(self):
        """Config is a default, not a lock: two clients can want different things."""
        body = build_upstream_request({
            "query": "q",
            "maxResults": 25,
            "language": "ja",
            "region": "JP",
            "location": "lat:40.753250;long:-74.003807",
            "contentFormat": "markdown",
            "maxLength": 120000,
            "safeSearch": "off",
        }, make_settings())
        self.assertEqual(body["maxResults"], 25)
        self.assertEqual(body["language"], "ja")
        self.assertEqual(body["region"], "JP")
        self.assertEqual(body["location"], "lat:40.753250;long:-74.003807")
        self.assertEqual(body["contentFormat"], "markdown")
        self.assertEqual(body["maxLength"], 120000)
        self.assertEqual(body["safeSearch"], "off")

    def test_location_is_omitted_unless_asked_for(self):
        self.assertNotIn("location", build_upstream_request({"query": "q"}, make_settings()))

    def test_out_of_spec_values_are_rejected(self):
        settings = make_settings()
        for payload in (
            {"query": "q", "maxResults": 51},
            {"query": "q", "maxResults": 0},
            {"query": "q", "maxResults": "3"},
            {"query": "q", "maxResults": True},
            {"query": "q", "maxLength": 500001},
            {"query": "q", "contentFormat": "pdf"},
            {"query": "q", "safeSearch": "moderate"},
            {"query": "q", "language": "english"},
            {"query": "q", "region": "USA"},
            {"query": "q", "location": "40.75,-74.00"},
        ):
            with self.assertRaises(WebIQError, msg=payload) as ctx:
                build_upstream_request(payload, settings)
            self.assertEqual(ctx.exception.status_code, 400)

    def test_local_cap_clamps_in_spec_values(self):
        """A cap is local policy protecting a paid key, so it clamps, not errors."""
        settings = make_settings(webiq_max_results_cap=8, webiq_max_length_cap=1000)
        body = build_upstream_request(
            {"query": "q", "maxResults": 50, "maxLength": 400000}, settings)
        self.assertEqual(body["maxResults"], 8)
        self.assertEqual(body["maxLength"], 1000)

    def test_unknown_parameters_are_rejected(self):
        with self.assertRaises(WebIQError) as ctx:
            build_upstream_request({"query": "q", "coun": 3}, make_settings())
        self.assertEqual(ctx.exception.status_code, 400)

    def test_retired_snake_case_names_name_their_replacement(self):
        with self.assertRaises(WebIQError) as ctx:
            build_upstream_request({"query": "q", "max_results": 3}, make_settings())
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("maxResults", str(ctx.exception))

    def test_non_object_body_is_rejected(self):
        with self.assertRaises(WebIQError) as ctx:
            build_upstream_request(["q"], make_settings())
        self.assertEqual(ctx.exception.status_code, 400)


class SearchTest(unittest.TestCase):
    @patch("ghc_api.webiq.requests.post")
    def test_sends_an_official_request_body(self, post):
        post.return_value = Mock(ok=True, **{"json.return_value": {"webResults": []}})

        search({"query": "latest python", "maxResults": 3}, make_settings())

        self.assertEqual(post.call_args[0][0], ENDPOINT)
        self.assertEqual(post.call_args.kwargs["headers"]["x-apikey"], "secret")
        # Host is HTTP boilerplate: urllib3 derives it from the URL.
        self.assertNotIn("host", {k.lower() for k in post.call_args.kwargs["headers"]})
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["query"], "latest python")
        self.assertEqual(body["maxResults"], 3)
        self.assertEqual(body["contentFormat"], "passage")

    @patch("ghc_api.webiq.requests.post")
    def test_endpoint_can_be_overridden(self, post):
        """For a mock, a recording proxy or a regional deployment."""
        post.return_value = Mock(ok=True, **{"json.return_value": {"webResults": []}})

        search({"query": "q"}, make_settings(webiq_endpoint="http://localhost:9/v3/search/web"))

        self.assertEqual(post.call_args[0][0], "http://localhost:9/v3/search/web")

    @patch("ghc_api.webiq.requests.post")
    def test_response_is_returned_verbatim(self, post):
        """Compatibility means every field survives, not just the convenient ones.

        Dropping these would make click instrumentation impossible, hide
        premium-tier billing, and leave support tickets without a trace id.
        """
        upstream = {
            "webResults": [{
                "title": "Python",
                "url": "https://python.org",
                "content": "3.13",
                "crawledAt": "2026-01-02T03:04:05Z",
                "lastUpdatedAt": "2026-01-01T00:00:00Z",
                "language": "en",
                "isAdult": False,
                "clickUrl": "https://bing.com/click?k=1",
                "instrumentationSuffix": "abc",
                "contentTier": "premium",
            }],
            "instrumentationClickBase": "https://bing.com/ping?",
            "traceId": "trace-123",
        }
        post.return_value = Mock(ok=True, **{"json.return_value": upstream})

        self.assertEqual(search({"query": "q"}, make_settings()), upstream)

    @patch("ghc_api.webiq.requests.post")
    def test_content_is_not_truncated_locally(self, post):
        """maxLength is enforced on the request; the answer is passed through."""
        post.return_value = Mock(ok=True, **{"json.return_value": {
            "webResults": [{"title": "t", "url": "u", "content": "x" * 9000}]
        }})

        body = search({"query": "q"}, make_settings(webiq_max_length=100))

        self.assertEqual(post.call_args.kwargs["json"]["maxLength"], 100)
        self.assertEqual(len(body["webResults"][0]["content"]), 9000)

    @patch("ghc_api.webiq.requests.post")
    def test_upstream_status_codes_keep_their_meaning(self, post):
        for upstream_status, expected in [
            (400, 400), (410, 410), (415, 415), (429, 429),
            (500, 500), (503, 503), (504, 504), (418, 502),
        ]:
            post.return_value = Mock(ok=False, status_code=upstream_status, text="")
            with self.assertRaises(WebIQError, msg=upstream_status) as ctx:
                search({"query": "q"}, make_settings())
            self.assertEqual(ctx.exception.status_code, expected)

    @patch("ghc_api.webiq.requests.post")
    def test_rejected_server_key_is_not_reported_as_client_auth_failure(self, post):
        """401/403 upstream is about our key, not about the caller's token."""
        for upstream_status in (401, 403):
            post.return_value = Mock(ok=False, status_code=upstream_status, text="bad key")
            with self.assertRaises(WebIQError) as ctx:
                search({"query": "q"}, make_settings())
            self.assertEqual(ctx.exception.status_code, 503)
            self.assertIn("webiq_api_key", str(ctx.exception))

    @patch("ghc_api.webiq.requests.post")
    def test_upstream_error_body_is_surfaced(self, post):
        post.return_value = Mock(ok=False, status_code=400, text="query too long")

        with self.assertRaises(WebIQError) as ctx:
            search({"query": "q"}, make_settings())

        self.assertIn("query too long", str(ctx.exception))

    def test_unconfigured_server_reports_503(self):
        with self.assertRaises(WebIQError) as ctx:
            search({"query": "q"}, make_settings(webiq_api_key=""))

        self.assertEqual(ctx.exception.status_code, 503)

    def test_a_bad_request_is_rejected_even_when_unconfigured(self):
        with self.assertRaises(WebIQError) as ctx:
            search({"query": "  "}, make_settings(webiq_api_key=""))

        self.assertEqual(ctx.exception.status_code, 400)

    @patch("ghc_api.webiq.requests.post")
    def test_trace_records_what_went_upstream_without_the_key(self, post):
        post.return_value = Mock(ok=True, status_code=200,
                                 **{"json.return_value": {"webResults": [{}]}})
        trace = {}

        search({"query": "q"}, make_settings(), trace=trace)

        self.assertEqual(trace["endpoint"], ENDPOINT)
        self.assertEqual(trace["request"]["query"], "q")
        self.assertEqual(trace["result_count"], 1)
        self.assertNotIn("secret", json.dumps(trace))


class RouteTest(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_search_endpoint_is_auth_gated(self):
        self.assertIn("/v3/search/web", PROTECTED_PATHS)

    def test_retired_path_is_gone(self):
        """No alias: the bespoke shape was removed, like webiq_search_options."""
        res = self.client.post("/v1/webiq/search", json={"query": "x"})
        self.assertEqual(res.status_code, 404)

    @patch("ghc_api.routes.webiq.search")
    def test_upstream_body_reaches_the_client_unchanged(self, search_mock):
        upstream = {
            "webResults": [{"title": "t", "url": "u", "content": "c", "contentTier": "premium"}],
            "traceId": "trace-9",
        }
        search_mock.return_value = upstream

        res = self.client.post("/v3/search/web", json={"query": "python"})

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json(), upstream)

    @patch("ghc_api.routes.webiq.search")
    def test_the_client_request_is_handed_over_as_given(self, search_mock):
        search_mock.return_value = {"webResults": []}
        payload = {"query": "python", "maxResults": 20, "safeSearch": "off"}

        self.client.post("/v3/search/web", json=payload)

        self.assertEqual(search_mock.call_args[0][0], payload)

    @patch("ghc_api.routes.webiq.search", side_effect=WebIQError("nope", 503))
    def test_propagates_status_code(self, _search):
        res = self.client.post("/v3/search/web", json={"query": "python"})

        self.assertEqual(res.status_code, 503)
        self.assertEqual(res.get_json()["error"]["type"], "webiq_search_error")

    def test_rejects_non_object_body(self):
        res = self.client.post("/v3/search/web", json=["python"])

        self.assertEqual(res.status_code, 400)

    def test_rejects_bad_query_before_searching(self):
        res = self.client.post("/v3/search/web", json={"query": "   "})

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
