import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ghc_api.app import PROTECTED_PATHS, create_app
from ghc_api.webiq import (
    DROPPED_RESPONSE_HEADERS,
    ENDPOINT,
    WebIQError,
    endpoint_for,
    passthrough_headers,
    pop_legacy_option,
    result_count,
    search,
    tool_definition,
)


def make_settings(**overrides):
    """Everything the proxy needs. Note what is absent: there is no search
    parameter here, because the client's body is forwarded as received."""
    settings = SimpleNamespace(
        enable_webiq_search=True,
        webiq_api_key="secret",
        webiq_endpoint="",
        webiq_timeout=30,
    )
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


def upstream_response(status=200, body=b'{"webResults": []}', headers=None):
    return Mock(
        status_code=status,
        ok=200 <= status < 300,
        content=body,
        text=body.decode("utf-8", "replace"),
        headers=headers or {"content-type": "application/json"},
    )


class ToolDefinitionTest(unittest.TestCase):
    """The model-facing tool is a prompt surface, not the HTTP contract.

    It stays deliberately narrow: a model handed every Web Search v3 knob fills
    them in badly. The client turns these arguments into a full official
    request.
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


class EndpointTest(unittest.TestCase):
    def test_defaults_to_the_spec_endpoint(self):
        self.assertEqual(endpoint_for(make_settings()), ENDPOINT)

    def test_config_can_override_it(self):
        self.assertEqual(
            endpoint_for(make_settings(webiq_endpoint="http://127.0.0.1:9/x")),
            "http://127.0.0.1:9/x",
        )


class SearchTest(unittest.TestCase):
    """search() forwards bytes and hands back the raw upstream response."""

    @patch("ghc_api.webiq.requests.post")
    def test_body_is_forwarded_byte_for_byte(self, post):
        post.return_value = upstream_response()
        raw = b'{"query":"q","someFieldMicrosoftAddsLater":true}'

        search(raw, make_settings())

        kwargs = post.call_args.kwargs
        self.assertEqual(kwargs["data"], raw)
        # Never re-serialized: no json= argument, so nothing can be reordered,
        # defaulted or dropped on the way out.
        self.assertNotIn("json", kwargs)
        self.assertEqual(post.call_args[0][0], ENDPOINT)

    @patch("ghc_api.webiq.requests.post")
    def test_unknown_parameters_are_not_rejected(self, post):
        """A parameter added to the spec tomorrow must work here today."""
        post.return_value = upstream_response()

        response = search(b'{"query":"q","freshness":"day"}', make_settings())

        self.assertEqual(response.status_code, 200)

    @patch("ghc_api.webiq.requests.post")
    def test_invalid_bodies_are_left_to_upstream(self, post):
        """No local schema check: upstream owns the contract, including its errors."""
        post.return_value = upstream_response(400, b'{"error":"query is required"}')

        response = search(b"not json at all", make_settings())

        self.assertEqual(response.status_code, 400)
        self.assertEqual(post.call_args.kwargs["data"], b"not json at all")

    @patch("ghc_api.webiq.requests.post")
    def test_only_the_servers_key_is_sent(self, post):
        post.return_value = upstream_response()

        search(b"{}", make_settings())

        headers = post.call_args.kwargs["headers"]
        self.assertEqual(headers["x-apikey"], "secret")
        self.assertNotIn("authorization", {k.lower() for k in headers})

    @patch("ghc_api.webiq.requests.post")
    def test_upstream_errors_are_returned_not_raised(self, post):
        """Only 401/403 is rewritten; every other status is the caller's to see."""
        for status in (400, 404, 410, 429, 500, 503, 504):
            post.return_value = upstream_response(status, b'{"upstream":"body"}')

            response = search(b"{}", make_settings())

            self.assertEqual(response.status_code, status)
            self.assertEqual(response.content, b'{"upstream":"body"}')

    @patch("ghc_api.webiq.requests.post")
    def test_rejected_server_key_is_not_reported_as_client_auth_failure(self, post):
        """401/403 upstream is about our key, not about the caller's token."""
        for status in (401, 403):
            post.return_value = upstream_response(status, b"bad key")
            with self.assertRaises(WebIQError) as ctx:
                search(b"{}", make_settings())
            self.assertEqual(ctx.exception.status_code, 503)
            self.assertIn("webiq_api_key", str(ctx.exception))

    def test_unconfigured_server_reports_503(self):
        with self.assertRaises(WebIQError) as ctx:
            search(b'{"query":"q"}', make_settings(webiq_api_key=""))

        self.assertEqual(ctx.exception.status_code, 503)


class PassthroughHeaderTest(unittest.TestCase):
    def test_retry_after_survives(self):
        """Without it a client cannot back off correctly against a shared key."""
        headers = dict(passthrough_headers({
            "Retry-After": "30",
            "content-type": "application/json",
            "x-ms-something": "1",
        }))
        self.assertEqual(headers["Retry-After"], "30")
        self.assertEqual(headers["x-ms-something"], "1")

    def test_hop_by_hop_and_framing_headers_are_dropped(self):
        headers = dict(passthrough_headers({
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked",
            "Content-Length": "12",
            "Content-Encoding": "gzip",
            "content-type": "application/json",
        }))
        self.assertEqual(set(headers), {"content-type"})
        for name in ("connection", "transfer-encoding", "content-length"):
            self.assertIn(name, DROPPED_RESPONSE_HEADERS)

    def test_self_generated_and_origin_scoped_headers_are_dropped(self):
        """Date/Server would be comma-joined with this server's own values, and
        Alt-Svc/HSTS are claims about upstream's origin, not this proxy's."""
        headers = dict(passthrough_headers({
            "Date": "Mon, 01 Jan 2024 00:00:00 GMT",
            "Server": "upstream/1.0",
            "Alt-Svc": 'h3=":443"',
            "Strict-Transport-Security": "max-age=31536000",
            "content-type": "application/json",
        }))
        self.assertEqual(set(headers), {"content-type"})


class ResultCountTest(unittest.TestCase):
    def test_counts_web_results(self):
        self.assertEqual(result_count({"webResults": [{}, {}]}), 2)

    def test_unknown_shapes_report_nothing_rather_than_zero(self):
        self.assertIsNone(result_count(None))
        self.assertIsNone(result_count("plain text error"))
        self.assertIsNone(result_count({"error": "x"}))


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
        body = (b'{"webResults":[{"title":"t","url":"u","contentTier":"premium"}],'
                b'"traceId":"trace-9"}')
        search_mock.return_value = upstream_response(200, body)

        res = self.client.post("/v3/search/web", json={"query": "python"})

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.data, body)

    @patch("ghc_api.routes.webiq.search")
    def test_the_client_body_is_handed_over_as_raw_bytes(self, search_mock):
        search_mock.return_value = upstream_response()
        raw = b'{"query":"python","maxResults":20,"safeSearch":"off"}'

        self.client.post("/v3/search/web", data=raw,
                         content_type="application/json")

        self.assertEqual(search_mock.call_args[0][0], raw)

    @patch("ghc_api.routes.webiq.search")
    def test_upstream_error_body_and_headers_are_passed_through(self, search_mock):
        search_mock.return_value = upstream_response(
            429, b'{"error":{"code":"TooManyRequests"}}',
            headers={"content-type": "application/json", "Retry-After": "30"},
        )

        res = self.client.post("/v3/search/web", json={"query": "python"})

        self.assertEqual(res.status_code, 429)
        self.assertEqual(res.data, b'{"error":{"code":"TooManyRequests"}}')
        self.assertEqual(res.headers["Retry-After"], "30")

    @patch("ghc_api.routes.webiq.search")
    def test_hop_by_hop_headers_never_reach_the_client(self, search_mock):
        """PEP 3333 forbids them and waitress answers 500 if one appears."""
        search_mock.return_value = upstream_response(
            200, b"{}", headers={"content-type": "application/json",
                                 "Connection": "keep-alive"},
        )

        res = self.client.post("/v3/search/web", json={"query": "python"})

        self.assertEqual(res.status_code, 200)
        self.assertNotEqual(res.headers.get("Connection"), "keep-alive")

    @patch("ghc_api.routes.webiq.search", side_effect=WebIQError("nope", 503))
    def test_local_failures_still_get_a_json_error(self, _search):
        res = self.client.post("/v3/search/web", json={"query": "python"})

        self.assertEqual(res.status_code, 503)
        self.assertEqual(res.get_json()["error"]["type"], "webiq_search_error")


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
