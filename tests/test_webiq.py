import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ghc_api.app import PROTECTED_PATHS, create_app
from ghc_api.webiq import (
    API_BASE_URL,
    API_PATHS,
    DROPPED_RESPONSE_HEADERS,
    ENDPOINT,
    MCP_PATH,
    WEB_PATH,
    WebIQError,
    endpoint_for,
    mcp_request,
    passthrough_headers,
    pop_legacy_option,
    result_count,
    search,
    timeout_for,
    tool_definition,
)


def make_settings(**overrides):
    """Everything the proxy needs. Note what is absent: there is no search
    parameter here, because the client's body is forwarded as received."""
    settings = SimpleNamespace(
        enable_webiq_search=True,
        webiq_api_key="secret",
        webiq_base_url="",
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

    def test_legacy_config_can_override_web_search(self):
        self.assertEqual(
            endpoint_for(make_settings(webiq_endpoint="http://127.0.0.1:9/x")),
            "http://127.0.0.1:9/x",
        )

    def test_all_service_base_override_applies_to_every_path(self):
        settings = make_settings(webiq_base_url="http://127.0.0.1:9/root/")
        for path in (*API_PATHS, MCP_PATH):
            self.assertEqual(endpoint_for(settings, path),
                             "http://127.0.0.1:9/root" + path)

    def test_default_urls_cover_every_official_service(self):
        for path in (*API_PATHS, MCP_PATH):
            self.assertEqual(endpoint_for(make_settings(), path), API_BASE_URL + path)

    def test_standard_shaped_legacy_endpoint_can_supply_a_base(self):
        settings = make_settings(
            webiq_endpoint="http://127.0.0.1:9/v3/search/web")
        self.assertEqual(endpoint_for(settings, "/v3/search/news"),
                         "http://127.0.0.1:9/v3/search/news")

    def test_nonstandard_legacy_endpoint_fails_closed_for_other_services(self):
        settings = make_settings(webiq_endpoint="http://127.0.0.1:9/mock-web")
        self.assertEqual(endpoint_for(settings), "http://127.0.0.1:9/mock-web")
        with self.assertRaisesRegex(ValueError, "webiq_base_url"):
            endpoint_for(settings, "/v3/browse")

    def test_arbitrary_paths_are_rejected(self):
        with self.assertRaises(ValueError):
            endpoint_for(make_settings(), "/v3/not-real")


class TimeoutTest(unittest.TestCase):
    def test_slower_services_have_independent_defaults(self):
        settings = make_settings(
            webiq_timeout=30,
            webiq_browse_timeout=120,
            webiq_classic_timeout=60,
        )
        self.assertEqual(timeout_for(settings, WEB_PATH), 30)
        self.assertEqual(timeout_for(settings, "/v3/browse"), 120)
        self.assertEqual(timeout_for(settings, "/v3/search/classic"), 60)


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
    def test_each_rest_api_uses_its_official_path(self, post):
        post.return_value = upstream_response()

        for path in API_PATHS:
            search(b"{}", make_settings(), api_path=path)
            self.assertEqual(post.call_args[0][0], API_BASE_URL + path)

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
        """401/403 on search APIs is about our key, not the caller's token."""
        for status in (401, 403):
            post.return_value = upstream_response(status, b"bad key")
            with self.assertRaises(WebIQError) as ctx:
                search(b"{}", make_settings())
            self.assertEqual(ctx.exception.status_code, 503)
            self.assertIn("webiq_api_key", str(ctx.exception))

    @patch("ghc_api.webiq.requests.post")
    def test_browse_403_is_preserved_because_it_can_be_content_policy(self, post):
        post.return_value = upstream_response(403, b'{"errorCode":"URL_BLOCKED"}')

        response = search(b"{}", make_settings(), api_path="/v3/browse")

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.content, b'{"errorCode":"URL_BLOCKED"}')

    def test_unconfigured_server_reports_503(self):
        with self.assertRaises(WebIQError) as ctx:
            search(b'{"query":"q"}', make_settings(webiq_api_key=""))

        self.assertEqual(ctx.exception.status_code, 503)


class MCPRequestTest(unittest.TestCase):
    @patch("ghc_api.webiq.requests.request")
    def test_transport_method_headers_and_body_are_forwarded(self, send):
        send.return_value = upstream_response()

        mcp_request("POST", b'{"jsonrpc":"2.0"}', make_settings(),
                    request_headers={
                        "accept": "application/json, text/event-stream",
                        "content-type": "application/json",
                        "mcp-protocol-version": "2026-07-28",
                        "mcp-method": "tools/call",
                        "mcp-name": "web",
                        "mcp-param-region": "us-west1",
                        "mcp-future-extension": "future-value",
                        "mcp-session-id": "session-1",
                        "authorization": "Bearer client-secret",
                        "x-apikey": "client-key",
                    })

        self.assertEqual(send.call_args.args, ("POST", API_BASE_URL + MCP_PATH))
        kwargs = send.call_args.kwargs
        self.assertEqual(kwargs["data"], b'{"jsonrpc":"2.0"}')
        self.assertTrue(kwargs["stream"])
        self.assertEqual(kwargs["headers"]["x-apikey"], "secret")
        self.assertEqual(kwargs["headers"]["mcp-session-id"], "session-1")
        self.assertEqual(kwargs["headers"]["mcp-method"], "tools/call")
        self.assertEqual(kwargs["headers"]["mcp-name"], "web")
        self.assertEqual(kwargs["headers"]["mcp-param-region"], "us-west1")
        self.assertEqual(kwargs["headers"]["mcp-future-extension"], "future-value")
        self.assertNotIn("authorization", kwargs["headers"])
        self.assertNotIn("x-apikey", {
            key for key, value in kwargs["headers"].items()
            if value == "client-key"
        })

    @patch("ghc_api.webiq.requests.request")
    def test_get_opens_a_stream_without_a_body(self, send):
        send.return_value = upstream_response()

        mcp_request("GET", b"", make_settings(), request_headers={
            "accept": "text/event-stream",
            "last-event-id": "event-7",
        })

        self.assertIsNone(send.call_args.kwargs["data"])
        self.assertEqual(send.call_args.kwargs["timeout"], (30, None))

    @patch("ghc_api.webiq.requests.request")
    def test_rejected_server_key_closes_response_and_becomes_503(self, send):
        response = upstream_response(401, b"bad key")
        send.return_value = response

        with self.assertRaises(WebIQError) as ctx:
            mcp_request("POST", b"{}", make_settings(), request_headers={})

        self.assertEqual(ctx.exception.status_code, 503)
        response.close.assert_called_once_with()


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

    def test_counts_each_vertical_and_combined_classic_results(self):
        cases = {
            "/v3/search/videos": ("videoResults", 2),
            "/v3/search/news": ("newsResults", 2),
            "/v3/search/images": ("imageResults", 2),
        }
        for path, (key, expected) in cases.items():
            self.assertEqual(result_count({key: [{}, {}]}, path), expected)
        self.assertEqual(result_count({"webResults": [{}], "imageResults": [{}, {}]},
                                      "/v3/search/classic"), 3)

    def test_unknown_shapes_report_nothing_rather_than_zero(self):
        self.assertIsNone(result_count(None))
        self.assertIsNone(result_count("plain text error"))
        self.assertIsNone(result_count({"error": "x"}))
        self.assertIsNone(result_count({"url": "https://example.com"}, "/v3/browse"))


class RouteTest(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_every_webiq_endpoint_is_auth_gated(self):
        for path in (*API_PATHS, MCP_PATH):
            self.assertIn(path, PROTECTED_PATHS)

    @patch("ghc_api.routes.webiq.search")
    def test_every_rest_endpoint_is_exposed(self, search_mock):
        search_mock.return_value = upstream_response()

        for path in API_PATHS:
            res = self.client.post(path, data=b"{}", content_type="application/json")
            self.assertEqual(res.status_code, 200, path)
            self.assertEqual(search_mock.call_args.kwargs["api_path"], path)

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
    def test_local_failures_use_a_service_specific_error_type(self, _search):
        cases = {
            "/v3/search/web": ("query", "webiq_search_error"),
            "/v3/search/videos": ("query", "webiq_videos_error"),
            "/v3/browse": ("url", "webiq_browse_error"),
        }
        for path, (field, error_type) in cases.items():
            res = self.client.post(path, json={field: "value"})
            self.assertEqual(res.status_code, 503)
            self.assertEqual(res.get_json()["error"]["type"], error_type)

    @patch("ghc_api.routes.webiq.mcp_request")
    def test_mcp_stream_and_session_header_reach_the_client(self, mcp_mock):
        upstream = upstream_response(
            200, b"ignored", headers={
                "content-type": "text/event-stream",
                "Mcp-Session-Id": "session-9",
            })
        upstream.iter_content.return_value = [b"event: message\n", b"data: {}\n\n"]
        mcp_mock.return_value = upstream

        res = self.client.post(MCP_PATH, data=b"{}", headers={
            "Accept": "application/json, text/event-stream",
            "Mcp-Session-Id": "session-1",
        }, content_type="application/json")

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.data, b"event: message\ndata: {}\n\n")
        self.assertEqual(res.headers["Mcp-Session-Id"], "session-9")
        self.assertEqual(res.headers["X-Accel-Buffering"], "no")
        self.assertEqual(res.headers["Cache-Control"], "no-cache")
        request_headers = mcp_mock.call_args.kwargs["request_headers"]
        self.assertEqual(request_headers["mcp-session-id"], "session-1")
        upstream.close.assert_called_once_with()


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
