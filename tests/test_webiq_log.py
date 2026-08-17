"""Web IQ search logging.

Two destinations, on purpose:

* ``<config_dir>/webiq/YYYY-MM-DD.jl`` - the full-fidelity record. It is the
  only place a search survives untruncated, so nothing here is abbreviated.
* the shared request cache - so a search shows up in the request list,
  full-text search, detail view and export next to the LLM requests. That copy
  obeys ``cache_max_request_size`` like any other entry.

There is deliberately no third copy in an in-memory Web IQ buffer.

The API key must never reach either destination: the server's key travels in a
header that is never logged, and a client's own ``x-apikey`` (which a client
that "only changed the base URL" still sends) is redacted before its headers
are cached.
"""

import glob
import json
import os
import tempfile
import unittest
from unittest import mock

from ghc_api.app import create_app
from ghc_api.auth import REDACTED_HEADERS, redact_auth_headers
from ghc_api.cache import cache
from ghc_api.routes.webiq import REQUEST_LIST_MODEL, SEARCH_PATH
from ghc_api.state import state
from ghc_api.webiq_log import log_dir, record_search_to_file


def upstream_response(status=200, body=b'{"webResults": []}', headers=None):
    return mock.Mock(
        status_code=status,
        ok=200 <= status < 300,
        content=body,
        text=body.decode("utf-8", "replace"),
        headers=headers or {"content-type": "application/json"},
    )


def read_log_lines():
    lines = []
    for path in sorted(glob.glob(os.path.join(log_dir(), "*.jl"))):
        with open(path, encoding="utf-8") as f:
            lines.extend(json.loads(line) for line in f if line.strip())
    return lines


class IsolatedConfigDirTest(unittest.TestCase):
    """Base class: every test writes into its own throwaway config dir."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        patcher = mock.patch.dict(os.environ, {"GHC_API_CONFIG_DIR": self.tmpdir.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        cache.cache.clear()
        self.addCleanup(cache.cache.clear)
        self.client = create_app().test_client()


class FileLoggingTest(IsolatedConfigDirTest):
    def test_records_are_appended_to_a_daily_file(self):
        record_search_to_file({"id": "a", "query": "one"})
        record_search_to_file({"id": "b", "query": "two"})

        entries = read_log_lines()
        self.assertEqual([e["query"] for e in entries], ["one", "two"])

    def test_file_logging_can_be_disabled(self):
        with mock.patch.object(state, "log_webiq_requests", False):
            record_search_to_file({"id": "a", "query": "one"})

        self.assertEqual(read_log_lines(), [])

    def test_disk_failure_does_not_break_the_search(self):
        with mock.patch("ghc_api.webiq_log.open", side_effect=OSError("disk full")):
            record_search_to_file({"id": "a", "query": "one"})  # must not raise

    def test_non_serializable_values_do_not_lose_the_record(self):
        record_search_to_file({"id": "a", "query": "one", "odd": object()})

        self.assertEqual(len(read_log_lines()), 1)


class RouteLoggingTest(IsolatedConfigDirTest):
    """Every call to the search endpoint is recorded, success or failure."""

    @mock.patch("ghc_api.routes.webiq.search")
    def test_successful_search_is_recorded_in_full(self, search_mock):
        body = (b'{"webResults":[{"title":"t","url":"u","content":"c"}],'
                b'"traceId":"trace-1"}')
        search_mock.return_value = upstream_response(200, body)

        res = self.client.post(SEARCH_PATH, json={"query": "python"})

        self.assertEqual(res.status_code, 200)
        entry = read_log_lines()[0]
        self.assertEqual(entry["query"], "python")
        self.assertEqual(entry["state"], "completed")
        self.assertEqual(entry["status_code"], 200)
        self.assertEqual(entry["result_count"], 1)
        # Full fidelity: the response body is stored as upstream sent it.
        self.assertEqual(entry["response_body"]["webResults"],
                         [{"title": "t", "url": "u", "content": "c"}])
        # A trace id is the first thing upstream support asks for.
        self.assertEqual(entry["trace_id"], "trace-1")
        self.assertEqual(entry["request_body"], {"query": "python"})
        self.assertEqual(entry["upstream"]["status_code"], 200)
        self.assertEqual(entry["user_id"], "anonymous")
        self.assertIsInstance(entry["duration_ms"], int)

    @mock.patch("ghc_api.routes.webiq.search")
    def test_other_rest_services_use_their_own_endpoint_model_and_count(self, search_mock):
        search_mock.return_value = upstream_response(
            200, b'{"newsResults":[{"title":"a"},{"title":"b"}]}')

        res = self.client.post("/v3/search/news", json={"query": "python"})

        self.assertEqual(res.status_code, 200)
        entry = read_log_lines()[0]
        self.assertEqual(entry["type"], "webiq_news")
        self.assertEqual(entry["endpoint"], "/v3/search/news")
        self.assertEqual(entry["result_count"], 2)
        item = self.client.get("/api/requests").get_json()["items"][0]
        self.assertEqual(item["model"], "webiq_news")
        self.assertEqual(item["endpoint"], "/v3/search/news")

    @mock.patch("ghc_api.routes.webiq.search")
    def test_upstream_failure_is_recorded_with_its_body(self, search_mock):
        search_mock.return_value = upstream_response(
            429, b'{"error":{"code":"TooManyRequests"}}')

        res = self.client.post(SEARCH_PATH, json={"query": "python"})

        self.assertEqual(res.status_code, 429)
        entry = read_log_lines()[0]
        self.assertEqual(entry["state"], "error")
        self.assertEqual(entry["status_code"], 429)
        self.assertEqual(entry["upstream"]["status_code"], 429)
        self.assertEqual(entry["response_body"], {"error": {"code": "TooManyRequests"}})
        self.assertIsNone(entry["result_count"])

    @mock.patch("ghc_api.routes.webiq.search")
    def test_local_failure_is_recorded_without_an_upstream_status(self, search_mock):
        from ghc_api.webiq import WebIQError
        search_mock.side_effect = WebIQError("not configured", 503)

        res = self.client.post(SEARCH_PATH, json={"query": "python"})

        self.assertEqual(res.status_code, 503)
        entry = read_log_lines()[0]
        self.assertEqual(entry["state"], "error")
        self.assertEqual(entry["error"], "not configured")
        # None, not 0: the request never reached upstream at all.
        self.assertIsNone(entry["upstream"]["status_code"])

    @mock.patch("ghc_api.routes.webiq.search")
    def test_a_malformed_body_is_still_logged(self, search_mock):
        """Nothing validates the body locally, so logging cannot assume JSON."""
        search_mock.return_value = upstream_response(400, b"query is required")

        res = self.client.post(SEARCH_PATH, data=b"not json",
                               content_type="application/json")

        self.assertEqual(res.status_code, 400)
        entry = read_log_lines()[0]
        self.assertEqual(entry["request_body"], "not json")
        self.assertEqual(entry["response_body"], "query is required")
        self.assertIsNone(entry["query"])

    def test_the_servers_api_key_is_never_recorded(self):
        with mock.patch.object(state, "enable_webiq_search", True), \
                mock.patch.object(state, "webiq_api_key", "super-secret"), \
                mock.patch("ghc_api.webiq.requests.post") as post:
            post.return_value = upstream_response()

            res = self.client.post(SEARCH_PATH, json={"query": "python"})

        self.assertEqual(res.status_code, 200)
        self.assertNotIn("super-secret", json.dumps(read_log_lines()))
        self.assertNotIn("super-secret", json.dumps(cache.get_recent_requests(5)))
        self.assertEqual(post.call_args.kwargs["headers"]["x-apikey"], "super-secret")

    @mock.patch("ghc_api.routes.webiq.search")
    def test_a_clients_own_api_key_is_redacted(self, search_mock):
        """A client that "only changed the base URL" still sends x-apikey.

        Its key must not end up in the request detail view, an export, or
        requests/YYYY-MM-DD.jl.
        """
        search_mock.return_value = upstream_response()

        self.client.post(SEARCH_PATH, json={"query": "python"},
                         headers={"x-apikey": "CLIENT-KEY",
                                  "Authorization": "Bearer CLIENT-TOKEN"})

        recorded = json.dumps(cache.get_recent_requests(5))
        self.assertNotIn("CLIENT-KEY", recorded)
        self.assertNotIn("CLIENT-TOKEN", recorded)

    def test_the_redaction_list_covers_the_web_iq_header(self):
        self.assertIn("x-apikey", REDACTED_HEADERS)
        redacted = redact_auth_headers({"X-Apikey": "k", "X-Api-Key": "k", "Api-Key": "k"})
        self.assertEqual(set(redacted.values()), {"***REDACTED***"})


class RequestListTest(IsolatedConfigDirTest):
    """A search is a request like any other, and shows up in the shared views."""

    @mock.patch("ghc_api.routes.webiq.search")
    def test_search_shows_up_in_the_request_list(self, search_mock):
        search_mock.return_value = upstream_response(
            200, b'{"webResults":[{"title":"t","url":"u","content":"c"}]}')

        self.client.post(SEARCH_PATH, json={"query": "python"})

        items = self.client.get("/api/requests").get_json()["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["model"], REQUEST_LIST_MODEL)
        self.assertEqual(items[0]["endpoint"], SEARCH_PATH)
        self.assertEqual(items[0]["state"], "completed")
        # A search spends Web IQ quota, not model tokens.
        self.assertEqual(items[0]["input_tokens"], 0)
        self.assertEqual(items[0]["output_tokens"], 0)

    @mock.patch("ghc_api.routes.webiq.search")
    def test_detail_view_carries_both_bodies(self, search_mock):
        search_mock.return_value = upstream_response(
            200, b'{"webResults":[{"title":"t","url":"u","content":"c"}]}')

        self.client.post(SEARCH_PATH, json={"query": "python"})

        request_id = self.client.get("/api/requests").get_json()["items"][0]["id"]
        detail = self.client.get(f"/api/request/{request_id}").get_json()
        self.assertEqual(detail["request_body"], {"query": "python"})
        self.assertEqual(detail["response_body"]["webResults"][0]["title"], "t")

    @mock.patch("ghc_api.routes.webiq.search")
    def test_failed_search_is_listed_as_an_error(self, search_mock):
        search_mock.return_value = upstream_response(429, b'{"error":"slow down"}')

        self.client.post(SEARCH_PATH, json={"query": "python"})

        items = self.client.get("/api/requests").get_json()["items"]
        self.assertEqual(items[0]["state"], "error")
        self.assertEqual(items[0]["status_code"], 429)

    @mock.patch("ghc_api.routes.webiq.search")
    def test_search_is_findable_by_full_text(self, search_mock):
        search_mock.return_value = upstream_response(
            200, b'{"webResults":[{"title":"Zebra facts","url":"u","content":"c"}]}')

        self.client.post(SEARCH_PATH, json={"query": "zebra"})

        found = self.client.get("/api/requests/search?q=Zebra").get_json()
        self.assertEqual(found["total"], 1)


class RuntimeConfigTest(IsolatedConfigDirTest):
    def test_log_toggle_is_exposed_and_updatable(self):
        original = state.log_webiq_requests
        self.addCleanup(setattr, state, "log_webiq_requests", original)

        config = self.client.get("/api/runtime-config").get_json()
        self.assertIn("log_webiq_requests", config)

        res = self.client.post("/api/runtime-config", json={"log_webiq_requests": False})
        self.assertEqual(res.status_code, 200)
        self.assertFalse(state.log_webiq_requests)

        res = self.client.post("/api/runtime-config", json={"log_webiq_requests": "yes"})
        self.assertEqual(res.status_code, 400)

    def test_retired_buffer_setting_is_gone(self):
        """The in-memory Web IQ buffer was removed; its knob must not linger."""
        config = self.client.get("/api/runtime-config").get_json()
        self.assertNotIn("webiq_log_max_entries", config)
        self.assertFalse(hasattr(state, "webiq_log_max_entries"))

    def test_retired_dashboard_apis_are_gone(self):
        """Web IQ searches are read from the shared request APIs now."""
        self.assertEqual(self.client.get("/api/webiq/requests").status_code, 404)
        self.assertEqual(self.client.get("/api/webiq/request/x").status_code, 404)


if __name__ == "__main__":
    unittest.main()
