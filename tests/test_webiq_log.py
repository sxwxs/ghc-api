import json
import os
import tempfile
import unittest
from unittest import mock

from ghc_api.app import create_app
from ghc_api.state import state
from ghc_api.webiq_log import MAX_MEMORY_ENTRIES_LIMIT, WebIQLog, webiq_log


class WebIQLogBufferTest(unittest.TestCase):
    """The in-memory ring buffer keeps only the newest N searches."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        patcher = mock.patch.dict(os.environ, {"GHC_API_CONFIG_DIR": self.tmpdir.name})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_defaults_to_twenty_entries(self):
        self.assertEqual(WebIQLog().max_entries, 20)

    def test_oldest_entries_are_evicted(self):
        log = WebIQLog(max_entries=3)
        for i in range(5):
            log.record({"query": f"q{i}"})

        queries = [entry["query"] for entry in log.recent()]
        self.assertEqual(queries, ["q4", "q3", "q2"])
        self.assertEqual(log.stats()["total_searches"], 5)
        self.assertEqual(log.stats()["buffered"], 3)

    def test_resize_keeps_newest(self):
        log = WebIQLog(max_entries=5)
        for i in range(5):
            log.record({"query": f"q{i}"})

        log.set_max_entries(2)

        self.assertEqual([e["query"] for e in log.recent()], ["q4", "q3"])
        self.assertEqual(log.max_entries, 2)

    def test_resize_is_clamped(self):
        log = WebIQLog()
        log.set_max_entries(10 ** 6)
        self.assertEqual(log.max_entries, MAX_MEMORY_ENTRIES_LIMIT)
        log.set_max_entries(0)
        self.assertEqual(log.max_entries, 1)

    def test_entries_get_id_timestamp_and_state(self):
        log = WebIQLog()

        ok = log.record({"query": "q"})
        failed = log.record({"query": "q", "error": "nope"})

        self.assertTrue(ok["id"])
        self.assertIsInstance(ok["timestamp"], int)
        self.assertEqual(ok["state"], "completed")
        self.assertEqual(failed["state"], "error")
        self.assertEqual(log.stats()["failed_searches"], 1)
        self.assertEqual(log.get(ok["id"])["query"], "q")

    def test_records_are_appended_to_daily_jl_file(self):
        log = WebIQLog()
        with mock.patch.object(state, "log_webiq_requests", True):
            log.record({"query": "python", "results": [{"title": "t"}]})
            log.record({"query": "rust"})

        files = os.listdir(os.path.join(self.tmpdir.name, "webiq"))
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].endswith(".jl"))
        with open(os.path.join(self.tmpdir.name, "webiq", files[0]), encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        self.assertEqual([line["query"] for line in lines], ["python", "rust"])

    def test_file_logging_can_be_disabled_without_losing_memory_buffer(self):
        log = WebIQLog()
        with mock.patch.object(state, "log_webiq_requests", False):
            log.record({"query": "python"})

        self.assertFalse(os.path.exists(os.path.join(self.tmpdir.name, "webiq")))
        self.assertEqual(len(log.recent()), 1)

    def test_disk_failure_does_not_break_the_search(self):
        log = WebIQLog()
        with mock.patch.object(state, "log_webiq_requests", True), \
                mock.patch("ghc_api.webiq_log.open", side_effect=OSError("disk full")):
            entry = log.record({"query": "python"})

        self.assertEqual(entry["query"], "python")
        self.assertEqual(len(log.recent()), 1)


class WebIQRouteLoggingTest(unittest.TestCase):
    """Every call to the search endpoint is recorded, success or failure."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        patcher = mock.patch.dict(os.environ, {"GHC_API_CONFIG_DIR": self.tmpdir.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        webiq_log.clear()
        self.addCleanup(webiq_log.clear)
        self.client = create_app().test_client()

    @mock.patch("ghc_api.routes.webiq.search")
    def test_successful_search_is_recorded(self, search_mock):
        def fake_search(query, settings, max_results=None, trace=None):
            trace.update({
                "endpoint": "https://api.microsoftol.com/v3/search/web",
                "request": {"query": query, "maxResults": 5},
                "status_code": 200,
            })
            return [{"title": "t", "url": "u", "content": "c"}]

        search_mock.side_effect = fake_search

        res = self.client.post("/v1/webiq/search", json={"query": "python"})

        self.assertEqual(res.status_code, 200)
        entries = webiq_log.recent()
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["query"], "python")
        self.assertEqual(entry["state"], "completed")
        self.assertEqual(entry["status_code"], 200)
        self.assertEqual(entry["result_count"], 1)
        self.assertEqual(entry["results"], [{"title": "t", "url": "u", "content": "c"}])
        self.assertEqual(entry["request_body"], {"query": "python"})
        self.assertEqual(entry["upstream"]["status_code"], 200)
        self.assertEqual(entry["upstream"]["request"], {"query": "python", "maxResults": 5})
        self.assertEqual(entry["user_id"], "anonymous")
        self.assertIsInstance(entry["duration_ms"], int)

    @mock.patch("ghc_api.routes.webiq.search")
    def test_upstream_failure_is_recorded(self, search_mock):
        from ghc_api.webiq import WebIQError

        def failing_search(query, settings, max_results=None, trace=None):
            trace["endpoint"] = "https://api.microsoftol.com/v3/search/web"
            trace["status_code"] = 429
            raise WebIQError("Web IQ returned HTTP 429.", 429)

        search_mock.side_effect = failing_search

        res = self.client.post("/v1/webiq/search", json={"query": "python"})

        self.assertEqual(res.status_code, 429)
        entry = webiq_log.recent()[0]
        self.assertEqual(entry["state"], "error")
        self.assertEqual(entry["status_code"], 429)
        self.assertEqual(entry["error"], "Web IQ returned HTTP 429.")
        self.assertEqual(entry["upstream"]["status_code"], 429)
        self.assertEqual(entry["result_count"], 0)

    def test_rejected_request_is_recorded(self):
        res = self.client.post("/v1/webiq/search", json={"query": "  "})

        self.assertEqual(res.status_code, 400)
        entry = webiq_log.recent()[0]
        self.assertEqual(entry["state"], "error")
        self.assertEqual(entry["status_code"], 400)
        self.assertIsNone(entry["query"])

    def test_api_key_is_never_recorded(self):
        with mock.patch.object(state, "enable_webiq_search", True), \
                mock.patch.object(state, "webiq_api_key", "super-secret"), \
                mock.patch("ghc_api.webiq.requests.post") as post:
            post.return_value = mock.Mock(ok=True, status_code=200, **{
                "json.return_value": {"webResults": [{"title": "t", "url": "u", "content": "c"}]}
            })

            res = self.client.post("/v1/webiq/search", json={"query": "python"})

        self.assertEqual(res.status_code, 200)
        recorded = json.dumps(webiq_log.recent()[0])
        self.assertNotIn("super-secret", recorded)
        self.assertIn("x-apikey", post.call_args.kwargs["headers"])


class WebIQDashboardApiTest(unittest.TestCase):
    """The dashboard lists Web IQ searches and can open one in full."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        patcher = mock.patch.dict(os.environ, {"GHC_API_CONFIG_DIR": self.tmpdir.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        webiq_log.clear()
        self.addCleanup(webiq_log.clear)
        self.client = create_app().test_client()

    def test_list_omits_bodies_and_reports_stats(self):
        webiq_log.record({
            "query": "python",
            "user_id": "anonymous",
            "results": [{"title": "t", "url": "u", "content": "c"}],
            "result_count": 1,
            "status_code": 200,
            "request_body": {"query": "python"},
            "upstream": {"request": {"query": "python"}},
        })

        data = self.client.get("/api/webiq/requests").get_json()

        self.assertEqual(len(data["items"]), 1)
        item = data["items"][0]
        self.assertEqual(item["query"], "python")
        self.assertEqual(item["result_count"], 1)
        self.assertNotIn("results", item)
        self.assertNotIn("request_body", item)
        self.assertNotIn("upstream", item)
        self.assertEqual(data["stats"]["total_searches"], 1)
        self.assertTrue(data["log_dir"].endswith("webiq"))

    def test_detail_returns_full_entry(self):
        entry = webiq_log.record({"query": "python", "results": [{"title": "t"}]})

        data = self.client.get(f"/api/webiq/request/{entry['id']}").get_json()

        self.assertEqual(data["results"], [{"title": "t"}])

    def test_missing_detail_is_404(self):
        res = self.client.get("/api/webiq/request/does-not-exist")
        self.assertEqual(res.status_code, 404)

    def test_user_filter(self):
        webiq_log.record({"query": "a", "user_id": "alice"})
        webiq_log.record({"query": "b", "user_id": "bob"})

        data = self.client.get("/api/webiq/requests?user=alice").get_json()

        self.assertEqual([i["query"] for i in data["items"]], ["a"])

    def test_limit_is_validated(self):
        self.assertEqual(self.client.get("/api/webiq/requests?limit=0").status_code, 400)
        self.assertEqual(
            self.client.get(f"/api/webiq/requests?limit={MAX_MEMORY_ENTRIES_LIMIT + 1}").status_code,
            400,
        )

    def test_runtime_config_exposes_and_updates_log_settings(self):
        original_enabled = state.log_webiq_requests
        original_entries = state.webiq_log_max_entries
        self.addCleanup(setattr, state, "log_webiq_requests", original_enabled)
        self.addCleanup(setattr, state, "webiq_log_max_entries", original_entries)
        self.addCleanup(webiq_log.set_max_entries, original_entries)

        config = self.client.get("/api/runtime-config").get_json()
        self.assertIn("log_webiq_requests", config)
        self.assertIn("webiq_log_max_entries", config)

        res = self.client.post("/api/runtime-config", json={
            "log_webiq_requests": False,
            "webiq_log_max_entries": 5,
        })

        self.assertEqual(res.status_code, 200)
        self.assertFalse(state.log_webiq_requests)
        self.assertEqual(webiq_log.max_entries, 5)

        bad = self.client.post("/api/runtime-config", json={"webiq_log_max_entries": 0})
        self.assertEqual(bad.status_code, 400)


class WebIQRequestListTest(unittest.TestCase):
    """A search must also appear in the shared request list, not only in the
    dedicated Web IQ panel."""

    def setUp(self):
        from ghc_api.cache import RequestCache
        from ghc_api.routes import webiq as webiq_routes

        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        patcher = mock.patch.dict(os.environ, {"GHC_API_CONFIG_DIR": self.tmpdir.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        webiq_log.clear()
        self.addCleanup(webiq_log.clear)
        # Isolate the global request cache so the assertions below see only
        # what this test produced.
        self.cache = RequestCache()
        for target in ("ghc_api.routes.webiq.cache", "ghc_api.routes.dashboard.cache"):
            p = mock.patch(target, self.cache)
            p.start()
            self.addCleanup(p.stop)
        self.client = create_app().test_client()

    @mock.patch("ghc_api.routes.webiq.search")
    def test_search_shows_up_in_the_request_list(self, search_mock):
        def fake_search(query, settings, max_results=None, trace=None):
            trace.update({"endpoint": "https://api.microsoftol.com/v3/search/web",
                          "request": {"query": query}, "status_code": 200})
            return [{"title": "t", "url": "u", "content": "c"}]

        search_mock.side_effect = fake_search

        self.client.post("/v1/webiq/search", json={"query": "python"})

        listing = self.client.get("/api/requests").get_json()
        self.assertEqual(listing["total"], 1)
        item = listing["items"][0]
        self.assertEqual(item["endpoint"], "/v1/webiq/search")
        self.assertEqual(item["model"], "webiq_search")
        self.assertEqual(item["status_code"], 200)
        self.assertEqual(item["state"], "completed")
        self.assertGreater(item["response_size"], 0)
        # A search spends Web IQ quota, not model tokens.
        self.assertEqual(item["input_tokens"], 0)
        self.assertEqual(item["output_tokens"], 0)

    @mock.patch("ghc_api.routes.webiq.search")
    def test_detail_view_carries_request_and_response_bodies(self, search_mock):
        search_mock.return_value = [{"title": "t", "url": "u", "content": "c"}]

        self.client.post("/v1/webiq/search", json={"query": "python", "max_results": 3})

        entry_id = self.client.get("/api/requests").get_json()["items"][0]["id"]
        detail = self.client.get(f"/api/request/{entry_id}").get_json()
        self.assertEqual(detail["request_body"], {"query": "python", "max_results": 3})
        self.assertEqual(detail["response_body"]["results"], [{"title": "t", "url": "u", "content": "c"}])
        # Both logs must describe the same search.
        self.assertEqual(webiq_log.recent()[0]["id"], entry_id)

    def test_failed_search_is_listed_as_an_error(self):
        self.client.post("/v1/webiq/search", json={"query": "  "})

        item = self.client.get("/api/requests").get_json()["items"][0]
        self.assertEqual(item["state"], "error")
        self.assertEqual(item["status_code"], 400)

    @mock.patch("ghc_api.routes.webiq.search")
    def test_search_is_findable_by_full_text(self, search_mock):
        search_mock.return_value = []

        self.client.post("/v1/webiq/search", json={"query": "quantum computing"})

        found = self.client.get("/api/requests/search?q=quantum").get_json()
        self.assertEqual(found["total"], 1)
        self.assertEqual(found["items"][0]["endpoint"], "/v1/webiq/search")


if __name__ == "__main__":
    unittest.main()
