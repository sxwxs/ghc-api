import time
import unittest

from flask import Flask
from unittest import mock

from ghc_api.cache import RequestCache
from ghc_api.routes import dashboard as dashboard_routes


class RequestCacheTimestampTests(unittest.TestCase):
    def test_start_request_uses_unix_timestamp(self) -> None:
        cache = RequestCache()

        cache.start_request("req-1", {"model": "gpt-5", "endpoint": "/v1/chat/completions"})

        item = cache.get_request("req-1")
        self.assertIsNotNone(item)
        self.assertIsInstance(item["timestamp"], int)
        self.assertGreater(item["timestamp"], 0)


class DashboardRequestRoutesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Flask(__name__)
        self.app.register_blueprint(dashboard_routes.dashboard_bp)
        self.client = self.app.test_client()

    def test_list_has_metadata_and_detail_reads_full_record_from_log(self) -> None:
        cache = RequestCache()
        cache.start_request("req-1", {
            "model": "gpt-5", "endpoint": "/v1/chat/completions",
            "request_body": {"messages": ["hello"]},
        })
        cache.complete_request("req-1", {"response_body": {"id": "resp-1"}, "status_code": 200})

        with mock.patch.object(dashboard_routes, "cache", cache):
            detail = self.client.get("/api/request/req-1")
            listing = self.client.get("/api/requests")
            response_body = self.client.get("/api/request/req-1/response-body")

        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.get_json()["request_body"], {"messages": ["hello"]})
        self.assertEqual(detail.get_json()["response_body"], {"id": "resp-1"})
        self.assertEqual(response_body.get_json(), {"id": "resp-1"})
        item = listing.get_json()["items"][0]
        self.assertEqual(item["id"], "req-1")
        self.assertIsInstance(item["timestamp"], int)
        self.assertNotIn("request_body", item)
        self.assertEqual(item["log"]["file"], cache.get_request("req-1")["log"]["file"])

    def test_in_flight_detail_comes_from_memory(self) -> None:
        cache = RequestCache()
        cache.start_request("req-1", {"model": "gpt-5", "request_body": {"q": 1}})

        with mock.patch.object(dashboard_routes, "cache", cache):
            detail = self.client.get("/api/request/req-1")

        self.assertEqual(detail.get_json()["request_body"], {"q": 1})
        self.assertEqual(detail.get_json()["state"], "pending")

    def test_unknown_request_and_mismatched_locator_are_404(self) -> None:
        cache = RequestCache()
        cache.start_request("other", {"model": "gpt-5"})
        cache.complete_request("other", {"response_body": {}})
        locator = cache.get_request("other")["log"]

        with mock.patch.object(dashboard_routes, "cache", cache):
            missing = self.client.get("/api/request/nope")
            wrong = self.client.get("/api/request/nope", query_string=locator)

        self.assertEqual(missing.status_code, 404)
        self.assertEqual(wrong.status_code, 404)


class CancelRequestTests(unittest.TestCase):
    def test_cancel_finalizes_in_flight_request_exactly_once(self) -> None:
        cache = RequestCache()
        cache.start_request("req", {
            "model": "gpt-5",
            "endpoint": "/v1/messages",
            "started_at": time.time() - 7,
        })

        cache.cancel_request("req")
        cache.cancel_request("req")

        item = cache.get_request("req")
        self.assertEqual(item["state"], RequestCache.STATE_ERROR)
        self.assertEqual(item["status_code"], 499)
        self.assertGreaterEqual(item["duration"], 7)
        self.assertEqual(cache.request_count, 1)
        self.assertEqual(cache.model_stats["gpt-5"]["request_count"], 1)

    def test_cancel_leaves_finished_request_alone(self) -> None:
        cache = RequestCache()
        cache.start_request("req", {"model": "gpt-5", "endpoint": "/v1/messages"})
        cache.complete_request("req", {"model": "gpt-5", "endpoint": "/v1/messages", "status_code": 502})

        cache.cancel_request("req")

        self.assertEqual(cache.get_request("req")["status_code"], 502)
        self.assertEqual(cache.request_count, 1)


class EvictionTests(unittest.TestCase):
    def test_eviction_skips_in_flight_requests(self) -> None:
        cache = RequestCache(max_entries=2)
        cache.start_request("streaming", {
            "model": "m", "request_headers": {"X-Trace": "t"},
            "original_request_body": {"prompt": "original"},
        })
        for n in range(3):
            cache.start_request(f"done{n}", {"model": "m"})
            cache.complete_request(f"done{n}", {"status_code": 200})

        # The oldest finished entries were evicted, never the running stream.
        self.assertEqual([e["id"] for e in cache.get_in_flight()], ["streaming"])
        self.assertEqual(len(cache.cache), 2)

        cache.complete_request("streaming", {"raw_events": ["e"], "status_code": 200})
        record = cache.get_record("streaming")
        self.assertEqual(record["original_request_body"], {"prompt": "original"})
        self.assertEqual(record["request_headers"], {"X-Trace": "t"})

    def test_list_grows_past_the_limit_only_while_everything_is_in_flight(self) -> None:
        cache = RequestCache(max_entries=1)
        cache.start_request("a", {"model": "m"})
        cache.start_request("b", {"model": "m"})
        self.assertEqual(len(cache.cache), 2)
        cache.complete_request("a", {"status_code": 200})
        cache.start_request("c", {"model": "m"})
        self.assertEqual(sorted(cache.cache), ["b", "c"])

    def test_list_is_trimmed_back_to_the_limit_after_a_burst(self) -> None:
        cache = RequestCache(max_entries=1)
        for key in ("a", "b", "c"):
            cache.start_request(key, {"model": "m"})
        for key in ("a", "b", "c"):
            cache.complete_request(key, {"status_code": 200})
        self.assertEqual(len(cache.cache), 3)
        cache.start_request("d", {"model": "m"})
        self.assertEqual(list(cache.cache), ["d"])


class FullFidelityRecordTests(unittest.TestCase):
    def test_finished_entry_keeps_no_bodies_in_memory(self) -> None:
        cache = RequestCache()
        cache.start_request("req", {
            "model": "gpt-5",
            "request_headers": {"X": "y"},
            "original_request_body": {"a": 1},
            "request_body": {"a": 1},
        })
        cache.complete_request("req", {
            "raw_events": ["raw"],
            "response_body": {"type": "message"},
            "conversion_report": {"big": "sidecar"},
        })
        entry = cache.get_request("req")
        for key in ("request_headers", "original_request_body", "request_body",
                    "response_body", "raw_events", "conversion_report"):
            self.assertNotIn(key, entry)
        self.assertEqual(entry["state"], RequestCache.STATE_COMPLETED)
        self.assertIsNotNone(entry["log"])

    def test_large_bodies_are_persisted_untruncated(self) -> None:
        cache = RequestCache()
        image = "iVBORw0KGgo" * 300_000  # ~3.3 MB of base64, like a generated PNG
        events = ["x" * 2_000_000, "y" * 2_000_000]
        cache.start_request("req", {"model": "gpt-image", "request_body": {"prompt": "p" * 2_000_000}})
        cache.complete_request("req", {
            "raw_events": events,
            "response_body": {"data": [{"b64_json": image}]},
            "response_size": 8_000_000,
        })
        record = cache.get_record("req")
        self.assertEqual(record["response_body"]["data"][0]["b64_json"], image)
        self.assertEqual(record["raw_events"], events)
        self.assertEqual(record["request_body"]["prompt"], "p" * 2_000_000)
        self.assertNotIn("_truncated", str(record)[:200])

    def test_raw_events_and_terminal_projection_are_both_recorded(self) -> None:
        cache = RequestCache()
        cache.start_request("req", {"request_size": 1})
        cache.complete_request("req", {
            "raw_events": ["raw"],
            "response_body": {"type": "message"},
            "response_size": 3,
        })
        record = cache.get_record("req")
        self.assertEqual(record["raw_events"], ["raw"])
        self.assertEqual(record["response_body"], {"type": "message"})

    def test_failed_write_keeps_metadata_without_locator(self) -> None:
        cache = RequestCache()
        cache.start_request("req", {"model": "gpt-5", "request_body": {"a": 1}})
        with mock.patch("ghc_api.request_log.append_jsonl_line", side_effect=OSError("disk full")):
            cache.complete_request("req", {"response_body": {}})
        entry = cache.get_request("req")
        self.assertIsNone(entry["log"])
        self.assertNotIn("request_body", entry)
        self.assertIsNone(cache.get_record("req"))
        self.assertEqual(cache.request_count, 1)


if __name__ == "__main__":
    unittest.main()
