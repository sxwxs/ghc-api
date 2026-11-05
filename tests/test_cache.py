import json
import time
import unittest
from datetime import datetime
from unittest import mock

from flask import Flask

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

    def test_import_request_converts_iso_timestamp_to_unix(self) -> None:
        cache = RequestCache()
        iso_timestamp = "2026-03-22T10:20:30"

        cache.import_request({
            "id": "req-1",
            "timestamp": iso_timestamp,
            "model": "gpt-5",
            "endpoint": "/v1/chat/completions",
        })

        item = cache.get_request("req-1")
        self.assertIsNotNone(item)
        self.assertEqual(
            item["timestamp"],
            int(datetime.fromisoformat(iso_timestamp).timestamp()),
        )

    def test_import_request_keeps_numeric_timestamp(self) -> None:
        cache = RequestCache()

        cache.import_request({
            "id": "req-1",
            "timestamp": 1711111111,
            "model": "gpt-5",
            "endpoint": "/v1/chat/completions",
        })

        item = cache.get_request("req-1")
        self.assertIsNotNone(item)
        self.assertEqual(item["timestamp"], 1711111111)


class DashboardRequestTimestampTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Flask(__name__)
        self.app.register_blueprint(dashboard_routes.dashboard_bp)
        self.client = self.app.test_client()

    def test_request_routes_return_unix_timestamp_for_imported_legacy_record(self) -> None:
        cache = RequestCache()
        iso_timestamp = "2026-03-22T10:20:30"
        expected_timestamp = int(datetime.fromisoformat(iso_timestamp).timestamp())
        cache.import_request({
            "id": "req-1",
            "timestamp": iso_timestamp,
            "request_body": {"messages": []},
            "response_body": {"id": "resp-1"},
            "model": "gpt-5",
            "endpoint": "/v1/chat/completions",
        })

        with mock.patch.object(dashboard_routes, "cache", cache):
            detail_response = self.client.get("/api/request/req-1")
            list_response = self.client.get("/api/requests")

        self.assertEqual(detail_response.status_code, 200)
        self.assertEqual(list_response.status_code, 200)

        detail_data = detail_response.get_json()
        self.assertEqual(detail_data["timestamp"], expected_timestamp)
        self.assertIsInstance(detail_data["timestamp"], int)

        list_data = list_response.get_json()
        self.assertEqual(list_data["items"][0]["timestamp"], expected_timestamp)
        self.assertIsInstance(list_data["items"][0]["timestamp"], int)


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


class RequestCacheTranslatedStreamTests(unittest.TestCase):
    def test_raw_events_and_terminal_projection_share_cache_when_small(self) -> None:
        cache = RequestCache(max_request_size=1024)
        cache.start_request("req", {"request_size": 1})
        cache.complete_request("req", {
            "raw_events": ["raw"],
            "response_body": {"type": "message"},
            "response_size": 3,
        })
        item = cache.get_request("req")
        self.assertEqual(item["raw_events"], ["raw"])
        self.assertEqual(item["response_body"], {"type": "message"})

    def test_both_stream_views_are_truncated_at_dashboard_limit(self) -> None:
        cache = RequestCache(max_request_size=4)
        cache.start_request("req", {"request_size": 1})
        cache.complete_request("req", {
            "raw_events": ["private raw event"],
            "response_body": {"private": "terminal projection"},
            "response_size": 100,
        })
        item = cache.get_request("req")
        self.assertTrue(json.loads(item["raw_events"][0])["_truncated"])
        self.assertTrue(item["response_body"]["_truncated"])


if __name__ == "__main__":
    unittest.main()
