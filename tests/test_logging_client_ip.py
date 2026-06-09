import json
import os
import tempfile
import unittest
from unittest import mock

from ghc_api import utils
from ghc_api.cache import RequestCache


class FakeRequest:
    def __init__(self, headers=None, remote_addr=None):
        self.headers = headers or {}
        self.remote_addr = remote_addr


class GetClientIpTests(unittest.TestCase):
    def test_prefers_x_forwarded_for_first_entry(self):
        request = FakeRequest(
            headers={"X-Forwarded-For": "203.0.113.5, 70.41.3.18, 150.172.238.178"},
            remote_addr="10.0.0.1",
        )
        self.assertEqual(utils.get_client_ip(request), "203.0.113.5")

    def test_falls_back_to_x_real_ip(self):
        request = FakeRequest(headers={"X-Real-IP": "198.51.100.7"}, remote_addr="10.0.0.1")
        self.assertEqual(utils.get_client_ip(request), "198.51.100.7")

    def test_falls_back_to_remote_addr(self):
        request = FakeRequest(remote_addr="192.0.2.44")
        self.assertEqual(utils.get_client_ip(request), "192.0.2.44")

    def test_unknown_when_nothing_available(self):
        request = FakeRequest(remote_addr=None)
        self.assertEqual(utils.get_client_ip(request), "unknown")


class LogErrorRequestTests(unittest.TestCase):
    def test_error_log_has_unix_timestamp_and_client_ip(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(utils, "get_config_dir", return_value=tmp):
                utils.log_error_request(
                    "/v1/messages",
                    {"model": "claude"},
                    "boom",
                    500,
                    client_ip="203.0.113.5",
                )

            with open(os.path.join(tmp, "error.log"), encoding="utf-8") as f:
                entry = json.loads(f.readline())

        self.assertIsInstance(entry["timestamp"], int)
        self.assertGreater(entry["timestamp"], 0)
        self.assertEqual(entry["client_ip"], "203.0.113.5")
        self.assertEqual(entry["endpoint"], "/v1/messages")
        self.assertEqual(entry["status_code"], 500)


class CacheClientIpTests(unittest.TestCase):
    def test_client_ip_stored_on_start_and_preserved_on_complete(self):
        cache = RequestCache()
        cache.start_request("req-1", {
            "client_ip": "203.0.113.5",
            "endpoint": "/v1/messages",
            "model": "claude",
        })
        cache.complete_request("req-1", {
            "response_body": {"ok": True},
            "status_code": 200,
        })

        item = cache.get_request("req-1")
        self.assertEqual(item["client_ip"], "203.0.113.5")

    def test_client_ip_stored_via_add_request_fallback(self):
        cache = RequestCache()
        cache.add_request("req-2", {
            "client_ip": "198.51.100.7",
            "endpoint": "/v1/chat/completions",
            "model": "gpt-5",
            "status_code": 200,
        })

        item = cache.get_request("req-2")
        self.assertEqual(item["client_ip"], "198.51.100.7")


if __name__ == "__main__":
    unittest.main()
