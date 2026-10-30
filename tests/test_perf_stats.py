"""Tests for live throughput metrics (tokens/second) and the /live dashboard."""

import json
import time
import unittest
from unittest import mock

from flask import Flask

from ghc_api.cache import RequestCache
from ghc_api.perf_stats import PerfStats, parse_window, perf_stats, tokens_per_second
from ghc_api.routes import dashboard as dashboard_routes
from ghc_api.sse import AnthropicDirectStreamHandler
from ghc_api.sse import base as sse_base


class _FakeResponse:
    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code

    def iter_lines(self):
        yield from self._lines

    def close(self):
        pass


class TokensPerSecondTests(unittest.TestCase):
    def test_excludes_time_to_first_token(self) -> None:
        # 100 tokens, 12s total, 2s waiting for the first one -> 10 tok/s.
        self.assertEqual(tokens_per_second(100, 12.0, 2.0), 10.0)

    def test_falls_back_to_total_duration_without_ttft(self) -> None:
        self.assertEqual(tokens_per_second(100, 10.0, None), 10.0)

    def test_unmeasurable_requests_return_none(self) -> None:
        self.assertIsNone(tokens_per_second(0, 10.0, 1.0))
        # Sub-50ms generation is clock noise, not a 10k tok/s model.
        self.assertIsNone(tokens_per_second(10, 1.01, 1.0))


class ParseWindowTests(unittest.TestCase):
    def test_presets_and_raw_seconds(self) -> None:
        self.assertEqual(parse_window("1h"), ("1h", 3600))
        self.assertEqual(parse_window("all"), ("all", None))
        self.assertEqual(parse_window("90"), ("90s", 90))

    def test_unknown_values_fall_back_to_default(self) -> None:
        self.assertEqual(parse_window("nonsense"), ("1h", 3600))
        self.assertEqual(parse_window(None), ("1h", 3600))


class PerfStatsSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stats = PerfStats(capacity=100)

    def _record(self, model: str, finished_at: float, **kwargs) -> None:
        defaults = {
            "endpoint": "/v1/messages",
            "status_code": 200,
            "duration": 11.0,
            "output_tokens": 100,
            "input_tokens": 1000,
            "ttft": 1.0,
        }
        defaults.update(kwargs)
        self.stats.record(model=model, finished_at=finished_at, **defaults)

    def test_window_filters_old_samples(self) -> None:
        now = 10_000.0
        self._record("fast-model", now - 10)
        self._record("fast-model", now - 7200)

        recent = self.stats.snapshot(window_seconds=3600, now=now)
        self.assertEqual(recent["overall"]["requests"], 1)

        everything = self.stats.snapshot(window_seconds=None, now=now)
        self.assertEqual(everything["overall"]["requests"], 2)

    def test_per_model_rates(self) -> None:
        now = 10_000.0
        # 100 tokens in 10 generating seconds -> 10 tok/s.
        self._record("slow-model", now - 5)
        # 500 tokens in 5 generating seconds -> 100 tok/s.
        self._record("fast-model", now - 5, duration=6.0, output_tokens=500)

        snapshot = self.stats.snapshot(window_seconds=3600, now=now)
        by_model = {m["model"]: m for m in snapshot["models"]}

        self.assertEqual(by_model["slow-model"]["tps"], 10.0)
        self.assertEqual(by_model["fast-model"]["tps"], 100.0)
        # Token-weighted overall: 600 tokens / 15 generating seconds.
        self.assertEqual(snapshot["overall"]["tps"], 40.0)
        # Sorted by produced tokens, so the busiest model leads.
        self.assertEqual(snapshot["models"][0]["model"], "fast-model")

    def test_errors_are_counted_but_not_measured(self) -> None:
        now = 10_000.0
        self._record("m", now - 5)
        self._record("m", now - 4, status_code=500, output_tokens=0)

        model = self.stats.snapshot(now=now)["models"][0]
        self.assertEqual(model["requests"], 2)
        self.assertEqual(model["errors"], 1)
        self.assertEqual(model["measured_requests"], 1)
        self.assertEqual(model["tps"], 10.0)

    def test_non_streaming_requests_excluded_from_ttft(self) -> None:
        now = 10_000.0
        self._record("m", now - 5, ttft=None)

        model = self.stats.snapshot(now=now)["models"][0]
        self.assertEqual(model["requests"], 1)
        self.assertEqual(model["streaming_requests"], 0)
        self.assertIsNone(model["ttft_median"])

    def test_capacity_wrap_is_reported_as_truncated(self) -> None:
        stats = PerfStats(capacity=3)
        now = 10_000.0
        for index in range(5):
            stats.record(
                model="m",
                endpoint="/v1/messages",
                status_code=200,
                duration=11.0,
                output_tokens=100,
                ttft=1.0,
                finished_at=now - 100 + index,
            )

        snapshot = stats.snapshot(window_seconds=86400, now=now)
        self.assertEqual(snapshot["samples_retained"], 3)
        self.assertEqual(snapshot["samples_dropped"], 2)
        self.assertEqual(snapshot["samples_recorded"], 5)
        self.assertTrue(snapshot["truncated"])

        # A window the retained samples do fully cover is not truncated.
        self.assertFalse(stats.snapshot(window_seconds=5, now=now)["truncated"])

    def test_user_filter(self) -> None:
        now = 10_000.0
        self._record("m", now - 5, user_id="alice")
        self._record("m", now - 5, user_id="bob")

        self.assertEqual(self.stats.snapshot(now=now, user_id="alice")["overall"]["requests"], 1)
        self.assertEqual(self.stats.snapshot(now=now)["overall"]["requests"], 2)


class CacheSpeedFieldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cache = RequestCache()

    def test_receiving_state_stamps_ttft(self) -> None:
        self.cache.start_request("req-1", {"model": "gpt-5", "endpoint": "/v1/responses"})
        self.cache.update_request_state("req-1", RequestCache.STATE_RECEIVING)

        entry = self.cache.get_request("req-1")
        self.assertIsNotNone(entry["ttft"])
        self.assertGreaterEqual(entry["ttft"], 0.0)

        first_ttft = entry["ttft"]
        # Later chunks must not move the first-token mark.
        self.cache.update_request_state("req-1", RequestCache.STATE_RECEIVING)
        self.assertEqual(self.cache.get_request("req-1")["ttft"], first_ttft)

    def test_complete_request_computes_tokens_per_second(self) -> None:
        self.cache.start_request("req-1", {"model": "gpt-5", "endpoint": "/v1/responses"})
        entry = self.cache.get_request("req-1")
        entry["ttft"] = 2.0

        self.cache.complete_request("req-1", {
            "status_code": 200,
            "output_tokens": 100,
            "duration": 12.0,
            "model": "gpt-5",
            "endpoint": "/v1/responses",
        })

        self.assertEqual(self.cache.get_request("req-1")["tokens_per_second"], 10.0)

    def test_in_flight_lists_unfinished_requests_only(self) -> None:
        self.cache.start_request("live-1", {"model": "gpt-5", "endpoint": "/v1/responses"})
        self.cache.update_request_state("live-1", RequestCache.STATE_RECEIVING)
        self.cache.start_request("done-1", {"model": "gpt-5", "endpoint": "/v1/responses"})
        self.cache.complete_request("done-1", {"status_code": 200, "duration": 1.0})

        in_flight = self.cache.get_in_flight()
        self.assertEqual([row["id"] for row in in_flight], ["live-1"])
        self.assertEqual(in_flight[0]["state"], "receiving")
        self.assertGreaterEqual(in_flight[0]["elapsed"], 0.0)

    def test_import_does_not_feed_live_metrics(self) -> None:
        from ghc_api.perf_stats import perf_stats

        before = perf_stats.snapshot(window_seconds=None)["samples_recorded"]
        self.cache.import_request({
            "id": "imported-1",
            "model": "gpt-5",
            "endpoint": "/v1/responses",
            "output_tokens": 100,
            "duration": 10.0,
        })
        after = perf_stats.snapshot(window_seconds=None)["samples_recorded"]

        self.assertEqual(before, after)
        # The derived speed is still shown for the imported row.
        self.assertEqual(self.cache.get_request("imported-1")["tokens_per_second"], 10.0)


class StreamingHandlerSpeedTests(unittest.TestCase):
    """End-to-end: a real SSE handler must produce TTFT, speed, and a sample."""

    def setUp(self) -> None:
        self.cache = RequestCache()
        self._cache_patch = mock.patch.object(sse_base, "cache", self.cache)
        self._cache_patch.start()
        perf_stats.reset()

    def tearDown(self) -> None:
        self._cache_patch.stop()
        perf_stats.reset()

    def test_stream_records_ttft_speed_and_sample(self) -> None:
        message_start = json.dumps({
            "type": "message_start",
            "message": {"model": "claude-opus-4", "usage": {"input_tokens": 11}},
        })
        message_delta = json.dumps({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 100},
        })
        handler = AnthropicDirectStreamHandler(
            response=_FakeResponse([
                b"event: message_start",
                message_start.encode(),
                b"event: message_delta",
                f"data: {message_delta}".encode(),
                b"data: [DONE]",
            ]),
            request_id="req-1",
            request_size=42,
            # Pretend the request started 10 seconds ago so the measured
            # generation time is stable rather than ~0.
            start_time=time.time() - 10,
            original_model="claude-opus-4",
            translated_model="claude-opus-4",
            request_body_for_cache={"model": "claude-opus-4"},
        )
        list(handler._generate())

        entry = self.cache.get_request("req-1")
        self.assertIsNotNone(entry["ttft"])
        self.assertEqual(entry["output_tokens"], 100)
        # 100 tokens over ~10 generating seconds.
        self.assertGreater(entry["tokens_per_second"], 5)
        self.assertLess(entry["tokens_per_second"], 20)

        snapshot = perf_stats.snapshot(window_seconds=3600)
        self.assertEqual(snapshot["overall"]["requests"], 1)
        self.assertEqual(snapshot["models"][0]["model"], "claude-opus-4")


class LiveStatsRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Flask(__name__, template_folder="../ghc_api/templates")
        self.app.register_blueprint(dashboard_routes.dashboard_bp)
        self.client = self.app.test_client()
        dashboard_routes.perf_stats.reset()

    def tearDown(self) -> None:
        dashboard_routes.perf_stats.reset()

    def test_live_stats_payload(self) -> None:
        dashboard_routes.perf_stats.record(
            model="claude-sonnet-5",
            endpoint="/v1/messages",
            status_code=200,
            duration=11.0,
            output_tokens=100,
            input_tokens=5000,
            ttft=1.0,
        )

        response = self.client.get("/api/live-stats?window=1h")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()

        self.assertEqual(payload["window"], "1h")
        self.assertEqual(payload["window_seconds"], 3600)
        self.assertEqual(payload["overall"]["requests"], 1)
        self.assertEqual(payload["models"][0]["model"], "claude-sonnet-5")
        self.assertEqual(payload["models"][0]["tps"], 10.0)
        self.assertIn("in_flight", payload)

    def test_live_stats_window_all(self) -> None:
        response = self.client.get("/api/live-stats?window=all")
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.get_json()["window_seconds"])

    def test_live_page_renders(self) -> None:
        response = self.client.get("/live")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Live Throughput", response.data)


if __name__ == "__main__":
    unittest.main()
