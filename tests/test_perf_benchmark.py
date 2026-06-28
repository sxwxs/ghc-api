import copy
import json
import os
import statistics
import time
import unittest
from unittest import mock

import ghc_api.routes.anthropic as anthropic_routes
import ghc_api.routes.openai as openai_routes
import ghc_api.state
from ghc_api.app import create_app
from ghc_api.cache import RequestCache


RUN_PERF_BENCHMARKS = os.environ.get("GHC_API_RUN_PERF_BENCHMARKS", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _env_int(name, default):
    value = os.environ.get(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _env_float(name):
    value = os.environ.get(name)
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _percentile(values, percentile):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


class _FakeResponse:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = json.dumps(body)

    def json(self):
        return copy.deepcopy(self._body)


@unittest.skipUnless(
    RUN_PERF_BENCHMARKS,
    "set GHC_API_RUN_PERF_BENCHMARKS=1 to run proxy perf benchmarks",
)
class EndpointPerfBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.iterations = _env_int("GHC_API_PERF_ITERATIONS", 1000)
        self.warmup = _env_int("GHC_API_PERF_WARMUP", 50)
        self.max_mean_ms = _env_float("GHC_API_PERF_MAX_MEAN_MS")
        self.max_p95_ms = _env_float("GHC_API_PERF_MAX_P95_MS")

        self._saved_state = {
            "models": ghc_api.state.state.models,
            "copilot_token": ghc_api.state.state.copilot_token,
            "token_expires_at": ghc_api.state.state.token_expires_at,
            "redirect_anthropic": ghc_api.state.state.redirect_anthropic,
            "enable_tool_call_recovery": ghc_api.state.state.enable_tool_call_recovery,
            "enable_auth": ghc_api.state.state.enable_auth,
            "system_prompt_remove": list(ghc_api.state.state.system_prompt_remove),
            "system_prompt_add": list(ghc_api.state.state.system_prompt_add),
            "tool_result_suffix_remove": list(ghc_api.state.state.tool_result_suffix_remove),
            "max_connection_retries": ghc_api.state.state.max_connection_retries,
            "enable_web_search_proxy": ghc_api.state.state.enable_web_search_proxy,
            "auto_remove_encrypted_content_on_parse_error": ghc_api.state.state.auto_remove_encrypted_content_on_parse_error,
        }

        ghc_api.state.state.models = {
            "data": [
                {"id": "claude-sonnet-4", "supported_endpoints": ["/v1/messages"]},
                {"id": "gpt-5", "supported_endpoints": ["/responses"]},
            ]
        }
        ghc_api.state.state.copilot_token = "test-copilot-token"
        ghc_api.state.state.token_expires_at = time.time() + 3600
        ghc_api.state.state.redirect_anthropic = False
        ghc_api.state.state.enable_tool_call_recovery = False
        ghc_api.state.state.enable_auth = False
        ghc_api.state.state.system_prompt_remove = []
        ghc_api.state.state.system_prompt_add = []
        ghc_api.state.state.tool_result_suffix_remove = []
        ghc_api.state.state.max_connection_retries = 0
        ghc_api.state.state.enable_web_search_proxy = False
        ghc_api.state.state.auto_remove_encrypted_content_on_parse_error = False

        self.cache = RequestCache(max_entries=self.iterations + self.warmup + 10)
        self._patches = [
            mock.patch.object(anthropic_routes, "cache", self.cache),
            mock.patch.object(openai_routes, "cache", self.cache),
            mock.patch("requests.post", side_effect=self._fake_upstream_post),
        ]
        for patcher in self._patches:
            patcher.start()

        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self):
        self.client = None
        for patcher in reversed(self._patches):
            patcher.stop()

        for key, value in self._saved_state.items():
            setattr(ghc_api.state.state, key, value)

    def test_v1_messages_proxy_perf(self):
        payload = {
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 16,
            "stream": False,
        }

        self._benchmark_endpoint("/v1/messages", payload)

    def test_v1_responses_proxy_perf(self):
        payload = {
            "model": "gpt-5",
            "input": "ping",
            "max_output_tokens": 16,
            "stream": False,
        }

        self._benchmark_endpoint("/v1/responses", payload)

    def _benchmark_endpoint(self, endpoint, payload):
        with mock.patch("builtins.print", lambda *args, **kwargs: None):
            for _ in range(self.warmup):
                response = self.client.post(endpoint, json=payload)
                self.assertEqual(response.status_code, 200, response.get_data(as_text=True))

            timings_ms = []
            started = time.perf_counter()
            for _ in range(self.iterations):
                request_started = time.perf_counter()
                response = self.client.post(endpoint, json=payload)
                timings_ms.append((time.perf_counter() - request_started) * 1000)
                self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
            elapsed = time.perf_counter() - started

        mean_ms = statistics.mean(timings_ms)
        median_ms = statistics.median(timings_ms)
        p95_ms = _percentile(timings_ms, 0.95)
        p99_ms = _percentile(timings_ms, 0.99)
        rps = self.iterations / elapsed if elapsed else 0.0

        print(
            f"\n{endpoint} proxy perf: "
            f"iterations={self.iterations} warmup={self.warmup} "
            f"mean={mean_ms:.3f}ms median={median_ms:.3f}ms "
            f"p95={p95_ms:.3f}ms p99={p99_ms:.3f}ms rps={rps:.1f}"
        )

        if self.max_mean_ms is not None:
            self.assertLessEqual(mean_ms, self.max_mean_ms)
        if self.max_p95_ms is not None:
            self.assertLessEqual(p95_ms, self.max_p95_ms)

    def _fake_upstream_post(self, url, **kwargs):
        payload = kwargs.get("json") or {}
        if url.endswith("/v1/messages"):
            return _FakeResponse({
                "id": "msg_perf",
                "type": "message",
                "role": "assistant",
                "model": payload.get("model"),
                "content": [{"type": "text", "text": "pong"}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            })
        if url.endswith("/v1/responses"):
            return _FakeResponse({
                "id": "resp_perf",
                "object": "response",
                "status": "completed",
                "model": payload.get("model"),
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "pong"}],
                    }
                ],
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                },
            })
        raise AssertionError(f"unexpected upstream URL: {url}")


if __name__ == "__main__":
    unittest.main()
