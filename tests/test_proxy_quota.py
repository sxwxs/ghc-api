"""Tests for upstream quota control.

Design: docs/decisions/UPSTREAM_QUOTA.md, section 11.

All "quota commands" are faked by mocking subprocess.run, so no real command
runs and no network request is made. Time is driven through a replaceable
clock so expiry / retry / calibration can be tested without sleeping.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import requests
import yaml

from ghc_api.app import PROTECTED_PATHS, create_app
from ghc_api.proxy.client import ProxyRequestError, ProxyRuntime
from ghc_api.proxy.config import ProxyConfigError, ProxyRegistry, parse_proxy_config
from ghc_api.proxy.quota import QuotaManager
from ghc_api.routes import dashboard as dashboard_routes
from ghc_api.routes import proxy as proxy_routes
from ghc_api.state import state


FIXED_NOW = 1_000_000.0
_UNSET = object()


def make_config(*, quota=True, cost=2.0, min_remaining=0.0,
                command=_UNSET, interval=3600, timeout=30):
    profile = {
        "auth": {"type": "none"},
        "apis": {
            "chat_completions": {
                "upstream_url": "https://gateway.example.test/chat/completions",
                "request_model": "preserve",
                "response_model": "preserve",
            },
        },
        "models": {
            "limited-model": {"display_name": "Limited", "quota_cost": cost},
            "free-model": {"display_name": "Free"},
        },
    }
    if quota:
        profile["quota"] = {
            "command": ["/path/to/quota-command"] if command is _UNSET else command,
            "refresh_interval_seconds": interval,
            "command_timeout_seconds": timeout,
            "min_remaining": min_remaining,
        }
    return {"proxies": {"demo": profile}}


def quota_output(batches):
    return mock.Mock(
        returncode=0,
        stdout=json.dumps({"batches": batches}),
        stderr="",
    )


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None, lines=None, content=None):
        self.status_code = status_code
        self.ok = status_code < 400
        self.headers = headers or {"Content-Type": "application/json"}
        self._payload = payload
        self._lines = lines or []
        self.closed = False
        if content is not None:
            self.content = content
        elif payload is None:
            self.content = b""
        else:
            self.content = json.dumps(payload).encode("utf-8")
        self.text = self.content.decode("utf-8", errors="replace")

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload

    def iter_lines(self):
        return iter(self._lines)

    def close(self):
        self.closed = True


class QuotaConfigTest(unittest.TestCase):
    def test_valid_quota_config(self):
        profile = parse_proxy_config(make_config()).profiles["demo"]
        self.assertEqual(profile.quota.command, ("/path/to/quota-command",))
        self.assertEqual(profile.quota.refresh_interval_seconds, 3600)
        self.assertEqual(profile.quota.command_timeout_seconds, 30)
        self.assertEqual(profile.quota.min_remaining, 0.0)
        self.assertEqual(profile.models["limited-model"].quota_cost, 2.0)
        self.assertIsNone(profile.models["free-model"].quota_cost)

    def test_quota_cost_requires_quota_block(self):
        with self.assertRaises(ProxyConfigError):
            parse_proxy_config(make_config(quota=False))

    def test_quota_cost_must_be_positive_number(self):
        for value in (0, -1, -0.5, True, "2", float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ProxyConfigError):
                parse_proxy_config(make_config(cost=value))

    def test_min_remaining_must_be_non_negative_number(self):
        for value in (-1, -0.5, True, "1", float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ProxyConfigError):
                parse_proxy_config(make_config(min_remaining=value))

    def test_quota_command_must_be_non_empty_string_list(self):
        for value in (None, [], "cmd", [""], [1]):
            with self.subTest(value=value), self.assertRaises(ProxyConfigError):
                parse_proxy_config(make_config(command=value))

    def test_fingerprint_covers_only_command_and_timeout(self):
        quota = parse_proxy_config(make_config()).profiles["demo"].quota
        self.assertEqual(quota.fingerprint(), (("/path/to/quota-command",), 30))
        self.assertEqual(
            quota.fingerprint(),
            parse_proxy_config(make_config(min_remaining=99, interval=1)).profiles["demo"].quota.fingerprint(),
        )
        self.assertNotEqual(
            quota.fingerprint(),
            parse_proxy_config(make_config(command=["other"])).profiles["demo"].quota.fingerprint(),
        )


class QuotaManagerTest(unittest.TestCase):
    def setUp(self):
        self.saved_enable_auth = state.enable_auth
        state.enable_auth = False
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_dir.name) / "proxies.yaml"
        self._mtime_counter = 0
        self.now = FIXED_NOW

        self.manager = QuotaManager(registry=ProxyRegistry(self.config_path))
        self.manager._now = lambda: self.now
        self.runtime = ProxyRuntime(registry=ProxyRegistry(self.config_path))
        self._patches = [
            mock.patch.object(proxy_routes, "quota_manager", self.manager),
            mock.patch.object(dashboard_routes, "quota_manager", self.manager),
            mock.patch.object(proxy_routes, "proxy_runtime", self.runtime),
        ]
        for patcher in self._patches:
            patcher.start()
        self.app = create_app()

    def tearDown(self):
        for patcher in self._patches:
            patcher.stop()
        self.temp_dir.cleanup()
        state.enable_auth = self.saved_enable_auth

    # -- helpers ----------------------------------------------------------

    def write_config(self, data, *, reset_ledgers=True):
        self.config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        # Force a distinct mtime so the mtime-based registry always reloads.
        # Step whole seconds: NTFS keeps mtime at 100 ns resolution only.
        self._mtime_counter += 1
        stamp = 1_000_000_000_000_000_000 + self._mtime_counter * 1_000_000_000
        os.utime(self.config_path, ns=(stamp, stamp))
        if reset_ledgers:
            with self.manager._lock:
                self.manager._accounts.clear()

    def profile(self):
        return self.manager.registry.snapshot().profiles["demo"]

    def fetch_with(self, completed):
        with mock.patch("ghc_api.proxy.quota.subprocess.run", return_value=completed) as run:
            self.manager._reconcile()
        return run

    def seed(self, batches):
        return self.fetch_with(quota_output(batches))

    def refetch(self, batches):
        with mock.patch("ghc_api.proxy.quota.subprocess.run", return_value=quota_output(batches)):
            self.manager.refresh_now("demo")

    def remaining(self):
        profiles = self.manager.api_snapshot()["profiles"]
        return None if not profiles else profiles[0]["remaining"]

    def alerts(self):
        return self.manager.api_snapshot()["profiles"][0]["alerts"]

    def post_model(self, model="limited-model", **extra):
        body = {"model": model, "messages": []}
        body.update(extra)
        with self.app.test_client() as client:
            return client.post("/proxy/demo/v1/chat/completions", json=body)

    # -- ledger -----------------------------------------------------------

    def test_deducts_earliest_expiring_first_and_spills_to_next(self):
        self.write_config(make_config())
        e1 = int(self.now + 1000)
        e2 = int(self.now + 2000)
        self.seed([
            {"amount": 1, "expires_at": e1},
            {"amount": 5, "expires_at": e2},
        ])
        result = self.manager.reserve(self.profile(), 3)
        self.assertTrue(result.allowed)
        self.assertEqual(result.receipt, [(e1, 1), (e2, 2)])
        self.assertEqual(self.remaining(), 3.0)

    def test_batches_with_same_expiry_are_merged(self):
        self.write_config(make_config())
        e = int(self.now + 3600)
        self.seed([
            {"amount": 2, "expires_at": e},
            {"amount": 3, "expires_at": e},
        ])
        self.assertEqual(self.remaining(), 5.0)
        self.assertEqual(len(self.manager.api_snapshot()["profiles"][0]["batches"]), 1)

    def test_expired_and_nearly_expired_batches_are_unusable(self):
        self.write_config(make_config())
        e_past = int(self.now - 10)
        e_soon = int(self.now + 30)   # inside the 60s margin
        e_ok = int(self.now + 3600)
        self.seed([
            {"amount": 100, "expires_at": e_past},
            {"amount": 100, "expires_at": e_soon},
            {"amount": 10, "expires_at": e_ok},
        ])
        result = self.manager.reserve(self.profile(), 5)
        self.assertTrue(result.allowed)
        self.assertEqual(result.receipt, [(e_ok, 5)])
        self.assertEqual(self.remaining(), 5.0)

    def test_min_remaining_zero_allows_using_exactly_zero(self):
        self.write_config(make_config(min_remaining=0))
        e = int(self.now + 3600)
        self.seed([{"amount": 2, "expires_at": e}])
        self.assertTrue(self.manager.reserve(self.profile(), 2).allowed)
        self.assertEqual(self.remaining(), 0.0)
        second = self.manager.reserve(self.profile(), 2)
        self.assertFalse(second.allowed)
        self.assertEqual(second.reason, "exhausted")

    def test_concurrent_requests_cannot_overdraw(self):
        self.write_config(make_config())
        e = int(self.now + 3600)
        self.seed([{"amount": 10, "expires_at": e}])
        profile = self.profile()
        results = []
        guard = threading.Lock()

        def worker():
            outcome = self.manager.reserve(profile, 2)
            with guard:
                results.append(outcome.allowed)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(results.count(True), 5)
        self.assertEqual(results.count(False), 5)
        self.assertEqual(self.remaining(), 0.0)

    def test_fetch_result_replaces_ledger_as_is(self):
        self.write_config(make_config())
        e_soon = int(self.now + 30)   # inside the 60s margin
        e_ok = int(self.now + 3600)
        self.seed([{"amount": 10, "expires_at": e_ok}])

        def fake_run(*args, **kwargs):
            # A request admitted while the command runs is not re-deducted.
            self.manager.reserve(self.profile(), 2)
            return quota_output([
                {"amount": 5, "expires_at": e_soon},
                {"amount": 10, "expires_at": e_ok},
            ])

        with mock.patch("ghc_api.proxy.quota.subprocess.run", side_effect=fake_run):
            self.manager.refresh_now("demo")
        self.assertEqual(self.remaining(), 10.0)

    def test_reserve_uses_request_profile_min_remaining(self):
        self.write_config(make_config(min_remaining=0))
        e = int(self.now + 3600)
        self.seed([{"amount": 10, "expires_at": e}])
        # The request path already sees the new threshold; the refresh loop
        # has not reconciled yet.
        self.write_config(make_config(min_remaining=9), reset_ledgers=False)
        result = self.manager.reserve(self.profile(), 2)
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "exhausted")

    # -- request admission ------------------------------------------------

    def test_429_below_min_remaining_and_upstream_not_called(self):
        self.write_config(make_config(min_remaining=10))
        e = int(self.now + 3600)
        self.seed([{"amount": 11, "expires_at": e}])
        with mock.patch("ghc_api.proxy.client.requests.post") as post:
            response = self.post_model()
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.get_json()["error"]["code"], "quota_exceeded")
        post.assert_not_called()

    def test_429_message_distinguishes_unavailable_from_exhausted(self):
        self.write_config(make_config())
        first = self.post_model()
        self.assertEqual(first.status_code, 429)
        self.assertIn("unavailable", first.get_json()["error"]["message"])

        self.write_config(make_config(min_remaining=10))
        e = int(self.now + 3600)
        self.seed([{"amount": 11, "expires_at": e}])
        second = self.post_model()
        self.assertEqual(second.status_code, 429)
        self.assertIn("insufficient", second.get_json()["error"]["message"])
        self.assertEqual(
            first.get_json()["error"]["code"], second.get_json()["error"]["code"]
        )

    def test_messages_api_429_has_quota_exceeded_code(self):
        config = make_config()
        config["proxies"]["demo"]["apis"]["messages"] = {
            "upstream_url": "https://gateway.example.test/messages",
            "request_model": "preserve",
            "response_model": "preserve",
        }
        self.write_config(config)
        with mock.patch("ghc_api.proxy.client.requests.post") as post:
            with self.app.test_client() as client:
                response = client.post(
                    "/proxy/demo/v1/messages",
                    json={"model": "limited-model", "messages": [], "max_tokens": 1},
                )
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.get_json()["error"]["code"], "quota_exceeded")
        post.assert_not_called()

    def test_model_without_quota_cost_is_unrestricted(self):
        self.write_config(make_config())
        # No successful fetch: restricted models are unavailable, but a model
        # without quota_cost must sail through the quota check entirely.
        profile = self.profile()
        result = self.manager.reserve(profile, profile.models["free-model"].quota_cost)
        self.assertTrue(result.allowed)
        self.assertEqual(result.receipt, [])

    def test_free_model_bypasses_quota_via_route(self):
        self.write_config(make_config())
        upstream = FakeResponse(payload={"choices": []})
        with mock.patch("ghc_api.proxy.client.requests.post", return_value=upstream) as post:
            response = self.post_model(model="free-model")
        self.assertEqual(response.status_code, 200)
        post.assert_called_once()

    # -- refunds ----------------------------------------------------------

    def test_refund_on_connection_failure(self):
        self.write_config(make_config())
        e = int(self.now + 3600)
        self.seed([{"amount": 10, "expires_at": e}])
        for error in (requests.exceptions.ConnectionError("refused"), ProxyRequestError("boom")):
            with self.subTest(error=type(error).__name__):
                with mock.patch("ghc_api.proxy.client.requests.post", side_effect=error):
                    response = self.post_model()
                self.assertEqual(response.status_code, 503)
                self.assertEqual(self.remaining(), 10.0)

    def test_no_refund_on_read_timeout(self):
        # The request reached the upstream and may have been billed.
        self.write_config(make_config())
        e = int(self.now + 3600)
        self.seed([{"amount": 10, "expires_at": e}])
        with mock.patch("ghc_api.proxy.client.requests.post",
                        side_effect=requests.exceptions.ReadTimeout("slow")):
            response = self.post_model()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.remaining(), 8.0)

    def test_refund_on_upstream_non_2xx(self):
        self.write_config(make_config())
        e = int(self.now + 3600)
        self.seed([{"amount": 10, "expires_at": e}])
        upstream = FakeResponse(status_code=500, payload={"error": "upstream"})
        with mock.patch("ghc_api.proxy.client.requests.post", return_value=upstream):
            response = self.post_model()
        self.assertEqual(response.status_code, 500)
        self.assertEqual(self.remaining(), 10.0)

    def test_streaming_response_keeps_deduction(self):
        self.write_config(make_config())
        e = int(self.now + 3600)
        self.seed([{"amount": 10, "expires_at": e}])
        upstream = FakeResponse(
            status_code=200,
            headers={"Content-Type": "text/event-stream"},
            lines=[b'data: {"choices":[]}', b"", b"data: [DONE]", b""],
        )
        with mock.patch("ghc_api.proxy.client.requests.post", return_value=upstream):
            with self.app.test_client() as client:
                response = client.post(
                    "/proxy/demo/v1/chat/completions",
                    json={"model": "limited-model", "messages": [], "stream": True},
                )
                response.get_data()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.remaining(), 8.0)

    def test_refund_skips_expired_batch(self):
        self.write_config(make_config())
        e = int(self.now + 3600)
        self.seed([{"amount": 10, "expires_at": e}])
        result = self.manager.reserve(self.profile(), 2)
        self.now = e + 1   # the original batch has now expired
        self.manager.refund("demo", result.receipt)
        with self.manager._lock:
            account = self.manager._accounts["demo"]
        with account.lock:
            self.assertEqual(sum(account.batches.values()), 8.0)

    # -- fetch failures / retries -----------------------------------------

    def test_no_fetch_yet_returns_429(self):
        self.write_config(make_config())
        with mock.patch("ghc_api.proxy.client.requests.post") as post:
            response = self.post_model()
        self.assertEqual(response.status_code, 429)
        post.assert_not_called()

    def test_failed_or_invalid_fetch_returns_429(self):
        e = int(self.now + 3600)
        invalid_outputs = [
            mock.Mock(returncode=1, stdout="", stderr="boom"),
            mock.Mock(returncode=0, stdout="not json", stderr=""),
            mock.Mock(returncode=0, stdout='{"foo": 1}', stderr=""),
            mock.Mock(returncode=0, stdout='{"batches": "no"}', stderr=""),
            mock.Mock(returncode=0, stdout=json.dumps(
                {"batches": [{"amount": 5, "expires_at": "soon"}]}), stderr=""),
            mock.Mock(returncode=0, stdout=json.dumps(
                {"batches": [{"amount": -1, "expires_at": e}]}), stderr=""),
            mock.Mock(returncode=0, stdout=json.dumps(
                {"batches": [{"amount": "5", "expires_at": e}]}), stderr=""),
            mock.Mock(returncode=0, stdout=json.dumps(
                {"batches": [{"amount": 5, "expires_at": True}]}), stderr=""),
        ]
        for completed in invalid_outputs:
            with self.subTest(stdout=completed.stdout):
                self.write_config(make_config())
                self.fetch_with(completed)
                result = self.manager.reserve(self.profile(), 2)
                self.assertFalse(result.allowed)
                self.assertEqual(result.reason, "unavailable")

    def test_fetch_timeout_returns_429(self):
        self.write_config(make_config())
        with mock.patch("ghc_api.proxy.quota.subprocess.run",
                        side_effect=subprocess.TimeoutExpired("cmd", 30)):
            self.manager._reconcile()
        result = self.manager.reserve(self.profile(), 2)
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "unavailable")

    def test_undecodable_command_output_is_a_failed_fetch(self):
        command = [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'\\xff\\xfe')"]
        self.write_config(make_config(command=command))
        self.manager._reconcile()
        result = self.manager.reserve(self.profile(), 2)
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "unavailable")
        self.assertIn("invalid quota command output",
                      self.manager.api_snapshot()["profiles"][0]["last_error"])

    def test_unexpected_command_error_does_not_stall_refresh(self):
        self.write_config(make_config())
        error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
        with mock.patch("ghc_api.proxy.quota.subprocess.run", side_effect=error):
            self.manager._reconcile()
        self.assertFalse(self.manager.reserve(self.profile(), 2).allowed)
        # Still scheduled for the 60s retry instead of stuck "in progress".
        self.now += 60
        e = int(self.now + 3600)
        self.seed([{"amount": 10, "expires_at": e}])
        self.assertTrue(self.manager.reserve(self.profile(), 2).allowed)

    def test_retry_after_60_seconds_then_recovers(self):
        self.write_config(make_config())
        e = int(self.now + 3600)
        self.fetch_with(mock.Mock(returncode=1, stdout="", stderr="boom"))
        self.assertFalse(self.manager.reserve(self.profile(), 2).allowed)

        success = quota_output([{"amount": 10, "expires_at": e}])
        self.now += 59
        with mock.patch("ghc_api.proxy.quota.subprocess.run", return_value=success) as run:
            self.manager._reconcile()
            self.assertFalse(run.called)

        self.now += 1
        with mock.patch("ghc_api.proxy.quota.subprocess.run", return_value=success) as run:
            self.manager._reconcile()
            self.assertTrue(run.called)
        self.assertTrue(self.manager.reserve(self.profile(), 2).allowed)
        self.assertEqual(self.remaining(), 8.0)

    # -- calibration ------------------------------------------------------

    def test_calibration_alerts_when_our_ledger_higher(self):
        self.write_config(make_config())
        e = int(self.now + 3600)
        self.seed([{"amount": 10, "expires_at": e}])
        self.refetch([{"amount": 5, "expires_at": e}])
        self.assertEqual(len(self.alerts()), 1)
        alert = self.alerts()[0]
        self.assertEqual(alert["ours"], 10.0)
        self.assertEqual(alert["fetched"], 5.0)
        self.assertEqual(alert["diff"], 5.0)
        self.assertEqual(alert["expires_at"], e)
        self.assertEqual(self.remaining(), 5.0)

    def test_calibration_no_alert_when_our_ledger_lower(self):
        self.write_config(make_config())
        e = int(self.now + 3600)
        self.seed([{"amount": 10, "expires_at": e}])
        self.manager.reserve(self.profile(), 2)   # our ledger 8
        self.refetch([{"amount": 10, "expires_at": e}])
        self.assertEqual(self.alerts(), [])

    def test_calibration_no_alert_for_new_batch(self):
        self.write_config(make_config())
        e1 = int(self.now + 3600)
        e2 = int(self.now + 7200)
        self.seed([{"amount": 10, "expires_at": e1}])
        self.refetch([
            {"amount": 10, "expires_at": e1},
            {"amount": 20, "expires_at": e2},
        ])
        self.assertEqual(self.alerts(), [])

    def test_calibration_alerts_when_batch_used_up_upstream(self):
        self.write_config(make_config())
        e = int(self.now + 3600)
        self.seed([{"amount": 5, "expires_at": e}])
        self.refetch([{"amount": 0, "expires_at": e}])
        self.assertEqual(len(self.alerts()), 1)
        self.assertEqual(self.alerts()[0]["fetched"], 0.0)
        self.assertEqual(self.alerts()[0]["diff"], 5.0)

    def test_calibration_ignores_expired_batch(self):
        self.write_config(make_config())
        e = int(self.now + 3600)
        self.seed([{"amount": 5, "expires_at": e}])
        self.now = e + 1
        self.refetch([])
        self.assertEqual(self.alerts(), [])

    def test_calibration_ignores_small_deltas(self):
        self.write_config(make_config())
        e = int(self.now + 3600)
        self.seed([{"amount": 10, "expires_at": e}])
        self.refetch([{"amount": 9.995, "expires_at": e}])
        self.assertEqual(self.alerts(), [])

    def test_calibration_skipped_after_failed_fetch(self):
        self.write_config(make_config())
        e = int(self.now + 3600)
        self.seed([{"amount": 10, "expires_at": e}])
        with mock.patch("ghc_api.proxy.quota.subprocess.run",
                        return_value=mock.Mock(returncode=1, stdout="", stderr="boom")):
            self.manager.refresh_now("demo")
        self.refetch([{"amount": 1, "expires_at": e}])
        self.assertEqual(self.alerts(), [])

    def test_calibration_keeps_only_last_50_alerts(self):
        self.write_config(make_config())
        e = int(self.now + 3600)
        self.seed([{"amount": 100, "expires_at": e}])
        for value in range(99, 39, -1):
            self.refetch([{"amount": value, "expires_at": e}])
        self.assertEqual(len(self.alerts()), 50)
        # The deque keeps the most recent entries.
        self.assertEqual(self.alerts()[-1]["ours"], 41.0)
        self.assertEqual(self.alerts()[-1]["fetched"], 40.0)

    # -- config hot reload ------------------------------------------------

    def test_changing_command_clears_ledger_and_refetches(self):
        self.write_config(make_config(command=["cmd-a"]))
        e = int(self.now + 3600)
        self.seed([{"amount": 10, "expires_at": e}])
        self.manager.reserve(self.profile(), 2)
        self.assertEqual(self.remaining(), 8.0)

        self.write_config(make_config(command=["cmd-b"]), reset_ledgers=False)
        with mock.patch("ghc_api.proxy.quota.subprocess.run",
                        return_value=quota_output([{"amount": 20, "expires_at": e}])) as run:
            self.manager._reconcile()
            self.assertTrue(run.called)
        self.assertEqual(self.remaining(), 20.0)

    def test_changing_min_remaining_does_not_refetch(self):
        self.write_config(make_config(min_remaining=0))
        e = int(self.now + 3600)
        self.seed([{"amount": 10, "expires_at": e}])
        self.write_config(make_config(min_remaining=5), reset_ledgers=False)
        with mock.patch("ghc_api.proxy.quota.subprocess.run") as run:
            self.manager._reconcile()
            self.assertFalse(run.called)
        self.assertEqual(self.remaining(), 10.0)

    def test_removing_profile_drops_ledger(self):
        self.write_config(make_config())
        e = int(self.now + 3600)
        self.seed([{"amount": 10, "expires_at": e}])
        self.write_config({"proxies": {}}, reset_ledgers=False)
        self.manager._reconcile()
        self.assertEqual(self.manager.api_snapshot()["profiles"], [])

    # -- dashboard --------------------------------------------------------

    def test_api_quota_payload_is_numeric_only(self):
        self.write_config(make_config())
        e = int(self.now + 3600)
        self.seed([{"amount": 10, "expires_at": e}])
        with self.app.test_client() as client:
            response = client.get("/api/quota")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(len(data["profiles"]), 1)
        entry = data["profiles"][0]
        self.assertEqual(entry["profile"], "demo")
        self.assertEqual(entry["remaining"], 10.0)
        self.assertEqual(entry["min_remaining"], 0.0)
        self.assertEqual(len(entry["batches"]), 1)
        self.assertIsNone(entry["last_error"])
        self.assertEqual(entry["alerts"], [])
        self.assertNotIn("/path/to/quota-command", json.dumps(data))

    def test_api_quota_empty_when_no_quota_profiles(self):
        self.write_config({"proxies": {}})
        with self.app.test_client() as client:
            response = client.get("/api/quota")
        self.assertEqual(response.get_json(), {"profiles": []})

    def test_api_quota_does_not_expose_command_on_failure(self):
        self.write_config(make_config())
        with mock.patch(
            "ghc_api.proxy.quota.subprocess.run",
            side_effect=FileNotFoundError(2, "No such file", "/path/to/quota-command"),
        ):
            self.manager._reconcile()
        with self.app.test_client() as client:
            data = client.get("/api/quota").get_json()
        self.assertEqual(data["profiles"][0]["remaining"], 0.0)
        self.assertNotIn("/path/to/quota-command", json.dumps(data))

    def test_api_quota_is_not_auth_gated(self):
        # Same as /api/stats: the dashboard has no way to send a token.
        self.assertNotIn("/api/quota", PROTECTED_PATHS)
        state.enable_auth = True
        with self.app.test_client() as client:
            self.assertEqual(client.get("/api/quota").status_code, 200)


if __name__ == "__main__":
    unittest.main()
