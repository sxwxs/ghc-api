import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from threading import Event, Thread
from unittest.mock import Mock, patch

from ghc_api.api_helpers import refresh_copilot_token
from ghc_api.app import create_app, initialize_app
from ghc_api.state import state
from ghc_api.token_manager import (
    GitHubDeviceFlowManager,
    delete_github_token_file,
    get_github_token,
    get_token_file_path,
    github_device_flow_manager,
    request_github_device_code,
    save_github_token_to_file,
)


class TokenFileTests(unittest.TestCase):
    def test_delete_token_file_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "ghc_api.token_manager.get_config_dir", return_value=directory
        ):
            self.assertTrue(save_github_token_to_file("secret"))
            self.assertTrue(os.path.exists(get_token_file_path()))
            self.assertTrue(delete_github_token_file())
            self.assertFalse(os.path.exists(get_token_file_path()))
            self.assertTrue(delete_github_token_file())

    def test_get_token_file_path_does_not_create_config_directory(self):
        with tempfile.TemporaryDirectory() as parent:
            config_dir = os.path.join(parent, "missing-config-dir")
            with patch("ghc_api.token_manager.get_config_dir", return_value=config_dir):
                self.assertEqual(
                    get_token_file_path(),
                    os.path.join(config_dir, "github_token.txt"),
                )
            self.assertFalse(os.path.exists(config_dir))

    def test_interactive_login_records_device_flow_source(self):
        previous_source = state.github_token_source
        try:
            state.github_token_source = "unconfigured"
            with patch.dict(os.environ, {"GITHUB_TOKEN": ""}), patch(
                "ghc_api.token_manager.load_github_token_from_file", return_value=None
            ), patch(
                "ghc_api.token_manager.authenticate_github_device_flow",
                return_value="github-token",
            ), patch(
                "ghc_api.token_manager.save_github_token_to_file", return_value=True
            ):
                self.assertEqual(get_github_token(), "github-token")
            self.assertEqual(state.github_token_source, "device_flow")
        finally:
            state.github_token_source = previous_source


class DeviceFlowRequestTests(unittest.TestCase):
    @patch("ghc_api.token_manager.requests.post")
    def test_device_code_error_response_is_truncated(self, post):
        post.return_value = Mock(ok=False, status_code=502, text="x" * 1000)

        with self.assertRaises(RuntimeError) as caught:
            request_github_device_code()

        message = str(caught.exception)
        self.assertIn("HTTP 502", message)
        self.assertIn("response truncated", message)
        self.assertNotIn("x" * 501, message)


class DeviceFlowManagerTests(unittest.TestCase):
    def test_concurrent_start_requests_create_only_one_device_flow(self):
        manager = GitHubDeviceFlowManager()
        request_started = Event()
        release_request = Event()

        def request_device_code():
            request_started.set()
            self.assertTrue(release_request.wait(timeout=2))
            return {
                "device_code": "secret",
                "user_code": "ABCD-1234",
                "verification_uri": "https://github.com/login/device",
                "expires_in": 900,
                "interval": 5,
            }

        with patch(
            "ghc_api.token_manager.request_github_device_code",
            side_effect=request_device_code,
        ) as request_mock, patch.object(manager, "_complete"):
            first = Thread(target=manager.start)
            first.start()
            self.assertTrue(request_started.wait(timeout=2))

            concurrent_status = manager.start()
            self.assertEqual(concurrent_status["status"], "starting")

            release_request.set()
            first.join(timeout=2)
            self.assertFalse(first.is_alive())

        self.assertEqual(request_mock.call_count, 1)
        self.assertEqual(manager.status()["status"], "pending")

    def test_stale_session_cannot_save_or_activate_its_token(self):
        manager = GitHubDeviceFlowManager()
        stale_session = {"status": "pending", "device_code": "old-secret"}
        replacement_session = {"status": "pending", "device_code": "new-secret"}
        manager._session = replacement_session

        with patch(
            "ghc_api.token_manager.poll_github_device_flow", return_value="old-token"
        ), patch("ghc_api.token_manager.save_github_token_to_file") as save_token:
            manager._complete(stale_session)

        save_token.assert_not_called()
        self.assertIs(manager._session, replacement_session)

    def test_completing_session_cannot_be_replaced(self):
        manager = GitHubDeviceFlowManager()
        completing_session = {"status": "completing"}
        manager._session = completing_session

        with patch("ghc_api.token_manager.request_github_device_code") as request_code:
            status = manager.start()

        request_code.assert_not_called()
        self.assertEqual(status["status"], "completing")
        self.assertIs(manager._session, completing_session)


class CopilotRefreshStatusTests(unittest.TestCase):
    def setUp(self):
        self.saved = {
            name: getattr(state, name)
            for name in (
                "github_token",
                "copilot_token",
                "token_expires_at",
                "token_refresh_last_attempt_at",
                "token_refresh_last_success_at",
                "token_refresh_last_succeeded",
                "token_refresh_last_error",
            )
        }
        state.github_token = "github-token"
        state.copilot_token = None
        state.token_expires_at = 0
        state.token_refresh_last_attempt_at = None
        state.token_refresh_last_success_at = None
        state.token_refresh_last_succeeded = None
        state.token_refresh_last_error = None

    def tearDown(self):
        for name, value in self.saved.items():
            setattr(state, name, value)

    @patch("ghc_api.api_helpers.requests.get")
    def test_success_is_recorded(self, get):
        response = Mock(ok=True)
        response.json.return_value = {"token": "copilot-token", "refresh_in": 1800}
        get.return_value = response

        refresh_copilot_token(force=True)

        self.assertEqual(state.copilot_token, "copilot-token")
        self.assertTrue(state.token_refresh_last_succeeded)
        self.assertIsNotNone(state.token_refresh_last_attempt_at)
        self.assertIsNotNone(state.token_refresh_last_success_at)
        self.assertIsNone(state.token_refresh_last_error)

    @patch("ghc_api.api_helpers.log_upstream_error")
    @patch("ghc_api.api_helpers.requests.get")
    def test_http_failure_is_recorded(self, get, log_upstream_error):
        get.return_value = Mock(ok=False, status_code=502, text="bad gateway")

        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(RuntimeError):
            refresh_copilot_token(force=True)

        self.assertFalse(state.token_refresh_last_succeeded)
        self.assertIn("502", state.token_refresh_last_error)
        message = output.getvalue()
        self.assertIn("temporary GitHub service issue", message)
        self.assertIn("written to error.log", message)
        self.assertIn("ghc-api --delete-github-token", message)
        self.assertIn("ghc-api --github-device-login", message)
        log_upstream_error.assert_called_once()
        logged = log_upstream_error.call_args.kwargs
        self.assertEqual(logged["operation"], "copilot_token_refresh")
        self.assertEqual(logged["status_code"], 502)
        self.assertEqual(logged["response_body"], "bad gateway")


class ApplicationInitializationTests(unittest.TestCase):
    def setUp(self):
        self.saved = {
            name: getattr(state, name)
            for name in (
                "github_token",
                "github_token_source",
                "copilot_token",
                "token_expires_at",
                "token_refresh_last_attempt_at",
                "token_refresh_last_success_at",
                "token_refresh_last_succeeded",
                "token_refresh_last_error",
                "token_usage_reporter_started",
            )
        }
        state.github_token_source = "unconfigured"
        state.copilot_token = None
        state.token_expires_at = 0
        state.token_usage_reporter_started = False

    def tearDown(self):
        for name, value in self.saved.items():
            setattr(state, name, value)

    @patch("ghc_api.token_usage_reporter.start_token_usage_reporter")
    @patch("ghc_api.token_manager.get_github_token", return_value="github-token")
    @patch("ghc_api.api_helpers.log_upstream_error")
    @patch("ghc_api.api_helpers.requests.get")
    def test_startup_survives_github_502(
        self, get, log_upstream_error, _get_github_token, start_reporter
    ):
        get.return_value = Mock(
            ok=False,
            status_code=502,
            text="<!DOCTYPE html>" + ("GitHub Unicorn error " * 100),
        )

        output = io.StringIO()
        with redirect_stdout(output):
            initialize_app()

        message = output.getvalue()
        self.assertIn("Copilot token refresh failed", message)
        self.assertIn("temporary GitHub service issue", message)
        self.assertIn("written to error.log", message)
        self.assertIn("ghc-api --delete-github-token", message)
        self.assertIn("ghc-api --github-device-login", message)
        self.assertIn("token initialization is incomplete", message)
        self.assertIn("response truncated", state.token_refresh_last_error)
        self.assertFalse(state.token_refresh_last_succeeded)
        log_upstream_error.assert_called_once()
        start_reporter.assert_called_once_with()


class TokenManagerRouteTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config.update(TESTING=True)
        self.client = self.app.test_client()

    @patch("ghc_api.routes.dashboard.get_token_usage_overview")
    def test_token_usage_defaults_to_last_week(self, overview):
        overview.return_value = {"machines": [], "rows": []}
        response = self.client.get("/api/config-manager/token-usage")
        self.assertEqual(response.status_code, 200)
        overview.assert_called_once_with("week", user_filter=None)

    def test_device_flow_status_never_exposes_device_code(self):
        with github_device_flow_manager._lock:
            previous = github_device_flow_manager._session
            github_device_flow_manager._session = {
                "status": "pending",
                "device_code": "secret-device-code",
                "user_code": "ABCD-1234",
                "verification_uri": "https://github.com/login/device",
            }
        try:
            with tempfile.TemporaryDirectory() as directory, patch(
                "ghc_api.token_manager.get_config_dir", return_value=directory
            ):
                response = self.client.get("/api/config-manager/token-status")
            self.assertEqual(response.status_code, 200)
            flow = response.get_json()["device_flow"]
            self.assertNotIn("device_code", flow)
            self.assertEqual(flow["user_code"], "ABCD-1234")
        finally:
            with github_device_flow_manager._lock:
                github_device_flow_manager._session = previous

    def test_config_details_page_exists(self):
        response = self.client.get("/code-agent-manager/config")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Config Hash Overview", response.data)

    def test_token_status_is_a_table_at_the_bottom_of_manager_page(self):
        response = self.client.get("/code-agent-manager")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('class="token-status-table"', html)
        self.assertIn('class="btn danger"', html)
        self.assertNotIn("refreshConfigManagerStatus", html)
        self.assertNotIn("refreshSoftwareVersions", html)
        self.assertNotIn("refreshConfigHashOverview", html)
        self.assertIn('url.protocol === "https:"', html)
        self.assertIn('url.hostname === "github.com"', html)
        self.assertGreater(
            html.index("GitHub / Copilot Token Status"),
            html.index("Code Agent Configuration and Tools"),
        )


if __name__ == "__main__":
    unittest.main()
