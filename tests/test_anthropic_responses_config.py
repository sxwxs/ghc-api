import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

import ghc_api.state
from ghc_api.app import create_app
from ghc_api.generate_config import generate_config_file
from ghc_api.main import apply_anthropic_responses_config


WEB_SEARCH_CONFIG_FIELDS = (
    "enable_web_search_proxy",
    "web_search_proxy_endpoint",
)

CONFIG_FIELDS = (
    "anthropic_responses_compat_enabled",
    "anthropic_responses_compat_mode",
    "anthropic_responses_wire_profile",
    "anthropic_responses_model_profiles",
    "anthropic_responses_replay_path",
    "anthropic_responses_replay_ttl_seconds",
    "anthropic_responses_replay_max_bytes",
    "anthropic_responses_replay_max_tenant_bytes",
    "anthropic_responses_replay_max_record_bytes",
    "anthropic_responses_replay_encryption_key_env",
    "anthropic_responses_replay_require_trusted_tenant",
    "anthropic_responses_replay_trusted_single_user",
)


class AnthropicResponsesRuntimeConfigTest(unittest.TestCase):
    def setUp(self):
        self.state = ghc_api.state.state
        self.saved = {
            field: getattr(self.state, field)
            if not isinstance(getattr(self.state, field), dict)
            else dict(getattr(self.state, field))
            for field in CONFIG_FIELDS
        }
        self.state.anthropic_responses_compat_enabled = True
        self.state.anthropic_responses_compat_mode = "compatibility"
        self.state.anthropic_responses_wire_profile = "copilot_responses_lite"
        self.state.anthropic_responses_model_profiles = {
            "gpt-5.6-sol": "copilot_responses_lite",
        }
        self.state.anthropic_responses_replay_path = ""
        self.state.anthropic_responses_replay_ttl_seconds = 86400
        self.state.anthropic_responses_replay_max_bytes = 1073741824
        self.state.anthropic_responses_replay_max_tenant_bytes = 268435456
        self.state.anthropic_responses_replay_max_record_bytes = 67108864
        self.state.anthropic_responses_replay_encryption_key_env = "GHC_REPLAY_ENCRYPTION_KEY"
        self.state.anthropic_responses_replay_require_trusted_tenant = True
        self.state.anthropic_responses_replay_trusted_single_user = False
        self.app = create_app()

    def tearDown(self):
        for field, value in self.saved.items():
            setattr(self.state, field, value)

    def test_get_exposes_every_anthropic_responses_setting(self):
        with self.app.test_client() as client:
            response = client.get("/api/runtime-config")

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(
            {field: body[field] for field in CONFIG_FIELDS},
            {field: getattr(self.state, field) for field in CONFIG_FIELDS},
        )

    def test_post_updates_every_anthropic_responses_setting(self):
        update = {
            "anthropic_responses_compat_enabled": False,
            "anthropic_responses_compat_mode": "lossless_required",
            "anthropic_responses_wire_profile": "public_responses",
            "anthropic_responses_model_profiles": {
                "gpt-sanitized": "public_responses",
                "gpt-copilot-sanitized": "copilot_responses_lite",
            },
            "anthropic_responses_replay_path": "replay-sanitized.sqlite3",
            "anthropic_responses_replay_ttl_seconds": 7200,
            "anthropic_responses_replay_max_bytes": 8388608,
            "anthropic_responses_replay_max_tenant_bytes": 4194304,
            "anthropic_responses_replay_max_record_bytes": 1048576,
            "anthropic_responses_replay_encryption_key_env": "TEST_REPLAY_KEY",
            "anthropic_responses_replay_require_trusted_tenant": False,
            "anthropic_responses_replay_trusted_single_user": True,
        }

        with self.app.test_client() as client:
            response = client.post("/api/runtime-config", json=update)

        self.assertEqual(response.status_code, 200)
        body = response.get_json()["config"]
        for field, expected in update.items():
            self.assertEqual(getattr(self.state, field), expected)
            self.assertEqual(body[field], expected)

    def test_post_rejects_invalid_types_and_unknown_values(self):
        invalid_values = (
            ("anthropic_responses_compat_enabled", 1),
            ("anthropic_responses_compat_mode", "strict"),
            ("anthropic_responses_compat_mode", 1),
            ("anthropic_responses_wire_profile", "future_profile"),
            ("anthropic_responses_wire_profile", ""),
            ("anthropic_responses_model_profiles", []),
            ("anthropic_responses_model_profiles", {"model": "future_profile"}),
            ("anthropic_responses_model_profiles", {"": "public_responses"}),
            ("anthropic_responses_replay_path", None),
            ("anthropic_responses_replay_ttl_seconds", True),
            ("anthropic_responses_replay_ttl_seconds", 0),
            ("anthropic_responses_replay_max_bytes", "1024"),
            ("anthropic_responses_replay_max_bytes", -1),
            ("anthropic_responses_replay_max_tenant_bytes", True),
            ("anthropic_responses_replay_max_tenant_bytes", 0),
            ("anthropic_responses_replay_max_record_bytes", "1024"),
            ("anthropic_responses_replay_max_record_bytes", -1),
            ("anthropic_responses_replay_encryption_key_env", None),
            ("anthropic_responses_replay_require_trusted_tenant", 0),
            ("anthropic_responses_replay_trusted_single_user", "true"),
        )

        with self.app.test_client() as client:
            for field, invalid in invalid_values:
                with self.subTest(field=field, invalid=invalid):
                    response = client.post("/api/runtime-config", json={field: invalid})
                    self.assertEqual(response.status_code, 400)
                    self.assertIn(field, response.get_json()["error"])

    def test_quota_hierarchy_is_validated_atomically(self):
        invalid_updates = (
            {
                "anthropic_responses_replay_max_bytes": 1024,
                "anthropic_responses_replay_max_tenant_bytes": 2048,
                "anthropic_responses_replay_max_record_bytes": 512,
            },
            {
                "anthropic_responses_replay_max_bytes": 4096,
                "anthropic_responses_replay_max_tenant_bytes": 2048,
                "anthropic_responses_replay_max_record_bytes": 3072,
            },
        )
        original = {
            field: getattr(self.state, field)
            for field in (
                "anthropic_responses_replay_max_bytes",
                "anthropic_responses_replay_max_tenant_bytes",
                "anthropic_responses_replay_max_record_bytes",
            )
        }
        with self.app.test_client() as client:
            for update in invalid_updates:
                with self.subTest(update=update):
                    response = client.post("/api/runtime-config", json=update)
                    self.assertEqual(response.status_code, 400)
                    self.assertEqual(
                        {field: getattr(self.state, field) for field in original},
                        original,
                    )

        for update in invalid_updates:
            with self.subTest(yaml_update=update):
                with self.assertRaises(ValueError):
                    apply_anthropic_responses_config(update)
                self.assertEqual(
                    {field: getattr(self.state, field) for field in original},
                    original,
                )

    def test_yaml_loader_uses_the_same_strict_validation(self):
        valid = {
            "anthropic_responses_compat_enabled": False,
            "anthropic_responses_compat_mode": "lossless_required",
            "anthropic_responses_wire_profile": "public_responses",
            "anthropic_responses_model_profiles": {
                "fixture-model": "public_responses"
            },
            "anthropic_responses_replay_path": "fixture.sqlite3",
            "anthropic_responses_replay_ttl_seconds": 60,
            "anthropic_responses_replay_max_bytes": 4096,
            "anthropic_responses_replay_max_tenant_bytes": 2048,
            "anthropic_responses_replay_max_record_bytes": 1024,
            "anthropic_responses_replay_encryption_key_env": "FIXTURE_KEY",
            "anthropic_responses_replay_require_trusted_tenant": False,
            "anthropic_responses_replay_trusted_single_user": True,
        }
        apply_anthropic_responses_config(valid)
        for field, expected in valid.items():
            self.assertEqual(getattr(self.state, field), expected)

        invalid = (
            {"anthropic_responses_compat_enabled": "false"},
            {"anthropic_responses_compat_mode": "strict"},
            {"anthropic_responses_wire_profile": "future"},
            {"anthropic_responses_model_profiles": {"model": "future"}},
            {"anthropic_responses_replay_path": None},
            {"anthropic_responses_replay_ttl_seconds": 0},
            {"anthropic_responses_replay_max_bytes": True},
            {"anthropic_responses_replay_max_tenant_bytes": 0},
            {"anthropic_responses_replay_max_record_bytes": True},
            {"anthropic_responses_replay_encryption_key_env": None},
            {"anthropic_responses_replay_require_trusted_tenant": 1},
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    apply_anthropic_responses_config(value)


class WebSearchRuntimeConfigTest(unittest.TestCase):
    def setUp(self):
        self.state = ghc_api.state.state
        self.saved = {
            field: getattr(self.state, field)
            for field in WEB_SEARCH_CONFIG_FIELDS
        }
        self.state.enable_web_search_proxy = False
        self.state.web_search_proxy_endpoint = "http://127.0.0.1:5002"
        self.app = create_app()

    def tearDown(self):
        for field, value in self.saved.items():
            setattr(self.state, field, value)

    def test_get_and_post_expose_web_search_settings(self):
        with self.app.test_client() as client:
            get_response = client.get("/api/runtime-config")
            post_response = client.post("/api/runtime-config", json={
                "enable_web_search_proxy": True,
                "web_search_proxy_endpoint": "http://search.internal:5002",
            })

        self.assertEqual(get_response.status_code, 200)
        self.assertEqual(
            {field: get_response.get_json()[field] for field in WEB_SEARCH_CONFIG_FIELDS},
            {
                "enable_web_search_proxy": False,
                "web_search_proxy_endpoint": "http://127.0.0.1:5002",
            },
        )
        self.assertEqual(post_response.status_code, 200)
        body = post_response.get_json()["config"]
        self.assertTrue(body["enable_web_search_proxy"])
        self.assertEqual(body["web_search_proxy_endpoint"], "http://search.internal:5002")
        self.assertTrue(self.state.enable_web_search_proxy)
        self.assertEqual(self.state.web_search_proxy_endpoint, "http://search.internal:5002")

    def test_post_rejects_invalid_web_search_types(self):
        with self.app.test_client() as client:
            invalid = (
                ("enable_web_search_proxy", 1),
                ("web_search_proxy_endpoint", None),
            )
            for field, value in invalid:
                with self.subTest(field=field):
                    response = client.post("/api/runtime-config", json={field: value})
                    self.assertEqual(response.status_code, 400)
                    self.assertIn(field, response.get_json()["error"])


class AnthropicResponsesGeneratedConfigTest(unittest.TestCase):
    def test_generated_yaml_documents_all_runtime_fields_and_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("ghc_api.generate_config.get_config_dir", return_value=directory), mock.patch(
                "ghc_api.generate_config.platform.system", return_value="Linux"
            ):
                generate_config_file()

            config_path = Path(directory) / "config.yaml"
            text = config_path.read_text(encoding="utf-8")
            config = yaml.safe_load(text)

        expected = {
            "anthropic_responses_compat_enabled": True,
            "anthropic_responses_compat_mode": "compatibility",
            "anthropic_responses_wire_profile": "copilot_responses_lite",
            "anthropic_responses_model_profiles": {
                "gpt-5.6-sol": "copilot_responses_lite",
            },
            "anthropic_responses_replay_path": "",
            "anthropic_responses_replay_ttl_seconds": 86400,
            "anthropic_responses_replay_max_bytes": 1073741824,
            "anthropic_responses_replay_max_tenant_bytes": 268435456,
            "anthropic_responses_replay_max_record_bytes": 67108864,
            "anthropic_responses_replay_encryption_key_env": "GHC_REPLAY_ENCRYPTION_KEY",
            "anthropic_responses_replay_require_trusted_tenant": True,
            "anthropic_responses_replay_trusted_single_user": False,
        }
        self.assertEqual({field: config[field] for field in CONFIG_FIELDS}, expected)
        self.assertEqual(
            {field: config[field] for field in WEB_SEARCH_CONFIG_FIELDS},
            {
                "enable_web_search_proxy": False,
                "web_search_proxy_endpoint": "http://127.0.0.1:5002",
            },
        )
        self.assertIn("every approximation", text)
        self.assertIn("urlsafe-base64 Fernet key", text)
        self.assertIn("/search?keyword=<query>&limit=3", text)
        self.assertIn("preprocessed before calling Copilot", text)


if __name__ == "__main__":
    unittest.main()
