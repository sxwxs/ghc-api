import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

import ghc_api.state
from ghc_api.app import create_app
from ghc_api.generate_config import generate_config_file
from ghc_api.main import apply_anthropic_responses_config


CONFIG_FIELDS = (
    "anthropic_responses_compat_enabled",
    "anthropic_responses_wire_profile",
    "anthropic_responses_model_profiles",
)


class AnthropicResponsesRuntimeConfigTest(unittest.TestCase):
    def setUp(self):
        self.state = ghc_api.state.state
        self.saved = {
            field: (
                dict(getattr(self.state, field))
                if isinstance(getattr(self.state, field), dict)
                else getattr(self.state, field)
            )
            for field in CONFIG_FIELDS
        }
        self.state.anthropic_responses_compat_enabled = True
        self.state.anthropic_responses_wire_profile = "copilot_responses_lite"
        self.state.anthropic_responses_model_profiles = {
            "gpt-5.6-sol": "copilot_responses_lite",
        }
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
            "anthropic_responses_wire_profile": "public_responses",
            "anthropic_responses_model_profiles": {
                "gpt-sanitized": "public_responses",
                "gpt-copilot-sanitized": "copilot_responses_lite",
            },
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
            ("anthropic_responses_wire_profile", "future_profile"),
            ("anthropic_responses_wire_profile", ""),
            ("anthropic_responses_model_profiles", []),
            ("anthropic_responses_model_profiles", {"model": "future_profile"}),
            ("anthropic_responses_model_profiles", {"": "public_responses"}),
        )
        with self.app.test_client() as client:
            for field, invalid in invalid_values:
                with self.subTest(field=field, invalid=invalid):
                    response = client.post("/api/runtime-config", json={field: invalid})
                    self.assertEqual(response.status_code, 400)
                    self.assertIn(field, response.get_json()["error"])

    def test_removed_settings_are_rejected_as_unknown(self):
        for removed in (
            "anthropic_responses_replay_path",
            "anthropic_responses_compat_mode",
        ):
            with self.subTest(removed=removed):
                with self.app.test_client() as client:
                    response = client.post(
                        "/api/runtime-config",
                        json={removed: "old-value"},
                    )
                self.assertEqual(response.status_code, 400)
                self.assertIn("Unknown config key", response.get_json()["error"])

    def test_yaml_loader_warns_once_for_removed_replay_settings(self):
        with mock.patch("builtins.print") as printed:
            apply_anthropic_responses_config({
                "anthropic_responses_replay_path": "old.sqlite3",
                "anthropic_responses_replay_max_bytes": 1024,
            })
        printed.assert_called_once()
        message = printed.call_args.args[0]
        self.assertIn("deprecated and ignored", message)
        self.assertIn("anthropic_responses_replay_path", message)

    def test_yaml_loader_warns_and_ignores_the_removed_compat_mode(self):
        with mock.patch("builtins.print") as printed:
            apply_anthropic_responses_config({
                "anthropic_responses_compat_mode": "lossless_required",
            })
        printed.assert_called_once()
        self.assertIn("removed and ignored", printed.call_args.args[0])
        self.assertFalse(hasattr(self.state, "anthropic_responses_compat_mode"))

    def test_yaml_loader_uses_the_same_strict_validation(self):
        valid = {
            "anthropic_responses_compat_enabled": False,
            "anthropic_responses_wire_profile": "public_responses",
            "anthropic_responses_model_profiles": {
                "fixture-model": "public_responses"
            },
        }
        apply_anthropic_responses_config(valid)
        for field, expected in valid.items():
            self.assertEqual(getattr(self.state, field), expected)

        invalid = (
            {"anthropic_responses_compat_enabled": "false"},
            {"anthropic_responses_wire_profile": "future"},
            {"anthropic_responses_model_profiles": {"model": "future"}},
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    apply_anthropic_responses_config(value)


class AnthropicResponsesGeneratedConfigTest(unittest.TestCase):
    def test_generated_yaml_documents_runtime_fields_and_stateless_carrier(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch(
                "ghc_api.generate_config.get_config_dir", return_value=directory
            ), mock.patch("ghc_api.generate_config.platform.system", return_value="Linux"):
                generate_config_file()
            text = (Path(directory) / "config.yaml").read_text(encoding="utf-8")
            config = yaml.safe_load(text)

        expected = {
            "anthropic_responses_compat_enabled": True,
            "anthropic_responses_wire_profile": "copilot_responses_lite",
            "anthropic_responses_model_profiles": {
                "gpt-5.6-sol": "copilot_responses_lite",
            },
        }
        self.assertEqual({field: config[field] for field in CONFIG_FIELDS}, expected)
        self.assertNotIn("anthropic_responses_compat_mode", config)
        self.assertIn("X-GHC-Compatibility-Warnings", text)
        self.assertIn("carried statelessly", text)
        self.assertNotIn("anthropic_responses_replay_", text)


if __name__ == "__main__":
    unittest.main()
