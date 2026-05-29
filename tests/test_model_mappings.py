import unittest
from unittest import mock

from ghc_api.config import DEFAULT_ANTHROPIC_THINKING, DEFAULT_MODEL_MAPPINGS, ModelMappings
from ghc_api.routes.anthropic import filter_payload_for_copilot, normalize_thinking_for_copilot
from ghc_api.state import state


class DefaultModelMappingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mappings = ModelMappings()
        self.mappings.load_from_config({"model_mappings": DEFAULT_MODEL_MAPPINGS})

    def test_opus_alias_maps_to_claude_opus_4_8(self) -> None:
        self.assertEqual(self.mappings.translate("opus"), "claude-opus-4.8")

    def test_anthropic_dash_version_maps_to_copilot_opus_4_8(self) -> None:
        self.assertEqual(self.mappings.translate("claude-opus-4-8"), "claude-opus-4.8")

    def test_opus_4_8_prefixes_take_precedence_over_generic_opus_4_prefix(self) -> None:
        self.assertEqual(
            self.mappings.translate("claude-opus-4-8-20260529"),
            "claude-opus-4.8",
        )
        self.assertEqual(
            self.mappings.translate("claude-opus-4.8-20260529"),
            "claude-opus-4.8",
        )


class AnthropicOpus48PayloadTests(unittest.TestCase):
    def test_opus_4_8_enabled_thinking_is_normalized_to_adaptive(self) -> None:
        payload = {
            "model": "claude-opus-4.8",
            "thinking": {"type": "enabled", "budget_tokens": 31999},
            "max_tokens": 32000,
        }

        with mock.patch.object(state, "anthropic_thinking", DEFAULT_ANTHROPIC_THINKING):
            normalized = normalize_thinking_for_copilot(payload)

        self.assertEqual(normalized["thinking"], {"type": "adaptive"})
        self.assertEqual(normalized["output_config"], {"effort": "medium"})

    def test_opus_4_8_effort_is_mapped_by_config(self) -> None:
        payload = {
            "model": "claude-opus-4.8",
            "thinking": {"type": "enabled", "budget_tokens": 31999},
            "output_config": {"effort": "xhigh"},
            "max_tokens": 32000,
        }

        with mock.patch.object(state, "anthropic_thinking", DEFAULT_ANTHROPIC_THINKING):
            normalized = normalize_thinking_for_copilot(payload)

        self.assertEqual(normalized["thinking"], {"type": "adaptive"})
        self.assertEqual(normalized["output_config"], {"effort": "medium"})

    def test_opus_4_8_unmapped_future_effort_is_passed_through(self) -> None:
        payload = {
            "model": "claude-opus-4.8",
            "thinking": {"type": "enabled", "budget_tokens": 31999},
            "output_config": {"effort": "future"},
            "max_tokens": 32000,
        }

        with mock.patch.object(state, "anthropic_thinking", DEFAULT_ANTHROPIC_THINKING):
            normalized = normalize_thinking_for_copilot(payload)

        self.assertEqual(normalized["thinking"], {"type": "adaptive"})
        self.assertEqual(normalized["output_config"], {"effort": "future"})

    def test_output_config_is_only_forwarded_for_adaptive_thinking_models(self) -> None:
        opus_48_payload = {
            "model": "claude-opus-4.8",
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "medium"},
        }
        opus_46_payload = {
            "model": "claude-opus-4.6",
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "output_config": {"effort": "medium"},
        }

        with mock.patch.object(state, "anthropic_thinking", DEFAULT_ANTHROPIC_THINKING):
            self.assertIn("output_config", filter_payload_for_copilot(opus_48_payload))
            self.assertNotIn("output_config", filter_payload_for_copilot(opus_46_payload))


if __name__ == "__main__":
    unittest.main()
