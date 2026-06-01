import unittest

from ghc_api.routes.anthropic import filter_payload_for_copilot


class AnthropicPayloadFilterTest(unittest.TestCase):
    def test_preserves_output_config_effort_for_opus_46_1m(self):
        payload = {
            "model": "claude-opus-4.6-1m",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1024,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
            "unsupported_field": "drop me",
        }

        filtered = filter_payload_for_copilot(payload)

        self.assertEqual(filtered["output_config"], {"effort": "high"})
        self.assertEqual(filtered["thinking"], {"type": "adaptive"})
        self.assertNotIn("unsupported_field", filtered)

    def test_preserves_output_config_effort_for_opus_47_1m_internal(self):
        payload = {
            "model": "claude-opus-4.7-1m-internal",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1024,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "xhigh"},
        }

        filtered = filter_payload_for_copilot(payload)

        self.assertEqual(filtered["output_config"], {"effort": "xhigh"})

    def test_filters_output_config_for_other_models(self):
        # Gating now lives in apply_effort_policy; the field-level filter forwards
        # output_config whenever it is still present in the payload.
        payload = {
            "model": "claude-opus-4.7",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1024,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "medium"},
        }

        filtered = filter_payload_for_copilot(payload)

        self.assertEqual(filtered["output_config"], {"effort": "medium"})


if __name__ == "__main__":
    unittest.main()
