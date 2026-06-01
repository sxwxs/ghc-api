import unittest

from ghc_api.routes.anthropic import apply_effort_policy, filter_payload_for_copilot


def _payload(model, effort=None):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 1024,
        "thinking": {"type": "adaptive"},
    }
    if effort is not None:
        payload["output_config"] = {"effort": effort}
    return payload


class ApplyEffortPolicyTest(unittest.TestCase):
    def test_keeps_supported_effort(self):
        result = apply_effort_policy(_payload("claude-opus-4.7", "medium"), "claude-opus-4.7")
        self.assertEqual(result["output_config"], {"effort": "medium"})

    def test_drops_unsupported_effort_for_model(self):
        result = apply_effort_policy(_payload("claude-opus-4.7", "high"), "claude-opus-4.7")
        self.assertNotIn("output_config", result)

    def test_full_range_model(self):
        for value in ("low", "medium", "high", "xhigh"):
            result = apply_effort_policy(
                _payload("claude-opus-4.7-1m-internal", value),
                "claude-opus-4.7-1m-internal",
            )
            self.assertEqual(result["output_config"], {"effort": value})

    def test_normalizes_max_to_xhigh(self):
        result = apply_effort_policy(
            _payload("claude-opus-4.7-1m-internal", "max"),
            "claude-opus-4.7-1m-internal",
        )
        self.assertEqual(result["output_config"], {"effort": "xhigh"})

    def test_normalized_value_dropped_when_unsupported(self):
        # max -> xhigh, but opus-4.6-1m only supports low/medium/high
        result = apply_effort_policy(
            _payload("claude-opus-4.6-1m", "max"),
            "claude-opus-4.6-1m",
        )
        self.assertNotIn("output_config", result)

    def test_model_with_no_effort_support(self):
        result = apply_effort_policy(_payload("claude-opus-4.5", "medium"), "claude-opus-4.5")
        self.assertNotIn("output_config", result)

    def test_model_not_in_table_is_dropped(self):
        result = apply_effort_policy(_payload("claude-sonnet-4.5", "high"), "claude-sonnet-4.5")
        self.assertNotIn("output_config", result)

    def test_no_output_config_is_passthrough(self):
        payload = _payload("claude-opus-4.7")
        result = apply_effort_policy(payload, "claude-opus-4.7")
        self.assertIs(result, payload)

    def test_baked_in_high_variant_keeps_high_drops_others(self):
        kept = apply_effort_policy(_payload("claude-opus-4.7-high", "high"), "claude-opus-4.7-high")
        self.assertEqual(kept["output_config"], {"effort": "high"})
        dropped = apply_effort_policy(_payload("claude-opus-4.7-high", "low"), "claude-opus-4.7-high")
        self.assertNotIn("output_config", dropped)

    def test_preserves_other_output_config_keys(self):
        payload = _payload("claude-opus-4.7", "medium")
        payload["output_config"]["extra"] = "keep"
        result = apply_effort_policy(payload, "claude-opus-4.7")
        self.assertEqual(result["output_config"], {"effort": "medium", "extra": "keep"})


class FilterPassesGatedOutputConfigTest(unittest.TestCase):
    def test_filter_keeps_output_config(self):
        # apply_effort_policy is the gate; filter forwards whatever it leaves.
        payload = _payload("claude-opus-4.7", "medium")
        payload["unsupported_field"] = "drop me"
        filtered = filter_payload_for_copilot(payload)
        self.assertEqual(filtered["output_config"], {"effort": "medium"})
        self.assertNotIn("unsupported_field", filtered)


if __name__ == "__main__":
    unittest.main()
