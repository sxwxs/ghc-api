import unittest

import ghc_api.state
from ghc_api.routes.anthropic import apply_effort_policy, filter_payload_for_copilot


FAKE_MODELS = {"data": [
    {"id": "claude-opus-4.8", "capabilities": {"supports": {"reasoning_effort": ["low", "medium", "high", "xhigh", "max"]}}},
    {"id": "claude-opus-4.6", "capabilities": {"supports": {"reasoning_effort": ["low", "medium", "high", "max"]}}},  # max, no xhigh
    {"id": "gpt-5.4", "capabilities": {"supports": {"reasoning_effort": ["none", "low", "medium", "high", "xhigh"]}}},  # xhigh, no max
    {"id": "claude-opus-4.5", "capabilities": {"supports": {}}},  # no reasoning_effort
]}


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
    def setUp(self):
        self._saved_models = ghc_api.state.state.models
        ghc_api.state.state.models = FAKE_MODELS

    def tearDown(self):
        ghc_api.state.state.models = self._saved_models

    def test_opus_48_max_kept_unconverted(self):
        result = apply_effort_policy(_payload("claude-opus-4.8", "max"), "claude-opus-4.8")
        self.assertEqual(result["output_config"], {"effort": "max"})

    def test_opus_48_high_kept(self):
        result = apply_effort_policy(_payload("claude-opus-4.8", "high"), "claude-opus-4.8")
        self.assertEqual(result["output_config"], {"effort": "high"})

    def test_opus_46_max_kept(self):
        result = apply_effort_policy(_payload("claude-opus-4.6", "max"), "claude-opus-4.6")
        self.assertEqual(result["output_config"], {"effort": "max"})

    def test_gpt54_max_dropped(self):
        result = apply_effort_policy(_payload("gpt-5.4", "max"), "gpt-5.4")
        self.assertNotIn("output_config", result)

    def test_gpt54_xhigh_kept(self):
        result = apply_effort_policy(_payload("gpt-5.4", "xhigh"), "gpt-5.4")
        self.assertEqual(result["output_config"], {"effort": "xhigh"})

    def test_model_with_no_effort_support(self):
        result = apply_effort_policy(_payload("claude-opus-4.5", "medium"), "claude-opus-4.5")
        self.assertNotIn("output_config", result)

    def test_unknown_model_is_dropped(self):
        result = apply_effort_policy(_payload("claude-sonnet-4.5", "high"), "claude-sonnet-4.5")
        self.assertNotIn("output_config", result)

    def test_no_output_config_is_passthrough(self):
        payload = _payload("claude-opus-4.8")
        result = apply_effort_policy(payload, "claude-opus-4.8")
        self.assertIs(result, payload)

    def test_preserves_other_output_config_keys(self):
        payload = _payload("claude-opus-4.8", "high")
        payload["output_config"]["extra"] = "keep"
        result = apply_effort_policy(payload, "claude-opus-4.8")
        self.assertEqual(result["output_config"], {"effort": "high", "extra": "keep"})


class FilterPassesGatedOutputConfigTest(unittest.TestCase):
    def setUp(self):
        self._saved_models = ghc_api.state.state.models
        ghc_api.state.state.models = FAKE_MODELS

    def tearDown(self):
        ghc_api.state.state.models = self._saved_models

    def test_filter_keeps_output_config(self):
        # apply_effort_policy is the gate; filter forwards whatever it leaves.
        payload = _payload("claude-opus-4.8", "high")
        payload["unsupported_field"] = "drop me"
        filtered = filter_payload_for_copilot(payload)
        self.assertEqual(filtered["output_config"], {"effort": "high"})
        self.assertNotIn("unsupported_field", filtered)


if __name__ == "__main__":
    unittest.main()
