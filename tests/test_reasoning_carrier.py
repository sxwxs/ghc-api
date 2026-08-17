import unittest

from ghc_api.anthropic_responses import (
    AnthropicResponsesConversionError,
    convert_anthropic_to_responses,
    convert_responses_to_anthropic,
)
from ghc_api.reasoning_carrier import (
    MAX_REASONING_CARRIER_CHARS,
    REASONING_CARRIER_PREFIX,
    build_reasoning_carrier,
    is_reasoning_carrier,
    parse_reasoning_carrier,
    redact_reasoning_carriers_for_cache,
    strip_reasoning_carriers_from_messages_payload,
)


class ReasoningCarrierTests(unittest.TestCase):
    def test_round_trip_is_strict_and_preserves_opaque_content(self):
        signature = build_reasoning_carrier(
            model="gpt-5.6-sol",
            wire_profile="copilot_responses_lite",
            encrypted_content="opaque:\u0000-json-safe",
        )
        self.assertTrue(is_reasoning_carrier(signature))
        self.assertNotIn("opaque", signature)
        carrier = parse_reasoning_carrier(signature)
        self.assertEqual(carrier.model, "gpt-5.6-sol")
        self.assertEqual(carrier.wire_profile, "copilot_responses_lite")
        self.assertIsNone(carrier.item_id)
        self.assertEqual(carrier.encrypted_content, "opaque:\u0000-json-safe")

    def test_summary_only_carrier_round_trips_without_encrypted_content(self):
        signature = build_reasoning_carrier(
            model="gpt",
            wire_profile="public_responses",
            encrypted_content=None,
        )
        self.assertIsNone(parse_reasoning_carrier(signature).encrypted_content)

    def test_foreign_signature_returns_none(self):
        self.assertIsNone(parse_reasoning_carrier("real-anthropic-signature"))

    def test_namespaced_malformed_carriers_fail_closed(self):
        invalid = (
            REASONING_CARRIER_PREFIX,
            REASONING_CARRIER_PREFIX + "***",
            REASONING_CARRIER_PREFIX + "A",
            REASONING_CARRIER_PREFIX + "e30",  # valid JSON, unsupported shape
            REASONING_CARRIER_PREFIX + "A" * MAX_REASONING_CARRIER_CHARS,
        )
        for signature in invalid:
            with self.subTest(signature=signature[:80]):
                with self.assertRaises(ValueError):
                    parse_reasoning_carrier(signature)

    def test_cache_redaction_removes_opaque_carrier_payload(self):
        signature = build_reasoning_carrier(
            model="gpt",
            wire_profile="public_responses",
            encrypted_content="sensitive-opaque-state",
        )
        redacted = redact_reasoning_carriers_for_cache({
            "content": [{"type": "thinking", "signature": signature}]
        })
        cached = redacted["content"][0]["signature"]
        self.assertIn("Responses reasoning carrier", cached)
        self.assertIn("sha256=", cached)
        self.assertNotIn(signature, cached)

    def test_native_anthropic_strip_removes_only_our_assistant_blocks(self):
        synthetic = build_reasoning_carrier(
            model="gpt",
            wire_profile="public_responses",
            encrypted_content="enc",
        )
        payload = {
            "model": "claude",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "synthetic", "signature": synthetic},
                        {"type": "thinking", "thinking": "real", "signature": "real-signature"},
                        {"type": "text", "text": "answer"},
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "thinking", "thinking": "odd", "signature": synthetic}],
                },
            ],
        }
        stripped, count = strip_reasoning_carriers_from_messages_payload(payload)
        self.assertEqual(count, 1)
        self.assertEqual(
            [block["thinking"] for block in stripped["messages"][0]["content"] if block["type"] == "thinking"],
            ["real"],
        )
        self.assertIsNot(stripped, payload)
        self.assertIs(stripped["messages"][1], payload["messages"][1])

    def test_native_strip_drops_an_encrypted_only_empty_assistant_turn(self):
        signature = build_reasoning_carrier(
            model="gpt",
            wire_profile="public_responses",
            encrypted_content="enc",
        )
        payload = {
            "messages": [
                {"role": "user", "content": "before"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "", "signature": signature}
                ]},
                {"role": "user", "content": "after"},
            ]
        }
        stripped, count = strip_reasoning_carriers_from_messages_payload(payload)
        self.assertEqual(count, 1)
        self.assertEqual(
            [message["role"] for message in stripped["messages"]],
            ["user", "user"],
        )


class ReasoningCarrierTranslationTests(unittest.TestCase):
    def test_multiple_reasoning_items_round_trip_independently(self):
        translated = convert_responses_to_anthropic(
            {
                "id": "resp",
                "model": "resolved-model",
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "first"}],
                        "encrypted_content": "enc-1",
                    },
                    {
                        "type": "reasoning",
                        "summary": [],
                        "encrypted_content": "enc-2",
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "phase": "final_answer",
                        "content": [{"type": "output_text", "text": "answer"}],
                    },
                ],
            },
            original_model="client-model",
            reasoning_model="target-model",
            wire_profile="copilot_responses_lite",
        )
        self.assertEqual(
            [block["type"] for block in translated.response["content"]],
            ["thinking", "thinking", "text"],
        )
        self.assertEqual(translated.response["content"][1]["thinking"], "")

        request = convert_anthropic_to_responses(
            {
                "model": "target-model",
                "messages": [
                    {"role": "assistant", "content": translated.response["content"]},
                    {"role": "user", "content": "continue"},
                ],
            },
            wire_profile="copilot_responses_lite",
        )
        reasoning = [item for item in request.payload["input"] if item["type"] == "reasoning"]
        self.assertEqual([item.get("encrypted_content") for item in reasoning], ["enc-1", "enc-2"])
        self.assertEqual(reasoning[0]["summary"], [{"type": "summary_text", "text": "first"}])
        self.assertEqual(reasoning[1]["summary"], [])

    def test_public_profile_restores_reasoning_id_but_lite_omits_it(self):
        translated = convert_responses_to_anthropic(
            {
                "id": "resp",
                "model": "gpt",
                "status": "completed",
                "output": [{
                    "id": "rs_public_1",
                    "type": "reasoning",
                    "summary": [],
                    "encrypted_content": "enc",
                }],
            },
            original_model="client",
            reasoning_model="gpt",
            wire_profile="public_responses",
        )
        thinking = translated.response["content"][0]
        carrier = parse_reasoning_carrier(thinking["signature"])
        self.assertEqual(carrier.item_id, "rs_public_1")

        public_request = convert_anthropic_to_responses(
            {"model": "gpt", "messages": [{"role": "assistant", "content": [thinking]}]},
            wire_profile="public_responses",
        )
        lite_request = convert_anthropic_to_responses(
            {"model": "gpt", "messages": [{"role": "assistant", "content": [thinking]}]},
            wire_profile="copilot_responses_lite",
        )
        self.assertEqual(public_request.payload["input"][0]["id"], "rs_public_1")
        self.assertNotIn("id", lite_request.payload["input"][0])

        # Preserve continuity for carriers emitted by the first Grok release,
        # which used public_responses before Copilot's rotating SSE ids were
        # observed and assigned a dedicated profile.
        migrated_request = convert_anthropic_to_responses(
            {"model": "gpt", "messages": [{"role": "assistant", "content": [thinking]}]},
            wire_profile="copilot_public_responses",
        )
        self.assertEqual(migrated_request.payload["input"][0]["id"], "rs_public_1")
        self.assertEqual(
            migrated_request.payload["input"][0]["encrypted_content"], "enc"
        )

    def test_request_restoration_preserves_block_order(self):
        first = build_reasoning_carrier(
            model="gpt",
            wire_profile="copilot_responses_lite",
            encrypted_content="enc-1",
        )
        second = build_reasoning_carrier(
            model="gpt",
            wire_profile="copilot_responses_lite",
            encrypted_content="enc-2",
        )
        converted = convert_anthropic_to_responses(
            {
                "model": "gpt",
                "messages": [{
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "r1", "signature": first},
                        {"type": "tool_use", "id": "call_1", "name": "Read", "input": {}},
                        {"type": "thinking", "thinking": "r2", "signature": second},
                        {"type": "text", "text": "done"},
                    ],
                }],
            },
            wire_profile="copilot_responses_lite",
        )
        self.assertEqual(
            [item["type"] for item in converted.payload["input"]],
            ["reasoning", "function_call", "reasoning", "message"],
        )

    def test_nonstream_rejects_reasoning_after_visible_output(self):
        with self.assertRaises(AnthropicResponsesConversionError) as raised:
            convert_responses_to_anthropic(
                {
                    "id": "resp",
                    "model": "gpt",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "answer"}],
                        },
                        {
                            "id": "rs_late",
                            "type": "reasoning",
                            "summary": [],
                            "encrypted_content": "enc",
                        },
                    ],
                },
                original_model="client",
                reasoning_model="gpt",
                wire_profile="copilot_responses_lite",
            )
        self.assertIn("reasoning after visible output", str(raised.exception))

    def test_model_or_profile_mismatch_keeps_summary_and_drops_opaque_state(self):
        signature = build_reasoning_carrier(
            model="old-model",
            wire_profile="public_responses",
            encrypted_content="old-encrypted",
        )
        converted = convert_anthropic_to_responses(
            {
                "model": "new-model",
                "messages": [{
                    "role": "assistant",
                    "content": [{
                        "type": "thinking",
                        "thinking": "visible summary",
                        "signature": signature,
                    }],
                }],
            },
            wire_profile="copilot_responses_lite",
        )
        item = converted.payload["input"][0]
        self.assertEqual(item["type"], "reasoning")
        self.assertEqual(item["summary"], [{"type": "summary_text", "text": "visible summary"}])
        self.assertNotIn("encrypted_content", item)
        self.assertIn(
            "conversion.approximation",
            {warning["code"] for warning in converted.report.warnings},
        )


if __name__ == "__main__":
    unittest.main()
