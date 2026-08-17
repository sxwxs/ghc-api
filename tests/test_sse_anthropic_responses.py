import json
import unittest

from ghc_api.anthropic_responses import convert_anthropic_to_responses
from ghc_api.sse.anthropic_responses import (
    ResponsesAnthropicEventTranslator,
    StopSequenceScanner,
)
from ghc_api.reasoning_carrier import parse_reasoning_carrier


def event_types(events):
    return [event_type for event_type, _ in events]


class StopSequenceScannerTests(unittest.TestCase):
    def test_matches_across_chunks_without_leaking_prefix(self):
        scanner = StopSequenceScanner(["<STOP>"])
        output = scanner.push("before<ST") + scanner.push("OP>after") + scanner.finish()
        self.assertEqual(output, "before")
        self.assertEqual(scanner.matched, "<STOP>")


class ResponsesAnthropicEventTranslatorTests(unittest.TestCase):
    def translator(self, **kwargs):
        kwargs.setdefault("wire_profile", "copilot_responses_lite")
        kwargs.setdefault("reasoning_model", "gpt-5.6-sol")
        return ResponsesAnthropicEventTranslator(original_model="claude-opus-4.8", **kwargs)

    def test_reasoning_text_usage_and_lifecycle(self):
        translator = self.translator()
        sequence = [
            ("response.created", {"response": {"id": "resp_1", "model": "gpt"}}),
            ("response.output_item.added", {"output_index": 0, "item": {"type": "reasoning", "encrypted_content": "mid-state"}}),
            ("response.output_item.done", {"output_index": 0, "item": {"type": "reasoning", "summary": [], "encrypted_content": "x"}}),
            ("response.output_item.added", {"output_index": 1, "item": {"type": "message", "role": "assistant", "phase": "final_answer", "content": []}}),
            ("response.output_text.delta", {"output_index": 1, "content_index": 0, "delta": "hello"}),
            ("response.output_item.done", {"output_index": 1, "item": {"type": "message", "role": "assistant", "phase": "final_answer", "content": [{"type": "output_text", "text": "hello"}]}}),
            ("response.completed", {"response": {
                "id": "resp_1", "model": "gpt", "status": "completed",
                "output": [
                    {"type": "reasoning", "summary": [], "encrypted_content": "x"},
                    {"type": "message", "role": "assistant", "phase": "final_answer", "content": [{"type": "output_text", "text": "hello"}]},
                ],
                "usage": {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 2}, "output_tokens": 4},
            }}),
        ]
        output = []
        for name, event in sequence:
            output.extend(translator.process(name, event))
        self.assertEqual(event_types(output), [
            "message_start",
            "content_block_start", "content_block_delta", "content_block_stop",
            "content_block_start", "content_block_delta", "content_block_stop",
            "message_delta", "message_stop",
        ])
        signature_event = next(
            event for event_type, event in output
            if event_type == "content_block_delta"
            and event.get("delta", {}).get("type") == "signature_delta"
        )
        carrier = parse_reasoning_carrier(signature_event["delta"]["signature"])
        self.assertEqual(carrier.encrypted_content, "x")
        self.assertEqual(output[-2][1]["usage"]["input_tokens"], 8)

    def test_copilot_public_replays_terminal_reasoning_id_with_encrypted_content(self):
        translator = self.translator(
            wire_profile="copilot_public_responses",
            reasoning_model="grok-4.6",
        )
        translator.process("response.output_item.added", {
            "output_index": 0,
            "item": {"type": "reasoning", "id": "per-frame-added-id"},
        })
        output = translator.process("response.output_item.done", {
            "output_index": 0,
            "item": {
                "type": "reasoning",
                "id": "terminal-done-id",
                "summary": [{"type": "summary_text", "text": "summary"}],
                "encrypted_content": "encrypted-reasoning",
            },
        })
        signature_event = next(
            event for event_type, event in output
            if event_type == "content_block_delta"
            and event.get("delta", {}).get("type") == "signature_delta"
        )
        signature = signature_event["delta"]["signature"]
        carrier = parse_reasoning_carrier(signature)
        self.assertEqual(carrier.item_id, "terminal-done-id")
        self.assertEqual(carrier.encrypted_content, "encrypted-reasoning")

        next_turn = convert_anthropic_to_responses(
            {
                "model": "grok-4.6",
                "messages": [{
                    "role": "assistant",
                    "content": [{
                        "type": "thinking",
                        "thinking": "summary",
                        "signature": signature,
                    }],
                }],
            },
            wire_profile="copilot_public_responses",
        )
        reasoning = next_turn.payload["input"][0]
        self.assertEqual(reasoning["id"], "terminal-done-id")
        self.assertEqual(
            reasoning["encrypted_content"], "encrypted-reasoning"
        )

    def test_web_search_lifecycle_is_sidecar_only_and_final_text_streams(self):
        translator = self.translator()
        search_added = {
            "type": "web_search_call",
            "id": "opaque-search-added",
            "status": "in_progress",
        }
        search_done = {
            "type": "web_search_call",
            "id": "opaque-search-done",
            "status": "completed",
            "action": {
                "type": "search",
                "query": "private query",
                "queries": ["private query"],
            },
        }
        message = {
            "type": "message",
            "role": "assistant",
            "phase": "final_answer",
            "content": [{
                "type": "output_text",
                "text": "search answer",
                "annotations": [{
                    "type": "url_citation",
                    "start_index": 0,
                    "end_index": 6,
                    "title": "private title",
                    "url": "https://example.invalid/private",
                }],
            }],
        }
        sequence = [
            ("response.created", {"response": {"id": "resp_search", "model": "gpt"}}),
            ("response.output_item.added", {"output_index": 0, "item": search_added}),
            ("response.web_search_call.in_progress", {"output_index": 0, "item_id": "opaque-search-progress"}),
            ("response.web_search_call.searching", {"output_index": 0, "item_id": "opaque-search-searching"}),
            ("response.web_search_call.completed", {"output_index": 0, "item_id": "opaque-search-completed"}),
            ("response.output_item.done", {"output_index": 0, "item": search_done}),
            ("response.output_item.added", {"output_index": 1, "item": {**message, "id": "opaque-message-added", "content": []}}),
            ("response.output_text.delta", {"output_index": 1, "content_index": 0, "delta": "search answer"}),
            ("response.output_text.annotation.added", {
                "output_index": 1,
                "content_index": 0,
                "item_id": "opaque-annotation-event",
                "annotation_index": 0,
                "annotation": message["content"][0]["annotations"][0],
            }),
            ("response.output_item.done", {"output_index": 1, "item": {**message, "id": "opaque-message-done"}}),
            ("response.completed", {"response": {
                "id": "resp_search",
                "model": "gpt",
                "status": "completed",
                "output": [search_done, message],
                "usage": {},
                "tool_usage": {"web_search": {"num_requests": 1}},
            }}),
        ]
        output = []
        for name, event in sequence:
            output.extend(translator.process(name, event))
        self.assertEqual(event_types(output), [
            "message_start", "content_block_start", "content_block_delta",
            "content_block_stop", "message_delta", "message_stop",
        ])
        self.assertNotIn("private query", str(output))
        self.assertFalse(any(
            event.get("content_block", {}).get("type") in (
                "server_tool_use", "web_search_tool_result",
            )
            for _, event in output
        ))
        self.assertEqual(translator.terminal_result.response["content"], [
            {"type": "text", "text": "search answer"}
        ])
        self.assertEqual(translator.terminal_result.report.unaccounted_paths, [])

    def test_web_search_lifecycle_uses_output_index_identity_and_rejects_unclosed_item(self):
        translator = self.translator()
        translator.process("response.output_item.added", {
            "output_index": 0,
            "item": {"type": "web_search_call", "id": "opaque-added", "status": "in_progress"},
        })
        self.assertEqual(translator.process(
            "response.web_search_call.in_progress",
            {"output_index": 0, "item_id": "opaque-in-progress"},
        ), [])
        self.assertEqual(translator.process(
            "response.web_search_call.searching",
            {"output_index": 0, "item_id": "opaque-searching"},
        ), [])
        self.assertEqual(translator.process(
            "response.web_search_call.completed",
            {"output_index": 0, "item_id": "opaque-completed"},
        ), [])

        translator = self.translator()
        translator.process("response.output_item.added", {
            "output_index": 0,
            "item": {"type": "web_search_call", "id": "search_1", "status": "in_progress"},
        })
        output = translator.process("response.completed", {"response": {
            "id": "resp_search",
            "model": "gpt",
            "status": "completed",
            "output": [],
            "usage": {},
        }})
        self.assertIn("error", event_types(output))
        self.assertNotIn("message_stop", event_types(output))

    def test_public_profile_rejects_web_search_id_mismatch(self):
        translator = ResponsesAnthropicEventTranslator(
            original_model="claude-opus-4.8",
            reasoning_model="gpt-5.6-sol",
            wire_profile="public_responses",
        )
        translator.process("response.output_item.added", {
            "output_index": 0,
            "item": {"type": "web_search_call", "id": "search_1", "status": "in_progress"},
        })
        output = translator.process("response.web_search_call.searching", {
            "output_index": 0,
            "item_id": "search_2",
        })
        self.assertEqual(event_types(output), ["error"])
        self.assertEqual(
            translator.compatibility_warnings[-1]["code"],
            "responses.web_search_id_mismatch",
        )

    def test_web_search_and_annotation_lifecycle_order_is_validated(self):
        translator = self.translator()
        translator.process("response.output_item.added", {
            "output_index": 0,
            "item": {"type": "web_search_call", "id": "search_1", "status": "in_progress"},
        })
        self.assertEqual(translator.process(
            "response.web_search_call.completed",
            {"output_index": 0, "item_id": "search_1"},
        ), [])
        output = translator.process(
            "response.web_search_call.searching",
            {"output_index": 0, "item_id": "search_1"},
        )
        self.assertEqual(event_types(output), ["error"])
        self.assertEqual(
            translator.compatibility_warnings[-1]["code"],
            "responses.web_search_status_regression",
        )

        translator = self.translator()
        translator.process("response.output_item.added", {
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "message_1",
                "role": "assistant",
                "content": [],
            },
        })
        output = translator.process("response.output_text.annotation.added", {
            "output_index": 1,
            "content_index": 0,
            "item_id": "wrong",
            "annotation_index": 0,
            "annotation": {"type": "url_citation"},
        })
        self.assertEqual(event_types(output), ["error"])
        self.assertEqual(
            translator.compatibility_warnings[-1]["code"],
            "responses.annotation_without_message",
        )

        translator = self.translator()
        translator.process("response.output_item.added", {
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "message_1",
                "role": "assistant",
                "content": [],
            },
        })
        self.assertEqual(translator.process(
            "response.output_text.annotation.added",
            {
                "output_index": 0,
                "content_index": 1,
                "item_id": "message_1",
                "annotation_index": 0,
                "annotation": {"type": "url_citation"},
            },
        ), [])

    def test_function_call_waits_for_done_name_and_hydrates_arguments(self):
        translator = self.translator()
        output = []
        output += translator.process("response.created", {"response": {"id": "resp"}})
        output += translator.process("response.output_item.added", {
            "output_index": 0, "item": {"type": "function_call", "call_id": "call_1"},
        })
        output += translator.process("response.function_call_arguments.delta", {"output_index": 0, "delta": '{"x"'})
        self.assertEqual(event_types(output), ["message_start"])
        output += translator.process("response.output_item.done", {
            "output_index": 0,
            "item": {"type": "function_call", "call_id": "call_1", "name": "Read", "arguments": '{"x":1}'},
        })
        self.assertEqual(event_types(output)[-3:], ["content_block_start", "content_block_delta", "content_block_stop"])
        self.assertEqual(output[-2][1]["delta"]["partial_json"], '{"x":1}')

    def test_interleaved_parallel_calls_each_start_and_stop_once(self):
        translator = self.translator()
        output = translator.process("response.created", {"response": {"id": "resp"}})
        output += translator.process("response.output_item.added", {
            "output_index": 0, "item": {"type": "function_call", "call_id": "a", "name": "A"},
        })
        output += translator.process("response.output_item.added", {
            "output_index": 1, "item": {"type": "function_call", "call_id": "b", "name": "B"},
        })
        output += translator.process("response.function_call_arguments.delta", {"output_index": 1, "delta": '{"b":2}'})
        output += translator.process("response.function_call_arguments.delta", {"output_index": 0, "delta": '{"a":1}'})
        output += translator.process("response.output_item.done", {
            "output_index": 0, "item": {"type": "function_call", "call_id": "a", "name": "A", "arguments": '{"a":1}'},
        })
        output += translator.process("response.output_item.done", {
            "output_index": 1, "item": {"type": "function_call", "call_id": "b", "name": "B", "arguments": '{"b":2}'},
        })
        starts = [event["index"] for name, event in output if name == "content_block_start"]
        stops = [event["index"] for name, event in output if name == "content_block_stop"]
        self.assertEqual(starts, [0, 1])
        self.assertEqual(stops, [0, 1])

    def test_function_arguments_fail_closed_unless_strict_json_object(self):
        for arguments in ("[]", "1", "null", "not-json", '{"x":1,"x":2}'):
            translator = self.translator()
            output = translator.process(
                "response.created", {"response": {"id": "resp"}}
            )
            output += translator.process("response.output_item.done", {
                "output_index": 0,
                "item": {
                    "type": "function_call", "call_id": "call_1",
                    "name": "Read", "arguments": arguments,
                },
            })
            with self.subTest(arguments=arguments):
                self.assertIn("error", event_types(output))
                self.assertNotIn("content_block_start", event_types(output))
                self.assertNotIn("message_stop", event_types(output))
                self.assertEqual(
                    translator.compatibility_warnings[-1]["code"],
                    "responses.invalid_function_arguments",
                )

    def test_custom_tool_wrapper_matches_nonstream_projection(self):
        raw_input = '{"looks":"like json"}'
        translator = self.translator()
        output = translator.process(
            "response.created", {"response": {"id": "resp"}}
        )
        output += translator.process("response.output_item.added", {
            "output_index": 0,
            "item": {
                "type": "custom_tool_call", "call_id": "call_custom",
                "name": "shell", "input": "",
            },
        })
        output += translator.process("response.custom_tool_call_input.delta", {
            "output_index": 0, "delta": raw_input,
        })
        output += translator.process("response.custom_tool_call_input.done", {
            "output_index": 0, "input": raw_input,
        })
        output += translator.process("response.output_item.done", {
            "output_index": 0,
            "item": {
                "type": "custom_tool_call", "call_id": "call_custom",
                "name": "shell", "input": raw_input,
            },
        })
        partial = "".join(
            event["delta"]["partial_json"]
            for name, event in output if name == "content_block_delta"
        )
        self.assertEqual(json.loads(partial), {"input": raw_input})
        self.assertEqual(
            [name for name, _ in output][-3:],
            ["content_block_start", "content_block_delta", "content_block_stop"],
        )

    def test_refusal_done_prefix_hydrates_without_duplicate_text(self):
        translator = self.translator()
        output = translator.process(
            "response.created", {"response": {"id": "resp"}}
        )
        output += translator.process("response.output_item.added", {
            "output_index": 0,
            "item": {"type": "message", "role": "assistant", "content": []},
        })
        output += translator.process("response.refusal.delta", {
            "output_index": 0, "delta": "can",
        })
        output += translator.process("response.refusal.done", {
            "output_index": 0, "refusal": "cannot",
        })
        terminal_item = {
            "type": "message", "role": "assistant",
            "content": [{"type": "refusal", "refusal": "cannot"}],
        }
        output += translator.process("response.output_item.done", {
            "output_index": 0, "item": terminal_item,
        })
        output += translator.process("response.completed", {"response": {
            "id": "resp", "model": "gpt", "status": "completed",
            "output": [terminal_item], "usage": {},
        }})
        visible = "".join(
            event["delta"]["text"]
            for name, event in output if name == "content_block_delta"
        )
        self.assertEqual(visible, "cannot")
        self.assertEqual(event_types(output)[-2:], ["message_delta", "message_stop"])

    def test_terminal_text_and_argument_mismatch_are_fatal(self):
        cases = (
            (
                {
                    "type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "streamed"}],
                },
                {
                    "type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "different"}],
                },
                "responses.text_mismatch",
            ),
            (
                {
                    "type": "function_call", "call_id": "call_1",
                    "name": "Read", "arguments": '{"x":1}',
                },
                {
                    "type": "function_call", "call_id": "call_1",
                    "name": "Read", "arguments": '{"x":2}',
                },
                "responses.arguments_mismatch",
            ),
        )
        for streamed_item, terminal_item, expected_code in cases:
            translator = self.translator()
            output = translator.process(
                "response.created", {"response": {"id": "resp"}}
            )
            output += translator.process("response.output_item.done", {
                "output_index": 0, "item": streamed_item,
            })
            output += translator.process("response.completed", {"response": {
                "id": "resp", "model": "gpt", "status": "completed",
                "output": [terminal_item], "usage": {},
            }})
            with self.subTest(code=expected_code):
                self.assertIn("error", event_types(output))
                self.assertNotIn("message_stop", event_types(output))
                self.assertEqual(
                    translator.compatibility_warnings[-1]["code"],
                    expected_code,
                )

    def test_tool_identity_mutation_is_fatal(self):
        for changed_field, changed_value, expected_code in (
            ("name", "Write", "responses.tool_name_mutation"),
            ("call_id", "call_2", "responses.call_id_mutation"),
        ):
            translator = self.translator()
            output = translator.process(
                "response.created", {"response": {"id": "resp"}}
            )
            output += translator.process("response.output_item.added", {
                "output_index": 0,
                "item": {
                    "type": "function_call", "call_id": "call_1",
                    "name": "Read", "arguments": "",
                },
            })
            done_item = {
                "type": "function_call", "call_id": "call_1",
                "name": "Read", "arguments": "{}",
            }
            done_item[changed_field] = changed_value
            output += translator.process("response.output_item.done", {
                "output_index": 0, "item": done_item,
            })
            with self.subTest(field=changed_field):
                self.assertEqual(event_types(output)[-1], "error")
                self.assertNotIn("message_stop", event_types(output))
                self.assertEqual(
                    translator.compatibility_warnings[-1]["code"],
                    expected_code,
                )

    def test_reasoning_after_committed_text_is_fatal(self):
        translator = self.translator()
        output = translator.process(
            "response.created", {"response": {"id": "resp"}}
        )
        output += translator.process(
            "response.output_item.added",
            {"output_index": 0, "item": {"type": "message", "role": "assistant", "content": []}},
        )
        output += translator.process(
            "response.output_text.delta",
            {"output_index": 0, "delta": "hello"},
        )
        output += translator.process(
            "response.output_item.done",
            {"output_index": 0, "item": {
                "type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            }},
        )
        output += translator.process(
            "response.output_item.added",
            {"output_index": 1, "item": {"type": "reasoning", "summary": []}},
        )
        output += translator.process(
            "response.output_item.done",
            {"output_index": 1, "item": {
                "type": "reasoning", "summary": [], "encrypted_content": "enc",
            }},
        )
        self.assertEqual(event_types(output)[-1], "error")
        self.assertEqual(
            translator.compatibility_warnings[-1]["code"],
            "responses.late_reasoning_item",
        )

    def test_multiple_content_parts_are_validated_independently(self):
        translator = self.translator()
        output = translator.process(
            "response.created", {"response": {"id": "resp", "model": "gpt"}}
        )
        output += translator.process("response.output_item.added", {
            "output_index": 0,
            "item": {"type": "message", "role": "assistant", "content": []},
        })
        for content_index, text in enumerate(("Hello", " World")):
            output += translator.process("response.content_part.added", {
                "output_index": 0,
                "content_index": content_index,
                "part": {"type": "output_text", "text": ""},
            })
            output += translator.process("response.output_text.delta", {
                "output_index": 0,
                "content_index": content_index,
                "delta": text,
            })
            output += translator.process("response.output_text.done", {
                "output_index": 0,
                "content_index": content_index,
                "text": text,
            })
            output += translator.process("response.content_part.done", {
                "output_index": 0,
                "content_index": content_index,
                "part": {"type": "output_text", "text": text},
            })
        output += translator.process("response.output_item.done", {
            "output_index": 0,
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Hello"},
                    {"type": "output_text", "text": " World"},
                ],
            },
        })
        output += translator.process("response.completed", {"response": {
            "id": "resp", "model": "gpt", "status": "completed",
            "output": [{
                "type": "message", "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Hello"},
                    {"type": "output_text", "text": " World"},
                ],
            }],
            "usage": {},
        }})
        visible = "".join(
            event["delta"]["text"]
            for name, event in output
            if name == "content_block_delta"
            and event.get("delta", {}).get("type") == "text_delta"
        )
        self.assertEqual(visible, "Hello World")
        self.assertNotIn("error", event_types(output))
        self.assertEqual(event_types(output)[-1], "message_stop")

    def test_terminal_response_identity_and_status_must_match_created_event(self):
        # The id comparison is only meaningful where ids are stable.
        public = ResponsesAnthropicEventTranslator(
            original_model="claude-opus-4.8",
            reasoning_model="gpt-5.6-sol",
            wire_profile="public_responses",
        )
        output = public.process(
            "response.created", {"response": {"id": "resp", "model": "gpt"}}
        )
        output += public.process("response.completed", {"response": {
            "id": "resp_other", "model": "gpt", "status": "completed",
            "output": [], "usage": {},
        }})
        self.assertEqual(event_types(output)[-1], "error")
        self.assertEqual(
            public.compatibility_warnings[-1]["code"],
            "responses.terminal_response_id_mismatch",
        )
        self.assertTrue(public.protocol_failed)

        translator = self.translator()
        output = translator.process(
            "response.created", {"response": {"id": "resp", "model": "gpt"}}
        )
        output += translator.process("response.completed", {"response": {
            "id": "resp", "model": "gpt", "status": "failed",
            "error": {"message": "failed"}, "output": [], "usage": {},
        }})
        self.assertEqual(event_types(output)[-1], "error")
        self.assertEqual(
            translator.compatibility_warnings[-1]["code"],
            "responses.terminal_status_mismatch",
        )
        self.assertTrue(translator.protocol_failed)

    def test_copilot_profiles_accept_per_event_opaque_response_ids(self):
        """Copilot returns a different encrypted response.id on every event of
        one stream, including for Grok's standard Responses request dialect.
        Only call_id is stable on these profiles."""
        for profile in (
            "copilot_responses_lite",
            "copilot_public_responses",
        ):
            with self.subTest(profile=profile):
                translator = ResponsesAnthropicEventTranslator(
                    original_model="claude-opus-4.8",
                    reasoning_model=(
                        "grok-4.6"
                        if profile == "copilot_public_responses"
                        else "gpt-5.6-sol"
                    ),
                    wire_profile=profile,
                )
                output = translator.process(
                    "response.created",
                    {"response": {"id": "opaque-A", "model": "model"}},
                )
                output += translator.process(
                    "response.in_progress",
                    {"response": {"id": "opaque-B", "model": "model"}},
                )
                output += translator.process("response.output_item.added", {
                    "output_index": 0,
                    "item": {
                        "type": "message", "id": "opaque-item-A",
                        "role": "assistant", "content": [],
                    },
                })
                output += translator.process("response.output_text.delta", {
                    "output_index": 0, "content_index": 0,
                    "item_id": "opaque-item-B", "delta": "hello",
                })
                output += translator.process("response.completed", {"response": {
                    "id": "opaque-C", "model": "model", "status": "completed",
                    "output": [{
                        "type": "message", "id": "opaque-item-C",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hello"}],
                    }],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }})
                self.assertFalse(translator.protocol_failed)
                self.assertNotIn("error", event_types(output))
                self.assertEqual(event_types(output)[-1], "message_stop")
        visible = "".join(
            event["delta"]["text"]
            for name, event in output
            if name == "content_block_delta"
            and event.get("delta", {}).get("type") == "text_delta"
        )
        self.assertEqual(visible, "hello")
        # The client-visible message id stays the one announced in
        # message_start; a rotated upstream id must not change it.
        message_start = next(
            event for name, event in output if name == "message_start"
        )
        self.assertTrue(message_start["message"]["id"].startswith("msg_"))

    def test_event_after_closed_output_index_is_fatal(self):
        translator = self.translator()
        output = translator.process(
            "response.created", {"response": {"id": "resp"}}
        )
        output += translator.process("response.output_item.done", {
            "output_index": 0,
            "item": {
                "type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            },
        })
        output += translator.process("response.output_text.delta", {
            "output_index": 0, "delta": "late",
        })
        self.assertEqual(event_types(output)[-1], "error")
        self.assertNotIn("message_stop", event_types(output))
        self.assertEqual(
            translator.compatibility_warnings[-1]["code"],
            "responses.event_after_closed_output_item",
        )

    def test_unknown_incomplete_reason_is_fatal_without_output_state(self):
        translator = self.translator()
        output = translator.process("response.incomplete", {"response": {
            "id": "resp", "model": "gpt", "status": "incomplete",
            "incomplete_details": {"reason": "future-private-reason"},
            "output": [], "usage": {},
        }})
        self.assertIn("error", event_types(output))
        self.assertNotIn("message_stop", event_types(output))

    def test_incomplete_event_type_is_preserved_when_response_status_is_absent(self):
        translator = self.translator()
        output = translator.process("response.incomplete", {"response": {
            "id": "resp", "model": "gpt",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [], "usage": {},
        }})
        message_delta = next(
            event for name, event in output if name == "message_delta"
        )
        self.assertEqual(message_delta["delta"]["stop_reason"], "max_tokens")
        self.assertEqual(event_types(output)[-1], "message_stop")

    def test_stop_sequence_across_text_deltas(self):
        translator = self.translator(stop_sequences=["<STOP>"])
        output = translator.process("response.created", {"response": {"id": "resp"}})
        output += translator.process("response.output_item.added", {
            "output_index": 0, "item": {"type": "message", "role": "assistant", "content": []},
        })
        output += translator.process("response.output_text.delta", {"output_index": 0, "delta": "before<ST"})
        output += translator.process("response.output_text.delta", {"output_index": 0, "delta": "OP>after"})
        visible = "".join(event["delta"]["text"] for name, event in output if name == "content_block_delta")
        self.assertEqual(visible, "before")
        self.assertEqual(translator.local_stop_sequence, "<STOP>")

    def test_stop_sequence_suppresses_later_tool(self):
        translator = self.translator(stop_sequences=["<STOP>"])
        output = translator.process(
            "response.created", {"response": {"id": "resp"}}
        )
        output += translator.process("response.output_item.added", {
            "output_index": 0,
            "item": {"type": "message", "role": "assistant", "content": []},
        })
        output += translator.process("response.output_text.delta", {
            "output_index": 0, "delta": "before<STOP>after",
        })
        output += translator.process("response.output_item.done", {
            "output_index": 0,
            "item": {
                "type": "message", "role": "assistant", "phase": "final_answer",
                "content": [{"type": "output_text", "text": "before<STOP>after"}],
            },
        })
        output += translator.process("response.output_item.added", {
            "output_index": 1,
            "item": {
                "type": "function_call", "call_id": "hidden-call",
                "name": "HiddenTool", "arguments": "{}",
            },
        })
        output += translator.process("response.output_item.done", {
            "output_index": 1,
            "item": {
                "type": "function_call", "call_id": "hidden-call",
                "name": "HiddenTool", "arguments": "{}",
            },
        })
        output += translator.process("response.completed", {"response": {
            "id": "resp", "model": "gpt", "status": "completed",
            "output": [
                {
                    "type": "message", "role": "assistant",
                    "phase": "final_answer", "content": [{
                        "type": "output_text", "text": "before<STOP>after",
                    }],
                },
                {
                    "type": "function_call", "call_id": "hidden-call",
                    "name": "HiddenTool", "arguments": "{}",
                },
            ],
            "usage": {},
        }})

        self.assertNotIn("HiddenTool", str(output))
        self.assertEqual(
            [event["content_block"]["type"] for name, event in output
             if name == "content_block_start"],
            ["text"],
        )
        self.assertEqual(event_types(output)[-2:], ["message_delta", "message_stop"])

    def test_terminal_output_repairs_missing_delta(self):
        translator = self.translator()
        output = translator.process("response.created", {"response": {"id": "resp"}})
        output += translator.process("response.completed", {"response": {
            "id": "resp", "model": "gpt", "status": "completed",
            "output": [{"type": "message", "role": "assistant", "phase": "final_answer", "content": [{"type": "output_text", "text": "terminal"}]}],
            "usage": {},
        }})
        text = "".join(event["delta"]["text"] for name, event in output if name == "content_block_delta")
        self.assertEqual(text, "terminal")
        self.assertEqual(event_types(output)[-2:], ["message_delta", "message_stop"])

    def test_unknown_event_is_skipped_and_terminal_hydration_recovers_output(self):
        """A skipped event costs incremental delivery, not model output: the
        terminal event carries the full output array."""
        translator = self.translator()
        output = translator.process(
            "response.future_content.delta", {"type": "response.future_content.delta"}
        )
        self.assertEqual(event_types(output), [])
        self.assertFalse(translator.protocol_failed)
        self.assertEqual(
            translator.compatibility_warnings[0]["code"],
            "responses.unknown_event_skipped",
        )
        self.assertEqual(translator.compatibility_warnings[0]["action"], "approximation")

        output = translator.process("response.completed", {"response": {
            "id": "resp", "model": "gpt", "status": "completed",
            "output": [{"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "recovered"},
            ]}],
            "usage": {},
        }})
        text = "".join(
            event["delta"]["text"]
            for name, event in output
            if name == "content_block_delta"
        )
        self.assertEqual(text, "recovered")
        self.assertEqual(event_types(output)[-2:], ["message_delta", "message_stop"])
        self.assertFalse(translator.protocol_failed)

    def test_unknown_event_without_a_terminal_event_still_fails_closed(self):
        translator = self.translator()
        translator.process(
            "response.future_content.delta", {"type": "response.future_content.delta"}
        )
        output = translator.finalize_interrupted()
        self.assertEqual(event_types(output), ["error"])
        self.assertTrue(translator.protocol_failed)
        self.assertIn(
            "responses.stream_ended_without_terminal",
            {warning["code"] for warning in translator.compatibility_warnings},
        )

    def test_failed_response_emits_anthropic_error(self):
        translator = self.translator()
        output = translator.process("response.failed", {
            "response": {"error": {"message": "failed"}},
        })
        self.assertEqual(event_types(output), ["error"])
        self.assertEqual(output[0][1]["type"], "error")


if __name__ == "__main__":
    unittest.main()
