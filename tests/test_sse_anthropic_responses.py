import json
import unittest

from ghc_api.anthropic_responses import MODE_LOSSLESS_REQUIRED
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

    def test_web_search_before_reasoning_is_sidecar_only_and_final_text_streams(self):
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
        reasoning = {
            "type": "reasoning",
            "summary": [],
            "encrypted_content": "opaque-reasoning",
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
            ("response.output_item.added", {"output_index": 1, "item": reasoning}),
            ("response.output_item.done", {"output_index": 1, "item": reasoning}),
            ("response.output_item.added", {"output_index": 2, "item": {**message, "id": "opaque-message-added", "content": []}}),
            ("response.output_text.delta", {"output_index": 2, "content_index": 0, "delta": "search answer"}),
            ("response.output_text.annotation.added", {
                "output_index": 2,
                "content_index": 0,
                "item_id": "opaque-annotation-event",
                "annotation_index": 0,
                "annotation": message["content"][0]["annotations"][0],
            }),
            ("response.output_item.done", {"output_index": 2, "item": {**message, "id": "opaque-message-done"}}),
            ("response.completed", {"response": {
                "id": "resp_search",
                "model": "gpt",
                "status": "completed",
                "output": [search_done, reasoning, message],
                "usage": {},
                "tool_usage": {"web_search": {"num_requests": 1}},
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
        self.assertNotIn("private query", str(output))
        self.assertFalse(any(
            event.get("content_block", {}).get("type") in (
                "server_tool_use", "web_search_tool_result",
            )
            for _, event in output
        ))
        self.assertEqual(
            [block["type"] for block in translator.terminal_result.response["content"]],
            ["thinking", "text"],
        )
        self.assertEqual(translator.terminal_result.response["content"][1], {
            "type": "text", "text": "search answer"
        })
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
        # Compatibility mode keeps the already-delivered turn and records the
        # dropped item; only a lossless contract refuses the exchange.
        self.assertNotIn("error", event_types(output))
        self.assertEqual(event_types(output)[-1], "message_stop")
        self.assertIn(
            "responses.terminal_output_missing_item",
            {warning["code"] for warning in translator.compatibility_warnings},
        )

        strict = self.translator(mode=MODE_LOSSLESS_REQUIRED)
        strict.process("response.output_item.added", {
            "output_index": 0,
            "item": {"type": "web_search_call", "id": "search_1", "status": "in_progress"},
        })
        output = strict.process("response.completed", {"response": {
            "id": "resp_search",
            "model": "gpt",
            "status": "completed",
            "output": [],
            "usage": {},
        }})
        self.assertIn("error", event_types(output))
        self.assertNotIn("message_stop", event_types(output))

    def test_lite_profile_tolerates_rotating_response_ids(self):
        # The live Copilot backend re-encrypts the response id in every frame,
        # so identity across frames carries no protocol meaning there.
        translator = self.translator()
        output = translator.process(
            "response.created", {"response": {"id": "enc_a", "model": "gpt"}}
        )
        output += translator.process("response.in_progress", {
            "response": {"id": "enc_b", "model": "gpt"},
        })
        output += translator.process("response.output_item.added", {
            "output_index": 0,
            "item": {"type": "message", "role": "assistant", "content": []},
        })
        output += translator.process("response.output_text.delta", {
            "output_index": 0, "content_index": 0, "delta": "hi",
        })
        output += translator.process("response.completed", {"response": {
            "id": "enc_c", "model": "gpt", "status": "completed",
            "output": [{
                "type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": "hi"}],
            }],
            "usage": {},
        }})
        self.assertFalse(translator.protocol_failed)
        self.assertNotIn("error", event_types(output))
        self.assertEqual(event_types(output)[-1], "message_stop")

    def test_lite_terminal_only_stream_uses_terminal_response_id(self):
        message_ids = []
        for response_id in ("enc_terminal_a", "enc_terminal_b"):
            translator = self.translator()
            output = translator.process("response.completed", {"response": {
                "id": response_id,
                "model": "gpt",
                "status": "completed",
                "output": [],
                "usage": {},
            }})
            message_ids.append(next(
                event["message"]["id"]
                for name, event in output
                if name == "message_start"
            ))
        self.assertNotEqual(message_ids[0], message_ids[1])

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

    def test_public_profile_rejects_foreign_content_item_ids(self):
        cases = (
            (
                {"type": "message", "id": "item_1", "role": "assistant", "content": []},
                "response.output_text.delta",
                {"content_index": 0, "delta": "secret"},
                "text",
            ),
            (
                {
                    "type": "function_call", "id": "item_1",
                    "call_id": "call_1", "name": "Read", "arguments": "",
                },
                "response.function_call_arguments.delta",
                {"delta": '{"path":"secret"}'},
                "arguments",
            ),
        )
        for item, event_type, event, state_field in cases:
            translator = self.translator(wire_profile="public_responses")
            translator.process("response.output_item.added", {
                "output_index": 0, "item": item,
            })
            output = translator.process(event_type, {
                "output_index": 0, "item_id": "foreign_item", **event,
            })
            with self.subTest(event_type=event_type):
                self.assertEqual(event_types(output), ["error"])
                self.assertEqual(getattr(translator.states[0], state_field), "")
                self.assertEqual(
                    translator.compatibility_warnings[-1]["code"],
                    "responses.content_item_id_mismatch",
                )

    def test_non_output_message_parts_are_not_streamed_as_assistant_text(self):
        for part_type in ("input_text", "summary_text", "encrypted_content"):
            translator = self.translator()
            output = translator.process("response.output_item.done", {
                "output_index": 0,
                "item": {
                    "type": "message", "role": "assistant",
                    "content": [{
                        "type": part_type,
                        "text": "PRIVATE PROMPT",
                        "encrypted_content": "PRIVATE ENCRYPTED",
                    }],
                },
            })
            with self.subTest(part_type=part_type):
                self.assertNotIn("PRIVATE", json.dumps(output))
                self.assertEqual(translator.states[0].text, "")

    def test_duplicate_or_regressing_sequence_number_is_fatal_before_replay(self):
        translator = self.translator()
        translator.process("response.output_item.added", {
            "sequence_number": 1,
            "output_index": 0,
            "item": {"type": "message", "role": "assistant", "content": []},
        })
        first = translator.process("response.output_text.delta", {
            "sequence_number": 2, "output_index": 0, "delta": "once",
        })
        replay = translator.process("response.output_text.delta", {
            "sequence_number": 2, "output_index": 0, "delta": "once",
        })
        self.assertIn("content_block_delta", event_types(first))
        self.assertEqual(event_types(replay), ["error"])
        self.assertEqual(translator.states[0].text, "once")

        translator = self.translator()
        translator.process("response.output_item.added", {
            "sequence_number": 5,
            "output_index": 0,
            "item": {
                "type": "function_call", "call_id": "call_1",
                "name": "Read", "arguments": "",
            },
        })
        translator.process("response.function_call_arguments.delta", {
            "sequence_number": 6, "output_index": 0, "delta": "{}",
        })
        replay = translator.process("response.function_call_arguments.delta", {
            "sequence_number": 4, "output_index": 0, "delta": "{}",
        })
        self.assertEqual(event_types(replay), ["error"])
        self.assertEqual(translator.states[0].arguments, "{}")

    def test_done_tool_items_require_nonempty_call_id_and_name(self):
        for item_type in ("function_call", "custom_tool_call"):
            argument_key = "arguments" if item_type == "function_call" else "input"
            for field, value in (("call_id", None), ("call_id", ""), ("name", None), ("name", "")):
                item = {
                    "type": item_type, "call_id": "call_1", "name": "Read",
                    argument_key: "{}",
                }
                if value is None:
                    item.pop(field)
                else:
                    item[field] = value
                translator = self.translator()
                output = translator.process("response.output_item.done", {
                    "output_index": 0, "item": item,
                })
                with self.subTest(item_type=item_type, field=field, value=value):
                    self.assertEqual(event_types(output), ["error"])
                    self.assertNotIn("content_block_start", event_types(output))

    def test_output_before_created_uses_request_scoped_unique_message_id(self):
        message_ids = []
        for _ in range(2):
            translator = self.translator()
            output = translator.process("response.output_text.delta", {
                "output_index": 0, "delta": "hello",
            })
            message_ids.append(next(
                event["message"]["id"]
                for name, event in output if name == "message_start"
            ))
        self.assertNotEqual(message_ids[0], message_ids[1])

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
        cases = (
            ({
                "id": "resp_other", "model": "gpt", "status": "completed",
                "output": [], "usage": {},
            }, "responses.terminal_response_id_mismatch", "public_responses"),
            ({
                "id": "resp", "model": "gpt", "status": "failed",
                "error": {"message": "failed"}, "output": [], "usage": {},
            }, "responses.terminal_status_mismatch", "copilot_responses_lite"),
        )
        for terminal, expected_code, profile in cases:
            translator = self.translator(wire_profile=profile)
            output = translator.process(
                "response.created", {"response": {"id": "resp", "model": "gpt"}}
            )
            output += translator.process(
                "response.completed", {"response": terminal}
            )
            with self.subTest(code=expected_code):
                self.assertEqual(event_types(output)[-1], "error")
                self.assertEqual(
                    translator.compatibility_warnings[-1]["code"], expected_code
                )
                self.assertTrue(translator.protocol_failed)

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

    def test_unknown_incomplete_reason_is_projected_as_truncation(self):
        translator = self.translator()
        output = translator.process("response.incomplete", {"response": {
            "id": "resp", "model": "gpt", "status": "incomplete",
            "incomplete_details": {"reason": "future-private-reason"},
            "output": [], "usage": {},
        }})
        self.assertNotIn("error", event_types(output))
        self.assertEqual(event_types(output)[-1], "message_stop")
        delta = next(
            event for name, event in output if name == "message_delta"
        )
        self.assertEqual(delta["delta"]["stop_reason"], "max_tokens")

        strict = self.translator(mode=MODE_LOSSLESS_REQUIRED)
        output = strict.process("response.incomplete", {"response": {
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

    def test_unknown_event_is_skipped_in_compatibility_and_fatal_when_lossless(self):
        translator = self.translator()
        output = translator.process(
            "response.future_content.delta",
            {"type": "response.future_content.delta"},
        )
        # A lifecycle event we have never seen must not destroy a live turn.
        self.assertEqual(output, [])
        self.assertFalse(translator.protocol_failed)
        warning = translator.compatibility_warnings[0]
        self.assertEqual(warning["code"], "responses.unknown_event")
        self.assertNotIn("future_content", json.dumps(translator.compatibility_warnings))

        strict = self.translator(mode=MODE_LOSSLESS_REQUIRED)
        output = strict.process(
            "response.future_content.delta",
            {"type": "response.future_content.delta"},
        )
        self.assertEqual(event_types(output), ["error"])
        self.assertTrue(strict.protocol_failed)

    def test_unknown_output_item_is_skipped_without_losing_sibling_content(self):
        translator = self.translator()
        output = translator.process(
            "response.created", {"response": {"id": "resp", "model": "gpt"}}
        )
        output += translator.process("response.output_item.added", {
            "output_index": 0,
            "item": {"type": "image_generation_call", "id": "img_1", "status": "in_progress"},
        })
        output += translator.process("response.image_generation_call.partial_image", {
            "output_index": 0, "partial_image_index": 0,
        })
        output += translator.process("response.output_item.done", {
            "output_index": 0,
            "item": {"type": "image_generation_call", "id": "img_1", "status": "completed"},
        })
        output += translator.process("response.output_item.added", {
            "output_index": 1,
            "item": {"type": "message", "role": "assistant", "content": []},
        })
        output += translator.process("response.output_text.delta", {
            "output_index": 1, "content_index": 0, "delta": "after the image",
        })
        output += translator.process("response.completed", {"response": {
            "id": "resp", "model": "gpt", "status": "completed",
            "output": [
                {"type": "image_generation_call", "id": "img_1", "status": "completed"},
                {
                    "type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "after the image"}],
                },
            ],
            "usage": {},
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
        self.assertEqual(visible, "after the image")
        self.assertIn(
            "responses.unknown_output_item",
            {warning["code"] for warning in translator.compatibility_warnings},
        )

    def test_event_after_closed_unknown_item_is_fatal(self):
        translator = self.translator()
        translator.process("response.output_item.added", {
            "output_index": 0,
            "item": {"type": "image_generation_call", "id": "img_1"},
        })
        translator.process("response.output_item.done", {
            "output_index": 0,
            "item": {"type": "image_generation_call", "id": "img_1"},
        })
        output = translator.process("response.output_text.delta", {
            "output_index": 0,
            "content_index": 0,
            "delta": "must not be silently discarded",
        })
        self.assertEqual(event_types(output), ["error"])
        self.assertTrue(translator.protocol_failed)
        self.assertIn(
            "responses.event_after_closed_output_item",
            {warning["code"] for warning in translator.compatibility_warnings},
        )

    def test_upstream_failure_keeps_its_http_class(self):
        translator = self.translator()
        output = translator.process("response.failed", {
            "response": {"error": {"code": "rate_limit_exceeded", "message": "slow down"}},
        })
        self.assertEqual(event_types(output), ["error"])
        self.assertEqual(output[0][1]["error"]["type"], "rate_limit_error")
        self.assertEqual(translator.error_status_code, 429)

    def test_failed_response_emits_anthropic_error(self):
        translator = self.translator()
        output = translator.process("response.failed", {
            "response": {"error": {"message": "failed"}},
        })
        self.assertEqual(event_types(output), ["error"])
        self.assertEqual(output[0][1]["type"], "error")


if __name__ == "__main__":
    unittest.main()
