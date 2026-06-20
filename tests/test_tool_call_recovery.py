import json
import unittest

from ghc_api.tool_call_recovery import LeakedToolCallTransformer


def _delta_event(index, text):
    return {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    }


def _run(events):
    """Feed a list of (event_type, event_dict) through the transformer.

    Returns the flat list of emitted (event_type, parsed_data_dict_or_str).
    """
    transformer = LeakedToolCallTransformer(enabled=True)
    emitted = []
    for event_type, event in events:
        raw = json.dumps(event)
        for et, data in transformer.process(event_type, event, raw):
            emitted.append((et, json.loads(data)))
    for et, data in transformer.finalize():
        emitted.append((et, json.loads(data)))
    return transformer, emitted


def _text_of(emitted):
    return "".join(
        e["delta"]["text"]
        for et, e in emitted
        if et == "content_block_delta" and e.get("delta", {}).get("type") == "text_delta"
    )


def _tool_uses(emitted):
    return [
        e["content_block"]
        for et, e in emitted
        if et == "content_block_start" and e.get("content_block", {}).get("type") == "tool_use"
    ]


def _input_jsons(emitted):
    return [
        json.loads(e["delta"]["partial_json"])
        for et, e in emitted
        if et == "content_block_delta" and e.get("delta", {}).get("type") == "input_json_delta"
    ]


# A representative leak: prose followed by a leaked invoke construct.
LEAK = (
    "Let me check the repository status.\n\ncall\n"
    '<invoke name="Bash">\n'
    '<parameter name="command">cd /repo\ngit status --short</parameter>\n'
    '<parameter name="description">Poll</parameter>\n'
    "</invoke>"
)


def _leak_events(chunks):
    """Build SSE events for a single text block carrying the given text chunks."""
    events = [
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
    ]
    for chunk in chunks:
        events.append(("content_block_delta", _delta_event(0, chunk)))
    events.append(("content_block_stop", {"type": "content_block_stop", "index": 0}))
    events.append(("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}))
    events.append(("message_stop", {"type": "message_stop"}))
    return events


class NormalPassthroughTest(unittest.TestCase):
    def test_plain_text_passes_through_unchanged(self):
        events = [
            ("message_start", {"type": "message_start", "message": {"model": "m"}}),
            ("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", _delta_event(0, "Hello, ")),
            ("content_block_delta", _delta_event(0, "world!")),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}),
            ("message_stop", {"type": "message_stop"}),
        ]
        transformer, emitted = _run(events)
        self.assertEqual(_text_of(emitted), "Hello, world!")
        self.assertEqual(_tool_uses(emitted), [])
        self.assertFalse(transformer.recovered_any)
        # stop_reason must be left as end_turn
        stop_reasons = [e["delta"]["stop_reason"] for et, e in emitted if et == "message_delta"]
        self.assertEqual(stop_reasons, ["end_turn"])

    def test_structured_tool_use_passes_through(self):
        events = [
            ("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "tool_use", "id": "toolu_x",
                                                       "name": "Bash", "input": {}}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "input_json_delta", "partial_json": "{\"a\":1}"}}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"}}),
        ]
        transformer, emitted = _run(events)
        self.assertFalse(transformer.recovered_any)
        self.assertEqual(len(emitted), 4)
        self.assertEqual(_input_jsons(emitted), [{"a": 1}])

    def test_prose_mentioning_invoke_without_name_is_text(self):
        leaked = "You can use the <invoke> tag for tool calls."
        events = [
            ("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", _delta_event(0, leaked)),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}),
        ]
        transformer, emitted = _run(events)
        self.assertFalse(transformer.recovered_any)
        self.assertEqual(_text_of(emitted), leaked)

    def test_backtick_mention_is_not_recovered(self):
        leaked = 'Example: `<invoke name="Bash">` is the markup.'
        events = [
            ("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", _delta_event(0, leaked)),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}),
        ]
        transformer, emitted = _run(events)
        self.assertFalse(transformer.recovered_any)
        self.assertEqual(_text_of(emitted), leaked)


class LeakRecoveryTest(unittest.TestCase):
    LEAK = LEAK

    def _leak_events(self, chunks):
        return _leak_events(chunks)

    def test_single_chunk_leak_is_recovered(self):
        transformer, emitted = _run(self._leak_events([self.LEAK]))
        self.assertTrue(transformer.recovered_any)
        # prose retained, residue + xml swallowed
        self.assertEqual(_text_of(emitted), "Let me check the repository status.")
        tools = _tool_uses(emitted)
        self.assertEqual([t["name"] for t in tools], ["Bash"])
        self.assertEqual(_input_jsons(emitted), [
            {"command": "cd /repo\ngit status --short", "description": "Poll"},
        ])
        # stop_reason rewritten
        stop_reasons = [e["delta"]["stop_reason"] for et, e in emitted if et == "message_delta"]
        self.assertEqual(stop_reasons, ["tool_use"])

    def test_leak_split_across_many_deltas(self):
        # Split the leak into single-character deltas to stress the holdback logic.
        chunks = list(self.LEAK)
        transformer, emitted = _run(self._leak_events(chunks))
        self.assertTrue(transformer.recovered_any)
        self.assertEqual(_text_of(emitted), "Let me check the repository status.")
        self.assertEqual(_input_jsons(emitted), [
            {"command": "cd /repo\ngit status --short", "description": "Poll"},
        ])

    def test_leak_with_function_calls_wrapper(self):
        leak = (
            "Working on it.\n<function_calls>\n"
            '<invoke name="Read">\n'
            '<parameter name="path">/etc/hosts</parameter>\n'
            "</invoke>\n</function_calls>"
        )
        transformer, emitted = _run(self._leak_events([leak]))
        self.assertTrue(transformer.recovered_any)
        self.assertEqual(_text_of(emitted), "Working on it.")
        self.assertEqual([t["name"] for t in _tool_uses(emitted)], ["Read"])
        self.assertEqual(_input_jsons(emitted), [{"path": "/etc/hosts"}])

    def test_multiple_leaked_tool_calls(self):
        leak = (
            "Doing two things.\ncall\n"
            '<invoke name="Bash"><parameter name="command">ls</parameter></invoke>\n'
            '<invoke name="Bash"><parameter name="command">pwd</parameter></invoke>'
        )
        transformer, emitted = _run(self._leak_events([leak]))
        self.assertTrue(transformer.recovered_any)
        self.assertEqual(_text_of(emitted), "Doing two things.")
        self.assertEqual(_input_jsons(emitted), [{"command": "ls"}, {"command": "pwd"}])

    def test_tool_use_blocks_get_unique_indices(self):
        leak = (
            "x\ncall\n"
            '<invoke name="A"><parameter name="p">1</parameter></invoke>\n'
            '<invoke name="B"><parameter name="p">2</parameter></invoke>'
        )
        transformer, emitted = _run(self._leak_events([leak]))
        start_indices = [e["index"] for et, e in emitted if et == "content_block_start"]
        # original text block (0) plus two recovered tool blocks with distinct indices
        self.assertEqual(len(start_indices), len(set(start_indices)))

    def test_incomplete_invoke_at_stream_end_is_recovered(self):
        leak = (
            "Run it.\ncall\n"
            '<invoke name="Bash">\n'
            '<parameter name="command">echo hi</parameter>'
            # no </invoke> before the block ends
        )
        transformer, emitted = _run(self._leak_events([leak]))
        self.assertTrue(transformer.recovered_any)
        self.assertEqual(_input_jsons(emitted), [{"command": "echo hi"}])


def _run_disabled(events):
    """Feed events through a recovery-disabled transformer (pure passthrough)."""
    transformer = LeakedToolCallTransformer(enabled=False)
    emitted = []
    for event_type, event in events:
        raw = json.dumps(event)
        for et, data in transformer.process(event_type, event, raw):
            emitted.append((et, json.loads(data)))
    for et, data in transformer.finalize():
        emitted.append((et, json.loads(data)))
    return transformer, emitted


class DisabledPassthroughTest(unittest.TestCase):
    def test_leak_is_not_recovered_when_disabled(self):
        # The same leak that LeakRecoveryTest recovers must pass through untouched.
        transformer, emitted = _run_disabled(_leak_events([LEAK]))
        self.assertFalse(transformer.recovered_any)
        # No tool_use blocks are injected; the leaked text is emitted verbatim.
        self.assertEqual(_tool_uses(emitted), [])
        self.assertEqual(_text_of(emitted), LEAK)
        # stop_reason is left untouched (not rewritten to tool_use).
        stop_reasons = [e["delta"]["stop_reason"] for et, e in emitted if et == "message_delta"]
        self.assertEqual(stop_reasons, ["end_turn"])

    def test_events_pass_through_unchanged_when_disabled(self):
        events = _leak_events([LEAK])
        transformer, emitted = _run_disabled(events)
        # Every input event is forwarded, in order, byte-for-byte.
        self.assertEqual(emitted, [(et, ev) for et, ev in events])

    def test_disabled_cache_content_captures_text(self):
        transformer, _ = _run_disabled(_leak_events([LEAK]))
        content = transformer.build_response_content()
        self.assertEqual(content, [{"type": "text", "text": LEAK}])


if __name__ == "__main__":
    unittest.main()
