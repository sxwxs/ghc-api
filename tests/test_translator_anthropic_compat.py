"""Tests for Anthropic-to-OpenAI translation of tool_result blocks and tool schemas."""

from ghc_api.routes.anthropic import _anthropic_messages_have_vision
from ghc_api.translator import (
    normalize_tool_parameters_schema,
    translate_anthropic_to_openai,
    translate_tool_result_content,
)


def test_tool_result_image_base64_translated_to_image_url():
    payload = {
        "model": "gemini-4-neon",
        "max_tokens": 128,
        "messages": [
            {"role": "user", "content": "read image"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "Read",
                        "input": {"file_path": "a.png"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "AAAA",
                                },
                            }
                        ],
                    }
                ],
            },
        ],
    }

    translated = translate_anthropic_to_openai(payload)
    tool_msgs = [m for m in translated["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "call_1"
    assert tool_msgs[0]["content"] == [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AAAA"},
        }
    ]
    assert _anthropic_messages_have_vision(payload["messages"]) is True


def test_tool_result_mixed_text_and_url_image():
    content = translate_tool_result_content(
        [
            {"type": "text", "text": "preview:"},
            {
                "type": "image",
                "source": {"type": "url", "url": "https://example.com/a.png"},
            },
        ]
    )
    assert content == [
        {"type": "text", "text": "preview:"},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
    ]


def test_tool_result_tool_reference_converted_to_text():
    content = translate_tool_result_content(
        [
            {"type": "tool_reference", "tool_name": "mcp__t3-code__t3_thread_read"},
        ]
    )
    assert content == "Loaded tool: mcp__t3-code__t3_thread_read"


def test_tool_result_text_blocks_joined_to_string():
    content = translate_tool_result_content(
        [
            {"type": "text", "text": "line 1"},
            {"type": "text", "text": "line 2"},
        ]
    )
    assert content == "line 1\n\nline 2"


def test_tool_parameters_schema_adds_empty_properties_for_object():
    raw = {"type": "object", "additionalProperties": False}
    normalized = normalize_tool_parameters_schema(raw)
    assert normalized == {"type": "object", "additionalProperties": False, "properties": {}}
    # Original dict should not be mutated
    assert "properties" not in raw


def test_translate_tools_normalizes_schema_and_skips_tool_search_builtin():
    payload = {
        "model": "gemini-4-neon",
        "max_tokens": 64,
        "tools": [
            {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"},
            {
                "name": "mcp__t3-code__t3_worktree_status",
                "description": "Status",
                "input_schema": {"type": "object", "additionalProperties": False},
            },
            {
                "name": "Read",
                "description": "Read file",
                "input_schema": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                },
            },
        ],
        "messages": [{"role": "user", "content": "hi"}],
    }

    translated = translate_anthropic_to_openai(payload)
    assert len(translated["tools"]) == 2
    assert translated["tools"][0]["function"]["name"] == "mcp__t3-code__t3_worktree_status"
    assert translated["tools"][0]["function"]["parameters"] == {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
    }
    assert translated["tools"][1]["function"]["name"] == "Read"
    assert translated["tools"][1]["function"]["parameters"]["properties"] == {
        "file_path": {"type": "string"}
    }


def test_vision_detection_handles_nested_tool_result_and_documents():
    msgs_no_vision = [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "1", "content": "plain text"}],
        }
    ]
    assert _anthropic_messages_have_vision(msgs_no_vision) is False

    msgs_doc = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "1",
                    "content": [{"type": "document", "source": {"type": "base64", "data": "AA"}}],
                }
            ],
        }
    ]
    assert _anthropic_messages_have_vision(msgs_doc, include_document=False) is False
    assert _anthropic_messages_have_vision(msgs_doc, include_document=True) is True
