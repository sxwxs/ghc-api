"""
Streaming response translation between OpenAI and Anthropic formats
"""

import json
import uuid
from typing import Any, Dict, List

from .translator import map_openai_stop_reason_to_anthropic


class AnthropicStreamState:
    """State for translating streaming responses to Anthropic format"""
    def __init__(self):
        self.message_start_sent = False
        self.content_block_index = 0
        self.content_block_open = False
        self.tool_calls: Dict[int, Dict] = {}


def translate_chunk_to_anthropic_events(chunk: Dict, stream_state: AnthropicStreamState) -> List[Dict]:
    """Translate OpenAI streaming chunk to Anthropic events"""
    events = []

    if not chunk.get("choices"):
        return events

    choice = chunk["choices"][0]
    delta = choice.get("delta", {})

    # Send message_start if not sent yet
    if not stream_state.message_start_sent:
        usage = chunk.get("usage", {})
        cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)

        events.append({
            "type": "message_start",
            "message": {
                "id": chunk.get("id", str(uuid.uuid4())),
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": chunk.get("model", ""),
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": usage.get("prompt_tokens", 0) - cached_tokens,
                    "output_tokens": 0,
                    **({"cache_read_input_tokens": cached_tokens} if cached_tokens else {}),
                },
            },
        })
        stream_state.message_start_sent = True

    # Handle text content
    if delta.get("content"):
        # Check if we need to close a tool block
        if stream_state.content_block_open and any(
            tc.get("anthropic_block_index") == stream_state.content_block_index
            for tc in stream_state.tool_calls.values()
        ):
            events.append({
                "type": "content_block_stop",
                "index": stream_state.content_block_index,
            })
            stream_state.content_block_index += 1
            stream_state.content_block_open = False

        if not stream_state.content_block_open:
            events.append({
                "type": "content_block_start",
                "index": stream_state.content_block_index,
                "content_block": {"type": "text", "text": ""},
            })
            stream_state.content_block_open = True

        events.append({
            "type": "content_block_delta",
            "index": stream_state.content_block_index,
            "delta": {"type": "text_delta", "text": delta["content"]},
        })

    # Handle tool calls
    if delta.get("tool_calls"):
        for tc in delta["tool_calls"]:
            tc_index = tc.get("index", 0)

            if tc.get("id") and tc.get("function", {}).get("name"):
                # New tool call starting
                if stream_state.content_block_open:
                    events.append({
                        "type": "content_block_stop",
                        "index": stream_state.content_block_index,
                    })
                    stream_state.content_block_index += 1
                    stream_state.content_block_open = False

                block_index = stream_state.content_block_index
                stream_state.tool_calls[tc_index] = {
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "anthropic_block_index": block_index,
                }

                events.append({
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": {},
                    },
                })
                stream_state.content_block_open = True

            if tc.get("function", {}).get("arguments"):
                tc_info = stream_state.tool_calls.get(tc_index)
                if tc_info:
                    events.append({
                        "type": "content_block_delta",
                        "index": tc_info["anthropic_block_index"],
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": tc["function"]["arguments"],
                        },
                    })

    # Handle finish
    if choice.get("finish_reason"):
        if stream_state.content_block_open:
            events.append({
                "type": "content_block_stop",
                "index": stream_state.content_block_index,
            })
            stream_state.content_block_open = False

        usage = chunk.get("usage", {})
        cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)

        events.append({
            "type": "message_delta",
            "delta": {
                "stop_reason": map_openai_stop_reason_to_anthropic(choice["finish_reason"]),
                "stop_sequence": None,
            },
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0) - cached_tokens,
                "output_tokens": usage.get("completion_tokens", 0),
                **({"cache_read_input_tokens": cached_tokens} if cached_tokens else {}),
            },
        })
        events.append({"type": "message_stop"})

    return events


def reconstruct_openai_response_from_chunks(chunks: List[Dict]) -> Dict:
    """Reconstruct a complete OpenAI response from streaming chunks"""
    if not chunks:
        return {}

    # Get metadata from first chunk
    first_chunk = chunks[0]
    response_id = first_chunk.get("id", "")
    model = first_chunk.get("model", "")
    created = first_chunk.get("created", 0)

    # Accumulate content and tool calls
    content_parts = []
    tool_calls: Dict[int, Dict] = {}
    finish_reason = None
    usage = {}

    for chunk in chunks:
        if chunk.get("usage"):
            usage = chunk["usage"]

        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})

            if delta.get("content"):
                content_parts.append(delta["content"])

            if delta.get("tool_calls"):
                for tc in delta["tool_calls"]:
                    tc_index = tc.get("index", 0)
                    if tc_index not in tool_calls:
                        tool_calls[tc_index] = {
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc.get("id"):
                        tool_calls[tc_index]["id"] = tc["id"]
                    if tc.get("function", {}).get("name"):
                        tool_calls[tc_index]["function"]["name"] = tc["function"]["name"]
                    if tc.get("function", {}).get("arguments"):
                        tool_calls[tc_index]["function"]["arguments"] += tc["function"]["arguments"]

            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    # Build the reconstructed response
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts) if content_parts else None,
    }

    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls.keys())]

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": usage,
    }
