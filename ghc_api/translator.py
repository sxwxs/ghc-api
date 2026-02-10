"""
API format translation between Anthropic and OpenAI formats
"""

import json
import uuid
from typing import Any, Dict, List, Optional


def translate_model_name(model: str) -> str:
    """Translate model names for Copilot API compatibility.

    Uses mappings loaded from config file. Supports:
    - Exact match: full model name matches exactly
    - Prefix match: model name starts with the prefix
    """
    from .config import model_mappings
    return model_mappings.translate(model)


def apply_system_prompt_filters(system_text: str) -> str:
    """Apply content filtering to system prompt text.

    - Removes strings specified in system_prompt_remove
    - Appends strings specified in system_prompt_add
    """
    from .state import state

    # Remove specified strings from system prompt
    for remove_str in state.system_prompt_remove:
        if remove_str in system_text:
            system_text = system_text.replace(remove_str, "")
            print(f"[Content Filter] Removed from system prompt: {remove_str[:50]}{'...' if len(remove_str) > 50 else ''}")

    # Add specified strings to system prompt (only if not already present)
    # for add_str in state.system_prompt_add:
    #     if add_str not in system_text:
    #         system_text = system_text + "\n\n" + add_str
    #         print(f"[Content Filter] Added to system prompt: {add_str[:50]}{'...' if len(add_str) > 50 else ''}")

    return system_text


def apply_tool_result_suffix_filter(content: str) -> str:
    """Remove trailing suffixes from tool result content.

    Only removes strings if they appear at the END of the content.
    """
    from .state import state

    for suffix in state.tool_result_suffix_remove:
        if content.endswith(suffix):
            content = content[:-len(suffix)]
            print(f"[Content Filter] Removed suffix from tool result: {suffix[:50]}{'...' if len(suffix) > 50 else ''}")

    return content


def translate_anthropic_to_openai(payload: Dict) -> Dict:
    """Translate Anthropic API format to OpenAI format"""
    messages = []

    # Handle system prompt
    system = payload.get("system")
    if system:
        if isinstance(system, str):
            system_text = apply_system_prompt_filters(system)
            messages.append({"role": "system", "content": system_text})
        elif isinstance(system, list):
            # Filter out billing headers which cause errors in GitHub API
            filtered_system = [
                block.get("text", "") for block in system
                if isinstance(block, dict) and
                block.get("type") == "text" and
                "x-anthropic-billing-header" not in block.get("text", "")
            ]
            if filtered_system:
                system_text = "\n\n".join(filtered_system)
                system_text = apply_system_prompt_filters(system_text)
                messages.append({"role": "system", "content": system_text})

    # Translate messages
    for msg in payload.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            if isinstance(content, list):
                # Handle tool results and other content blocks
                tool_results = [b for b in content if b.get("type") == "tool_result"]
                other_blocks = [b for b in content if b.get("type") != "tool_result"]

                # Tool results become tool messages
                for tr in tool_results:
                    tool_content = tr.get("content", "")
                    if isinstance(tool_content, str):
                        tool_content = apply_tool_result_suffix_filter(tool_content)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tr.get("tool_use_id"),
                        "content": tool_content,
                    })

                # Other content
                if other_blocks:
                    translated_content = translate_content_blocks(other_blocks)
                    if translated_content:
                        messages.append({"role": "user", "content": translated_content})
            else:
                messages.append({"role": "user", "content": content})

        elif role == "assistant":
            if isinstance(content, list):
                tool_uses = [b for b in content if b.get("type") == "tool_use"]
                text_blocks = [b for b in content if b.get("type") in ("text", "thinking")]

                text_content = "\n\n".join(
                    b.get("text", "") if b.get("type") == "text" else b.get("thinking", "")
                    for b in text_blocks
                )

                if tool_uses:
                    messages.append({
                        "role": "assistant",
                        "content": text_content or None,
                        "tool_calls": [
                            {
                                "id": tu.get("id"),
                                "type": "function",
                                "function": {
                                    "name": tu.get("name"),
                                    "arguments": json.dumps(tu.get("input", {})),
                                },
                            }
                            for tu in tool_uses
                        ],
                    })
                else:
                    messages.append({"role": "assistant", "content": text_content})
            else:
                messages.append({"role": "assistant", "content": content})

    # Build OpenAI payload
    openai_payload = {
        "model": translate_model_name(payload.get("model", "")),
        "messages": messages,
        "max_tokens": payload.get("max_tokens"),
        "stream": payload.get("stream", False),
    }

    if payload.get("temperature") is not None:
        openai_payload["temperature"] = payload["temperature"]
    if payload.get("top_p") is not None:
        openai_payload["top_p"] = payload["top_p"]
    if payload.get("stop_sequences"):
        openai_payload["stop"] = payload["stop_sequences"]

    # Translate tools
    if payload.get("tools"):
        openai_payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description"),
                    "parameters": tool.get("input_schema", {}),
                },
            }
            for tool in payload["tools"]
        ]

    # Translate tool_choice
    tool_choice = payload.get("tool_choice")
    if tool_choice:
        choice_type = tool_choice.get("type")
        if choice_type == "auto":
            openai_payload["tool_choice"] = "auto"
        elif choice_type == "any":
            openai_payload["tool_choice"] = "required"
        elif choice_type == "none":
            openai_payload["tool_choice"] = "none"
        elif choice_type == "tool" and tool_choice.get("name"):
            openai_payload["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice["name"]},
            }

    return openai_payload


def translate_content_blocks(blocks: List[Dict]) -> Any:
    """Translate Anthropic content blocks to OpenAI format"""
    has_image = any(b.get("type") == "image" for b in blocks)

    if not has_image:
        # Just combine text
        texts = []
        for b in blocks:
            if b.get("type") == "text":
                texts.append(b.get("text", ""))
            elif b.get("type") == "thinking":
                texts.append(b.get("thinking", ""))
        return "\n\n".join(texts) if texts else None

    # Handle mixed content with images
    parts = []
    for b in blocks:
        if b.get("type") == "text":
            parts.append({"type": "text", "text": b.get("text", "")})
        elif b.get("type") == "thinking":
            parts.append({"type": "text", "text": b.get("thinking", "")})
        elif b.get("type") == "image":
            source = b.get("source", {})
            parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{source.get('media_type')};base64,{source.get('data')}",
                },
            })
    return parts if parts else None


def map_openai_stop_reason_to_anthropic(reason: Optional[str]) -> Optional[str]:
    """Map OpenAI stop reason to Anthropic format"""
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "refusal",
    }
    return mapping.get(reason) if reason else None


def translate_openai_to_anthropic(response: Dict) -> Dict:
    """Translate OpenAI response to Anthropic format"""
    content = []
    stop_reason = None

    for choice in response.get("choices", []):
        message = choice.get("message", {})

        # Handle text content
        if message.get("content"):
            content.append({"type": "text", "text": message["content"]})

        # Handle tool calls
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                # Safely parse arguments - handle malformed JSON from incomplete streams
                args_str = tc["function"].get("arguments", "{}")
                try:
                    args_input = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    # If arguments are malformed (e.g., incomplete stream), use raw string
                    args_input = {"_raw_arguments": args_str}
                content.append({
                    "type": "tool_use",
                    "id": tc.get("id"),
                    "name": tc["function"]["name"],
                    "input": args_input,
                })

        if choice.get("finish_reason"):
            stop_reason = choice["finish_reason"]

    usage = response.get("usage", {})
    cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)

    return {
        "id": response.get("id", str(uuid.uuid4())),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": response.get("model", ""),
        "stop_reason": map_openai_stop_reason_to_anthropic(stop_reason),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0) - cached_tokens,
            "output_tokens": usage.get("completion_tokens", 0),
            **({"cache_read_input_tokens": cached_tokens} if cached_tokens else {}),
        },
    }
