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
    from .counters import counters
    result = model_mappings.translate(model)
    if result != model:
        counters.incr("mod.model_name_mapping")
    return result


def apply_system_prompt_filters(system_text: str) -> str:
    """Apply content filtering to system prompt text.

    - Removes strings specified in system_prompt_remove
    - Appends strings specified in system_prompt_add
    """
    from .state import state
    from .counters import counters

    # Remove specified strings from system prompt
    for remove_str in state.system_prompt_remove:
        if remove_str in system_text:
            system_text = system_text.replace(remove_str, "")
            counters.incr("mod.system_prompt_remove")
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
    from .counters import counters

    for suffix in state.tool_result_suffix_remove:
        if content.endswith(suffix):
            content = content[:-len(suffix)]
            counters.incr("mod.tool_result_suffix_remove")
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
                    tool_content = translate_tool_result_content(tr.get("content", ""))
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
        translated_tools = []
        for tool in payload["tools"]:
            if not isinstance(tool, dict):
                continue
            # Skip Anthropic server/built-in tool_search declarations (no input_schema)
            tool_type = tool.get("type")
            if isinstance(tool_type, str) and tool_type.startswith("tool_search_tool_"):
                continue
            translated_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description"),
                    "parameters": normalize_tool_parameters_schema(tool.get("input_schema")),
                },
            })
        if translated_tools:
            openai_payload["tools"] = translated_tools

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


def normalize_tool_parameters_schema(schema: Any) -> Dict[str, Any]:
    """Normalize Anthropic tool input_schema for OpenAI / Gemini chat/completions.

    Some MCP tools emit {"type": "object", "additionalProperties": false} without
    a "properties" key, which upstream chat/completions backends reject with 400.
    """
    if not isinstance(schema, dict) or not schema:
        return {"type": "object", "properties": {}}
    if schema.get("type") == "object" and "properties" not in schema:
        normalized = dict(schema)
        normalized["properties"] = {}
        return normalized
    return schema


def _anthropic_image_to_openai_part(block: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an Anthropic image content block to an OpenAI image_url part."""
    source = block.get("source") if isinstance(block.get("source"), dict) else {}
    source_type = source.get("type")
    if source_type == "url" and source.get("url"):
        url = source["url"]
    else:
        media_type = source.get("media_type") or "image/png"
        data = source.get("data") or ""
        url = f"data:{media_type};base64,{data}"
    return {
        "type": "image_url",
        "image_url": {"url": url},
    }


def _block_to_text_or_part(block: Any) -> Optional[Dict[str, Any]]:
    """Translate a single Anthropic content block to an OpenAI text or image_url part."""
    if not isinstance(block, dict):
        return {"type": "text", "text": str(block)} if block is not None else None
    btype = block.get("type")
    if btype == "text":
        return {"type": "text", "text": block.get("text", "")}
    if btype == "thinking":
        return {"type": "text", "text": block.get("thinking", "")}
    if btype == "image":
        return _anthropic_image_to_openai_part(block)
    if btype == "image_url":
        return block
    if btype == "tool_reference":
        tool_name = block.get("tool_name") or block.get("name") or ""
        text = f"Loaded tool: {tool_name}" if tool_name else json.dumps(block, ensure_ascii=False)
        return {"type": "text", "text": text}
    return {"type": "text", "text": json.dumps(block, ensure_ascii=False)}


def translate_tool_result_content(tool_content: Any) -> Any:
    """Translate Anthropic tool_result.content to OpenAI tool message content.

    Strings pass through the suffix filter. Block lists are converted to OpenAI
    content parts (image -> image_url, tool_reference -> text, etc.) when images
    are present, or joined into text when all blocks are textual.
    """
    if isinstance(tool_content, str):
        return apply_tool_result_suffix_filter(tool_content)
    if isinstance(tool_content, list):
        has_image = any(
            isinstance(b, dict) and b.get("type") in ("image", "image_url")
            for b in tool_content
        )
        if not has_image:
            texts: List[str] = []
            for b in tool_content:
                part = _block_to_text_or_part(b)
                if part and part.get("type") == "text":
                    texts.append(apply_tool_result_suffix_filter(part.get("text", "")))
            return "\n\n".join(texts)

        parts: List[Dict[str, Any]] = []
        for b in tool_content:
            part = _block_to_text_or_part(b)
            if part is None:
                continue
            if part.get("type") == "text":
                part = {"type": "text", "text": apply_tool_result_suffix_filter(part.get("text", ""))}
            parts.append(part)
        return parts if parts else ""
    if tool_content is None:
        return ""
    return json.dumps(tool_content, ensure_ascii=False)


def translate_content_blocks(blocks: List[Dict]) -> Any:
    """Translate Anthropic content blocks to OpenAI format"""
    has_image = any(
        isinstance(b, dict) and b.get("type") in ("image", "image_url")
        for b in blocks
    )

    if not has_image:
        # Just combine text
        texts = []
        for b in blocks:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                texts.append(b.get("text", ""))
            elif b.get("type") == "thinking":
                texts.append(b.get("thinking", ""))
            elif b.get("type") == "tool_reference":
                part = _block_to_text_or_part(b)
                if part:
                    texts.append(part.get("text", ""))
        return "\n\n".join(texts) if texts else None

    # Handle mixed content with images
    parts = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if b.get("type") in ("text", "thinking", "image", "image_url", "tool_reference"):
            part = _block_to_text_or_part(b)
            if part is not None:
                parts.append(part)
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
