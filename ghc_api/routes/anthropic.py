"""
Anthropic-compatible API routes
"""

import json
import time
import uuid
from typing import Dict, Generator, Any

import requests
from flask import Blueprint, Response, jsonify, request, stream_with_context

from ..api_helpers import (
    ensure_copilot_token,
    get_copilot_base_url,
    get_copilot_headers,
    supports_direct_anthropic_api,
    count_tokens,
)
from ..cache import cache
from ..streaming import AnthropicStreamState, reconstruct_openai_response_from_chunks, translate_chunk_to_anthropic_events
from ..translator import (
    translate_anthropic_to_openai,
    translate_model_name,
    translate_openai_to_anthropic,
    apply_system_prompt_filters,
    apply_tool_result_suffix_filter,
)
from ..utils import log_error_request, is_orphaned_tool_result_error, remove_orphaned_tool_results, extract_orphaned_tool_use_ids, log_tool_result_cleanup
from ..state import state

anthropic_bp = Blueprint('anthropic', __name__)


# Fields supported by Copilot's Anthropic API endpoint
COPILOT_SUPPORTED_FIELDS = {
    "model", "messages", "max_tokens", "system", "metadata",
    "stop_sequences", "stream", "temperature", "top_p", "top_k",
    "tools", "tool_choice", "thinking", "service_tier",
}


def filter_payload_for_copilot(payload: Dict) -> Dict:
    """Filter payload to only include fields supported by Copilot's Anthropic API."""
    filtered = {}
    unsupported_fields = []

    for key, value in payload.items():
        if key in COPILOT_SUPPORTED_FIELDS:
            filtered[key] = value
        else:
            unsupported_fields.append(key)

    if unsupported_fields:
        print(f"[DirectAnthropic] Filtered unsupported fields: {', '.join(unsupported_fields)}")

    return filtered


def adjust_max_tokens_for_thinking(payload: Dict) -> Dict:
    """Adjust max_tokens if thinking is enabled.

    According to Anthropic docs, max_tokens must be greater than thinking.budget_tokens.
    """
    thinking = payload.get("thinking")
    if not thinking:
        return payload

    budget_tokens = thinking.get("budget_tokens")
    if not budget_tokens:
        return payload

    max_tokens = payload.get("max_tokens", 0)
    if max_tokens <= budget_tokens:
        response_buffer = min(16384, budget_tokens)
        new_max_tokens = budget_tokens + response_buffer
        print(f"[DirectAnthropic] Adjusted max_tokens: {max_tokens} → {new_max_tokens} (thinking.budget_tokens={budget_tokens})")
        return {**payload, "max_tokens": new_max_tokens}

    return payload


def get_anthropic_headers(enable_vision: bool = False) -> Dict[str, str]:
    """Get headers for direct Anthropic API requests to Copilot."""
    headers = get_copilot_headers(enable_vision)
    headers["anthropic-version"] = "2023-06-01"
    return headers


def apply_system_prompt_filters_to_payload(payload: Dict) -> Dict:
    """Apply system prompt filters to Anthropic payload.

    Handles both string and list formats of system prompt.
    - Removes strings specified in state.system_prompt_remove
    - Adds strings specified in state.system_prompt_add (if not already present)
    Returns a new payload with filtered system prompt.
    """
    system = payload.get("system")

    # If no system prompt exists but we have content to add, create one
    if not system:
        if state.system_prompt_add:
            # Create new system prompt with all add strings
            new_system = []
            for add_str in state.system_prompt_add:
                print(f"[Content Filter] Added new system prompt block: {add_str[:50]}{'...' if len(add_str) > 50 else ''}")
                new_system.append({"type": "text", "text": add_str})
            if new_system:
                return {**payload, "system": new_system}
        return payload

    if isinstance(system, str):
        filtered_system = apply_system_prompt_filters(system)
        if filtered_system != system or state.system_prompt_add:
            for add_str in state.system_prompt_add:
                if add_str not in filtered_system:
                    filtered_system = filtered_system + "\n\n" + add_str
                    print(f"[Content Filter] Added to system prompt: {add_str[:50]}{'...' if len(add_str) > 50 else ''}")
            return {**payload, "system": filtered_system}
    elif isinstance(system, list):
        # Handle list format: filter each text block for removals
        new_system = []
        modified = False

        # Collect all text content for checking if add strings exist
        all_text_content = ""
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                all_text_content += block.get("text", "") + "\n"

        # Apply removal filters to each text block
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                original_text = block.get("text", "")
                # Only apply removal filters, not additions (we'll handle additions separately)
                filtered_text = original_text
                for remove_str in state.system_prompt_remove:
                    if remove_str in filtered_text:
                        filtered_text = filtered_text.replace(remove_str, "")
                        print(f"[Content Filter] Removed from system prompt: {remove_str[:50]}{'...' if len(remove_str) > 50 else ''}")

                if filtered_text != original_text:
                    modified = True
                    new_system.append({**block, "text": filtered_text})
                else:
                    new_system.append(block)
            else:
                print(f"[Content Filter] Non-text system block passed through without modification. type={block.get('type') if isinstance(block, dict) else type(block)}")
                new_system.append(block)

        # Check if each add string exists in the combined text content
        # If not, add a new text block for it
        for add_str in state.system_prompt_add:
            if add_str not in all_text_content:
                print(f"[Content Filter] Added new system prompt block: {add_str[:50]}{'...' if len(add_str) > 50 else ''}")
                new_system.append({"type": "text", "text": add_str})
                modified = True

        if modified:
            return {**payload, "system": new_system}

    return payload


def apply_tool_result_suffix_filter_to_payload(payload: Dict) -> Dict:
    """Apply tool result suffix filters to Anthropic payload.

    Removes trailing suffixes from tool_result content in messages.
    Returns a new payload with filtered tool results.
    """
    messages = payload.get("messages", [])
    if not messages:
        return payload

    new_messages = []
    modified = False

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            new_messages.append(msg)
            continue

        new_content = []
        msg_modified = False

        for block in content:
            if block.get("type") == "tool_result":
                tool_content = block.get("content", "")
                if isinstance(tool_content, str):
                    filtered_content = apply_tool_result_suffix_filter(tool_content)
                    if filtered_content != tool_content:
                        msg_modified = True
                        new_content.append({**block, "content": filtered_content})
                    else:
                        new_content.append(block)
                else:
                    new_content.append(block)
            else:
                new_content.append(block)

        if msg_modified:
            modified = True
            new_messages.append({**msg, "content": new_content})
        else:
            new_messages.append(msg)

    if modified:
        return {**payload, "messages": new_messages}

    return payload


@anthropic_bp.route("/v1/messages/count_tokens", methods=["POST"])
def anthropic_count_tokens():
    """Handle Anthropic token counting API.

    This endpoint counts tokens in the request payload for context window management.
    """
    try:
        ensure_copilot_token()
        payload = request.get_json()

        model_id = payload.get("model", "")

        # Find the model in cached models
        selected_model = None
        if state.models and state.models.get("data"):
            selected_model = next(
                (m for m in state.models["data"] if m.get("id") == model_id),
                None
            )

        if not selected_model:
            print(f"[count_tokens] Model {model_id} not found, returning default token count")
            return jsonify({"input_tokens": 1})

        # Count tokens from system prompt
        total_tokens = 0
        system = payload.get("system")
        if system:
            if isinstance(system, str):
                total_tokens += count_tokens(system, model_id)
            elif isinstance(system, list):
                for block in system:
                    if isinstance(block, dict) and block.get("type") == "text":
                        total_tokens += count_tokens(block.get("text", ""), model_id)

        # Count tokens from messages
        for msg in payload.get("messages", []):
            content = msg.get("content", "")
            if isinstance(content, str):
                total_tokens += count_tokens(content, model_id)
            elif isinstance(content, list):
                for block in content:
                    if block.get("type") == "text":
                        total_tokens += count_tokens(block.get("text", ""), model_id)
                    elif block.get("type") == "tool_result":
                        tool_content = block.get("content", "")
                        if isinstance(tool_content, str):
                            total_tokens += count_tokens(tool_content, model_id)
                    elif block.get("type") == "tool_use":
                        # Count tool input as JSON string
                        tool_input = block.get("input", {})
                        total_tokens += count_tokens(json.dumps(tool_input), model_id)

        # Count tokens from tools definitions
        tools = payload.get("tools", [])
        if tools:
            # Add base overhead for tool use capability (similar to copilot-api-js)
            if model_id.startswith("claude"):
                total_tokens += 346
            elif model_id.startswith("grok"):
                total_tokens += 480

            # Count tool definition tokens
            for tool in tools:
                total_tokens += count_tokens(tool.get("name", ""), model_id)
                total_tokens += count_tokens(tool.get("description", ""), model_id)
                input_schema = tool.get("input_schema", {})
                total_tokens += count_tokens(json.dumps(input_schema), model_id)

        # Apply buffer for non-Anthropic vendors (similar to copilot-api-js)
        vendor = selected_model.get("vendor", "")
        if vendor != "Anthropic":
            if model_id.startswith("grok"):
                total_tokens = int(total_tokens * 1.03)
            else:
                total_tokens = int(total_tokens * 1.05)

        return jsonify({"input_tokens": total_tokens})

    except Exception as e:
        print(f"[count_tokens] Error: {e}")
        return jsonify({"input_tokens": 1})


@anthropic_bp.route("/v1/messages", methods=["POST"])
def anthropic_messages():
    """Handle Anthropic messages API"""
    start_time = time.time()
    ensure_copilot_token()
    anthropic_payload = request.get_json()
    request_id = str(uuid.uuid4())

    original_model = anthropic_payload.get("model", "unknown")

    # Translate model name (applies to both paths)
    translated_model = translate_model_name(original_model)
    if translated_model != original_model:
        print(f"[Anthropic API] Model name translated: {original_model} -> {translated_model}")
        anthropic_payload = {**anthropic_payload, "model": translated_model}

    # Apply system prompt filters (applies to both paths)
    anthropic_payload = apply_system_prompt_filters_to_payload(anthropic_payload)

    # Apply tool result suffix filters (applies to both paths)
    anthropic_payload = apply_tool_result_suffix_filter_to_payload(anthropic_payload)

    # Check if this model supports direct Anthropic API
    use_direct_api = supports_direct_anthropic_api(translated_model)

    if use_direct_api:
        print(f"[Anthropic API] Using direct Anthropic API path for model: {translated_model}")
        return handle_direct_anthropic_request(anthropic_payload, request_id, start_time, original_model, translated_model)
    else:
        print(f"[Anthropic API] Using OpenAI translation path for model: {translated_model}")
        return handle_translated_request(anthropic_payload, request_id, start_time, original_model, translated_model)


def handle_direct_anthropic_request(anthropic_payload: Dict, request_id: str, start_time: float,
                                     original_model: str, translated_model: str) -> Response:
    """Handle request using direct Anthropic API (no translation needed)."""
    request_size = len(json.dumps(anthropic_payload))

    # Check for vision content
    enable_vision = any(
        isinstance(msg.get("content"), list) and
        any(p.get("type") == "image" for p in msg.get("content", []))
        for msg in anthropic_payload.get("messages", [])
    )

    # Agent/user check for X-Initiator header
    is_agent_call = any(
        msg.get("role") == "assistant"
        for msg in anthropic_payload.get("messages", [])
    )

    headers = get_anthropic_headers(enable_vision)
    headers["X-Initiator"] = "agent" if is_agent_call else "user"

    max_retries = 3
    current_payload = anthropic_payload
    cleanup_log_entry = None

    for attempt in range(max_retries + 1):
        # Filter and adjust payload for Copilot
        filtered_payload = filter_payload_for_copilot(current_payload)
        filtered_payload = adjust_max_tokens_for_thinking(filtered_payload)

        if filtered_payload.get("stream"):
            return stream_direct_anthropic(filtered_payload, headers, request_id,
                                            current_payload, request_size, start_time,
                                            original_model, translated_model)

        # Non-streaming request
        response = requests.post(
            f"{get_copilot_base_url()}/v1/messages",
            headers=headers,
            json=filtered_payload,
            timeout=1200,
        )

        duration = round(time.time() - start_time, 2)

        if response.ok:
            anthropic_response = response.json()

            # Cache the request/response
            usage = anthropic_response.get("usage", {})
            cache.add_request(request_id, {
                "request_body": current_payload,
                "response_body": anthropic_response,
                "model": original_model,
                "translated_model": translated_model if translated_model != original_model else None,
                "endpoint": "/v1/messages",
                "status_code": response.status_code,
                "request_size": request_size,
                "response_size": len(json.dumps(anthropic_response)),
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "duration": duration,
            })

            if cleanup_log_entry is not None:
                cleanup_log_entry["modified_request"] = current_payload
                cleanup_log_entry["final_status_code"] = response.status_code
                cleanup_log_entry["final_response"] = anthropic_response
                log_tool_result_cleanup(cleanup_log_entry)

            return jsonify(anthropic_response)
        else:
            log_error_request("/v1/messages", current_payload, response.text, response.status_code)

            # Handle orphaned tool_result error with retry
            if is_orphaned_tool_result_error(response.status_code, response.text):
                orphaned_ids = extract_orphaned_tool_use_ids(response.text)
                if orphaned_ids:
                    print(f"[Direct Anthropic] Attempt {attempt + 1}: Found orphaned tool_result IDs: {orphaned_ids}")

                    if cleanup_log_entry is None:
                        cleanup_log_entry = {
                            "request_id": request_id,
                            "original_request": anthropic_payload,
                            "error_response": response.text,
                            "error_status_code": response.status_code,
                            "orphaned_ids": orphaned_ids,
                        }
                    else:
                        cleanup_log_entry["orphaned_ids"].extend(orphaned_ids)

                    cleaned_messages = remove_orphaned_tool_results(
                        current_payload.get("messages", []), orphaned_ids
                    )
                    current_payload = dict(current_payload)
                    current_payload["messages"] = cleaned_messages
                    print(f"[Direct Anthropic] Retrying with cleaned messages...")
                    continue

    # Final failure after all retries
    if cleanup_log_entry is not None:
        cleanup_log_entry["modified_request"] = current_payload
        cleanup_log_entry["final_status_code"] = response.status_code
        cleanup_log_entry["final_response"] = response.text
        log_tool_result_cleanup(cleanup_log_entry)

    return Response(response.text, status=response.status_code, mimetype="application/json")


def stream_direct_anthropic(filtered_payload: Dict, headers: Dict, request_id: str,
                            anthropic_payload: Dict, request_size: int, start_time: float,
                            original_model: str, translated_model: str) -> Response:
    """Handle streaming direct Anthropic response (passthrough SSE events)."""
    # Start tracking request immediately
    cache.start_request(request_id, {
        "request_body": anthropic_payload,
        "model": original_model,
        "translated_model": translated_model if translated_model != original_model else None,
        "endpoint": "/v1/messages",
        "request_size": request_size,
    })

    def generate() -> Generator[str, None, None]:
        total_input_tokens = 0
        total_output_tokens = 0
        error_occurred = False
        status_code = 200
        first_chunk_received = False
        accumulated_content = []
        accumulated_model = original_model

        try:
            cache.update_request_state(request_id, cache.STATE_SENDING)

            response = requests.post(
                f"{get_copilot_base_url()}/v1/messages",
                headers=headers,
                json=filtered_payload,
                stream=True,
                timeout=1200,
            )
            status_code = response.status_code

            for line in response.iter_lines():
                if not line:
                    continue

                line = line.decode("utf-8")

                # Handle SSE format
                if line.startswith("event: "):
                    event_type = line[7:]
                    continue

                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break

                    try:
                        event = json.loads(data)

                        if not first_chunk_received:
                            first_chunk_received = True
                            cache.update_request_state(request_id, cache.STATE_RECEIVING)

                        # Extract usage info from events
                        event_type = event.get("type", "")
                        if event_type == "message_start":
                            msg = event.get("message", {})
                            accumulated_model = msg.get("model", original_model)
                            usage = msg.get("usage", {})
                            total_input_tokens = usage.get("input_tokens", 0)
                        elif event_type == "message_delta":
                            usage = event.get("usage", {})
                            total_output_tokens = usage.get("output_tokens", 0)
                        elif event_type == "content_block_delta":
                            delta = event.get("delta", {})
                            if delta.get("type") == "text_delta":
                                accumulated_content.append(delta.get("text", ""))

                        # Forward event directly to client
                        yield f"event: {event_type}\ndata: {data}\n\n"

                    except json.JSONDecodeError:
                        continue

        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            error_occurred = True
            status_code = 504
            print(f"[Stream Direct Anthropic] Upstream timeout/connection error for request {request_id}: {type(e).__name__}")
        except GeneratorExit:
            error_occurred = True
            print(f"[Stream Direct Anthropic] Client disconnected for request {request_id}")
            cache.update_request_state(request_id, cache.STATE_ERROR, status_code=499)
            return
        except Exception as e:
            error_occurred = True
            status_code = 500
            print(f"[Stream Direct Anthropic] Error for request {request_id}: {type(e).__name__}: {e}")
            try:
                error_event = {"type": "error", "error": {"type": "api_error", "message": str(e)}}
                yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
            except GeneratorExit:
                cache.update_request_state(request_id, cache.STATE_ERROR, status_code=499)
                return

        duration = round(time.time() - start_time, 2)

        # Build response for cache
        anthropic_response = {
            "id": request_id,
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "".join(accumulated_content)}] if accumulated_content else [],
            "model": accumulated_model,
            "usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
            },
        }

        if error_occurred and not accumulated_content:
            anthropic_response = {"error": {"type": "api_error", "message": "Stream interrupted"}}

        cache.complete_request(request_id, {
            "request_body": anthropic_payload,
            "response_body": anthropic_response,
            "model": original_model,
            "translated_model": translated_model if translated_model != original_model else None,
            "endpoint": "/v1/messages",
            "status_code": status_code,
            "request_size": request_size,
            "response_size": len(json.dumps(anthropic_response)),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "duration": duration,
        })

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def handle_translated_request(anthropic_payload: Dict, request_id: str, start_time: float,
                               original_model: str, translated_model: str) -> Response:
    """Handle request using OpenAI translation path."""
    # Check for vision content
    enable_vision = any(
        isinstance(msg.get("content"), list) and
        any(p.get("type") == "image" for p in msg.get("content", []))
        for msg in anthropic_payload.get("messages", [])
    )

    request_size = len(json.dumps(anthropic_payload))

    max_retries = 3
    current_payload = anthropic_payload
    cleanup_log_entry = None

    for attempt in range(max_retries + 1):
        # Translate to OpenAI format
        openai_payload = translate_anthropic_to_openai(current_payload)
        is_agent_call = any(
            msg.get("role") in ("assistant", "tool")
            for msg in openai_payload.get("messages", [])
        )

        headers = get_copilot_headers(enable_vision)
        headers["X-Initiator"] = "agent" if is_agent_call else "user"

        if anthropic_payload.get("stream"):
            return stream_anthropic_messages(openai_payload, headers, request_id,
                                            anthropic_payload, request_size, start_time,
                                            original_model, translated_model)

        # Non-streaming request
        response = requests.post(
            f"{get_copilot_base_url()}/chat/completions",
            headers=headers,
            json=openai_payload,
            timeout=1200,
        )

        duration = round(time.time() - start_time, 2)

        if response.ok:
            openai_response = response.json()
            anthropic_response = translate_openai_to_anthropic(openai_response)

            # Cache the request/response
            usage = openai_response.get("usage", {})
            cache.add_request(request_id, {
                "request_body": current_payload,
                "response_body": anthropic_response,
                "model": original_model,
                "translated_model": translated_model if translated_model != original_model else None,
                "endpoint": "/v1/messages",
                "status_code": response.status_code,
                "request_size": request_size,
                "response_size": len(json.dumps(anthropic_response)),
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "duration": duration,
            })

            return jsonify(anthropic_response)
        else:
            log_error_request("/v1/messages", anthropic_payload, response.text, response.status_code)
            if is_orphaned_tool_result_error(response.status_code, response.text):
                orphaned_ids = extract_orphaned_tool_use_ids(response.text)
                if orphaned_ids:
                    print(f"[Anthropic API] Attempt {attempt + 1}: Found orphaned tool_result IDs: {orphaned_ids}")

                    if cleanup_log_entry is None:
                        cleanup_log_entry = {
                            "request_id": request_id,
                            "original_request": anthropic_payload,
                            "error_response": response.text,
                            "error_status_code": response.status_code,
                            "orphaned_ids": orphaned_ids,
                        }
                    else:
                        cleanup_log_entry["orphaned_ids"].extend(orphaned_ids)

                    cleaned_messages = remove_orphaned_tool_results(
                        current_payload.get("messages", []), orphaned_ids
                    )
                    current_payload = dict(current_payload)
                    current_payload["messages"] = cleaned_messages
                    print(f"[Anthropic API] Retrying with cleaned messages...")
                    continue

        if cleanup_log_entry is not None:
            cleanup_log_entry["modified_request"] = current_payload
            cleanup_log_entry["final_status_code"] = response.status_code
            cleanup_log_entry["final_response"] = response.text
            log_tool_result_cleanup(cleanup_log_entry)

        return Response(response.text, status=response.status_code, mimetype="application/json")


def stream_anthropic_messages(openai_payload: Dict, headers: Dict, request_id: str,
                              anthropic_payload: Dict, request_size: int, start_time: float,
                              original_model: str, translated_model: str) -> Response:
    """Handle streaming Anthropic messages"""
    # Start tracking request immediately
    cache.start_request(request_id, {
        "request_body": anthropic_payload,
        "model": original_model,
        "translated_model": translated_model if translated_model != original_model else None,
        "endpoint": "/v1/messages",
        "request_size": request_size,
    })

    def generate() -> Generator[str, None, None]:
        stream_state = AnthropicStreamState()
        response_chunks = []
        total_output_tokens = 0
        total_input_tokens = 0
        error_occurred = False
        status_code = 200
        first_chunk_received = False

        try:
            # Update state to sending
            cache.update_request_state(request_id, cache.STATE_SENDING)

            response = requests.post(
                f"{get_copilot_base_url()}/chat/completions",
                headers=headers,
                json=openai_payload,
                stream=True,
                timeout=1200,
            )
            status_code = response.status_code

            for line in response.iter_lines():
                if not line:
                    continue

                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        response_chunks.append(chunk)

                        # Update state to receiving on first chunk
                        if not first_chunk_received:
                            first_chunk_received = True
                            cache.update_request_state(request_id, cache.STATE_RECEIVING)

                        if chunk.get("usage"):
                            total_output_tokens = chunk["usage"].get("completion_tokens", 0)
                            total_input_tokens = chunk["usage"].get("prompt_tokens", 0)

                        # Translate to Anthropic events
                        events = translate_chunk_to_anthropic_events(chunk, stream_state)
                        for event in events:
                            yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                    except json.JSONDecodeError:
                        continue
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            # Timeout or connection error from upstream - log but don't try to yield after client disconnect
            error_occurred = True
            status_code = 504
            print(f"[Stream Anthropic] Upstream timeout/connection error for request {request_id}: {type(e).__name__}")
        except GeneratorExit:
            # Client disconnected - this is normal, just clean up
            error_occurred = True
            print(f"[Stream Anthropic] Client disconnected for request {request_id}")
            # Update state to error since client disconnected
            cache.update_request_state(request_id, cache.STATE_ERROR, status_code=499)
            return
        except Exception as e:
            error_occurred = True
            status_code = 500
            print(f"[Stream Anthropic] Error for request {request_id}: {type(e).__name__}: {e}")
            try:
                yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': str(e)}})}\n\n"
            except GeneratorExit:
                # Client already disconnected, can't yield
                cache.update_request_state(request_id, cache.STATE_ERROR, status_code=499)
                return

        duration = round(time.time() - start_time, 2)

        # Reconstruct the OpenAI response then translate to Anthropic format
        reconstructed_openai = reconstruct_openai_response_from_chunks(response_chunks)
        anthropic_response = translate_openai_to_anthropic(reconstructed_openai) if reconstructed_openai else {}
        if error_occurred and not anthropic_response:
            anthropic_response = {"error": {"type": "api_error", "message": "Stream interrupted"}}

        # Complete the request in cache
        cache.complete_request(request_id, {
            "request_body": anthropic_payload,
            "response_body": anthropic_response,
            "model": original_model,
            "translated_model": translated_model if translated_model != original_model else None,
            "endpoint": "/v1/messages",
            "status_code": status_code,
            "request_size": request_size,
            "response_size": sum(len(json.dumps(c)) for c in response_chunks),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "duration": duration,
        })

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
