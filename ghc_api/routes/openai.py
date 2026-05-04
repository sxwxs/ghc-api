"""
OpenAI-compatible API routes
"""

import json
import time
import uuid
from datetime import datetime
from typing import Dict, Generator

import requests
from flask import Blueprint, Response, jsonify, request, stream_with_context

from ..api_helpers import (
    ensure_copilot_token,
    get_copilot_base_url,
    get_copilot_headers,
    supports_chat_completions_api,
    supports_responses_api,
)
from ..cache import cache
from ..state import state
from ..streaming import reconstruct_openai_response_from_chunks
from ..translator import translate_model_name
from ..utils import log_error_request, log_connection_retry, is_encrypted_content_parse_error

openai_bp = Blueprint('openai', __name__)


@openai_bp.route("/v1/models", methods=["GET"])
@openai_bp.route("/models", methods=["GET"])
def list_models():
    """List available models"""
    from ..api_helpers import fetch_models

    try:
        ensure_copilot_token()
        if not state.models:
            fetch_models()

        models = [
            {
                "id": m["id"],
                "object": "model",
                "type": "model",
                "created": 0,
                "created_at": datetime.utcfromtimestamp(0).isoformat() + "Z",
                "owned_by": m.get("vendor", "unknown"),
                "display_name": m.get("name", m["id"]),
            }
            for m in state.models.get("data", [])
        ]

        return jsonify({
            "object": "list",
            "data": models,
            "has_more": False,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@openai_bp.route("/v1/models/full/", methods=["GET"])
@openai_bp.route("/models/full/", methods=["GET"])
def list_models_full():
    return jsonify(state.models)

@openai_bp.route("/v1/chat/completions", methods=["POST"])
@openai_bp.route("/chat/completions", methods=["POST"])
def chat_completions():
    """Handle chat completions (OpenAI format)"""
    try:
        start_time = time.time()
        ensure_copilot_token()
        payload = request.get_json()
        request_id = str(uuid.uuid4())

        # Capture incoming request headers
        request_headers = dict(request.headers)

        # Get the original and translated model names
        original_model = payload.get("model", "unknown")
        translated_model = translate_model_name(original_model)

        # Update payload with translated model if different
        if translated_model != original_model:
            payload = dict(payload)
            payload["model"] = translated_model

        # Check for vision content
        enable_vision = False
        for msg in payload.get("messages", []):
            content = msg.get("content")
            if isinstance(content, list):
                if any(p.get("type") == "image_url" for p in content):
                    enable_vision = True
                    break

        # Detect agent vs user call
        is_agent_call = any(
            msg.get("role") in ("assistant", "tool")
            for msg in payload.get("messages", [])
        )

        headers = get_copilot_headers(enable_vision)
        headers["X-Initiator"] = "agent" if is_agent_call else "user"

        request_body = json.dumps(payload)
        request_size = len(request_body)

        if should_use_gpt_chat_responses_compat(translated_model):
            if payload.get("stream"):
                return stream_chat_completions_via_responses(
                    payload, headers, request_id, request_body, request_size, start_time,
                    original_model, translated_model, request_headers
                )
            return chat_completions_via_responses(
                payload, headers, request_id, request_body, request_size, start_time,
                original_model, translated_model, request_headers
            )

        if payload.get("stream"):
            return stream_chat_completions(payload, headers, request_id, request_body, request_size, start_time,
                                           original_model, translated_model, request_headers)

        # Non-streaming request
        connection_retries = state.max_connection_retries
        last_connection_error = None
        for conn_attempt in range(connection_retries + 1):
            try:
                response = requests.post(
                    f"{get_copilot_base_url()}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=1200,
                )
                last_connection_error = None
                break
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                last_connection_error = e
                log_connection_retry(request_id, "/v1/chat/completions", conn_attempt, connection_retries, e)
                ensure_copilot_token()  # Refresh token in case it's a token expiration issue
                if conn_attempt < connection_retries:
                    print(f"[OpenAI API] Connection error (attempt {conn_attempt + 1}/{connection_retries + 1}) for request {request_id}: {type(e).__name__}: {e}")
                    time.sleep(min(2 ** conn_attempt, 8))
                    continue
                else:
                    print(f"[OpenAI API] Connection error (final attempt) for request {request_id}: {type(e).__name__}: {e}")

        if last_connection_error is not None:
            return jsonify({"error": f"Upstream connection error after {connection_retries + 1} attempts: {type(last_connection_error).__name__}"}), 504

        if state.auto_remove_encrypted_content_on_parse_error and is_encrypted_content_parse_error(response.status_code, response.text):
            request_input = payload.get("input")
            if isinstance(request_input, list):
                cleaned_input = []
                removed_count = 0
                for item in request_input:
                    if isinstance(item, dict) and "encrypted_content" in item:
                        removed_count += 1
                        continue
                    cleaned_input.append(item)

                if removed_count > 0:
                    retry_payload = dict(payload)
                    retry_payload["input"] = cleaned_input
                    response = requests.post(
                        f"{get_copilot_base_url()}/v1/responses",
                        headers=headers,
                        json=retry_payload,
                        timeout=1200,
                    )
                    payload = retry_payload

        duration = round(time.time() - start_time, 2)
        response_body = response.text
        response_size = len(response_body)

        if response.ok:
            result = response.json()

            # Cache the request/response
            usage = result.get("usage", {})
            cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            cache.add_request(request_id, {
                "request_headers": request_headers,
                "request_body": payload,
                "response_body": result,
                "model": original_model,
                "translated_model": translated_model if translated_model != original_model else None,
                "endpoint": "/v1/chat/completions",
                "status_code": response.status_code,
                "request_size": request_size,
                "response_size": response_size,
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "cache_read_input_tokens": cached_tokens,
                "duration": duration,
            })

            return jsonify(result)
        else:
            log_error_request("/v1/chat/completions", payload, response.text, response.status_code)
            return Response(response.text, status=response.status_code, mimetype="application/json")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def should_use_gpt_chat_responses_compat(model_id: str) -> bool:
    """Return True when legacy chat clients should be shimmed to Responses."""
    if not state.enable_gpt_chat_completions_responses_compat:
        return False
    if not isinstance(model_id, str) or not model_id.lower().startswith("gpt-"):
        return False
    return supports_responses_api(model_id) and not supports_chat_completions_api(model_id)


def chat_completions_via_responses(payload: Dict, headers: Dict, request_id: str,
                                   request_body: str, request_size: int, start_time: float,
                                   original_model: str, translated_model: str,
                                   request_headers: Dict = None) -> Response:
    """Handle non-streaming Chat Completions via the Responses API."""
    responses_payload = chat_payload_to_responses_payload(payload)
    connection_retries = state.max_connection_retries
    last_connection_error = None

    for conn_attempt in range(connection_retries + 1):
        try:
            response = requests.post(
                f"{get_copilot_base_url()}/v1/responses",
                headers=headers,
                json=responses_payload,
                timeout=1200,
            )
            last_connection_error = None
            break
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            last_connection_error = e
            log_connection_retry(request_id, "/v1/chat/completions (responses compat)", conn_attempt, connection_retries, e)
            ensure_copilot_token()
            if conn_attempt < connection_retries:
                print(f"[Chat Compat] Connection error (attempt {conn_attempt + 1}/{connection_retries + 1}) for request {request_id}: {type(e).__name__}: {e}")
                time.sleep(min(2 ** conn_attempt, 8))
                continue
            print(f"[Chat Compat] Connection error (final attempt) for request {request_id}: {type(e).__name__}: {e}")

    if last_connection_error is not None:
        return jsonify({"error": f"Upstream connection error after {connection_retries + 1} attempts: {type(last_connection_error).__name__}"}), 504

    duration = round(time.time() - start_time, 2)

    if response.ok:
        responses_result = response.json()
        result = responses_response_to_chat_completion(responses_result)
        usage = result.get("usage", {})
        cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        response_size = len(json.dumps(result))

        cache.add_request(request_id, {
            "request_headers": request_headers,
            "request_body": payload,
            "upstream_request_body": responses_payload,
            "response_body": result,
            "upstream_response_body": responses_result,
            "model": original_model,
            "translated_model": translated_model if translated_model != original_model else None,
            "endpoint": "/v1/chat/completions",
            "upstream_endpoint": "/v1/responses",
            "status_code": response.status_code,
            "request_size": request_size,
            "response_size": response_size,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_read_input_tokens": cached_tokens,
            "duration": duration,
        })

        return jsonify(result)

    log_error_request("/v1/chat/completions (responses compat)", responses_payload, response.text, response.status_code)
    return Response(response.text, status=response.status_code, mimetype="application/json")


def chat_payload_to_responses_payload(payload: Dict) -> Dict:
    responses_payload = {
        "model": payload.get("model"),
        "input": chat_messages_to_responses_input(payload.get("messages", [])),
    }

    passthrough_fields = [
        "metadata",
        "parallel_tool_calls",
        "reasoning",
        "store",
        "stream",
        "temperature",
        "top_p",
        "truncation",
    ]
    for field in passthrough_fields:
        if field in payload:
            responses_payload[field] = payload[field]

    if "tools" in payload:
        responses_payload["tools"] = chat_tools_to_responses_tools(payload.get("tools"))
    if "tool_choice" in payload:
        responses_payload["tool_choice"] = chat_tool_choice_to_responses_tool_choice(payload.get("tool_choice"))

    if "max_completion_tokens" in payload:
        responses_payload["max_output_tokens"] = payload["max_completion_tokens"]
    elif "max_tokens" in payload:
        responses_payload["max_output_tokens"] = payload["max_tokens"]

    if "stop" in payload:
        responses_payload["stop"] = payload["stop"]

    return {key: value for key, value in responses_payload.items() if value is not None}


def chat_tools_to_responses_tools(tools):
    if not isinstance(tools, list):
        return tools

    converted = []
    for tool in tools:
        if not isinstance(tool, dict):
            converted.append(tool)
            continue

        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            function = tool["function"]
            converted_tool = {
                "type": "function",
                "name": function.get("name"),
            }
            if "description" in function:
                converted_tool["description"] = function["description"]
            if "parameters" in function:
                converted_tool["parameters"] = function["parameters"]
            if "strict" in function:
                converted_tool["strict"] = function["strict"]
            converted.append({key: value for key, value in converted_tool.items() if value is not None})
            continue

        converted.append(tool)

    return converted


def chat_tool_choice_to_responses_tool_choice(tool_choice):
    if not isinstance(tool_choice, dict):
        return tool_choice

    if tool_choice.get("type") == "function" and isinstance(tool_choice.get("function"), dict):
        return {
            "type": "function",
            "name": tool_choice["function"].get("name"),
        }

    return tool_choice


def chat_messages_to_responses_input(messages: list) -> list:
    responses_input = []
    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = message.get("content")

        if role == "tool":
            responses_input.append({
                "type": "function_call_output",
                "call_id": message.get("tool_call_id"),
                "output": chat_content_to_text(content),
            })
            continue

        if role == "assistant" and message.get("tool_calls"):
            if content:
                responses_input.append({
                    "role": "assistant",
                    "content": chat_content_to_responses_content(content),
                })
            for tool_call in message.get("tool_calls", []):
                function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                responses_input.append({
                    "type": "function_call",
                    "call_id": tool_call.get("id") if isinstance(tool_call, dict) else None,
                    "name": function.get("name"),
                    "arguments": function.get("arguments", "{}"),
                })
            continue

        if role not in ("developer", "system", "user", "assistant"):
            role = "user"

        responses_input.append({
            "role": role,
            "content": chat_content_to_responses_content(content),
        })

    return responses_input


def chat_content_to_responses_content(content):
    if isinstance(content, list):
        converted = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                converted.append({"type": "input_text", "text": part.get("text", "")})
            elif part_type == "image_url":
                image_url = part.get("image_url", {})
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                converted.append({"type": "input_image", "image_url": url})
            else:
                converted.append(part)
        return converted
    return "" if content is None else content


def chat_content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
                else:
                    parts.append(json.dumps(part))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content)


def responses_response_to_chat_completion(response_json: Dict) -> Dict:
    message = responses_output_to_chat_message(response_json)
    usage = responses_usage_to_chat_usage(response_json.get("usage", {}))
    finish_reason = "tool_calls" if message.get("tool_calls") else "stop"

    return {
        "id": response_json.get("id", f"chatcmpl-{uuid.uuid4()}"),
        "object": "chat.completion",
        "created": int(response_json.get("created_at") or time.time()),
        "model": response_json.get("model", ""),
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": usage,
    }


def responses_output_to_chat_message(response_json: Dict) -> Dict:
    text_parts = []
    tool_calls = []
    has_output_text = bool(response_json.get("output_text"))

    if has_output_text:
        text_parts.append(response_json["output_text"])

    for item in response_json.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "message":
            for part in item.get("content", []) or []:
                if not isinstance(part, dict):
                    continue
                if not has_output_text and part.get("type") in ("output_text", "text"):
                    text_parts.append(part.get("text", ""))
                elif not has_output_text and part.get("type") == "refusal":
                    text_parts.append(part.get("refusal", ""))
        elif item_type == "function_call":
            tool_calls.append({
                "id": item.get("call_id") or item.get("id"),
                "type": "function",
                "function": {
                    "name": item.get("name"),
                    "arguments": item.get("arguments", ""),
                },
            })

    content = "".join(text_parts)
    message = {
        "role": "assistant",
        "content": content if content else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def responses_usage_to_chat_usage(usage: Dict) -> Dict:
    if not isinstance(usage, dict):
        usage = {}
    prompt_tokens = usage.get("input_tokens", 0) or 0
    completion_tokens = usage.get("output_tokens", 0) or 0
    result = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": usage.get("total_tokens", prompt_tokens + completion_tokens) or 0,
    }
    input_details = usage.get("input_tokens_details")
    if isinstance(input_details, dict):
        result["prompt_tokens_details"] = {
            "cached_tokens": input_details.get("cached_tokens", 0) or 0,
        }
    output_details = usage.get("output_tokens_details")
    if isinstance(output_details, dict):
        result["completion_tokens_details"] = output_details
    return result


def stream_chat_completions_via_responses(payload: Dict, headers: Dict, request_id: str,
                                          request_body: str, request_size: int, start_time: float,
                                          original_model: str, translated_model: str,
                                          request_headers: Dict = None) -> Response:
    """Handle streaming Chat Completions via the Responses API."""
    responses_payload = chat_payload_to_responses_payload(payload)
    responses_payload["stream"] = True

    cache.start_request(request_id, {
        "request_headers": request_headers,
        "request_body": payload,
        "upstream_request_body": responses_payload,
        "model": original_model,
        "translated_model": translated_model if translated_model != original_model else None,
        "endpoint": "/v1/chat/completions",
        "upstream_endpoint": "/v1/responses",
        "request_size": request_size,
    })

    def generate() -> Generator[str, None, None]:
        response_chunks = []
        total_output_tokens = 0
        total_input_tokens = 0
        total_cache_read_input_tokens = 0
        error_occurred = False
        status_code = 200
        first_chunk_sent = False
        response_id = f"chatcmpl-{uuid.uuid4()}"
        created = int(time.time())
        model = translated_model
        function_indexes = {}
        text_delta_seen = False

        def make_chunk(delta: Dict, finish_reason=None, usage=None) -> Dict:
            chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }],
            }
            if usage is not None:
                chunk["usage"] = usage
            return chunk

        def emit_chunk(delta: Dict, finish_reason=None, usage=None):
            chunk = make_chunk(delta, finish_reason, usage)
            response_chunks.append(chunk)
            return f"data: {json.dumps(chunk)}\n\n"

        def emit_role_if_needed():
            nonlocal first_chunk_sent
            if first_chunk_sent:
                return None
            first_chunk_sent = True
            cache.update_request_state(request_id, cache.STATE_RECEIVING)
            return emit_chunk({"role": "assistant"})

        try:
            cache.update_request_state(request_id, cache.STATE_SENDING)
            response = requests.post(
                f"{get_copilot_base_url()}/v1/responses",
                headers=headers,
                json=responses_payload,
                stream=True,
                timeout=1200,
            )
            status_code = response.status_code

            if not response.ok:
                error_occurred = True
                log_error_request("/v1/chat/completions (responses compat stream)", responses_payload, response.text, response.status_code)
                error_payload = {"error": response.text, "status_code": response.status_code}
                yield f"data: {json.dumps(error_payload)}\n\n"

            current_event = None
            for line in response.iter_lines():
                if not line:
                    continue

                line = line.decode("utf-8")
                if line.startswith("event: "):
                    current_event = line[7:]
                    continue
                if not line.startswith("data: "):
                    continue

                data = line[6:]
                if data == "[DONE]":
                    continue

                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type") or current_event
                if event_type == "response.created":
                    response_data = event.get("response", {})
                    response_id = response_data.get("id", response_id)
                    created = int(response_data.get("created_at") or created)
                    model = response_data.get("model", model)
                    role_chunk = emit_role_if_needed()
                    if role_chunk:
                        yield role_chunk
                elif event_type in ("response.output_text.delta", "response.refusal.delta"):
                    role_chunk = emit_role_if_needed()
                    if role_chunk:
                        yield role_chunk
                    delta_text = event.get("delta") or event.get("text") or ""
                    if delta_text:
                        text_delta_seen = True
                        yield emit_chunk({"content": delta_text})
                elif event_type == "response.output_item.added":
                    item = event.get("item", {})
                    if item.get("type") == "function_call":
                        role_chunk = emit_role_if_needed()
                        if role_chunk:
                            yield role_chunk
                        output_index = event.get("output_index", len(function_indexes))
                        tool_index = len(function_indexes)
                        function_indexes[output_index] = tool_index
                        yield emit_chunk({
                            "tool_calls": [{
                                "index": tool_index,
                                "id": item.get("call_id") or item.get("id"),
                                "type": "function",
                                "function": {
                                    "name": item.get("name"),
                                    "arguments": "",
                                },
                            }]
                        })
                elif event_type == "response.function_call_arguments.delta":
                    role_chunk = emit_role_if_needed()
                    if role_chunk:
                        yield role_chunk
                    output_index = event.get("output_index", 0)
                    tool_index = function_indexes.setdefault(output_index, len(function_indexes))
                    yield emit_chunk({
                        "tool_calls": [{
                            "index": tool_index,
                            "function": {"arguments": event.get("delta", "")},
                        }]
                    })
                elif event_type == "response.completed":
                    response_data = event.get("response", {})
                    usage = responses_usage_to_chat_usage(response_data.get("usage", {}))
                    total_input_tokens = usage.get("prompt_tokens", 0)
                    total_output_tokens = usage.get("completion_tokens", 0)
                    total_cache_read_input_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)

                    final_message = responses_output_to_chat_message(response_data)
                    if not text_delta_seen:
                        final_text = final_message.get("content")
                        if final_text:
                            role_chunk = emit_role_if_needed()
                            if role_chunk:
                                yield role_chunk
                            yield emit_chunk({"content": final_text})

                    if not function_indexes and final_message.get("tool_calls"):
                        role_chunk = emit_role_if_needed()
                        if role_chunk:
                            yield role_chunk
                        for tool_index, tool_call in enumerate(final_message["tool_calls"]):
                            function = tool_call.get("function", {})
                            function_indexes[tool_index] = tool_index
                            yield emit_chunk({
                                "tool_calls": [{
                                    "index": tool_index,
                                    "id": tool_call.get("id"),
                                    "type": "function",
                                    "function": {
                                        "name": function.get("name"),
                                        "arguments": function.get("arguments", ""),
                                    },
                                }]
                            })

                    finish_reason = "tool_calls" if function_indexes else "stop"
                    role_chunk = emit_role_if_needed()
                    if role_chunk:
                        yield role_chunk
                    yield emit_chunk({}, finish_reason=finish_reason, usage=usage)
                    yield "data: [DONE]\n\n"
                elif event_type in ("response.failed", "error"):
                    error_occurred = True
                    error_payload = event.get("error", event)
                    yield f"data: {json.dumps({'error': error_payload})}\n\n"
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            error_occurred = True
            status_code = 504
            print(f"[Chat Compat Stream] Upstream timeout/connection error for request {request_id}: {type(e).__name__}")
        except GeneratorExit:
            error_occurred = True
            cache.update_request_state(request_id, cache.STATE_ERROR, status_code=499)
            return
        except Exception as e:
            error_occurred = True
            status_code = 500
            print(f"[Chat Compat Stream] Error for request {request_id}: {type(e).__name__}: {e}")
            try:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            except GeneratorExit:
                cache.update_request_state(request_id, cache.STATE_ERROR, status_code=499)
                return

        duration = round(time.time() - start_time, 2)
        reconstructed_response = reconstruct_openai_response_from_chunks(response_chunks)
        if error_occurred and not reconstructed_response:
            reconstructed_response = {"error": "Stream interrupted"}

        cache.complete_request(request_id, {
            "request_body": payload,
            "upstream_request_body": responses_payload,
            "response_body": reconstructed_response,
            "model": original_model,
            "translated_model": translated_model if translated_model != original_model else None,
            "endpoint": "/v1/chat/completions",
            "upstream_endpoint": "/v1/responses",
            "status_code": status_code,
            "request_size": request_size,
            "response_size": sum(len(json.dumps(c)) for c in response_chunks),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cache_read_input_tokens": total_cache_read_input_tokens,
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


def stream_chat_completions(payload: Dict, headers: Dict, request_id: str,
                            request_body: str, request_size: int, start_time: float,
                            original_model: str, translated_model: str,
                            request_headers: Dict = None) -> Response:
    """Handle streaming chat completions"""
    # Start tracking request immediately
    cache.start_request(request_id, {
        "request_headers": request_headers,
        "request_body": payload,
        "model": original_model,
        "translated_model": translated_model if translated_model != original_model else None,
        "endpoint": "/v1/chat/completions",
        "request_size": request_size,
    })

    def generate() -> Generator[str, None, None]:
        response_chunks = []
        total_output_tokens = 0
        total_input_tokens = 0
        total_cache_read_input_tokens = 0
        error_occurred = False
        status_code = 200
        first_chunk_received = False

        try:
            # Update state to sending
            cache.update_request_state(request_id, cache.STATE_SENDING)

            response = requests.post(
                f"{get_copilot_base_url()}/chat/completions",
                headers=headers,
                json=payload,
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
                        yield "data: [DONE]\n\n"
                        break

                    try:
                        chunk = json.loads(data)
                        response_chunks.append(chunk)

                        # Update state to receiving on first chunk
                        if not first_chunk_received:
                            first_chunk_received = True
                            cache.update_request_state(request_id, cache.STATE_RECEIVING)

                        # Track tokens from streaming chunks
                        if chunk.get("usage"):
                            total_output_tokens = chunk["usage"].get("completion_tokens", 0)
                            total_input_tokens = chunk["usage"].get("prompt_tokens", 0)
                            total_cache_read_input_tokens = chunk["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)

                        yield f"data: {data}\n\n"
                    except json.JSONDecodeError:
                        continue
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            # Timeout or connection error from upstream - log but don't try to yield after client disconnect
            error_occurred = True
            status_code = 504
            print(f"[Stream] Upstream timeout/connection error for request {request_id}: {type(e).__name__}")
        except GeneratorExit:
            # Client disconnected - this clean up
            error_occurred = True
            print(f"[Stream] Client disconnected for request {request_id}")
            # Update state to error since client disconnected
            cache.update_request_state(request_id, cache.STATE_ERROR, status_code=499)
            return
        except Exception as e:
            error_occurred = True
            status_code = 500
            print(f"[Stream] Error for request {request_id}: {type(e).__name__}: {e}")
            try:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            except GeneratorExit:
                # Client already disconnected, can't yield
                cache.update_request_state(request_id, cache.STATE_ERROR, status_code=499)
                return

        duration = round(time.time() - start_time, 2)

        # Reconstruct the full response from chunks
        reconstructed_response = reconstruct_openai_response_from_chunks(response_chunks)
        if error_occurred and not reconstructed_response:
            reconstructed_response = {"error": "Stream interrupted"}

        # Complete the request in cache
        cache.complete_request(request_id, {
            "request_body": payload,
            "response_body": reconstructed_response,
            "model": original_model,
            "translated_model": translated_model if translated_model != original_model else None,
            "endpoint": "/v1/chat/completions",
            "status_code": status_code,
            "request_size": request_size,
            "response_size": sum(len(json.dumps(c)) for c in response_chunks),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cache_read_input_tokens": total_cache_read_input_tokens,
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


@openai_bp.route("/v1/responses", methods=["POST"])
@openai_bp.route("/responses", methods=["POST"])
def responses():
    """Handle OpenAI Responses API (passthrough for supported models only)"""
    try:
        start_time = time.time()
        ensure_copilot_token()
        try:
            payload = request.get_json()
        except Exception as e:
            print(f"Get json parse error {e}, try to remove \\r \\n in the request body")
            raw = request.get_data()
            raw = raw.replace(b'\n', b'')
            raw = raw.replace(b'\r', b'')
            payload = json.loads(raw.decode('utf8'))
            print("after remove \\r \\n the json can be parsed")
        request_id = str(uuid.uuid4())

        # Capture incoming request headers
        request_headers = dict(request.headers)

        original_model = payload.get("model", "unknown")
        translated_model = translate_model_name(original_model)

        # Update payload with translated model if different
        if translated_model != original_model:
            payload = dict(payload)
            payload["model"] = translated_model

        # Check if this model supports the Responses API
        if not supports_responses_api(translated_model):
            return jsonify({
                "error": {
                    "message": f"Model '{original_model}' does not support the /v1/responses endpoint.",
                    "type": "invalid_request_error",
                    "code": "unsupported_model",
                }
            }), 400

        # Check for vision content in input
        enable_vision = False
        input_content = payload.get("input")
        if isinstance(input_content, list):
            for item in input_content:
                if isinstance(item, dict):
                    content = item.get("content")
                    if isinstance(content, list):
                        if any(p.get("type") == "input_image" for p in content):
                            enable_vision = True
                            break
                    elif item.get("type") == "input_image":
                        enable_vision = True
                        break

        headers = get_copilot_headers(enable_vision)

        if 'tools' in payload:
            tools = payload['tools']
            i = 0
            while i < len(tools):
                # '{"error":{"message":"rejected tool(s): web_search","code":"invalid_request_body"}}\n'
                if tools[i]['type'] in ('web_search', 'image_generation'):
                    removed_type = tools[i]['type']
                    tools.pop(i)
                    print(f"Removed unsupported tool '{removed_type}' from payload for request {request_id}")
                else:
                    i += 1

        request_size = len(json.dumps(payload))

        # Non-streaming request
        connection_retries = state.max_connection_retries
        last_connection_error = None
        use_streaming = payload.get("stream", False)
        for conn_attempt in range(connection_retries + 1):
            try:
                response = requests.post(
                    f"{get_copilot_base_url()}/v1/responses",
                    headers=headers,
                    json=payload,
                    stream=use_streaming,
                    timeout=1200,
                )
                if use_streaming:
                    if response.ok:
                        return stream_responses(response, request_id, request_size, start_time,
                                        original_model, translated_model, payload, request_headers)
                if not response.ok:
                    print(f"Received error response for request {request_id}: {response.status_code} - {response.text}")
                    log_error_request("/v1/responses", payload, response.text, response.status_code)
                    if state.auto_remove_encrypted_content_on_parse_error and is_encrypted_content_parse_error(response.status_code, response.text):
                        request_input = payload.get("input")
                        if isinstance(request_input, list):
                            cleaned_input = []
                            removed_count = 0
                            for item in request_input:
                                if isinstance(item, dict) and "encrypted_content" in item:
                                    removed_count += 1
                                    continue
                                cleaned_input.append(item)
                            print("Warning: Detected possible encrypted content parse error in response. auto_remove_encrypted_content_on_parse_error is enabled, so will remove encrypted content and retry the request. May cause loss of information.")
                            print("Try to remove encrypted content and retry", f"Removed {removed_count} encrypted content items from input for request {request_id}")
                            if removed_count > 0:
                                retry_payload = dict(payload)
                                retry_payload["input"] = cleaned_input
                                payload = retry_payload
                                continue
                last_connection_error = None
                break
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                last_connection_error = e
                log_connection_retry(request_id, "/v1/responses", conn_attempt, connection_retries, e)
                ensure_copilot_token()  # Refresh token in case it's a token expiration issue
                if conn_attempt < connection_retries:
                    print(f"[Responses API] Connection error (attempt {conn_attempt + 1}/{connection_retries + 1}) for request {request_id}: {type(e).__name__}: {e}")
                    time.sleep(min(2 ** conn_attempt, 8))
                    continue
                else:
                    print(f"[Responses API] Connection error (final attempt) for request {request_id}: {type(e).__name__}: {e}")

        if last_connection_error is not None:
            return jsonify({"error": f"Upstream connection error after {connection_retries + 1} attempts: {type(last_connection_error).__name__}"}), 504

        duration = round(time.time() - start_time, 2)
        response_size = len(response.text)
        if response.ok:
            result = response.json()

            usage = result.get("usage", {})
            cache.add_request(request_id, {
                "request_headers": request_headers,
                "request_body": payload,
                "response_body": result,
                "model": original_model,
                "translated_model": translated_model if translated_model != original_model else None,
                "endpoint": "/v1/responses",
                "status_code": response.status_code,
                "request_size": request_size,
                "response_size": response_size,
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cache_creation_input_tokens": usage.get("input_tokens_details", {}).get("cached_tokens", 0),
                "duration": duration,
            })

            return jsonify(result)
        else:
            log_error_request("/v1/responses", payload, response.text, response.status_code)
            usage = {}
            try:
                result = response.json()
            except:
                result = response.text
            cache.add_request(request_id, {
                "request_headers": request_headers,
                "request_body": payload,
                "response_body": result,
                "model": original_model,
                "translated_model": translated_model if translated_model != original_model else None,
                "endpoint": "/v1/responses",
                "status_code": response.status_code,
                "request_size": request_size,
                "response_size": response_size,
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "duration": duration,
            })
            return Response(response.text, status=response.status_code, mimetype="application/json")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def stream_responses(response: requests.Response, request_id: str,
                     request_size: int, start_time: float,
                     original_model: str, translated_model: str, payload: dict,
                     request_headers: Dict = None) -> Response:
    """Handle streaming Responses API (passthrough SSE events)"""
    cache.start_request(request_id, {
        "request_headers": request_headers,
        "request_body": payload,
        "model": original_model,
        "translated_model": translated_model if translated_model != original_model else None,
        "endpoint": "/v1/responses",
        "request_size": request_size,
    })

    def generate() -> Generator[str, None, None]:
        total_input_tokens = 0
        total_output_tokens = 0
        total_cache_creation_input_tokens = 0
        error_occurred = False
        status_code = response.status_code
        first_chunk_received = False
        accumulated_text = []
        response_data = {}

        try:
            cache.update_request_state(request_id, cache.STATE_SENDING)

            status_code = response.status_code

            for line in response.iter_lines():
                if not line:
                    continue

                line = line.decode("utf-8")

                # The Responses API uses SSE with 'event:' and 'data:' lines
                if line.startswith("event: "):
                    # Forward event line directly
                    yield f"{line}\n"
                    continue

                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break

                    try:
                        event = json.loads(data)

                        if not first_chunk_received:
                            first_chunk_received = True
                            cache.update_request_state(request_id, cache.STATE_RECEIVING)

                        # Extract usage from response.completed event
                        event_type = event.get("type", "")
                        if event_type == "response.completed":
                            resp = event.get("response", {})
                            response_data = resp
                            usage = resp.get("usage", {})
                            total_input_tokens = usage.get("input_tokens", 0)
                            total_output_tokens = usage.get("output_tokens", 0)
                            total_cache_creation_input_tokens = usage.get("input_tokens_details", {}).get("cached_tokens", 0)
                        elif event_type == "response.output_text.delta":
                            accumulated_text.append(event.get("delta", ""))

                        yield f"data: {data}\n\n"
                    except json.JSONDecodeError:
                        yield f"data: {data}\n\n"
                        continue

        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            error_occurred = True
            status_code = 504
            print(f"[Stream Responses] Upstream timeout/connection error for request {request_id}: {type(e).__name__}")
        except GeneratorExit:
            error_occurred = True
            print(f"[Stream Responses] Client disconnected for request {request_id}")
            cache.update_request_state(request_id, cache.STATE_ERROR, status_code=499)
            return
        except Exception as e:
            error_occurred = True
            status_code = 500
            print(f"[Stream Responses] Error for request {request_id}: {type(e).__name__}: {e}")
            try:
                error_payload = {"error": str(e)}
                yield f"data: {json.dumps(error_payload)}\n\n"
            except GeneratorExit:
                cache.update_request_state(request_id, cache.STATE_ERROR, status_code=499)
                return

        duration = round(time.time() - start_time, 2)

        # Use the response.completed data if available, otherwise build a minimal response
        cached_response = response_data if response_data else {}
        if not cached_response and accumulated_text:
            cached_response = {
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "".join(accumulated_text)}]}],
            }
        if error_occurred and not response_data and not accumulated_text:
            cached_response = {"error": "Stream interrupted"}

        cache.complete_request(request_id, {
            "request_body": payload,
            "response_body": cached_response,
            "model": original_model,
            "translated_model": translated_model if translated_model != original_model else None,
            "endpoint": "/v1/responses",
            "status_code": status_code,
            "request_size": request_size,
            "response_size": len(json.dumps(cached_response)),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cache_creation_input_tokens": total_cache_creation_input_tokens,
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
