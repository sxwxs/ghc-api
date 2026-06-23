"""
OpenAI-compatible API routes
"""

import copy
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Generator

import requests
from flask import Blueprint, Response, g, jsonify, request, stream_with_context

from ..api_helpers import (
    ensure_copilot_token,
    get_copilot_base_url,
    get_copilot_headers,
    is_configured_chat_completions_support_added,
    supports_responses_api,
)
from ..auth import redact_auth_headers
from ..cache import cache
from ..config import chat_completions_model_support
from ..state import state
from ..streaming import reconstruct_openai_response_from_chunks
from ..translator import translate_model_name
from ..utils import log_error_request, log_connection_retry, is_encrypted_content_parse_error, get_client_ip

openai_bp = Blueprint('openai', __name__)


def _current_user_id() -> str:
    """Read user_id from flask.g; falls back to anonymous outside a request
    context (defensive — should never happen in production)."""
    return getattr(g, "user_id", "anonymous") or "anonymous"


def _is_gpt_model(model_id: str) -> bool:
    return isinstance(model_id, str) and model_id.lower().startswith("gpt-")


def _model_supported_endpoints(model_id: str) -> list:
    if not state.models or not state.models.get("data"):
        return []
    model = next((m for m in state.models["data"] if m.get("id") == model_id), None)
    if not model:
        return []
    endpoints = model.get("supported_endpoints", [])
    return endpoints if isinstance(endpoints, list) else []


def _has_native_chat_completions_endpoint(model_id: str) -> bool:
    endpoints = _model_supported_endpoints(model_id)
    if "/chat/completions" in endpoints:
        return True
    return (
        "/v1/chat/completions" in endpoints
        and not is_configured_chat_completions_support_added(state.models, model_id)
    )


def _should_route_chat_completions_via_responses(model_id: str) -> bool:
    """Use Responses API as a compatibility backend for configured models.

    Copilot metadata uses /responses for newer models and /chat/completions for
    older native chat models. The local endpoint remains /v1/chat/completions.
    """
    if not chat_completions_model_support.matches(model_id):
        return False
    if not supports_responses_api(model_id):
        return False
    return not _has_native_chat_completions_endpoint(model_id)


def _message_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type in ("text", "input_text", "output_text"):
                parts.append(part.get("text", ""))
            elif part_type == "image_url":
                image_url = part.get("image_url", {})
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                if url:
                    parts.append(f"[image: {url}]")
        return "\n".join(part for part in parts if part)
    return str(content)


def _chat_content_to_responses_content(content, role: str = "user"):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts = []
    text_parts = []
    has_structured_part = False
    for part in content:
        if not isinstance(part, dict):
            continue

        part_type = part.get("type")
        if part_type in ("text", "input_text", "output_text"):
            text = part.get("text", "")
            if text:
                text_parts.append(text)
                parts.append({
                    "type": "output_text" if role == "assistant" else "input_text",
                    "text": text,
                })
        elif part_type == "image_url":
            image_url = part.get("image_url", {})
            if isinstance(image_url, dict):
                url = image_url.get("url")
                detail = image_url.get("detail")
            else:
                url = image_url
                detail = None
            if url:
                image_part = {
                    "type": "input_image",
                    "image_url": url,
                }
                if detail:
                    image_part["detail"] = detail
                parts.append(image_part)
                has_structured_part = True
        elif part_type == "input_image":
            parts.append(part)
            has_structured_part = True

    if has_structured_part:
        return parts
    return "\n".join(text_parts)


def _tool_arguments_to_string(arguments) -> str:
    if arguments is None:
        return ""
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments)
    except Exception:
        return str(arguments)


def _chat_tool_call_to_responses_item(tool_call: Dict) -> Dict | None:
    if not isinstance(tool_call, dict):
        return None
    if tool_call.get("type") != "function":
        return None

    function = tool_call.get("function")
    if not isinstance(function, dict) or not function.get("name"):
        return None

    return {
        "type": "function_call",
        "call_id": tool_call.get("id") or f"call_{uuid.uuid4().hex}",
        "name": function.get("name", ""),
        "arguments": _tool_arguments_to_string(function.get("arguments", "")),
    }


def _chat_completions_to_responses_payload(payload: Dict) -> Dict:
    responses_payload = {
        "model": payload.get("model"),
        "input": [],
        "stream": bool(payload.get("stream")),
    }

    instructions = []
    for message in payload.get("messages", []):
        if not isinstance(message, dict):
            continue

        role = message.get("role", "user")
        content = message.get("content")
        if role in ("system", "developer"):
            text = _message_text(content)
            if text:
                instructions.append(text)
            continue

        if role in ("tool", "function"):
            call_id = message.get("tool_call_id") or message.get("call_id") or message.get("name")
            if call_id:
                responses_payload["input"].append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": _message_text(content),
                })
            continue

        if role == "assistant":
            assistant_content = _chat_content_to_responses_content(content, role="assistant")
            if assistant_content:
                responses_payload["input"].append({
                    "role": "assistant",
                    "content": assistant_content,
                })
            for tool_call in message.get("tool_calls", []) or []:
                responses_item = _chat_tool_call_to_responses_item(tool_call)
                if responses_item:
                    responses_payload["input"].append(responses_item)
            continue

        if role not in ("user",):
            role = "user"

        responses_content = _chat_content_to_responses_content(content, role=role)

        responses_payload["input"].append({
            "role": role,
            "content": responses_content,
        })

    if instructions:
        responses_payload["instructions"] = "\n\n".join(instructions)

    field_map = {
        "temperature": "temperature",
        "top_p": "top_p",
        "parallel_tool_calls": "parallel_tool_calls",
        "reasoning": "reasoning",
    }
    for source, target in field_map.items():
        if source in payload:
            responses_payload[target] = payload[source]

    if payload.get("max_completion_tokens") is not None:
        responses_payload["max_output_tokens"] = payload["max_completion_tokens"]
    elif payload.get("max_tokens") is not None:
        responses_payload["max_output_tokens"] = payload["max_tokens"]

    if payload.get("stop") is not None:
        responses_payload["stop"] = payload["stop"]

    if payload.get("tools") is not None:
        responses_payload["tools"] = _chat_tools_to_responses_tools(payload["tools"])

    if payload.get("tool_choice") is not None:
        responses_payload["tool_choice"] = _chat_tool_choice_to_responses_tool_choice(payload["tool_choice"])

    return responses_payload


def _chat_tools_to_responses_tools(tools) -> list:
    if not isinstance(tools, list):
        return []

    translated_tools = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue

        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            function = tool["function"]
            if function.get("name"):
                translated_tools.append({
                    "type": "function",
                    "name": function.get("name"),
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters", {}),
                })
        else:
            translated_tools.append(tool)

    return translated_tools


def _chat_tool_choice_to_responses_tool_choice(tool_choice):
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        function = tool_choice.get("function")
        if isinstance(function, dict) and function.get("name"):
            return {"type": "function", "name": function["name"]}
    return tool_choice


def _extract_responses_text(response_body: Dict) -> str:
    output_text = response_body.get("output_text")
    if isinstance(output_text, str):
        return output_text

    parts = []
    for item in response_body.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for block in item.get("content", []) or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") in ("output_text", "text"):
                parts.append(block.get("text", ""))
    return "".join(parts)


def _responses_usage_to_chat_usage(usage: Dict) -> Dict:
    input_tokens = usage.get("input_tokens", 0) if isinstance(usage, dict) else 0
    output_tokens = usage.get("output_tokens", 0) if isinstance(usage, dict) else 0
    result = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": usage.get("total_tokens", input_tokens + output_tokens) if isinstance(usage, dict) else input_tokens + output_tokens,
    }
    if isinstance(usage, dict) and usage.get("input_tokens_details"):
        result["prompt_tokens_details"] = usage["input_tokens_details"]
    return result


def _responses_to_chat_completion(response_body: Dict, original_model: str) -> Dict:
    usage = response_body.get("usage", {}) if isinstance(response_body, dict) else {}
    content = _extract_responses_text(response_body) if isinstance(response_body, dict) else ""
    tool_calls = _extract_responses_tool_calls(response_body) if isinstance(response_body, dict) else []
    message = {
        "role": "assistant",
        "content": content if content or not tool_calls else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": response_body.get("id", str(uuid.uuid4())) if isinstance(response_body, dict) else str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": original_model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if tool_calls else "stop",
        }],
        "usage": _responses_usage_to_chat_usage(usage),
    }


def _extract_responses_tool_calls(response_body: Dict) -> list:
    tool_calls = []
    for item in response_body.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        tool_calls.append({
            "id": item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}",
            "type": "function",
            "function": {
                "name": item.get("name", ""),
                "arguments": item.get("arguments", ""),
            },
        })
    return tool_calls


def _chat_completion_chunk(chunk_id: str, model: str, delta: Dict = None,
                           finish_reason: str = None, usage: Dict = None) -> Dict:
    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta or {},
            "finish_reason": finish_reason,
        }],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def stream_chat_completions_via_responses(response: requests.Response, request_id: str,
                                          request_body: str, request_size: int, start_time: float,
                                          original_model: str, translated_model: str, chat_payload: Dict,
                                          responses_payload: Dict, original_request_body: Dict = None,
                                          request_headers: Dict = None, client_ip: str = None,
                                          user_id: str = "anonymous") -> Response:
    """Translate a streaming Responses API result back to Chat Completions SSE."""
    cache.start_request(request_id, {
        "request_headers": request_headers,
        "client_ip": client_ip,
        "original_request_body": original_request_body,
        "request_body": chat_payload,
        "model": original_model,
        "translated_model": translated_model if translated_model != original_model else None,
        "endpoint": "/v1/chat/completions",
        "request_size": request_size,
        "user_id": user_id,
    })

    def generate() -> Generator[str, None, None]:
        response_chunks = []
        accumulated_text = []
        response_data = {}
        total_input_tokens = 0
        total_output_tokens = 0
        total_cache_read_input_tokens = 0
        status_code = response.status_code
        error_occurred = False
        first_chunk_received = False
        chunk_id = f"chatcmpl-{request_id}"
        tool_index_by_output_index = {}
        saw_tool_call = False

        try:
            cache.update_request_state(request_id, cache.STATE_SENDING)

            if not response.ok:
                error_occurred = True
                error_text = response.text
                if error_text:
                    try:
                        response_data = response.json()
                    except Exception:
                        response_data = {"error": error_text}
            else:
                role_chunk = _chat_completion_chunk(chunk_id, original_model, delta={"role": "assistant"})
                response_chunks.append(role_chunk)
                yield f"data: {json.dumps(role_chunk)}\n\n"

                for line in response.iter_lines():
                    if not line:
                        continue

                    line = line.decode("utf-8")
                    if not line.startswith("data: "):
                        continue

                    data = line[6:].strip()
                    if data == "[DONE]":
                        continue

                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    if not first_chunk_received:
                        first_chunk_received = True
                        cache.update_request_state(request_id, cache.STATE_RECEIVING)

                    event_type = event.get("type", "")
                    if event_type == "response.output_text.delta":
                        text_delta = event.get("delta", "")
                        if text_delta:
                            accumulated_text.append(text_delta)
                            chunk = _chat_completion_chunk(chunk_id, original_model, delta={"content": text_delta})
                            response_chunks.append(chunk)
                            yield f"data: {json.dumps(chunk)}\n\n"
                    elif event_type == "response.output_item.added":
                        item = event.get("item", {}) or {}
                        if item.get("type") == "function_call":
                            output_index = event.get("output_index", len(tool_index_by_output_index))
                            tool_index = len(tool_index_by_output_index)
                            tool_index_by_output_index[output_index] = tool_index
                            saw_tool_call = True
                            chunk = _chat_completion_chunk(
                                chunk_id,
                                original_model,
                                delta={
                                    "tool_calls": [{
                                        "index": tool_index,
                                        "id": item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}",
                                        "type": "function",
                                        "function": {
                                            "name": item.get("name", ""),
                                            "arguments": "",
                                        },
                                    }]
                                },
                            )
                            response_chunks.append(chunk)
                            yield f"data: {json.dumps(chunk)}\n\n"
                    elif event_type == "response.function_call_arguments.delta":
                        output_index = event.get("output_index")
                        if output_index in tool_index_by_output_index:
                            chunk = _chat_completion_chunk(
                                chunk_id,
                                original_model,
                                delta={
                                    "tool_calls": [{
                                        "index": tool_index_by_output_index[output_index],
                                        "function": {
                                            "arguments": event.get("delta", ""),
                                        },
                                    }]
                                },
                            )
                            response_chunks.append(chunk)
                            yield f"data: {json.dumps(chunk)}\n\n"
                    elif event_type == "response.completed":
                        response_data = event.get("response", {}) or {}
                        usage = response_data.get("usage", {})
                        chat_usage = _responses_usage_to_chat_usage(usage)
                        total_input_tokens = chat_usage.get("prompt_tokens", 0)
                        total_output_tokens = chat_usage.get("completion_tokens", 0)
                        total_cache_read_input_tokens = chat_usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                        finish_chunk = _chat_completion_chunk(
                            chunk_id,
                            original_model,
                            delta={},
                            finish_reason="tool_calls" if saw_tool_call else "stop",
                            usage=chat_usage,
                        )
                        response_chunks.append(finish_chunk)
                        yield f"data: {json.dumps(finish_chunk)}\n\n"
                    elif event_type == "response.failed":
                        error_occurred = True
                        status_code = 500
                        response_data = event.get("response", {}) or event
                        error_payload = response_data.get("error") if isinstance(response_data, dict) else None
                        if not error_payload:
                            error_payload = {"message": "Responses API stream failed"}
                        yield f"data: {json.dumps({'error': error_payload})}\n\n"
                        break

                if not error_occurred:
                    yield "data: [DONE]\n\n"

        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            error_occurred = True
            status_code = 504
            response_data = {"error": f"Upstream connection error: {type(e).__name__}"}
            print(f"[Stream Chat via Responses] Upstream timeout/connection error for request {request_id}: {type(e).__name__}")
        except GeneratorExit:
            cache.update_request_state(request_id, cache.STATE_ERROR, status_code=499)
            return
        except Exception as e:
            error_occurred = True
            status_code = 500
            response_data = {"error": str(e)}
            print(f"[Stream Chat via Responses] Error for request {request_id}: {type(e).__name__}: {e}")
            try:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            except GeneratorExit:
                cache.update_request_state(request_id, cache.STATE_ERROR, status_code=499)
                return

        duration = round(time.time() - start_time, 2)
        if not response_data and accumulated_text:
            response_data = _responses_to_chat_completion({
                "id": chunk_id,
                "output_text": "".join(accumulated_text),
                "usage": {
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                },
            }, original_model)
        elif response_data and not error_occurred and response_data.get("object") != "chat.completion":
            response_data = _responses_to_chat_completion(response_data, original_model)
        elif error_occurred and not response_data:
            response_data = {"error": "Stream interrupted"}

        cache.complete_request(request_id, {
            "request_body": chat_payload,
            "response_body": response_data,
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
            "user_id": user_id,
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


def chat_completions_via_responses(payload: Dict, headers: Dict, request_id: str,
                                   request_body: str, request_size: int, start_time: float,
                                   original_model: str, translated_model: str,
                                   original_request_body: Dict = None,
                                   request_headers: Dict = None,
                                   client_ip: str = None,
                                   user_id: str = "anonymous") -> Response:
    """Handle Chat Completions compatibility through Copilot's Responses API."""
    responses_payload = _chat_completions_to_responses_payload(payload)
    if payload.get("stream"):
        responses_payload["stream"] = True

    _filter_responses_web_search_tools(responses_payload, translated_model, request_id)

    connection_retries = state.max_connection_retries
    last_connection_error = None
    for conn_attempt in range(connection_retries + 1):
        try:
            response = requests.post(
                f"{get_copilot_base_url()}/v1/responses",
                headers=headers,
                json=responses_payload,
                stream=bool(payload.get("stream")),
                timeout=1200,
            )
            last_connection_error = None
            break
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            last_connection_error = e
            log_connection_retry(request_id, "/v1/chat/completions (responses compat)", conn_attempt, connection_retries, e)
            ensure_copilot_token()
            if conn_attempt < connection_retries:
                print(f"[OpenAI Compat] Connection error (attempt {conn_attempt + 1}/{connection_retries + 1}) for request {request_id}: {type(e).__name__}: {e}")
                time.sleep(min(2 ** conn_attempt, 8))
                continue
            print(f"[OpenAI Compat] Connection error (final attempt) for request {request_id}: {type(e).__name__}: {e}")

    if last_connection_error is not None:
        return jsonify({"error": f"Upstream connection error after {connection_retries + 1} attempts: {type(last_connection_error).__name__}"}), 504

    if payload.get("stream"):
        if not response.ok:
            duration = round(time.time() - start_time, 2)
            response_size = len(response.text)
            log_error_request("/v1/chat/completions", responses_payload, response.text, response.status_code, client_ip)
            try:
                result = response.json()
            except Exception:
                result = response.text
            cache.add_request(request_id, {
                "request_headers": request_headers,
                "client_ip": client_ip,
                "original_request_body": original_request_body,
                "request_body": payload,
                "response_body": result,
                "model": original_model,
                "translated_model": translated_model if translated_model != original_model else None,
                "endpoint": "/v1/chat/completions",
                "status_code": response.status_code,
                "request_size": request_size,
                "response_size": response_size,
                "input_tokens": 0,
                "output_tokens": 0,
                "duration": duration,
                "user_id": user_id,
            })
            return Response(response.text, status=response.status_code, mimetype="application/json")

        return stream_chat_completions_via_responses(
            response, request_id, request_body, request_size, start_time,
            original_model, translated_model, payload, responses_payload,
            original_request_body, request_headers, client_ip, user_id,
        )

    duration = round(time.time() - start_time, 2)
    response_size = len(response.text)
    if response.ok:
        responses_body = response.json()
        result = _responses_to_chat_completion(responses_body, original_model)
        usage = result.get("usage", {})
        cache.add_request(request_id, {
            "request_headers": request_headers,
            "client_ip": client_ip,
            "original_request_body": original_request_body,
            "request_body": payload,
            "response_body": result,
            "model": original_model,
            "translated_model": translated_model if translated_model != original_model else None,
            "endpoint": "/v1/chat/completions",
            "status_code": response.status_code,
            "request_size": request_size,
            "response_size": len(json.dumps(result)),
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_read_input_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
            "duration": duration,
            "user_id": user_id,
        })
        return jsonify(result)

    log_error_request("/v1/chat/completions", responses_payload, response.text, response.status_code, client_ip)
    try:
        result = response.json()
    except Exception:
        result = response.text
    cache.add_request(request_id, {
        "request_headers": request_headers,
        "client_ip": client_ip,
        "original_request_body": original_request_body,
        "request_body": payload,
        "response_body": result,
        "model": original_model,
        "translated_model": translated_model if translated_model != original_model else None,
        "endpoint": "/v1/chat/completions",
        "status_code": response.status_code,
        "request_size": request_size,
        "response_size": response_size,
        "input_tokens": 0,
        "output_tokens": 0,
        "duration": duration,
        "user_id": user_id,
    })
    return Response(response.text, status=response.status_code, mimetype="application/json")


def _filter_responses_web_search_tools(payload: Dict, model_id: str, request_id: str) -> None:
    """Remove unsupported Responses tools before forwarding to Copilot."""
    is_gpt_model = _is_gpt_model(model_id)

    tools = payload.get("tools")
    if not isinstance(tools, list):
        return

    filtered_tools = []
    removed_counts = {}
    for tool in tools:
        tool_type = tool.get("type") if isinstance(tool, dict) else None
        if tool_type == "image_generation" or (tool_type == "web_search" and not is_gpt_model):
            removed_counts[tool_type] = removed_counts.get(tool_type, 0) + 1
            continue
        filtered_tools.append(tool)

    if removed_counts:
        payload["tools"] = filtered_tools
        removed_tools = ", ".join(f"{count} '{tool_type}'" for tool_type, count in sorted(removed_counts.items()))
        print(f"Removed unsupported tool(s) {removed_tools} from payload for request {request_id}")


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
        original_request_body = copy.deepcopy(payload)
        request_id = str(uuid.uuid4())

        # Capture incoming request headers (auth values redacted before caching).
        request_headers = redact_auth_headers(dict(request.headers))
        client_ip = get_client_ip(request)
        user_id = _current_user_id()

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

        if _should_route_chat_completions_via_responses(translated_model):
            return chat_completions_via_responses(
                payload, headers, request_id, request_body, request_size, start_time,
                original_model, translated_model, original_request_body, request_headers,
                client_ip=client_ip, user_id=user_id,
            )

        if payload.get("stream"):
            return stream_chat_completions(payload, headers, request_id, request_body, request_size, start_time,
                                           original_model, translated_model, original_request_body, request_headers,
                                           client_ip=client_ip, user_id=user_id)

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
                    request_size = len(json.dumps(payload))

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
                "client_ip": client_ip,
                "original_request_body": original_request_body,
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
                "user_id": user_id,
            })

            return jsonify(result)
        else:
            log_error_request("/v1/chat/completions", payload, response.text, response.status_code, client_ip)
            try:
                result = response.json()
            except:
                result = response.text
            cache.add_request(request_id, {
                "request_headers": request_headers,
                "client_ip": client_ip,
                "original_request_body": original_request_body,
                "request_body": payload,
                "response_body": result,
                "model": original_model,
                "translated_model": translated_model if translated_model != original_model else None,
                "endpoint": "/v1/chat/completions",
                "status_code": response.status_code,
                "request_size": request_size,
                "response_size": response_size,
                "input_tokens": 0,
                "output_tokens": 0,
                "duration": duration,
                "user_id": user_id,
            })
            return Response(response.text, status=response.status_code, mimetype="application/json")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def stream_chat_completions(payload: Dict, headers: Dict, request_id: str,
                            request_body: str, request_size: int, start_time: float,
                            original_model: str, translated_model: str,
                            original_request_body: Dict = None,
                            request_headers: Dict = None,
                            client_ip: str = None,
                            user_id: str = "anonymous") -> Response:
    """Handle streaming chat completions"""
    # Start tracking request immediately
    cache.start_request(request_id, {
        "request_headers": request_headers,
        "client_ip": client_ip,
        "original_request_body": original_request_body,
        "request_body": payload,
        "model": original_model,
        "translated_model": translated_model if translated_model != original_model else None,
        "endpoint": "/v1/chat/completions",
        "request_size": request_size,
        "user_id": user_id,
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
            "user_id": user_id,
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
        original_request_body = copy.deepcopy(payload)
        request_id = str(uuid.uuid4())

        # Capture incoming request headers (auth values redacted before caching).
        request_headers = redact_auth_headers(dict(request.headers))
        client_ip = get_client_ip(request)
        user_id = _current_user_id()

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

        _filter_responses_web_search_tools(payload, translated_model, request_id)

        # Non-streaming request
        connection_retries = state.max_connection_retries
        last_connection_error = None
        use_streaming = payload.get("stream", False)
        for conn_attempt in range(connection_retries + 1):
            request_size = len(json.dumps(payload))
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
                                        original_model, translated_model, payload, original_request_body, request_headers,
                                        client_ip=client_ip, user_id=user_id)
                if not response.ok:
                    print(f"Received error response for request {request_id}: {response.status_code} - {response.text}")
                    log_error_request("/v1/responses", payload, response.text, response.status_code, client_ip)
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
                "client_ip": client_ip,
                "original_request_body": original_request_body,
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
                "user_id": user_id,
            })

            return jsonify(result)
        else:
            log_error_request("/v1/responses", payload, response.text, response.status_code, client_ip)
            usage = {}
            try:
                result = response.json()
            except:
                result = response.text
            cache.add_request(request_id, {
                "request_headers": request_headers,
                "client_ip": client_ip,
                "original_request_body": original_request_body,
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
                "user_id": user_id,
            })
            return Response(response.text, status=response.status_code, mimetype="application/json")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def stream_responses(response: requests.Response, request_id: str,
                     request_size: int, start_time: float,
                     original_model: str, translated_model: str, payload: dict,
                     original_request_body: Dict = None,
                     request_headers: Dict = None,
                     client_ip: str = None,
                     user_id: str = "anonymous") -> Response:
    """Handle streaming Responses API (passthrough SSE events)"""
    cache.start_request(request_id, {
        "request_headers": request_headers,
        "client_ip": client_ip,
        "original_request_body": original_request_body,
        "request_body": payload,
        "model": original_model,
        "translated_model": translated_model if translated_model != original_model else None,
        "endpoint": "/v1/responses",
        "request_size": request_size,
        "user_id": user_id,
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
            "user_id": user_id,
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
