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

from ..api_helpers import ensure_copilot_token, get_copilot_base_url, get_copilot_headers, supports_responses_api
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

        if payload.get("stream"):
            return stream_chat_completions(payload, headers, request_id, request_body, request_size, start_time,
                                           original_model, translated_model)

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
            cache.add_request(request_id, {
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
                "duration": duration,
            })

            return jsonify(result)
        else:
            log_error_request("/v1/chat/completions", payload, response.text, response.status_code)
            return Response(response.text, status=response.status_code, mimetype="application/json")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def stream_chat_completions(payload: Dict, headers: Dict, request_id: str,
                            request_body: str, request_size: int, start_time: float,
                            original_model: str, translated_model: str) -> Response:
    """Handle streaming chat completions"""
    # Start tracking request immediately
    cache.start_request(request_id, {
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
                if tools[i]['type'] == 'web_search':
                    tools.pop(i)
                    print(f"Removed unsupported tool 'web_search' from payload for request {request_id}")
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
                                        original_model, translated_model, payload)
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

            return jsonify(result)
        else:
            log_error_request("/v1/responses", payload, response.text, response.status_code)
            usage = {}
            try:
                result = response.json()
            except:
                result = response.text
            cache.add_request(request_id, {
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
                     original_model: str, translated_model: str, payload: dict) -> Response:
    """Handle streaming Responses API (passthrough SSE events)"""
    cache.start_request(request_id, {
        "request_body": payload,
        "model": original_model,
        "translated_model": translated_model if translated_model != original_model else None,
        "endpoint": "/v1/responses",
        "request_size": request_size,
    })

    def generate() -> Generator[str, None, None]:
        total_input_tokens = 0
        total_output_tokens = 0
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
