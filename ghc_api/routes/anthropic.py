"""
Anthropic-compatible API routes
"""

import json
import time
import uuid
from typing import Dict, Generator

import requests
from flask import Blueprint, Response, jsonify, request, stream_with_context

from ..api_helpers import ensure_copilot_token, get_copilot_base_url, get_copilot_headers
from ..cache import cache
from ..streaming import AnthropicStreamState, reconstruct_openai_response_from_chunks, translate_chunk_to_anthropic_events
from ..translator import translate_anthropic_to_openai, translate_model_name, translate_openai_to_anthropic
from ..utils import log_error_request, is_orphaned_tool_result_error, remove_orphaned_tool_results, extract_orphaned_tool_use_ids, log_tool_result_cleanup

anthropic_bp = Blueprint('anthropic', __name__)


@anthropic_bp.route("/v1/messages", methods=["POST"])
def anthropic_messages():
    """Handle Anthropic messages API"""
    if True:
        start_time = time.time()
        ensure_copilot_token()
        anthropic_payload = request.get_json()
        request_id = str(uuid.uuid4())

        # Get the original and translated model names
        original_model = anthropic_payload.get("model", "unknown")
        translated_model = translate_model_name(original_model)

        # Check for vision content
        enable_vision = any(
            isinstance(msg.get("content"), list) and
            any(p.get("type") == "image" for p in msg.get("content", []))
            for msg in anthropic_payload.get("messages", [])
        )

        is_agent_call = any(
            msg.get("role") in ("assistant", "tool")
            for msg in openai_payload.get("messages", [])
        )

        headers = get_copilot_headers(enable_vision)
        headers["X-Initiator"] = "agent" if is_agent_call else "user"

        request_size = len(json.dumps(anthropic_payload))

        max_retries = 3
        current_payload = anthropic_payload
        duration = round(time.time() - start_time, 2)
        cleanup_log_entry = None
        for attempt in range(max_retries + 1):
            # Translate to OpenAI format
            openai_payload = translate_anthropic_to_openai(current_payload)
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

                        # Start building log entry on first cleanup attempt
                        if cleanup_log_entry is None:
                            cleanup_log_entry = {
                                "request_id": request_id,
                                "original_request": anthropic_payload,
                                "error_response": response.text,
                                "error_status_code": response.status_code,
                                "orphaned_ids": orphaned_ids,
                            }
                        else:
                            # Append additional orphaned IDs if multiple retries needed
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
                cleanup_log_entry["final_response"] = anthropic_response
                log_tool_result_cleanup(cleanup_log_entry)
            return Response(response.text, status=response.status_code, mimetype="application/json")

    # except Exception as e:
    #     return jsonify({"error": {"type": "api_error", "message": str(e)}}), 500


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
