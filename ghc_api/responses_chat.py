"""Serve the Responses protocol using a native Chat Completions backend."""

import copy
import json
import time
import uuid


class ResponsesChatRequestError(ValueError):
    def __init__(self, message, param):
        super().__init__(message)
        self.param = param


def _arguments(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _chat_content(content):
    if isinstance(content, str) or content is None:
        return content
    if not isinstance(content, list):
        raise ResponsesChatRequestError("Message content must be a string or array.", "input")
    parts = []
    for part in content:
        if not isinstance(part, dict):
            raise ResponsesChatRequestError("Each content part must be an object.", "input")
        kind = part.get("type")
        if kind in ("input_text", "output_text", "text"):
            parts.append({"type": "text", "text": part.get("text", "")})
        elif kind == "input_image":
            if not isinstance(part.get("image_url"), str) or not part["image_url"]:
                raise ResponsesChatRequestError("Gemini image inputs require an 'image_url'.", "input")
            image = {"url": part["image_url"]}
            if part.get("detail"):
                image["detail"] = part["detail"]
            parts.append({"type": "image_url", "image_url": image})
        elif kind == "refusal":
            parts.append({"type": "text", "text": part.get("refusal", "")})
        else:
            raise ResponsesChatRequestError(
                f"Content type '{kind}' is not supported by the Gemini Responses bridge.", "input",
            )
    return parts


def responses_to_chat_payload(payload):
    """Translate stateless Responses messages, vision and function tool turns."""
    for field in ("previous_response_id", "conversation", "background"):
        if payload.get(field):
            raise ResponsesChatRequestError(
                f"The Gemini Responses bridge does not support '{field}'; "
                "send the complete conversation in 'input'.", field,
            )

    messages = []
    if payload.get("instructions"):
        messages.append({"role": "system", "content": payload["instructions"]})
    inputs = payload.get("input", [])
    if isinstance(inputs, str):
        inputs = [{"role": "user", "content": inputs}]
    if not isinstance(inputs, list):
        raise ResponsesChatRequestError("The 'input' field must be a string or array.", "input")

    for item in inputs:
        if not isinstance(item, dict):
            raise ResponsesChatRequestError("Each input item must be an object.", "input")
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role", "user")
            if role not in ("user", "assistant", "system", "developer"):
                raise ResponsesChatRequestError(f"Unsupported message role '{role}'.", "input")
            messages.append({
                "role": "system" if role == "developer" else role,
                "content": _chat_content(item.get("content", "")),
            })
        elif kind == "function_call":
            if not item.get("call_id") or not item.get("name"):
                raise ResponsesChatRequestError("Function calls require 'call_id' and 'name'.", "input")
            call = {
                "id": item["call_id"],
                "type": "function",
                "function": {"name": item.get("name", ""), "arguments": _arguments(item.get("arguments", ""))},
            }
            if messages and messages[-1].get("role") == "assistant":
                messages[-1].setdefault("tool_calls", []).append(call)
            else:
                messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
        elif kind == "function_call_output":
            if not item.get("call_id"):
                raise ResponsesChatRequestError("Function call outputs require 'call_id'.", "input")
            output = item.get("output", "")
            if isinstance(output, list):
                output = _chat_content(output)
            messages.append({"role": "tool", "tool_call_id": item.get("call_id"), "content": output})
        elif kind == "reasoning":
            # Responses reasoning items are opaque backend state. The visible
            # assistant messages and tool calls carry the conversation here.
            continue
        else:
            raise ResponsesChatRequestError(
                f"Input item type '{kind}' is not supported by the Gemini Responses bridge.", "input",
            )

    result = {"model": payload["model"], "messages": messages, "stream": bool(payload.get("stream"))}
    for field in ("temperature", "top_p", "parallel_tool_calls", "user", "stop"):
        if field in payload:
            result[field] = payload[field]
    if payload.get("max_output_tokens") is not None:
        result["max_tokens"] = payload["max_output_tokens"]
    if isinstance(payload.get("reasoning"), dict) and payload["reasoning"].get("effort"):
        result["reasoning_effort"] = payload["reasoning"]["effort"]
    text_options = payload.get("text") or {}
    if not isinstance(text_options, dict):
        raise ResponsesChatRequestError("The 'text' field must be an object.", "text")
    text_format = text_options.get("format")
    if text_format:
        if not isinstance(text_format, dict):
            raise ResponsesChatRequestError("The 'text.format' field must be an object.", "text.format")
        if text_format.get("type") == "json_schema":
            result["response_format"] = {
                "type": "json_schema",
                "json_schema": {key: value for key, value in text_format.items() if key != "type"},
            }
        else:
            result["response_format"] = text_format
    if "tools" in payload:
        if payload["tools"] is not None and not isinstance(payload["tools"], list):
            raise ResponsesChatRequestError("The 'tools' field must be an array.", "tools")
        tools = []
        for tool in payload["tools"] or []:
            if not isinstance(tool, dict):
                raise ResponsesChatRequestError("Each tool must be an object.", "tools")
            if tool.get("type") != "function":
                raise ResponsesChatRequestError(
                    f"Tool type '{tool.get('type')}' is not supported by the Gemini Responses bridge; "
                    "use function tools.", "tools",
                )
            if not tool.get("name"):
                raise ResponsesChatRequestError("Function tools require a 'name'.", "tools")
            tools.append({
                "type": "function",
                "function": {key: tool[key] for key in ("name", "description", "parameters", "strict") if key in tool},
            })
        result["tools"] = tools
    if "tool_choice" in payload:
        choice = payload["tool_choice"]
        if isinstance(choice, dict):
            if choice.get("type") != "function" or not choice.get("name"):
                raise ResponsesChatRequestError("Only function tool choices are supported.", "tool_choice")
            choice = {"type": "function", "function": {"name": choice["name"]}}
        result["tool_choice"] = choice
    if result["stream"]:
        result["stream_options"] = {"include_usage": True}
    return result


def _usage(usage):
    usage = usage or {}
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)},
        "total_tokens": usage.get("total_tokens", input_tokens + output_tokens),
    }


def _response(payload):
    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "model": payload["model"],
        "status": "in_progress",
        "error": None,
        "incomplete_details": None,
        "output": [],
        "usage": None,
        "instructions": payload.get("instructions"),
        "parallel_tool_calls": payload.get("parallel_tool_calls", True),
        "tools": payload.get("tools", []),
        "tool_choice": payload.get("tool_choice", "auto"),
        "metadata": payload.get("metadata") or {},
        "store": False,
    }


def _finish(result, finish_reason, usage):
    result["status"] = "completed"
    if finish_reason in ("length", "content_filter"):
        result["status"] = "incomplete"
        result["incomplete_details"] = {
            "reason": "max_output_tokens" if finish_reason == "length" else "content_filter",
        }
    result["usage"] = _usage(usage)


def chat_completion_to_response(completion, payload):
    result = _response(payload)
    choices = completion.get("choices") or []
    if not choices:
        raise ValueError("Upstream Chat Completions response has no choices")
    choice = choices[0]
    message = choice.get("message") or {}
    content = []
    if message.get("content"):
        content.append({"type": "output_text", "text": message["content"], "annotations": []})
    if message.get("refusal"):
        content.append({"type": "refusal", "refusal": message["refusal"]})
    if content:
        result["output"].append({
            "id": f"msg_{uuid.uuid4().hex}", "type": "message", "role": "assistant",
            "status": "completed", "content": content,
        })
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        result["output"].append({
            "id": f"fc_{uuid.uuid4().hex}", "type": "function_call", "status": "completed",
            "call_id": call.get("id") or f"call_{uuid.uuid4().hex}",
            "name": function.get("name", ""), "arguments": function.get("arguments", ""),
        })
    _finish(result, choice.get("finish_reason"), completion.get("usage"))
    if result["status"] == "incomplete":
        for item in result["output"]:
            item["status"] = "incomplete"
    return result


class ChatCompletionResponsesAdapter:
    """Present a Chat Completions response to the existing Responses pipeline.

    HTTP errors retain their status and body. Successful SSE bodies are
    translated lazily so keepalives, cancellation and retries remain owned by
    the route's existing stream handler.
    """

    def __init__(self, response, payload):
        self._response = response
        self._payload = payload
        self._result = None
        self.status_code = response.status_code
        self.ok = response.ok

    def json(self):
        if not self.ok:
            return self._response.json()
        if self._result is None:
            self._result = chat_completion_to_response(self._response.json(), self._payload)
        return self._result

    @property
    def text(self):
        return json.dumps(self.json(), ensure_ascii=False) if self.ok else self._response.text

    def close(self):
        self._response.close()

    def iter_lines(self):
        translator = ChatToResponsesStream(self._payload)
        try:
            for event in translator.start():
                yield from self._event_lines(event)
            for line in self._response.iter_lines():
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    error = chunk["error"]
                    if not isinstance(error, dict):
                        error = {"message": str(error)}
                    yield from self._event_lines(translator.fail(error))
                    return
                for event in translator.feed(chunk):
                    yield from self._event_lines(event)
            for event in translator.finish():
                yield from self._event_lines(event)
        except (ValueError, OSError) as exc:
            yield from self._event_lines(translator.fail({"code": "upstream_error", "message": str(exc)}))

    @staticmethod
    def _event_lines(event):
        yield f"event: {event['type']}".encode("utf-8")
        yield ("data: " + json.dumps(event, ensure_ascii=False)).encode("utf-8")
        yield b""


class ChatToResponsesStream:
    def __init__(self, payload):
        self.response = _response(payload)
        self.sequence = 0
        self.message = None
        self.message_index = None
        self.parts = {}
        self.calls = {}
        self.usage = None
        self.finish_reason = None

    def event(self, kind, **fields):
        event = {"type": kind, "sequence_number": self.sequence, **fields}
        self.sequence += 1
        return event

    def start(self):
        yield self.event("response.created", response=copy.deepcopy(self.response))
        yield self.event("response.in_progress", response=copy.deepcopy(self.response))

    def feed(self, chunk):
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta") or {}
            for field in ("content", "refusal"):
                value = delta.get(field)
                if not value:
                    continue
                if self.message is None:
                    self.message_index = len(self.response["output"])
                    self.message = {
                        "id": f"msg_{uuid.uuid4().hex}", "type": "message", "role": "assistant",
                        "status": "in_progress", "content": [],
                    }
                    self.response["output"].append(self.message)
                    yield self.event("response.output_item.added", output_index=self.message_index, item=self.message)
                if field not in self.parts:
                    part = {"type": "output_text", "text": "", "annotations": []} if field == "content" else {"type": "refusal", "refusal": ""}
                    self.parts[field] = len(self.message["content"])
                    self.message["content"].append(part)
                    yield self.event(
                        "response.content_part.added", item_id=self.message["id"],
                        output_index=self.message_index, content_index=self.parts[field], part=part,
                    )
                part = self.message["content"][self.parts[field]]
                part["text" if field == "content" else "refusal"] += value
                yield self.event(
                    "response.output_text.delta" if field == "content" else "response.refusal.delta",
                    item_id=self.message["id"], output_index=self.message_index,
                    content_index=self.parts[field], delta=value,
                )

            for call in delta.get("tool_calls") or []:
                call_index = call.get("index", 0)
                function = call.get("function") or {}
                if call_index not in self.calls:
                    output_index = len(self.response["output"])
                    item = {
                        "id": f"fc_{uuid.uuid4().hex}", "type": "function_call", "status": "in_progress",
                        "call_id": call.get("id") or f"call_{uuid.uuid4().hex}",
                        "name": function.get("name", ""), "arguments": "",
                    }
                    self.calls[call_index] = output_index
                    self.response["output"].append(item)
                    yield self.event("response.output_item.added", output_index=output_index, item=item)
                else:
                    output_index = self.calls[call_index]
                    item = self.response["output"][output_index]
                    if function.get("name"):
                        item["name"] += function["name"]
                arguments = function.get("arguments", "")
                if arguments:
                    item["arguments"] += arguments
                    yield self.event(
                        "response.function_call_arguments.delta", item_id=item["id"],
                        output_index=output_index, delta=arguments,
                    )
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]

    def fail(self, error):
        self.response["status"] = "failed"
        self.response["error"] = {
            "code": error.get("code") or error.get("type") or "upstream_error",
            "message": error.get("message") or "Upstream Chat Completions stream failed.",
        }
        return self.event("response.failed", response=self.response)

    def finish(self):
        if not self.finish_reason:
            yield self.fail({"code": "upstream_error", "message": "Upstream stream ended without a finish reason."})
            return
        for index, item in enumerate(self.response["output"]):
            item["status"] = "incomplete" if self.finish_reason in ("length", "content_filter") else "completed"
            if item["type"] == "message":
                for content_index, part in enumerate(item["content"]):
                    is_text = part["type"] == "output_text"
                    yield self.event(
                        "response.output_text.done" if is_text else "response.refusal.done",
                        item_id=item["id"], output_index=index, content_index=content_index,
                        **({"text": part["text"]} if is_text else {"refusal": part["refusal"]}),
                    )
                    yield self.event("response.content_part.done", item_id=item["id"], output_index=index, content_index=content_index, part=part)
            else:
                yield self.event(
                    "response.function_call_arguments.done", item_id=item["id"],
                    output_index=index, arguments=item["arguments"],
                )
            yield self.event("response.output_item.done", output_index=index, item=item)
        _finish(self.response, self.finish_reason, self.usage)
        yield self.event(
            "response.incomplete" if self.response["status"] == "incomplete" else "response.completed",
            response=self.response,
        )
