"""Gemini GenerateContent translation without a separate proxy process."""

import json
import uuid

from .sse.events import DONE, json_events
from .sse.keepalive import KEEPALIVE


CLOUD_MODELS = {
    "MODEL_PLACEHOLDER_M460": "gemini-4-argon",
    "MODEL_PLACEHOLDER_M461": "gemini-3.8-flash",
}


class IncompleteChatStream(ValueError):
    """The upstream closed without a terminal completion."""


def object_value(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def array_value(value, name):
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def schema_value(value):
    if isinstance(value, list):
        return [schema_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key == "type" and isinstance(item, str):
            result[key] = item.lower()
        else:
            result[key] = schema_value(item)
    return result


def text_parts(parts):
    result = []
    for part in array_value(parts, "parts"):
        part = object_value(part, "part")
        if set(part) != {"text"} or not isinstance(part["text"], str):
            raise ValueError("systemInstruction supports text parts only")
        result.append(part["text"])
    return "\n".join(result)


def to_chat(body, model, streaming):
    body = object_value(body, "request")
    unknown = set(body) - {
        "contents", "systemInstruction", "tools", "toolConfig", "generationConfig",
    }
    if unknown:
        raise ValueError("Unsupported Gemini request fields: " + ", ".join(sorted(unknown)))
    if not isinstance(model, str) or not model:
        raise ValueError("model must be a nonempty string")
    model = CLOUD_MODELS.get(model, model[7:] if model.startswith("models/") else model)
    payload = {"model": model, "messages": [], "stream": streaming}
    messages = payload["messages"]
    if "systemInstruction" in body:
        instruction = object_value(body["systemInstruction"], "systemInstruction")
        messages.append({"role": "system", "content": text_parts(instruction.get("parts"))})
    contents = array_value(body.get("contents"), "contents")
    if not contents:
        raise ValueError("contents must not be empty")
    pending = []
    for content in contents:
        content = object_value(content, "content")
        role = content.get("role", "user")
        if role not in ("user", "model"):
            raise ValueError("content role must be user or model")
        parts = array_value(content.get("parts"), "parts")
        # Antigravity CLI replays tool-only responses with the model role.
        if role == "model" and any(isinstance(part, dict) and "functionResponse" in part for part in parts):
            if not all(isinstance(part, dict) and set(part) == {"functionResponse"} for part in parts):
                raise ValueError("Model-role function responses must contain only functionResponse parts")
        text, calls, results = [], [], []
        reasoning = []
        for part in parts:
            part = object_value(part, "part")
            kinds = set(part) & {"text", "inlineData", "functionCall", "functionResponse"}
            if len(kinds) != 1 or set(part) - kinds - {"thought"}:
                raise ValueError("A part must contain one supported value; native thought signatures are unsupported")
            if "text" in part:
                if not isinstance(part["text"], str):
                    raise ValueError("text must be a string")
                if part.get("thought"):
                    reasoning.append(part["text"])
                else:
                    text.append({"type": "text", "text": part["text"]})
            elif "inlineData" in part:
                image = object_value(part["inlineData"], "inlineData")
                mime, data = image.get("mimeType"), image.get("data")
                if not isinstance(mime, str) or not mime.startswith("image/") or not isinstance(data, str):
                    raise ValueError("inlineData requires image mimeType and base64 data")
                text.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
            elif "functionCall" in part:
                if role != "model":
                    raise ValueError("functionCall requires model role")
                call = object_value(part["functionCall"], "functionCall")
                name = call.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError("functionCall requires a name")
                args = object_value(call.get("args", {}), "functionCall.args")
                call_id = call.get("id") or "call_" + uuid.uuid4().hex
                if not isinstance(call_id, str):
                    raise ValueError("functionCall.id must be a string")
                calls.append({"id": call_id, "type": "function", "function": {
                    "name": name, "arguments": json.dumps(args),
                }})
                pending.append((call_id, name))
            elif "functionResponse" in part:
                response = object_value(part["functionResponse"], "functionResponse")
                matches = [(ident, name) for ident, name in pending
                           if name == response.get("name")
                           and (not response.get("id") or ident == response["id"])]
                if not matches:
                    raise ValueError("functionResponse has no matching functionCall")
                ident, name = matches[0]
                pending.remove((ident, name))
                if "response" not in response:
                    raise ValueError("functionResponse.response is required")
                results.append({"role": "tool", "tool_call_id": ident,
                                "content": json.dumps(response["response"])})
            else:
                raise ValueError("Unsupported Gemini part (supported: text, inlineData, functionCall, functionResponse)")
        if results:
            if calls:
                raise ValueError("Cannot mix function calls and results in one content")
            messages.extend(results)
        if text or calls or reasoning:
            message = {"role": "assistant" if role == "model" else "user",
                       "content": text or None}
            if text and all(part["type"] == "text" for part in text):
                message["content"] = "\n".join(part["text"] for part in text)
            if calls:
                message["tool_calls"] = calls
            if reasoning:
                message["reasoning_content"] = "".join(reasoning)
            messages.append(message)
    if pending:
        raise ValueError("Every historical functionCall must have a functionResponse")
    tools = []
    for group in array_value(body.get("tools", []), "tools"):
        group = object_value(group, "tool")
        if set(group) != {"functionDeclarations"}:
            raise ValueError("Only functionDeclarations tools are supported")
        for declaration in array_value(group["functionDeclarations"], "functionDeclarations"):
            declaration = object_value(declaration, "functionDeclaration")
            name = declaration.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("Function declaration requires a name")
            tools.append({"type": "function", "function": {
                "name": name, "description": declaration.get("description", ""),
                "parameters": schema_value(object_value(
                    declaration.get("parametersJsonSchema", declaration.get("parameters", {"type": "object"})),
                    "function parameters")),
            }})
    if tools:
        payload["tools"] = tools
    if "toolConfig" in body:
        config = object_value(body["toolConfig"], "toolConfig")
        calling = object_value(config.get("functionCallingConfig"), "functionCallingConfig")
        if set(config) != {"functionCallingConfig"} or set(calling) - {"mode", "allowedFunctionNames"}:
            raise ValueError("Unsupported toolConfig")
        mode = calling.get("mode", "AUTO")
        if mode not in ("AUTO", "NONE", "ANY"):
            raise ValueError("Unsupported function calling mode")
        allowed = calling.get("allowedFunctionNames")
        if allowed is not None:
            if mode != "ANY" or not isinstance(allowed, list) or len(allowed) != 1:
                raise ValueError("allowedFunctionNames supports exactly one function with ANY mode")
            if allowed[0] not in [tool["function"]["name"] for tool in tools]:
                raise ValueError("allowedFunctionNames must name a declared function")
            payload["tool_choice"] = {"type": "function", "function": {"name": allowed[0]}}
        else:
            payload["tool_choice"] = {"AUTO": "auto", "NONE": "none", "ANY": "required"}[mode]
    generation = object_value(body.get("generationConfig", {}), "generationConfig")
    keys = {"temperature": "temperature", "topP": "top_p",
            "maxOutputTokens": "max_tokens", "stopSequences": "stop"}
    if set(generation) - set(keys) - {"thinkingConfig", "candidateCount"}:
        raise ValueError("Unsupported generationConfig field")
    if generation.get("candidateCount", 1) != 1:
        raise ValueError("Only candidateCount=1 is supported")
    for source, target in keys.items():
        if source in generation:
            payload[target] = generation[source]
    if "thinkingConfig" in generation:
        thinking = object_value(generation["thinkingConfig"], "thinkingConfig")
        if set(thinking) - {"thinkingLevel", "includeThoughts"}:
            raise ValueError("Only thinkingConfig.thinkingLevel and includeThoughts are supported")
        if "includeThoughts" in thinking and not isinstance(thinking["includeThoughts"], bool):
            raise ValueError("includeThoughts must be a boolean")
        if "thinkingLevel" in thinking:
            level = thinking["thinkingLevel"]
            if not isinstance(level, str) or level.lower() not in ("low", "medium", "high"):
                raise ValueError("thinkingLevel must be LOW, MEDIUM or HIGH")
            payload["reasoning_effort"] = level.lower()
    return payload


def candidate(message, finish=None, include_thoughts=True):
    parts = []
    if include_thoughts and message.get("reasoning_content"):
        parts.append({"text": message["reasoning_content"], "thought": True})
    if message.get("content"):
        parts.append({"text": message["content"]})
    for call in message.get("tool_calls", []):
        function = call["function"]
        args = object_value(json.loads(function.get("arguments") or "{}"), "tool arguments")
        parts.append({"functionCall": {
            "id": call["id"], "name": function["name"], "args": args,
        }})
    result = {"index": 0, "content": {"role": "model", "parts": parts}}
    if finish:
        result["finishReason"] = {"stop": "STOP", "tool_calls": "STOP",
                                  "length": "MAX_TOKENS", "content_filter": "SAFETY"}.get(finish, "OTHER")
    return result


def usage_metadata(usage):
    return {"promptTokenCount": usage.get("prompt_tokens", 0),
            "candidatesTokenCount": usage.get("completion_tokens", 0),
            "totalTokenCount": usage.get("total_tokens", 0)}


def from_chat(body, include_thoughts=True):
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Upstream response has no choices")
    choice = choices[0]
    result = {"candidates": [candidate(choice["message"], choice.get("finish_reason"), include_thoughts)]}
    if "usage" in body:
        result["usageMetadata"] = usage_metadata(body["usage"])
    return result


def stream_events(chunks):
    for event in json_events(chunks):
        if event is DONE:
            return
        yield event


def from_chat_stream(chunks, include_thoughts=True):
    calls = {}
    finished = False
    for event in stream_events(chunks):
        if event is KEEPALIVE:
            yield KEEPALIVE
            continue
        if "error" in event:
            yield {"error": event["error"]}
            return
        if event.get("choices"):
            choice = event["choices"][0]
            delta = choice.get("delta", {})
            message = {key: delta[key] for key in ("content", "reasoning_content") if key in delta}
            for call in delta.get("tool_calls", []):
                stored = calls.setdefault(call["index"], {"id": "", "function": {"name": "", "arguments": ""}})
                if call.get("id"):
                    stored["id"] = call["id"]
                for key in ("name", "arguments"):
                    stored["function"][key] += call.get("function", {}).get(key, "")
            finish = choice.get("finish_reason")
            finished = finished or bool(finish)
            if finish and calls:
                message["tool_calls"] = list(calls.values())
                calls.clear()
            converted = candidate(message, finish, include_thoughts)
            if converted["content"]["parts"] or finish:
                yield {"candidates": [converted]}
        if event.get("usage"):
            yield {"usageMetadata": usage_metadata(event["usage"])}
    if calls:
        raise ValueError("Upstream stream ended before tool calls completed")
    if not finished:
        raise IncompleteChatStream("Upstream stream ended without a finish reason")
