"""Opt-in live checks against a running server; uses synthetic prompts only."""

import argparse
import json
import urllib.request


def generate(base, model, contents, streaming, cloud, tools=None, calling=None,
             system=None, max_output_tokens=4096):
    payload = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
            "thinkingConfig": {"thinkingLevel": "HIGH"},
        },
    }
    if tools:
        payload["tools"] = tools
    if calling:
        payload["toolConfig"] = {"functionCallingConfig": calling}
    if system is not None:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    action = "streamGenerateContent" if streaming else "generateContent"
    if cloud:
        url = base + "/v1internal:" + action
        placeholder = {"gemini-4-argon": "MODEL_PLACEHOLDER_M460",
                       "gemini-3.8-flash": "MODEL_PLACEHOLDER_M461"}[model]
        payload = {"model": placeholder, "request": payload}
    else:
        url = base + "/v1beta/models/" + model + ":" + action
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        if not streaming:
            result = json.load(response)
            if "error" in result:
                raise RuntimeError(result["error"])
            candidate = result.get("response", result)["candidates"][0]
            if candidate.get("finishReason") == "MAX_TOKENS":
                raise RuntimeError(f"Model exhausted maxOutputTokens={max_output_tokens}")
            return candidate["content"]
        parts = []
        finished = False
        for line in response:
            if not line.startswith(b"data:"):
                continue
            result = json.loads(line[5:])
            if "error" in result:
                raise RuntimeError(result["error"])
            result = result.get("response", result)
            for candidate in result.get("candidates", []):
                if candidate.get("finishReason") == "MAX_TOKENS":
                    raise RuntimeError(f"Model exhausted maxOutputTokens={max_output_tokens}")
                parts.extend(candidate.get("content", {}).get("parts", []))
                finished = finished or "finishReason" in candidate
        if not finished:
            raise RuntimeError("Stream did not finish normally")
        return {"role": "model", "parts": parts}


def visible_text(content):
    return "".join(part.get("text", "") for part in content["parts"] if not part.get("thought")).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gemini-4-argon", "gemini-3.8-flash"], required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8314")
    args = parser.parse_args()
    tools = [{"functionDeclarations": [{
        "name": "add_numbers",
        "description": "Add the supplied integers.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"a": {"type": "INTEGER"}, "b": {"type": "INTEGER"}},
            "required": ["a", "b"],
        },
    }]}]
    for cloud in (False, True):
        for streaming in (False, True):
            mode = ("CloudCode" if cloud else "Gemini") + ("/SSE" if streaming else "/JSON")
            plain = generate(args.base_url, args.model, [{"role": "user", "parts": [{
                "text": "Reply with exactly NATIVE_GEMINI_OK and nothing else.",
            }]}], streaming, cloud)
            if visible_text(plain) != "NATIVE_GEMINI_OK":
                raise RuntimeError(f"{mode}: unexpected plain text response")
            contents = [{"role": "user", "parts": [{
                "text": (
                    "Call add_numbers with a=19, b=23. Once the tool returns, "
                    "reply with exactly TOOL_RESULT_OK_42 and nothing else."
                ),
            }]}]
            first = generate(args.base_url, args.model, contents, streaming, cloud, tools,
                             {"mode": "ANY", "allowedFunctionNames": ["add_numbers"]})
            calls = [part["functionCall"] for part in first["parts"] if "functionCall" in part]
            if len(calls) != 1 or calls[0]["name"] != "add_numbers" or calls[0]["args"] != {"a": 19, "b": 23}:
                raise RuntimeError(f"{mode}: incorrect structured tool call")
            call = calls[0]
            contents.append(first)
            contents.append({"role": "user", "parts": [{"functionResponse": {
                "name": "add_numbers", "id": call["id"], "response": {"result": 42},
            }}]})
            second = generate(args.base_url, args.model, contents, streaming, cloud, tools)
            if visible_text(second) != "TOOL_RESULT_OK_42":
                raise RuntimeError(f"{mode}: incorrect tool continuation")
            print(f"PASS {args.model} {mode}: text, structured tool call, tool-result continuation", flush=True)


if __name__ == "__main__":
    main()
