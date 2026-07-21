import json

import requests

r = requests.post(
    "http://localhost:8313/v1/messages",
    headers={"anthropic-version": "2023-06-01"},
    json={
        "model": "gpt-5.6-luna",
        # "model": "gpt-5.6-sol",
        # "model": "claude-opus-4.8", 400: {"error":{"message":"The use of the web search tool is not supported.","code":"unsupported_value"}}
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": "搜索 Python 官网，告诉我当前最新版本。",
            }
        ],
        "tools": [
            {
                "type": "web_search_20250305",
                "name": "web_search",
            }
        ],
        "tool_choice": {
            "type": "tool",
            "name": "web_search",
        },
    },
    timeout=180,
)

print(r.status_code)
try:
    print(json.dumps(r.json(), ensure_ascii=False, indent=2))
except ValueError:
    print(r.text)
