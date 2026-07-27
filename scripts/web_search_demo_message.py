"""Demonstrate web search through the local Messages API endpoint."""

import json

import requests


def main() -> None:
    response = requests.post(
        "http://localhost:8313/v1/messages",
        headers={"anthropic-version": "2023-06-01"},
        json={
            "model": "gpt-5.6-luna",
            # "model": "gpt-5.6-sol",
            # Web search is not supported by claude-opus-4.8.
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

    print(response.status_code)
    try:
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    except ValueError:
        print(response.text)


if __name__ == "__main__":
    main()
