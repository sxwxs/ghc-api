"""Demonstrate web search through the local Responses API endpoint."""

import json

import requests


def main() -> None:
    response = requests.post(
        "http://localhost:8313/v1/responses",
        json={
            "model": "gpt-5.6-sol",
            "input": "搜索 Python 官网，告诉我当前最新版本。",
            "tools": [{"type": "web_search"}],
            "tool_choice": {"type": "web_search"},
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
