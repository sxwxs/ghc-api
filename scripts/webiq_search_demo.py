"""Demonstrate Microsoft Web IQ grounding through the local Responses API.

Web IQ credentials are read by ghc-api from config.yaml; this client never
receives or sends the Web IQ API key.
"""

import argparse
import json

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Web IQ search through ghc-api")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8313",
        help="Local ghc-api base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default="gemini-3.1-pro-preview",
        help="Model to ground with Web IQ (default: %(default)s)",
    )
    parser.add_argument(
        "--query",
        default="搜索 Python 官网，告诉我当前最新稳定版本，并附上来源链接。",
        help="Question to ask",
    )
    args = parser.parse_args()

    response = requests.post(
        f"{args.base_url.rstrip('/')}/v1/responses",
        json={
            "model": args.model,
            "input": args.query,
            # This proxy-only option makes ghc-api call Microsoft Web IQ,
            # inject the returned passages into instructions, and then call
            # the selected model. It works with non-GPT models as well.
            "webiq_search_options": {"enabled": True},
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
