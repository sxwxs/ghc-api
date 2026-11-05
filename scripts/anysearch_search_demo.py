#!/usr/bin/env python3
"""Call all three native AnySearch APIs without handling the provider key.

The historical filename and positional search usage remain compatible.
Examples:
    python scripts/anysearch_search_demo.py 'Python asyncio TaskGroup docs'
    python scripts/anysearch_search_demo.py --domain code --domain finance
    python scripts/anysearch_search_demo.py 'Go context docs' --tag code.doc --params '{"library":"golang"}'
    python scripts/anysearch_search_demo.py --extract https://example.com/article

Enable AnySearch in the server config first. If ghc-api user authentication is
on, set GHC_API_TOKEN to your ghc-api user token (not the AnySearch API key).
"""

import argparse
import json
import os

import requests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--extract", metavar="URL", help="Extract a public HTTP(S) page")
    mode.add_argument("--domain", action="append", help="Look up a search capability domain (repeatable; not DNS)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8313")
    parser.add_argument("--max-results", type=int, choices=range(1, 11), default=5)
    parser.add_argument("--tag")
    parser.add_argument("--params", help="JSON object of parameters defined for the selected tag")
    parser.add_argument("--zone", choices=("cn", "intl"))
    parser.add_argument("--language")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args()
    if args.query and (args.extract or args.domain):
        parser.error("Choose either a search query, --extract, or --domain")
    if not (args.query or args.extract or args.domain):
        parser.error("Provide a search query, --extract URL, or --domain DOMAIN")
    if (args.extract or args.domain) and any((args.tag, args.params, args.zone, args.language)):
        parser.error("--tag, --params, --zone and --language apply only to Search")
    headers = {} if args.domain else {"Content-Type": "application/json"}
    if args.domain:
        endpoint, timeout = "/v1/sub-domains", 30
        sender = requests.get
        request_args = {"params": [("domain", domain) for domain in args.domain]}
    elif args.extract:
        endpoint, timeout = "/v1/extract", 60
        sender = requests.post
        request_args = {"json": {"url": args.extract}}
    else:
        endpoint, timeout = "/v1/search", 30
        sender = requests.post
        payload = {"query": args.query, "max_results": args.max_results, "format": args.format}
        for name in ("tag", "zone", "language"):
            if getattr(args, name):
                payload[name] = getattr(args, name)
        if args.params:
            try:
                params = json.loads(args.params)
            except ValueError:
                parser.error("--params must be a JSON object")
            if not isinstance(params, dict):
                parser.error("--params must be a JSON object")
            payload["params"] = params
        request_args = {"json": payload}
    token = os.environ.get("GHC_API_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    try:
        with sender(args.base_url.rstrip("/") + endpoint, **request_args,
                    headers=headers, timeout=timeout, allow_redirects=False) as response:
            if not 200 <= response.status_code < 300:
                # Never dump error bodies: a misconfigured endpoint might
                # return credentials in a 402 response rather than proxy errors.
                print(f"AnySearch {endpoint} failed: HTTP {response.status_code}")
                return 1
            body = response.json()
            if not isinstance(body, dict) or type(body.get("code")) is not int or body["code"] != 0:
                print("AnySearch returned an unsuccessful response")
                return 1
            print(json.dumps(body, ensure_ascii=False, indent=2))
    except (requests.RequestException, ValueError):
        print("AnySearch failed: network error or invalid JSON")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
