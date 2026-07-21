import json

import requests

r = requests.post(
    "http://localhost:8313/v1/responses",
    json={
        "model": "gpt-5.6-sol",
        "input": "搜索 Python 官网，告诉我当前最新版本。",
        "tools": [{"type": "web_search"}],
        "tool_choice": {"type": "web_search"},
    },
    timeout=180,
)

print(r.status_code)
try:
    print(json.dumps(r.json(), ensure_ascii=False, indent=2))
except ValueError:
    print(r.text)

# Messages API

# import requests

# r = requests.post(
#     "http://localhost:8313/v1/messages",
#     headers={"anthropic-version": "2023-06-01"},
#     json={
#         "model": "gpt-5.6-sol",
#         "max_tokens": 512,
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "搜索 Python 官网，告诉我当前最新版本。",
#             }
#         ],
#         "tools": [
#             {
#                 "type": "web_search_20250305",
#                 "name": "web_search",
#             }
#         ],
#         "tool_choice": {
#             "type": "tool",
#             "name": "web_search",
#         },
#     },
#     timeout=180,
# )

# print(r.status_code)
# print(r.text)