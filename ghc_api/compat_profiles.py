"""Claude Code and Responses protocol compatibility auditing.

The converter deliberately keeps this module transport-free.  Callers pass
the request headers and decoded JSON body and receive a serialisable profile
plus warnings.  Warnings describe *shape*, never request values: prose, tool
input, JSON schema contents, metadata, and identities must not escape through
logs or cache metadata.

This is a drift detector, not a JSON-schema replacement.  It recognises the
Claude Code profiles observed in the supplied 2.1.197 and 2.1.207 captures and
checks the protocol surfaces whose meaning affects Messages/Responses
translation.  Dynamic tool schemas, tool inputs, and metadata are opaque.
Built-in tool contracts can instead be compared as canonical hashes by
supplying a baseline manifest.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


MODE_COMPATIBILITY = "compatibility"
MODE_LOSSLESS_REQUIRED = "lossless_required"
VALID_MODES = frozenset((MODE_COMPATIBILITY, MODE_LOSSLESS_REQUIRED))

KNOWN_CLAUDE_CLI_VERSIONS = frozenset(("2.1.197", "2.1.207"))

# SHA-256 fingerprints of all 29 built-in tool contracts observed in each
# supplied capture (ten changed between the versions). Hashes let the runtime
# detect same-version contract drift without embedding proprietary prompt or
# description text in this repository. Environment-provided MCP tools are
# deliberately excluded because their contracts are deployment-specific.
CLAUDE_CLI_TOOL_CONTRACT_BASELINES: Dict[str, Dict[str, str]] = {
    "2.1.197": {
        "Agent": "sha256:d5d6ada52ef3877257ea27100d6fd994ae6c41fdf3ecf7e13d4a7c6957534710",
        "AskUserQuestion": "sha256:98e072997d5f09ae87dd342d8bce9b7325b172e17827afbdb05ab38a01ae3940",
        "Bash": "sha256:4f8cb5b937d235ce4e4a768ce896cf65cb98d79738d0c2ae7efbc2725b246da4",
        "CronCreate": "sha256:e36ab387b6eb860b52b38a0b14b52e8866832fdda16aac9f3fcd13afb1a25724",
        "CronDelete": "sha256:3df063cab8649491ef59b364e5784408dc59b7fca809b72f47ba8adea9cb8968",
        "CronList": "sha256:f8434e2ceaa7dde9676700dcf662b3d8ef87aeea929795ab2909905b6f3fbafe",
        "Edit": "sha256:2920bee1258ac22d0b2cddaea52c19a594bba03d2b2c037f39915f5300e1ea80",
        "EnterPlanMode": "sha256:ad7d62247fa400bb0669ab9feb140cafb402238fb48da95609a3f570ce85d873",
        "EnterWorktree": "sha256:d8f96f9d45a63f246cd796005b581984a9094f8ab901ca355dcc964ae91f9680",
        "ExitPlanMode": "sha256:7477bddd0188ce850c0c03663a93084581a07b4ab3c3362e2cecd208d0aac6e5",
        "ExitWorktree": "sha256:5031a9c58e9d456457cd5cda99a51792a066a53e442e4fe8f5459f7a95b97714",
        "Glob": "sha256:a6fe4e2131d38c61c799b54e7946acaad071253f3528e19ec2125f858920584e",
        "Grep": "sha256:048981839c7659acaa16599fdbb721ad9458741cd5090acb8413ffab8ff37cf1",
        "NotebookEdit": "sha256:cbcd42978626cbaf82821f221b4c20515910cc433dbd301917df9111dabe77b9",
        "Read": "sha256:76184d62cf690bf77a82f9059888174122780462372a9a5d86ed8db07f653753",
        "ReportFindings": "sha256:d91b6a737503fab4668472f3bb80ecaa8a2426386056c090bf119067a5682337",
        "ScheduleWakeup": "sha256:5083ddcea825c6cd5bbc64717225188f7bb2908016b8fc4dd46bfecc09378d29",
        "SendMessage": "sha256:20b9e4adadcaa5d65bb2e03f1c69326ab39cba6316dd8b78f2a1534515b28d12",
        "Skill": "sha256:97b97a90f84ef9a7eb770189f7874de1699afce25fff5923e767bbcd00766778",
        "TaskCreate": "sha256:33749aa1759968fd3e3d70419e7a5c71d46f8dd78b7cab3f73b33ec3662826d3",
        "TaskGet": "sha256:48fc8f8b9d065eba687d0e4334db180d60744e4fba9614d447b8bc1c3e2493a6",
        "TaskList": "sha256:315dd2f40c1cdcd0665a9d397385db9bc61a8e47136bb1c96e89c54f6283348b",
        "TaskOutput": "sha256:e777b0ea933cb0a1193966b2b6ddd1341f879494500902fa28bcd1aa9976bf72",
        "TaskStop": "sha256:43f6f7973947413996cee8cfd8bfe05ce983cd2075feb5d1b210ad924cf35251",
        "TaskUpdate": "sha256:767f19e1b6bca1adb0a661fb6bdf8f3d2927321075a0fb63801a379784f5fbbc",
        "WebFetch": "sha256:aeb103a4c73c58945eedb4776e7b19d5b1e6ca86dbad12377cfd51bead4b17ff",
        "WebSearch": "sha256:7d645172684106a9418dfac03516ea7aea726098dff5a5fc233c29f8f2444af8",
        "Workflow": "sha256:e348757fdb4c661ac5cb746ca1960bfc789d15354eb85eb2301abd21b9766148",
        "Write": "sha256:6784fbdfbfc0bbd086fbed843f220a6c375c08dfb332ae60ffa7a82ac84bacf0",
    },
    "2.1.207": {
        "Agent": "sha256:886a03fdda7274c5fabc49813f8ad70bcff5eef1cb958e67119d44aa5a6fe306",
        "AskUserQuestion": "sha256:98e072997d5f09ae87dd342d8bce9b7325b172e17827afbdb05ab38a01ae3940",
        "Bash": "sha256:c5e28c6f6184da2058de0acc1f107820079816360c60a58619d666de21df807b",
        "CronCreate": "sha256:e36ab387b6eb860b52b38a0b14b52e8866832fdda16aac9f3fcd13afb1a25724",
        "CronDelete": "sha256:3df063cab8649491ef59b364e5784408dc59b7fca809b72f47ba8adea9cb8968",
        "CronList": "sha256:f8434e2ceaa7dde9676700dcf662b3d8ef87aeea929795ab2909905b6f3fbafe",
        "Edit": "sha256:2920bee1258ac22d0b2cddaea52c19a594bba03d2b2c037f39915f5300e1ea80",
        "EnterPlanMode": "sha256:c618d2f0661cc1cc35324a9ba9e790a4e3a8ecf0c3540c52799024cea27da500",
        "EnterWorktree": "sha256:cf6fb03e82b8c1df4841fe5412ca389bcd6c758cdb734fad9a80e4be7f711193",
        "ExitPlanMode": "sha256:eaa7b3517bda947ea2f8d18c2ca963d429f662e5ebd75f2bfc858424389e3feb",
        "ExitWorktree": "sha256:5031a9c58e9d456457cd5cda99a51792a066a53e442e4fe8f5459f7a95b97714",
        "Glob": "sha256:a6fe4e2131d38c61c799b54e7946acaad071253f3528e19ec2125f858920584e",
        "Grep": "sha256:048981839c7659acaa16599fdbb721ad9458741cd5090acb8413ffab8ff37cf1",
        "NotebookEdit": "sha256:cbcd42978626cbaf82821f221b4c20515910cc433dbd301917df9111dabe77b9",
        "Read": "sha256:76184d62cf690bf77a82f9059888174122780462372a9a5d86ed8db07f653753",
        "ReportFindings": "sha256:2260b4e8347ae897204e5e54ef49050f982a245ea81c635ebd9a026cc550a8ad",
        "ScheduleWakeup": "sha256:ead6624e70f3e00d6f1d4a68da2f66133cc164ee074c9e49851959eaa500a6e4",
        "SendMessage": "sha256:25918b5c2aaa9d85088bf8940b2c76045d3dd381d6241a994b5e65ff6091ddd3",
        "Skill": "sha256:97b97a90f84ef9a7eb770189f7874de1699afce25fff5923e767bbcd00766778",
        "TaskCreate": "sha256:33749aa1759968fd3e3d70419e7a5c71d46f8dd78b7cab3f73b33ec3662826d3",
        "TaskGet": "sha256:48fc8f8b9d065eba687d0e4334db180d60744e4fba9614d447b8bc1c3e2493a6",
        "TaskList": "sha256:315dd2f40c1cdcd0665a9d397385db9bc61a8e47136bb1c96e89c54f6283348b",
        "TaskOutput": "sha256:e777b0ea933cb0a1193966b2b6ddd1341f879494500902fa28bcd1aa9976bf72",
        "TaskStop": "sha256:2ce9cf8b4f24ac9e4054ce8a3e3841a91392aa466fe596daab88408ca7bbbeb3",
        "TaskUpdate": "sha256:767f19e1b6bca1adb0a661fb6bdf8f3d2927321075a0fb63801a379784f5fbbc",
        "WebFetch": "sha256:aeb103a4c73c58945eedb4776e7b19d5b1e6ca86dbad12377cfd51bead4b17ff",
        "WebSearch": "sha256:7d645172684106a9418dfac03516ea7aea726098dff5a5fc233c29f8f2444af8",
        "Workflow": "sha256:468eea10ef36adfacac20fd96438f5ed6bf56e26342ad83e7f3f95e4f0f2c95f",
        "Write": "sha256:6784fbdfbfc0bbd086fbed843f220a6c375c08dfb332ae60ffa7a82ac84bacf0",
    },
}
KNOWN_ANTHROPIC_VERSION = "2023-06-01"
KNOWN_ANTHROPIC_BETAS = frozenset(
    (
        "claude-code-20250219",
        "context-1m-2025-08-07",
        "context-management-2025-06-27",
        "interleaved-thinking-2025-05-14",
        "prompt-caching-scope-2026-01-05",
        "redact-thinking-2026-02-12",
        "thinking-token-count-2026-05-13",
    )
)
# The 1M-context flag is conditional in both captured CLI versions.  Treat the
# two observed normalised sets as profiles rather than requiring every known
# token on every request.
_BASE_ANTHROPIC_BETAS = KNOWN_ANTHROPIC_BETAS - {"context-1m-2025-08-07"}
KNOWN_ANTHROPIC_BETA_SETS = frozenset(
    (tuple(sorted(_BASE_ANTHROPIC_BETAS)), tuple(sorted(KNOWN_ANTHROPIC_BETAS)))
)

# These are the request keys understood by the compatibility converter.  The
# two captured Claude Code versions use a subset; keeping converter-supported
# Anthropic fields here avoids reporting legitimate API clients as drift.
_TOP_LEVEL_TYPES: Dict[str, Tuple[str, ...]] = {
    "model": ("string",),
    "messages": ("array",),
    "max_tokens": ("integer",),
    "system": ("string", "array"),
    "metadata": ("object",),
    "stop_sequences": ("array",),
    "stream": ("boolean",),
    "temperature": ("number",),
    "top_p": ("number",),
    "top_k": ("integer",),
    "tools": ("array",),
    "tool_choice": ("object", "string"),
    "thinking": ("object",),
    "context_management": ("object",),
    "output_config": ("object",),
    "service_tier": ("string",),
}

_TOP_LEVEL_REQUIRED_FIELDS: Dict[str, Tuple[str, ...]] = {
    "model": ("string",),
    "messages": ("array",),
    "max_tokens": ("integer",),
}

_CONTENT_BLOCK_FIELDS: Dict[str, Set[str]] = {
    "text": {"type", "text", "cache_control", "metadata"},
    "image": {"type", "source", "cache_control", "metadata"},
    "document": {
        "type", "source", "cache_control", "title", "context", "citations", "metadata"
    },
    "tool_use": {"type", "id", "name", "input", "cache_control", "metadata"},
    "tool_result": {
        "type", "tool_use_id", "content", "is_error", "cache_control", "metadata"
    },
    "thinking": {"type", "thinking", "signature", "metadata"},
    "redacted_thinking": {"type", "data", "metadata"},
}

_CONTENT_BLOCK_REQUIRED_FIELDS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "text": {"text": ("string",)},
    "image": {"source": ("object",)},
    "document": {"source": ("object",)},
    "tool_use": {
        "id": ("string",),
        "name": ("string",),
        "input": ("object",),
    },
    "tool_result": {"tool_use_id": ("string",)},
    "thinking": {
        "thinking": ("string",),
        "signature": ("string",),
    },
    "redacted_thinking": {"data": ("string",)},
}

_SOURCE_REQUIRED_FIELDS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "base64": {
        "media_type": ("string",),
        "data": ("string",),
    },
    "url": {"url": ("string",)},
    # Anthropic text documents observed in the captures use ``data``.  The
    # converter also understands ``text`` as a compatibility alias, handled
    # explicitly by the source auditor below.
    "text": {"data": ("string",)},
}

_TOOL_FIELDS = {
    "name", "description", "input_schema", "type", "cache_control",
    "defer_loading", "allowed_callers", "metadata",
}

# Event names accepted by the Responses SSE state machine.  Includes the
# public lifecycle events and the Copilot Responses-Lite keepalive observed in
# the capture.
KNOWN_RESPONSES_EVENT_TYPES = frozenset(
    (
        "response.created",
        "response.queued",
        "response.in_progress",
        "response.output_item.added",
        "response.output_item.done",
        "response.content_part.added",
        "response.content_part.done",
        "response.output_text.delta",
        "response.output_text.done",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.custom_tool_call_input.delta",
        "response.custom_tool_call_input.done",
        "response.refusal.delta",
        "response.refusal.done",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_part.done",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
        "response.reasoning_text.delta",
        "response.reasoning_text.done",
        "response.completed",
        "response.incomplete",
        "response.failed",
        "error",
        "keepalive",
    )
)

# The union covers terminal output items and the Responses-Lite input items in
# the supplied GPT-5.6 dump.  Conversion support is still decided separately;
# this set answers only whether an item discriminator is known to the profile.
KNOWN_RESPONSES_ITEM_TYPES = frozenset(
    (
        "additional_tools",
        "agent_message",
        "custom_tool_call",
        "custom_tool_call_output",
        "function_call",
        "function_call_output",
        "message",
        "reasoning",
    )
)

_RESPONSES_EVENT_FIELDS: Dict[str, Set[str]] = {
    "keepalive": {"type", "sequence_number"},
    "response.created": {"type", "sequence_number", "response"},
    "response.queued": {"type", "sequence_number", "response"},
    "response.in_progress": {"type", "sequence_number", "response"},
    "response.output_item.added": {"type", "sequence_number", "output_index", "item"},
    "response.output_item.done": {"type", "sequence_number", "output_index", "item"},
    "response.content_part.added": {
        "type", "sequence_number", "output_index", "content_index", "item_id", "part"
    },
    "response.content_part.done": {
        "type", "sequence_number", "output_index", "content_index", "item_id", "part"
    },
    "response.output_text.delta": {
        "type", "sequence_number", "output_index", "content_index", "item_id",
        "delta", "logprobs", "obfuscation",
    },
    "response.output_text.done": {
        "type", "sequence_number", "output_index", "content_index", "item_id",
        "text", "logprobs",
    },
    "response.function_call_arguments.delta": {
        "type", "sequence_number", "output_index", "item_id", "delta", "obfuscation"
    },
    "response.function_call_arguments.done": {
        "type", "sequence_number", "output_index", "item_id", "arguments"
    },
    "response.custom_tool_call_input.delta": {
        "type", "sequence_number", "output_index", "item_id", "delta", "obfuscation"
    },
    "response.custom_tool_call_input.done": {
        "type", "sequence_number", "output_index", "item_id", "input"
    },
    "response.refusal.delta": {
        "type", "sequence_number", "output_index", "content_index", "item_id", "delta"
    },
    "response.refusal.done": {
        "type", "sequence_number", "output_index", "content_index", "item_id", "refusal"
    },
    "response.reasoning_summary_part.added": {
        "type", "sequence_number", "output_index", "summary_index", "item_id", "part"
    },
    "response.reasoning_summary_part.done": {
        "type", "sequence_number", "output_index", "summary_index", "item_id", "part"
    },
    "response.reasoning_summary_text.delta": {
        "type", "sequence_number", "output_index", "summary_index", "item_id", "delta", "obfuscation"
    },
    "response.reasoning_summary_text.done": {
        "type", "sequence_number", "output_index", "summary_index", "item_id", "text"
    },
    "response.reasoning_text.delta": {
        "type", "sequence_number", "output_index", "content_index", "item_id", "delta", "obfuscation"
    },
    "response.reasoning_text.done": {
        "type", "sequence_number", "output_index", "content_index", "item_id", "text"
    },
    "response.completed": {"type", "sequence_number", "response", "copilot_usage"},
    "response.incomplete": {"type", "sequence_number", "response", "copilot_usage"},
    "response.failed": {"type", "sequence_number", "response", "copilot_usage"},
    "error": {"type", "sequence_number", "code", "message", "param", "error"},
}

_RESPONSES_ITEM_FIELDS: Dict[str, Set[str]] = {
    "additional_tools": {"type", "role", "tools"},
    "agent_message": {"type", "author", "content", "recipient"},
    "custom_tool_call": {"type", "id", "call_id", "name", "input", "status", "namespace"},
    "custom_tool_call_output": {"type", "id", "call_id", "output", "status"},
    "function_call": {"type", "id", "call_id", "name", "arguments", "namespace", "status"},
    "function_call_output": {"type", "id", "call_id", "output", "status"},
    "message": {"type", "id", "role", "content", "phase", "status"},
    "reasoning": {"type", "id", "summary", "content", "encrypted_content", "status"},
}

_RESPONSES_CONTENT_PART_FIELDS: Dict[str, Set[str]] = {
    "input_text": {"type", "text"},
    "output_text": {"type", "text", "annotations", "logprobs"},
    "refusal": {"type", "refusal"},
    "summary_text": {"type", "text"},
    "encrypted_content": {"type", "encrypted_content"},
}

_RESPONSES_RESPONSE_FIELDS = {
    "id", "object", "created_at", "completed_at", "status", "background",
    "error", "incomplete_details", "instructions", "max_output_tokens",
    "max_tool_calls", "model", "output", "parallel_tool_calls",
    "previous_response_id", "prompt_cache_key", "prompt_cache_retention",
    "reasoning", "safety_identifier", "service_tier", "store", "temperature",
    "text", "tool_choice", "tools", "top_logprobs", "top_p", "truncation",
    "usage", "user", "metadata", "frequency_penalty", "presence_penalty",
    "moderation", "tool_usage", "client_metadata", "billing",
}

_RESPONSES_EVENT_FIELD_TYPES: Dict[str, Tuple[str, ...]] = {
    "sequence_number": ("integer",),
    "output_index": ("integer",),
    "content_index": ("integer",),
    "summary_index": ("integer",),
    "item_id": ("string",),
    "delta": ("string",),
    "text": ("string",),
    "arguments": ("string",),
    "input": ("string",),
    "obfuscation": ("string",),
    "refusal": ("string",),
    "logprobs": ("array",),
    "item": ("object",),
    "part": ("object",),
    "response": ("object",),
}

# Minimum fields needed to interpret each event without guessing lifecycle
# identity or payload placement.  Copilot-only optional fields (for example
# ``copilot_usage`` and delta ``obfuscation``) are intentionally not required.
_RESPONSES_EVENT_REQUIRED_FIELDS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "keepalive": {"sequence_number": ("integer",)},
    "response.created": {
        "sequence_number": ("integer",),
        "response": ("object",),
    },
    "response.queued": {
        "sequence_number": ("integer",),
        "response": ("object",),
    },
    "response.in_progress": {
        "sequence_number": ("integer",),
        "response": ("object",),
    },
    "response.output_item.added": {
        "sequence_number": ("integer",),
        "output_index": ("integer",),
        "item": ("object",),
    },
    "response.output_item.done": {
        "sequence_number": ("integer",),
        "output_index": ("integer",),
        "item": ("object",),
    },
    "response.content_part.added": {
        "sequence_number": ("integer",),
        "output_index": ("integer",),
        "content_index": ("integer",),
        "item_id": ("string",),
        "part": ("object",),
    },
    "response.content_part.done": {
        "sequence_number": ("integer",),
        "output_index": ("integer",),
        "content_index": ("integer",),
        "item_id": ("string",),
        "part": ("object",),
    },
    "response.output_text.delta": {
        "sequence_number": ("integer",),
        "output_index": ("integer",),
        "content_index": ("integer",),
        "item_id": ("string",),
        "delta": ("string",),
        "logprobs": ("array",),
    },
    "response.output_text.done": {
        "sequence_number": ("integer",),
        "output_index": ("integer",),
        "content_index": ("integer",),
        "item_id": ("string",),
        "text": ("string",),
        "logprobs": ("array",),
    },
    "response.function_call_arguments.delta": {
        "sequence_number": ("integer",),
        "output_index": ("integer",),
        "item_id": ("string",),
        "delta": ("string",),
    },
    "response.function_call_arguments.done": {
        "sequence_number": ("integer",),
        "output_index": ("integer",),
        "item_id": ("string",),
        "arguments": ("string",),
    },
    "response.custom_tool_call_input.delta": {
        "sequence_number": ("integer",),
        "output_index": ("integer",),
        "item_id": ("string",),
        "delta": ("string",),
    },
    "response.custom_tool_call_input.done": {
        "sequence_number": ("integer",),
        "output_index": ("integer",),
        "item_id": ("string",),
        "input": ("string",),
    },
    "response.refusal.delta": {
        "sequence_number": ("integer",),
        "output_index": ("integer",),
        "content_index": ("integer",),
        "item_id": ("string",),
        "delta": ("string",),
    },
    "response.refusal.done": {
        "sequence_number": ("integer",),
        "output_index": ("integer",),
        "content_index": ("integer",),
        "item_id": ("string",),
        "refusal": ("string",),
    },
    "response.reasoning_summary_part.added": {
        "sequence_number": ("integer",),
        "output_index": ("integer",),
        "summary_index": ("integer",),
        "item_id": ("string",),
        "part": ("object",),
    },
    "response.reasoning_summary_part.done": {
        "sequence_number": ("integer",),
        "output_index": ("integer",),
        "summary_index": ("integer",),
        "item_id": ("string",),
        "part": ("object",),
    },
    "response.reasoning_summary_text.delta": {
        "sequence_number": ("integer",),
        "output_index": ("integer",),
        "summary_index": ("integer",),
        "item_id": ("string",),
        "delta": ("string",),
    },
    "response.reasoning_summary_text.done": {
        "sequence_number": ("integer",),
        "output_index": ("integer",),
        "summary_index": ("integer",),
        "item_id": ("string",),
        "text": ("string",),
    },
    "response.reasoning_text.delta": {
        "sequence_number": ("integer",),
        "output_index": ("integer",),
        "content_index": ("integer",),
        "item_id": ("string",),
        "delta": ("string",),
    },
    "response.reasoning_text.done": {
        "sequence_number": ("integer",),
        "output_index": ("integer",),
        "content_index": ("integer",),
        "item_id": ("string",),
        "text": ("string",),
    },
    "response.completed": {
        "sequence_number": ("integer",),
        "response": ("object",),
    },
    "response.incomplete": {
        "sequence_number": ("integer",),
        "response": ("object",),
    },
    "response.failed": {
        "sequence_number": ("integer",),
        "response": ("object",),
    },
    "error": {
        "sequence_number": ("integer",),
        "code": ("string",),
        "message": ("string",),
        "param": ("string", "null"),
    },
}

_RESPONSES_ITEM_REQUIRED_FIELDS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "additional_tools": {
        "role": ("string",),
        "tools": ("array",),
    },
    "agent_message": {
        "author": ("string",),
        "content": ("array",),
        "recipient": ("string",),
    },
    "custom_tool_call": {
        "call_id": ("string",),
        "name": ("string",),
        "input": ("string",),
    },
    "custom_tool_call_output": {
        "call_id": ("string",),
        "output": ("string", "array"),
    },
    "function_call": {
        "call_id": ("string",),
        "name": ("string",),
        "arguments": ("string",),
    },
    "function_call_output": {
        "call_id": ("string",),
        "output": ("string", "array"),
    },
    "message": {
        "role": ("string",),
        "content": ("array",),
    },
    "reasoning": {
        "summary": ("array",),
        "encrypted_content": ("string",),
    },
}

_RESPONSES_CONTENT_REQUIRED_FIELDS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "input_text": {"text": ("string",)},
    "output_text": {"text": ("string",)},
    "refusal": {"refusal": ("string",)},
    "summary_text": {"text": ("string",)},
    "encrypted_content": {"encrypted_content": ("string",)},
}

_CLI_VERSION_RE = re.compile(
    r"(?:^|[\s;(])claude-cli/([0-9]+(?:\.[0-9]+){2}(?:[-+][^\s;)]+)?)",
    re.IGNORECASE,
)
_SAFE_WARNING_PATH_COMPONENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    encoded = _canonical_json(value).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _json_type(value: Any) -> str:
    if value is _MISSING:
        return "missing"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return type(value).__name__


def _matches_types(value: Any, expected: Sequence[str]) -> bool:
    observed = _json_type(value)
    if observed in expected:
        return True
    # JSON Schema's number includes integers.
    return observed == "integer" and "number" in expected


def _pointer_escape(component: Any) -> str:
    return str(component).replace("~", "~0").replace("/", "~1")


def _join_path(base: str, component: Any) -> str:
    escaped = _pointer_escape(component)
    return f"{base}/{escaped}" if base else f"/{escaped}"


_MISSING = object()


@dataclass(frozen=True)
class CompatibilityProfile:
    """The non-sensitive compatibility identity selected for one audit."""

    name: str
    protocol: str
    cli_version: Optional[str]
    known_cli_version: bool
    anthropic_version: Optional[str]
    anthropic_betas: Tuple[str, ...]
    fingerprint: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "protocol": self.protocol,
            "cli_version": self.cli_version,
            "known_cli_version": self.known_cli_version,
            "anthropic_version": self.anthropic_version,
            "anthropic_betas": list(self.anthropic_betas),
            "fingerprint": self.fingerprint,
        }


@dataclass
class CompatibilityAudit:
    """Structured result returned to routes, logs, and request-cache code."""

    mode: str
    profile: CompatibilityProfile
    warnings: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def should_fail(self) -> bool:
        return any(warning.get("action") == "reject" for warning in self.warnings)

    @property
    def allowed(self) -> bool:
        return not self.should_fail

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "profile": self.profile.to_dict(),
            "warnings": copy.deepcopy(self.warnings),
            "should_fail": self.should_fail,
            "allowed": self.allowed,
        }


class _WarningCollector:
    """Build value-free warnings and deduplicate them in insertion order."""

    def __init__(self, mode: str, version: Optional[str]) -> None:
        if mode not in VALID_MODES:
            raise ValueError("Unknown compatibility mode: %s" % mode)
        self.mode = mode
        self.version = version or "unknown"
        self.warnings: List[Dict[str, Any]] = []
        self._seen: Set[str] = set()

    def add(
        self,
        code: str,
        path: str,
        observed_type: str,
        expected_types: Sequence[str],
        *,
        evidence: Any = None,
        fail_in_lossless: bool = True,
        fail_always: bool = False,
        fingerprint: Optional[str] = None,
    ) -> None:
        action = "reject" if fail_always or (
            fail_in_lossless and self.mode == MODE_LOSSLESS_REQUIRED
        ) else "warn"
        expected = tuple(sorted(set(str(item) for item in expected_types)))
        if fingerprint is None:
            # Evidence is hashed but never copied into the warning.  For type
            # errors callers intentionally omit it, making fingerprints stable
            # across different prose/identity values of the same shape.
            fingerprint = _digest(
                {
                    "code": code,
                    "path": path,
                    "observed": observed_type,
                    "expected": expected,
                    "version": self.version,
                    "evidence": evidence,
                }
            )
        warning: Dict[str, Any] = {
            "code": code,
            "path": path or "/",
            "types": {
                "observed": observed_type,
                "expected": list(expected),
            },
            "observed_type": observed_type,
            "expected_types": list(expected),
            "version": self.version,
            "fingerprint": fingerprint,
            "action": action,
        }
        key = _canonical_json(warning)
        if key not in self._seen:
            self._seen.add(key)
            self.warnings.append(warning)


def _header_value(headers: Any, name: str) -> Any:
    # Werkzeug's EnvironHeaders exposes ``items``/``get`` but intentionally
    # does not register as collections.abc.Mapping.
    if not hasattr(headers, "items") or not callable(getattr(headers, "items", None)):
        return _MISSING
    wanted = name.lower()
    matches: List[Any] = []
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == wanted:
            matches.append(value)
    if not matches:
        return _MISSING
    if len(matches) == 1:
        return matches[0]
    # Duplicate beta header lines have comma-list semantics.  Other duplicate
    # headers use the final value, matching common WSGI header containers.
    if wanted == "anthropic-beta" and all(isinstance(value, str) for value in matches):
        return ",".join(matches)
    return matches[-1]


def normalize_anthropic_betas(value: Any) -> Tuple[str, ...]:
    """Return a deterministic, case-normalised beta-token set."""

    if value is _MISSING or value is None or value == "":
        return ()
    if not isinstance(value, str):
        return ()
    return tuple(sorted({token.strip().lower() for token in value.split(",") if token.strip()}))


def _parse_cli_version(user_agent: Any) -> Optional[str]:
    if not isinstance(user_agent, str):
        return None
    match = _CLI_VERSION_RE.search(user_agent)
    return match.group(1) if match else None


def _make_anthropic_profile(
    cli_version: Optional[str], anthropic_version: Optional[str], betas: Tuple[str, ...]
) -> CompatibilityProfile:
    known = cli_version in KNOWN_CLAUDE_CLI_VERSIONS
    name = "claude_cli_" + cli_version.replace(".", "_") if known and cli_version else "claude_cli_unknown"
    fingerprint = _digest(
        {
            "profile": name,
            "cli_version": cli_version,
            "anthropic_version": anthropic_version,
            "anthropic_betas": betas,
        }
    )
    return CompatibilityProfile(
        name=name,
        protocol="anthropic-messages-2023-06-01",
        cli_version=cli_version,
        known_cli_version=known,
        anthropic_version=anthropic_version,
        anthropic_betas=betas,
        fingerprint=fingerprint,
    )


def _responses_profile() -> CompatibilityProfile:
    name = "openai_responses_v1"
    return CompatibilityProfile(
        name=name,
        protocol="openai-responses-v1",
        cli_version=None,
        known_cli_version=False,
        anthropic_version=None,
        anthropic_betas=(),
        fingerprint=_digest({"profile": name}),
    )


def _check_type(
    value: Any,
    expected: Sequence[str],
    path: str,
    collector: _WarningCollector,
    code: str = "request.invalid_type",
) -> bool:
    if _matches_types(value, expected):
        return True
    collector.add(code, path, _json_type(value), expected)
    return False


def _check_enum(
    value: Any,
    allowed: Iterable[str],
    path: str,
    collector: _WarningCollector,
    code: str = "request.unknown_enum",
) -> bool:
    if not isinstance(value, str):
        return False
    allowed_values = set(allowed)
    if value in allowed_values:
        return True
    collector.add(code, path, "string", ("string",), evidence={"enum": value})
    return False


def _unknown_fields(
    value: Mapping,
    allowed: Set[str],
    path: str,
    collector: _WarningCollector,
    code: str,
) -> None:
    for key in sorted((key for key in value.keys() if isinstance(key, str) and key not in allowed)):
        # JSON object keys can technically contain identities or prose.  Keep
        # conventional protocol identifiers useful in paths and redact any
        # suspicious key; the full key still contributes only to the digest.
        display_key = key if _SAFE_WARNING_PATH_COMPONENT_RE.match(key) else "<redacted-key>"
        child_path = _join_path(path, display_key)
        collector.add(
            code,
            child_path,
            _json_type(value[key]),
            ("absent",),
            evidence={"field": key},
        )
    # A decoded JSON object can only have string keys, but accepting arbitrary
    # Mapping objects makes the public helper safer in unit/embedding use.
    for key in value.keys():
        if not isinstance(key, str):
            collector.add(code, path, "non_string_key", ("string_key",))


def _require_fields(
    value: Mapping,
    required: Mapping[str, Sequence[str]],
    path: str,
    collector: _WarningCollector,
    code: str,
) -> None:
    """Report missing protocol fields without including any field value."""

    for key in sorted(required):
        if key not in value:
            collector.add(
                code,
                _join_path(path, key),
                "missing",
                required[key],
            )


def _audit_cache_control(value: Any, path: str, collector: _WarningCollector) -> None:
    if not _check_type(value, ("object",), path, collector):
        return
    _unknown_fields(value, {"type", "ttl", "scope"}, path, collector, "request.unknown_field")
    if "type" in value:
        if _check_type(value["type"], ("string",), _join_path(path, "type"), collector):
            _check_enum(value["type"], ("ephemeral",), _join_path(path, "type"), collector)
    if "ttl" in value:
        if _check_type(value["ttl"], ("string",), _join_path(path, "ttl"), collector):
            _check_enum(value["ttl"], ("5m", "1h"), _join_path(path, "ttl"), collector)


def _audit_source(value: Any, path: str, block_type: str, collector: _WarningCollector) -> None:
    if not _check_type(value, ("object",), path, collector):
        return
    source_type = value.get("type", _MISSING)
    if not _check_type(source_type, ("string",), _join_path(path, "type"), collector):
        return
    allowed_types = ("base64", "url") if block_type == "image" else ("base64", "url", "text")
    _check_enum(source_type, allowed_types, _join_path(path, "type"), collector, "content_source.unknown_type")
    fields = {
        "base64": {"type", "media_type", "data"},
        "url": {"type", "url"},
        "text": {"type", "data", "text", "media_type"},
    }.get(source_type, {"type"})
    _unknown_fields(value, fields, path, collector, "content_source.unknown_field")
    required = _SOURCE_REQUIRED_FIELDS.get(source_type)
    if required is not None:
        # ``text`` is accepted as a compatibility alias for document text.
        # Require at least one of data/text without disclosing its value.
        if source_type == "text" and "data" not in value and "text" in value:
            required = {}
        _require_fields(
            value,
            required,
            path,
            collector,
            "content_source.missing_required_field",
        )


def _audit_content_block(
    block: Any,
    path: str,
    collector: _WarningCollector,
    *,
    allowed_types: Optional[Set[str]] = None,
) -> None:
    if not _check_type(block, ("object",), path, collector, "content_block.invalid_type"):
        return
    block_type = block.get("type", _MISSING)
    if not isinstance(block_type, str) or block_type not in _CONTENT_BLOCK_FIELDS:
        collector.add(
            "content_block.unknown_type",
            _join_path(path, "type"),
            _json_type(block_type),
            ("string",),
            evidence={"block_type": None if block_type is _MISSING else block_type},
        )
        return
    if allowed_types is not None and block_type not in allowed_types:
        collector.add(
            "content_block.unknown_type",
            _join_path(path, "type"),
            "string",
            ("string",),
            evidence={"block_type": block_type, "context": sorted(allowed_types)},
        )
        return

    _require_fields(
        block,
        _CONTENT_BLOCK_REQUIRED_FIELDS[block_type],
        path,
        collector,
        "content_block.missing_required_field",
    )

    _unknown_fields(
        block,
        _CONTENT_BLOCK_FIELDS[block_type],
        path,
        collector,
        "content_block.unknown_field",
    )

    scalar_types: Dict[str, Tuple[str, ...]] = {
        "text": ("string",),
        "id": ("string",),
        "name": ("string",),
        "tool_use_id": ("string",),
        "is_error": ("boolean",),
        "thinking": ("string",),
        "signature": ("string",),
        "data": ("string",),
    }
    for key, expected in scalar_types.items():
        if key in block:
            _check_type(block[key], expected, _join_path(path, key), collector)
    if "cache_control" in block:
        _audit_cache_control(block["cache_control"], _join_path(path, "cache_control"), collector)

    # These subtrees can contain arbitrary user/tool data and are deliberately
    # opaque.  Only the root type is checked.
    if block_type == "tool_use" and "input" in block:
        _check_type(block["input"], ("object",), _join_path(path, "input"), collector)
    if block_type in ("image", "document"):
        _audit_source(block.get("source", _MISSING), _join_path(path, "source"), block_type, collector)
    if block_type == "tool_result" and "content" in block:
        content = block["content"]
        content_path = _join_path(path, "content")
        if _check_type(content, ("string", "array"), content_path, collector):
            if isinstance(content, list):
                for index, child in enumerate(content):
                    _audit_content_block(
                        child,
                        _join_path(content_path, index),
                        collector,
                        allowed_types={"text", "image", "document"},
                    )


def _audit_messages(value: Any, collector: _WarningCollector) -> None:
    if not isinstance(value, list):
        return
    for index, message in enumerate(value):
        path = f"/messages/{index}"
        if not _check_type(message, ("object",), path, collector):
            continue
        _unknown_fields(message, {"role", "content", "metadata"}, path, collector, "message.unknown_field")
        role = message.get("role", _MISSING)
        if _check_type(role, ("string",), _join_path(path, "role"), collector):
            _check_enum(role, ("user", "assistant"), _join_path(path, "role"), collector)
        content = message.get("content", _MISSING)
        content_path = _join_path(path, "content")
        if not _check_type(content, ("string", "array"), content_path, collector):
            continue
        if isinstance(content, list):
            for block_index, block in enumerate(content):
                _audit_content_block(block, _join_path(content_path, block_index), collector)


def _audit_system(value: Any, collector: _WarningCollector) -> None:
    if not isinstance(value, list):
        return
    for index, block in enumerate(value):
        _audit_content_block(block, f"/system/{index}", collector, allowed_types={"text"})


def _audit_context_management(value: Any, collector: _WarningCollector) -> None:
    if not isinstance(value, Mapping):
        return
    _unknown_fields(value, {"edits"}, "/context_management", collector, "context_management.unknown_field")
    edits = value.get("edits", _MISSING)
    if not _check_type(edits, ("array",), "/context_management/edits", collector):
        return
    for index, edit in enumerate(edits):
        path = f"/context_management/edits/{index}"
        if not _check_type(edit, ("object",), path, collector, "context_edit.invalid_type"):
            continue
        _unknown_fields(edit, {"type", "keep"}, path, collector, "context_edit.unknown_field")
        edit_type = edit.get("type", _MISSING)
        if not isinstance(edit_type, str) or edit_type != "clear_thinking_20251015":
            collector.add(
                "context_edit.unknown_type",
                _join_path(path, "type"),
                _json_type(edit_type),
                ("string",),
                evidence={"edit_type": None if edit_type is _MISSING else edit_type},
            )
        if "keep" in edit:
            if _check_type(edit["keep"], ("string",), _join_path(path, "keep"), collector):
                _check_enum(edit["keep"], ("all",), _join_path(path, "keep"), collector)


def _audit_thinking(value: Any, collector: _WarningCollector) -> None:
    if not isinstance(value, Mapping):
        return
    _unknown_fields(value, {"type", "budget_tokens"}, "/thinking", collector, "thinking.unknown_field")
    thinking_type = value.get("type", _MISSING)
    if _check_type(thinking_type, ("string",), "/thinking/type", collector):
        _check_enum(thinking_type, ("disabled", "enabled", "adaptive", "auto"), "/thinking/type", collector)
    if "budget_tokens" in value:
        _check_type(value["budget_tokens"], ("integer",), "/thinking/budget_tokens", collector)


def _audit_tool_choice(value: Any, collector: _WarningCollector) -> None:
    if isinstance(value, str):
        _check_enum(value, ("auto", "any", "none"), "/tool_choice", collector)
        return
    if not isinstance(value, Mapping):
        return
    _unknown_fields(
        value,
        {"type", "name", "disable_parallel_tool_use"},
        "/tool_choice",
        collector,
        "tool_choice.unknown_field",
    )
    choice_type = value.get("type", _MISSING)
    if _check_type(choice_type, ("string",), "/tool_choice/type", collector):
        _check_enum(choice_type, ("auto", "any", "none", "tool"), "/tool_choice/type", collector)
    if "name" in value:
        _check_type(value["name"], ("string",), "/tool_choice/name", collector)
    if "disable_parallel_tool_use" in value:
        _check_type(
            value["disable_parallel_tool_use"],
            ("boolean",),
            "/tool_choice/disable_parallel_tool_use",
            collector,
        )


def canonical_tool_contract_hash(tool: Mapping) -> str:
    """Hash a complete tool contract while excluding transient cache hints.

    Description and ``input_schema`` are intentionally included.  Nested
    values remain opaque to warnings; only this digest is exposed on mismatch.
    ``cache_control`` is request-placement state rather than part of a tool's
    callable contract and is therefore excluded.
    """

    if not isinstance(tool, Mapping):
        raise TypeError("tool contract must be an object")
    contract = {str(key): value for key, value in tool.items() if key != "cache_control"}
    return _digest(contract)


def _normalise_hash(value: str) -> str:
    return value if value.startswith("sha256:") else "sha256:" + value


def _baseline_tool_hashes(manifest: Any, cli_version: Optional[str]) -> Dict[str, str]:
    """Accept compact hashes or full contracts in a baseline manifest."""

    if not isinstance(manifest, Mapping):
        return {}
    selected: Any = manifest
    profiles = manifest.get("profiles")
    if isinstance(profiles, Mapping):
        profile_keys = [cli_version]
        if cli_version:
            profile_keys.append("claude_cli_" + cli_version.replace(".", "_"))
        for profile_key in profile_keys:
            if profile_key in profiles:
                selected = profiles[profile_key]
                break
    if isinstance(selected, Mapping) and "tools" in selected:
        selected = selected["tools"]

    result: Dict[str, str] = {}
    if isinstance(selected, list):
        entries = []
        for contract in selected:
            if isinstance(contract, Mapping) and isinstance(contract.get("name"), str):
                entries.append((contract["name"], contract))
    elif isinstance(selected, Mapping):
        entries = list(selected.items())
    else:
        return result

    for name, expected in entries:
        if not isinstance(name, str):
            continue
        if isinstance(expected, str):
            result[name] = _normalise_hash(expected)
            continue
        if not isinstance(expected, Mapping):
            continue
        digest = expected.get("hash") or expected.get("contract_hash")
        if isinstance(digest, str):
            result[name] = _normalise_hash(digest)
            continue
        contract = dict(expected)
        contract.setdefault("name", name)
        result[name] = canonical_tool_contract_hash(contract)
    return result


def _audit_tools(
    value: Any,
    collector: _WarningCollector,
    baseline_manifest: Any,
    cli_version: Optional[str],
) -> None:
    if not isinstance(value, list):
        return
    baselines = _baseline_tool_hashes(baseline_manifest, cli_version)
    for index, tool in enumerate(value):
        path = f"/tools/{index}"
        if not _check_type(tool, ("object",), path, collector, "tool.invalid_type"):
            continue
        _require_fields(
            tool,
            {"name": ("string",), "input_schema": ("object",)},
            path,
            collector,
            "tool.missing_required_field",
        )
        _unknown_fields(tool, _TOOL_FIELDS, path, collector, "tool.unknown_field")
        for key, expected in (
            ("name", ("string",)),
            ("description", ("string",)),
            ("input_schema", ("object",)),
            ("metadata", ("object",)),
            ("cache_control", ("object",)),
            ("defer_loading", ("boolean",)),
            ("allowed_callers", ("array",)),
        ):
            if key in tool:
                _check_type(tool[key], expected, _join_path(path, key), collector)
        # input_schema and metadata are opaque: never recurse into their
        # arbitrary property names, descriptions, defaults, or identities.
        if "cache_control" in tool and isinstance(tool["cache_control"], Mapping):
            _audit_cache_control(tool["cache_control"], _join_path(path, "cache_control"), collector)

        name = tool.get("name")
        if isinstance(name, str) and name in baselines:
            try:
                observed_hash = canonical_tool_contract_hash(tool)
            except (TypeError, ValueError):
                collector.add("tool.contract_invalid", path, "object", ("json_object",))
                continue
            if observed_hash != baselines[name]:
                collector.add(
                    "tool.contract_mismatch",
                    path,
                    "object",
                    ("baseline_contract",),
                    fingerprint=observed_hash,
                )


def _audit_output_config(value: Any, collector: _WarningCollector) -> None:
    if not isinstance(value, Mapping):
        return
    _unknown_fields(value, {"effort", "format"}, "/output_config", collector, "output_config.unknown_field")
    if "effort" in value:
        if _check_type(value["effort"], ("string",), "/output_config/effort", collector):
            _check_enum(
                value["effort"],
                ("none", "minimal", "low", "medium", "high", "xhigh", "max"),
                "/output_config/effort",
                collector,
            )
    # output_config.format is schema-like and therefore opaque.


def _audit_request_payload(
    payload: Any,
    collector: _WarningCollector,
    baseline_manifest: Any,
    cli_version: Optional[str],
) -> None:
    if not _check_type(payload, ("object",), "/", collector, "request.invalid_root_type"):
        return

    _require_fields(
        payload,
        _TOP_LEVEL_REQUIRED_FIELDS,
        "",
        collector,
        "request.missing_required_field",
    )
    _unknown_fields(payload, set(_TOP_LEVEL_TYPES), "", collector, "request.unknown_field")
    valid_types: Dict[str, bool] = {}
    for key, expected in _TOP_LEVEL_TYPES.items():
        if key in payload:
            valid_types[key] = _check_type(payload[key], expected, "/" + key, collector)

    if valid_types.get("messages"):
        _audit_messages(payload["messages"], collector)
    if valid_types.get("system"):
        _audit_system(payload["system"], collector)
    if valid_types.get("tools"):
        _audit_tools(payload["tools"], collector, baseline_manifest, cli_version)
    if valid_types.get("thinking"):
        _audit_thinking(payload["thinking"], collector)
    if valid_types.get("tool_choice"):
        _audit_tool_choice(payload["tool_choice"], collector)
    if valid_types.get("context_management"):
        _audit_context_management(payload["context_management"], collector)
    if valid_types.get("output_config"):
        _audit_output_config(payload["output_config"], collector)

    if valid_types.get("service_tier"):
        _check_enum(
            payload["service_tier"],
            ("auto", "standard_only", "default", "priority"),
            "/service_tier",
            collector,
        )
    if valid_types.get("stop_sequences"):
        for index, stop in enumerate(payload["stop_sequences"]):
            _check_type(stop, ("string",), f"/stop_sequences/{index}", collector)


def audit_anthropic_request(
    headers: Any,
    payload: Any,
    *,
    mode: str = MODE_COMPATIBILITY,
    baseline_manifest: Any = None,
) -> CompatibilityAudit:
    """Audit one Anthropic Messages request against the Claude Code profile.

    Unknown Claude CLI versions always warn but do not by themselves reject a
    request.  In ``lossless_required`` mode unknown protocol fields, beta
    tokens, content blocks, edits, or contracts set ``should_fail``.  The
    ``compatibility`` mode records those as warnings and lets the conversion
    report make its own preservation decision.
    """

    if mode not in VALID_MODES:
        raise ValueError("Unknown compatibility mode: %s" % mode)

    user_agent = _header_value(headers, "user-agent")
    cli_version = _parse_cli_version(user_agent)
    raw_anthropic_version = _header_value(headers, "anthropic-version")
    anthropic_version = (
        raw_anthropic_version.strip()
        if isinstance(raw_anthropic_version, str) and raw_anthropic_version.strip()
        else None
    )
    raw_betas = _header_value(headers, "anthropic-beta")
    betas = normalize_anthropic_betas(raw_betas)
    profile = _make_anthropic_profile(cli_version, anthropic_version, betas)
    collector = _WarningCollector(mode, cli_version)

    if not hasattr(headers, "items") or not callable(getattr(headers, "items", None)):
        collector.add("headers.invalid_type", "/headers", _json_type(headers), ("object",))
    if cli_version not in KNOWN_CLAUDE_CLI_VERSIONS:
        collector.add(
            "claude_cli.unknown_version",
            "/headers/user-agent",
            _json_type(user_agent),
            ("string",),
            evidence={"cli_version": cli_version},
            fail_in_lossless=False,
        )

    if raw_anthropic_version is _MISSING:
        collector.add(
            "anthropic.version_missing",
            "/headers/anthropic-version",
            "missing",
            ("string",),
        )
    elif not isinstance(raw_anthropic_version, str):
        collector.add(
            "anthropic.version_invalid_type",
            "/headers/anthropic-version",
            _json_type(raw_anthropic_version),
            ("string",),
        )
    elif anthropic_version != KNOWN_ANTHROPIC_VERSION:
        collector.add(
            "anthropic.version_unknown",
            "/headers/anthropic-version",
            "string",
            ("string",),
            evidence={"anthropic_version": anthropic_version},
        )

    if raw_betas is not _MISSING and not isinstance(raw_betas, str):
        collector.add(
            "anthropic.beta_invalid_type",
            "/headers/anthropic-beta",
            _json_type(raw_betas),
            ("string",),
        )
    unknown_betas = sorted(set(betas) - set(KNOWN_ANTHROPIC_BETAS))
    if unknown_betas:
        collector.add(
            "anthropic.beta_unknown",
            "/headers/anthropic-beta",
            "string",
            ("string",),
            evidence={"unknown_beta_set": unknown_betas},
        )
    if not unknown_betas and tuple(betas) not in KNOWN_ANTHROPIC_BETA_SETS:
        # A previously unseen subset introduces no unrepresentable input by
        # itself, but beta-set drift should remain visible for every mode.
        collector.add(
            "anthropic.beta_set_unknown",
            "/headers/anthropic-beta",
            "missing" if not betas else "string",
            ("string",),
            evidence={"normalised_beta_set": betas},
            fail_in_lossless=False,
        )

    _audit_request_payload(payload, collector, baseline_manifest, cli_version)
    return CompatibilityAudit(mode=mode, profile=profile, warnings=collector.warnings)


def _audit_responses_item_into(
    item: Any,
    path: str,
    collector: _WarningCollector,
) -> None:
    if not isinstance(item, Mapping):
        collector.add(
            "responses.invalid_item_type",
            path,
            _json_type(item),
            ("object",),
            fail_always=True,
        )
        return
    item_type = item.get("type", _MISSING)
    if not isinstance(item_type, str) or item_type not in KNOWN_RESPONSES_ITEM_TYPES:
        collector.add(
            "responses.unknown_item",
            _join_path(path, "type"),
            _json_type(item_type),
            ("string",),
            evidence={"item_type": None if item_type is _MISSING else item_type},
            fail_always=True,
        )
        return
    _require_fields(
        item,
        _RESPONSES_ITEM_REQUIRED_FIELDS[item_type],
        path,
        collector,
        "responses.missing_item_field",
    )
    _unknown_fields(
        item,
        _RESPONSES_ITEM_FIELDS[item_type],
        path,
        collector,
        "responses.unknown_item_field",
    )
    for content_key in ("content", "summary"):
        content = item.get(content_key)
        if not isinstance(content, list):
            continue
        for index, part in enumerate(content):
            part_path = _join_path(_join_path(path, content_key), index)
            if not isinstance(part, Mapping):
                collector.add(
                    "responses.invalid_content_part",
                    part_path,
                    _json_type(part),
                    ("object",),
                    fail_always=True,
                )
                continue
            part_type = part.get("type", _MISSING)
            if (
                not isinstance(part_type, str)
                or part_type not in _RESPONSES_CONTENT_PART_FIELDS
            ):
                collector.add(
                    "responses.unknown_content_part",
                    _join_path(part_path, "type"),
                    _json_type(part_type),
                    ("string",),
                    evidence={
                        "part_type": None if part_type is _MISSING else part_type
                    },
                    fail_always=True,
                )
                continue
            _require_fields(
                part,
                _RESPONSES_CONTENT_REQUIRED_FIELDS[part_type],
                part_path,
                collector,
                "responses.missing_content_field",
            )
            _unknown_fields(
                part,
                _RESPONSES_CONTENT_PART_FIELDS[part_type],
                part_path,
                collector,
                "responses.unknown_content_field",
            )


def audit_responses_item(
    item: Any,
    *,
    mode: str = MODE_COMPATIBILITY,
    path: str = "/output/0",
) -> CompatibilityAudit:
    """Audit a Responses input/output item discriminator.

    Unknown items are lifecycle-unsafe and therefore fail closed in both
    modes.  No item payload values are included in the warning.
    """

    collector = _WarningCollector(mode, "responses-v1")
    _audit_responses_item_into(item, path, collector)
    return CompatibilityAudit(mode=mode, profile=_responses_profile(), warnings=collector.warnings)


def audit_responses_event(
    event: Any,
    *,
    mode: str = MODE_COMPATIBILITY,
) -> CompatibilityAudit:
    """Audit a Responses SSE event and any directly embedded output items."""

    collector = _WarningCollector(mode, "responses-v1")
    if not isinstance(event, Mapping):
        collector.add(
            "responses.invalid_event_type",
            "/events",
            _json_type(event),
            ("object",),
            fail_always=True,
        )
        return CompatibilityAudit(mode=mode, profile=_responses_profile(), warnings=collector.warnings)

    event_type = event.get("type", _MISSING)
    if not isinstance(event_type, str) or event_type not in KNOWN_RESPONSES_EVENT_TYPES:
        collector.add(
            "responses.unknown_event",
            "/events/type",
            _json_type(event_type),
            ("string",),
            evidence={"event_type": None if event_type is _MISSING else event_type},
            fail_always=True,
        )
    else:
        _require_fields(
            event,
            _RESPONSES_EVENT_REQUIRED_FIELDS[event_type],
            "/events",
            collector,
            "responses.missing_event_field",
        )
        _unknown_fields(
            event,
            _RESPONSES_EVENT_FIELDS[event_type],
            "/events",
            collector,
            "responses.unknown_event_field",
        )
        for field_name, expected_types in _RESPONSES_EVENT_FIELD_TYPES.items():
            if field_name in event:
                _check_type(
                    event[field_name],
                    expected_types,
                    _join_path("/events", field_name),
                    collector,
                    "responses.invalid_event_field_type",
                )
    if isinstance(event.get("item"), Mapping) or "item" in event:
        index = event.get("output_index")
        path = "/output/%s" % index if isinstance(index, int) and not isinstance(index, bool) else "/output/item"
        _audit_responses_item_into(event.get("item"), path, collector)
    response = event.get("response")
    if isinstance(response, Mapping):
        _unknown_fields(
            response,
            _RESPONSES_RESPONSE_FIELDS,
            "/response",
            collector,
            "responses.unknown_response_field",
        )
        if isinstance(response.get("output"), list):
            for index, item in enumerate(response["output"]):
                _audit_responses_item_into(item, f"/output/{index}", collector)
    part = event.get("part")
    if isinstance(part, Mapping):
        part_type = part.get("type", _MISSING)
        if (
            not isinstance(part_type, str)
            or part_type not in _RESPONSES_CONTENT_PART_FIELDS
        ):
            collector.add(
                "responses.unknown_content_part",
                "/events/part/type",
                _json_type(part_type),
                ("string",),
                evidence={"part_type": None if part_type is _MISSING else part_type},
                fail_always=True,
            )
        else:
            _require_fields(
                part,
                _RESPONSES_CONTENT_REQUIRED_FIELDS[part_type],
                "/events/part",
                collector,
                "responses.missing_content_field",
            )
            _unknown_fields(
                part,
                _RESPONSES_CONTENT_PART_FIELDS[part_type],
                "/events/part",
                collector,
                "responses.unknown_content_field",
            )
    return CompatibilityAudit(mode=mode, profile=_responses_profile(), warnings=collector.warnings)


# Descriptive aliases for callers that use the client/protocol name first.
audit_claude_cli_request = audit_anthropic_request
audit_response_event = audit_responses_event
audit_response_item = audit_responses_item


__all__ = [
    "MODE_COMPATIBILITY",
    "MODE_LOSSLESS_REQUIRED",
    "KNOWN_CLAUDE_CLI_VERSIONS",
    "CLAUDE_CLI_TOOL_CONTRACT_BASELINES",
    "KNOWN_ANTHROPIC_VERSION",
    "KNOWN_ANTHROPIC_BETAS",
    "KNOWN_ANTHROPIC_BETA_SETS",
    "KNOWN_RESPONSES_EVENT_TYPES",
    "KNOWN_RESPONSES_ITEM_TYPES",
    "CompatibilityProfile",
    "CompatibilityAudit",
    "normalize_anthropic_betas",
    "canonical_tool_contract_hash",
    "audit_anthropic_request",
    "audit_claude_cli_request",
    "audit_responses_event",
    "audit_response_event",
    "audit_responses_item",
    "audit_response_item",
]
