"""Public model metadata shared by Copilot and configured proxy catalogs."""

import copy
from typing import Dict, Optional


def _positive_int(value) -> Optional[int]:
    return value if type(value) is int and value > 0 else None


def _boolean(value) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def _string_list(value):
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    return None


def normalize_model_metadata(model: Dict) -> Dict:
    """Expose known limits and capabilities without inventing missing values.

    Copilot nests these under capabilities; configured proxies already supply
    flat fields. Keep the original capabilities object for future extensions.
    A context window is a combined budget, not a substitute for an input limit.
    """
    capabilities = model.get("capabilities")
    capabilities = capabilities if isinstance(capabilities, dict) else None
    limits = (capabilities or {}).get("limits")
    limits = limits if isinstance(limits, dict) else {}
    supports = (capabilities or {}).get("supports")
    supports = supports if isinstance(supports, dict) else {}

    efforts = _string_list(model.get("reasoning_efforts", supports.get("reasoning_effort")))
    reasoning = _boolean(model.get("reasoning", supports.get("reasoning")))
    if reasoning is None and efforts:
        reasoning = True

    input_types = _string_list(model.get("input"))
    if input_types is None:
        vision = _boolean(supports.get("vision"))
        if vision is True:
            input_types = ["text", "image"]
        elif vision is False or (capabilities or {}).get("type") in ("chat", "embeddings"):
            input_types = ["text"]

    return {
        "model_type": model.get("model_type", (capabilities or {}).get("type")),
        "context_window": _positive_int(model.get("context_window", limits.get("max_context_window_tokens"))),
        "max_input_tokens": _positive_int(model.get("max_input_tokens", limits.get("max_prompt_tokens"))),
        "max_output_tokens": _positive_int(model.get("max_output_tokens", limits.get("max_output_tokens"))),
        "input": input_types,
        "reasoning": reasoning,
        "reasoning_efforts": efforts,
        "tool_calls": _boolean(model.get("tool_calls", supports.get("tool_calls"))),
        "streaming": _boolean(model.get("streaming", supports.get("streaming"))),
        "supported_endpoints": _string_list(model.get("supported_endpoints")),
        "capabilities": copy.deepcopy(capabilities),
    }
