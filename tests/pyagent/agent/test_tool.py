"""Tool schemas: what the model sees for typed params."""

from typing import Annotated

import msgspec

from ghc_api.pyagent.agent.tool import json_schema
from ghc_api.pyagent.ai.types import Wire


class Item(Wire):
    old_text: str


class Params(Wire):
    title: Annotated[str, msgspec.Meta(description="A property that happens to be called title")]
    items: list[Item]
    limit: Annotated[int, msgspec.Meta(ge=1, description="Optional limit")] | None = None


def test_schema_is_self_contained_and_free_of_noise() -> None:
    assert json_schema(Params) == {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "A property that happens to be called title"},
            "items": {
                "type": "array",
                "items": {"type": "object", "properties": {"oldText": {"type": "string"}}, "required": ["oldText"]},
            },
            "limit": {"type": "integer", "minimum": 1, "description": "Optional limit"},
        },
        "required": ["title", "items"],
    }
