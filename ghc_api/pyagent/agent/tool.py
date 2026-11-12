"""Tool contract: typed params, the schema the model sees, the execution context, and results."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

import msgspec

from ..ai.types import TextContent, ToolResultContent, ToolSpec, Wire

ToolKind = Literal["read", "edit", "execute", "search", "other"]


class ToolResult(Wire):
    """Tool output: `content` goes to the model, `details` only to UIs."""

    content: list[ToolResultContent]
    details: Any = None
    is_error: bool = False


class ToolError(Exception):
    """An expected tool failure; its message is what the model sees."""


@dataclass
class ToolContext:
    """What a running tool may use. `on_update` streams partial results to UIs."""

    cwd: str
    on_update: Callable[[ToolResult], None]


P = TypeVar("P", bound=msgspec.Struct)


class Tool(ABC, Generic[P]):
    """A tool the model can call. Subclasses set the class attributes and implement `execute`.

    Tools must be cancellation-safe: an abort cancels `execute`, so cleanup belongs in `finally`.
    """

    name: str
    description: str
    params: type[P]
    kind: ToolKind = "other"
    parallel_safe: bool = False
    snippet: str = ""  # one line for the system prompt's tool list
    guidelines: tuple[str, ...] = ()

    def prepare_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Repair common model mistakes in raw arguments before validation. Default: unchanged."""
        return arguments

    def title(self, params: P) -> str:
        """One line describing this call for UIs."""
        return self.name

    @abstractmethod
    async def execute(self, params: P, ctx: ToolContext) -> ToolResult:
        """Run the call. Raise ToolError (or any exception) to report a failure to the model."""

    def spec(self) -> ToolSpec:
        """The declaration sent to the model."""
        return ToolSpec(name=self.name, description=self.description, parameters=json_schema(self.params))


def text_result(text: str, details: Any = None, is_error: bool = False) -> ToolResult:
    """A result with one text block."""
    return ToolResult(content=[TextContent(text=text)], details=details, is_error=is_error)


def json_schema(params: type[msgspec.Struct]) -> dict[str, Any]:
    """A self-contained JSON Schema for `params`: refs inlined, titles and null branches of optional fields removed."""
    raw = msgspec.json.schema(params)
    return _clean(raw, raw.get("$defs", {}))


def _clean(node: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    if "$ref" in node:
        return _clean(defs[node["$ref"].rsplit("/", 1)[-1]], defs)
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in ("title", "$defs"):
            continue
        if key == "properties":
            out[key] = {name: _clean(schema, defs) for name, schema in value.items()}
        elif key == "anyOf":
            out[key] = [_clean(schema, defs) for schema in value]
        elif key in ("items", "additionalProperties") and isinstance(value, dict):
            out[key] = _clean(value, defs)  # pyright: ignore[reportUnknownArgumentType]
        else:
            out[key] = value
    # An optional field is simply omitted, so its null branch and null default only add noise.
    branches = [b for b in out.get("anyOf", []) if b != {"type": "null"}]
    if "anyOf" in out and len(branches) == 1:
        del out["anyOf"]
        out = {**branches[0], **out}
    if "default" in out and out["default"] is None:
        del out["default"]
    return out
