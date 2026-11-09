"""The edit tool: exact, unique, non-overlapping text replacements in one file, keeping its BOM and line endings."""

import asyncio
import contextlib
import difflib
import itertools
import json
from typing import Annotated, Any

import msgspec

from ...agent.tool import Tool, ToolContext, ToolError, ToolResult, text_result
from ...ai.types import Wire
from .paths import resolve_path


class Replacement(Wire):
    old_text: Annotated[
        str,
        msgspec.Meta(
            description="Exact text for one targeted replacement. It must be unique in the original file and must "
            "not overlap with any other edits[].oldText in the same call."
        ),
    ]
    new_text: Annotated[str, msgspec.Meta(description="Replacement text for this targeted edit.")]


class EditParams(Wire):
    path: Annotated[str, msgspec.Meta(description="Path to the file to edit (relative or absolute)")]
    edits: Annotated[
        list[Replacement],
        msgspec.Meta(
            min_length=1,
            description="One or more targeted replacements. Each edit is matched against the original file, not "
            "incrementally. Do not include overlapping or nested edits; merge nearby changes into one edit.",
        ),
    ]


class EditDetails(Wire):
    diff: str
    first_changed_line: int


class EditTool(Tool[EditParams]):
    name = "edit"
    description = (
        "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping "
        "region of the original file. If two changes affect the same block or nearby lines, merge them into one edit "
        "instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes."
    )
    params = EditParams
    kind = "edit"
    snippet = "Make precise file edits with exact text replacement, including multiple disjoint edits in one call"
    guidelines = (
        "Use edit for targeted changes. Batch changes to one file in a single call.",
        "Each edits[].oldText must exactly and uniquely match the original file. Keep matches short and "
        "non-overlapping; merge nearby changes without including large unchanged regions.",
    )

    def prepare_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        # Models sometimes send edits as a JSON string, as one object, or as top-level oldText/newText.
        args = dict(arguments)
        edits: Any = args.get("edits")
        if isinstance(edits, str):
            with contextlib.suppress(json.JSONDecodeError):
                edits = json.loads(edits)
        if isinstance(edits, dict):
            edits = [edits]
        old, new = args.get("oldText"), args.get("newText")
        if isinstance(old, str) and isinstance(new, str):
            del args["oldText"], args["newText"]
            edits = [*(edits if isinstance(edits, list) else []), {"oldText": old, "newText": new}]
        if edits is not None:
            args["edits"] = edits
        return args

    def title(self, params: EditParams) -> str:
        return params.path

    async def execute(self, params: EditParams, ctx: ToolContext) -> ToolResult:
        path = resolve_path(params.path, ctx.cwd)
        if not path.is_file():
            raise ToolError(f"File not found: {params.path}")
        try:
            # Strict: the whole file is written back, so replacement characters would destroy bytes outside the edits.
            raw = (await asyncio.to_thread(path.read_bytes)).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ToolError(
                f"{params.path} is not UTF-8 text ({error.reason}); edit cannot change it safely"
            ) from error
        bom = "\ufeff" if raw.startswith("\ufeff") else ""
        crlf = "\r\n" in raw
        # Match on LF text: the model will not reproduce an invisible BOM or CR characters.
        text = raw.removeprefix(bom).replace("\r\n", "\n")
        new_text, first_offset = apply_edits(text, params.edits, params.path)
        output = bom + (new_text.replace("\n", "\r\n") if crlf else new_text)
        await asyncio.to_thread(path.write_bytes, output.encode())
        details = EditDetails(
            diff=_diff(params.path, text, new_text), first_changed_line=text.count("\n", 0, first_offset) + 1
        )
        return text_result(f"Replaced {len(params.edits)} block(s) in {params.path}.", details)


def apply_edits(text: str, edits: list[Replacement], label: str) -> tuple[str, int]:
    """Apply replacements matched against the original text. Returns the new text and the first changed offset."""
    spans: list[tuple[int, int, str, int]] = []
    for number, edit in enumerate(edits):
        old = edit.old_text.replace("\r\n", "\n")
        if not old:
            raise ToolError(f"edits[{number}].oldText must not be empty")
        start = text.find(old)
        if start < 0:
            raise ToolError(
                f"edits[{number}].oldText was not found in {label}. It must match the file exactly, "
                "including whitespace and indentation."
            )
        if text.find(old, start + 1) >= 0:
            raise ToolError(
                f"edits[{number}].oldText matches {text.count(old)} places in {label}. "
                "Add surrounding context to make it unique."
            )
        spans.append((start, start + len(old), edit.new_text.replace("\r\n", "\n"), number))
    spans.sort()
    for previous, current in itertools.pairwise(spans):
        if current[0] < previous[1]:
            raise ToolError(
                f"edits[{previous[3]}] and edits[{current[3]}] overlap in {label}; merge them into one edit"
            )
    parts: list[str] = []
    cursor = 0
    for start, end, replacement, _ in spans:
        parts += [text[cursor:start], replacement]
        cursor = end
    parts.append(text[cursor:])
    new_text = "".join(parts)
    if new_text == text:
        raise ToolError(f"The edits leave {label} unchanged")
    return new_text, spans[0][0]


def _diff(label: str, before: str, after: str) -> str:
    lines = difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile=label, tofile=label, n=3
    )
    return "".join(lines)
