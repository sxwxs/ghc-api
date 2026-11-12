"""The write tool: creates or overwrites a whole file."""

import asyncio
from pathlib import Path
from typing import Annotated

import msgspec

from ...agent.tool import Tool, ToolContext, ToolResult, text_result
from ...ai.types import Wire
from .paths import resolve_path


class WriteParams(Wire):
    path: Annotated[str, msgspec.Meta(description="Path to the file to write (relative or absolute)")]
    content: Annotated[str, msgspec.Meta(description="Content to write to the file")]


class WriteTool(Tool[WriteParams]):
    name = "write"
    description = (
        "Write content to a file. Creates the file if it doesn't exist, overwrites it if it does. "
        "Automatically creates parent directories."
    )
    params = WriteParams
    kind = "edit"
    snippet = "Create or overwrite files"
    guidelines = ("Use write only for new files or complete rewrites.",)

    def title(self, params: WriteParams) -> str:
        return params.path

    async def execute(self, params: WriteParams, ctx: ToolContext) -> ToolResult:
        path = resolve_path(params.path, ctx.cwd)
        data = params.content.encode()
        await asyncio.to_thread(_write, path, data)
        return text_result(f"Wrote {len(data)} bytes to {params.path}")


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Bytes, not text mode: text mode would turn \n into \r\n on Windows.
    path.write_bytes(data)
