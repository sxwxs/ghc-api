"""The read tool: text files with offset/limit paging and truncation, images as image blocks."""

import asyncio
import base64
from typing import Annotated

import msgspec

from ...agent.tool import Tool, ToolContext, ToolError, ToolResult, text_result
from ...ai.types import ImageContent, TextContent, Wire
from .paths import resolve_path
from .truncate import MAX_BYTES, MAX_LINES, format_size, truncate_head

IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
# Providers cap images at about 5MB of base64.
MAX_IMAGE_BYTES = 3_500_000


class ReadParams(Wire):
    path: Annotated[str, msgspec.Meta(description="Path to the file to read (relative or absolute)")]
    offset: Annotated[int, msgspec.Meta(ge=1, description="Line number to start reading from (1-indexed)")] | None = (
        None
    )
    limit: Annotated[int, msgspec.Meta(ge=1, description="Maximum number of lines to read")] | None = None


class ReadTool(Tool[ReadParams]):
    name = "read"
    description = (
        f"Read the contents of a file. Supports text files and images (png, jpg, gif, webp). Text output is truncated "
        f"to {MAX_LINES} lines or {MAX_BYTES // 1024}KB, whichever is hit first; use offset/limit for large files."
    )
    params = ReadParams
    kind = "read"
    parallel_safe = True
    snippet = "Read file contents"
    guidelines = ("Use read to examine files instead of cat or sed.",)

    def title(self, params: ReadParams) -> str:
        return params.path if params.offset is None else f"{params.path}:{params.offset}"

    async def execute(self, params: ReadParams, ctx: ToolContext) -> ToolResult:
        path = resolve_path(params.path, ctx.cwd)
        if not path.exists():
            raise ToolError(f"File not found: {params.path}")
        if path.is_dir():
            raise ToolError(f"{params.path} is a directory; list it with bash instead")
        raw = await asyncio.to_thread(path.read_bytes)
        mime = IMAGE_TYPES.get(path.suffix.lower())
        if mime is not None:
            if len(raw) > MAX_IMAGE_BYTES:
                raise ToolError(f"Image is {format_size(len(raw))}; the limit is {format_size(MAX_IMAGE_BYTES)}")
            data = base64.b64encode(raw).decode("ascii")
            return ToolResult(
                content=[TextContent(text=f"Read image file [{mime}]"), ImageContent(data=data, mime_type=mime)]
            )
        if b"\x00" in raw[:8192]:
            return text_result(f"Binary file ({format_size(len(raw))}); not shown")
        # LF only, matching how edit compares text: models do not reproduce CR characters.
        text = raw.decode("utf-8", "replace").removeprefix("\ufeff").replace("\r\n", "\n")
        return text_result(_page(text, params))


def _page(text: str, params: ReadParams) -> str:
    lines = text.split("\n")
    if len(lines) > 1 and text.endswith("\n"):
        lines.pop()
    total = len(lines)
    start = (params.offset or 1) - 1
    if start >= total:
        raise ToolError(f"Offset {params.offset} is beyond the end of the file ({total} lines)")
    end = total if params.limit is None else min(total, start + params.limit)
    page = truncate_head("\n".join(lines[start:end]))
    if page.first_line_too_long:
        return (
            f"[Line {start + 1} exceeds {format_size(MAX_BYTES)}. "
            f"Use bash to read part of it, e.g. head -c 2000 on that line.]"
        )
    if page.truncated:
        shown_end = start + page.output_lines
        notice = f"[Showing lines {start + 1}-{shown_end} of {total}. Use offset={shown_end + 1} to continue.]"
        return f"{page.content}\n\n{notice}"
    if end < total:
        return f"{page.content}\n\n[{total - end} more lines in file. Use offset={end + 1} to continue.]"
    return page.content
