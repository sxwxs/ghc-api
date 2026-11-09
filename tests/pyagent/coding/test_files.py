"""read, write and edit tools against a temporary directory."""

import base64
from pathlib import Path
from typing import Any

import msgspec
import pytest

from ghc_api.pyagent.agent.tool import Tool, ToolContext, ToolError, ToolResult
from ghc_api.pyagent.ai.types import ImageContent, TextContent
from ghc_api.pyagent.coding.tools.edit import EditDetails, EditTool
from ghc_api.pyagent.coding.tools.read import ReadTool
from ghc_api.pyagent.coding.tools.write import WriteTool

pytestmark = pytest.mark.anyio


async def call(tool: Tool[Any], cwd: Path, **arguments: Any) -> ToolResult:
    params = msgspec.convert(tool.prepare_arguments(arguments), tool.params)
    return await tool.execute(params, ToolContext(cwd=str(cwd), on_update=lambda _: None))


def text(result: ToolResult) -> str:
    return "".join(block.text for block in result.content if isinstance(block, TextContent))


async def test_read_pages_with_offset_and_limit(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("".join(f"l{i}\n" for i in range(1, 11)))
    assert text(await call(ReadTool(), tmp_path, path="a.txt")) == "\n".join(f"l{i}" for i in range(1, 11))
    paged = text(await call(ReadTool(), tmp_path, path="a.txt", offset=3, limit=2))
    assert paged == "l3\nl4\n\n[6 more lines in file. Use offset=5 to continue.]"
    with pytest.raises(ToolError, match="beyond the end"):
        await call(ReadTool(), tmp_path, path="a.txt", offset=20)


async def test_read_truncates_long_files_with_a_continuation_hint(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text("\n".join("x" for _ in range(3000)))
    result = text(await call(ReadTool(), tmp_path, path="big.txt"))
    assert result.endswith("[Showing lines 1-2000 of 3000. Use offset=2001 to continue.]")


async def test_read_returns_images_and_refuses_binaries_and_missing_files(tmp_path: Path) -> None:
    (tmp_path / "p.png").write_bytes(b"\x89PNG...")
    image = await call(ReadTool(), tmp_path, path="@p.png")
    assert image.content[1] == ImageContent(data=base64.b64encode(b"\x89PNG...").decode(), mime_type="image/png")
    (tmp_path / "b.bin").write_bytes(b"\x00\x01\x02")
    assert "Binary file" in text(await call(ReadTool(), tmp_path, path="b.bin"))
    with pytest.raises(ToolError, match="File not found"):
        await call(ReadTool(), tmp_path, path="missing.txt")


async def test_write_creates_directories_and_keeps_line_endings(tmp_path: Path) -> None:
    result = await call(WriteTool(), tmp_path, path="sub/dir/f.txt", content="a\nb\n")
    assert text(result) == "Wrote 4 bytes to sub/dir/f.txt"
    assert (tmp_path / "sub/dir/f.txt").read_bytes() == b"a\nb\n"


async def test_edit_applies_disjoint_replacements_and_reports_a_diff(tmp_path: Path) -> None:
    (tmp_path / "f.py").write_text("a = 1\nb = 2\nc = 3\n")
    result = await call(
        EditTool(),
        tmp_path,
        path="f.py",
        edits=[{"oldText": "c = 3", "newText": "c = 30"}, {"oldText": "a = 1", "newText": "a = 10"}],
    )
    assert (tmp_path / "f.py").read_text() == "a = 10\nb = 2\nc = 30\n"
    assert text(result) == "Replaced 2 block(s) in f.py."
    assert isinstance(result.details, EditDetails)
    assert result.details.first_changed_line == 1
    assert "-a = 1\n+a = 10\n" in result.details.diff


async def test_edit_keeps_bom_and_crlf(tmp_path: Path) -> None:
    (tmp_path / "w.txt").write_bytes("\ufeffone\r\ntwo\r\n".encode())
    await call(EditTool(), tmp_path, path="w.txt", edits=[{"oldText": "one\ntwo", "newText": "uno\ndos"}])
    assert (tmp_path / "w.txt").read_bytes() == "\ufeffuno\r\ndos\r\n".encode()


@pytest.mark.parametrize(
    ("edits", "message"),
    [
        ([{"oldText": "zzz", "newText": "y"}], "was not found"),
        ([{"oldText": "x", "newText": "y"}], "matches 2 places"),
        ([{"oldText": "x x", "newText": "a"}, {"oldText": "x\ny", "newText": "b"}], "overlap"),
        ([{"oldText": "y", "newText": "y"}], "unchanged"),
    ],
)
async def test_edit_rejects_ambiguous_missing_overlapping_and_empty_changes(
    tmp_path: Path, edits: list[dict[str, str]], message: str
) -> None:
    (tmp_path / "f.txt").write_text("x x\ny\n")
    with pytest.raises(ToolError, match=message):
        await call(EditTool(), tmp_path, path="f.txt", edits=edits)
    assert (tmp_path / "f.txt").read_text() == "x x\ny\n"


def test_edit_repairs_common_argument_shapes() -> None:
    tool = EditTool()
    single = {"oldText": "a", "newText": "b"}
    expected = {"path": "f", "edits": [single]}
    assert tool.prepare_arguments({"path": "f", "edits": '[{"oldText": "a", "newText": "b"}]'}) == expected
    assert tool.prepare_arguments({"path": "f", "edits": single}) == expected
    assert tool.prepare_arguments({"path": "f", "oldText": "a", "newText": "b"}) == expected
