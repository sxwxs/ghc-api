"""The shell tool: output, exit codes, timeouts, aborts, truncation, and output cleanup."""

import asyncio
import time
from pathlib import Path

import msgspec
import pytest

from ghc_api.pyagent.agent.tool import ToolContext, ToolError, ToolResult
from ghc_api.pyagent.ai.types import TextContent
from ghc_api.pyagent.coding.tools.bash import BashDetails, BashParams, BashTool, clean_output, find_shell

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(find_shell()[0] == "powershell", reason="these commands need a POSIX shell"),
]


def text(result: ToolResult) -> str:
    return "".join(block.text for block in result.content if isinstance(block, TextContent))


async def run(
    command: str, cwd: Path, timeout: float | None = None, updates: list[ToolResult] | None = None
) -> ToolResult:
    sink = updates if updates is not None else []
    params = msgspec.convert({"command": command, "timeout": timeout}, BashParams)
    return await BashTool().execute(params, ToolContext(cwd=str(cwd), on_update=sink.append))


async def test_returns_output_from_the_working_directory(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("")
    result = await run("ls; echo err >&2", tmp_path)
    assert text(result) == "marker.txt\nerr\n"
    assert result.details == BashDetails(exit_code=0)


async def test_nonzero_exit_is_an_error_with_the_output(tmp_path: Path) -> None:
    result = await run("echo partial; exit 3", tmp_path)
    assert result.is_error
    assert text(result) == "partial\n\n\nCommand exited with code 3"


async def test_timeout_kills_the_command(tmp_path: Path) -> None:
    started = time.monotonic()
    with pytest.raises(ToolError, match=r"timed out after 0\.5 seconds"):
        await run("echo before; sleep 10", tmp_path, timeout=0.5)
    assert time.monotonic() - started < 5


async def test_cancellation_kills_the_process_tree(tmp_path: Path) -> None:
    marker = tmp_path / "late.txt"
    task = asyncio.create_task(run(f"sleep 1.5; touch '{marker.as_posix()}'", tmp_path))
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(2)
    assert not marker.exists()


async def test_large_output_keeps_the_tail_and_saves_everything(tmp_path: Path) -> None:
    updates: list[ToolResult] = []
    result = await run("for i in $(seq 1 30000); do echo line$i; done", tmp_path, updates=updates)
    assert isinstance(result.details, BashDetails) and result.details.truncated
    assert result.details.full_output_path is not None
    full = Path(result.details.full_output_path).read_text()
    assert full.startswith("line1\n") and full.endswith("line30000\n")
    assert "line30000\n\n[Output truncated: showing the last 2000 lines" in text(result)
    assert updates  # streamed partial output


def test_clean_output_strips_escapes_redraws_and_control_characters() -> None:
    assert clean_output("\x1b[31mred\x1b[0m\r\n10%\r50%\r100%\ndone\x07") == "red\n100%\ndone"


def test_title_is_the_first_command_line() -> None:
    tool = BashTool()
    assert tool.title(BashParams(command="pytest -q\necho done")) == "pytest -q"
    assert tool.title(BashParams(command="x" * 100)).endswith("...")
