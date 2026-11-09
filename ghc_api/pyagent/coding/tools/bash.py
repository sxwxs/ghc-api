"""The shell tool: runs a command, streams its output, enforces timeouts, and kills the whole process tree on abort."""

import asyncio
import contextlib
import functools
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Annotated, BinaryIO

import msgspec

from ...agent.tool import Tool, ToolContext, ToolError, ToolResult, text_result
from ...ai.types import Wire
from .truncate import MAX_BYTES, MAX_LINES, format_size, truncate_tail

UPDATE_INTERVAL = 0.1
NEW_GROUP_FLAGS = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
# After the shell exits, background processes it started may keep the output pipe open.
DRAIN_GRACE = 1.0


class BashParams(Wire):
    command: Annotated[str, msgspec.Meta(description="Command to execute")]
    timeout: (
        Annotated[float, msgspec.Meta(gt=0, description="Timeout in seconds (optional, no default timeout)")] | None
    ) = None


class BashDetails(Wire):
    exit_code: int | None
    truncated: bool = False
    full_output_path: str | None = None


class BashTool(Tool[BashParams]):
    params = BashParams
    kind = "execute"

    def __init__(self, shell: tuple[str, list[str]] | None = None) -> None:
        shell_name, self.argv = shell or find_shell()
        self.name = "powershell" if shell_name == "powershell" else "bash"
        self.description = (
            f"Execute a {shell_name} command in the current working directory. Returns stdout and stderr. Output is "
            f"truncated to the last {MAX_LINES} lines or {MAX_BYTES // 1024}KB, whichever is hit first; if truncated, "
            "the full output is saved to a temp file. Optionally provide a timeout in seconds."
        )
        self.snippet = f"Execute {shell_name} commands (ls, grep, find, etc.)"

    def title(self, params: BashParams) -> str:
        first = params.command.strip().splitlines()[0] if params.command.strip() else ""
        return first if len(first) <= 80 else first[:77] + "..."

    async def execute(self, params: BashParams, ctx: ToolContext) -> ToolResult:
        process = await asyncio.create_subprocess_exec(
            *self.argv,
            params.command,
            cwd=ctx.cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # A separate group lets us kill the command with its children and keeps the user's Ctrl+C away from it.
            creationflags=NEW_GROUP_FLAGS,
            start_new_session=sys.platform != "win32",
        )
        output = Output()
        reader = asyncio.create_task(_pump(process, output, ctx))
        timed_out = False
        try:
            try:
                async with asyncio.timeout(params.timeout):
                    await process.wait()
            except TimeoutError:
                timed_out = True
                _kill_tree(process.pid)
                await process.wait()
            await asyncio.wait({reader}, timeout=DRAIN_GRACE)
        finally:
            reader.cancel()
            if process.returncode is None:
                _kill_tree(process.pid)
            output.close()
        text, details = output.finish(_exit_code(process.returncode))
        if timed_out:
            raise ToolError(_with_status(text, f"Command timed out after {params.timeout:g} seconds"))
        if details.exit_code != 0:
            return text_result(
                _with_status(text, f"Command exited with code {details.exit_code}"), details, is_error=True
            )
        return text_result(text or "(no output)", details)


async def _pump(process: asyncio.subprocess.Process, output: "Output", ctx: ToolContext) -> None:
    assert process.stdout is not None  # spawned with stdout=PIPE
    last_update = 0.0
    while chunk := await process.stdout.read(65536):
        output.append(chunk)
        now = time.monotonic()
        if now - last_update >= UPDATE_INTERVAL:
            last_update = now
            ctx.on_update(text_result(truncate_tail(output.text()).content))


class Output:
    """Collects command output: all of it in memory up to the byte limit; beyond that everything also goes to a
    temp file and memory keeps a tail."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.total = 0
        self.file: BinaryIO | None = None
        self.path: str | None = None

    def append(self, data: bytes) -> None:
        """Add a chunk."""
        self.total += len(data)
        self.buffer += data
        if self.file is None and self.total > MAX_BYTES:
            self._spill()
        elif self.file is not None:
            self.file.write(data)
        if len(self.buffer) > 2 * MAX_BYTES:
            del self.buffer[: len(self.buffer) - MAX_BYTES]

    def text(self) -> str:
        """The buffered output, cleaned of terminal control sequences."""
        return clean_output(self.buffer.decode("utf-8", "replace"))

    def finish(self, exit_code: int | None) -> tuple[str, BashDetails]:
        """The model-facing text, tail-truncated with a pointer to the full output, and the details."""
        tail = truncate_tail(self.text())
        if tail.truncated and self.file is None:
            self._spill()
            self.close()
        truncated = self.file is not None
        text = tail.content
        if truncated:
            text += (
                f"\n\n[Output truncated: showing the last {tail.output_lines} lines of {format_size(self.total)}. "
                f"Full output: {self.path}]"
            )
        return text, BashDetails(exit_code=exit_code, truncated=truncated, full_output_path=self.path)

    def close(self) -> None:
        """Close the temp file, if any."""
        if self.file is not None and not self.file.closed:
            self.file.close()

    def _spill(self) -> None:
        handle, self.path = tempfile.mkstemp(prefix="pyagent-bash-", suffix=".log")
        self.file = os.fdopen(handle, "wb")
        self.file.write(self.buffer)


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")
# A carriage return without a newline redraws the line (progress bars); keep only the final redraw.
REDRAWN = re.compile(r"[^\n]*\r(?!\n)")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_output(text: str) -> str:
    """Strip ANSI escapes, progress-bar redraws and control characters; they only cost tokens."""
    text = ANSI.sub("", text).replace("\r\n", "\n")
    return CONTROL.sub("", REDRAWN.sub("", text))


def _with_status(text: str, status: str) -> str:
    return f"{text}\n\n{status}" if text else status


def _exit_code(code: int | None) -> int | None:
    # A signal-killed process reports -signal; shells report 128 + signal.
    return 128 - code if code is not None and code < 0 else code


@functools.cache
def find_shell() -> tuple[str, list[str]]:
    """Return the shell name and argv prefix: bash where available, else PowerShell on Windows, else sh."""
    if sys.platform != "win32":
        if os.path.exists("/bin/bash"):
            return "bash", ["/bin/bash", "-c"]
        bash = shutil.which("bash")
        return ("bash", [bash, "-c"]) if bash else ("sh", ["sh", "-c"])
    for base in (os.environ.get("PROGRAMFILES"), os.environ.get("PROGRAMFILES(X86)")):
        if base and os.path.exists(candidate := os.path.join(base, "Git", "bin", "bash.exe")):
            return "bash", [candidate, "-c"]
    bash = shutil.which("bash")
    # System32\bash.exe launches WSL, which runs commands in a different file system.
    if bash and "system32" not in bash.lower():
        return "bash", [bash, "-c"]
    system_root = os.environ.get("SYSTEMROOT", r"C:\Windows")
    powershell = (
        shutil.which("pwsh")
        or shutil.which("powershell")
        or os.path.join(system_root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
    )
    return "powershell", [powershell, "-NoProfile", "-NonInteractive", "-Command"]


def _kill_tree(pid: int) -> None:
    if sys.platform == "win32":
        system_root = os.environ.get("SYSTEMROOT", r"C:\Windows")
        taskkill = os.path.join(system_root, "System32", "taskkill.exe")
        # Fire and forget: the caller waits for the process itself.
        subprocess.Popen(
            [taskkill, "/F", "/T", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)
