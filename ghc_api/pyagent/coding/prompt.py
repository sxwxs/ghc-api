"""System prompt: base instructions, the tool list and rules, project context files, and environment facts."""

import datetime
import platform
from pathlib import Path
from typing import Any

from ..agent.tool import Tool

# The first name found in a directory wins.
CONTEXT_FILE_NAMES = ("AGENTS.md", "CLAUDE.md")


def load_context_files(cwd: str, home: Path) -> list[tuple[str, str]]:
    """Context files from the pyagent home, then from every directory from the file system root down to cwd."""
    here = Path(cwd).resolve()
    directories = [home, *reversed(here.parents), here]
    files: list[tuple[str, str]] = []
    seen: set[Path] = set()
    for directory in directories:
        for name in CONTEXT_FILE_NAMES:
            path = directory / name
            if path.is_file():
                if path.resolve() not in seen:
                    seen.add(path.resolve())
                    files.append((str(path), path.read_text("utf-8", errors="replace")))
                break
    return files


def build_system_prompt(cwd: str, tools: list[Tool[Any]], context_files: list[tuple[str, str]]) -> str:
    """Assemble the system prompt."""
    sections = [
        "You are a coding assistant running inside pyagent, a coding agent harness. "
        "Help users inspect files, run commands, and modify code.",
        "Available tools:\n" + "\n".join(f"- {tool.name}: {tool.snippet}" for tool in tools if tool.snippet),
        "Guidelines:\n" + "\n".join(f"- {rule}" for rule in _rules(tools)),
    ]
    if context_files:
        instructions = "\n\n".join(
            f'<project_instructions path="{path}">\n{content}\n</project_instructions>'
            for path, content in context_files
        )
        sections.append(
            f"<project_context>\nProject-specific instructions and guidelines:\n\n{instructions}\n</project_context>"
        )
    sections.append(
        "<environment>\n"
        f"Working directory: {Path(cwd).as_posix()}\n"
        f"Platform: {platform.system()}\n"
        f"Date: {datetime.date.today().isoformat()}\n"
        "</environment>"
    )
    return "\n\n".join(sections)


def _rules(tools: list[Tool[Any]]) -> list[str]:
    names = {tool.name for tool in tools}
    rules: list[str] = []
    shell = "bash" if "bash" in names else "powershell" if "powershell" in names else None
    if shell and not names & {"grep", "find", "ls"}:
        rules.append(f"Use {shell} for file operations like listing, searching, and finding files")
    for tool in tools:
        rules += [rule for rule in tool.guidelines if rule not in rules]
    rules += ["Be concise in your responses", "Show file paths clearly when working with files"]
    return rules
