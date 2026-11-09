"""Output truncation by line and byte limits, whichever is hit first: head for file reads, tail for command output."""

from dataclasses import dataclass

MAX_LINES = 2000
MAX_BYTES = 50 * 1024


@dataclass
class Truncation:
    content: str
    truncated: bool
    total_lines: int
    output_lines: int
    first_line_too_long: bool = False  # head: the first line alone exceeds the byte limit, so nothing fits
    last_line_partial: bool = False  # tail: only the end of the last line fits


def truncate_head(text: str, max_lines: int = MAX_LINES, max_bytes: int = MAX_BYTES) -> Truncation:
    """Keep whole lines from the start."""
    lines = _lines(text)
    if len(lines) <= max_lines and len(text.encode()) <= max_bytes:
        return Truncation(text, False, len(lines), len(lines))
    kept: list[str] = []
    size = 0
    for line in lines[:max_lines]:
        cost = len(line.encode()) + (1 if kept else 0)
        if size + cost > max_bytes:
            break
        kept.append(line)
        size += cost
    return Truncation("\n".join(kept), True, len(lines), len(kept), first_line_too_long=not kept)


def truncate_tail(text: str, max_lines: int = MAX_LINES, max_bytes: int = MAX_BYTES) -> Truncation:
    """Keep whole lines from the end; if even the last line is too long, keep its end."""
    lines = _lines(text)
    if len(lines) <= max_lines and len(text.encode()) <= max_bytes:
        return Truncation(text, False, len(lines), len(lines))
    kept: list[str] = []
    size = 0
    partial = False
    for line in reversed(lines[-max_lines:]):
        cost = len(line.encode()) + (1 if kept else 0)
        if size + cost > max_bytes:
            if not kept:
                kept.append(line.encode()[-max_bytes:].decode("utf-8", "ignore"))
                partial = True
            break
        kept.append(line)
        size += cost
    kept.reverse()
    return Truncation("\n".join(kept), True, len(lines), len(kept), last_line_partial=partial)


def format_size(size: int) -> str:
    """Human-readable byte size."""
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / (1024 * 1024):.1f}MB"


def _lines(text: str) -> list[str]:
    if not text:
        return []
    lines = text.split("\n")
    if text.endswith("\n"):
        lines.pop()
    return lines
