"""Session files: append-only JSONL entries forming a tree, where the last appended entry is the active leaf."""

import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import msgspec

from ..ai.types import Message, TextContent, ThinkingLevel, UserMessage, Wire, new_id, now_ms

VERSION = 1


class Header(Wire, tag="session", tag_field="type"):
    """First line of a session file."""

    version: int
    id: str
    cwd: str
    timestamp: int
    parent_session: str | None = None  # the file this session was forked from


class EntryBase(Wire, tag_field="type"):
    """Base of entries. `parent_id` links each entry to the one before it on its branch."""

    id: str
    parent_id: str | None
    timestamp: int


class MessageEntry(EntryBase, tag="message", kw_only=True):
    message: Message


class ModelChangeEntry(EntryBase, tag="model_change", kw_only=True):
    provider: str
    model_id: str


class ThinkingChangeEntry(EntryBase, tag="thinking_change", kw_only=True):
    level: ThinkingLevel


class SessionInfoEntry(EntryBase, tag="session_info", kw_only=True):
    name: str


Entry = MessageEntry | ModelChangeEntry | ThinkingChangeEntry | SessionInfoEntry
LINE = msgspec.json.Decoder(Header | Entry)


class SessionError(Exception):
    """A session file or entry does not exist or cannot be read."""


@dataclass
class SessionContext:
    """What the active branch says: its messages and the latest settings on it."""

    messages: list[Message]
    provider: str | None
    model_id: str | None
    thinking: ThinkingLevel | None
    name: str | None


class SessionSummary(Wire):
    """One line of a session list."""

    session_id: str
    cwd: str
    updated_at: int
    message_count: int
    name: str | None = None
    first_message: str | None = None


class SessionStore:
    """One session. A new session writes its file only once the first message arrives, so empty sessions leave
    nothing behind. Without a path the session lives in memory only."""

    def __init__(self, header: Header, path: Path | None, entries: list[Entry], on_disk: bool) -> None:
        self.header = header
        self.path = path
        self.entries: dict[str, Entry] = {entry.id: entry for entry in entries}
        self.leaf: str | None = entries[-1].id if entries else None
        self._file: BinaryIO | None = None
        # Lines waiting for the first message; None once the file exists.
        self._pending: list[bytes] | None = None if on_disk else [msgspec.json.encode(header) + b"\n"]

    @classmethod
    def create(cls, cwd: str, directory: Path | None, parent_session: str | None = None) -> "SessionStore":
        """A new session in `directory`, or in memory when `directory` is None."""
        session_id = secrets.token_hex(6)
        header = Header(version=VERSION, id=session_id, cwd=cwd, timestamp=now_ms(), parent_session=parent_session)
        path = directory / f"{session_id}.jsonl" if directory is not None else None
        return cls(header, path, [], on_disk=False)

    @classmethod
    def open(cls, path: Path) -> "SessionStore":
        """Read a session file. Lines that cannot be decoded, such as a line cut short by a crash, are skipped."""
        header, entries = _read(path)
        return cls(header, path, entries, on_disk=True)

    @property
    def id(self) -> str:
        return self.header.id

    @property
    def cwd(self) -> str:
        return self.header.cwd

    def append_message(self, message: Message) -> None:
        """Record a message; its id becomes the entry id."""
        self._append(MessageEntry(id=message.id, parent_id=self.leaf, timestamp=now_ms(), message=message))

    def append_model_change(self, provider: str, model_id: str) -> None:
        """Record a model selection."""
        self._append(
            ModelChangeEntry(id=new_id(), parent_id=self.leaf, timestamp=now_ms(), provider=provider, model_id=model_id)
        )

    def append_thinking_change(self, level: ThinkingLevel) -> None:
        """Record a thinking level selection."""
        self._append(ThinkingChangeEntry(id=new_id(), parent_id=self.leaf, timestamp=now_ms(), level=level))

    def append_session_info(self, name: str) -> None:
        """Record the session's display name."""
        self._append(SessionInfoEntry(id=new_id(), parent_id=self.leaf, timestamp=now_ms(), name=name))

    def branch(self, leaf: str | None = None) -> list[Entry]:
        """The entries from the root to `leaf` (default: the active leaf)."""
        entries: list[Entry] = []
        current = leaf if leaf is not None else self.leaf
        while current is not None:
            entry = self.entries.get(current)
            if entry is None:
                raise SessionError(f"Entry {current} not found in session {self.id}")
            entries.append(entry)
            current = entry.parent_id
        entries.reverse()
        return entries

    def context(self) -> SessionContext:
        """Messages and settings of the active branch."""
        context = SessionContext(messages=[], provider=None, model_id=None, thinking=None, name=None)
        for entry in self.branch():
            if isinstance(entry, MessageEntry):
                context.messages.append(entry.message)
            elif isinstance(entry, ModelChangeEntry):
                context.provider, context.model_id = entry.provider, entry.model_id
            elif isinstance(entry, ThinkingChangeEntry):
                context.thinking = entry.level
            else:
                context.name = entry.name
        return context

    def fork(self, entry_id: str, directory: Path | None) -> "SessionStore":
        """A new session holding this session's entries from the root through `entry_id`, written at once."""
        entries = self.branch(entry_id)
        store = SessionStore.create(self.cwd, directory, parent_session=str(self.path) if self.path else None)
        store.entries = {entry.id: entry for entry in entries}
        store.leaf = entry_id
        assert store._pending is not None  # a new store has not written its file yet
        store._pending += [msgspec.json.encode(entry) + b"\n" for entry in entries]
        store._write([])
        return store

    def close(self) -> None:
        """Close the file handle, if any."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def _append(self, entry: Entry) -> None:
        self.entries[entry.id] = entry
        self.leaf = entry.id
        line = msgspec.json.encode(entry) + b"\n"
        if self._pending is not None and not isinstance(entry, MessageEntry):
            self._pending.append(line)
        else:
            self._write([line])

    def _write(self, lines: list[bytes]) -> None:
        if self.path is None:
            return
        if self._file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("ab")
        if self._pending is not None:
            lines = self._pending + lines
            self._pending = None
        self._file.write(b"".join(lines))
        self._file.flush()


def _read(path: Path) -> tuple[Header, list[Entry]]:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise SessionError(f"Cannot read session file {path}: {error}") from error
    header: Header | None = None
    entries: list[Entry] = []
    for raw in data.split(b"\n"):
        if not raw.strip():
            continue
        try:
            line = LINE.decode(raw)
        except msgspec.DecodeError:
            continue
        if isinstance(line, Header):
            header = line
        elif header is not None:
            entries.append(line)
    if header is None:
        raise SessionError(f"{path} is not a session file")
    return header, entries


def sessions_dir(home: Path, cwd: str) -> Path:
    """Where sessions for `cwd` live: one directory per working directory."""
    return home / "sessions" / re.sub(r"[\\/:]+", "-", cwd).strip("-")


def find_session(home: Path, session_id: str) -> Path:
    """The file of a session id, in any working directory's folder."""
    if re.fullmatch(r"[0-9a-f]+", session_id):
        for path in (home / "sessions").glob(f"*/{session_id}.jsonl"):
            return path
    raise SessionError(f"Session {session_id} not found")


def list_sessions(directory: Path) -> list[SessionSummary]:
    """Sessions in `directory`, most recently updated first."""
    summaries: list[SessionSummary] = []
    for path in directory.glob("*.jsonl"):
        try:
            header, entries = _read(path)
        except SessionError:
            continue
        messages = [entry.message for entry in entries if isinstance(entry, MessageEntry)]
        names = [entry.name for entry in entries if isinstance(entry, SessionInfoEntry)]
        first_user = next((m for m in messages if isinstance(m, UserMessage)), None)
        first_text = "".join(b.text for b in first_user.content if isinstance(b, TextContent)) if first_user else None
        summaries.append(
            SessionSummary(
                session_id=header.id,
                cwd=header.cwd,
                updated_at=int(os.path.getmtime(path) * 1000),
                message_count=len(messages),
                name=names[-1] if names else None,
                first_message=first_text,
            )
        )
    return sorted(summaries, key=lambda s: s.updated_at, reverse=True)
