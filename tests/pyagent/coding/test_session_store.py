"""Session files: lazy creation, round trips, branches, forks, damaged lines, and listing."""

from pathlib import Path

import pytest

from ghc_api.pyagent.ai.types import AssistantMessage, TextContent, user_message
from ghc_api.pyagent.coding.session_store import (
    MessageEntry,
    SessionError,
    SessionStore,
    find_session,
    list_sessions,
    sessions_dir,
)


def reply(text: str) -> AssistantMessage:
    return AssistantMessage(content=[TextContent(text=text)], provider="p", model="m", stop_reason="stop")


def test_the_file_appears_with_the_first_message(tmp_path: Path) -> None:
    store = SessionStore.create("/work", tmp_path)
    store.append_model_change("p", "m")
    store.append_thinking_change("high")
    assert store.path is not None and not store.path.exists()
    store.append_message(user_message("hi"))
    store.close()
    lines = store.path.read_bytes().splitlines()
    assert [line.split(b'"type":"', 1)[1].split(b'"', 1)[0] for line in lines] == [
        b"session",
        b"model_change",
        b"thinking_change",
        b"message",
    ]


def test_reopening_restores_the_branch_and_settings(tmp_path: Path) -> None:
    store = SessionStore.create("/work", tmp_path)
    store.append_model_change("p", "m")
    first = user_message("hi")
    store.append_message(first)
    store.append_message(reply("hello"))
    store.append_session_info("greeting")
    store.append_thinking_change("low")
    store.close()

    assert store.path is not None
    reopened = SessionStore.open(store.path)
    context = reopened.context()
    assert [m.id for m in context.messages] == [first.id, store.branch()[2].id]
    assert (context.provider, context.model_id, context.thinking, context.name) == ("p", "m", "low", "greeting")
    assert reopened.leaf == store.leaf


def test_entries_chain_through_parent_ids_and_forks_copy_one_branch(tmp_path: Path) -> None:
    store = SessionStore.create("/work", tmp_path)
    question = user_message("q1")
    answer = reply("a1")
    store.append_message(question)
    store.append_message(answer)
    store.append_message(user_message("q2"))
    branch = store.branch()
    assert [e.parent_id for e in branch] == [None, question.id, answer.id]

    fork = store.fork(answer.id, tmp_path)
    assert fork.path is not None and fork.path.exists()
    assert fork.header.parent_session == str(store.path)
    reopened = SessionStore.open(fork.path)
    assert [m.id for m in reopened.context().messages] == [question.id, answer.id]
    assert reopened.leaf == answer.id


def test_damaged_lines_are_skipped(tmp_path: Path) -> None:
    store = SessionStore.create("/work", tmp_path)
    store.append_message(user_message("kept"))
    store.close()
    assert store.path is not None
    with store.path.open("ab") as file:
        file.write(b'{"type":"message","id":"x","parentId"')  # a crash in the middle of a write
    reopened = SessionStore.open(store.path)
    assert len(reopened.context().messages) == 1


def test_memory_only_sessions_write_nothing(tmp_path: Path) -> None:
    store = SessionStore.create("/work", None)
    store.append_message(user_message("hi"))
    assert store.path is None and len(store.branch()) == 1
    assert list(tmp_path.iterdir()) == []


def test_find_and_list_sessions(tmp_path: Path) -> None:
    directory = sessions_dir(tmp_path, "C:\\src\\project")
    assert directory == tmp_path / "sessions" / "C-src-project"
    store = SessionStore.create("C:\\src\\project", directory)
    store.append_message(user_message("first question"))
    store.append_message(reply("answer"))
    store.append_session_info("named")
    store.close()
    assert find_session(tmp_path, store.id) == store.path
    [summary] = list_sessions(directory)
    assert (summary.session_id, summary.message_count, summary.name, summary.first_message) == (
        store.id,
        2,
        "named",
        "first question",
    )
    assert isinstance(store.branch()[0], MessageEntry)
    with pytest.raises(SessionError):
        find_session(tmp_path, "../../etc")
