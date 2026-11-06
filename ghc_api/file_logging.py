"""Synchronous JSONL appends serialized within a single server process."""

import threading
from os import PathLike
from typing import Union


_append_lock = threading.Lock()


def append_jsonl_line(path: Union[str, PathLike], line: str) -> None:
    """Append an already-serialized record, including its trailing newline.

    Callers serialize outside the lock and keep their own error handling.
    Hold the lock through close: buffered output may be flushed there, not
    just in write(). This does not coordinate writes from other processes.
    """
    with _append_lock:
        with open(path, "a", encoding="utf-8") as file:
            file.write(line)
