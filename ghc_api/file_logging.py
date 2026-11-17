"""Synchronous JSONL appends serialized within a single server process."""

import threading
from os import PathLike
from typing import Tuple, Union


_append_lock = threading.Lock()


def append_jsonl_line(path: Union[str, PathLike], line: str) -> Tuple[int, int]:
    """Append an already-serialized record, including its trailing newline.

    Returns ``(offset, length)`` of the written bytes so callers can locate the
    record again. The file is written in binary mode, so the offset is a byte
    offset and line endings are ``\\n`` on every OS.

    Callers serialize outside the lock and keep their own error handling.
    Hold the lock through close: buffered output may be flushed there, not
    just in write(). This does not coordinate writes from other processes.
    """
    data = line.encode("utf-8")
    with _append_lock:
        with open(path, "ab") as file:
            offset = file.tell()
            file.write(data)
    return offset, len(data)
