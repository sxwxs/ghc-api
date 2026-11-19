"""Bounded JSON SSE decoding shared by Chat Completions and native Gemini."""

import codecs
import json

from .keepalive import KEEPALIVE


DONE = object()


def json_events(chunks):
    buffer = ""
    data = []
    data_size = 0
    decoder = codecs.getincrementaldecoder("utf-8")()
    for chunk in chunks:
        if chunk is KEEPALIVE:
            yield KEEPALIVE
            continue
        buffer += decoder.decode(chunk) if isinstance(chunk, bytes) else chunk
        if len(buffer) + data_size > 2 * 1024 * 1024:
            raise ValueError("Upstream SSE event exceeds 2 MiB")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line.startswith(":"):
                yield KEEPALIVE
            elif line.startswith("data:"):
                value = line[5:]
                if value.startswith(" "):
                    value = value[1:]
                data.append(value)
                data_size += len(value)
            elif not line and data:
                value = "\n".join(data)
                data = []
                data_size = 0
                if value == "[DONE]":
                    yield DONE
                    return
                event = json.loads(value)
                if not isinstance(event, dict):
                    raise ValueError("Upstream SSE JSON must be an object")
                yield event
    buffer += decoder.decode(b"", final=True)
    if data or (buffer.strip() and not buffer.strip().startswith(":")):
        raise ValueError("Incomplete upstream SSE event")
