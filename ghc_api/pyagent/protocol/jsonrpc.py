"""JSON-RPC 2.0 envelopes, error codes, and line framing, shared by the server and the client."""

from typing import Any

import msgspec

PROTOCOL_VERSION = 1

# Standard JSON-RPC codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
# pyagent codes.
NOT_INITIALIZED = -32000
UNSUPPORTED_VERSION = -32001
SESSION_NOT_FOUND = -32002
ENTRY_NOT_FOUND = -32003
MODEL_ERROR = -32004  # unknown provider or model, or missing credentials


class ErrorObject(msgspec.Struct, omit_defaults=True):
    code: int
    message: str
    data: Any = None


class Incoming(msgspec.Struct):
    """Any JSON-RPC message, decoded leniently: one decode tells requests, notifications and responses apart."""

    jsonrpc: str = ""
    id: int | str | None = None
    method: str | None = None
    params: msgspec.Raw = msgspec.Raw()  # empty when absent; Raw cannot be optional
    result: msgspec.Raw = msgspec.Raw()
    error: ErrorObject | None = None


class Request(msgspec.Struct, omit_defaults=True):
    jsonrpc: str
    id: int | str
    method: str
    params: Any = None


class Notification(msgspec.Struct, omit_defaults=True):
    jsonrpc: str
    method: str
    params: Any = None


class Success(msgspec.Struct):
    jsonrpc: str
    id: int | str
    result: Any


class Failure(msgspec.Struct):
    jsonrpc: str
    id: int | str | None
    error: ErrorObject


class ProtocolError(Exception):
    """A request failed with a protocol error code."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


INCOMING = msgspec.json.Decoder(Incoming)


def encode_line(record: Any) -> bytes:
    """Encode one record as a line. U+2028/U+2029 are escaped so readers that also split on them stay correct."""
    data = msgspec.json.encode(record).replace(b"\xe2\x80\xa8", b"\\u2028").replace(b"\xe2\x80\xa9", b"\\u2029")
    return data + b"\n"
