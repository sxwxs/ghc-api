"""HTTP plumbing for provider adapters: the shared client, error responses, and a Server-Sent Events reader."""

from collections.abc import AsyncIterator

import httpx
import msgspec

from .provider import ProviderError

_client: httpx.AsyncClient | None = None


def shared_client() -> httpx.AsyncClient:
    """Return the process-wide client, so connections are reused across requests."""
    global _client
    if _client is None:
        # Reasoning models can pause for minutes between chunks.
        _client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=600.0))
    return _client


async def raise_for_status(response: httpx.Response) -> None:
    """Raise a ProviderError carrying the provider's own error message for non-2xx responses."""
    if response.status_code < 400:
        return
    body = (await response.aread()).decode("utf-8", "replace")
    raise ProviderError(error_message(body), response.status_code)


class _ErrorDetail(msgspec.Struct):
    message: str | None = None


class _ErrorBody(msgspec.Struct):
    error: _ErrorDetail | str | None = None


def error_message(body: str) -> str:
    """Extract `error.message` (or a string `error`) from a JSON error body, else return the body shortened."""
    try:
        error = msgspec.json.decode(body, type=_ErrorBody).error
    except msgspec.DecodeError:
        return body[:2000]
    if isinstance(error, str):
        return error
    return error.message if error is not None and error.message else body[:2000]


async def sse_data(response: httpx.Response) -> AsyncIterator[str]:
    """Yield the data payload of each server-sent event; multi-line data is joined with newlines."""
    data: list[str] = []
    async for line in response.aiter_lines():
        if not line:
            if data:
                yield "\n".join(data)
                data = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if field == "data":
            data.append(value[1:] if value.startswith(" ") else value)
    if data:
        yield "\n".join(data)
