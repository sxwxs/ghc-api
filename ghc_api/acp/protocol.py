"""JSON-RPC 2.0 Protocol over subprocess stdio."""

import asyncio
import json
import logging
from typing import Any, Callable, Coroutine, Dict, Optional

logger = logging.getLogger(__name__)


class JsonRpcProtocol:
    """Bidirectional JSON-RPC 2.0 over stdin/stdout of a subprocess."""

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        client_handler: Optional[Any] = None,
    ):
        self.process = process
        self.client_handler = client_handler
        self._next_id = 1
        self._pending = {}  # type: Dict[int, asyncio.Future]
        self._notification_callbacks = {}  # type: Dict[str, Callable]
        self._reader_task = None  # type: Optional[asyncio.Task]
        self._send_lock = asyncio.Lock()  # protects concurrent writes to stdin

    def start(self):
        """Start the background reader that dispatches incoming messages."""
        self._reader_task = asyncio.ensure_future(self._read_loop())

    async def _read_loop(self):
        """Continuously read JSON-RPC messages from agent's stdout."""
        assert self.process.stdout is not None
        msg_count = 0
        while True:
            try:
                line = await self.process.stdout.readline()
            except Exception as e:
                print(f"[ACP] read_loop: exception reading stdout: {e}")
                break
            if not line:
                rc = self.process.returncode
                print(f"[ACP] read_loop: stdout closed (process returncode={rc}, msgs read={msg_count})")
                # Resolve any pending futures with error
                for fid, fut in list(self._pending.items()):
                    if not fut.done():
                        fut.set_exception(RuntimeError("Agent process exited"))
                break
            msg_count += 1
            try:
                msg = json.loads(line.decode("utf-8").strip())
            except (json.JSONDecodeError, UnicodeDecodeError):
                print(f"[ACP] read_loop: non-JSON line ({len(line)} bytes): {line[:100]!r}")
                continue

            # Protect the read loop — _dispatch must never crash this loop
            try:
                await self._dispatch(msg)
            except Exception as e:
                print(f"[ACP] read_loop: dispatch error (msg #{msg_count}): {e}")

        print(f"[ACP] read_loop: EXITED after {msg_count} messages")

    async def _dispatch(self, msg: Dict):
        """Route an incoming message to the right handler."""
        if "id" in msg and "method" in msg:
            # Agent -> Client REQUEST (agent calling us)
            # Handle in a separate task so the read loop is not blocked
            asyncio.ensure_future(self._handle_agent_request(msg))
        elif "id" in msg and "result" in msg:
            # Response to our request
            req_id = msg["id"]
            fut = self._pending.get(req_id)
            if fut and not fut.done():
                fut.set_result(msg["result"])
        elif "id" in msg and "error" in msg:
            # Error response to our request
            req_id = msg["id"]
            fut = self._pending.get(req_id)
            if fut and not fut.done():
                error = msg["error"]
                code = error.get("code", -1)
                message = error.get("message", "Unknown error")
                data = error.get("data")
                fut.set_exception(JsonRpcError(code, message, data))
        elif "method" in msg and "id" not in msg:
            # Agent -> Client NOTIFICATION
            method = msg["method"]
            params = msg.get("params", {})
            cb = self._notification_callbacks.get(method)
            if cb:
                try:
                    result = cb(params)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    logger.exception("Notification callback error for %s", method)

    async def _handle_agent_request(self, msg: Dict):
        """Handle a request from the agent (file read/write, permissions, etc.)."""
        method = msg["method"]
        params = msg.get("params", {})
        req_id = msg["id"]
        print(f"[ACP] agent request: {method} (id={req_id})")

        if self.client_handler is None:
            await self._send_error(req_id, -32601, "Method not found")
            return

        try:
            result = await self.client_handler.handle(method, params)
            await self._send({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": result,
            })
        except NotImplementedError:
            await self._send_error(req_id, -32601, f"Method not supported: {method}")
        except Exception as e:
            logger.exception("Error handling agent request %s", method)
            try:
                await self._send_error(req_id, -32603, str(e))
            except Exception:
                logger.warning("Failed to send error response for %s", method)

    async def request(self, method: str, params: Dict, timeout: float = None) -> Any:
        """Send a JSON-RPC request and wait for the response.
        timeout=None means wait indefinitely (used for session/prompt)."""
        req_id = self._next_id
        self._next_id += 1

        loop = asyncio.get_event_loop()
        future = loop.create_future()  # type: asyncio.Future
        self._pending[req_id] = future

        await self._send({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        })

        try:
            if timeout is not None:
                return await asyncio.wait_for(future, timeout=timeout)
            else:
                return await future
        finally:
            self._pending.pop(req_id, None)

    async def notify(self, method: str, params: Dict):
        """Send a JSON-RPC notification (no response expected)."""
        await self._send({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        })

    def on_notification(self, method: str, callback: Callable):
        """Register a handler for incoming notifications."""
        self._notification_callbacks[method] = callback

    async def _send(self, msg: Dict):
        """Write a JSON-RPC message to the agent's stdin.
        Uses a lock to prevent concurrent writes from corrupting messages."""
        assert self.process.stdin is not None
        data = json.dumps(msg, ensure_ascii=False) + "\n"
        async with self._send_lock:
            self.process.stdin.write(data.encode("utf-8"))
            await self.process.stdin.drain()

    async def _send_error(self, req_id: int, code: int, message: str):
        await self._send({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        })

    async def close(self):
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        if self.process.stdin:
            try:
                self.process.stdin.close()
            except Exception:
                pass


class JsonRpcError(Exception):
    """JSON-RPC error with code and message."""

    METHOD_NOT_FOUND = -32601
    AUTH_REQUIRED = -32000
    RESOURCE_NOT_FOUND = -32001

    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        detail = f"JSON-RPC error {code}: {message}"
        if data:
            detail += f" | data: {data}"
        super().__init__(detail)
