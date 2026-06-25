"""Global, process-wide activity counters.

A tiny thread-safe key->int store mirroring the existing global-singleton
pattern (``state``, ``cache``). Used to track SSE keepalive pings sent to
clients, pings received from the Copilot upstream, and how often each
conditional request/response modification the proxy performs actually fires.

Counters are in-memory only and reset on process restart; they back the
dashboard's "Proxy Activity" panel.
"""

import threading
from collections import defaultdict
from typing import Dict


class Counters:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._c: Dict[str, int] = defaultdict(int)

    def incr(self, key: str, n: int = 1) -> None:
        with self._lock:
            self._c[key] += n

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._c)

    def reset(self) -> None:
        with self._lock:
            self._c.clear()


# Global counters instance.
counters = Counters()
