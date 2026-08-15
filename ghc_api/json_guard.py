"""Nesting guard for client-supplied JSON bodies.

Every endpoint that parses a request body hands the result to something
recursive -- ``copy.deepcopy`` for the request cache, the Anthropic <-> Responses
translation, ``json.dumps`` on the way upstream. Those give out around 500
levels of nesting, while the JSON decoder itself happily accepts thousands
(CPython's C scanner only raises ``RecursionError`` near 10k levels on 3.12, and
the exact point moves between versions and builds). Without a guard a small
hostile body -- a few KB of ``[[[[...`` -- turns into an unhandled
``RecursionError`` and a 500.

The limit is checked on the raw bytes before anything recursive touches the
value, so it is deterministic and independent of the interpreter's stack.
"""

import re

# Orders of magnitude above any real request (tool JSON Schemas nest a dozen
# levels at most) and comfortably under every recursive consumer above.
MAX_JSON_NESTING_DEPTH = 100

# One pass that swallows a whole string literal (unrolled loop form, which is
# markedly faster than the naive alternation) or a run of bytes that carries no
# structure, leaving only the brackets to count. On a 4.6 MB body this costs
# ~20 ms, against ~90 ms for the readable two-regex version. Matching bytes
# rather than text keeps multi-byte UTF-8 out of the way for free: its bytes are
# never structural.
_STRUCTURE_ONLY_RE = re.compile(rb'"[^"\\]*(?:\\.[^"\\]*)*"|[^\[\]{}"]+', re.S)


def exceeds_max_nesting(raw, limit: int = MAX_JSON_NESTING_DEPTH) -> bool:
    """Return True when ``raw`` nests containers deeper than ``limit``.

    ``raw`` is an undecoded request body (``bytes``, or ``str`` for callers that
    already decoded it). String literals are stripped first so brackets inside
    them do not count as structure; only the remaining brackets are scanned.
    """
    if isinstance(raw, str):
        raw = raw.encode("utf-8", errors="surrogatepass")

    structure = _STRUCTURE_ONLY_RE.sub(b"", raw)
    depth = 0
    for byte in structure:
        if byte in b"[{":
            depth += 1
            if depth > limit:
                return True
        elif byte in b"]}":
            depth -= 1
        # A leftover quote only survives for an unterminated string, which the
        # decoder rejects a moment later; it must not disturb the depth count.
    return False
