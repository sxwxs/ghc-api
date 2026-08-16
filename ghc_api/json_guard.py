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

The scan must stay linear in the body size. It runs before authentication can
help (``state.enable_auth`` is off by default, and unprotected paths accept JSON
too), so any super-linear behaviour here is a cheaper denial of service than the
``RecursionError`` this guard exists to prevent. That rules out matching string
literals with a regex: every ``"`` that opens an unterminated literal costs a
full-length failed match, and re-scanning from the next byte makes the whole
thing quadratic. Instead the structural bytes are visited once, in order, with
the in-string flag carried explicitly.
"""

import re

# Orders of magnitude above any real request (tool JSON Schemas nest a dozen
# levels at most) and comfortably under every recursive consumer above.
MAX_JSON_NESTING_DEPTH = 100

# Only these bytes can change the depth or the in-string state. Everything else
# -- including every byte of a multi-byte UTF-8 sequence, which is never
# structural -- is skipped by the regex engine in C.
_STRUCTURAL_RE = re.compile(rb'["\[\]{}]')

_QUOTE = 0x22
_BACKSLASH = 0x5C
_OPENERS = (0x5B, 0x7B)  # [ {


def exceeds_max_nesting(raw, limit: int = MAX_JSON_NESTING_DEPTH) -> bool:
    """Return True when ``raw`` nests containers deeper than ``limit``.

    ``raw`` is an undecoded request body (``bytes``, or ``str`` for callers that
    already decoded it). Brackets inside string literals are not structure, so
    the scan tracks whether it is inside one.

    An unterminated literal makes everything after it look like string content
    and the reported depth stops growing. That is safe rather than merely
    tolerable: the decoder reads left to right, so it can only recurse into
    containers that open *before* the unterminated literal -- exactly the ones
    already counted -- and it then rejects the body outright.
    """
    if isinstance(raw, str):
        raw = raw.encode("utf-8", errors="surrogatepass")

    depth = 0
    in_string = False
    for match in _STRUCTURAL_RE.finditer(raw):
        index = match.start()
        byte = raw[index]
        if byte == _QUOTE:
            if not in_string:
                in_string = True
                continue
            # A closing quote, unless an odd number of backslashes escapes it.
            # Each run of backslashes is consumed by at most one quote, so the
            # walk back stays linear over the whole body.
            probe = index - 1
            backslashes = 0
            while probe >= 0 and raw[probe] == _BACKSLASH:
                backslashes += 1
                probe -= 1
            if backslashes % 2 == 0:
                in_string = False
        elif in_string:
            continue
        elif byte in _OPENERS:
            depth += 1
            if depth > limit:
                return True
        else:
            depth -= 1
    return False
