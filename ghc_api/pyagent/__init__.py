"""pyagent: a lightweight coding agent with a JSON-RPC protocol for user interfaces.

Self-contained: nothing here imports the rest of ghc_api (tests/pyagent/test_independence.py checks it).
ghc_api reaches pyagent through its protocol only; see docs/pyagent/design.md.
"""

__version__ = "0.1.0"
