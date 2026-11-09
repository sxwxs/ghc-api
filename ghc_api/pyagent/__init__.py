"""pyagent: a lightweight coding agent with a JSON-RPC protocol for user interfaces.

Self-contained: nothing here imports the rest of ghc_api (tests/pyagent/test_independence.py checks it).
ghc_api builds a setup and a protocol server, then talks to it in protocol messages only (ghc_api/agent_bridge.py).
"""

__version__ = "0.1.0"
