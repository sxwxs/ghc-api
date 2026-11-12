"""Command-line options, and how they turn into a session setup and an open session."""

import argparse
import os

from .. import __version__
from ..ai.models import merge_providers
from ..client.client import Client
from ..coding.agent_session import SessionSetup
from ..coding.config import home_dir, load_config
from ..coding.session_events import SessionState


def parser() -> argparse.ArgumentParser:
    """The pyagent argument parser."""
    result = argparse.ArgumentParser(prog="pyagent", description="A lightweight coding agent.")
    result.add_argument("prompt", nargs="*", help="run once with this prompt; without one, start the REPL")
    result.add_argument("-p", "--print", action="store_true", help="run once (reads the prompt from stdin if omitted)")
    result.add_argument("--mode", choices=["text", "json", "rpc"], default="text", help="output, or serve the protocol")
    result.add_argument(
        "-c", "--continue", dest="resume_latest", action="store_true", help="continue the latest session"
    )
    result.add_argument("--session", help="resume the session with this id")
    result.add_argument("--no-session", action="store_true", help="do not save the session")
    result.add_argument("--provider", help="provider name, e.g. openai, anthropic, deepseek, or one from config.json")
    result.add_argument("--model", help="model id; defaults to the provider's default model")
    result.add_argument("--thinking", choices=["off", "low", "medium", "high", "max"], help="reasoning level")
    result.add_argument("--cwd", help="working directory (default: current directory)")
    result.add_argument("--version", action="version", version=f"pyagent {__version__}")
    return result


def setup(args: argparse.Namespace) -> tuple[SessionSetup, str]:
    """The session setup and working directory the options select."""
    home = home_dir()
    config = load_config(home)
    cwd = os.path.abspath(args.cwd or os.getcwd())
    return SessionSetup(
        home=home, config=config, providers=merge_providers(config.providers), persist=not args.no_session
    ), cwd


async def open_session(client: Client, args: argparse.Namespace, cwd: str) -> SessionState:
    """Resume (--session, --continue) or create a session; model flags also apply to a resumed one."""
    session_id: str | None = args.session
    if session_id is None and args.resume_latest:
        sessions = await client.list_sessions(cwd)
        session_id = sessions[0].session_id if sessions else None
    if session_id is None:
        return await client.new_session(cwd, args.provider, args.model, args.thinking)
    state = await client.load_session(session_id)
    if args.provider or args.model:
        state = await client.set_model(state.session_id, args.provider or state.model.provider, args.model)
    if args.thinking:
        state = await client.set_thinking(state.session_id, args.thinking)
    return state
