"""Built-in coding tools."""

from typing import Any

from ...agent.tool import Tool
from .bash import BashTool
from .edit import EditTool
from .read import ReadTool
from .write import WriteTool


def default_tools() -> list[Tool[Any]]:
    """The tools enabled by default: read, the shell, edit, write."""
    return [ReadTool(), BashTool(), EditTool(), WriteTool()]
