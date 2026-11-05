"""Run the real Chat JS against fake DOM, search HTTP, and model SSE streams."""

from pathlib import Path
import shutil
import subprocess

import pytest

from ghc_api.app import create_app


@pytest.mark.skipif(not shutil.which("node"), reason="Node.js is required for Chat JS checks")
def test_search_selection_execution_and_three_protocol_loops():
    with create_app().test_client() as client:
        html = client.get("/chat").get_data(as_text=True)
    script = html.rsplit("<script>", 1)[-1].split("</script>", 1)[0]
    checked = subprocess.run(["node", "--check"], input=script, text=True, capture_output=True)
    assert checked.returncode == 0, checked.stderr
    result = subprocess.run(
        ["node", str(Path(__file__).parent / "fixtures" / "chat_search_ui.js")],
        input=script, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
