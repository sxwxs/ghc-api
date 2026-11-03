"""Exercise Image mode against a fake DOM, IndexedDB and fetch; no paid calls."""
import shutil
import subprocess
from pathlib import Path

import pytest

from ghc_api.app import create_app


@pytest.mark.skipif(not shutil.which("node"), reason="Node.js is required for Chat UI checks")
def test_image_mode_generation_editing_storage_and_errors():
    with create_app().test_client() as client:
        response = client.get("/chat")
    assert response.status_code == 200
    script = response.get_data(as_text=True).rsplit("<script>", 1)[-1].split("</script>", 1)[0]
    result = subprocess.run(["node", "--check"], input=script, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        ["node", str(Path(__file__).parent / "fixtures/chat_image_ui.js")],
        input=script, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
