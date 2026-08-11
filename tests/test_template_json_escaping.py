"""The JSON viewers write into innerHTML, so they must escape first.

Request and response bodies contain arbitrary third-party text: Web IQ search
results (titles/content copied from indexed pages) and model output. Without
escaping, `<img src=x onerror=...>` in such a body executes on the dashboard
origin, which also serves the admin APIs.
"""

import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# template -> highlighter function name
HIGHLIGHTERS = {
    "ghc_api/templates/dashboard.html": "syntaxHighlight",
    "ghc_api/templates/chat.html": "syntaxHighlightJson",
    "ghc_api/templates/requests.html": "syntaxHighlight",
}


class TemplateJsonEscapingTest(unittest.TestCase):
    def test_highlighters_escape_before_building_html(self):
        for template, name in HIGHLIGHTERS.items():
            source = (ROOT / template).read_text(encoding="utf-8")
            match = re.search(
                rf"function {name}\(json\)[\s\S]*?\n        \}}", source
            )
            self.assertIsNotNone(match, f"{name}() not found in {template}")
            body = match.group(0)

            escape_at = body.find("'&lt;'")
            # The line that wraps a matched token: '<span class="' + cls + ...
            span_at = body.find("'<span class=\"' +")
            self.assertNotEqual(escape_at, -1, f"{template}: {name}() does not escape '<'")
            self.assertNotEqual(span_at, -1, f"{template}: {name}() token wrapper not found")
            self.assertLess(
                escape_at, span_at,
                f"{template}: {name}() must escape before wrapping tokens in spans",
            )


if __name__ == "__main__":
    unittest.main()
