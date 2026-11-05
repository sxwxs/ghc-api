"""Real-browser visibility regression: fake-DOM JS tests cannot see CSS hiding controls.

All fetches are stubbed before Chat runs. No model/search services are contacted,
and Chromium uses a disposable profile, never the operator's browser profile.
"""

from html.parser import HTMLParser
import json
import shutil
import subprocess
from unittest.mock import patch

import pytest

from ghc_api.app import create_app
from ghc_api.state import state

CHROME = next((path for name in ("google-chrome", "chromium", "chromium-browser")
               if (path := shutil.which(name))), None)

BROWSER_CHECK = r"""
<script>
window.fetch = async url => {
    if (url === '/v1/models' || url === '/v1/models/full/') {
        return {ok:true, json:async () => ({data:[{
            id:'gpt-layout-test', supported_endpoints:['/responses'],
        }]})};
    }
    if (url === '/proxy/models') return {ok:true, json:async () => ({data:[]})};
    throw new Error('Unexpected network call: ' + url);
};
window.addEventListener('load', () => setTimeout(() => {
    const result = {};
    try {
        const check = (condition, message) => { if (!condition) throw new Error(message); };
        const search = document.getElementById('search-provider');
        const endpoint = document.getElementById('endpoint-select');
        const visible = element => getComputedStyle(element).display !== 'none' &&
            element.getBoundingClientRect().width > 0 && element.getBoundingClientRect().height > 0;
        check(visible(search), 'Search selector is hidden in Chat mode');
        check(!search.disabled, 'Search selector is disabled despite a configured provider and model');
        check(!visible(endpoint), 'Raw endpoint selector must remain hidden in Chat mode');
        check([...search.options].map(o => o.value).join(',') === 'off,native,webiq,anysearch',
            'Search provider options missing');
        const label = document.querySelector('label[for="search-provider"]');
        label.click();
        check(document.activeElement === search, 'Search label does not focus the visible selector');
        search.value = 'anysearch';
        search.dispatchEvent(new Event('change', {bubbles:true}));
        check(localStorage.getItem('ghc_chat_search_provider') === 'anysearch', 'Selection was not saved');
        check(searchTools('gpt-layout-test', 'responses')[0].name === 'anysearch_search',
            'Selected provider did not control the declared tool');
        setRawMode(true);
        check(visible(search) && search.disabled, 'Search must remain visible but disabled in Raw mode');
        check(visible(endpoint), 'Endpoint selector must be visible in Raw mode');
        setRawMode(false);
        check(visible(search) && !search.disabled, 'Search did not recover when returning to Chat mode');
        check(!visible(endpoint), 'Endpoint selector remained visible outside Raw mode');
        check(search.value === 'anysearch', 'Mode switch changed the selected provider');
        result.ok = true;
    } catch (error) {
        result.ok = false;
        result.error = error.message;
    }
    document.body.setAttribute('data-search-layout-result', JSON.stringify(result));
}, 0));
</script>
"""


class ResultParser(HTMLParser):
    result = None

    def handle_starttag(self, tag, attrs):
        if tag == "body":
            value = dict(attrs).get("data-search-layout-result")
            if value:
                self.result = json.loads(value)


@pytest.mark.skipif(CHROME is None, reason="Chrome/Chromium is required for real CSS layout checks")
def test_search_selector_visible_and_interactive_in_real_browser(tmp_path):
    with patch.multiple(state, enable_anysearch=True, anysearch_api_key="fake-key",
                        enable_webiq_search=True, webiq_api_key="fake-key"):
        with create_app().test_client() as client:
            html = client.get("/chat").get_data(as_text=True)
    # Install the network stubs before the page's own script and remove the
    # favicon request. file:// keeps the fixture independent of any live server.
    html = html.replace('<link rel="icon"', '<link rel="unused-icon"', 1)
    html = html.replace("<script>", BROWSER_CHECK + "<script>", 1)
    fixture = tmp_path / "chat.html"
    fixture.write_text(html, encoding="utf-8")
    profile = tmp_path / "chrome-profile"
    result = subprocess.run([
        CHROME, "--headless", "--no-sandbox", "--disable-gpu",
        "--disable-background-networking", "--disable-component-update", "--disable-sync",
        "--no-first-run", "--no-default-browser-check", "--disable-extensions",
        f"--user-data-dir={profile}", "--virtual-time-budget=1500", "--dump-dom", fixture.as_uri(),
    ], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr[-2000:]
    parsed = ResultParser()
    parsed.feed(result.stdout)
    assert parsed.result is not None, "Browser checks did not finish: " + result.stderr[-1000:]
    assert parsed.result.get("ok"), parsed.result
