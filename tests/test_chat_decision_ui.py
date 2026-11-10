"""Browser-free checks for the Chat page Decision mode (System One and OpenAI Decisions models).

The fake fetch never contacts a model service. Skip when Node.js is unavailable.
"""
import shutil
import subprocess
import unittest

from ghc_api.app import create_app


NODE_CHECK = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const script = fs.readFileSync(0, 'utf8').split('        // ---- Input handling ----')[0];

const escapeText = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
function element() {
    const classes = new Set();
    const el = {
        children: [], value: '', placeholder: '', disabled: false, options: [], style: {},
        classList: {
            toggle(name, on) { (on === undefined ? !classes.has(name) : on) ? classes.add(name) : classes.delete(name); },
            add(name) { classes.add(name); },
            remove(name) { classes.delete(name); },
            contains(name) { return classes.has(name); },
        },
        appendChild(child) { this.children.push(child); return child; },
        insertBefore(child) { this.children.push(child); return child; },
        querySelector() { return null; },
        remove() {},
        focus() {},
    };
    Object.defineProperty(el, 'innerHTML', {
        get() { return this._innerHTML || ''; },
        set(value) { this._innerHTML = value; if (value === '') this.children = []; },
    });
    // escapeHtml() relies on textContent -> innerHTML escaping.
    Object.defineProperty(el, 'textContent', {
        get() { return this._text || ''; },
        set(value) { this._text = String(value ?? ''); this._innerHTML = escapeText(this._text); },
    });
    return el;
}
const elements = {};
const document = {
    getElementById(name) { return elements[name] || (elements[name] = element()); },
    createElement() { return element(); },
};
document.getElementById('search-provider').options =
    ['off', 'native', 'webiq', 'anysearch'].map(value => ({value}));
document.getElementById('search-provider').value = 'off';
const saved = {};
const localStorage = {
    getItem(key) { return saved[key] || null; },
    setItem(key, value) { saved[key] = value; },
};

const DECISION_RESPONSE = {
    model: 'decision-icebox-9b',
    answers: {
        team: {type: 'choice', choice: 'engineering', confidence: 0.9978,
               probabilities: {billing: 0.0022, engineering: 0.9978, '<img src=x>"': 0}},
        urgent: {type: 'noul', noul: 0.9466},
    },
    usage: {input_tokens: 72, output_tokens: 2},
};
let nextResponse = {status: 200, body: JSON.stringify(DECISION_RESPONSE)};
const calls = [];
async function fetch(url, options) {
    if (url === '/v1/models') {
        return { ok: true, json: async () => ({ data: [
            {id: 'gpt-chat', supported_endpoints: ['/responses']},
            {id: 'microsoft-decision-1', display_name: 'Microsoft-Decision-1',
             model_type: 'decisions', supported_endpoints: ['/v1/systemone']},
            {id: 'decision-typed', model_type: 'decisions', supported_endpoints: []},
            {id: 'gpt-6-luna-decisions', display_name: 'GPT-6 Luna Decisions',
             model_type: 'decisions', supported_endpoints: ['/v1/decisions']},
            {id: 'gpt-image', model_type: 'image', supported_endpoints: ['/v1/images/generations']},
        ] }) };
    }
    if (url === '/proxy/models') {
        return { ok: true, json: async () => ({ data: [
            {id: 'proxy-decision', profile: 'dec', base_url: '/proxy/dec/v1',
             supported_endpoints: ['/systemone']},
        ] }) };
    }
    calls.push({ url, options });
    const response = nextResponse;
    return { ok: response.status < 400, status: response.status, text: async () => response.body };
}
const sandbox = {
    document, localStorage, fetch, crypto: require('node:crypto').webcrypto,
    TextDecoder, performance, console, AbortController,
    autoResize() {},  // defined after the "Input handling" cut
};
vm.createContext(sandbox);
vm.runInContext(script, sandbox);
const run = code => vm.runInContext(code, sandbox);
const container = document.getElementById('messages-container');
const lastMessage = () => container.children.at(-1);

(async () => {
    await run('loadModels()');

    // Decision models are listed and labelled; proxy systemone models too.
    const proxyKey = run("configuredModelKey('dec', 'proxy-decision')");
    const keys = run('Object.keys(modelTargets)');
    for (const key of ['gpt-chat', 'microsoft-decision-1', 'decision-typed', 'gpt-6-luna-decisions', proxyKey]) {
        assert.ok(keys.includes(key), key + ' missing from the model list');
    }
    const labels = elements['model-select'].children.flatMap(group => group.children)
        .map(option => option.textContent);
    assert.ok(labels.some(l => l.startsWith('Microsoft-Decision-1') && l.includes('[decision]')));
    assert.ok(labels.some(l => l.startsWith('GPT-6 Luna Decisions') && l.includes('[decision]')));
    assert.ok(labels.some(l => l.startsWith('gpt-chat') && !l.includes('[decision]')));
    assert.equal(run("isDecisionOnly('microsoft-decision-1')"), true);
    assert.equal(run("isDecisionOnly('decision-typed')"), true);
    assert.equal(run("isDecisionOnly('gpt-6-luna-decisions')"), true);
    assert.equal(run("endpointForModel('gpt-6-luna-decisions', 'decisions')"), '/v1/decisions');
    assert.equal(run("isDecisionOnly('gpt-chat')"), false);
    assert.equal(run("endpointForModel('microsoft-decision-1', 'systemone')"), '/v1/systemone');
    assert.equal(run(`endpointForModel(${JSON.stringify(proxyKey)}, 'systemone')`), '/proxy/dec/v1/systemone');

    // Selecting a decision-only model in Chat mode switches to Decision mode
    // and offers the request template.
    elements['model-select'].value = 'microsoft-decision-1';
    run('onModelChange()');
    assert.equal(run('chatMode'), 'decision');
    assert.ok(elements['mode-decision'].classList.contains('active'));
    assert.ok(!elements['mode-chat'].classList.contains('active'));
    assert.equal(elements['chat-input'].value, run('SYSTEMONE_DECISION_TEMPLATE'));
    assert.equal(elements['search-provider'].disabled, true);

    // Raw mode routes a decision-only model to System One automatically.
    run("setMode('raw')");
    assert.equal(elements['chat-input'].value, '', 'untouched template must not leak into Raw mode');
    assert.equal(run('resolveEndpoint({})'), '/v1/systemone');
    assert.ok(elements['endpoint-select'].children.some(o => o.value === '/v1/systemone'));

    // Chat mode refuses a decision-only model without calling any API.
    run("setMode('chat')");
    elements['chat-input'].value = 'hello';
    await run('sendMessage()');
    assert.equal(calls.length, 0);
    assert.match(lastMessage().children[0].textContent, /Decision mode/);

    // Decision mode sends the request with the model injected and renders answers.
    run("setMode('decision')");
    elements['chat-input'].value = JSON.stringify({
        state: 'The payment callback API returns 500 on every call.',
        questions: {
            team: {type: 'choice', instructions: 'Which team?', criteria: {billing: 'x', engineering: 'y'}},
            urgent: {type: 'noul', instructions: 'Urgent?'},
        },
    });
    await run('sendMessage()');
    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, '/v1/systemone');
    const sent = JSON.parse(calls[0].options.body);
    assert.equal(sent.model, 'microsoft-decision-1');
    assert.equal(sent.state, 'The payment callback API returns 500 on every call.');
    assert.equal(sent.stream, undefined);
    assert.notEqual(elements['chat-input'].value, '', 'Decision input is kept for re-runs');

    const session = run('getActiveSession()');
    assert.equal(session.messages.map(m => m.role).join(','), 'decision-request,decision-response');
    assert.equal(session.messages[1].statusCode, 200);
    assert.match(session.title, /^\[Decision\] The payment callback/);
    assert.equal(run('conversationTurns(getActiveSession().messages).length'), 0);

    const rendered = lastMessage().innerHTML;
    assert.match(rendered, /decision-option chosen"><span class="decision-option-label" title="engineering">engineering/);
    assert.match(rendered, /99\.8%/);
    assert.match(rendered, /yes 94\.7%/);
    assert.match(rendered, /confidence 99\.8%/);
    assert.match(rendered, /tokens 72 in \/ 2 out/);
    assert.ok(!rendered.includes('<img src=x>'), 'option labels must be escaped');
    assert.ok(!rendered.includes('title="&lt;img src=x&gt;""'), 'attribute quotes must be escaped');

    // Reloading the session re-renders stored decision entries.
    run('renderMessages()');
    assert.equal(container.children.length, 2);
    assert.match(container.children[1].innerHTML, /99\.8%/);

    // Restoring a Decision session after an Image session must not enter Chat.
    const decisionSessionId = run('activeSessionId');
    run("sessions.push({id: 'image-session', model: 'gpt-image', messages: []})");
    run("setActiveSession('image-session')");
    assert.equal(run('chatMode'), 'image');
    run(`setActiveSession(${JSON.stringify(decisionSessionId)})`);
    assert.equal(run('chatMode'), 'decision');
    assert.equal(elements['model-select'].value, 'microsoft-decision-1');
    assert.ok(elements['mode-decision'].classList.contains('active'));
    assert.match(container.children[1].innerHTML, /99\.8%/);
    const beforeRestoredSend = calls.length;
    await run('sendMessage()');
    assert.equal(calls.length, beforeRestoredSend + 1);
    assert.equal(calls.at(-1).url, '/v1/systemone');

    // Explicit Raw mode remains Raw when either session is restored.
    run("setMode('raw'); setActiveSession('image-session')");
    run(`setActiveSession(${JSON.stringify(decisionSessionId)})`);
    assert.equal(run('chatMode'), 'raw');
    run("setMode('decision')");

    // Upstream validation errors are shown, not hidden behind empty cards.
    nextResponse = {status: 422, body: JSON.stringify({detail: 'At least one question is required.'})};
    elements['chat-input'].value = JSON.stringify({state: 'x', questions: {}});
    await run('sendMessage()');
    assert.match(lastMessage().innerHTML, /decision-error">At least one question is required\./);

    // A model without decision support is refused in Decision mode.
    elements['model-select'].value = 'gpt-chat';
    const before = calls.length;
    await run('sendMessage()');
    assert.equal(calls.length, before);
    assert.match(lastMessage().children[0].textContent, /does not support the decision API/);

    // OpenAI Decisions models (GPT-6 Luna Decisions) use /v1/decisions and its
    // array-based schema; an untouched template follows the selected model.
    elements['model-select'].value = 'microsoft-decision-1';
    elements['chat-input'].value = '';
    run("setMode('decision')");
    assert.equal(elements['chat-input'].value, run('SYSTEMONE_DECISION_TEMPLATE'));
    elements['model-select'].value = 'gpt-6-luna-decisions';
    run('onModelChange()');
    assert.equal(run('chatMode'), 'decision');
    assert.equal(elements['chat-input'].value, run('OPENAI_DECISION_TEMPLATE'));
    assert.equal(run('resolveEndpoint({})'), '/v1/decisions');
    const lunaEndpoints = elements['endpoint-select'].children.map(o => o.value);
    assert.ok(lunaEndpoints.includes('/v1/decisions'));
    assert.ok(!lunaEndpoints.includes('/v1/systemone'));

    nextResponse = {status: 200, body: JSON.stringify({
        model: 'gpt-6-luna',
        answers: [
            {type: 'choice', name: 'team', choice: 'billing', confidence: 0.93, probabilities: [
                {value: 'support', probability: 0.05}, {value: 'billing', probability: 0.93},
                {value: 'sales', probability: 0.02}]},
            {type: 'predicate', name: 'urgent', probability: 0.3},
            {type: 'score', name: 'severity', score: 1.1, confidence: 0.8, probabilities: [
                {value: 0, label: 'Cosmetic', probability: 0.1},
                {value: 1, label: 'Workaround', probability: 0.7},
                {value: 2, label: 'Blocked', probability: 0.2}]},
            {type: 'choice', name: 'refund', choice: true, confidence: 0.8, probabilities: [
                {value: true, probability: 0.8}, {value: false, probability: 0.2}]},
            {type: 'refusal', name: 'unsafe'},
        ],
        usage: {input_tokens: 298, input_tokens_details: {cached_tokens: 256, cache_write_tokens: 0},
                output_tokens: 0},
    })};
    await run('sendMessage()');
    assert.equal(calls.at(-1).url, '/v1/decisions');
    const lunaSent = JSON.parse(calls.at(-1).options.body);
    assert.equal(lunaSent.model, 'gpt-6-luna-decisions');
    assert.match(lunaSent.input, /charged twice/);
    assert.ok(Array.isArray(lunaSent.questions));

    const lunaRequest = container.children.at(-2).innerHTML;
    assert.match(lunaRequest, /decision-state">Customer: I was charged twice/);
    assert.match(lunaRequest, /choice<\/span> <strong>team<\/strong>/);
    assert.match(lunaRequest, /predicate<\/span> <strong>urgent<\/strong>/);

    const lunaRendered = lastMessage().innerHTML;
    assert.match(lunaRendered, /decision-option chosen"><span class="decision-option-label" title="billing">billing/);
    assert.match(lunaRendered, /billing \u00b7 confidence 93\.0%/);
    assert.match(lunaRendered, /yes 30\.0%/);
    assert.match(lunaRendered, /score 1\.10/);
    assert.match(lunaRendered, /decision-option chosen"><span class="decision-option-label" title="1 \u2014 Workaround">/);
    assert.match(lunaRendered, /decision-option chosen"><span class="decision-option-label" title="true">true/);
    assert.match(lunaRendered, /decision-id">unsafe<\/span><span class="decision-type">refusal<\/span><span class="decision-summary">refused/);
    assert.match(lunaRendered, /tokens 298 in \(256 cached\) \/ 0 out/);

    // Restoring a session for the other decision API swaps an untouched
    // template too; an edited request is kept.
    const lunaSessionId = run('activeSessionId');
    assert.equal(run('getActiveSession().model'), 'gpt-6-luna-decisions');
    run("sessions.push({id: 'systemone-session', model: 'microsoft-decision-1', messages: []})");
    elements['chat-input'].value = run('OPENAI_DECISION_TEMPLATE');
    run("setActiveSession('systemone-session')");
    assert.equal(run('chatMode'), 'decision');
    assert.equal(elements['chat-input'].value, run('SYSTEMONE_DECISION_TEMPLATE'));
    run(`setActiveSession(${JSON.stringify(lunaSessionId)})`);
    assert.equal(elements['chat-input'].value, run('OPENAI_DECISION_TEMPLATE'));
    elements['chat-input'].value = '{"input": "edited"}';
    run("setActiveSession('systemone-session')");
    assert.equal(elements['chat-input'].value, '{"input": "edited"}');
})().catch(err => { console.error(err); process.exitCode = 1; });
"""


@unittest.skipUnless(shutil.which("node"), "Node.js is required for Chat UI JS checks")
class ChatDecisionModeTest(unittest.TestCase):
    def test_decision_mode_lists_sends_and_renders_decision_models(self):
        with create_app().test_client() as client:
            response = client.get("/chat")
        self.assertEqual(response.status_code, 200)
        script = response.get_data(as_text=True).rsplit("<script>", 1)[-1].split("</script>", 1)[0]
        checked = subprocess.run(["node", "--check"], input=script, text=True, capture_output=True)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        result = subprocess.run(["node", "-e", NODE_CHECK], input=script, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
