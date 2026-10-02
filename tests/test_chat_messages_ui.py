"""Browser-free checks for the configured-proxy Messages Chat path.

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
const script = fs.readFileSync(0, 'utf8').split('        // ---- Input handling ----')[0]
    .replace(/const WEBIQ_AVAILABLE = (true|false);/, 'const WEBIQ_AVAILABLE = true;');
function element() {
    const el = {
        children: [], value: '', checked: false,
        classList: { toggle() {} },
        appendChild(child) { this.children.push(child); },
        insertBefore(child) { this.children.push(child); },
        querySelector() { return null; },
        remove() {},
    };
    Object.defineProperty(el, 'innerHTML', {
        get() { return this._innerHTML || ''; },
        set(value) { this._innerHTML = value; if (value === '') this.children = []; },
    });
    return el;
}
const elements = {};
for (const name of ['model-select', 'endpoint-select', 'web-search-enabled',
    'web-search-toggle', 'webiq-search-enabled', 'webiq-search-toggle', 'messages-container']) {
    elements[name] = element();
}
const document = {
    getElementById(name) { return elements[name]; },
    createElement() { return element(); },
};
const saved = {};
const localStorage = {
    getItem(key) { return saved[key] || null; },
    setItem(key, value) { saved[key] = value; },
};
let sse = [
    'event: message_start\ndata: {"type":"message_start","message":{"content":[]}}\n\n',
    'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"OK"}}\n\n',
    'event: message_stop\ndata: {"type":"message_stop"}\n\n',
].join('');
const normalSse = sse;
let responseStreams = null;
let calls = [];
async function fetch(url, options) {
    if (url === '/v1/models' || url === '/v1/models/full/') {
        return { ok: true, json: async () => ({ data: [] }) };
    }
    if (url === '/proxy/models') {
        return { ok: true, json: async () => ({ data: [{
            id: 'example-model', profile: 'example-profile',
            base_url: '/proxy/example-profile/v1', supported_endpoints: ['/messages'],
            max_output_tokens: 2000,
            request_headers: {'/messages': [
                {name:'X-Test-Client-Context', value_type:'uuid_v4'},
                {name:'X-Test-Client-Trace', value_type:'uuid_v4'},
            ]},
        }] }) };
    }
    calls.push({ url, options });
    const bytes = new TextEncoder().encode(responseStreams ? responseStreams.shift() : sse);
    const chunks = [bytes.slice(0, 57), bytes.slice(57, 119), bytes.slice(119)];
    return {
        ok: true, status: 200,
        body: { getReader() { return { async read() {
            return chunks.length ? { done: false, value: chunks.shift() } : { done: true };
        } }; } },
    };
}
const sandbox = {
    document, localStorage, fetch, crypto: require('node:crypto').webcrypto,
    TextDecoder, performance, console,
};
vm.createContext(sandbox);
vm.runInContext(script, sandbox);
(async () => {
    await vm.runInContext('loadModels()', sandbox);
    const key = vm.runInContext("configuredModelKey('example-profile', 'example-model')", sandbox);
    elements['model-select'].value = key;
    vm.runInContext('updateEndpointOptions()', sandbox);
    assert.equal(vm.runInContext('shouldUseMessages(document.getElementById("model-select").value)', sandbox), true);
    assert.equal(vm.runInContext('canUseWebIq(document.getElementById("model-select").value)', sandbox), true);
    const endpoints = elements['endpoint-select'].children.map(e => e.value);
    assert.deepEqual(endpoints, ['auto', '/proxy/example-profile/v1/messages']);
    assert.equal(vm.runInContext('resolveEndpoint({model:"example-model"})', sandbox),
        '/proxy/example-profile/v1/messages');

    const session = { messages: [{ role: 'user', content: 'hello' }] };
    sandbox.currentSession = session;
    vm.runInContext('sessions.push(currentSession); abortController = { signal: {} }', sandbox);
    const stream = vm.runInContext('streamAnthropicMessages', sandbox);
    const content = element();
    const result = await stream(key, session, element(), content);
    assert.equal(result.text, 'OK');
    assert.equal(calls[0].url, '/proxy/example-profile/v1/messages');
    assert.equal(JSON.parse(calls[0].options.body).max_tokens, 2000);
    assert.equal(JSON.parse(calls[0].options.body).messages.length, 1);
    const id = calls[0].options.headers['X-Test-Client-Context'];
    assert.equal(id, session.headerValues[key]['X-Test-Client-Context']);
    const traceId = calls[0].options.headers['X-Test-Client-Trace'];
    assert.equal(traceId, session.headerValues[key]['X-Test-Client-Trace']);
    assert.notEqual(traceId, id);
    assert.match(id, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
    assert.equal(JSON.parse(saved['ghc_chat_sessions'])[0].headerValues[key]['X-Test-Client-Context'], id);

    session.messages.push({ role: 'assistant', content: result.text }, { role: 'user', content: 'again' });
    const next = await stream(key, session, element(), element());
    assert.equal(next.text, 'OK');
    assert.equal(calls[1].options.headers['X-Test-Client-Context'], id);
    assert.equal(calls[1].options.headers['X-Test-Client-Trace'], traceId);
    assert.deepEqual(JSON.parse(calls[1].options.body).messages.map(m => m.role), ['user', 'assistant', 'user']);

    // A truncated stream must not be committed as a successful assistant turn.
    sse = sse.slice(0, sse.indexOf('event: message_stop'));
    await assert.rejects(() => stream(key, session, element(), element()),
        /ended before message_stop/);

    // A Messages tool_use is followed by a user tool_result, then a final answer.
    // Thinking and redacted thinking must survive both the tool round and later turns.
    const frame = (e) => 'event: ' + e.type + '\ndata: ' + JSON.stringify(e) + '\n\n';
    const toolStream = [
        {type:'message_start', message:{content:[]}},
        {type:'content_block_start', index:0, content_block:{type:'thinking', thinking:''}},
        {type:'content_block_delta', index:0, delta:{type:'thinking_delta', thinking:'Let me think'}},
        {type:'content_block_delta', index:0, delta:{type:'signature_delta', signature:'signed-thinking'}},
        {type:'content_block_stop', index:0},
        {type:'content_block_start', index:1, content_block:{type:'redacted_thinking', data:'sealed-blob'}},
        {type:'content_block_stop', index:1},
        {type:'content_block_start', index:2, content_block:{type:'tool_use', id:'tool_1', name:'webiq_search', input:{}}},
        {type:'content_block_delta', index:2, delta:{type:'input_json_delta', partial_json:'{"query":"news"}'}},
        {type:'content_block_stop', index:2},
        {type:'message_stop'},
    ].map(frame).join('');
    elements['webiq-search-enabled'].checked = true;
    vm.runInContext('executeWebIqCall = async args => ({ results: [], output: JSON.stringify({ query: JSON.parse(args).query, hits: 1 }) })', sandbox);
    responseStreams = [toolStream, normalSse];
    const toolSession = { messages: [{ role: 'user', content: 'search news' }] };
    sandbox.currentToolSession = toolSession;
    vm.runInContext('sessions.push(currentToolSession)', sandbox);
    const answer = await stream(key, toolSession, element(), element());
    assert.equal(answer.text, 'OK');
    assert.equal(answer.thinking, 'Let me think');
    assert.equal(answer.toolCalls.length, 1);
    assert.equal(answer.toolCalls[0].name, 'webiq_search');
    assert.equal(answer.nativeMessages.length, 3);
    const toolRequest = JSON.parse(calls[3].options.body);
    const finalRequest = JSON.parse(calls[4].options.body);
    assert.equal(toolRequest.tools[0].name, 'webiq_search');
    assert.equal(toolRequest.tools[0].input_schema.properties.query.type, 'string');
    assert.equal(toolRequest.tool_choice.type, 'auto');
    assert.deepEqual(finalRequest.messages[1].content[0],
        {type:'thinking', thinking:'Let me think', signature:'signed-thinking'});
    assert.deepEqual(finalRequest.messages[1].content[1],
        {type:'redacted_thinking', data:'sealed-blob'});
    assert.equal(finalRequest.messages[1].content[2].type, 'tool_use');
    assert.deepEqual(finalRequest.messages[1].content[2].input, {query:'news'});
    assert.equal(finalRequest.messages[2].content[0].type, 'tool_result');
    assert.equal(finalRequest.messages[2].content[0].tool_use_id, 'tool_1');
    assert.equal(calls[3].options.headers['X-Test-Client-Context'], calls[4].options.headers['X-Test-Client-Context']);
    toolSession.messages.push({role:'assistant', content:answer.text, nativeMessages:answer.nativeMessages},
        {role:'user', content:'follow up'});
    const replay = vm.runInContext('buildAnthropicMessages(currentToolSession.messages)', sandbox);
    assert.deepEqual(Array.from(replay, m => m.role), ['user', 'assistant', 'user', 'assistant', 'user']);
    assert.deepEqual(JSON.parse(JSON.stringify(replay[1])), finalRequest.messages[1]);
    assert.equal(replay[2].content[0].type, 'tool_result');

    responseStreams = [toolStream.replace('webiq_search', 'unapproved_tool')];
    await assert.rejects(() => stream(key, toolSession, element(), element()),
        /Unsupported Messages tool call/);

    // No configured header: neither send nor persist a synthetic session ID.
    vm.runInContext('modelTargets[' + JSON.stringify(key) + '].requestHeaders = {}', sandbox);
    responseStreams = [normalSse];
    const plainSession = {messages:[{role:'user', content:'no session header'}]};
    await stream(key, plainSession, element(), element());
    assert.deepEqual(Object.keys(calls.at(-1).options.headers), ['Content-Type']);
    assert.equal(plainSession.headerValues, undefined);
})().catch(err => { console.error(err); process.exitCode = 1; });
"""


@unittest.skipUnless(shutil.which("node"), "Node.js is required for Chat UI JS checks")
class ConfiguredMessagesChatTest(unittest.TestCase):
    def test_chat_displays_messages_model_and_reuses_session_id_while_streaming(self):
        with create_app().test_client() as client:
            response = client.get("/chat")
        self.assertEqual(response.status_code, 200)
        script = response.get_data(as_text=True).rsplit("<script>", 1)[-1].split("</script>", 1)[0]
        checked = subprocess.run(["node", "--check"], input=script, text=True, capture_output=True)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        result = subprocess.run(["node", "-e", NODE_CHECK], input=script, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
