const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const script = fs.readFileSync(0, 'utf8').split('        // ---- Input handling ----')[0];
function element(tag = 'div') {
    const classes = new Set();
    const el = {tag, children: [], value: '', options: [], style: {}, disabled: false,
        classList: {
            toggle(name, on) { on ? classes.add(name) : classes.delete(name); },
            add(name) { classes.add(name); }, remove(name) { classes.delete(name); },
            contains(name) { return classes.has(name); },
        },
        appendChild(child) { this.children.push(child); return child; },
        insertBefore(child) { this.children.push(child); return child; },
        querySelector() { return null; }, remove() {}, focus() {},
    };
    Object.defineProperty(el, 'innerHTML', {
        get() { return this.html || ''; },
        set(value) { this.html = value; if (value === '') this.children = []; },
    });
    Object.defineProperty(el, 'textContent', {
        get() { return this.text || ''; },
        set(value) { this.text = String(value); this.html = this.text.replace(/&/g, '&amp;').replace(/</g, '&lt;'); },
    });
    return el;
}
const elements = {};
const document = {
    getElementById(id) { return elements[id] || (elements[id] = element()); },
    createElement(tag) { return element(tag); },
};
const el = id => document.getElementById(id);
el('search-provider').value = 'off';
el('search-provider').options = ['off', 'native', 'webiq', 'anysearch'].map(value => ({value}));
el('image-count').value = '1';
el('image-operation').value = 'generations';
const saved = {};
const localStorage = {getItem: k => saved[k] || null, setItem: (k, v) => { saved[k] = v; }};
const images = new Map();
let storageFailure = false;
const indexedDB = {open() {
    const opening = {};
    queueMicrotask(() => {
        opening.result = {close() {}, transaction() {
            const tx = {objectStore() {
                function op(action, value, key) {
                    const result = {};
                    queueMicrotask(() => {
                        if (storageFailure && action === 'put') {
                            tx.error = new Error('Quota exceeded'); tx.onabort(); return;
                        }
                        if (action === 'put') images.set(key, structuredClone(value));
                        else if (action === 'delete') images.delete(value);
                        else result.result = structuredClone(images.get(value));
                        tx.oncomplete();
                    });
                    return result;
                }
                return {put: (v,k) => op('put',v,k), get: k => op('get',k), delete: k => op('delete',k)};
            }};
            return tx;
        }};
        opening.onsuccess();
    });
    return opening;
}};
const model = 'gpt-image-2.5-flare';
let nextResponse = {status: 200, body: {
    data: [{b64_json: 'aW1hZ2U='}], output_format: 'png',
    usage: {input_tokens: 17, output_tokens: 196},
}};
let networkFailure = false;
const calls = [];
async function fetch(url, options) {
    if (url === '/v1/models') return {ok: true, json: async () => ({data: [
        {id: 'gpt-chat', supported_endpoints: ['/responses']},
        {id: model, model_type: 'image', supported_endpoints: ['/v1/images/generations', '/v1/images/edits']},
    ]})};
    if (url === '/proxy/models') return {ok: true, json: async () => ({data: []})};
    calls.push({url, options});
    if (networkFailure) throw new Error('Connection lost');
    return {ok: nextResponse.status < 400, status: nextResponse.status,
        text: async () => JSON.stringify(nextResponse.body), headers: {get: () => 'application/json'}};
}
const sandbox = {document, localStorage, indexedDB, fetch, crypto: require('node:crypto').webcrypto,
    TextDecoder, performance, console, AbortController, FormData, File, URL, atob,
    window: {location: {origin: 'http://localhost'}}, autoResize() {},
};
vm.createContext(sandbox);
vm.runInContext(script, sandbox);
const run = code => vm.runInContext(code, sandbox);
const last = () => el('messages-container').children.at(-1);
function descendants(element, tag) {
    return element.children.flatMap(c => [...(c.tag === tag ? [c] : []), ...descendants(c, tag)]);
}
(async () => {
    await run('loadModels()');
    const labels = el('model-select').children.flatMap(g => g.children.map(o => o.textContent));
    assert.ok(labels.some(label => label.includes('[image]')));
    el('model-select').value = model;
    run('onModelChange()');
    assert.equal(run('chatMode'), 'image');
    assert.equal(el('search-provider').disabled, true);
    assert.ok(el('image-controls').classList.contains('visible'));
    assert.equal(run('resolveEndpoint({})'), '/v1/images/generations');
    assert.equal(run('resolveEndpoint({images: []})'), '/v1/images/edits');
    assert.deepEqual(el('endpoint-select').children.map(o => o.value), ['auto', '/v1/images/generations', '/v1/images/edits']);

    run("setMode('chat')");
    el('chat-input').value = 'draw';
    await run('sendMessage()');
    assert.equal(calls.length, 0);
    assert.match(last().children[0].textContent, /Image mode/);
    run("setMode('image')");
    el('chat-input').value = '<img src=x onerror=alert(1)> blue circle';
    await run('sendMessage()');
    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, '/v1/images/generations');
    assert.equal(JSON.parse(calls[0].options.body).model, model);
    assert.equal(JSON.parse(calls[0].options.body).quality, 'low');
    assert.equal(images.size, 1);
    assert.ok(!JSON.stringify(saved).includes('aW1hZ2U='), 'binary must not enter localStorage');
    assert.equal(run('conversationTurns(getActiveSession().messages).length'), 0);
    const rendered = descendants(last(), 'img');
    assert.equal(rendered[0].src, 'data:image/png;base64,aW1hZ2U=');
    assert.equal(descendants(last(), 'a')[0].download, 'image-1.png');
    assert.match(last().children[0].textContent, /tokens 17 in \/ 196 out/);

    // Reuse a generated image for an edit; no image re-encoding takes place.
    descendants(last(), 'button')[0].onclick();
    assert.equal(el('image-operation').value, 'edits');
    el('chat-input').value = 'make it red';
    await run('sendMessage()');
    assert.equal(calls[1].url, '/v1/images/edits');
    assert.ok(calls[1].options.body instanceof FormData);
    assert.equal(calls[1].options.headers, undefined);
    assert.equal(calls[1].options.body.get('image[]').name, 'reference.png');
    assert.equal(calls[1].options.body.get('model'), model);

    // Restore from IndexedDB, not a short-lived URL or a huge localStorage item.
    await run('restoreImageResponse(getActiveSession().messages[1])');
    assert.equal(descendants(last(), 'img').length, 1);
    // Raw image requests/responses use the same binary store and safe previews.
    run("setMode('raw')");
    el('endpoint-select').value = 'auto';
    el('chat-input').value = JSON.stringify({prompt: 'edit', images: [{image_url: 'data:image/png;base64,c2VjcmV0'}]});
    await run('sendMessage()');
    assert.equal(calls.at(-1).url, '/v1/images/edits');
    assert.equal(descendants(last(), 'img').length, 1);
    assert.ok(!JSON.stringify(saved).includes('c2VjcmV0'));
    assert.ok(!JSON.stringify(saved).includes('aW1hZ2U='));
    run("setMode('image')");
    el('chat-input').value = 'make it red';
    run('selectImageFiles([])');
    const before = calls.length;
    await run('sendMessage()');
    assert.equal(calls.length, before, 'edit without a reference must not spend quota');

    el('image-operation').value = 'generations';
    storageFailure = true;
    await run('sendMessage()');
    assert.equal(descendants(last(), 'img').length, 1, 'quota failure must not lose generated image');
    assert.match(last().children.at(-1).textContent, /not saved/);
    assert.ok(!JSON.stringify(saved).includes('aW1hZ2U='));
    storageFailure = false;

    nextResponse = {status: 403, body: {error: {message: '<script>policy disabled</script>'}}};
    await run('sendMessage()');
    assert.match(last().children.at(-1).textContent, /policy disabled/);
    assert.ok(!last().children.at(-1).innerHTML.includes('<script>'));
    networkFailure = true;
    const count = calls.length;
    await run('sendMessage()');
    assert.equal(calls.length, count + 1);
    assert.equal(el('send-btn').disabled, false);
    assert.match(last().children.at(-1).textContent, /No retry/);

    // Reject active-content URLs and invalid base64 without rendering HTML.
    const div = element();
    sandbox.testDiv = div;
    run(`renderImageResponse(testDiv, {data: [{url: 'javascript:alert(1)'}, {url: 'data:image/svg+xml,bad'}, {b64_json:'<script>'}]}, 200, 1)`);
    assert.equal(descendants(div, 'img').length, 0);
    el('model-select').value = 'gpt-chat';
    run('onModelChange()');
    assert.equal(run('chatMode'), 'chat');
    const id = run('activeSessionId');
    run(`deleteSession(${JSON.stringify(id)}, {stopPropagation(){}})`);
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(images.size, 0);
})().catch(err => {console.error(err); process.exitCode = 1;});
