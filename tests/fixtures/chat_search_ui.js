// Browser-free search contract and three-protocol tool-loop checks. No network.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(0, 'utf8').split('        // ---- Input handling ----')[0]
    .replace(/const WEBIQ_AVAILABLE = (true|false);/, 'const WEBIQ_AVAILABLE = true;')
    .replace(/const ANYSEARCH_AVAILABLE = (true|false);/, 'const ANYSEARCH_AVAILABLE = true;');
function element() {
    return {children: [], options: [], value: '', innerHTML: '', textContent: '',
        classList: {toggle() {}}, appendChild(c) {this.children.push(c);},
        insertBefore(c) {this.children.push(c);}, querySelector() {return null;}, remove() {}};
}
const elements = Object.fromEntries(['model-select', 'endpoint-select', 'search-provider', 'messages-container']
    .map(id => [id, element()]));
elements['search-provider'].options = ['off', 'native', 'webiq', 'anysearch'].map(value => ({value}));
elements['model-select'].value = 'm';
elements['endpoint-select'].value = 'auto';
const saved = {};
const calls = [];
let streams = [];
let searchReply;
let searchFailure;
let toolReplies = {};
let toolSequence = 0;
function jsonReply(body, status = 200) {return {ok: status === 200, status, json: async () => body};}
const successBody = {code: 0, data: {results: [
    {title: '', url: 'https://example.com/one', snippet: 'summary'},
    {title: 'Long', url: 'https://example.com/two', content: 'x'.repeat(4000)},
    {title: 'Missing content', url: 'https://example.com/three'},
]}};
function sseReply(text) {
    // Fragment transport bytes as well as tool-argument deltas.
    const bytes = new TextEncoder().encode(text);
    const chunks = [bytes.slice(0, 17), bytes.slice(17, 72), bytes.slice(72)];
    return {ok: true, body: {getReader() {return {async read() {
        return chunks.length ? {done: false, value: chunks.shift()} : {done: true};
    }}}}};
}
async function fetch(url, options) {
    calls.push({url, method: options.method, body: options.body ? JSON.parse(options.body) : null, signal: options.signal});
    if (url === '/v1/search' || url === '/v1/extract' || url.startsWith('/v1/sub-domains?')) {
        if (searchFailure) throw searchFailure;
        return toolReplies[url.split('?')[0]] || searchReply;
    }
    assert.notEqual(url, '/v3/search/web', 'AnySearch must never fall back to Web IQ');
    assert.ok(streams.length, 'unexpected model call');
    return sseReply(streams.shift());
}
const sandbox = {document: {getElementById: id => elements[id], createElement: element},
    localStorage: {getItem: key => saved[key] ?? null, setItem: (key, value) => {saved[key] = value;}},
    fetch, TextDecoder, performance, console, URL, URLSearchParams, window: {location: {origin: 'https://proxy.test'}},
    crypto: require('node:crypto').webcrypto};
vm.createContext(sandbox);
vm.runInContext(source, sandbox);
vm.runInContext(`modelTargets.m = {id:'gpt-test', type:'copilot', baseUrl:'/v1',
    supportedEndpoints:['/responses'], requestHeaders:{}};
    abortController = {signal: {testSignal:true}};`, sandbox);
const run = expr => vm.runInContext(expr, sandbox);
const frame = data => 'data: ' + JSON.stringify(data) + '\n\n';
function toolStream(style, name = 'anysearch_search', argumentsJson = '{"query":"news"}') {
    const itemId = name + '_' + (++toolSequence);
    const callId = 'call_' + itemId;
    if (style === 'messages') return [
        {type:'message_start', message:{content:[]}},
        {type:'content_block_start', index:0, content_block:{type:'tool_use', id:callId, name, input:{}}},
        {type:'content_block_delta', index:0, delta:{type:'input_json_delta', partial_json:argumentsJson.slice(0, 4)}},
        {type:'content_block_delta', index:0, delta:{type:'input_json_delta', partial_json:argumentsJson.slice(4)}},
        {type:'content_block_stop', index:0}, {type:'message_stop'},
    ].map(frame).join('');
    if (style === 'responses') {
        const item = {type:'function_call', id:'fc_' + itemId, call_id:callId, name, arguments:argumentsJson};
        return [
            {type:'response.output_item.done', item:{type:'reasoning', id:'reason_' + itemId, summary:[]}},
            {type:'response.output_item.done', item},
            {type:'response.completed', response:{output:[item]}}, // must dedupe
        ].map(frame).join('');
    }
    return [
        {choices:[{delta:{tool_calls:[{index:0, id:callId, type:'function',
            function:{name:name.slice(0, 4), arguments:argumentsJson.slice(0, 4)}}]}}]},
        {choices:[{delta:{tool_calls:[{index:0, function:{name:name.slice(4), arguments:argumentsJson.slice(4)}}]}}]},
    ].map(frame).join('') + 'data: [DONE]\n\n';
}
function finalStream(style) {
    if (style === 'messages') return [
        {type:'message_start', message:{content:[]}},
        {type:'content_block_start', index:0, content_block:{type:'text', text:''}},
        {type:'content_block_delta', index:0, delta:{type:'text_delta', text:'OK'}},
        {type:'message_stop'},
    ].map(frame).join('');
    if (style === 'responses') return [
        {type:'response.output_text.delta', delta:'OK'},
        {type:'response.completed', response:{output:[]}},
    ].map(frame).join('');
    return frame({choices:[{delta:{content:'OK'}}]}) + 'data: [DONE]\n\n';
}
function reset() {
    calls.length = 0;
    streams = [];
    searchReply = jsonReply(successBody);
    searchFailure = null;
    toolReplies = {};
    toolSequence = 0;
    elements['search-provider'].value = 'anysearch';
}
async function turn(style) {
    const messages = [{role:'user', content:'news'}];
    if (style === 'messages') return run('streamAnthropicMessages')('m', {messages}, element(), element());
    return run(style === 'responses' ? 'streamResponses' : 'streamChatCompletions')('m', messages, element(), element());
}
(async () => {
    // Migration, exclusivity, availability, and Native model restrictions.
    run('restoreSearchProvider()');
    assert.equal(elements['search-provider'].value, 'native');
    saved.ghc_chat_web_search_enabled = 'false';
    run('restoreSearchProvider()');
    assert.equal(elements['search-provider'].value, 'off');
    saved.ghc_chat_webiq_search_enabled = 'true';
    run('restoreSearchProvider()');
    assert.equal(elements['search-provider'].value, 'webiq');
    saved.ghc_chat_search_provider = 'anysearch';
    run('restoreSearchProvider()');
    assert.equal(elements['search-provider'].value, 'anysearch');
    run('onSearchProviderChange()');
    for (const provider of ['webiq', 'anysearch']) {
        elements['search-provider'].value = provider;
        for (const style of ['chat', 'responses', 'messages']) {
            const tools = run(`searchTools('m', '${style}')`);
            assert.equal(tools.length, provider === 'webiq' ? 1 : 3);
            assert.equal(tools[0].function?.name || tools[0].name, provider === 'webiq' ? 'webiq_search' : 'anysearch_search');
        }
    }
    elements['search-provider'].value = 'native';
    assert.equal(run("searchTools('m', 'responses')")[0].type, 'web_search');
    assert.equal(run("searchTools('m', 'chat')").length, 0);
    run("modelTargets.m.type = 'configured_proxy'");
    assert.equal(run("searchTools('m', 'responses')").length, 0);
    run("modelTargets.m.type = 'copilot'; chatMode = 'raw'");
    elements['search-provider'].value = 'anysearch';
    assert.equal(run("searchTools('m', 'responses')").length, 0);
    run("chatMode = 'chat'; SEARCH_PROVIDERS.anysearch.available = false; updateWebSearchAvailability()");
    assert.equal(elements['search-provider'].value, 'anysearch', 'no silent switch');
    assert.equal(elements['search-provider'].options.find(o => o.value === 'anysearch').disabled, true);
    assert.equal(run("searchTools('m', 'chat')").length, 0);
    run('SEARCH_PROVIDERS.anysearch.available = true');

    // UI policies do not leak into the native proxy contract.
    reset();
    let result = await run('executeAnySearchCall')('{"query":"q"}');
    assert.deepEqual(calls[0].body, {query:'q', max_results:5, format:'markdown'});
    const normalized = JSON.parse(result.output);
    assert.equal(normalized.results[0].content, 'summary');
    assert.equal(normalized.results[1].content.length, 3000);
    assert.equal(normalized.results[1].truncated, true);
    assert.equal(normalized.results[2].content, '');
    assert.match(normalized.note, /Untrusted/);
    assert.equal(calls[0].signal.testSignal, true);
    for (const args of ['[]', 'null', 'bad JSON', '{}', '{"query":"q","max_results":0}',
        '{"query":"q","max_results":11}', '{"query":"q","max_results":true}']) {
        reset();
        assert.ok(JSON.parse((await run('executeAnySearchCall')(args)).output).error);
        assert.equal(calls.length, 0);
    }
    for (const body of [{code:-1, message:'SENSITIVE'}, {code:0, data:{}},
        {code:0, data:{results:[null]}}, {code:0, data:{results:[{title:'', url:'u', content:null}]}}]) {
        reset();
        searchReply = jsonReply(body);
        const out = (await run('executeAnySearchCall')('{"query":"q"}')).output;
        assert.ok(JSON.parse(out).error);
        assert.ok(!out.includes('SENSITIVE'));
    }
    reset();
    searchReply = jsonReply({code:0, data:{results:[]}});
    assert.equal(JSON.parse((await run('executeAnySearchCall')('{"query":"q"}')).output).results.length, 0);
    reset();
    searchReply = {ok:false, status:402, json() {throw new Error('must not read 402');}};
    assert.match((await run('executeAnySearchCall')('{"query":"q"}')).output, /HTTP 402/);
    reset();
    searchReply = {ok:true, json:async () => {throw new Error('SENSITIVE malformed response');}};
    assert.ok(!(await run('executeAnySearchCall')('{"query":"q"}')).output.includes('SENSITIVE'));
    reset();
    searchFailure = Object.assign(new Error('cancelled'), {name:'AbortError'});
    await assert.rejects(() => run('executeAnySearchCall')('{"query":"q"}'), {name:'AbortError'});
    assert.equal(calls.length, 1);

    const catalogBody = {code:0, data:{domains:[{domain:'code', sub_domains:[{
        sub_domain:'code.doc', params:{library:{description:'Library name', required:true, sort_order:0}},
    }]}]}};
    const extractBody = {code:0, data:{url:'https://example.com/one', title:'Read source', content:'x'.repeat(13000)}};
    for (const [name, args, body] of [
        ['anysearch_extract', {url:'https://example.com/one'}, extractBody],
        ['anysearch_sub_domains', {domain:['code', 'finance']}, catalogBody],
    ]) {
        reset();
        searchReply = jsonReply(body);
        const result = await run('executeAnySearchCall')(JSON.stringify(args), name);
        const parsed = JSON.parse(result.output);
        assert.ok(!parsed.error);
        if (name === 'anysearch_extract') {
            assert.equal(calls[0].url, '/v1/extract');
            assert.equal(calls[0].method, 'POST');
            assert.equal(parsed.content.length, 12000);
            assert.equal(parsed.truncated, true);
            assert.equal(result.results[0].url, args.url);
        } else {
            assert.equal(calls[0].url, '/v1/sub-domains?domain=code&domain=finance');
            assert.equal(calls[0].method, 'GET');
            assert.equal(calls[0].body, null);
            assert.equal(parsed.domains[0].sub_domains[0].params.library.required, true);
            assert.match(parsed.note, /not DNS/);
            assert.equal(result.results.length, 0);
        }
        reset();
        searchReply = {ok:false, status:402, json() {throw new Error('must not read 402');}};
        assert.match((await run('executeAnySearchCall')(JSON.stringify(args), name)).output, /HTTP 402/);
        reset();
        searchFailure = Object.assign(new Error('cancelled'), {name:'AbortError'});
        await assert.rejects(() => run('executeAnySearchCall')(JSON.stringify(args), name), {name:'AbortError'});
        assert.equal(calls.length, 1);
        reset();
        elements['search-provider'].value = 'webiq';
        await assert.rejects(() => run('executeSearchCall')(name, JSON.stringify(args), run("searchTools('m', 'responses')")),
            /Unsupported search tool call/);
        assert.equal(calls.length, 0);
    }
    for (const [name, args] of [['anysearch_extract', {}], ['anysearch_extract', {url:4}],
        ['anysearch_extract', {url:'https://example.com', format:'markdown'}],
        ['anysearch_sub_domains', {domain:'code'}], ['anysearch_sub_domains', {domain:[]}],
        ['anysearch_sub_domains', {domain:['code', null]}], ['anysearch_sub_domains', {domains:['code']}],
        ['anysearch_search', {query:'q', params:[]}], ['anysearch_search', {query:'q', zone:'bad'}]]) {
        reset();
        assert.ok(JSON.parse((await run('executeAnySearchCall')(JSON.stringify(args), name)).output).error);
        assert.equal(calls.length, 0);
    }
    for (const [name, body] of [['anysearch_extract', {code:0, data:{url:'u', title:'t'}}],
        ['anysearch_sub_domains', {code:0, data:{domains:[null]}}]]) {
        reset();
        searchReply = jsonReply(body);
        const args = name === 'anysearch_extract' ? {url:'https://example.com'} : {domain:['code']};
        assert.ok(JSON.parse((await run('executeAnySearchCall')(JSON.stringify(args), name)).output).error);
    }

    for (const style of ['messages', 'responses', 'chat']) {
        // Capability discovery -> tagged search -> read a source -> answer.
        reset();
        toolReplies = {'/v1/sub-domains':jsonReply(catalogBody), '/v1/extract':jsonReply(extractBody)};
        streams = [
            toolStream(style, 'anysearch_sub_domains', '{"domain":["code","finance"]}'),
            toolStream(style, 'anysearch_search', '{"query":"q","tag":"code.doc","params":{"library":"golang"},"zone":"intl","language":"en"}'),
            toolStream(style, 'anysearch_extract', '{"url":"https://example.com/one"}'),
            finalStream(style),
        ];
        const pipeline = await turn(style);
        assert.equal(pipeline.text, 'OK');
        assert.deepEqual(Array.from(pipeline.toolCalls, c => c.name),
            ['anysearch_sub_domains', 'anysearch_search', 'anysearch_extract']);
        assert.equal(pipeline.sources.length, 3, 'extract citation must dedupe the existing source URL');
        assert.equal(new Set(pipeline.toolCalls.map(c => c.id)).size, 3, 'tool call IDs must remain distinct');
        sandbox.pipelineMessages = [{role:'user', content:'research'}, {role:'assistant', content:pipeline.text,
            toolCalls:pipeline.toolCalls, nativeMessages:pipeline.nativeMessages}, {role:'user', content:'follow up'}];
        const replay = run(style === 'messages' ? 'buildAnthropicMessages(pipelineMessages)' :
            style === 'responses' ? 'buildResponsesInput(pipelineMessages)' : 'buildChatMessages(pipelineMessages)');
        for (const name of ['anysearch_sub_domains', 'anysearch_search', 'anysearch_extract']) {
            assert.ok(JSON.stringify(replay).includes(name), 'missing tool in history: ' + name);
        }
        assert.match(JSON.stringify(replay), /sort_order/);
        const tagged = calls.find(c => c.url === '/v1/search').body;
        assert.equal(tagged.tag, 'code.doc');
        assert.deepEqual(tagged.params, {library:'golang'});
        assert.equal(tagged.zone, 'intl');
        assert.equal(tagged.language, 'en');
        assert.equal(tagged.format, 'markdown');
        assert.match(pipeline.toolCalls[0].output, /sort_order/);
        assert.equal(JSON.parse(pipeline.toolCalls[2].output).truncated, true);
        assert.equal(calls.filter(c => c.url.startsWith('/v1/sub-domains?')).length, 1);
        assert.equal(calls.filter(c => c.url === '/v1/extract').length, 1);
        const finalChoice = calls.at(-1).body.tool_choice;
        assert.equal(finalChoice.type || finalChoice, 'none');
        // Inspect outgoing schemas too, not only the synthetic SSE tool calls.
        for (const tool of calls.flatMap(c => c.body?.tools || [])) {
            const fn = tool.function || tool;
            if (style === 'messages') assert.equal(Object.hasOwn(fn, 'strict'), false);
            else assert.equal(fn.strict, false);
        }
        for (const name of ['anysearch_extract', 'anysearch_sub_domains']) {
            reset();
            const args = name === 'anysearch_extract' ? {url:'https://example.com'} : {domain:['code']};
            searchReply = jsonReply({}, 422);
            streams = [toolStream(style, name, JSON.stringify(args)), finalStream(style)];
            assert.match((await turn(style)).toolCalls[0].output, /HTTP 422/);
        }
        reset();
        streams = [toolStream(style), finalStream(style)];
        const answer = await turn(style);
        assert.equal(answer.text, 'OK');
        assert.equal(answer.toolCalls.length, 1);
        assert.equal(answer.toolCalls[0].name, 'anysearch_search');
        assert.equal(answer.sources.length, 3);
        assert.equal(calls.filter(c => c.url === '/v1/search').length, 1);
        const modelCalls = calls.filter(c => c.url !== '/v1/search');
        assert.equal(modelCalls.length, 2);
        assert.match(JSON.stringify(modelCalls[1].body), /Untrusted web content/);
        if (style === 'responses') assert.match(JSON.stringify(modelCalls[1].body), /reason_anysearch_search_1/);
        sandbox.replayMessages = [{role:'user', content:'news'},
            {role:'assistant', content:answer.text, toolCalls:answer.toolCalls, nativeMessages:answer.nativeMessages},
            {role:'user', content:'follow-up'}];
        const projection = run(style === 'messages' ? 'buildAnthropicMessages(replayMessages)' :
            style === 'responses' ? 'buildResponsesInput(replayMessages)' : 'buildChatMessages(replayMessages)');
        assert.match(JSON.stringify(projection), /anysearch_search/);
        assert.match(JSON.stringify(projection), /Untrusted web content/);

        // The executor uses the original declared tool set, not a selector
        // change that happens while a model call is in flight.
        reset();
        const declared = run(`searchTools('m', '${style}')`);
        elements['search-provider'].value = 'webiq';
        await run('executeSearchCall')('anysearch_search', '{"query":"q"}', declared);
        assert.equal(calls[0].url, '/v1/search');

        // Search errors are tool results, not failed chat turns or fallbacks.
        reset();
        searchReply = jsonReply({}, 429);
        streams = [toolStream(style), finalStream(style)];
        const degraded = await turn(style);
        assert.equal(degraded.text, 'OK');
        assert.match(degraded.toolCalls[0].output, /HTTP 429/);

        // Known but unselected providers are just as forbidden as unknown tools.
        for (const toolName of ['webiq_search', 'unapproved_tool']) {
            reset();
            streams = [toolStream(style, toolName)];
            await assert.rejects(() => turn(style), /Unsupported.*tool call/);
            assert.equal(calls.filter(c => c.url === '/v1/search').length, 0);
        }
        reset();
        streams = Array.from({length:4}, () => toolStream(style));
        await assert.rejects(() => turn(style), /Unexpected.*tool call/);
        assert.equal(calls.filter(c => c.url === '/v1/search').length, 3);
        const last = calls.at(-1).body;
        assert.equal(last.tool_choice.type || last.tool_choice, 'none');
    }
})().catch(error => {console.error(error); process.exitCode = 1;});
