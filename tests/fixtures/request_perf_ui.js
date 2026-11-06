const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));

function element(tag = 'div') {
    const classes = new Set();
    const node = {
        tag, children: [], attributes: {}, disabled: false, hidden: false, checked: false,
        style: {setProperty(key, value) { this[key] = value; }},
        classList: {
            add(name) { classes.add(name); }, remove(name) { classes.delete(name); },
            contains(name) { return classes.has(name); },
        },
        appendChild(child) { this.children.push(child); return child; },
        replaceChildren(...children) { this.children = children; this._text = ''; },
        setAttribute(name, value) { this.attributes[name] = String(value); },
        removeAttribute(name) { delete this.attributes[name]; },
    };
    Object.defineProperty(node, 'textContent', {
        get() { return (this._text || '') + this.children.map(c => c.textContent).join(' '); },
        set(value) { this._text = String(value); this.children = []; },
    });
    Object.defineProperty(node, 'innerHTML', {set() { throw new Error('Logged labels must not enter innerHTML'); }});
    Object.defineProperty(node, 'value', {
        get() { return this._value ?? (this.tag === 'select' ? this.children[0]?.value || '' : ''); },
        set(value) { this._value = value; },
    });
    return node;
}
const elements = {};
for (const match of input.html.matchAll(/<([a-z]+)[^>]*\bid="([^"]+)"[^>]*>/g)) {
    elements[match[2]] = element(match[1]);
    elements[match[2]].hidden = match[0].includes(' hidden');
}
const $ = id => { assert.ok(elements[id], `Missing template element ${id}`); return elements[id]; };
for (const [id, value] of [['heat-metric', 'tps'], ['volume-metric', 'output'], ['context-metric', 'tps']]) $(id).value = value;
const document = {getElementById: $, createElement: element, createElementNS: (_, tag) => element(tag)};
const timers = new Map();
let timerId = 0;
let responseMode = 'completed';
let postCount = 0;
const job = {id: 'job-1', status: 'queued', files_completed: 0, files_total: 1,
    bytes_processed: 0, bytes_total: 1000, cache_hits: 0, incremental_scans: 0, rebuild_scans: 0};
const context = {
    document, console,
    setTimeout(fn) { const id = ++timerId; timers.set(id, fn); return id; },
    clearTimeout(id) { timers.delete(id); },
    addEventListener() {},
    async fetch(url, options) {
        let data;
        let ok = true;
        if (url === '/api/request-stats/files') data = {files: [{name: '2026-07-31.jl', size_bytes: 1000, index_status: 'ready'}]};
        else if (url === '/api/request-stats/jobs') {
            assert.equal(options.method, 'POST');
            const payload = JSON.parse(options.body);
            assert.equal(payload.kind, 'performance');
            assert.deepEqual(payload.files, ['2026-07-31.jl']);
            postCount++;
            if (responseMode === 'busy') { ok = false; data = {error: 'busy', message: 'Another job is running'}; }
            else data = {job};
        } else if (url === '/api/request-stats/jobs/job-1/cancel') {
            assert.equal(options.method, 'POST'); responseMode = 'cancelled'; data = {job};
        } else if (url === '/api/request-stats/jobs/job-1') {
            if (responseMode === 'missing') { ok = false; data = {error: 'Job expired'}; }
            else if (responseMode === 'empty') data = {job: {...job, status: 'completed', result: {...input.result, perf: {...input.result.perf, dates: [], cells: [], total_rows: 0}}}};
            else if (responseMode === 'running') data = {job: {...job, status: 'running', current_mode: 'aggregating'}};
            else data = {job: {...job, status: responseMode, result: input.result, error: 'test failure'}};
        } else throw new Error(`Unexpected external/API call: ${url}`);
        return {ok, status: ok ? 200 : 409, async json() { return data; }};
    },
};
context.window = context;
vm.createContext(context);
vm.runInContext(input.math, context);
const M = context.RequestPerfMath;
const fitPoints = Array.from({length: 1000}, (_, i) => [2 + (10 + i * 10) / 20, 10 + i * 10, 8000]);
const fit = M.fit(fitPoints);
assert.ok(Math.abs(fit.ttft - 2) < 1e-8);
assert.ok(Math.abs(fit.decode - 20) < 1e-8);
assert.equal(fit.count, 220);
assert.equal(M.fit(fitPoints.slice(0, 9)), null);
assert.equal(M.fit(Array.from({length: 20}, (_, i) => [i, 100, 0])), null);
assert.equal(M.fit(Array.from({length: 20}, (_, i) => [100 - i, i * 10, 0])), null);
assert.equal(M.percentile([0, 0], [1, 10, 100], .5), null);
assert.ok(Math.abs(M.percentile([0, 10], [1, 10, 100], .5) - Math.sqrt(1000)) < 1e-8);
assert.equal(M.shiftDate('2026-03-01', -1), '2026-02-28');
assert.equal(M.change(null, 2), null);
assert.equal(M.change(5, 0), null);
const combined = M.aggregate(input.result.perf.cells, input.result.perf.total_fields);
const summary = M.summary(combined, input.result.perf);
assert.equal(summary.ttftSource, 'measured');
assert.equal(summary.ttft, 2);
assert.equal(summary.decodeSource, 'measured');
const legacy = M.aggregate(input.result.perf.cells.filter(c => c.model.startsWith('legacy')), input.result.perf.total_fields);
assert.equal(M.summary(legacy, input.result.perf).ttftSource, 'estimated');
const zero = {...combined, ttft_sum: 0};
assert.equal(M.summary(zero, input.result.perf).ttft, 0);

async function flush() { for (let i = 0; i < 10; i++) await Promise.resolve(); }
async function tick() {
    const [id, fn] = timers.entries().next().value || [];
    assert.ok(fn, 'Expected a job poll'); timers.delete(id); await fn(); await flush();
}
function allNodes(node) { return [node, ...node.children.flatMap(allNodes)]; }
function assertFiniteCharts() {
    for (const id of ['trend', 'volume', 'heat', 'context', 'scatter', 'distribution']) {
        assert.equal($(id).children[0].tag, 'svg', `${id} must render SVG`);
        for (const node of allNodes($(id))) {
            for (const value of Object.values(node.attributes)) assert.ok(!/NaN|Infinity|undefined/.test(value), `${id}: bad SVG attribute ${value}`);
        }
    }
}
(async () => {
    vm.runInContext(input.script, context);
    await flush();
    assert.equal(postCount, 0, 'Loading the page must not scan logs');
    assert.equal($('generate').disabled, true);
    $('select-all').onclick();
    assert.equal($('generate').disabled, false);
    const start = $('generate').onclick();
    await $('generate').onclick(); // double-click cannot submit twice
    await start;
    assert.equal(postCount, 1);
    assert.equal($('generate').disabled, true);
    await tick();
    assert.equal($('report').hidden, false);
    assert.equal($('generate').disabled, false);
    assert.ok($('notices').textContent.includes('2 invalid JSON'));
    assert.ok($('model-table').textContent.includes('measured'));
    assert.ok($('model-table').textContent.includes('estimated'));
    assert.ok($('model-chips').textContent.includes('<img src=x'));
    assert.ok($('upstream-table').textContent.includes('<svg onload='));
    assertFiniteCharts();

    $('range-7').onclick();
    assert.equal($('from').value, '2026-07-28', 'Seven calendar days must not select all four available dates');
    $('range-all').onclick();
    assert.equal($('from').value, '2026-07-01');
    assert.ok($('diagnosis').textContent.includes('effective TPS'));
    for (const option of $('trend-metric').children) {
        $('trend-metric').value = option.value; $('trend-metric').onchange(); assertFiniteCharts();
    }
    for (const value of ['tps', 'ttft', 'decode', 'ttft_est', 'decode_est', 'duration', 'count']) {
        $('context-metric').value = value; $('context-metric').onchange(); assertFiniteCharts();
    }
    for (const value of ['output', 'count', 'wait', 'context']) {
        $('volume-metric').value = value; $('volume-metric').onchange(); assertFiniteCharts();
    }
    for (const value of ['tps', 'duration', 'count']) {
        $('heat-metric').value = value; $('heat-metric').onchange(); assertFiniteCharts();
    }
    const headerButton = $('model-table').children[0].children[0].children[1].children[0];
    headerButton.onclick(); assertFiniteCharts();
    $('clear-models').onclick();
    assertFiniteCharts();
    assert.equal($('endpoint-table').children[1].children.length, 0, 'Auxiliary tables must respect the model filter');
    $('all-models').onclick();
    assertFiniteCharts();
    $('from').value = '2026-07-31'; $('from').onchange();
    assert.ok($('diagnosis-hint').textContent.includes('at least two'));

    responseMode = 'running'; await $('generate').onclick(); await tick();
    assert.ok($('job-title').textContent.includes('Aggregating'));
    await $('cancel').onclick(); await tick();
    assert.ok($('notices').textContent.includes('cancelled'));
    assert.equal($('report').hidden, false, 'Cancellation preserves the old report');
    assert.equal($('generate').disabled, false);
    responseMode = 'failed'; await $('generate').onclick(); await tick();
    assert.ok($('notices').textContent.includes('test failure'));
    responseMode = 'busy'; await $('generate').onclick();
    assert.ok($('notices').textContent.includes('Another job'));
    assert.equal($('generate').disabled, false);
    responseMode = 'missing'; await $('generate').onclick(); await tick();
    assert.ok($('notices').textContent.includes('Job expired'));
    assert.equal(timers.size, 0, 'Expired jobs must not poll forever');
    responseMode = 'empty'; await $('generate').onclick(); await tick();
    assert.equal($('report').hidden, true);
    assert.ok($('notices').textContent.includes('No valid indexed requests'));
    console.log('Performance math, all charts, filters, safe labels, and job lifecycle passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
