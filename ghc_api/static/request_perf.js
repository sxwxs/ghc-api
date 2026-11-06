/* Historical performance UI. Data comes only from the shared local index jobs. */
(function () {
    'use strict';
    const M = window.RequestPerfMath;
    const $ = id => document.getElementById(id);
    const colors = ['#008575', '#2563eb', '#be185d', '#a16207', '#7c3aed', '#15803d', '#dc2626', '#0369a1'];
    const terminal = new Set(['completed', 'cancelled', 'failed']);
    const state = {files: [], selected: new Set(), job: null, submitting: false, timer: null,
        data: null, from: '', to: '', models: new Set(), order: [], colors: new Map(), sorts: {}};
    const number = (value, digits = 1) => value == null || !Number.isFinite(value) ? '—' : value.toLocaleString(undefined, {maximumFractionDigits: digits});
    const compact = value => value == null ? '—' : Math.abs(value) >= 1e6 ? number(value / 1e6) + 'M' : Math.abs(value) >= 1e3 ? number(value / 1e3) + 'k' : number(value);
    const bytes = value => value >= 1e9 ? number(value / 1e9, 2) + ' GB' : number(value / 1e6, 2) + ' MB';
    const text = (tag, value, parent) => {
        const node = document.createElement(tag);
        node.textContent = value;
        if (parent) parent.appendChild(node);
        return node;
    };
    const sum = rows => M.aggregate(rows, state.data.total_fields);
    const selected = rows => rows.filter(r => r.date >= state.from && r.date <= state.to && state.models.has(r.model));
    const selectedModels = () => state.order.filter(model => state.models.has(model));
    const contextTotal = a => a.input + a.cache_read + a.cache_write;

    const metrics = {
        tps: {label: 'Effective TPS', unit: 'tok/s', calc: a => M.ratio(a.gen_output, a.gen_duration), hint: 'Output / total duration on successful generating requests, including first-token wait.'},
        tps_median: {label: 'Approx. per-request TPS median', unit: 'tok/s', calc: a => M.percentile(a.tps_hist, state.data.tps_edges, .5)},
        tps_p10: {label: 'Approx. per-request TPS P10', unit: 'tok/s', calc: a => M.percentile(a.tps_hist, state.data.tps_edges, .1)},
        ttft: {label: 'Measured TTFT mean', unit: 's', calc: a => M.ratio(a.ttft_sum, a.ttft_count), hint: 'Only persisted, valid TTFT values. Missing historical values leave gaps; no estimates are mixed into this curve.'},
        decode: {label: 'Measured decode TPS', unit: 'tok/s', calc: a => M.ratio(a.decode_output, a.decode_duration), hint: 'Output / (duration − TTFT), only on requests with valid measured timing and a positive generation interval.'},
        ttft_est: {label: 'Estimated TTFT (fit)', unit: 's', calc: a => M.fit(a.samples)?.ttft ?? null, hint: 'Theil–Sen fitted intercept from sampled requests, not measured first-token time. At least 10 samples required.'},
        decode_est: {label: 'Estimated decode TPS (fit)', unit: 'tok/s', calc: a => M.fit(a.samples)?.decode ?? null, hint: 'Reciprocal of the fitted slope. Changing request mix can affect this estimate.'},
        duration: {label: 'Mean duration', unit: 's', calc: a => M.ratio(a.duration, a.duration_count)},
        p50: {label: 'Approx. duration P50', unit: 's', calc: a => M.percentile(a.duration_hist, state.data.duration_edges, .5)},
        p90: {label: 'Approx. duration P90', unit: 's', calc: a => M.percentile(a.duration_hist, state.data.duration_edges, .9)},
        p99: {label: 'Approx. duration P99', unit: 's', calc: a => M.percentile(a.duration_hist, state.data.duration_edges, .99)},
        context: {label: 'Mean context / generating request', unit: 'tokens', calc: a => M.ratio(a.context_sum, a.context_count)},
        output_mean: {label: 'Mean output / generating request', unit: 'tokens', calc: a => M.ratio(a.gen_output, a.gen_count)},
        errors: {label: 'Failures + cancellations', unit: '%', calc: a => M.ratio(100 * (a.errors + a.cancelled), a.count)},
        wait: {label: 'Cumulative request hours', unit: 'h', calc: a => a.duration / 3600},
        count: {label: 'Requests', unit: '', calc: a => a.count},
        output: {label: 'Output tokens', unit: '', calc: a => a.output},
    };

    async function api(url, options) {
        const response = await fetch(url, options);
        const data = await response.json();
        if (!response.ok) throw new Error(data.message || data.error || `HTTP ${response.status}`);
        return data;
    }
    function notice(message, error = false) {
        const node = text('div', message, $('notices'));
        node.className = 'notice' + (error ? ' error' : '');
    }
    function selectionSummary() {
        const size = state.files.filter(f => state.selected.has(f.name)).reduce((n, f) => n + f.size_bytes, 0);
        $('selection').textContent = `${state.selected.size} files selected · ${bytes(size)}`;
        $('generate').disabled = !state.selected.size || state.selected.size > 366 || !!state.job || state.submitting;
    }
    function renderFiles() {
        $('file-list').replaceChildren();
        if (!state.files.length) text('p', 'No daily logs found. Enable save_request_to_file to record new requests.', $('file-list'));
        for (const file of state.files) {
            const label = document.createElement('label'); label.className = 'file-row';
            const input = document.createElement('input'); input.type = 'checkbox'; input.checked = state.selected.has(file.name);
            input.onchange = () => { input.checked ? state.selected.add(file.name) : state.selected.delete(file.name); selectionSummary(); };
            label.appendChild(input);
            text('span', file.name, label);
            text('small', `${bytes(file.size_bytes)} · ${file.index_status}`, label);
            $('file-list').appendChild(label);
        }
        selectionSummary();
    }
    async function loadFiles() {
        $('refresh-files').disabled = true;
        try {
            state.files = (await api('/api/request-stats/files')).files;
            const names = new Set(state.files.map(f => f.name));
            state.selected = new Set([...state.selected].filter(name => names.has(name)));
            renderFiles();
        } catch (error) { notice(`Could not list files: ${error.message}`, true); }
        finally { $('refresh-files').disabled = false; }
    }
    function renderJob(job) {
        $('job-panel').hidden = false;
        $('job-title').textContent = job.current_mode === 'aggregating' ? 'Aggregating lightweight indexes…' : `${job.current_file || 'Performance job'} · ${job.current_mode || job.status}`;
        $('job-meta').textContent = `${job.files_completed}/${job.files_total} files · ${bytes(job.bytes_processed)} / ${bytes(job.bytes_total)} · ${job.cache_hits} cached · ${job.incremental_scans} incremental · ${job.rebuild_scans} rebuilt`;
        if (job.current_mode === 'aggregating' || !job.bytes_total) $('progress').removeAttribute('value');
        else $('progress').value = Math.min(100, 100 * job.bytes_processed / job.bytes_total);
        $('cancel').disabled = terminal.has(job.status) || job.cancel_requested;
    }
    function finishJob(job) {
        state.job = null;
        $('report').classList.remove('busy');
        selectionSummary();
        if (job.status === 'completed') {
            $('job-title').textContent = 'Performance report ready';
            $('progress').value = 100;
            showResult(job.result);
            loadFiles();
        } else {
            notice(job.status === 'cancelled' ? 'Scan cancelled. Completed indexes are preserved; any report below is the previous result.' : `Scan failed: ${job.error || 'Unknown error'}. Any report below is the previous result.`, job.status === 'failed');
        }
    }
    function schedulePoll() { clearTimeout(state.timer); state.timer = setTimeout(pollJob, 700); }
    async function pollJob() {
        if (!state.job) return;
        try {
            const {job} = await api(`/api/request-stats/jobs/${encodeURIComponent(state.job)}`);
            renderJob(job);
            if (terminal.has(job.status)) finishJob(job); else schedulePoll();
        } catch (error) {
            notice(`Progress unavailable: ${error.message}. Analyze the same files again to reconnect if the job is still running.`, true);
            state.job = null; $('cancel').disabled = true; $('report').classList.remove('busy'); selectionSummary();
        }
    }
    async function startJob() {
        if (state.submitting || state.job || !state.selected.size || state.selected.size > 366) return;
        state.submitting = true; selectionSummary(); $('notices').replaceChildren();
        $('report').classList.add('busy');
        try {
            const {job} = await api('/api/request-stats/jobs', {method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({files: [...state.selected], kind: 'performance'})});
            state.job = job.id; renderJob(job);
            if (terminal.has(job.status)) finishJob(job); else schedulePoll();
        } catch (error) { notice(`Unable to start analysis: ${error.message}`, true); $('report').classList.remove('busy'); }
        finally { state.submitting = false; selectionSummary(); }
    }
    async function cancelJob() {
        if (!state.job) return;
        $('cancel').disabled = true;
        try { await api(`/api/request-stats/jobs/${encodeURIComponent(state.job)}/cancel`, {method: 'POST'}); schedulePoll(); }
        catch (error) { notice(`Unable to cancel: ${error.message}`, true); $('cancel').disabled = false; }
    }
    function setRange(days) {
        const dates = state.data.dates;
        state.to = dates[dates.length - 1];
        const cutoff = days ? M.shiftDate(state.to, 1 - days) : dates[0];
        state.from = dates.find(date => date >= cutoff) || state.to;
        $('from').value = state.from; $('to').value = state.to;
        render();
    }
    function showResult(result) {
        state.data = result.perf;
        const q = result.quality;
        const skipped = q.invalid_json_lines + q.non_object_lines + q.oversize_lines;
        if (skipped || q.partial_tail_bytes) notice(`Source quality: ${skipped} skipped lines (${q.invalid_json_lines} invalid JSON, ${q.non_object_lines} non-objects, ${q.oversize_lines} oversized); ${q.partial_tail_bytes} partial-tail bytes deferred.`);
        for (const warning of result.warnings || []) notice(`${warning.file}: ${warning.message}`);
        for (const skip of result.skipped_files || []) notice(`Skipped file ${skip.file}: ${skip.message}`);
        const descriptions = {date_fallback: 'timestamps missing/invalid (file date used)', missing_duration: 'durations unavailable',
            missing_output: 'output token counts unavailable', missing_context: 'context sizes unavailable', invalid_ttft: 'inconsistent TTFT values ignored',
            duration_hist_clipped: 'durations outside histogram range', tps_hist_clipped: 'rates outside histogram range'};
        const issues = Object.entries(result.perf.quality).filter(([, n]) => n).map(([key, n]) => `${n} ${descriptions[key] || key}`);
        if (issues.length) notice('Metric quality: ' + issues.join('; ') + '.');
        if (result.perf.model_overflow) notice('Excess model names were grouped as __other__.');
        $('report').hidden = !state.data.dates.length;
        if (!state.data.dates.length) { notice('No valid indexed requests in the selected files.'); return; }
        $('report-meta').textContent = `${number(state.data.total_rows, 0)} requests · ${result.files.length} files · ${state.data.timezone} · generated ${result.generated_at}`;
        for (const id of ['from', 'to']) {
            $(id).replaceChildren();
            for (const date of state.data.dates) { const option = text('option', date, $(id)); option.value = date; }
        }
        const totals = new Map(state.data.models.map(model => [model, 0]));
        for (const cell of state.data.cells) totals.set(cell.model, totals.get(cell.model) + cell.output);
        state.order = [...state.data.models].sort((a, b) => totals.get(b) - totals.get(a) || a.localeCompare(b));
        state.colors = new Map(state.order.map((model, i) => [model, colors[i % colors.length]]));
        state.models = new Set(state.order.slice(0, 6));
        setRange(30);
    }

    // Native SVG title tooltips and text nodes keep logged labels out of HTML.
    const NS = 'http://www.w3.org/2000/svg';
    function svgNode(tag, attrs, parent, value) {
        const node = document.createElementNS(NS, tag);
        for (const [key, val] of Object.entries(attrs)) node.setAttribute(key, val);
        if (value != null) node.textContent = value;
        parent.appendChild(node); return node;
    }
    function plot(id, height = 260) {
        const host = $(id); host.replaceChildren();
        const width = Math.max(320, host.clientWidth || 800);
        const svg = svgNode('svg', {viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': id}, host);
        return {host, svg, width, height, left: 55, top: 15, iw: width - 70, ih: height - 50};
    }
    function axes(p, maximum, unit = '') {
        maximum = Math.max(1, maximum);
        for (let i = 0; i <= 4; i++) {
            const y = p.top + p.ih * (1 - i / 4);
            svgNode('line', {x1: p.left, x2: p.left + p.iw, y1: y, y2: y, class: 'gridline'}, p.svg);
            svgNode('text', {x: p.left - 5, y: y + 3, 'text-anchor': 'end', class: 'axis'}, p.svg, compact(maximum * i / 4));
        }
        if (unit) svgNode('text', {x: p.left, y: 10, class: 'axis'}, p.svg, unit);
        return value => p.top + p.ih * (1 - value / maximum);
    }
    function legend(host, names) {
        const box = document.createElement('div'); box.className = 'legend';
        for (const name of names) { const label = text('span', name, box); label.style.borderColor = state.colors.get(name); }
        host.appendChild(box);
    }
    function xDates(p, dates) {
        const start = M.dateMs(state.from), span = M.dateMs(state.to) - start;
        const x = date => p.left + (span ? (M.dateMs(date) - start) / span : .5) * p.iw;
        const step = Math.max(1, Math.ceil(dates.length / 10));
        dates.forEach((date, i) => { if (i % step === 0) svgNode('text', {x: x(date), y: p.height - 10, 'text-anchor': 'middle', class: 'axis'}, p.svg, date.slice(5)); });
        return x;
    }
    function line(p, points, color) {
        let run = [];
        function flush() {
            if (run.length > 1) svgNode('path', {d: 'M' + run.map(pt => `${pt[0]} ${pt[1]}`).join('L'), fill: 'none', stroke: color, 'stroke-width': 2}, p.svg);
            run = [];
        }
        for (const point of points) { if (point) run.push(point); else flush(); }
        flush();
    }
    function renderTrend() {
        const metric = metrics[$('trend-metric').value];
        $('trend-hint').textContent = metric.hint || 'Missing metrics leave gaps. All values use the selected models and dates.';
        const dates = state.data.dates.filter(date => date >= state.from && date <= state.to);
        const lookup = new Map(selected(state.data.cells).map(c => [JSON.stringify([c.date, c.model]), c]));
        const series = selectedModels().map(model => ({model, values: dates.map(date => {
            const cell = lookup.get(JSON.stringify([date, model])); return cell ? metric.calc(cell) : null;
        })}));
        const p = plot('trend', 300);
        let maximum = 1; series.forEach(s => s.values.forEach(v => { if (v != null) maximum = Math.max(maximum, v); }));
        const y = axes(p, maximum * 1.05, metric.unit), x = xDates(p, dates);
        for (const s of series) {
            const points = [];
            s.values.forEach((v, i) => {
                if (i && M.dateMs(dates[i]) - M.dateMs(dates[i - 1]) > 86400000) points.push(null);
                points.push(v == null ? null : [x(dates[i]), y(v)]);
            });
            line(p, points, state.colors.get(s.model));
            s.values.forEach((v, i) => {
                if (v == null) return;
                const circle = svgNode('circle', {cx: x(dates[i]), cy: y(v), r: 3, fill: state.colors.get(s.model)}, p.svg);
                svgNode('title', {}, circle, `${dates[i]} · ${s.model}: ${number(v, 2)} ${metric.unit}`);
            });
        }
        legend(p.host, selectedModels());
    }
    function renderVolume() {
        const mode = $('volume-metric').value;
        const dates = state.data.dates.filter(date => date >= state.from && date <= state.to);
        const groups = new Map(dates.map(date => [date, []]));
        for (const cell of selected(state.data.cells)) groups.get(cell.date).push(cell);
        const value = c => mode === 'wait' ? c.duration / 3600 : mode === 'context' ? contextTotal(c) : c[mode];
        const totals = dates.map(date => groups.get(date).reduce((n, c) => n + value(c), 0));
        const p = plot('volume'), y = axes(p, Math.max(1, ...totals));
        const countDays = (M.dateMs(state.to) - M.dateMs(state.from)) / 86400000 + 1;
        // Bar centers use calendar positions; missing dates remain empty gaps.
        const width = Math.max(.5, p.iw / countDays * .8);
        const x = date => p.left + ((M.dateMs(date) - M.dateMs(state.from)) / 86400000 + .5) / countDays * p.iw;
        dates.forEach((date, i) => {
            let bottom = 0;
            for (const cell of groups.get(date)) {
                const v = value(cell);
                const rect = svgNode('rect', {x: x(date) - width / 2, y: y(bottom + v), width, height: y(bottom) - y(bottom + v), fill: state.colors.get(cell.model)}, p.svg);
                svgNode('title', {}, rect, `${date} · ${cell.model}: ${compact(v)}`); bottom += v;
            }
            if (i % Math.max(1, Math.ceil(dates.length / 8)) === 0) svgNode('text', {x: x(date), y: p.height - 10, 'text-anchor': 'middle', class: 'axis'}, p.svg, date.slice(5));
        });
        legend(p.host, selectedModels());
    }
    function renderHeat() {
        const grouped = new Map();
        for (const row of selected(state.data.hourly)) {
            const key = row.date + '/' + row.hour;
            if (!grouped.has(key)) grouped.set(key, []);
            grouped.get(key).push(row);
        }
        const metric = metrics[$('heat-metric').value];
        const values = new Map([...grouped].map(([key, rows]) => [key, metric.calc(sum(rows))]));
        let maximum = 1; for (const v of values.values()) if (v != null) maximum = Math.max(maximum, v);
        const p = plot('heat', 300), dates = state.data.dates.filter(date => date >= state.from && date <= state.to);
        const cw = p.iw / Math.max(1, dates.length), ch = p.ih / 24;
        dates.forEach((date, i) => {
            for (let hour = 0; hour < 24; hour++) {
                const value = values.get(date + '/' + hour);
                const fill = value == null ? '#eee' : `hsl(170 60% ${92 - 60 * value / maximum}%)`;
                const rect = svgNode('rect', {x: p.left + i * cw, y: p.top + hour * ch, width: Math.max(.3, cw - .5), height: ch - .5, fill}, p.svg);
                svgNode('title', {}, rect, `${date} ${String(hour).padStart(2, '0')}:00 · ${metric.label}: ${number(value, 2)} ${metric.unit}`);
            }
            if (i % Math.max(1, Math.ceil(dates.length / 8)) === 0) svgNode('text', {x: p.left + (i + .5) * cw, y: p.height - 10, 'text-anchor': 'middle', class: 'axis'}, p.svg, date.slice(5));
        });
        for (let hour = 0; hour < 24; hour += 3) svgNode('text', {x: p.left - 5, y: p.top + (hour + .8) * ch, 'text-anchor': 'end', class: 'axis'}, p.svg, hour + ':00');
    }
    function renderContext() {
        const metric = metrics[$('context-metric').value], data = state.data;
        const models = selectedModels();
        const series = models.map(model => {
            const samples = sum(selected(data.cells).filter(c => c.model === model)).samples;
            return data.context_labels.map((_, bucket) => {
                const group = sum(selected(data.contexts).filter(r => r.model === model && r.bucket === bucket));
                group.samples = samples.filter(p => p[2] != null && data.context_bounds.filter(bound => p[2] >= bound).length === bucket);
                return group.count ? metric.calc(group) : null;
            });
        });
        let maximum = 1; series.forEach(values => values.forEach(v => { if (v != null) maximum = Math.max(maximum, v); }));
        const p = plot('context'), y = axes(p, maximum, metric.unit);
        const groupWidth = p.iw / data.context_labels.length, width = groupWidth * .8 / Math.max(1, models.length);
        data.context_labels.forEach((label, bucket) => {
            const center = p.left + (bucket + .5) * groupWidth;
            svgNode('text', {x: center, y: p.height - 10, 'text-anchor': 'middle', class: 'axis'}, p.svg, label);
            series.forEach((values, i) => {
                const v = values[bucket]; if (v == null) return;
                const rect = svgNode('rect', {x: center - groupWidth * .4 + i * width, y: y(v), width: Math.max(.3, width - 1), height: p.top + p.ih - y(v), fill: state.colors.get(models[i])}, p.svg);
                svgNode('title', {}, rect, `${models[i]} · ${label}: ${number(v, 2)} ${metric.unit}`);
            });
        });
        legend(p.host, models);
    }
    function renderScatter(groups) {
        const selectedGroups = groups.slice(0, 6).map(g => ({...g, points: M.sampleEvenly(g.data.samples, 300), fit: M.fit(g.data.samples)}));
        let maxX = 1, maxY = 1;
        for (const g of selectedGroups) for (const pt of g.points) { maxX = Math.max(maxX, pt[1]); maxY = Math.max(maxY, pt[0]); }
        for (const g of selectedGroups) if (g.fit) maxY = Math.max(maxY, g.fit.ttft + maxX / g.fit.decode);
        const p = plot('scatter'), y = axes(p, maxY, 'duration (s)'), x = value => p.left + value / maxX * p.iw;
        for (let i = 0; i <= 4; i++) svgNode('text', {x: x(maxX * i / 4), y: p.height - 10, 'text-anchor': 'middle', class: 'axis'}, p.svg, compact(maxX * i / 4));
        for (const g of selectedGroups) {
            for (const pt of g.points) {
                const circle = svgNode('circle', {cx: x(pt[1]), cy: y(pt[0]), r: 2.5, fill: state.colors.get(g.model), opacity: .6}, p.svg);
                svgNode('title', {}, circle, `${g.model} · ${number(pt[1], 0)} output tokens · ${number(pt[0], 2)} s · context ${compact(pt[2])}`);
            }
            if (g.fit) svgNode('line', {x1: x(0), x2: x(maxX), y1: y(g.fit.ttft), y2: y(g.fit.ttft + maxX / g.fit.decode), stroke: state.colors.get(g.model), 'stroke-dasharray': '5 3'}, p.svg);
        }
        legend(p.host, selectedGroups.map(g => g.model));
        text('small', 'x: output tokens · y: duration (s)', p.host);
    }
    function renderDistribution(groups) {
        const edges = state.data.duration_edges, bins = edges.length - 1;
        const p = plot('distribution'), series = groups.map(g => {
            const total = g.data.duration_hist.reduce((n, v) => n + v, 0);
            return {model: g.model, counts: g.data.duration_hist, values: g.data.duration_hist.map(v => total ? 100 * v / total : 0)};
        });
        let maximum = 1; series.forEach(s => s.values.forEach(v => { maximum = Math.max(maximum, v); }));
        const y = axes(p, maximum, '%'), x = i => p.left + i / (bins - 1) * p.iw;
        for (let i = 0; i < bins; i += 8) svgNode('text', {x: x(i), y: p.height - 10, 'text-anchor': 'middle', class: 'axis'}, p.svg, number(edges[i], 2) + 's');
        for (const s of series) {
            line(p, s.values.map((v, i) => [x(i), y(v)]), state.colors.get(s.model));
            s.values.forEach((v, i) => {
                const point = svgNode('circle', {cx: x(i), cy: y(v), r: 2.5, fill: state.colors.get(s.model)}, p.svg);
                svgNode('title', {}, point, `${s.model} · ${number(edges[i], 2)}–${number(edges[i + 1], 2)} s: ${s.counts[i]} requests (${number(v)}%)`);
            });
        }
        legend(p.host, groups.map(g => g.model));
    }

    function table(id, headings, rows) {
        const host = $(id); host.replaceChildren();
        const sort = state.sorts[id];
        if (sort) rows.sort((a, b) => {
            const x = a[sort.index], y = b[sort.index];
            if (x == null) return y == null ? 0 : 1;
            if (y == null) return -1;
            return sort.direction * (typeof x === 'number' ? x - y : String(x).localeCompare(String(y)));
        });
        const header = document.createElement('thead'), tr = document.createElement('tr'); header.appendChild(tr); host.appendChild(header);
        headings.forEach((heading, index) => {
            const th = document.createElement('th'); tr.appendChild(th);
            const button = text('button', heading, th); button.type = 'button';
            button.onclick = () => { state.sorts[id] = {index, direction: sort?.index === index ? -sort.direction : -1}; render(); };
        });
        const body = document.createElement('tbody'); host.appendChild(body);
        for (const row of rows) {
            const tr = document.createElement('tr'); body.appendChild(tr);
            row.forEach(value => text('td', typeof value === 'number' ? number(value, 2) : value ?? '—', tr));
        }
    }
    function renderTables(groups) {
        table('model-table', ['Model', 'Requests', 'Fail + cancel %', 'Output', 'Mean context', 'Hours', 'Effective TPS', 'TPS median ≈', 'TTFT (s)', 'TTFT source', 'Measured TTFT n', 'Decode TPS', 'Decode source', 'Measured decode n', 'Duration P50 ≈', 'P90 ≈', 'P99 ≈'], groups.map(g => {
            const a = g.data, s = M.summary(a, state.data);
            return [g.model, a.count, s.errors, a.output, s.context, a.duration / 3600, s.tps, s.tpsMedian,
                s.ttft, s.ttftSource, a.ttft_count, s.decode, s.decodeSource, a.decode_count,
                metrics.p50.calc(a), s.p90, metrics.p99.calc(a)];
        }));
        for (const [id, source] of [['upstream-table', state.data.upstream], ['endpoint-table', state.data.endpoints]]) {
            const byName = new Map();
            for (const row of selected(source)) { if (!byName.has(row.name)) byName.set(row.name, []); byName.get(row.name).push(row); }
            table(id, ['Name', 'Requests', 'Fail + cancel %', 'Output', 'Effective TPS', 'Mean duration (s)', 'Hours'], [...byName].map(([name, rows]) => {
                const a = sum(rows); return [name, a.count, metrics.errors.calc(a), a.output, metrics.tps.calc(a), metrics.duration.calc(a), a.duration / 3600];
            }));
        }
    }
    function renderKpis(all) {
        const s = M.summary(all, state.data), host = $('kpis'); host.replaceChildren();
        const values = [
            ['Requests', number(all.count, 0), `${state.models.size} selected models`],
            ['Cumulative request time', number(all.duration / 3600, 2) + ' h', `${all.duration_count} known durations (not wall-clock)`],
            ['Output tokens', compact(all.output), 'All attempts with logged usage'],
            ['Context tokens', compact(contextTotal(all)), `Cache read share: ${number(M.ratio(100 * all.cache_read, contextTotal(all)))}%`],
            ['Effective TPS', number(s.tps), `${all.gen_count} successful generating requests`],
            ['TPS median ≈', number(s.tpsMedian), `P10 ≈ ${number(metrics.tps_p10.calc(all))}`],
            [`TTFT (${s.ttftSource})`, number(s.ttft, 2) + ' s', `${all.ttft_count} requests with measured timing`],
            [`Decode TPS (${s.decodeSource})`, number(s.decode), `${all.decode_count} requests with measured decode intervals`],
            ['Duration P50 / P90 ≈', `${number(metrics.p50.calc(all))} / ${number(s.p90)} s`, `P99 ≈ ${number(metrics.p99.calc(all))} s`],
            ['Failures + cancellations', number(s.errors, 2) + '%', `${all.errors} failures · ${all.cancelled} cancellations · ${all.unknown_status} unknown status`],
        ];
        for (const [label, value, note] of values) {
            const item = document.createElement('div'); item.className = 'kpi';
            text('small', label, item); text('b', value, item); text('small', note, item); host.appendChild(item);
        }
    }
    function renderDiagnosis() {
        const host = $('diagnosis'); host.replaceChildren();
        const days = Math.floor((M.dateMs(state.to) - M.dateMs(state.from)) / 86400000) + 1;
        if (days < 2) { $('diagnosis-hint').textContent = 'Select at least two calendar days for period comparisons.'; return; }
        const windowDays = Math.min(7, Math.floor(days / 2));
        const recentStart = M.shiftDate(state.to, 1 - windowDays), prevEnd = M.shiftDate(recentStart, -1), prevStart = M.shiftDate(recentStart, -windowDays);
        $('diagnosis-hint').textContent = `Comparing ${recentStart}–${state.to} with ${prevStart}–${prevEnd} (${windowDays} calendar days each). At least 20 generating requests per period are required.`;
        let compared = 0, changed = 0;
        for (const model of selectedModels()) {
            const rows = state.data.cells.filter(c => c.model === model);
            const recent = sum(rows.filter(c => c.date >= recentStart && c.date <= state.to));
            const previous = sum(rows.filter(c => c.date >= prevStart && c.date <= prevEnd));
            if (recent.gen_count < 20 || previous.gen_count < 20) continue;
            compared++;
            const r = M.summary(recent, state.data), p = M.summary(previous, state.data), delta = M.change(r.tps, p.tps);
            if (delta == null || Math.abs(delta) <= 12) continue;
            changed++;
            const indicators = [];
            if (M.change(r.context, p.context) > 15) indicators.push(`context +${number(M.change(r.context, p.context))}%`);
            if (r.ttftSource === p.ttftSource && M.change(r.ttft, p.ttft) > 15) indicators.push(`${r.ttftSource} TTFT +${number(M.change(r.ttft, p.ttft))}%`);
            const decodeChange = M.change(r.decode, p.decode);
            if (r.decodeSource === p.decodeSource && decodeChange != null && decodeChange < -10) indicators.push(`${r.decodeSource} decode speed ${number(decodeChange)}%`);
            if (M.change(r.output, p.output) > 25) indicators.push(`output length +${number(M.change(r.output, p.output))}%`);
            if (r.errors > p.errors + 1) indicators.push(`failure/cancellation rate ${number(p.errors)}% → ${number(r.errors)}%`);
            text('li', `${model}: effective TPS ${number(p.tps)} → ${number(r.tps)} (${delta > 0 ? '+' : ''}${number(delta)}%). ${indicators.length ? 'Concurrent changes: ' + indicators.join('; ') + '.' : 'No clear accompanying change in the available metrics.'}`, host);
        }
        if (!changed) text('li', compared ? 'No effective TPS change above 12% in models with enough samples.' : 'Insufficient comparable data. Missing days are not treated as zero-valued measurements.', host);
    }
    function render() {
        if (!state.data?.dates.length) return;
        $('model-chips').replaceChildren();
        for (const model of state.order) {
            const button = text('button', model, $('model-chips')); button.type = 'button'; button.className = 'chip';
            button.style.setProperty('--model-color', state.colors.get(model)); button.setAttribute('aria-pressed', state.models.has(model));
            button.onclick = () => { state.models.has(model) ? state.models.delete(model) : state.models.add(model); render(); };
        }
        const cells = selected(state.data.cells), groups = selectedModels().map(model => ({model, data: sum(cells.filter(c => c.model === model))}));
        renderKpis(sum(cells)); renderDiagnosis(); renderTrend(); renderVolume(); renderHeat(); renderContext();
        renderScatter(groups); renderDistribution(groups); renderTables(groups);
    }

    $('refresh-files').onclick = loadFiles;
    $('select-all').onclick = () => { state.selected = new Set(state.files.slice(0, 366).map(f => f.name)); renderFiles(); };
    $('clear-files').onclick = () => { state.selected.clear(); renderFiles(); };
    $('generate').onclick = startJob; $('cancel').onclick = cancelJob;
    for (const n of [7, 30]) $('range-' + n).onclick = () => setRange(n);
    $('range-all').onclick = () => setRange(0);
    $('from').onchange = () => { state.from = $('from').value; if (state.from > state.to) state.to = state.from; $('to').value = state.to; render(); };
    $('to').onchange = () => { state.to = $('to').value; if (state.to < state.from) state.from = state.to; $('from').value = state.from; render(); };
    $('all-models').onclick = () => { state.models = new Set(state.order); render(); };
    $('clear-models').onclick = () => { state.models.clear(); render(); };
    for (const [key, metric] of Object.entries(metrics)) { const option = text('option', metric.label, $('trend-metric')); option.value = key; }
    $('trend-metric').value = 'tps'; $('trend-metric').onchange = renderTrend;
    $('volume-metric').onchange = renderVolume; $('heat-metric').onchange = renderHeat; $('context-metric').onchange = renderContext;
    let resizeTimer;
    window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(render, 200); });
    loadFiles();
})();
