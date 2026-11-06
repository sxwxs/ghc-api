/* Pure calculations for the historical dashboard; no DOM or dependencies. */
(function (root) {
    'use strict';

    function aggregate(rows, fields) {
        const result = Object.fromEntries(fields.map(key => [key, 0]));
        result.duration_hist = [];
        result.tps_hist = [];
        result.samples = [];
        for (const row of rows) {
            for (const key of fields) result[key] += row[key] || 0;
            for (const key of ['duration_hist', 'tps_hist']) {
                (row[key] || []).forEach((value, i) => { result[key][i] = (result[key][i] || 0) + value; });
            }
            for (const sample of row.samples || []) result.samples.push(sample);
        }
        return result;
    }

    function percentile(histogram, edges, q) {
        const total = histogram.reduce((sum, n) => sum + n, 0);
        if (!total) return null;
        const target = q * total;
        let count = 0;
        for (let i = 0; i < histogram.length; i++) {
            if (histogram[i] > 0 && count + histogram[i] >= target) {
                const fraction = (target - count) / histogram[i];
                return Math.exp(Math.log(edges[i]) + fraction * Math.log(edges[i + 1] / edges[i]));
            }
            count += histogram[i];
        }
        return null;
    }

    function sampleEvenly(points, limit) {
        if (points.length <= limit) return points;
        return Array.from({length: limit}, (_, i) => points[Math.floor(i * points.length / limit)]);
    }

    // Theil–Sen over bounded day/model reservoirs: duration ≈ intercept + output / rate.
    function fit(samples) {
        if (samples.length < 10) return null;
        const points = sampleEvenly(samples, 220);
        const slopes = [];
        for (let i = 0; i < points.length; i++) {
            for (let j = i + 1; j < points.length; j++) {
                const dx = points[j][1] - points[i][1];
                if (Math.abs(dx) >= 5) slopes.push((points[j][0] - points[i][0]) / dx);
            }
        }
        if (slopes.length < 20) return null;
        slopes.sort((a, b) => a - b);
        const slope = slopes[slopes.length >> 1];
        if (!(slope > 1e-6)) return null;
        const intercepts = points.map(p => p[0] - slope * p[1]).sort((a, b) => a - b);
        return {ttft: Math.max(0, intercepts[intercepts.length >> 1]), decode: 1 / slope, count: points.length};
    }

    const ratio = (a, b) => b > 0 ? a / b : null;
    const change = (now, before) => now != null && before > 0 ? 100 * (now - before) / before : null;
    const dateMs = date => Date.parse(date + 'T00:00:00Z');
    const shiftDate = (date, days) => new Date(dateMs(date) + days * 86400000).toISOString().slice(0, 10);

    function summary(group, data) {
        const estimated = fit(group.samples);
        return {
            tps: ratio(group.gen_output, group.gen_duration),
            tpsMedian: percentile(group.tps_hist, data.tps_edges, 0.5),
            ttft: group.ttft_count ? group.ttft_sum / group.ttft_count : estimated?.ttft ?? null,
            ttftSource: group.ttft_count ? 'measured' : estimated ? 'estimated' : 'unavailable',
            decode: group.decode_count ? ratio(group.decode_output, group.decode_duration) : estimated?.decode ?? null,
            decodeSource: group.decode_count ? 'measured' : estimated ? 'estimated' : 'unavailable',
            context: ratio(group.context_sum, group.context_count),
            output: ratio(group.gen_output, group.gen_count),
            errors: ratio(100 * (group.errors + group.cancelled), group.count),
            p90: percentile(group.duration_hist, data.duration_edges, 0.9),
        };
    }

    const api = {aggregate, percentile, sampleEvenly, fit, ratio, change, dateMs, shiftDate, summary};
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else root.RequestPerfMath = api;
})(typeof window !== 'undefined' ? window : globalThis);
