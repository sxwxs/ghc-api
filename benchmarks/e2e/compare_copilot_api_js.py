"""Compare ghc-api with puxu-msft/copilot-api-js against one fake backend."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import requests
import yaml

try:
    import psutil
except ImportError:
    psutil = None

from .runner import (
    RESULTS,
    ROOT,
    _free_port,
    _process_metrics_end,
    _process_metrics_start,
    _run_load,
    _scenario_payloads,
    _start_process,
    _stop_process,
    _write_config,
)


def _wait_200(url: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=0.5)
            last = response.status_code
            if response.status_code == 200:
                return
        except requests.RequestException as exc:
            last = exc
        time.sleep(0.1)
    raise RuntimeError(f"Timed out waiting for HTTP 200 from {url}: {last}")


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


SECURITY_PROCESS_MARKERS = ("msmpeng", "mssense", "sense", "securityhealth")


def _security_processes():
    if psutil is None:
        return []
    return [
        process for process in psutil.process_iter(["pid", "name"])
        if any(marker in (process.info.get("name") or "").lower() for marker in SECURITY_PROCESS_MARKERS)
    ]


def _wait_for_idle(max_security_cpu: float, max_system_cpu: float, stable_seconds: int, timeout: int) -> Dict:
    if psutil is None:
        return {"skipped": "psutil unavailable"}
    security = _security_processes()
    for process in security:
        try:
            process.cpu_percent(None)
        except psutil.Error:
            pass
    psutil.cpu_percent(None)
    deadline = time.time() + timeout
    stable = 0
    samples = []
    while time.time() < deadline:
        time.sleep(1)
        system_cpu = psutil.cpu_percent(None)
        security_cpu = 0.0
        for process in security:
            try:
                security_cpu += process.cpu_percent(None)
            except psutil.Error:
                pass
        samples.append({"system_cpu_percent": system_cpu, "security_cpu_percent": security_cpu})
        samples = samples[-stable_seconds:]
        if system_cpu <= max_system_cpu and security_cpu <= max_security_cpu:
            stable += 1
            if stable >= stable_seconds:
                return {"stable_seconds": stable_seconds, "samples": samples}
        else:
            stable = 0
    raise RuntimeError(
        f"Machine did not become idle within {timeout}s "
        f"(thresholds: security<={max_security_cpu}%, system<={max_system_cpu}%)"
    )


def _run_measured_load(process, url: str, payload: Dict, requests_count: int, concurrency: int, warmup: int):
    metrics_start = _process_metrics_start(process.pid)
    if psutil is None:
        proxy = _run_load(url, payload, requests_count, concurrency, warmup)
        return proxy, _process_metrics_end(metrics_start, requests_count)

    target = psutil.Process(process.pid)
    security = _security_processes()
    target.cpu_percent(None)
    for item in security:
        try:
            item.cpu_percent(None)
        except psutil.Error:
            pass
    psutil.cpu_percent(None)
    samples = []
    stop = threading.Event()

    def sample():
        while not stop.wait(0.2):
            try:
                target_cpu = target.cpu_percent(None)
                rss = target.memory_info().rss
            except psutil.Error:
                break
            security_cpu = 0.0
            for item in security:
                try:
                    security_cpu += item.cpu_percent(None)
                except psutil.Error:
                    pass
            samples.append({
                "target_cpu_percent": target_cpu,
                "target_rss_bytes": rss,
                "system_cpu_percent": psutil.cpu_percent(None),
                "security_cpu_percent": security_cpu,
            })

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        proxy = _run_load(url, payload, requests_count, concurrency, warmup)
    finally:
        stop.set()
        thread.join(timeout=1)
    metrics = _process_metrics_end(metrics_start, requests_count)
    elapsed = proxy["elapsed_s"]
    cpu_seconds = metrics.get("cpu_seconds", 0)
    logical_cpus = psutil.cpu_count(logical=True) or 1
    metrics.update({
        "cpu_core_equivalent_percent": cpu_seconds / elapsed * 100 if elapsed else 0,
        "cpu_machine_capacity_percent": cpu_seconds / elapsed / logical_cpus * 100 if elapsed else 0,
        "cpu_sample_peak_percent": max((sample["target_cpu_percent"] for sample in samples), default=0),
        "rss_peak_bytes": max((sample["target_rss_bytes"] for sample in samples), default=metrics.get("rss_after_bytes", 0)),
        "system_cpu_average_percent": sum(sample["system_cpu_percent"] for sample in samples) / len(samples) if samples else 0,
        "system_cpu_peak_percent": max((sample["system_cpu_percent"] for sample in samples), default=0),
        "security_cpu_average_percent": sum(sample["security_cpu_percent"] for sample in samples) / len(samples) if samples else 0,
        "security_cpu_peak_percent": max((sample["security_cpu_percent"] for sample in samples), default=0),
        "sample_count": len(samples),
    })
    return proxy, metrics


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _aggregate(raw_runs: List[Dict]) -> List[Dict]:
    groups: Dict[tuple, List[Dict]] = {}
    for run in raw_runs:
        groups.setdefault((run["implementation"], run["scenario"], run["concurrency"]), []).append(run)

    rows = []
    for (implementation, scenario, concurrency), trials in sorted(groups.items()):
        rows.append({
            "implementation": implementation,
            "scenario": scenario,
            "concurrency": concurrency,
            "trials": len(trials),
            "direct_p50_ms": _median([trial["direct"]["latency_ms"]["p50"] for trial in trials]),
            "proxy_p50_ms": _median([trial["proxy"]["latency_ms"]["p50"] for trial in trials]),
            "overhead_p50_ms": _median([
                trial["proxy"]["latency_ms"]["p50"] - trial["direct"]["latency_ms"]["p50"] for trial in trials
            ]),
            "proxy_p95_ms": _median([trial["proxy"]["latency_ms"]["p95"] for trial in trials]),
            "ttfb_overhead_p50_ms": _median([
                trial["proxy"]["ttfb_ms"]["p50"] - trial["direct"]["ttfb_ms"]["p50"] for trial in trials
            ]),
            "proxy_rps": _median([trial["proxy"]["rps"] for trial in trials]),
            "throughput_ratio": _median([trial["proxy"]["rps"] / trial["direct"]["rps"] for trial in trials]),
            "cpu_ms_per_request": _median([trial["process"].get("cpu_ms_per_request", 0) for trial in trials]),
            "cpu_core_equivalent_percent": _median([trial["process"].get("cpu_core_equivalent_percent", 0) for trial in trials]),
            "cpu_sample_peak_percent": _median([trial["process"].get("cpu_sample_peak_percent", 0) for trial in trials]),
            "rss_after_mb": _median([trial["process"].get("rss_after_bytes", 0) / 1024 / 1024 for trial in trials]),
            "rss_peak_mb": _median([trial["process"].get("rss_peak_bytes", 0) / 1024 / 1024 for trial in trials]),
            "system_cpu_average_percent": _median([trial["process"].get("system_cpu_average_percent", 0) for trial in trials]),
            "security_cpu_average_percent": _median([trial["process"].get("security_cpu_average_percent", 0) for trial in trials]),
            "security_cpu_peak_percent": _median([trial["process"].get("security_cpu_peak_percent", 0) for trial in trials]),
        })
    return rows


def _markdown(result: Dict) -> str:
    lines = [
        "# ghc-api vs copilot-api-js E2E 性能对比",
        "",
        f"生成时间：{result['generated_at']}",
        "",
        f"copilot-api-js commit：`{result['copilot_api_js']['commit']}`（{result['copilot_api_js']['commit_date']}）",
        "",
        "## 汇总",
        "",
        "| 实现 | 场景 | 并发 | Proxy p50 | 配对 p50 overhead | RPS | CPU/request | 平均占用核 | CPU peak | RSS peak | Security CPU avg |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["summary"]:
        lines.append(
            f"| {row['implementation']} | {row['scenario']} | {row['concurrency']} | "
            f"{row['proxy_p50_ms']:.2f} ms | {row['overhead_p50_ms']:+.2f} ms | "
            f"{row['proxy_rps']:.1f} | {row['cpu_ms_per_request']:.3f} ms | "
            f"{row['cpu_core_equivalent_percent'] / 100:.2f} | {row['cpu_sample_peak_percent']:.1f}% | "
            f"{row['rss_peak_mb']:.1f} MiB | {row['security_cpu_average_percent']:.1f}% |"
        )

    lines.extend([
        "",
        "## 方法与公平性",
        "",
        "- 两个实现、direct baseline 使用同一个本地 fake backend 和完全相同的请求 payload。",
        "- 每个数据点由多轮 trial 组成；表中是各 trial 指标的中位数。overhead 在每轮内先做 Proxy − Direct，再取中位数。",
        "- ghc-api 使用当前项目 CLI（Flask threaded development server）。copilot-api-js 使用 production bundle `node dist/main.mjs`。",
        "- 两者 request-history retention 均设为 1000；ghc-api 是内存 cache，copilot-api-js 仍使用其正常 SQLite history pipeline。",
        "- 两者都关闭外部网络、rate limiting、SSE keepalive delay；fake backend 不包含真实数据。",
        "- 为使 copilot-api-js 的启动 token/VSCode 请求可指向 fake backend，对其 clone 应用了两个环境变量 URL override。该补丁只影响启动期常量，不在被测请求热路径。",
        "",
        "## 限制",
        "",
        f"- 结果代表本机 `{result['environment']['platform']}` 环境和各自当前服务器栈，不是 Python 与 JavaScript 语言本身的微基准。",
        "- fake backend 本身是 Flask threaded server；并发升高后会成为共享瓶颈，因此主要看配对 overhead，而不是只看总延迟。",
        "- 两个项目的功能管线不同：copilot-api-js 默认包含 SQLite history 和更复杂的 codec/observability；ghc-api 默认包含内存 request cache。",
    ])
    return "\n".join(lines) + "\n"


def run(
    js_repo: Path,
    output_dir: Path,
    requests_count: int = 60,
    warmup: int = 5,
    trials: int = 3,
    concurrencies: List[int] | None = None,
    wait_for_idle: bool = False,
    idle_timeout: int = 1800,
    js_runtime: str = "node",
) -> Path:
    concurrencies = concurrencies or [1, 8]
    output_dir.mkdir(parents=True, exist_ok=True)
    fake_port, python_port, js_port = _free_port(), _free_port(), _free_port()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    processes = []
    raw_runs = []
    idle_check = _wait_for_idle(10.0, 35.0, 10, idle_timeout) if wait_for_idle else {"skipped": True}
    post_start_idle_check = {"skipped": True}

    fake = python_proxy = js_proxy = None
    python_config_temp = tempfile.TemporaryDirectory(prefix="ghc-api-compare-")
    js_data_dir = output_dir / "copilot-api-js-data"
    shutil.rmtree(js_data_dir, ignore_errors=True)
    js_data_dir.mkdir(parents=True)

    try:
        fake = _start_process(
            [sys.executable, "-m", "benchmarks.e2e.fake_backend.app", "--port", str(fake_port)],
            env,
            output_dir / "fake-backend.log",
        )
        processes.append(fake)
        _wait_200(f"http://127.0.0.1:{fake_port}/health")

        python_config = Path(python_config_temp.name)
        _write_config(python_config / "config.yaml", fake_port, python_port, {"name": "comparison"})
        # Match the JS retention configured below.
        config = yaml.safe_load((python_config / "config.yaml").read_text(encoding="utf-8"))
        config["cache_max_entries"] = 1000
        (python_config / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        python_env = env.copy()
        python_env["GITHUB_TOKEN"] = "fake_github_token_for_benchmark_only"
        python_env["GHC_API_CONFIG_DIR"] = str(python_config)
        python_proxy = _start_process(
            [sys.executable, "-m", "ghc_api.cli", "--port", str(python_port), "--address", "127.0.0.1"],
            python_env,
            output_dir / "ghc-api.log",
        )
        processes.append(python_proxy)
        _wait_200(f"http://127.0.0.1:{python_port}/v1/models")

        js_config_dir = js_data_dir / "copilot-api"
        js_config_dir.mkdir(parents=True)
        (js_config_dir / "config.yaml").write_text(
            yaml.safe_dump({
                "ghc_api_base_url": f"http://127.0.0.1:{fake_port}",
                "history": {"success_limit": 1000, "failure_limit": 1000, "reaper_interval": 3600},
            }, sort_keys=False),
            encoding="utf-8",
        )
        js_env = env.copy()
        js_env["XDG_DATA_HOME"] = str(js_data_dir)
        js_env["NODE_ENV"] = "production"
        js_env["COPILOT_API_BENCH_GITHUB_BASE_URL"] = f"http://127.0.0.1:{fake_port}"
        js_env["COPILOT_API_BENCH_VSCODE_RELEASE_URL"] = f"http://127.0.0.1:{fake_port}/repos/microsoft/vscode/releases/latest"
        js_proxy = _start_process(
            [
                js_runtime, str(js_repo / "dist" / "main.mjs"), "start", "--port", str(js_port), "--host", "127.0.0.1",
                "--ghc-api-base-url", f"http://127.0.0.1:{fake_port}",
                "--github-token", "fake_github_token_for_benchmark_only",
                "--no-rate-limit", "--no-http-proxy-from-env",
            ],
            js_env,
            output_dir / "copilot-api-js.log",
        )
        processes.append(js_proxy)
        _wait_200(f"http://127.0.0.1:{js_port}/v1/models")
        if wait_for_idle:
            post_start_idle_check = _wait_for_idle(10.0, 35.0, 10, idle_timeout)

        implementations = [
            ("ghc-api", python_proxy, python_port),
            ("copilot-api-js", js_proxy, js_port),
        ]
        for scenario in _scenario_payloads():
            for concurrency in concurrencies:
                for trial in range(trials):
                    direct = _run_load(
                        f"http://127.0.0.1:{fake_port}{scenario['path']}",
                        scenario["payload"], requests_count, concurrency, warmup,
                    )
                    ordered = implementations if trial % 2 == 0 else list(reversed(implementations))
                    for implementation, process, port in ordered:
                        proxy, metrics = _run_measured_load(
                            process,
                            f"http://127.0.0.1:{port}{scenario['path']}",
                            scenario["payload"], requests_count, concurrency, warmup,
                        )
                        raw_runs.append({
                            "implementation": implementation,
                            "scenario": scenario["name"],
                            "concurrency": concurrency,
                            "trial": trial + 1,
                            "direct": direct,
                            "proxy": proxy,
                            "process": metrics,
                        })
    finally:
        for process in reversed(processes):
            _stop_process(process)
        python_config_temp.cleanup()

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "platform": sys.platform,
            "js_runtime": js_runtime,
            "js_runtime_version": subprocess.check_output([js_runtime, "--version"], text=True).strip(),
        },
        "parameters": {"requests": requests_count, "warmup": warmup, "trials": trials, "concurrencies": concurrencies},
        "idle_check": idle_check,
        "post_start_idle_check": post_start_idle_check,
        "copilot_api_js": {
            "repository": "https://github.com/puxu-msft/copilot-api-js",
            "commit": _git(js_repo, "rev-parse", "HEAD"),
            "commit_date": _git(js_repo, "log", "-1", "--format=%cI"),
            "version": json.loads((js_repo / "package.json").read_text(encoding="utf-8"))["version"],
            "benchmark_patch": ["GITHUB_API_BASE_URL env override", "VSCODE_RELEASE_URL env override"],
        },
        "raw_runs": raw_runs,
        "summary": _aggregate(raw_runs),
    }
    (output_dir / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / "comparison.md").write_text(_markdown(result), encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--js-repo", type=Path, default=ROOT / "build" / "copilot-api-js")
    parser.add_argument("--output", type=Path, default=RESULTS / "copilot-api-js-comparison")
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--wait-for-idle", action="store_true")
    parser.add_argument("--idle-timeout", type=int, default=1800)
    parser.add_argument("--js-runtime", default="node")
    args = parser.parse_args()
    print(run(
        args.js_repo.resolve(), args.output, args.requests, args.warmup, args.trials,
        wait_for_idle=args.wait_for_idle, idle_timeout=args.idle_timeout,
        js_runtime=args.js_runtime,
    ))


if __name__ == "__main__":
    main()
