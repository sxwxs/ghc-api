"""Process-level E2E benchmark runner for the synthetic Opus and GPT backends."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import socket
import statistics
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
except ImportError:  # resource metrics are optional
    psutil = None


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "results"
_thread_local = threading.local()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait(url: str, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=0.5)
            if response.status_code < 500:
                return
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def _session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session


def _request_once(url: str, payload: Dict) -> Dict[str, float]:
    start = time.perf_counter_ns()
    response = _session().post(url, json=payload, stream=True, timeout=30)
    first_byte = None
    total_bytes = 0
    try:
        # A 1-byte iterator makes the Python load generator itself the
        # bottleneck on event-rich streams. 4 KiB still observes logical SSE
        # termination while keeping client-side parsing overhead bounded.
        for line in response.iter_lines(chunk_size=4096):
            if first_byte is None:
                first_byte = time.perf_counter_ns()
            total_bytes += len(line) + 1
            if line == b"data: [DONE]":
                break
            if line.startswith(b"data: "):
                try:
                    event = json.loads(line[6:])
                except (json.JSONDecodeError, UnicodeDecodeError):
                    event = None
                if isinstance(event, dict) and event.get("type") in {"message_stop", "response.completed"}:
                    break
    finally:
        response.close()
    end = time.perf_counter_ns()
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code} from {url}")
    first_byte = first_byte or end
    return {
        "latency_ms": (end - start) / 1_000_000,
        "ttfb_ms": (first_byte - start) / 1_000_000,
        "bytes": total_bytes,
    }


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _summarize(samples: List[Dict[str, float]], elapsed: float) -> Dict:
    latencies = [sample["latency_ms"] for sample in samples]
    ttfb = [sample["ttfb_ms"] for sample in samples]
    return {
        "requests": len(samples),
        "elapsed_s": elapsed,
        "rps": len(samples) / elapsed if elapsed else 0,
        "bytes": int(sum(sample["bytes"] for sample in samples)),
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
            "max": max(latencies),
        },
        "ttfb_ms": {
            "mean": statistics.fmean(ttfb),
            "p50": _percentile(ttfb, 0.50),
            "p95": _percentile(ttfb, 0.95),
            "p99": _percentile(ttfb, 0.99),
            "max": max(ttfb),
        },
    }


def _run_load(url: str, payload: Dict, requests_count: int, concurrency: int, warmup: int) -> Dict:
    for _ in range(warmup):
        _request_once(url, payload)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        samples = list(pool.map(lambda _: _request_once(url, payload), range(requests_count)))
    elapsed = time.perf_counter() - started
    return _summarize(samples, elapsed)


def _write_config(path: Path, fake_port: int, proxy_port: int, variant: Dict) -> None:
    config = {
        "address": "127.0.0.1",
        "port": proxy_port,
        "debug": False,
        "account_type": "individual",
        "github_api_base_url": f"http://127.0.0.1:{fake_port}",
        "copilot_api_base_url": f"http://127.0.0.1:{fake_port}",
        "disable_onedrive_access": True,
        "save_request_to_file": variant.get("save_request_to_file", False),
        "enable_tool_call_recovery": variant.get("enable_tool_call_recovery", False),
        "enable_auth": False,
        "sse_keepalive_interval": 0,
        "max_connection_retries": 0,
        "cache_max_entries": 1000,
        "cache_max_request_size": 1048576,
        "model_mappings": {"exact": {}, "prefix": {}},
        "chat_completions_model_support": {"exact": [], "prefix": []},
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _start_process(command: List[str], env: Dict[str, str], log_path: Path) -> subprocess.Popen:
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    process._benchmark_log = log  # type: ignore[attr-defined]
    return process


def _stop_process(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    log = getattr(process, "_benchmark_log", None)
    if log:
        log.close()


def _process_metrics_start(pid: int):
    if psutil is None:
        return None
    process = psutil.Process(pid)
    cpu = process.cpu_times()
    return process, cpu.user + cpu.system, process.memory_info().rss


def _process_metrics_end(start, requests_count: int) -> Dict:
    if start is None:
        return {}
    process, cpu_start, rss_start = start
    cpu = process.cpu_times()
    cpu_seconds = cpu.user + cpu.system - cpu_start
    rss = process.memory_info().rss
    return {
        "cpu_seconds": cpu_seconds,
        "cpu_ms_per_request": cpu_seconds * 1000 / requests_count,
        "rss_before_bytes": rss_start,
        "rss_after_bytes": rss,
        "rss_delta_bytes": rss - rss_start,
        "threads": process.num_threads(),
    }


def _scenario_payloads() -> List[Dict]:
    common = {"profile": "full", "text_bytes": 1024, "text_chunks": 16, "argument_chunks": 4}
    return [
        {
            "name": "opus-messages-stream-full",
            "path": "/v1/messages",
            "payload": {
                "model": "fake-opus", "stream": True, "max_tokens": 1024,
                "system": "Operate only on the isolated load-test fixture.",
                "messages": [{"role": "user", "content": "Inspect the fixture and return a deterministic protocol response."}],
                "tools": [{"name": "inspect_fixture_asset", "description": "Inspect an isolated fixture", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}],
                "metadata": {"ghc_benchmark": common},
            },
        },
        {
            "name": "gpt-responses-stream-full",
            "path": "/v1/responses",
            "payload": {
                "model": "fake-gpt", "stream": True, "max_output_tokens": 1024,
                "input": [{"role": "user", "content": "Inspect the fixture and return a deterministic protocol response."}],
                "tools": [{"type": "function", "name": "inspect_fixture_asset", "description": "Inspect an isolated fixture", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}],
                "metadata": {"ghc_benchmark": common},
            },
        },
    ]


def _markdown(result: Dict) -> str:
    lines = [
        "# ghc-api E2E Benchmark Report",
        "",
        f"Generated: {result['generated_at']}",
        "",
        "Synthetic backends: `fake-opus` and `fake-gpt`; no GitHub endpoint was used.",
        "",
        "| Variant | Scenario | C | Direct p50 | Proxy p50 | Overhead | Direct p95 | Proxy p95 | Proxy RPS | CPU ms/request |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in result["runs"]:
        direct = run["direct"]
        proxy = run["proxy"]
        overhead = proxy["latency_ms"]["p50"] - direct["latency_ms"]["p50"]
        cpu = run.get("process", {}).get("cpu_ms_per_request", 0)
        lines.append(
            f"| {run['variant']} | {run['scenario']} | {run['concurrency']} | "
            f"{direct['latency_ms']['p50']:.2f} ms | {proxy['latency_ms']['p50']:.2f} ms | "
            f"{overhead:+.2f} ms | {direct['latency_ms']['p95']:.2f} ms | "
            f"{proxy['latency_ms']['p95']:.2f} ms | {proxy['rps']:.1f} | {cpu:.3f} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- Values are local-machine measurements and are intended for relative comparisons.",
        "- Direct and proxied requests use the same deterministic response profile.",
        "- The proxy is started through its real CLI and performs fake token/model initialization over HTTP.",
        "- Flask's current threaded development server is part of the measured proxy stack.",
    ])
    return "\n".join(lines) + "\n"


def run(suite: str, output_dir: Path | None = None) -> Path:
    requests_count = 30 if suite == "smoke" else 200
    warmup = 3 if suite == "smoke" else 15
    concurrencies = [1, 8] if suite == "smoke" else [1, 8, 32]
    variants = [
        {"name": "baseline"},
        {"name": "request-file-logging", "save_request_to_file": True},
        {"name": "tool-call-recovery", "enable_tool_call_recovery": True},
    ]
    generated = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_dir or RESULTS / generated
    output_dir.mkdir(parents=True, exist_ok=True)
    fake_port = _free_port()
    fake = None
    runs = []
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        fake = _start_process(
            [sys.executable, "-m", "benchmarks.e2e.fake_backend.app", "--port", str(fake_port)],
            env,
            output_dir / "fake-backend.log",
        )
        _wait(f"http://127.0.0.1:{fake_port}/health")

        for variant in variants:
            proxy_port = _free_port()
            with tempfile.TemporaryDirectory(prefix="ghc-api-bench-") as temp:
                config_dir = Path(temp)
                _write_config(config_dir / "config.yaml", fake_port, proxy_port, variant)
                proxy_env = env.copy()
                proxy_env["GITHUB_TOKEN"] = "fake_github_token_for_benchmark_only"
                proxy_env["GHC_API_CONFIG_DIR"] = str(config_dir)
                proxy = _start_process(
                    [sys.executable, "-m", "ghc_api.cli", "--port", str(proxy_port), "--address", "127.0.0.1"],
                    proxy_env,
                    output_dir / f"ghc-api-{variant['name']}.log",
                )
                try:
                    _wait(f"http://127.0.0.1:{proxy_port}/v1/models")
                    for scenario in _scenario_payloads():
                        for concurrency in concurrencies:
                            direct = _run_load(
                                f"http://127.0.0.1:{fake_port}{scenario['path']}",
                                scenario["payload"], requests_count, concurrency, warmup,
                            )
                            metrics_start = _process_metrics_start(proxy.pid)
                            proxied = _run_load(
                                f"http://127.0.0.1:{proxy_port}{scenario['path']}",
                                scenario["payload"], requests_count, concurrency, warmup,
                            )
                            process_metrics = _process_metrics_end(metrics_start, requests_count)
                            runs.append({
                                "variant": variant["name"], "scenario": scenario["name"],
                                "concurrency": concurrency, "direct": direct, "proxy": proxied,
                                "process": process_metrics,
                            })
                finally:
                    _stop_process(proxy)
    finally:
        _stop_process(fake)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite": suite,
        "python": sys.version,
        "platform": sys.platform,
        "requests_per_measurement": requests_count,
        "warmup_requests": warmup,
        "runs": runs,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(_markdown(result), encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = run(args.suite, args.output)
    print(output)


if __name__ == "__main__":
    main()
