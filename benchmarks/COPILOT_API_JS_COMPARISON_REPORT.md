# ghc-api 与 copilot-api-js E2E 性能对比报告

日期：2026-07-23

## 1. 对比对象

### ghc-api

- 当前工作区版本：`1.0.21`
- 运行方式：真实 CLI，Flask threaded server
- request history：内存 `RequestCache`

### copilot-api-js

- 仓库：`https://github.com/puxu-msft/copilot-api-js`
- commit：`3867514aabb5d6359fe71a5ab85e7447617845b7`
- commit 日期：`2026-07-14T02:43:43Z`
- package version：`0.8.4-beta.16`
- 运行方式：production bundle，Node.js `v24.18.0`
- request history：SQLite history pipeline

copilot-api-js 已有 `--ghc-api-base-url`，可直接将 LLM 请求指向 fake backend。但它的 GitHub token endpoint 和 VSCode release endpoint 是硬编码 URL。为了保证完全离线，对 clone 应用了两个只影响启动阶段的环境变量 URL override，并重新构建 backend bundle：

```text
GITHUB_API_BASE_URL
VSCODE_RELEASE_URL
```

请求转发热路径没有修改。

## 2. Benchmark 场景

两个实现使用同一个 fake backend、同一个 payload、相同 response bytes：

- `fake-opus`：Anthropic `/v1/messages` full streaming
  - thinking
  - text
  - tool use
  - usage/cache usage
- `fake-gpt`：OpenAI `/v1/responses` full streaming
  - reasoning summary
  - function call
  - web search
  - annotation
  - usage

参数：

```text
response text: 1024 bytes
text chunks: 16
argument chunks: 4
requests per trial: 60
warmup: 5
trials per run: 3
independent runs: 2
combined trials per point: 6
concurrency: 1, 8
```

两个实现的 history retention 均设为 1000。ghc-api 仍为内存 history，copilot-api-js 仍为正常 SQLite history，因此结果表示两者当前功能栈，而不是只比较 HTTP framework。

## 3. Windows Security 干扰处理

第一次原生 Windows 测试期间观察到：

```text
MsMpEng.exe: 约 70%～110% CPU
MsSense.exe: 约 30%～55% CPU
系统总 CPU: 约 50%～70%
```

因此第一次结果不作为最终结论。

之后执行了以下处理：

1. 等待 Windows Security 总 CPU 降至 10% 以下并连续稳定。
2. 将两个项目、fake backend、SQLite DB 和 benchmark result 全部移动到 WSL2 ext4 文件系统。
3. 在 WSL2 内重新运行两轮完整对比，共 6 个 trial。
4. WSL2 benchmark 启动前内部 system CPU 分别约为 2.5% 和 2.9%。
5. 使用独立 Windows host monitor 监视第三轮测试。

Host monitor 发现即使 benchmark 文件全部位于 WSL2 ext4，企业 Windows Security/EDR 仍会在 benchmark 执行期间重新活跃：

```text
Windows Security CPU average: 38.96%
Windows Security CPU p95:     102.5%
Windows Security CPU peak:    148.6%
```

因此：

- wall-clock latency/RPS 仍可能受 host 调度影响，可信度为中等。
- 两轮 WSL2 结果非常接近，说明相对排序具有重复性。
- CPU/request 使用每个被测进程自己的 user+system CPU time，不包含 MsMpEng/MsSense CPU，可信度高于 wall-clock latency。
- 若需要发布级绝对 latency，仍建议在没有企业 EDR 的独立 Linux 主机上运行。

## 4. 合并结果

下表为两轮 WSL2 run、每个点合计 6 个 trial 的中位数。

| 实现 | 场景 | 并发 | Proxy p50 | 配对 p50 overhead | RPS | CPU/request | 平均占用核 | CPU peak | RSS peak |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ghc-api | Opus Messages | 1 | **5.23 ms** | **+2.71 ms** | **179.2** | **2.75 ms** | 0.50 | 47.3% | **39.8 MiB** |
| copilot-api-js | Opus Messages | 1 | 19.93 ms | +17.46 ms | 40.4 | 27.33 ms | 1.09 | 132.2% | 250.9 MiB |
| ghc-api | Opus Messages | 8 | **123.58 ms** | **+25.36 ms** | **64.1** | **24.00 ms** | 1.55 | 160.6% | **43.7 MiB** |
| copilot-api-js | Opus Messages | 8 | 205.13 ms | +111.88 ms | 37.2 | 30.00 ms | 1.16 | 154.6% | 322.5 MiB |
| ghc-api | GPT Responses | 1 | **6.17 ms** | **+3.21 ms** | **153.4** | **3.25 ms** | 0.50 | 49.8% | **48.5 MiB** |
| copilot-api-js | GPT Responses | 1 | 31.31 ms | +28.21 ms | 26.9 | 39.58 ms | 1.07 | 146.1% | 327.6 MiB |
| ghc-api | GPT Responses | 8 | **293.07 ms** | **+145.26 ms** | **27.4** | 56.25 ms | 1.54 | 168.2% | **53.4 MiB** |
| copilot-api-js | GPT Responses | 8 | 325.87 ms | +186.05 ms | 24.2 | **43.67 ms** | 1.08 | 144.6% | 336.4 MiB |

## 5. Latency 与吞吐对比

### Concurrency 1

低并发时 ghc-api 明显更快：

| 场景 | p50 latency | RPS |
|---|---:|---:|
| Opus | ghc-api 快约 **3.81×** | ghc-api 高约 **4.43×** |
| GPT | ghc-api 快约 **5.08×** | ghc-api 高约 **5.70×** |

主要差异不是 fake backend，因为 direct baseline 在同一个 trial 内配对扣除。copilot-api-js 在单请求下仍执行 codec、observability、SQLite history 等完整管线。

### Concurrency 8

- Opus：ghc-api p50 快约 **1.66×**，RPS 高约 **1.72×**。
- GPT：两者接近；ghc-api p50 快约 **1.11×**，RPS 高约 **1.13×**。

GPT concurrency 8 时 fake backend 和 load client 已出现明显排队，因此这一点不适合用于推导纯 proxy 固定成本；它更接近整体系统饱和表现。

## 6. CPU 对比

### CPU/request

CPU/request 是本次最可靠的 CPU 指标，因为它只统计被测进程自身 CPU time，不包含 Windows Security 进程。

| 场景 | 并发 | ghc-api | copilot-api-js | 结论 |
|---|---:|---:|---:|---|
| Opus | 1 | **2.75 ms** | 27.33 ms | ghc-api 每请求约省 **89.9%** CPU |
| Opus | 8 | **24.00 ms** | 30.00 ms | ghc-api 每请求约省 **20.0%** CPU |
| GPT | 1 | **3.25 ms** | 39.58 ms | ghc-api 每请求约省 **91.8%** CPU |
| GPT | 8 | 56.25 ms | **43.67 ms** | copilot-api-js 每请求约省 **22.4%** CPU |

GPT concurrency 8 是唯一一个 copilot-api-js CPU/request 更低的场景。这说明：

- ghc-api 的低并发固定成本很低。
- event-rich Responses 流在高并发下，Python JSON/SSE/cache 路径出现明显 CPU 扩张。
- copilot-api-js 的单请求固定管线较重，但 Node pipeline 在高并发 GPT Responses 场景下 CPU scaling 更好。

### 平均占用核数

平均占用核数按：

```text
process CPU seconds / wall-clock measurement seconds
```

计算。

- concurrency 1：ghc-api 约使用 0.50 core；copilot-api-js 约使用 1.07～1.09 core。
- concurrency 8：ghc-api 约使用 1.54～1.55 core；copilot-api-js 约使用 1.08～1.16 core。

需要注意，平均占用核数不能脱离吞吐解释。例如 Opus concurrency 8 时 ghc-api 使用更多即时 CPU core，但它同时完成了约 1.72 倍的 RPS；按每请求 CPU 计算仍更省。

## 7. 内存对比

copilot-api-js 的 peak RSS 在本次测试中约为 ghc-api 的 6～7 倍：

| 场景 | ghc-api | copilot-api-js |
|---|---:|---:|
| Opus C1 | 39.8 MiB | 250.9 MiB |
| Opus C8 | 43.7 MiB | 322.5 MiB |
| GPT C1 | 48.5 MiB | 327.6 MiB |
| GPT C8 | 53.4 MiB | 336.4 MiB |

这包含 Node runtime、完整 codec/observability pipeline、UI/backend bundle 运行时状态和 SQLite history 等，不应解释为单纯语言 runtime 差异。

## 8. 结论

### ghc-api 优势

- concurrency 1 的固定 overhead 显著更低。
- Opus direct Anthropic path 在 concurrency 1 和 8 都明显领先。
- 内存占用显著更低。
- 除高并发 GPT Responses 外，CPU/request 更低。

### copilot-api-js 优势

- 高并发 GPT Responses 的 CPU/request 更低，表现出更好的 event-rich stream CPU scaling。
- 功能管线更丰富，默认包含 SQLite history、codec、observability 和更多兼容处理；这些能力本身会产生可测成本。

### 当前最值得优化的 ghc-api 热点

根据 CPU 曲线，优先级最高的是：

1. `/v1/responses` 高并发 streaming 的 JSON decode/encode。
2. raw SSE event 保存和 response-size 统计。
3. request cache 全局 lock 与每请求 statistics 更新。
4. `requests` 每请求连接和 Flask threaded server 调度。
5. 将 production benchmark 与当前 Flask development server 分离，增加正式 WSGI server 结果。

## 9. 复现和原始数据

Benchmark runner：

```text
benchmarks/e2e/compare_copilot_api_js.py
```

最终合并数据：

```text
benchmarks/results/copilot-api-js-comparison-combined/comparison.json
```

两轮 WSL2 原始结果：

```text
benchmarks/results/copilot-api-js-comparison-clean-wsl-v2/
benchmarks/results/copilot-api-js-comparison-clean-wsl-v3/
```

Windows host monitor：

```text
benchmarks/results/copilot-api-js-comparison-clean-wsl-v3-host-monitor.json
```

`benchmarks/results/` 已被 Git 忽略，不会提交日志、SQLite DB 或原始运行数据。

## 10. 原生 Windows 复测

2026-07-23 又执行了一轮原生 Windows 对比。启动 benchmark 前已经等待到：

```text
System CPU:          33.7%
Windows Security:     2.8%
连续稳定时间:          10 秒
```

但 benchmark 开始后 Security/EDR 再次被触发：

```text
System CPU average:           51.16%
System CPU peak:              96.5%
Windows Security CPU average: 82.66%
Windows Security CPU p95:     105.2%
Windows Security CPU peak:    243.0%
```

主要来源：

```text
MsSense.exe active-sample average: 75.2%
MsSense.exe peak:                 145.0%
MsMpEng.exe active-sample average: 11.5%
MsMpEng.exe peak:                145.3%
```

因此原生 Windows wall-clock latency/RPS 仍不能视为无干扰结果。进程自身 CPU/request 如下：

| 场景 | ghc-api | copilot-api-js | ghc-api CPU/request 优势 |
|---|---:|---:|---:|
| Opus C1 | 10.94 ms | 37.50 ms | 约 3.43× |
| Opus C8 | 15.10 ms | 54.95 ms | 约 3.64× |
| GPT C1 | 19.79 ms | 52.34 ms | 约 2.64× |
| GPT C8 | 32.81 ms | 61.72 ms | 约 1.88× |

原生 Windows peak RSS：

| 场景 | ghc-api | copilot-api-js |
|---|---:|---:|
| Opus C1 | 43.3 MiB | 138.2 MiB |
| Opus C8 | 46.7 MiB | 162.6 MiB |
| GPT C1 | 57.8 MiB | 196.7 MiB |
| GPT C8 | 58.9 MiB | 224.3 MiB |

虽然 wall-clock 受 EDR 干扰，原生 Windows 的进程 CPU 与内存方向和 WSL2 结果一致：ghc-api 在本轮四个场景中都使用更少 CPU/request，并维持明显更低的 RSS。

Windows 原始结果：

```text
benchmarks/results/copilot-api-js-comparison-windows-final/
benchmarks/results/copilot-api-js-comparison-windows-final-host-monitor.json
```
