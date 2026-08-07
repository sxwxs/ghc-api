# ghc-api Fake LLM E2E 测试报告

日期：2026-07-23

## 1. 测试范围

本次实现并验证了一个完全离线的真实 HTTP E2E 测试链路：

```text
benchmark client -> fake backend
benchmark client -> ghc-api -> fake backend
```

Fake backend 提供两个只包含合成数据的模型：

- `fake-opus`
  - Anthropic `/v1/messages`，stream/non-stream
  - OpenAI `/v1/responses`，stream/non-stream
  - thinking、text、tool use、usage、cache usage、Copilot usage 等字段
- `fake-gpt`
  - OpenAI `/v1/responses`，stream/non-stream
  - reasoning summary、text、function call、web search、annotation、usage 等事件

基准进程通过 ghc-api 的真实 CLI 启动。启动时的 token refresh、model listing 和后续 LLM 请求全部指向 loopback fake backend，不访问 GitHub。

## 2. 隐私和脱敏检查

采取的保护措施：

- `opus.jl`、`gpt.jl` 已加入 `.gitignore`。
- Fake backend 运行时不读取 request dump。
- Response builder 为手写协议结构，不复制 dump 中的 response value。
- Fake backend 不 echo 用户请求、tool arguments 或认证 header。
- ID 使用 `*_fake_*` 前缀。
- 路径仅使用 `/isolated-fixture/...`。
- URL 仅使用保留测试域名 `perf-fixture.invalid`。
- encrypted content 使用明确的 placeholder，不生成或复用真实密文。

执行了本地 dump overlap 检查：

```text
python -m benchmarks.e2e.verify_no_sample_leak opus.jl gpt.jl
PASS: no long synthetic response value was found in the supplied dumps
```

该检查只在内存中比较长 leaf value，不打印或保存 dump 内容。

## 3. 功能测试结果

执行命令：

```text
python -m pytest -q
```

结果：

```text
155 passed in 2.88s
```

覆盖内容包括：

- Fake models metadata
- Opus thinking/text/tool-use SSE 事件顺序
- GPT reasoning/function-call/web-search/annotation 事件族
- Opus/GPT non-stream response object
- request 内容不被 response echo
- upstream GitHub/Copilot URL override
- `GHC_API_CONFIG_DIR` 隔离
- 原有 ghc-api 单元测试和 SSE 测试

另外执行了：

```text
python -m py_compile ...
git diff --check
```

均通过。

## 4. E2E smoke benchmark

环境：

- Windows `win32`
- Python 3.11.9
- Flask threaded development server
- 每个 measurement 30 个请求
- 每轮 3 个 warmup 请求
- concurrency：1、8
- response profile：full
- text：1024 bytes / 16 chunks
- function arguments：4 chunks

原始结果位于本地忽略目录：

```text
benchmarks/results/implementation-smoke-final/
```

### 4.1 Baseline 结果

| Scenario | Concurrency | Direct p50 | Proxy p50 | p50 Overhead | Direct p95 | Proxy p95 | Proxy RPS | ghc-api CPU/request |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Opus `/v1/messages` stream full | 1 | 31.41 ms | 44.16 ms | +12.75 ms | 35.58 ms | 48.14 ms | 25.4 | 7.812 ms |
| Opus `/v1/messages` stream full | 8 | 80.73 ms | 92.51 ms | +11.78 ms | 109.13 ms | 109.53 ms | 80.8 | 13.021 ms |
| GPT `/v1/responses` stream full | 1 | 62.86 ms | 76.86 ms | +14.00 ms | 64.14 ms | 80.42 ms | 13.9 | 9.375 ms |
| GPT `/v1/responses` stream full | 8 | 304.20 ms | 317.78 ms | +13.58 ms | 411.70 ms | 418.50 ms | 23.5 | 23.438 ms |

在本次 smoke run 中，两个 full streaming path 的 p50 E2E overhead 约为：

```text
11.78 ms ～ 14.00 ms
```

这里包含：

- client 到 ghc-api 的额外 HTTP hop
- Flask route 和 threaded server
- `requests.post` 上游调用
- JSON/SSE parse 与 serialization
- request cache/statistics lock
- raw event 保存和 usage 提取
- Opus direct stream handler 或 Responses stream handler

### 4.2 开关 smoke run

本次还启动并执行了：

- `save_request_to_file: true`
- `enable_tool_call_recovery: true`

所有请求均成功完成，三个 ghc-api variant 均完成 fake token refresh 和 2 个 fake model 加载，日志中没有 traceback 或 startup error。

但是当前 smoke 配置每组只有 30 个请求，Windows 调度和 Flask development server 带来的波动明显。例如某些 file-logging 组反而比 baseline 更快，这不具有因果意义。因此，本报告不根据本轮数据给出 file logging 或 recovery 的精确 overhead 结论。

要评估开关增量，应运行：

```text
python -m benchmarks.e2e.runner --suite full
```

并在空闲、固定 CPU/电源策略的机器上重复多轮，使用 trial median 比较同一 scenario 的 proxy-to-proxy 差值。

## 5. 发现和限制

1. 当前 CLI 使用 Flask development server，因此本结果表示“当前 ghc-api CLI 整体栈”的 overhead，而不是纯业务函数耗时。
2. GPT full profile 包含更多 SSE events，direct backend 本身已经有明显开销；必须使用 direct/proxy 对照，不能只看 proxy latency。
3. smoke run 足以验证测试框架和获得 baseline 量级，但不足以设置 CI regression threshold。
4. runner 在看到 Anthropic `message_stop` 或 Responses `[DONE]` 后结束客户端计时，符合正常 SSE 客户端的协议终止行为。
5. 当前资源指标记录 ghc-api CPU time 和 RSS before/after；尚未记录 fake backend CPU、峰值 RSS 和 context switch。

## 6. 结论

Fake Opus/GPT backend、离线上游覆盖、真实进程 E2E runner、开关 variant、资源采集、隐私检查和 Markdown/JSON 报告均已实现并运行通过。

当前 smoke baseline 表明，在这台 Windows 机器、当前 Flask threaded server 和 full SSE response profile 下，ghc-api 增加的 p50 E2E latency 大约为 **12～14 ms**。该数字应作为本机当前版本的初始参考点，而不是跨机器或生产部署的固定结论。
