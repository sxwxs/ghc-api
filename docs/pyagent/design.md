# pyagent 设计

pyagent 是一个轻量的 Python 编程 agent。架构参考 pi（`earendil-works/pi`）：分层的模型接入层、通用 agent 循环、编程场景层，以及一套供所有界面共用的协议。

代码在 `ghc_api/pyagent/`，最初来自独立仓库 pyagent（提交 3dd6e04，现仅作存档），之后只在 ghc-api 里维护。它在 ghc-api 内保持独立：

- `ghc_api/pyagent/` 不 import ghc_api 的其他部分（`tests/pyagent/test_independence.py` 检查）。
- ghc-api 只通过第 9 节的协议驱动它（`ghc_api/agent_bridge.py`），不碰 agent、会话的内部对象。
- 有自己的 lint 与严格类型检查：`ghc_api/pyagent/ruff.toml`、根目录 `pyrightconfig.json`。
- 下文的 `pyagent` 命令即 `python -m ghc_api.pyagent`。

## 1. 目标

- 单进程、asyncio。核心依赖只有 `httpx` 和 `msgspec`，核心代码目标 5k 行以内。
- 所有界面（CLI、TUI、Web、IDE）都通过同一套协议驱动 agent。内置 REPL 也是协议客户端（进程内传输），协议缺什么会先被自己发现。
- v1 不做：插件市场、OAuth、MCP、HTML 导出、富 TUI。保留扩展点。

## 2. 分层

```
ghc_api/pyagent/
  ai/        模型调用：消息类型、流式事件、模型与厂商预设、厂商适配器
  agent/     通用 agent 循环：工具契约、事件、循环、Agent 类。不知道文件、会话、协议
  coding/    编程场景：内置工具、系统提示词、配置；之后加会话存储、压缩、重试、权限
  protocol/  (M2) 协议类型（唯一事实源）、错误码、JSON Schema 导出
  server/    (M2) 分发、事件映射、传输（stdio，之后 WebSocket）
  client/    (M2) Python 客户端
  cli/       入口
```

依赖单向：`ai ← agent ← coding ← server ← cli`。

## 3. 线上约定（所有层共用）

消息、事件、工具结果都是 msgspec Struct，进程内和线上是同一个类型，不需要转换层。

- JSON 字段名用 camelCase（Python 属性 snake_case，由 msgspec 自动转换）。
- 类型标签值用 snake_case：`"type": "tool_call"`、`"role": "tool_result"`、`"type": "block_delta"`。
- 可选字段缺省即零值（`null`、`false`、`0`、空数组）。有业务含义的字段（如 `stopReason`）一律必填。
- 客户端必须忽略不认识的字段和事件类型。

## 4. 数据模型（`ai/types.py`）

内容块：`text{text, signature?}`、`image{data, mimeType}`、`thinking{thinking, signature?, redacted?}`、`tool_call{id, name, arguments}`。`signature` 是回放给同一模型所需的厂商数据（Anthropic 签名、Responses 的加密推理项和消息 phase），换模型时丢弃。

消息（都带 `id` 和 `timestamp`；`id` 同时是会话记录的 id）：

| role | 字段 |
|---|---|
| `user` | `content[]` |
| `assistant` | `content[]`、`provider`、`model`、`stopReason`（`stop`/`length`/`tool_use`/`error`/`aborted`）、`usage`、`errorMessage?` |
| `tool_result` | `toolCallId`、`toolName`、`content[]`、`details?`、`isError?` |

`content` 给模型看；`details` 只给 UI（edit 的 diff、bash 的退出码），不发给模型。

## 5. 流式事件（`ai/events.py`）

厂商适配器边收边改同一个 `AssistantMessage`，并产出块事件：

| 事件 | 字段 |
|---|---|
| `message_start` | `message` |
| `block_start` | `messageId`、`index`、`kind`（`text`/`thinking`/`tool_call`）、`toolCallId?`、`toolName?` |
| `block_delta` | `messageId`、`index`、`delta` |
| `block_end` | `messageId`、`index` |
| `message_end` | `message`（权威完整版） |

只有三种块事件，UI 按 `index` 建块、按 `delta` 追加，`message_end` 到达时整体替换。事件不带累计快照，流总大小与输出长度线性相关。`block_end` 不重复块内容：完整内容只在 `message_end` 出现一次。

适配器可以抛异常；`stream_assistant()` 把异常记为 `stopReason=error` 的消息，把取消记为 `aborted`。循环只有一种结束路径。

## 6. Agent 循环（`agent/loop.py`）

```
append(prompts)
loop:
  turn_start
  reply = stream_assistant()            # message_start / block_* / message_end
  if reply 出错或中止: 结束
  if reply 有工具调用: 执行工具          # tool_start / tool_update / tool_end，然后按原顺序产出 tool_result 消息
  turn_end
  pending = 取 steering 队列
  if 有工具调用 or pending: continue
  pending = 取 follow-up 队列
  if 没有: 结束
```

- 中止：`Agent.abort()` 标记后取消运行任务。循环在最外层捕获取消：把正在流式的回复标为 `aborted`，给未完成的工具调用补 `Aborted` 结果，保证“每个工具调用都有结果”这个不变量。bash 在 `finally` 里杀掉整个进程组。
- 工具并发：一批调用全部 `parallel_safe`（read 等只读工具）时并发执行，否则逐个“准备 + 执行”。
- Hook：`before_tool_call` 可拦截调用（权限在 M3 接在这里）。
- 重试、压缩、持久化不在循环里，放在之后的 `coding/agent_session.py`。

Agent 事件 = 第 5 节的消息事件 + 以下事件：

| 事件 | 字段 |
|---|---|
| `agent_start` / `agent_end` | `reason`（`completed`/`aborted`/`error`）、`errorMessage?` |
| `turn_start` / `turn_end` | `turn` |
| `tool_start` | `toolCallId`、`name`、`kind`、`title` |
| `tool_update` | `toolCallId`、`partial` |
| `tool_end` | `toolCallId`、`result`、`durationMs?` |
| `queue_changed` | `steering[]`、`followUp[]` |

`tool_start` 不重复参数：参数已在 assistant 消息里；`title` 是工具生成的一行说明（如 `pytest -q`、`src/app.py`），UI 不需要认识每个工具。

## 7. 工具（`coding/tools`）

| 工具 | 要点 | kind | 并发 |
|---|---|---|---|
| read | offset/limit；默认 2000 行或 50KB 截断；图片返回 image 块 | read | 是 |
| write | 写整文件，自动建目录，按字节写入不改换行 | edit | 否 |
| edit | `edits[]` 精确替换，每段必须唯一匹配且不重叠；保留 BOM 和 CRLF；`details` 带 diff | edit | 否 |
| bash | 流式输出（100ms 节流）、超时、输出保留末尾 50KB，超出部分完整落临时文件；Windows 优先 Git Bash，否则 PowerShell | execute | 否 |
| grep/find/ls | (M3) 有 rg 用 rg | search | 是 |

参数用 msgspec Struct 定义，自动生成给模型的 JSON Schema，并在执行前校验。

## 8. 会话存储（`coding/session_store.py`）

沿用 pi 格式：每会话一个 JSONL 文件 `~/.pyagent/sessions/<工作目录>/<sessionId>.jsonl`，首行 header `{type: "session", version, id, cwd, timestamp, parentSession?}`，其余每行一个 entry `{type, id, parentId, timestamp, ...}`。时间戳统一为毫秒。

- 类型：`message`（entry id 即消息 id）、`model_change`、`thinking_change`、`session_info`；M3 加 `compaction`。
- 最后追加的 entry 是当前叶子；从叶子沿 `parentId` 走到根得到当前分支，分支上最后的设置生效。
- 新会话在第一条消息到来时才写文件，空会话不留痕迹；`--no-session` 只在内存中。
- 只追加写；读取时跳过无法解码的行（如崩溃时写了一半的最后一行）。
- 分叉：新文件，复制从根到指定 entry 的分支，header 记录 `parentSession`。

`AgentSession`（`coding/agent_session.py`）把 Agent 和存储接起来：每个 `message_end` 写一条 entry，设置变化写对应 entry，给事件编号并加上 `runId`。

压缩（M3）：上下文 token 超过「窗口 − 16k」时触发，保留最近约 20k token，之前的用模型写摘要。

## 9. 协议

完整参考见 [protocol.md](protocol.md)。

### 9.1 原则

1. JSON-RPC 2.0，双向：客户端请求、服务端通知（事件）、服务端反向请求（权限、对话框）。方法命名向 ACP（Agent Client Protocol）靠拢，之后可加 ACP 适配层。
2. 与传输无关：v1 stdio，每行一个 JSON，只按 `\n` 切分，stdout 只放协议；之后 WebSocket 承载同样的消息。输出时转义 U+2028/U+2029。
3. 所有方法和事件带 `sessionId`，一个进程可有多个会话。
4. 每会话事件带单调递增 `seq`。
5. 类型唯一事实源在 `protocol/types.py`，导出 JSON Schema，UI 生成 TS 类型。

### 9.2 客户端 → 服务端

| 方法 | 作用 |
|---|---|
| `initialize` | 版本协商与能力交换，必须第一条 |
| `session/new` `load` `list` `close` | 会话生命周期 |
| `session/state` | 快照：模型、思考级别、是否运行、队列、用量、上下文占用、最新 seq |
| `session/messages` | 当前分支全部消息 |
| `session/fork` | 从某条消息分叉出新会话 |
| `session/compact`、`session/set_name` | |
| `session/subscribe` | `{sinceSeq}` 断线补发（WebSocket 阶段） |
| `run/prompt` | `{content[], delivery: auto\|steer\|follow_up}` → `{runId, disposition: started\|queued}`；`auto` 空闲时启动，运行中作为 steer |
| `run/abort`、`run/clear_queue` | |
| `config/models`、`set_model`、`set_thinking`、`set_permission_mode` | |
| `commands/list` | 斜杠命令、技能（输入框补全） |

### 9.3 事件

通知 `session/event`，参数 `{sessionId, seq, runId?, event}`。`event` 为第 5、6 节的事件，另有会话级事件：

| 事件 | 字段 |
|---|---|
| `run_start` | 无（`runId` 在信封里） |
| `run_end` | `reason`（`completed`/`aborted`/`error`）、`errorMessage?` |
| `retry`（M3） | `attempt`、`delayMs`、`error` |
| `compaction_start` / `compaction_end`（M3） | `reason`、`result?` |
| `state_changed` | 完整 `SessionState`；设置变化和每次运行结束后发出 |

实现时的取舍：`run_start` 不带 `cause`（follow-up 在同一次运行内处理，没有其他起因）；`state_changed` 带完整状态而非差量，UI 直接替换即可，状态很小。

`agent_start`/`agent_end` 是循环内部事件；协议只暴露 `run_start`/`run_end`。原因：上下文溢出时 pi 的序列是 `agent_end → 压缩 → 重试 → agent_end → agent_settled`，UI 必须知道第一个 `agent_end` 不是结束。合并后只看 `run_end`。

### 9.4 服务端 → 客户端请求

- `permission/request {toolCallId, kind, title, options[]}` → `{optionId}`；选项 `allow_once`、`allow_always`、`deny`。
- `ui/confirm`、`ui/select`、`ui/input`（插件用）。

客户端在 `initialize` 声明不支持时，服务端按策略处理（权限默认拒绝），不挂起。

### 9.5 断线恢复

WebSocket 重连带 `sinceSeq`：缓冲区还在就补发，否则返回 `resync_required`，客户端用 `session/state` + `session/messages` 重建。stdio 下进程即连接，只发 seq 不做缓冲。

### 9.6 错误

JSON-RPC 标准码，加应用码：`NOT_INITIALIZED`、`UNSUPPORTED_VERSION`、`SESSION_NOT_FOUND`、`ENTRY_NOT_FOUND`、`MODEL_ERROR`（未知厂商或模型、缺少凭据合为一个码，消息说明原因）。运行中也允许修改设置，下一次运行生效，所以没有 `session_busy`。

### 9.7 版本

`initialize` 协商主版本号；新增字段和事件不升主版本。

## 10. 模型与配置

- 三类接口：`openai_chat`（OpenAI Chat Completions 兼容）、`openai_responses`（Responses API）、`anthropic`（Messages API）。httpx 直连，自己解析 SSE，不依赖官方 SDK。接口按厂商设置，单个模型可用 `models.<id>.api` 覆盖（同一网关按模型走不同接口，如 ghc-api）。流式数据用宽松的 msgspec Struct 解码：字段都可选，未知字段忽略。
- `openai_responses` 无状态：`store=false`，请求 `reasoning.encrypted_content`，推理项原样存进 `thinking.signature`，下一轮回放给同一模型；消息的 `phase`（`commentary`/`final_answer`）存进 `text.signature` 一并回放。
- 思考级别：`openai_responses` 映射为 `reasoning.effort`（`off` 不发 effort，不能关闭推理的模型用默认值）；`openai_chat` 和 `anthropic` 按厂商格式：`reasoning_effort`、`deepseek`、`openrouter`、`anthropic_adaptive`、`anthropic_budget`、`none`。
- 内置厂商预设：anthropic、openai（Responses）、deepseek、moonshot、openrouter、ollama。未知模型用默认参数（200k 上下文、32k 输出）。
- Anthropic 请求在系统提示词和最后一条消息上打缓存标记。
- 配置：`~/.pyagent/config.json`（可用 `PYAGENT_HOME` 改目录），camelCase，与线上格式一致：

```json
{
  "defaultProvider": "deepseek",
  "defaultModel": "deepseek-v4-pro",
  "defaultThinking": "high",
  "providers": {
    "deepseek": { "apiKey": "sk-..." },
    "myproxy": {
      "api": "openai_chat",
      "baseUrl": "http://localhost:8080/v1",
      "apiKeyEnv": "MYPROXY_KEY",
      "models": { "foo": { "contextWindow": 128000, "reasoning": true } }
    }
  }
}
```

## 11. 运行模式

| 命令 | 行为 |
|---|---|
| `pyagent "..."` / `-p` | 一次性运行，流式输出回答 |
| `pyagent --mode json "..."` | 每行一个事件信封（与协议 `session/event` 的参数相同） |
| `pyagent` | REPL：斜杠命令 `/new` `/sessions` `/resume` `/model` `/thinking` `/name` `/state`；Ctrl+C 中止回复 |
| `pyagent --mode rpc` | 协议服务，stdio |
| `pyagent serve --port N` | WebSocket（M4） |

公共参数：`-c/--continue` 续接最近会话，`--session ID` 恢复指定会话，`--no-session` 不保存，`--provider` `--model` `--thinking` `--cwd`。

所有模式都经由协议：一次性运行和 REPL 是进程内客户端，rpc 模式是同一个服务端挂在 stdio 上。REPL 的事件循环跑在后台线程，主线程阻塞在 `input()` 并接收 Ctrl+C，这在 Windows 上最可靠。

## 12. 测试

- faux 适配器按脚本回放回复，覆盖循环、中止、并发、steering，不花 token。
- 厂商适配器用 `httpx.MockTransport` 喂录制的 SSE，检查请求体和解析结果。
- 工具在临时目录测。
- 协议：进程内客户端跑完整流程，断言事件序列、`seq` 连续性、错误码、持久化与重载；另有一个真实 `--mode rpc` 子进程对接本地假 HTTP 端点的集成测试。
- 真实 API 测试单独标记，默认不跑。

## 13. 里程碑

| 阶段 | 内容 |
|---|---|
| M1 内核 | ai 层（openai_chat、openai_responses、anthropic、faux）、agent 循环、read/write/edit/bash、print 与 json 模式 |
| M2 会话与协议（已完成） | JSONL 会话、AgentSession、协议 v1 + stdio、Python 客户端、REPL |
| M3 实用性 | 压缩、重试、权限模式与 `permission/request`、grep/find/ls、技能（skills）、`commands/list`、REPL 中运行时 steering |
| M4 UI 支撑 | WebSocket + seq 补发、Schema/TS 类型导出、ACP 适配（可选）、参考 UI |
