# pyagent 协议 v1

供 UI 和集成开发者使用的参考。类型定义的唯一来源：

- `ghc_api/pyagent/protocol/jsonrpc.py`：信封、错误码、分帧
- `ghc_api/pyagent/protocol/types.py`：每个方法的参数与结果，方法表 `METHODS`
- `ghc_api/pyagent/coding/session_events.py`：事件信封、会话事件、`SessionState`
- `ghc_api/pyagent/ai/types.py`、`ai/events.py`、`agent/events.py`：消息、内容块、流式事件、工具事件

## 1. 传输

- JSON-RPC 2.0，双向。v1 只有客户端发起的请求和服务端发出的通知（事件）；服务端发起的请求（权限、对话框）在后续版本加入。
- stdio：启动 `python -m ghc_api.pyagent --mode rpc [--cwd DIR] [--no-session]`。
- 分帧：每行一个 JSON 对象，UTF-8，以 `\n` 结尾。只按 `\n` 切分；行尾的 `\r` 可容忍。服务端输出时转义 U+2028/U+2029，按这两个字符分行的读取器（如 Node.js `readline`）也不会切错。
- stdout 只有协议消息，诊断信息走 stderr。
- 请求按到达顺序逐个处理，响应顺序与请求顺序一致。客户端仍应按 `id` 匹配响应。
- 关闭 stdin：服务端处理完已收到的请求，中止进行中的运行（会话照常保存），然后以退出码 0 退出。
- Python：`pyagent.client.transports.ChildProcess` 启动子进程并连接；`connect_in_process` 在同一进程内连接（内置 CLI 和 REPL 用它，消息同样编码成行传输）。

## 2. 约定

- 字段名 camelCase；类型标签值 snake_case（`"type": "tool_call"`、`"role": "tool_result"`）。
- 可选字段缺省即零值：`null`、`false`、`0`、空数组。
- 客户端必须忽略不认识的字段和事件类型。新增字段、事件、方法不升主版本。
- 所有 id 都是不透明字符串。消息 `id` 同时是会话记录（entry）的 id。

## 3. 生命周期

```
→ initialize                       必须是第一个请求
→ session/new 或 session/load       得到 SessionState
→ run/prompt                       立即返回 {runId, disposition}
← session/event ...                run_start … run_end，然后 state_changed
→ run/prompt                       下一轮
```

一个连接可以同时打开多个会话；每个事件都带 `sessionId`。

## 4. 方法

| 方法 | 参数 | 结果 | 说明 |
|---|---|---|---|
| `initialize` | `{protocolVersion: 1, clientInfo?: {name, version?}}` | `{protocolVersion, agentInfo: {name, version}, capabilities: {multiSession, fork, images}}` | 版本不符返回 `UNSUPPORTED_VERSION` |
| `session/new` | `{cwd?, provider?, model?, thinking?}` | `SessionState` | 未填的用服务端工作目录和配置默认值 |
| `session/load` | `{sessionId}` | `SessionState` | 打开已保存的会话；已打开则直接返回状态 |
| `session/list` | `{cwd?}` | `{sessions: SessionSummary[]}` | 该目录的会话，最近更新的在前 |
| `session/close` | `{sessionId}` | `{}` | 中止运行并释放 |
| `session/state` | `{sessionId}` | `SessionState` | |
| `session/messages` | `{sessionId}` | `{messages: Message[]}` | 当前分支的全部消息 |
| `session/fork` | `{sessionId, entryId}` | `SessionState` | 新会话，内容为从开头到 `entryId`（含）的消息。要“编辑并重发第 N 条用户消息”，传它前一条消息的 id |
| `session/set_name` | `{sessionId, name}` | `SessionState` | |
| `run/prompt` | `{sessionId, content: (TextContent \| ImageContent)[], delivery?: "auto" \| "steer" \| "follow_up"}` | `{runId, disposition: "started" \| "queued"}` | 空闲时开始新运行（不论 delivery）；运行中则入队：`auto`/`steer` 在当前轮的工具调用之后插入，`follow_up` 在模型本该停下时插入 |
| `run/abort` | `{sessionId}` | `{texts: string[]}` | 中止运行；返回被丢弃的排队消息文本，UI 可放回输入框 |
| `run/clear_queue` | `{sessionId}` | `{texts: string[]}` | 清空队列 |
| `config/models` | `{}` | `{models: ModelInfo[]}` | 所有已配置厂商的模型 |
| `config/set_model` | `{sessionId, provider, model?}` | `SessionState` | 下一次运行生效；`model` 缺省用厂商默认模型 |
| `config/set_thinking` | `{sessionId, level}` | `SessionState` | `off` \| `low` \| `medium` \| `high` \| `max` |

## 5. 事件

服务端通知 `session/event`，参数为事件信封：

```json
{"sessionId": "8d0723428ae1", "seq": 12, "runId": "5134a5ff", "event": {"type": "...", "...": "..."}}
```

- `seq`：每个会话从 1 开始，每个事件加 1，不跳号。
- `runId`：运行期间的事件都带；`state_changed` 等运行之外的事件不带。
- `run/prompt` 的响应总是先于该次运行的 `run_start` 到达。

### 运行与轮次

| 事件 | 字段 | 含义 |
|---|---|---|
| `run_start` | | 运行开始 |
| `turn_start` / `turn_end` | `turn` | 一轮 = 一次模型回复及其工具调用 |
| `run_end` | `reason`（`completed` \| `aborted` \| `error`），`errorMessage?` | 运行结束；不发新 prompt 就不会再有动作 |
| `state_changed` | `state: SessionState` | 设置变化或运行结束后发出，携带完整状态 |
| `queue_changed` | `steering: string[]`，`followUp: string[]` | 完整的排队消息文本 |

### 消息流式

| 事件 | 字段 |
|---|---|
| `message_start` | `message` |
| `block_start` | `messageId`，`index`，`kind`（`text` \| `thinking` \| `tool_call`），`toolCallId?`，`toolName?` |
| `block_delta` | `messageId`，`index`，`delta` |
| `block_end` | `messageId`，`index` |
| `message_end` | `message`（权威完整版） |

还原规则：

1. `message_start` 时按 `message.id` 建立消息。用户消息和工具结果消息会紧接着收到 `message_end`，中间没有块事件。
2. `block_start` 在 `index` 处建块。`text`、`thinking` 块的 `delta` 是追加的文本；`tool_call` 块的 `delta` 是参数 JSON 的片段，拼完才是合法 JSON。
3. `message_end` 到达时，用其中的 `message` 整体替换拼出来的内容。

一次运行里事件的典型顺序：

```
run_start
message_start/end (user)
turn_start
  message_start (assistant) → block_* … → message_end
  tool_start → tool_update* → tool_end            每个工具调用一组
  message_start/end (tool_result)                 按模型调用的顺序
turn_end
turn_start … turn_end                             直到模型不再调用工具
run_end
state_changed
```

### 工具

| 事件 | 字段 | 说明 |
|---|---|---|
| `tool_start` | `toolCallId`，`name`，`kind`（`read` \| `edit` \| `execute` \| `search` \| `other`），`title` | `title` 是一行说明（如 `pytest -q`、`src/app.py`）；参数在 assistant 消息的 `tool_call` 块里 |
| `tool_update` | `toolCallId`，`partial: ToolResult` | 中间结果（bash 输出末尾，最多 10 次/秒）；每次是完整快照，不是增量 |
| `tool_end` | `toolCallId`，`result: ToolResult`，`durationMs?` | 未真正执行（参数无效、被拦截、中止）时没有 `durationMs` |

`ToolResult`：`{content: (TextContent | ImageContent)[], details?, isError?}`。`content` 是发给模型的内容；`details` 只给 UI：

- edit：`{diff, firstChangedLine}`，统一 diff 格式
- bash：`{exitCode, truncated?, fullOutputPath?}`

## 6. 数据类型

**内容块**

| type | 字段 |
|---|---|
| `text` | `text`，`signature?` |
| `image` | `data`（base64），`mimeType` |
| `thinking` | `thinking`，`signature?`，`redacted?` |
| `tool_call` | `id`，`name`，`arguments` |

`signature` 是回放给同一模型所需的厂商数据，UI 不需要理会。

**消息**（都带 `id`、`timestamp`，毫秒）

| role | 字段 |
|---|---|
| `user` | `content: (text \| image)[]` |
| `assistant` | `content: (text \| thinking \| tool_call)[]`，`provider`，`model`，`stopReason`（`stop` \| `length` \| `tool_use` \| `error` \| `aborted`），`usage`，`errorMessage?` |
| `tool_result` | `toolCallId`，`toolName`，`content`，`details?`，`isError?` |

`usage`：`{input, output, cacheRead, cacheWrite, cost: {input, output, cacheRead, cacheWrite, total}}`，token 数量和美元金额。`input` 不含缓存命中的部分。

**SessionState**

```
sessionId, cwd, model: {provider, id}, thinking, running, queue: {steering, followUp},
usage（全部回复之和）, contextTokens（最近一次回复的上下文大小）, contextWindow, lastSeq,
name?, runId?, file?（未保存时为空）
```

**SessionSummary**：`sessionId, cwd, updatedAt, messageCount, name?, firstMessage?`

**ModelInfo**：`provider, id, contextWindow, reasoning, images, available`（`available`：有凭据或该厂商不需要凭据）

## 7. 错误

| code | 名称 | 含义 |
|---|---|---|
| -32700 | PARSE_ERROR | 行不是 JSON（此时 `id` 为 `null`） |
| -32600 | INVALID_REQUEST | `jsonrpc` 不是 `"2.0"` |
| -32601 | METHOD_NOT_FOUND | 未知方法 |
| -32602 | INVALID_PARAMS | 参数不符，消息里有具体字段，如 ``Expected `str`, got `int` - at `$.sessionId` `` |
| -32603 | INTERNAL_ERROR | 服务端缺陷，详情在服务端 stderr |
| -32000 | NOT_INITIALIZED | 先调用 `initialize` |
| -32001 | UNSUPPORTED_VERSION | 协议版本不支持 |
| -32002 | SESSION_NOT_FOUND | 会话不存在或未打开 |
| -32003 | ENTRY_NOT_FOUND | 分叉点不存在 |
| -32004 | MODEL_ERROR | 未知厂商或模型，或缺少 API key |

模型调用失败（限流、网络等）不是请求错误：`run/prompt` 已成功返回，失败表现为 `stopReason: "error"` 的 assistant 消息和 `run_end {reason: "error"}`。

## 8. 示例

```
→ {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientInfo":{"name":"my-ui"}}}
← {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":1,"agentInfo":{"name":"pyagent","version":"0.1.0"},"capabilities":{"multiSession":true,"fork":true,"images":true}}}
→ {"jsonrpc":"2.0","id":2,"method":"session/new","params":{}}
← {"jsonrpc":"2.0","id":2,"result":{"sessionId":"8d0723428ae1","cwd":"/repo","model":{"provider":"local","id":"gpt-6.1-sol"},"thinking":"off","running":false,"queue":{"steering":[],"followUp":[]},"usage":{"cost":{}},"contextTokens":0,"contextWindow":922000,"lastSeq":0}}
→ {"jsonrpc":"2.0","id":3,"method":"run/prompt","params":{"sessionId":"8d0723428ae1","content":[{"type":"text","text":"hi"}]}}
← {"jsonrpc":"2.0","id":3,"result":{"runId":"5134a5ff","disposition":"started"}}
← {"jsonrpc":"2.0","method":"session/event","params":{"sessionId":"8d0723428ae1","seq":1,"event":{"type":"run_start"},"runId":"5134a5ff"}}
← ... message_start / message_end（用户消息）, turn_start, message_start（assistant）
← {"jsonrpc":"2.0","method":"session/event","params":{"sessionId":"8d0723428ae1","seq":6,"event":{"type":"block_start","messageId":"a1b2c3d4","index":0,"kind":"text"},"runId":"5134a5ff"}}
← {"jsonrpc":"2.0","method":"session/event","params":{"sessionId":"8d0723428ae1","seq":7,"event":{"type":"block_delta","messageId":"a1b2c3d4","index":0,"delta":"Hello"},"runId":"5134a5ff"}}
← ... block_end, message_end, turn_end
← {"jsonrpc":"2.0","method":"session/event","params":{"sessionId":"8d0723428ae1","seq":11,"event":{"type":"run_end","reason":"completed"},"runId":"5134a5ff"}}
← {"jsonrpc":"2.0","method":"session/event","params":{"sessionId":"8d0723428ae1","seq":12,"event":{"type":"state_changed","state":{"...":"..."}}}}
```

## 9. 后续版本

- 服务端发起的请求：`permission/request`（工具执行前确认），`ui/confirm`、`ui/select`、`ui/input`；`initialize` 将带客户端能力声明。
- `session/compact`、`commands/list`。
- WebSocket 传输，以及 `session/subscribe {sinceSeq}` 断线补发。
- 从类型定义导出 JSON Schema，供 UI 生成 TypeScript 类型。
