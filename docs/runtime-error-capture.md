# 运行中故障的旁路记录

`scripts/capture_runtime_errors.py` 是独立的 Linux 诊断进程，不导入业务模块，不连接数据库，不调用模型，不修改流量或服务配置。可在原 Full 续跑时使用，无需重启 API。

## 持续记录模式（2026-11-03）

服务器已启用独立单元 `csig-error-capture.service`，单元模板位于 `deploy/csig-error-capture.service`。它与 API/代理没有 `Requires`、`PartOf` 等生命周期依赖，启停记录器不会重启业务服务。

- 日志：`/root/CSIG/shared/diagnostics/runtime-live/events.jsonl`，UTC 时间；北京时间为 UTC 加八小时。
- 持续运行，每 30 秒写 `capture_heartbeat`；记录 journal 线程是否存活、代理数据包计数以及自上次统计以来 socket 的收包/丢包数。
- 每个事件文件最多 16 MiB，保留当前文件及三个历史文件 `events.1.jsonl` 至 `events.3.jsonl`，总量最多 64 MiB。记录器重启会追加当前日志，并继续轮转，不覆盖整段历史。
- `Restart=always` 仅重启记录器。诊断进程内存上限 160 MiB、CPU 配额为单核的 10%，降低优先级；这些限制没有施加给 API。达到诊断资源上限可能导致记录缺口，不代表业务请求失败。
- 连续模式仅保存元数据，不保存 Key、HTTP 头、问题/原文正文或 TLS 正文；禁止与 `--save-failed-add` 同时使用。

新增事件包括：`api_tcp_open` / `api_tcp_accept`、`api_http_headers`（仅方法、固定接口名或状态码）、`api_tcp_close`（FIN/RST 方向及待响应请求哈希）。即使请求体未收齐、响应采用 chunked 或正文超限，也尽量保留已观察到的首部状态；不能将“收到响应头”当成“完整响应已送达调用方”。

续跑后的公网方向复核发现，本机 Linux 上 `AF_PACKET + ETH_P_IP` 能看见物理网卡入站，却漏掉服务器出站的 SYN-ACK 和响应。已改为 `ETH_P_ALL` 接收，再由现有 `decode_ipv4()` 过滤 IPv4/TCP；仍不保存原始包。通过外部 `/health` GET 验证物理网卡出站 HTTP 状态，部署时只重启记录器并追加原日志，API 与代理均不重启。记录器切换期间有短暂观测空窗，业务请求继续处理。

已有应用 journal 的 `request_failed` / `request_finished`、`provider_started` / `provider_finished`、LLM 流完成/超时、embedding/rerank 重试和上游状态等只按字段白名单收集。没有修改业务调用链，缺失于原应用日志的供应商调用不能通过旁路记录器补造。代理 TLS 仍只记录大小、时间和 FIN/RST，不能读取加密错误正文。

常用操作（只针对记录器）：

```bash
systemctl status csig-error-capture.service --no-pager
tail -n 30 /root/CSIG/shared/diagnostics/runtime-live/events.jsonl
systemctl stop csig-error-capture.service
systemctl start csig-error-capture.service
```

首次安装前须创建输出目录（权限 `0700`），将脚本复制到 `/root/CSIG/shared/diagnostics/tools/`，将单元安装到 `/etc/systemd/system/`，执行 `systemctl daemon-reload` 和 `systemctl enable --now csig-error-capture.service`。不应重启 `csig-aml.service` 或 `csig-aml-proxy.service`。

本轮验证：本机与服务器各运行同一组 8 项离线测试，均通过；服务器隔离端口人工返回 502，验证请求/响应关联、连接事件、无密钥/原文落盘及记录器重启追加。仅通过生产 `/health` GET 验证真实端口旁路抓取，没有向生产 Add 写入测试消息，没有调用模型。

安装前后核对 API PID `291452`、代理 PID `585`、启动时间及重启计数保持一致；业务文件和 `.env` 校验和一致，数据库路径与 inode 未变。此核对未连接数据库或扫描业务表；保持原 Full 的请求 ID、幂等记录和续跑数据，没有创建新的评测任务。

安装证据：`/root/CSIG/shared/diagnostics/capture-install-20261103T105412Z/`，包含安装前状态、隔离结果及 `verification-result.json`。记录从启动时开始，无法补回此前 18:44 的 TCP 数据。无包、抓包丢包、日志轮转或不支持的协议均可能造成观测缺口，不能仅凭没有日志判定调用方未发请求。

## 有界单次抓取

记录内容：

- `/add`、`/search` 的请求哈希、正文哈希、消息数量、字符数、响应状态和耗时。
- 原服务 journal 中已有的失败阶段、异常类型、源码位置及上游 HTTP 状态；只复制允许的字段。
- 应用到本地代理、代理到远端节点的 TCP 数据包大小、时间间隔，以及 FIN/RST 的方向。TLS 正文不解密、不落盘。
- 使用 `--save-failed-add` 时，仅把返回 5xx 的 Add JSON 正文保存到私有文件；不保存 HTTP 请求头。成功请求不保存正文，重复失败正文按 SHA256 去重。

若公网响应未能重组，应用的 `request_failed` 日志也会触发保存最近五分钟内同一请求 ID 的正文候选，并明确标记 `failed_add_candidate`。候选缓存最多 16 条、合计 16 MiB；同一请求 ID 若出现不同正文，会分别保存，不会仅凭 ID 判定其内容相同。

示例（root 或 CAP_NET_RAW）：

```bash
python scripts/capture_runtime_errors.py \
  --output /root/CSIG/shared/diagnostics/runtime-errors-UNIQUE_TIME \
  --duration 7200 --max-mb 64 --save-failed-add
```

输出目录权限 `0700`、文件权限 `0600`。必须为每次运行使用新目录，防止覆盖历史记录。默认两小时或 64 MiB 后停止。建议通过独立的 systemd 临时单元启动，并设置 `MemoryMax=160M`、`CPUQuota=10%`；限制只作用于诊断进程。

`events.jsonl` 中用 `request_hash` 关联应用日志与请求响应，用 `connection` 关联同层 TCP 事件，用 UTC 时间对齐不同层。原文文件只供私下复现，不应提交 Git 或直接粘贴聊天。

捕获边界：IPv4 TCP、HTTP/1 Content-Length，单方向缓冲最多 8 MiB，最多保留 16 个 API 连接。超限、缺失首部或 chunked 请求不会猜测正文边界；可识别的超限会记录 `http_capture_unavailable`。退出时记录 socket 丢包统计。代理连接复用时，外层 TCP 不能单凭时序精确映射到某一个内部请求；该工具不能替代供应商日志或进程内部的 HTTPX trace。

离线验证：

```bash
python -m unittest discover -s tests -p test_runtime_capture.py -v
```

服务器隔离验证采用临时本地 HTTP 服务返回人工 502，检查正文保存、鉴权头过滤和 TCP 事件；不向生产 `/add` 写入测试数据。

## 2026-11-02 真实续跑记录

- 北京时间 23:53:05，`/add` 在 `write_classifier` 阶段失败，返回 502；异常链为 `LLMError → RemoteProtocolError`，应用位置 `memory/llm.py:42`。该轮包含三次模型请求，总耗时约 18.27 秒。
- 对应请求包含一条 8000 字符消息。其正文通过最近请求与应用失败日志关联保存，候选数量为 1；保存文件权限为 `0600`。这次关联来自应用失败日志，公网响应仍未成功重组。
- 三次尝试都先观察到代理外连的远端 FIN，随后本地代理关闭应用连接；最后一次响应静默约 5.18 秒。该观测支持外部链路先断开，不能单凭 TLS 包确定具体哪个中间组件设置了超时。
- 对保存的实际输入进行两次隔离分类调用，仅改变是否使用流式响应：非流式 5.874 秒后发生 `RemoteProtocolError`；流式 HTTP 200，1.464 秒收到响应头，12.270 秒完成，795 个输出 token，JSON Schema 验证通过。没有调用生产 `/add`，没有写入数据库。
- 两次隔离调用使用相同模型、分类提示词、输入和结构化输出 schema。这验证了该输入可被模型处理，并为“非流式等待期间链路断开”的判断提供直接对照；不能据此保证所有请求改成流式后都不再失败。

诊断会话：`/root/CSIG/shared/diagnostics/runtime-errors-20261102T155156Z/`。隔离结果保存在其中的 `isolated-classifier-replay.json`；原文不复制到仓库。API 主进程在安装和更新记录器期间保持为 `267920`，未重启 API 或代理。

后续修复方向是让模型适配器接收流式结构化输出，完整拼接后执行现有 schema 校验，并向调用者返回原有对象；保持 `/add` 接口、分类语义、数据库和请求幂等逻辑不变。本轮只安装诊断工具，未部署该业务改动。

## 2026-11-03 原因复核与修复方案

北京时间 10:01 核对应用 journal：从 11 月 2 日 23:00 起，共有 64 条 `request_failed`，全部属于同一个请求哈希，阶段均为 `write_classifier`，状态均为 502，原因链均含 `RemoteProtocolError`。最新一条发生于 11 月 3 日 09:56:40，耗时 18.711 秒。统计只计算失败记录，不把对应的 `request_finished` 再算一次。

10:04 使用此前保存的一条 8000 字符真实输入再次隔离复现，顺序为先流式、后非流式，模型、提示词与 schema 相同：

| 模式 | 结果 | 总耗时 | 输出 token |
| --- | --- | --- | --- |
| 原 schema，流式 | HTTP 200，收到 `[DONE]`、`finish_reason=stop`，通过 schema 和分类程序校验 | 8.232 秒 | 800 |
| 原 schema，非流式 | `RemoteProtocolError`，未收到响应头 | 5.482 秒 | 无有效响应，无法统计 |
| 仅将索引数组 `maxItems` 从 200 改为实际消息数 1，非流式 | HTTP 200，通过 schema 校验，分类仍为 `event` | 3.049 秒 | 192 |

第二行 HTTPX trace 显示：请求正文在 0.298 秒发送完成，等待响应头到 5.481 秒时失败，静默等待约 5.18 秒。因此不是本机配置的 60 秒读取超时先触发。前一晚 TCP 抓包也显示远端先 FIN，再由代理关闭应用连接。具体是代理节点还是后续网关设置了短静默限制，现有日志不能确定。

流式复查同时发现：仅有一条消息时，模型仍在选中标签的 `message_indices` 中输出了 200 个重复的 `0`，现有程序随后去除了 199 个重复项。`indexed_result_schema()` 虽限制索引只能为 0，但数组上限仍固定为 200，允许模型生成这些冗余索引。它增加输出量和生成耗时，是本例反复超过外部链路等待时间的重要诱因。将数组上限收紧后，单例输出从 800 token 降为 192 token；这是同一输入的隔离对照，不代表全数据集收益。

建议按以下顺序修复：

1. `memory/selector.py`：将动态结果 schema 的 `message_indices` 长度上限设为实际消息数。唯一有效索引数量不可能超过消息数，因此不减少合法来源覆盖；保留已有越界校验和重复去除。
2. `memory/llm.py`：内部采用流式接收，完整聚合后执行原 schema 校验；确认正常结束，拒绝截断、拒答和无效 JSON，绝不将部分结果交给写入器。外部 `/add`、`/search` 仍按现有协议返回完整 JSON。
3. 复用现有三次重试预算，避免传输层与判断器重复放大次数。传输失败重试保持原始提示词；确有格式错误时才附加修复提示。记录各次调用的响应头耗时、完成耗时和错误类型，继续不记录密钥及原文。

上线验证应覆盖真实失败原文、1/2/200 条消息的索引上限、重复索引、流中断、非正常结束和非法 JSON。验证通过后，在原任务停止时部署；不迁移或清空数据库，不改变请求 ID、用户 ID、源文 ID、幂等记录或 embedding 配置，保留原 Full 任务的续跑数据。本轮只做隔离诊断，API 主进程与代理进程均未重启。

原始复查数据保存在服务器诊断目录的 `isolated-recheck-20261103T020430Z.json` 和 `bounded-schema-probe-20261103T020608Z.json`，原文仍仅存私有文件。

## 2026-11-03 修复实现与部署验证

本轮只替换三个业务文件：`memory/llm.py`、新增的 `memory/llm_transport.py`、`memory/selector.py`。远端补丁从其原文件生成，未覆盖本机与服务器之间已有的其他诊断改动。

- 判断器来源索引数组的上限改为实际消息数，保留现有分类标准、索引越界校验和去重。
- LLM 通过 SSE 接收结构化输出，收到完整内容、正常结束标志和 `[DONE]` 后，继续执行原 JSON Schema 校验。外部接口仍返回一次完整 JSON。拒答、截断、无效 JSON 和中断均报错，不会把半成品交给写入器。
- 传输层不新增重试；分类器仍使用原有三次尝试。网络/HTTP 失败重试原提示词，实际格式或证据校验错误才进入修复提示。
- 保留原 60 秒 I/O 超时，另在收到行时检查累计 60 秒预算及 8 MiB 流上限；行间检查不是严格的总墙钟截止时间。日志只记录耗时和错误类型，不记录密钥或原文。

验证范围：

| 检查 | 结果及边界 |
| --- | --- |
| 本机回归 | 110 项通过；覆盖分片 UTF-8、流结束/中断、错误状态、schema、索引、三次重试、来源绑定、写入与幂等 |
| 真实失败输入 | 服务器 Python 3.10 隔离调用，原 8000 字符输入约 4.861 秒完成，分类为 `event`，来源索引数为 1；未写生产库 |
| 隔离接口 | 真实 `gpt-4o-mini` 分类，人工确定性向量；Add/Search/Health 均 200，重复 Add 和重新打开数据库后重试均只分类一次，更改正文的同 ID 请求返回 409 |
| 公网与鉴权 | 本机直接访问新服务器 `/health` 返回 200；无鉴权 Search 返回 401，有鉴权 Search 返回 200，无数据测试用户返回空证据 |

用户确认原 Full 已失败停止、尚未续跑后才重启 API，未重启代理。部署前已有 44,251 条请求记录、552,264 条原文关联记录和 552,264 条向量记录。采用 SQLite 在线备份 API 生成一致性副本，核对副本大小、schema 和上述计数；额外的完整 `quick_check` 扫描因磁盘 I/O 开销停止，未宣称它已通过。

部署后再次核对：20 张业务表的行数和 schema 均与部署前一致；完整请求记录及原文关联记录的逐字段 SHA256 摘要也一致。结果保存在 `data-after-result.json`，没有把成功写入的数据恢复到较早备份。

首次部署把健康等待设为 45 秒，短于该库既有的启动来源回填耗时，触发了代码自动回退。随后把等待上限改为 600 秒再部署；没有回退或替换数据库。最终 API 进程为 `291452`，代理仍为 `585`，本次成功部署重启前后同一只读连接的 `PRAGMA data_version` 未变化，根 `.env` 和 embedding 标识也未变化。

原 Full 的任务标识、用户/请求/来源 ID、请求摘要、幂等逻辑和 embedding 配置均保留。没有向生产 Add 注入原失败请求，也没有创建新的 Full；该请求由平台在原任务续跑时再次提交。

部署记录目录：`/root/CSIG/shared/diagnostics/stream-fix-20261103T022100Z/`。其中保存补丁文件校验和、代码原件、一致性数据库备份，以及真实输入和接口测试的无原文结果。此验证针对本次故障与兼容性，不是一次全数据集准确率或吞吐量测试。


## 2026-11-03 生产修复归档

本次归档服务器实际使用的并发等待、超长查询保护、流式结构化响应与分类索引修复，并纳入旁路日志工具和配套测试。生产业务文件保持原样，未重启服务或替换数据库。

在无生产 .env 的临时副本中，使用服务器 Python 3.10 运行 13 个相关测试文件，182 项离线测试通过。覆盖流完整性与用量保留、超长查询、并发等待、分类索引、接口协议、幂等恢复和诊断脱敏；这不是模型准确率或吞吐量测试。测试依赖仅安装在临时目录，未修改生产虚拟环境。
