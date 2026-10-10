# 原文证据包上线记录（2026-10-10）

按用户要求部署上一版 raw 行为，不启用本轮候选翻页/缺口补查增强。

- 代码：`f3e941f`，从 GitHub `codex/state-evidence-pilot` 获取。
- 仓库保留覆盖增强代码，运行明确配置 `RAG_RAW_COVERAGE_MODE=off`。
- 旧服务器提交：`aeedc6007ce189995ba683a8d85cfbeb36007bec`。
- 服务器：`geeai-aiagent`；目录 `/GeeAI/AIAgent/CSIG`，按提交固定 checkout。
- 服务：`csig-aml-v1.service`；公网 `http://210.16.160.209:18092`。

启用 raw 原文包、固定读取版本、基础脱敏、规则适用性、按缺口补查开关。
关闭 coverage 增强、邻近组扩展、链条裁决、事实替代、状态选择、语义隐私；
多跳采用 long 标准路径，bindings/needs 关闭。旧有 supplemental 与此次新加的
coverage 缺口补查是不同开关，后者未上线。时间增强保持服务器原来的 off。
后续按用户要求补齐显式配置：写入构建器 `RAG_BUILD_MODE=on`，原有默认预算显式
设为3轮/6次查询/5次LLM/12条审阅证据/每条1200字符。正式服务开启独立位置日志
`data/logs/production-search-positions.jsonl`；隔离测试继续关闭位置日志，避免多
实例共享滚动文件。该调整不启用coverage、时间增强或事实裁决。

保留服务器自己的 LLM、embedding、reranker 凭据、接口鉴权、代理、并发限制和
生产数据库路径；没有复制本机密钥。模型仍为服务器原配置。

先通过服务器凭据、生产库副本、隔离工作目录运行 TestClient；确认迁移及功能
正常后，停止正式服务，保存最新 SQLite 一致性快照，再切换代码与配置并重启。
配置与快照保存在 `/home/agent/csig-deploy/rollback-raw-20261010/`，目录700、敏感
文件600。隔离工作目录 `/home/agent/csig-deploy/raw-stage-20261010` 保留供回查。

## 正式服务验证

使用独立 smoke 用户调用正式公网地址，不改写已有用户原文。

|项目|结果|耗时|
|---|---|---:|
|health|200|0.05秒|
|add，5条原文|200|7.81秒|
|相同请求再次add|200|0.14秒|
|关系题search|200，3/3所需原文，1个包|8.93秒|
|新旧信息search|200，2/2所需原文，1个包|14.21秒|
|其他用户查询|200，空结果|5.35秒|
|无凭据search|401|—|

另从本机访问公网 health 得到200，未鉴权search得到401。携带服务器凭据的
正式 Add/Search 由服务器发向公网地址完成；未将凭据导出到本机。
最后检查服务 active/running，NRestarts=0。

这是上线可用性验证，不是新一轮性能评测。包保留原文及来源时间，不声明关系
正确或事实有效。原始无密钥报告位于本机
`data/deployment/raw-production-smoke-20261010.json`，服务器对应备份目录的
`production-smoke.json`。旧代码与配置可回滚；快照仅作恢复依据，不应在发生
新写入后直接覆盖生产库，以免丢失新增记忆。
