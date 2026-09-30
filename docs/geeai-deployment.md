# geeai-aiagent 部署记录

部署日期：2026-09-30。远端账户 agent；部署根目录 /home/agent/csig-aml，应用版本 ed799c8（基线 c38a690），当前研究分支 codex/agentmemories-evaluation。

## 地址与状态

- Add：http://210.16.160.209:18092/add
- Search：http://210.16.160.209:18092/search
- Health：http://210.16.160.209:18092/health
- 鉴权：Authorization: Bearer <Memory System Key>，密钥只在本地 .env 与远端 shared/.env 保存，不在本记录中。
- 公网 Health 已验证 200；公网未认证 Search 已验证 401。
- 服务器经本机 HTTP 端口执行真实 gpt-4o-mini 合成 smoke，通过全部检查，用时 25.594 秒。摘要位于本机 data/agentmemories/remote-smoke-report.json。
- 当前是 HTTP 端点，尚无 TLS。对外提交前建议配置用户提供域名及 HTTPS，避免 Bearer 凭据明文传输。

## 布局

- current -> releases/ed799c8，固定代码快照。
- .venv：Python 3.14.4 独立运行时。依赖版本记录在 deploy/geeai-requirements.lock。
- shared/.env：0600，配置仅从该文件读取；release/.env 是其符号链接。
- shared/data/memory.sqlite3：独立记忆库，目前仅有自建合成 smoke 数据，无官方私有数据。
- 用户 systemd 服务 csig-aml.service；已启用自动启动和失败重启，agent 的 Linger=yes，SSH 断开后继续运行。
- 专用出站代理 csig-aml-proxy.service，仅监听 127.0.0.1:18093。它使用当前 agent 账号已有代理配置中的单个节点，独立运行，不修改既有共享代理。
- 代理私有配置 shared/proxy.yaml 不入库。订阅或节点改变后需更新这个专用配置并重跑模型连通性测试。

原先试用的 127.0.0.1:7890 出口被模型接口以 unsupported_country_region_territory 拒绝；agent 原代理端口 55916 的连接也失败。专用代理已通过认证模型列表请求及完整真实 Add/Search smoke。

## 运维

```sh
ssh geeai-aiagent
systemctl --user status csig-aml.service csig-aml-proxy.service
journalctl --user -u csig-aml.service -n 50 --no-pager
cd /home/agent/csig-aml/current
/home/agent/csig-aml/.venv/bin/python scripts/smoke_aml.py --live
```

正式评测期间保持固定代码与接口版本。不要用研究测试脚本记录官方请求正文/候选。生产前还需明确数据清理、备份、容量监控与节点续期责任，正式评测数据和派生副本按官网要求在结束后 30 天内删除。

## 官方申请准备

官方申请入口位于 https://agentmemories.ai/api-guide 的 Apply for Evaluation Access。当前页面申请请求为 POST /evaluation-access-requests，开源/学术选项值 academic，接入方式 participant_api。页面会要求邮箱、系统版本、公网端点、Memory System Key、公开 GitHub 仓库及部署/来源说明。

已生成 data/agentmemories/application-draft.json。其中缺失字段为空，memory_api_key 刻意不保存；不能直接作为完整申请提交。未向官方发送申请、未取得 Eval Key、未消耗官方 smoke/full 配额。

仍需用户提供：联系邮箱；联系人与团队（个人可说明）；公开 GitHub 仓库；原创/复用及许可声明；如需 HTTPS，提供域名或现有反向代理入口。当前仓库没有 Git remote；未擅自发布源码。
