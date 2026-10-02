"""Summarize the fixed F/Q/M factorial, including regressions and uncached costs."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('out');args=p.parse_args();out=Path(args.out)
    report=json.loads((out/'report.json').read_text())
    latency=json.loads((out/'latency.json').read_text())
    cases=[json.loads(l) for l in (out/'cases.jsonl').read_text().splitlines()]
    assert len(cases)==861 and len({(c['sample_id'],c['question_index']) for c in cases})==861
    metrics=report['metrics']['10'];cost=latency['metrics']
    ranked=sorted(metrics,key=lambda n:(-metrics[n]['hit_rate'],-metrics[n]['micro_recall'],-metrics[n]['recall'],cost[n]['mean']))
    winner=ranked[0];paired={};samples=sorted({c['sample_id'] for c in cases})
    sizes=np.array([sum(bool(c['evidence']) for c in cases if c['sample_id']==sid) for sid in samples])
    rng=np.random.default_rng(20261002);resampled=rng.integers(0,len(samples),size=(10000,len(samples)))
    for name in ranked:
        wins=[dict(sample_id=c['sample_id'],question_index=c['question_index'],question=c['question']) for c in cases if c[name]['hit_evidence'] and not c['ABC']['hit_evidence']]
        losses=[dict(sample_id=c['sample_id'],question_index=c['question_index'],question=c['question']) for c in cases if not c[name]['hit_evidence'] and c['ABC']['hit_evidence']]
        changes=np.array([sum(bool(c[name]['hit_evidence'])-bool(c['ABC']['hit_evidence']) for c in cases if c['sample_id']==sid) for sid in samples])
        deltas=changes[resampled].sum(axis=1)/sizes[resampled].sum(axis=1)
        paired[name]=dict(wins=wins,losses=losses,hit_delta_cluster_percentile_95=np.percentile(deltas,[2.5,97.5]).tolist())
    summary=dict(winner=winner,ranked=ranked,metrics=report['metrics'],changes=report['changes'],latency=cost,
                 paired=paired,abc_parity=report['abc_parity'],uncached_parity=sum(c['exact'] for c in latency['checks']),uncached_checks=len(latency['checks']))
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# ABC 三项改进的八组合完整测试','',
           'F：上下文/目标融合与局部问答关联；Q：多人子查询与排序合并；M：人物/日期软召回。表中 F、Q、M 均指在原始 ABC 上追加改进。', '',
           '共同起点 ddaf4b4；独立分支 codex/abc-fusion-qa（6822ce8）、codex/abc-multi-query（fe872fa）、codex/abc-soft-recall（de43a6e）；组合分支 codex/abc-factorial。', '',
           '同一远端服务器 GPU 1、相同 BGE/MiniLM、完整记忆库的独立快照。861 道纯文本问题全部执行；859 道有证据问题进入准确率统计，证据总数 1,074。', '',
           '| 方案 | Hit@10 | Recall@10 | Micro Recall@10 | 全证据@10 | 新增/退化题 | 平均检索秒 | P95 秒 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for n in ranked:
        m=metrics[n];t=cost[n];ch=report['changes'][n]
        lines.append(f"| {n} | {m['hit_rate']*100:.2f}% | {m['recall']*100:.2f}% | {m['micro_recall']*100:.2f}% | {m['all_evidence_hit_rate']*100:.2f}% | +{ch['wins']}/−{ch['losses']} | {t['mean']:.3f} | {t['p95']:.3f} |")
    m=metrics[winner];b=metrics['ABC'];ci=paired[winner]['hit_delta_cluster_percentile_95']
    lines+=['',f"按事先固定的选择规则：先最大化 Hit@10，平分时依次比较 Micro Recall、逐题 Recall、平均延迟。本次最优为 **{winner}**。Hit@10 相对 ABC 增加 {(m['hit_rate']-b['hit_rate'])*100:.2f} 个百分点；Micro Recall 增加 {(m['micro_recall']-b['micro_recall'])*100:.2f} 个百分点。",'',
            f"冻结 ABC 的 Top10 ID 精确一致：{report['abc_parity']}/861；无缓存复核精确一致：{summary['uncached_parity']}/{summary['uncached_checks']}。",'',
            '准确率运行共享完全相同 query/document 的模型分数，仅减少重复推理，未使用答案或金标改变检索。6888 次生产 search 路径执行完成。延迟来自预先固定的均匀抽样 60 题，每种配置独立无缓存推理且交替顺序；不含 HTTP、启动或写入，不是全量延迟或并发吞吐量。', '',
            f"按 10 个会话聚类重采样，{winner} 相对 ABC 的 Hit 差值 95% 描述性区间为 {ci[0]*100:+.2f} 至 {ci[1]*100:+.2f} 个百分点。数据已用于开发与诊断，此区间不是独立泛化保证，也未校正多方案选优偏差。",'',
            '所有实验参数在本次全量测试前固定，没有根据这次逐题结果回调。最优仅指本次八个实现/组合，不代表三类方法的全局最优。线上比赛服务未被切换。', '', '## Top5 指标','',
            '| 方案 | Hit@5 | Recall@5 | Micro Recall@5 | 全证据@5 |','|---|---:|---:|---:|---:|']
    for n in ranked:
        m=report['metrics']['5'][n]
        lines.append(f"| {n} | {m['hit_rate']*100:.2f}% | {m['recall']*100:.2f}% | {m['micro_recall']*100:.2f}% | {m['all_evidence_hit_rate']*100:.2f}% |")
    lines+=['','## 最优方案的新增命中与退化','']
    for field,label in [('wins','新增命中'),('losses','退化')]:
        lines+=['### '+label,'']
        lines += [f"- {c['sample_id']} / q{c['question_index']}：{c['question']}" for c in paired[winner][field]] or ['无。']
        lines+=['']
    (out/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('winner',winner,'ranked',ranked,'abc_parity',report['abc_parity'],'uncached_parity',summary['uncached_parity'])


if __name__=='__main__':main()
