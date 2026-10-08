"""Authored multi-type live HTTP audit; not a public benchmark."""
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import json
import time
from fastapi.testclient import TestClient
from memory.config import load_settings
from memory.vanilla import VanillaMemory
from memory.aml_api import create_app

FACTS = [
 '小林最喜欢的足球队是阿森纳。',
 '小林对花生过敏，不能吃含花生的食品。',
 '老周在滨河学校工作，职业是音乐老师。',
 '小梅喜欢无糖豆浆，不喜欢甜豆浆。',
 '小林的姐姐叫小梅。',
 '小梅的老师是老周。',
 '小王和小林是同事。',
 '小陈是小王的父亲。',
 '为小林撰写周报时，先写结论，再用项目符号列出进度。',
 '只有紧急事故报告可以省略背景介绍，其他报告都要保留背景。',
 '为小梅导出表格时，日期必须用YYYY-MM-DD格式。',
 '发布正式文档前，必须先由小王审核，再由小林批准。',
 '小林在2023年10月参加了杭州马拉松。',
 '小梅在2024年5月3日去了北京听音乐会。',
 '小王原计划2024年6月去上海出差，但后来取消了这次出差。',
 '小陈先在2024年2月完成培训，随后在2024年3月通过考试。',
 '小林以前住杭州，2024年3月搬到了上海，现在住在上海。',
 '老周教的是小提琴，而不是钢琴。',
 '小王最喜欢曼联。',
 '小林昨天看了切尔西的比赛，但他并不支持切尔西。',
 '小梅给小王买了一件热刺球衣作为生日礼物。',
 '小王喜欢甜豆浆，并不喜欢无糖豆浆。',
 '小陈在山城学校工作，负责数学教学。',
 '小张的老师是老李。',
 '为小王写周报时先写背景，使用连续段落。',
 '为小陈导出表格时，日期用DD/MM/YYYY格式。',
 '小王在2022年参加过上海马拉松。',
 '小林计划明年去北京出差，尚未确定。',
 '小梅曾经住在上海，现在住在苏州。',
 '小陈对虾过敏，但不过敏于花生。',
 '普通报告应保留背景介绍，不能套用紧急事故报告的例外。',
 '小张的同事是小刘。',
]
CASES = [
 ('profile','小林最喜欢哪支足球队？',[0]),
 ('profile','小林对什么食物过敏？',[1]),
 ('profile','老周在哪里工作？',[2]),
 ('profile','小梅喜欢什么口味的豆浆？',[3]),
 ('relationship','小林的姐姐叫什么名字？',[4]),
 ('relationship','谁是小梅的老师？',[5]),
 ('relationship','小王和小林是什么关系？',[6]),
 ('relationship','小王的父亲是谁？',[7]),
 ('rule','小林的周报应该如何组织格式？',[8]),
 ('rule','什么情况下报告可以省略背景介绍？',[9]),
 ('rule','为小梅导出表格时日期格式是什么？',[10]),
 ('rule','正式文档发布前需要按什么顺序审批？',[11]),
 ('event','小林在2023年10月参加了什么比赛？',[12]),
 ('event','小梅去北京听音乐会是哪一天？',[13]),
 ('event','小王原计划六月去上海的出差后来怎样了？',[14]),
 ('event','小陈完成培训和通过考试的先后顺序是什么？',[15]),
 ('cross','小林搬家后现在住哪里，之前住哪里？',[16]),
 ('cross','小林姐姐的老师教什么乐器？',[4,5,17]),
 ('cross','小林姐姐喜欢什么口味的豆浆？',[4,3]),
 ('cross','小王的父亲在哪里工作？',[7,22]),
]


def main():
    out=Path('data/partition-types')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True)
    cfg=dict(load_settings(),RAG_MEMORY_DB=str((out/'memory.sqlite3').resolve()))
    assert cfg.get('RAG_PARTITION_MODE')=='dual' and cfg.get('RAG_MULTIHOP_MODE')=='off'
    b=VanillaMemory(cfg)
    traces=[]
    search=b.search
    def traced(payload):
        trace={}
        result=search(payload,trace=trace)
        traces.append(trace)
        return result
    b.search=traced
    rows=[]
    headers={'Authorization':'Bearer test'}
    with TestClient(create_app(settings={'AML_AUTH_MODE':'bearer','AML_API_KEY':'test'},backend=b)) as c:
        for start in range(0,len(FACTS),8):
            response=c.post('/add',headers=headers,json=dict(user_id='types-test',request_id=f'r{start}',
                session_id=f's{start}',messages=[dict(role='user',content=t) for t in FACTS[start:start+8]]))
            if response.status_code!=200: raise RuntimeError(f'Add {start} status {response.status_code}')
            print('Added',start+8,flush=True)
        with closing(b.connect()) as db:
            originals=[dict(r) for r in db.execute('SELECT m.id,m.content,s.source_id FROM rag_memories m JOIN rag_sources s ON s.memory_id=m.id')]
            labels={}
            for kind in ['profile','relationship','rule','event']:
                for r in db.execute(f'SELECT source_id FROM {kind}_sources'):
                    labels.setdefault(r[0],[]).append(kind)
        source_map={r['content']:dict(r,types=labels.get(r['source_id'],[])) for r in originals}
        (out/'sources.json').write_text(json.dumps(source_map,ensure_ascii=False,indent=2),encoding='utf8')
        for category,query,gold in CASES:
            row=dict(category=category,query=query,gold=[source_map[FACTS[i]] for i in gold])
            # Same live store, ranking configuration and top-k for the baseline.
            for mode in ['off','dual']:
                b.partition_search.mode=mode
                start=time.perf_counter()
                response=c.post('/search',headers=headers,json=dict(user_id='types-test',query=query,top_k=10))
                elapsed=time.perf_counter()-start
                hits=response.json().get('data',[])
                found={h['id'] for h in hits}
                ids={g['id'] for g in row['gold']}
                trace=traces[-1]
                selected=set(trace.get('partitions',[]))
                partition=set(trace.get('partition_channel_ids',[]))
                row[mode]=dict(status=response.status_code,seconds=elapsed,hits=hits,trace=trace,
                    hit10=int(bool(ids&found)),recall10=len(ids&found)/len(ids),all10=int(ids<=found),
                    gold_in_selected_partition=sum(bool(set(g['types'])&selected) for g in row['gold']),
                    gold_in_partition_channel=len(ids&partition),
                    retrieved_without_selected_label=[g['content'] for g in row['gold'] if g['id'] in found and not set(g['types'])&selected])
            rows.append(row)
            (out/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
            print('Tested',len(rows),'/',len(CASES),category,flush=True)
    summary={}
    for category in ['profile','relationship','rule','event','cross','all']:
        group=[r for r in rows if category=='all' or r['category']==category]
        summary[category]={mode:{metric:sum(r[mode][metric] for r in group)/len(group)
            for metric in ['hit10','recall10','all10','seconds']} for mode in ['off','dual']}
        summary[category]['gold_count']=sum(len(r['gold']) for r in group)
        summary[category]['gold_in_selected_partition']=sum(r['dual']['gold_in_selected_partition'] for r in group)
        summary[category]['gold_in_partition_channel']=sum(r['dual']['gold_in_partition_channel'] for r in group)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(out, json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
