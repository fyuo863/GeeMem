"""Reproducible rule storage benchmark and optional live extraction smoke test."""
import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from memory.rule import RuleBuilder, RuleRetriever, RuleCandidate, RuleExtraction


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--output',default='data/rule-tests/results.json')
    args=parser.parse_args()
    result={}
    with tempfile.TemporaryDirectory() as folder:
        b=RuleBuilder(Path(folder)/'rules.db'); r=RuleRetriever(b.db_path)
        writes=[]; reads=[]; hits=0
        for i in range(1000):
            ex=RuleExtraction(rules=[RuleCandidate(name=f'procedure{i}',condition=f'When running task{i}',
                actions=[f'Check document{i} before proceeding'],scope='work',message_indices=[0],confidence=1)])
            messages=[dict(message_index=0,source_id=f's{i}',content=f'For task{i}, check document{i}.')]
            start=time.perf_counter(); b.write('u',ex,messages); writes.append((time.perf_counter()-start)*1000)
        for i in range(100):
            start=time.perf_counter(); found=r.find_applicable('u',f'task{i}',limit=5); reads.append((time.perf_counter()-start)*1000)
            hits+=bool(found and found[0]['name']==f'procedure{i}')
        def stats(values):
            return dict(mean_ms=statistics.mean(values),p95_ms=sorted(values)[int(len(values)*.95)-1])
        result['storage']=dict(count=1000,write=stats(writes),read=stats(reads),hit1=f'{hits}/100',
                               evidence_complete=all(x['evidence'] for x in r.find('u',limit=1000)))
        if args.live:
            cases=[
                ('以后写报告先给结论，再列证据；紧急情况先说明风险。',True),
                ('处理退款时先核对订单，再确认金额，最后提交退款。已发货订单先联系客户。',True),
                ('以后发送邮件前一定先核对收件人，上次发错人造成了麻烦。',True),
                ('这次请帮我写一份报告。',False),
                ('我喜欢足球。',False),
                ('今天我和小王去看了电影。',False),
                ('我同事的团队规定上线前必须经过代码审查。',True),
                ('怎样才能写好报告？',False)]
            live=[]
            for i,(text,expected) in enumerate(cases):
                messages=[dict(message_index=0,source_id=f'live{i}',content=text,role='user',source_kind='evidence')]
                start=time.perf_counter()
                try:
                    ex=b.extract(messages); ids=b.write('live',ex,messages)
                    live.append(dict(input=text,expected=expected,correct=bool(ids)==expected,
                        output=ex.model_dump(),elapsed_s=time.perf_counter()-start))
                except Exception as error:
                    live.append(dict(input=text,correct=False,error=type(error).__name__,elapsed_s=time.perf_counter()-start))
            result['live']=live
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'storage':result['storage'],'live_correct':sum(x['correct'] for x in result.get('live',[])),
                      'live_count':len(result.get('live',[]))},ensure_ascii=False))


if __name__=='__main__': main()
