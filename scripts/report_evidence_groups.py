"""Summarize source visibility and delivery; these are not semantic correctness scores."""
import argparse
import json
from pathlib import Path
from statistics import mean, median


def main():
    parser=argparse.ArgumentParser();parser.add_argument('report',type=Path);args=parser.parse_args()
    report=json.loads(args.report.read_text(encoding='utf8'));mapping=report['source_mapping'];rows=[]
    for case in report['cases']:
        if 'off' not in case or 'combined' not in case: continue
        row={'name':case['name'],'question':case['question'],'kind':case['kind']}
        if 'answerable' in case:row['answerable']=case['answerable']
        for mode in ['off','combined']:
            run=case[mode];trace=run['trace'];reviewed={sid for step in trace['rounds'] for sid in step['evidence_ids']}
            source_gold=set(case['gold'])
            visible=len(source_gold & {mapping.get(sid) for sid in reviewed}) if case['kind']=='dataset' else None
            row[mode]=dict(found=len(run['found']),required=len(source_gold),reviewed_gold=visible,
                status=trace.get('evidence_chain',{}).get('status'),fallback=trace.get('fallback'),
                llm_calls=trace['llm_calls'],search_calls=trace['search_calls'],seconds=run['seconds'],
                groups=len(trace.get('adjacency_groups',[])),bundle=trace.get('evidence_bundle',{}).get('status'))
        rows.append(row)
    summary={'cases':rows,'questions':len(rows),'source_coverage_increased':[r['name'] for r in rows if r['combined']['found']>r['off']['found']],
        'source_coverage_decreased':[r['name'] for r in rows if r['combined']['found']<r['off']['found']]}
    for mode in ['off','combined']:
        summary[mode]=dict(mean_source_coverage=mean(r[mode]['found']/r[mode]['required'] for r in rows),
            all_sources=sum(r[mode]['found']==r[mode]['required'] for r in rows),
            median_seconds=median(r[mode]['seconds'] for r in rows),
            llm_calls=sum(r[mode]['llm_calls'] for r in rows),
            false_complete_controls=[r['name'] for r in rows if r.get('answerable') is False and r[mode]['status']=='complete'])
    dest=args.report.with_name('group_summary.json');dest.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in summary.items() if k!='cases'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
