"""Uncached real HTTP validation against frozen extension rankings."""
import argparse
import json
from pathlib import Path
import sys
import time
import httpx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from memory.config import load_settings


def main():
    parser=argparse.ArgumentParser();parser.add_argument('results');parser.add_argument('--variant',default='FM');args=parser.parse_args()
    out=Path(args.results);cfg=load_settings()
    cases={(c['sample_id'],c['question_index']):c for c in map(json.loads,(out/'cases.jsonl').read_text().splitlines())}
    latency=json.loads((out/'latency.json').read_text());identity=json.loads((out/'identity.json').read_text())
    selected=[c for c in latency['checks'] if c['variant']==args.variant]
    checks=[]
    with httpx.Client(base_url=cfg['AML_BASE_URL'],headers={'Authorization':'Bearer '+cfg['AML_API_KEY']},trust_env=False,timeout=1800) as client:
        assert client.get('/health').status_code==200
        anonymous=httpx.post(cfg['AML_BASE_URL']+'/search',json=dict(user_id='unknown',query='test',top_k=10),trust_env=False)
        assert anonymous.status_code==401
        for j,ref in enumerate(selected,1):
            case=cases[(ref['sample_id'],ref['question_index'])]
            body=dict(user_id=identity['prefix']+':'+ref['sample_id'],query=case['question'],top_k=10)
            before=time.perf_counter();response=client.post('/search',json=body);elapsed=time.perf_counter()-before;response.raise_for_status()
            hits=response.json()['data'];ids=[h['id'] for h in hits]
            checks.append(dict(sample_id=ref['sample_id'],question_index=ref['question_index'],exact=ids==case[args.variant]['ids'],seconds=elapsed))
            if j%10==0:print('HTTP_CHECKED',j,'/',len(selected),flush=True)
    report=dict(variant=args.variant,count=len(checks),exact=sum(c['exact'] for c in checks),unauthenticated_status=401,checks=checks,transport='real loopback HTTP; uncached production service')
    (out/'http-validation.json').write_text(json.dumps(report,indent=2))
    print('HTTP_PARITY',report['exact'],'/',report['count'],flush=True)
    assert report['count']==60 and report['exact']==60


if __name__=='__main__':main()
