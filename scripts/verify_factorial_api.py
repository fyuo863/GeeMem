"""Uncached real-model API parity on saved ablation cases; no external service."""
import argparse
import json
from pathlib import Path
import sys
from fastapi.testclient import TestClient

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from memory.aml_api import create_app
from memory.config import PROJECT_ROOT
from memory.rerank import LocalReranker
from memory.vanilla import LocalEmbedder
from benchmark_factorial import config,stores_for


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='data/factorial-benchmarks/20261001-v2');args=p.parse_args()
    out=PROJECT_ROOT/args.out;cfg=config(out)
    cases=[json.loads(l) for l in (out/'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    stores=stores_for(cfg,LocalEmbedder(cfg),LocalReranker(cfg))
    # First and last source-order questions cover different users and histories.
    checked=[]
    for name,store in stores.items():
        app=create_app(backend=store,settings=dict(AML_AUTH_MODE='none',AML_ALLOW_UNAUTHENTICATED='true'))
        with TestClient(app) as client:
            for c in [cases[0],cases[-1]]:
                response=client.post('/search',json=dict(user_id='factorial:'+c['sample_id'],query=c['question'],top_k=10))
                assert response.status_code==200,response.text
                ids=[h['id'] for h in response.json()['data']]
                assert ids==c[name]['ids'],(name,c['sample_id'])
                checked.append(dict(variant=name,sample_id=c['sample_id'],question_index=c['question_index'],match=True))
    (out/'api-parity.json').write_text(json.dumps(checked,indent=2),encoding='utf-8')
    print('Actual-model HTTP /search parity:',len(checked))


if __name__=='__main__':main()
