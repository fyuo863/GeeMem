"""Synthetic local/endpoint compatibility smoke, NOT an official AML evaluation."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import httpx
from fastapi.testclient import TestClient
from memory.aml_api import create_app
from memory.config import load_settings
from memory.store import Store


def auth_headers(cfg):
    mode=cfg.get('AML_AUTH_MODE','bearer').lower();key=cfg.get('AML_API_KEY','')
    if mode=='none':return {}
    if not key:raise ValueError('Configure AML_API_KEY in root .env')
    if mode=='x-api-key':return {'X-Api-Key':key}
    if mode not in ('bearer','token'):raise ValueError('Unsupported AML_AUTH_MODE')
    return {'Authorization':mode.title()+' '+key}


@contextmanager
def client_for(live):
    cfg=load_settings()
    if live:
        url=cfg.get('AML_BASE_URL','')
        parsed=httpx.URL(url)
        if parsed.scheme not in ('http','https') or not parsed.host or parsed.userinfo:
            raise ValueError('Set AML_BASE_URL to your adapter HTTP(S) URL without credentials')
        with httpx.Client(base_url=url.rstrip('/'),headers=auth_headers(cfg),timeout=1800,
                          trust_env=False,proxy=cfg.get('AML_HTTP_PROXY') or None) as c:
            yield c
    else:
        # No listener is opened. None authentication is confined to this isolated TestClient.
        with TemporaryDirectory(prefix='csig-aml-smoke-') as tmp:
            settings=dict(cfg,AML_AUTH_MODE='none',AML_ALLOW_UNAUTHENTICATED='true')
            with TestClient(create_app(store=Store(Path(tmp)/'memory.sqlite3'),settings=settings)) as c:
                yield c


def smoke(client):
    run='local-smoke:'+uuid4().hex
    user=run+':primary';other=run+':isolated'
    items=[
        dict(request_id=run+':1',user_id=user,session_id=run+':s1',messages=[
            dict(role='user',content='My name is Nora. I live in Hangzhou.')]),
        dict(request_id=run+':2',user_id=user,session_id=run+':s2',messages=[
            dict(role='user',content='My name is Nora. I train for a marathon three times a week.',timestamp=1704067200000)]),
        dict(request_id=run+':3',user_id=other,session_id=run+':s1',messages=[
            dict(role='user',content='My name is Evan. I visited Hangzhou last year.')])]
    health=client.get('/health');assert 200<=health.status_code<300,'Health failed'
    for p in items:
        r=client.post('/add',json=p)
        assert r.status_code==200,f'Add failed with HTTP {r.status_code}'
        assert r.json()==dict(success=True,**{k:p[k] for k in ('request_id','user_id','session_id')}),'Invalid Add contract'
    repeat=client.post('/add',json=items[0]);assert repeat.status_code==200,'Retry failed'
    allowed={p['messages'][0]['content'] for p in items[:2]}
    for question_index,query in enumerate(['Where does Nora live?','How often does Nora train for a marathon?']):
        body=dict(user_id=user,query=query,top_k=100)
        if 'live' in query:body['options']=['Hangzhou','Tokyo']
        r=client.post('/search',json=body);assert r.status_code==200,f'Search failed with HTTP {r.status_code}'
        data=r.json()['data'];assert 0<len(data)<=100,'No immediately searchable memory or too many results'
        assert all(h['content'] in allowed and isinstance(h['id'],str) and h['id'] for h in data),'Non-source evidence or isolation failure'
        assert items[question_index]['messages'][0]['content'] in {h['content'] for h in data},'Expected source evidence not retrieved'
        limited=client.post('/search',json=dict(body,top_k=1))
        assert limited.status_code==200 and len(limited.json()['data'])<=1
    r=client.post('/search',json=dict(user_id=run+':empty',query='Hangzhou',top_k=100))
    assert r.status_code==200 and r.json()=={'data':[]},'Unknown user isolation failed'
    return dict(run_id=run,checks=['health','official_add_envelope','optional_timestamp','idempotent_retry',
                'cross_session_search','options','top_k','source_only_results','user_isolation'],
                synthetic_users=[user,other],retrieval_nonempty=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true',help='Use AML_BASE_URL from .env; writes synthetic test memories to that endpoint')
    args=parser.parse_args();start=time.perf_counter()
    report=dict(official_evaluation=False,transport='http' if args.live else 'in_process',model='gpt-4o-mini')
    try:
        with client_for(args.live) as c:report.update(smoke(c))
        report['passed']=True
    except (AssertionError,ValueError,httpx.HTTPError) as exc:
        report.update(passed=False,error_type=type(exc).__name__)
        # Never print transport exception URLs/headers or evaluation request bodies.
        if isinstance(exc,AssertionError):report['check_error']=str(exc)
    report['seconds']=round(time.perf_counter()-start,3)
    out=ROOT/'data/aml-smoke'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True);(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2));print('REPORT',out/'report.json')
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
