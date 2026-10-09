"""Real gpt-4o-mini multi-hop recovery smoke on an isolated database."""
from datetime import datetime, timezone
from pathlib import Path
import json
import time
from fastapi.testclient import TestClient
from memory.config import load_settings
from memory.vanilla import VanillaMemory
from memory.aml_api import create_app


class TracedMemory(VanillaMemory):
    def search(self, payload, **kwargs):
        self.last_trace = {}
        return super().search(payload, trace=self.last_trace)


CASES = [
    ('chain', 'Where does the spouse of Leona work?', [
        'Leona is married to Tomas.', 'Tomas works at Cedar Labs.']),
    ('chain', 'Who is the CEO of the company where the spouse of Nadia works?', [
        'Nadia is married to Elias.', 'Elias works at Solstice Analytics.',
        'The CEO of Solstice Analytics is Ruth Chen.']),
    ('split', 'What is the total cost of my bicycle and tent?', [
        'I paid $240 for my bicycle.', 'My tent cost $180.']),
    ('direct', 'Where does Mira live?', ['Mira lives in Tallinn.']),
]


def main():
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=Path('data/unified-search-live')/stamp;out.mkdir(parents=True)
    cfg=dict(load_settings(),RAG_MEMORY_DB=str((out/'memory.sqlite3').resolve()),
             RAG_WRITE_GATE_MODE='on',RAG_BUILD_MODE='on',RAG_MULTIHOP_MODE='llm',
             RAG_MULTIHOP_RECOVERY='on',RAG_MULTIHOP_BINDINGS='on',RAG_MULTIHOP_NEEDS='on',
             RAG_MULTIHOP_VALIDATION='strict',RAG_MULTIHOP_PROMPT_STYLE='long',
             RAG_PARTITION_MODE='dual',RAG_RESULT_WINDOW='0',RAG_RERANK_MODE='onnx',
             RAG_RERANK_CONTEXT='0',RAG_RERANK_CANDIDATES='20',RAG_MULTIHOP_CANDIDATES='8',
             RAG_MULTIHOP_QUERIES='6',RAG_MULTIHOP_ROUNDS='3',RAG_MULTIHOP_LLM_CALLS='8',
             AML_AUTH_MODE='bearer',AML_API_KEY='live-test')
    backend=TracedMemory(cfg)
    rows=[]
    headers={'Authorization':'Bearer live-test'}
    with TestClient(create_app(settings=cfg,backend=backend)) as client:
        for index,(kind,question,evidence) in enumerate(CASES):
            texts=['Unrelated garden planning and a routine appointment.']*8+evidence+[
                'Leona visited Cedar Labs last year.', 'Nadia recommended a book about Paris.',
                'A bicycle repair cost $30.', 'The rental tent cost $20 per day.']
            body=dict(user_id=f'live-{index}',request_id='r1',session_id='s1',
                      messages=[dict(role='user',content=t,timestamp=1704067200000+j*60000)
                                for j,t in enumerate(texts)])
            added=client.post('/add',json=body,headers=headers)
            if added.status_code!=200: raise RuntimeError(f'add failed {added.status_code}: {added.text}')
            start=time.perf_counter()
            response=client.post('/search',json=dict(user_id=body['user_id'],query=question,top_k=10),headers=headers)
            elapsed=time.perf_counter()-start
            trace=dict(getattr(backend,'last_trace',{}))
            hits=response.json().get('data',[]) if response.status_code==200 else []
            found=sum(any(e in h['content'] for h in hits) for e in evidence)
            rows.append(dict(kind=kind,question=question,status=response.status_code,seconds=elapsed,
                evidence=evidence,found=found,trace=trace,hits=hits))
            (out/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
            print(kind,response.status_code,found,'/',len(evidence),
                  trace.get('strategy'),trace.get('stop'),trace.get('fallback'),flush=True)
    summary=dict(questions=len(rows),http_success=sum(r['status']==200 for r in rows),
        evidence_found=sum(r['found'] for r in rows),evidence_total=sum(len(r['evidence']) for r in rows),
        mean_seconds=sum(r['seconds'] for r in rows)/len(rows),
        recovery_queries=sum(r['trace'].get('recovery_queries',0) for r in rows),
        recovery_sources=sum(r['trace'].get('recovery_sources',0) for r in rows),
        fallback_count=sum(bool(r['trace'].get('fallback')) for r in rows))
    (out/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(out,json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
