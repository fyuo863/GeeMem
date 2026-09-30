"""Graph-only real /add test, no retrieval questions or gold sent to the model."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fastapi.testclient import TestClient
from memory.api import create_app
from memory.llm import LLM, LLMError
from memory.store import Store
from scripts.test_single_scene import graph_view


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    baseline=ROOT/'data/scene-tests/20260929T133355258176Z'
    payload=json.loads((baseline/'add_request.json').read_text(encoding='utf-8'))
    source=json.loads((baseline/'source_session.json').read_text(encoding='utf-8'))
    out=ROOT/'data/identity-tests'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True)
    save(out/'add_request.json',payload);save(out/'source_session.json',source)
    print('OUTPUT',out,flush=True)
    class Recorded(LLM):
        calls=0
        candidates=0
        def complete(self,instruction,payload,schema):
            self.calls+=1
            save(out/f'input-{self.calls:02d}.json',payload)
            return super().complete(instruction,payload,schema)
        def on_graph_candidate(self,candidate):
            self.candidates+=1
            save(out/f'draft-{self.candidates:02d}.json',dict(position=self.extraction_chunk_index,draft=candidate.model_dump()))
        def on_conversation_graph(self,graph,audit):
            save(out/'graph.json',graph.model_dump());save(out/'identity_audit.json',audit)
            graph_view(out,graph.model_dump(),source['messages'])
        def extract(self,request):
            try:return super().extract(request)
            except LLMError as exc:
                cause=exc.__cause__
                save(out/'failure.json',dict(error=str(exc),cause_type=type(cause).__name__,
                    validation_error=str(cause) if isinstance(cause,ValueError) else None))
                raise
    llm=Recorded();store=Store(out/'memory.sqlite3')
    with TestClient(create_app(store,llm)) as client:
        start=time.perf_counter();response=client.post('/add',json=payload);elapsed=time.perf_counter()-start
        save(out/'add_response.json',response.json())
        report=dict(model=llm.model,status=response.status_code,seconds=elapsed,calls=llm.calls,
                    message_count=len(payload['messages']),search_tested=False,
                    commit=subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    working_tree_dirty=bool(subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','status','--porcelain'],cwd=ROOT,text=True).strip()))
        if response.status_code==200:
            repeat=client.post('/add',json=payload)
            report['idempotency_passed']=repeat.status_code==200 and repeat.json()['deduplicated']
            graph=json.loads((out/'graph.json').read_text(encoding='utf-8'))
            report['nodes']=len(graph['nodes']);report['edges']=len(graph['edges'])
            report['speakers']=[n for n in graph['nodes'] if n['key'].startswith('speaker:')]
            report['all_quotes_exact']=all(q['text']==payload['messages'][q['message_index']]['content'] for e in graph['edges'] for q in e['evidence'])
        save(out/'report.json',report)
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
        if response.status_code!=200:raise SystemExit(1)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
