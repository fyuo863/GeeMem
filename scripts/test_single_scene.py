"""Run one real, source-only LoCoMo session through /add and /search."""
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import sys
import time
import subprocess
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient
from memory.api import create_app
from memory.llm import LLM, LLMError
from memory.store import Store


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def graph_view(out, graph, messages):
    payload = json.dumps({"graph": graph, "messages": messages}, ensure_ascii=False).replace("<", "\\u003c")
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>单场景记忆图</title>
<style>body{margin:0;font:15px system-ui;background:#f6f7f9;color:#182736}header{padding:20px 28px;background:white;border-bottom:1px solid #ddd}h1{font-size:24px;margin:0 0 8px}main{display:grid;grid-template-columns:2fr 1fr;height:78vh}svg{width:100%;height:100%}aside{overflow:auto;background:white;padding:22px;border-left:1px solid #ddd}.edge{fill:none;stroke:#9aa7b5;stroke-width:1.5;cursor:pointer}.edge:hover{stroke:#c25319;stroke-width:4}.node{cursor:pointer}.node circle{stroke:white;stroke-width:2}text{font-size:12px;fill:#182736;pointer-events:none}.evidence{margin:14px 0;padding:12px;background:#f4f6f9;line-height:1.55}small{color:#586773}button{background:#e4eaf0;border:0;padding:8px;cursor:pointer}pre{white-space:pre-wrap}</style>
<header><h1>单场景无向记忆图 · John 与 Maria</h1><small>LoCoMo-Refined / conv-41 / session-20 · 点击节点或连边查看关联原文。连边表示无向关联，端点顺序不表示施受方向。</small></header>
<main><svg viewBox="0 0 1100 850"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#718294"/></marker></defs><g id="edges"></g><g id="nodes"></g></svg><aside id="detail"></aside></main>
<script>const DATA=__PAYLOAD__;
const ns='http://www.w3.org/2000/svg',g=DATA.graph,ms=DATA.messages;
const nodes=g.nodes.map((n,i)=>({...n,x:550+300*Math.cos(i*2*Math.PI/g.nodes.length),y:425+300*Math.sin(i*2*Math.PI/g.nodes.length)}));
const lookup=Object.fromEntries(nodes.map(n=>[n.key,n]));
for(let step=0;step<500;step++){let forces=nodes.map(()=>({x:0,y:0}));for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){let dx=nodes[i].x-nodes[j].x,dy=nodes[i].y-nodes[j].y,d=Math.max(20,Math.hypot(dx,dy)),f=12000/(d*d);forces[i].x+=dx/d*f;forces[i].y+=dy/d*f;forces[j].x-=dx/d*f;forces[j].y-=dy/d*f;}for(const e of g.edges){let a=lookup[e.source],b=lookup[e.target],dx=b.x-a.x,dy=b.y-a.y,d=Math.max(1,Math.hypot(dx,dy)),f=(d-200)*.008;let i=nodes.indexOf(a),j=nodes.indexOf(b);forces[i].x+=dx/d*f;forces[i].y+=dy/d*f;forces[j].x-=dx/d*f;forces[j].y-=dy/d*f;}nodes.forEach((n,i)=>{n.x=Math.max(95,Math.min(1000,n.x+forces[i].x));n.y=Math.max(50,Math.min(800,n.y+forces[i].y));});}
function element(tag,attrs){let e=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);return e;}
function details(title,indices,extra=''){let box=document.getElementById('detail');box.replaceChildren();let h=document.createElement('h2');h.textContent=title;box.append(h);let p=document.createElement('p');p.textContent=extra;box.append(p);for(const i of [...new Set(indices)]){let d=document.createElement('div');d.className='evidence';let small=document.createElement('small');small.textContent=ms[i].dia_id+' · '+ms[i].speaker;let t=document.createElement('p');t.textContent=ms[i].text;d.append(small,t);box.append(d);}}
g.edges.forEach((e,i)=>{const a=lookup[e.source],b=lookup[e.target],dx=b.x-a.x,dy=b.y-a.y,d=Math.max(1,Math.hypot(dx,dy));let x1=a.x+dx/d*22,y1=a.y+dy/d*22,x2=b.x-dx/d*26,y2=b.y-dy/d*26;let path=element('path',{d:`M ${x1} ${y1} Q ${(x1+x2)/2-dy*.1} ${(y1+y2)/2+dx*.1} ${x2} ${y2}`,class:'edge'});let title=element('title',{});title.textContent=a.name+' — '+e.relation+' — '+b.name;path.append(title);path.onclick=()=>details(title.textContent,e.message_indices,'关系证据');document.getElementById('edges').append(path);let label=element('text',{x:(x1+x2)/2-dy*.05,y:(y1+y2)/2+dx*.05-5,'text-anchor':'middle'});label.textContent=e.relation;document.getElementById('edges').append(label);});
for(const n of nodes){let el=element('g',{class:'node',transform:`translate(${n.x},${n.y})`});el.append(element('circle',{r:21,fill:['person','user','assistant'].includes(n.kind.toLowerCase())?'#bc672c':'#4c8295'}));let text=element('text',{y:39,'text-anchor':'middle'});text.textContent=n.name;el.append(text);el.onclick=()=>details(n.name,n.message_indices,n.kind+' · '+n.key);document.getElementById('nodes').append(el);}
details('构图结果',[],nodes.length+' 个节点，'+g.edges.length+' 条无向关系。选择一个节点或连边查看证据。');
</script></html>'''
    (out / "graph.html").write_text(page.replace("__PAYLOAD__", payload), encoding="utf-8")


def main():
    public = ROOT / "data/locomo-refined/data/public"
    conversations = [json.loads(line) for line in (public / "conversations.jsonl").read_text(encoding="utf-8").splitlines()]
    conversation = next(c for c in conversations if c["sample_id"] == "conv-41")
    session = next(s for s in conversation["sessions"] if s["session_index"] == 20)
    source = session["messages"]
    assert len(source) <= 200 and not any(m["has_multimodal_context"] for m in source)
    # The source specifies a local session clock without timezone. UTC is only
    # a deterministic adapter convention, not a claim about the actual timezone.
    dt = datetime.strptime(session["date_time"], "%I:%M %p on %d %B, %Y").replace(tzinfo=timezone.utc)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = ROOT / "data/scene-tests" / stamp
    out.mkdir(parents=True)
    user_id = "scene:locomo:conv-41"
    payload = dict(request_id="scene-write-001", user_id=user_id, session_id="session-20",
                   messages=[dict(role=m["role"], content=m["text"], timestamp=int(dt.timestamp()*1000)) for m in source])
    save(out / "source_session.json", session)
    save(out / "add_request.json", payload)
    report = dict(dataset="LoCoMo-Refined", sample_id="conv-41", session_index=20,
                  graph_mode="undirected", extraction_llm_calls=1,
                  message_count=len(source), output=str(out), status="started",
                  timestamp_note="Source timezone unspecified; adapter assumes UTC; same timestamp for all session turns.")
    save(out / "report.json", report)
    print("OUTPUT", out, flush=True)

    class RecordingLLM(LLM):
        def on_graph_candidate(self, candidate):
            save(out / "candidate_graph.json", candidate.model_dump())

        def extract(self, request):
            try:
                graph = super().extract(request)
            except LLMError as exc:
                cause = exc.__cause__
                response = getattr(cause, "response", None)
                detail = {"error_type": type(cause).__name__ if cause else type(exc).__name__,
                          "http_status": response.status_code if response is not None else None}
                if response is not None:
                    try:
                        error = response.json().get("error", {})
                        detail["provider_error_code"] = error.get("code")
                        detail["provider_error_type"] = error.get("type")
                    except (ValueError, AttributeError):
                        pass
                if hasattr(cause, "errors"):
                    detail["validation_errors"] = cause.errors(include_input=False, include_url=False)
                save(out / "llm_error.json", detail)
                print("LLM_ERROR", json.dumps(detail), flush=True)
                raise
            save(out / "graph.json", graph.model_dump())
            graph_view(out, graph.model_dump(), source)
            return graph

    llm = RecordingLLM()
    report["llm_model"] = llm.model
    report["commit"] = subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report["input_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    store = Store(out / "memory.sqlite3")
    with TestClient(create_app(store, llm)) as client:
        started = time.perf_counter()
        response = client.post("/add", json=payload)
        report["add_seconds"] = round(time.perf_counter()-started, 3)
        report["add_status"] = response.status_code
        save(out / "add_response.json", response.json())
        if response.status_code != 200:
            report["status"] = "add_failed"
            save(out / "report.json", report)
            print(json.dumps(report), flush=True)
            raise SystemExit(1)
        added = response.json()
        report.update(nodes=added["nodes"], edges=added["edges"])
        graph = json.loads((out / "graph.json").read_text(encoding="utf-8"))
        names = {n["key"]: n["name"] for n in graph["nodes"]}
        report["question_edges"] = [dict(source=names[e["source"]], target=names[e["target"]],
                                          relation=e["relation"], evidence=[source[i]["dia_id"] for i in e["message_indices"]])
                                    for e in graph["edges"] if e["relation"] == "询问"]
        repeated = client.post("/add", json=payload)
        report["idempotency_passed"] = repeated.status_code == 200 and repeated.json()["message_ids"] == added["message_ids"] and repeated.json()["deduplicated"]
        # Gold labels are loaded only after construction and never passed to LLM.
        questions = [json.loads(line) for line in (public / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
        question = next(q for q in questions if q["qa_id"] == "conv-41#q0096")
        search = dict(user_id=user_id, query=question["question"], limit=5, max_hops=2)
        save(out / "search_request.json", search)
        started = time.perf_counter()
        result = client.post("/search", json=search)
        report["search_seconds"] = round(time.perf_counter()-started, 3)
        report["search_status"] = result.status_code
        save(out / "search_response.json", result.json())
        report["question"] = question["question"]
        report["gold_evidence"] = question["evidence"]
        report["gold_answer_for_evaluation_only"] = question["answer"]
        if result.status_code == 200:
            mapping = dict(zip(added["message_ids"], source))
            hits = result.json()["data"]
            retrieved = [mapping[h["id"]]["dia_id"] for h in hits]
            report["retrieved_evidence"] = retrieved
            report["evidence_recall_at_5"] = len(set(retrieved)&set(question["evidence"]))/len(set(question["evidence"]))
            report["source_exact_match"] = all(h["content"] == mapping[h["id"]]["text"] for h in hits)
        report["status"] = "completed" if result.status_code == 200 else "search_failed"
        report["limitations"] = "Single 18-message session and one in-session QA only; not a benchmark score or multi-session quality estimate."
        save(out / "report.json", report)
        (out / "report.md").write_text("# 单场景构图测试\n\n```json\n"+json.dumps(report,ensure_ascii=False,indent=2)+"\n```\n",encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
