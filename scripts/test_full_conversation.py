"""Real API-handler retrieval evaluation; gold never enters /add or /search."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient
from memory.api import create_app
from memory.llm import LLM
from memory.store import Store


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def metrics(gold, retrieved):
    gold = set(gold)
    if not gold:
        raise ValueError("Evidence metrics require nonempty gold")
    result = {}
    for k in (1, 5, 10):
        matches = len(gold & set(retrieved[:k]))
        result.update({f"hit@{k}": float(matches > 0),
                       f"recall@{k}": matches / len(gold),
                       f"precision@{k}": matches / k,
                       f"all_evidence@{k}": float(matches == len(gold))})
    result["mrr@10"] = next((1 / (i+1) for i, e in enumerate(retrieved[:10]) if e in gold), 0.0)
    return result


def aggregate(rows):
    scored = [r for r in rows if r.get("metrics") is not None]
    return dict(questions=len(rows), scored=len(scored),
                successful_requests=sum(r["status"] == 200 for r in rows),
                metrics={key: statistics.mean(r["metrics"][key] for r in scored)
                         for key in scored[0]["metrics"]} if scored else {})


def main():
    public = ROOT / "data/locomo-refined/data/public"
    conv = next(json.loads(line) for line in (public / "conversations.jsonl").read_text(encoding="utf-8").splitlines()
                if json.loads(line)["sample_id"] == "conv-41")
    out = ROOT / "data/full-tests" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True)
    save(out / "source.json", conv)
    store = Store(out / "memory.sqlite3")
    user = "full:locomo:conv-41"
    git = ["git", "-c", f"safe.directory={ROOT.as_posix()}"]
    meta = dict(sample_id="conv-41", model=LLM().model, workers=4,
                commit=subprocess.check_output(git+["rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
                working_tree_dirty=bool(subprocess.check_output(git+["status", "--porcelain"], text=True, cwd=ROOT).strip()),
                source_sha256=hashlib.sha256((public / "conversations.jsonl").read_bytes()).hexdigest(),
                input_mode="Original text only; no images, captions or gold supplied; session clock assumed UTC.")
    save(out / "meta.json", meta)
    print(f"OUTPUT {out}", flush=True)

    class RecordingLLM(LLM):
        def __init__(self, folder):
            super().__init__()
            self.folder = folder
            self.candidate_count = 0
            self.calls = 0

        def complete(self, instruction, payload, schema):
            self.calls += 1
            save(self.folder / "calls.json", {"logical_calls": self.calls})
            return super().complete(instruction, payload, schema)

        def on_graph_candidate(self, candidate):
            self.candidate_count += 1
            save(self.folder / f"candidate-{self.candidate_count}.json", candidate.model_dump())
            save(self.folder / f"candidate-{self.candidate_count}-context.json", {
                "chunk_index": self.extraction_chunk_index,
                "visible_indices": self.extraction_visible_indices})
            save(self.folder / "candidate.json", candidate.model_dump())

        def extract(self, request):
            graph = super().extract(request)
            save(self.folder / "graph.json", graph.model_dump())
            return graph

        def keywords(self, query):
            result = super().keywords(query)
            save(self.folder / "keywords.json", result)
            return result

    def write_session(session):
        folder = out / f"session-{session['session_index']:02d}"
        folder.mkdir()
        timestamp = int(datetime.strptime(session["date_time"], "%I:%M %p on %d %B, %Y").replace(tzinfo=timezone.utc).timestamp()*1000)
        payload = dict(request_id=f"write-{session['session_index']}", user_id=user,
                       session_id=f"session-{session['session_index']}",
                       messages=[dict(role=m["role"], content=m["text"], timestamp=timestamp) for m in session["messages"]])
        save(folder / "request.json", payload)
        attempts = []
        for attempt in range(1, 3):
            attempt_folder = folder / f"attempt-{attempt}"
            attempt_folder.mkdir()
            with TestClient(create_app(store, RecordingLLM(attempt_folder)), raise_server_exceptions=False) as client:
                start = time.perf_counter()
                response = client.post("/add", json=payload)
                elapsed = time.perf_counter() - start
                try:
                    body = response.json()
                except ValueError:
                    body = {"error": "Non-JSON server response"}
                save(attempt_folder / "response.json", body)
                attempts.append(dict(status=response.status_code, seconds=elapsed))
                if response.status_code == 200:
                    repeat = client.post("/add", json=payload)
                    idempotent = repeat.status_code == 200 and repeat.json().get("deduplicated") is True and repeat.json()["message_ids"] == body["message_ids"]
                    break
        result = dict(session_index=session["session_index"], status=response.status_code,
                      attempts=attempts, message_count=len(payload["messages"]),
                      idempotency=idempotent if response.status_code == 200 else False,
                      message_map=dict(zip(body["message_ids"], [m["dia_id"] for m in session["messages"]])) if response.status_code == 200 else {})
        save(folder / "result.json", result)
        return result

    start = time.perf_counter()
    writes = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(write_session, s) for s in conv["sessions"]]):
            result = future.result()
            writes.append(result)
            save(out / "writes.json", sorted(writes, key=lambda r:r["session_index"]))
            print(f"ADD {len(writes)}/{len(conv['sessions'])} session={result['session_index']} status={result['status']}", flush=True)
    add_wall = time.perf_counter()-start
    mapping = {mid:dia for r in writes for mid,dia in r["message_map"].items()}
    originals = {m["dia_id"]:m["text"] for s in conv["sessions"] for m in s["messages"]}
    # Gold is loaded only after the complete ingestion phase.
    questions = [json.loads(line) for line in (public / "questions.jsonl").read_text(encoding="utf-8").splitlines()
                 if json.loads(line)["sample_id"] == "conv-41"]

    def search_question(q):
        folder = out / q["qa_id"].split("#")[1]
        folder.mkdir()
        payload = dict(user_id=user, query=q["question"], limit=10, max_hops=2)
        save(folder / "request.json", payload)
        with TestClient(create_app(store, RecordingLLM(folder)), raise_server_exceptions=False) as client:
            start = time.perf_counter()
            response = client.post("/search", json=payload)
            elapsed = time.perf_counter()-start
        try:
            body = response.json()
        except ValueError:
            body = {"error": "Non-JSON server response"}
        save(folder / "response.json", body)
        hits = body.get("data", []) if response.status_code == 200 else []
        retrieved = [mapping[h["id"]] for h in hits]
        result = dict(qa_id=q["qa_id"], question=q["question"], category=q["category"],
                      multimodal=q["is_multi_modality"], status=response.status_code, seconds=elapsed,
                      gold=q["evidence"], retrieved=retrieved,
                      source_exact_match=all(h["content"] == originals[mapping[h["id"]]] for h in hits) if response.status_code == 200 else None,
                      metrics=metrics(q["evidence"], retrieved) if q["evidence"] else None)
        save(folder / "result.json", result)
        return result

    start = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(search_question, q) for q in questions]):
            results.append(future.result())
            save(out / "searches.json", sorted(results, key=lambda r:r["qa_id"]))
            if len(results) % 10 == 0 or len(results) == len(questions):
                print(f"SEARCH {len(results)}/{len(questions)}", flush=True)
    with store.connect() as db:
        counts = {table:db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in ("messages", "nodes", "edges")}
    report = dict(meta=meta, add_wall_seconds=add_wall, search_wall_seconds=time.perf_counter()-start,
                  add_sessions=len(writes), successful_add_sessions=sum(r["status"]==200 for r in writes),
                  counts=counts, all_idempotent=all(r["idempotency"] for r in writes),
                  all=aggregate(results), text_only=aggregate([r for r in results if not r["multimodal"]]),
                  multimodal=aggregate([r for r in results if r["multimodal"]]),
                  by_category={cat:aggregate([r for r in results if r["category"]==cat]) for cat in sorted({r["category"] for r in results})},
                  source_exact_match=all(r["source_exact_match"] is True for r in results if r["status"]==200),
                  search_mean_seconds=statistics.mean(r["seconds"] for r in results))
    save(out / "report.json", report)
    lines = ["# 完整对话检索评测", "", "LoCoMo conv-41 全部会话；真实 LLM 与 FastAPI /add、/search 处理器（TestClient），非公网部署压测。", "",
             f"写入成功 {report['successful_add_sessions']}/{len(writes)} 个会话；数据库计数：{counts}。", "",
             "| 范围 | 题数 | Hit@1 | Hit@5 | Recall@5 | 全证据命中@5 | MRR@10 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for label,key in [("全部题目","all"),("纯文本","text_only"),("多模态（仅文本输入）","multimodal")]:
        a=report[key]; m=a["metrics"]
        lines.append(f"| {label} | {a['questions']} | {m['hit@1']:.2%} | {m['hit@5']:.2%} | {m['recall@5']:.2%} | {m['all_evidence@5']:.2%} | {m['mrr@10']:.4f} |")
    lines += ["", "Hit@k：前 k 条至少命中一个金标证据的题目比例；Recall@k：每题金标证据召回比例的宏平均；全证据命中：该题全部证据均检出。Precision@k 固定除以 k。失败请求按空结果计零，不从分母排除。无金标题不计分，数目单独记录。", "",
              "写入先完成，再加载金标题；金标答案和证据不传入 LLM。图构建失败最多重试一次，各次结果保留。搜索不额外重试（适配器自带连接重试）。每次独立数据库。", "",
              "限制：只测一份完整对话，不是全数据集成绩或生成答案准确率；多模态题未提供图片。归属判断和跨会话实体稳定键仍依赖模型。4 并发下耗时不能直接和先前串行单场景对比。", "", "完整原始统计见 [report.json](report.json)，逐题排名见 [searches.json](searches.json)，逐会话写入见 [writes.json](writes.json)。"]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
