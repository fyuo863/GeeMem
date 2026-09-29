"""Download public text evaluation datasets; raw data stays git-ignored."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import httpx

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def get_json(url):
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()


def hf_files(repo, revision):
    url = f"https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true&limit=1000"
    files = []
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        while url:
            response = client.get(url)
            response.raise_for_status()
            files.extend(item for item in response.json() if item["type"] == "file")
            url = response.links.get("next", {}).get("url")
    return files


def plan():
    jobs = []
    repo = "mem-eval-suite/LoCoMo_refined"
    revision = get_json(f"https://api.github.com/repos/{repo}/commits/main")["sha"]
    for name in ["README.md", "LICENSE.txt", "NOTICE", "data/public/manifest.json",
                 "data/public/conversations.jsonl", "data/public/questions.jsonl",
                 "data/raw/locomo_refined.json"]:
        jobs.append(dict(dataset="locomo-refined", file=name,
                         url=f"https://raw.githubusercontent.com/{repo}/{revision}/{name}",
                         revision=revision, license="CC-BY-NC-4.0"))
    specs = [
        ("longmemeval", "xiaowu0162/longmemeval-cleaned", "MIT",
         lambda p: p in {"README.md", "longmemeval_oracle.json", "longmemeval_s_cleaned.json"}),
        ("personamem-v2", "bowen-upenn/PersonaMem-v2", "CC-BY-4.0",
         lambda p: p in {"README.md", "column_descriptions.md", "benchmark/text/benchmark.csv"}
         or p.startswith("data/chat_history_32k/")),
        ("beam", "Mohammadta/BEAM", "CC-BY-SA-4.0",
         lambda p: p == "README.md" or p.endswith(".parquet")),
        ("cl-bench", "tencent/CL-bench", "CL-Bench custom evaluation-only license",
         lambda p: p in {"README.md", "LICENSE.txt", "CL-bench.jsonl"}),
    ]
    for folder, repo, license_name, include in specs:
        meta = get_json(f"https://huggingface.co/api/datasets/{repo}")
        revision = meta["sha"]
        selected = [item for item in hf_files(repo, revision) if include(item["path"])]
        print(f"PLAN {folder}: {len(selected)} files, {sum(x['size'] for x in selected):,} bytes", flush=True)
        for item in selected:
            jobs.append(dict(dataset=folder, file=item["path"], expected_bytes=item["size"],
                             url=f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{item['path']}?download=true",
                             revision=revision, license=license_name))
    return jobs


def download(job):
    target = DATA / job["dataset"] / job["file"]
    target.parent.mkdir(parents=True, exist_ok=True)
    marker = target.with_name(target.name + ".download.json")
    if target.exists() and marker.exists():
        old = json.loads(marker.read_text(encoding="utf-8"))
        with target.open("rb") as existing:
            digest = hashlib.file_digest(existing, "sha256").hexdigest()
        if old.get("url") == job["url"] and old.get("sha256") == digest:
            return old
    temporary = target.with_name(target.name + ".part")
    for attempt in range(3):
        try:
            digest = hashlib.sha256()
            size = 0
            with httpx.Client(timeout=httpx.Timeout(90, connect=30), follow_redirects=True) as client:
                with client.stream("GET", job["url"]) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as output:
                        for chunk in response.iter_bytes(1024 * 1024):
                            output.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
            if job.get("expected_bytes") is not None and size != job["expected_bytes"]:
                raise ValueError("Downloaded size does not match source metadata")
            temporary.replace(target)
            result = dict(job, bytes=size, sha256=digest.hexdigest(), status="downloaded")
            marker.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return result
        except (httpx.HTTPError, OSError, ValueError) as exc:
            if attempt == 2:
                return dict(job, status="failed", error=type(exc).__name__,
                            http_status=getattr(getattr(exc, "response", None), "status_code", None))
            time.sleep(1 + attempt)


def main():
    DATA.mkdir(exist_ok=True)
    jobs = plan()
    (DATA / "download_plan.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            if len(results) % 50 == 0 or result["status"] == "failed":
                print(f"PROGRESS {len(results)}/{len(jobs)}; {result['dataset']}/{result['file']} {result['status']}", flush=True)
    results.sort(key=lambda r: (r["dataset"], r["file"]))
    (DATA / "download_manifest.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for dataset in sorted({r["dataset"] for r in results}):
        entries = [r for r in results if r["dataset"] == dataset]
        print(dataset, "files=", len(entries), "failed=", sum(r["status"] == "failed" for r in entries),
              "bytes=", sum(r.get("bytes", 0) for r in entries), flush=True)
    raise SystemExit(1 if any(r["status"] == "failed" for r in results) else 0)


if __name__ == "__main__":
    main()
