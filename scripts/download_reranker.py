"""Download a pinned local cross-encoder snapshot using ONLY root .env network settings."""
import hashlib
import json
from pathlib import Path
import sys
import httpx
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory.config import PROJECT_ROOT, load_settings


def main():
    cfg = load_settings()
    model = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    revision = cfg.get('RAG_RERANK_REVISION', '233902d25c440f23af6f7d6e94d2946bac0bee0a')
    target = Path(cfg.get('RAG_RERANK_PATH', 'data/models/ms-marco-MiniLM-L-6-v2'))
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    target.mkdir(parents=True, exist_ok=True)
    with httpx.Client(trust_env=False, proxy=cfg.get('RAG_DOWNLOAD_PROXY') or cfg.get('LLM_PROXY') or None,
                      timeout=180, follow_redirects=True) as client:
        response = client.get(f'https://huggingface.co/api/models/{model}/revision/{revision}')
        response.raise_for_status()
        meta = response.json()
        sha = meta['sha']
        allowed = {'config.json','config_sentence_transformers.json','modules.json','sentence_bert_config.json',
                   'special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.txt',
                   'model.safetensors','1_Pooling/config.json'}
        files = sorted(x['rfilename'] for x in meta['siblings'] if x['rfilename'] in allowed)
        if not {'model.safetensors','config.json','tokenizer.json'} <= set(files):
            raise ValueError('Incomplete model snapshot')
        hashes = {}
        for name in files:
            dest = target / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_suffix(dest.suffix + '.partial')
            digest = hashlib.sha256()
            with client.stream('GET', f'https://huggingface.co/{model}/resolve/{sha}/{name}') as stream:
                stream.raise_for_status()
                with temp.open('wb') as out:
                    for block in stream.iter_bytes():
                        digest.update(block)
                        out.write(block)
            temp.replace(dest)
            hashes[name] = digest.hexdigest()
            print('Downloaded', name, flush=True)
        (target / 'manifest.json').write_text(json.dumps(dict(model=model, revision=sha, sha256=hashes), indent=2), encoding='utf-8')
    print('Model snapshot ready:', sha)


if __name__ == '__main__':
    main()
