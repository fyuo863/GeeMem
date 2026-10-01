"""Local cross-encoder scoring; loads only a verified on-disk snapshot."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .config import PROJECT_ROOT


class LocalReranker:
    def __init__(self,cfg):
        from sentence_transformers import CrossEncoder
        path=Path(cfg.get('RAG_RERANK_PATH','data/models/ms-marco-MiniLM-L-6-v2'))
        if not path.is_absolute():path=PROJECT_ROOT/path
        manifest=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
        for name,digest in manifest['sha256'].items():
            source=(path/name).resolve()
            if not source.is_relative_to(path.resolve()):raise ValueError('Invalid reranker manifest')
            with source.open('rb') as handle:
                if hashlib.file_digest(handle,'sha256').hexdigest()!=digest:raise ValueError('Reranker checksum mismatch')
        self.identity=manifest
        self.batch=int(cfg.get('RAG_RERANK_BATCH_SIZE','32'))
        self.max_length=int(cfg.get('RAG_RERANK_MAX_LENGTH','512'))
        if self.batch<1 or not 32<=self.max_length<=512:raise ValueError('Invalid reranker limits')
        self.model=CrossEncoder(str(path),device=cfg.get('RAG_RERANK_DEVICE','cpu'),
            max_length=self.max_length,local_files_only=True,trust_remote_code=False)

    def score(self,query,documents):
        return np.asarray(self.model.predict([(query,d) for d in documents],batch_size=self.batch,
                          show_progress_bar=False,convert_to_numpy=True),dtype=float).reshape(-1)


def context_support_scores(rows, candidates, reranked, penalty):
    """Credit a context neighbor without treating context relevance as target proof.

    Rows must already be scoped to one user and ordered by insertion position.
    A single propagation step is used; transferred scores never propagate again.
    """
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError('Invalid context support penalty')
    original = dict(zip(candidates, reranked))
    boosted = dict(original)
    for position, score in original.items():
        for neighbor in (position-1, position+1):
            if 0 <= neighbor < len(rows) and rows[neighbor]['session_id'] == rows[position]['session_id']:
                boosted[neighbor] = max(boosted.get(neighbor, -float('inf')), score-penalty)
    order = sorted(boosted, key=lambda i: (-boosted[i], -original.get(i, -float('inf')), i))
    return order, boosted
