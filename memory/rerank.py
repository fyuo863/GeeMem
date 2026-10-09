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


class ONNXReranker:
    """Local single-logit cross encoder, using the existing ONNX snapshot."""
    def __init__(self, cfg):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        path = Path(cfg['RAG_RERANK_PATH'])
        if not path.is_absolute(): path = PROJECT_ROOT / path
        manifest_path = Path(cfg.get('RAG_RERANK_MANIFEST', str(path/'manifest.json')))
        if not manifest_path.is_absolute(): manifest_path = PROJECT_ROOT / manifest_path
        manifest = json.loads(manifest_path.read_text(encoding='utf8'))
        required = {'onnx/model.onnx', 'config.json', 'tokenizer.json', 'tokenizer_config.json'}
        if not required <= set(manifest['sha256']):
            raise ValueError('Incomplete ONNX reranker manifest')
        for name, digest in manifest['sha256'].items():
            source = (path/name).resolve()
            if not source.is_relative_to(path.resolve()): raise ValueError('Invalid reranker manifest')
            with source.open('rb') as handle:
                if hashlib.file_digest(handle,'sha256').hexdigest()!=digest:
                    raise ValueError('Reranker checksum mismatch')
        self.identity = manifest
        self.batch = int(cfg.get('RAG_RERANK_BATCH_SIZE','8'))
        self.max_length = int(cfg.get('RAG_RERANK_MAX_LENGTH','512'))
        threads = int(cfg.get('RAG_RERANK_THREADS','4'))
        if self.batch < 1 or not 32 <= self.max_length <= 512 or threads < 1:
            raise ValueError('Invalid ONNX reranker limits')
        self.tokenizer = AutoTokenizer.from_pretrained(str(path),local_files_only=True,trust_remote_code=False)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path/'onnx/model.onnx'),sess_options=options,
                                           providers=['CPUExecutionProvider'])
        self.inputs = self.session.get_inputs()
        self.output = self.session.get_outputs()[0].name

    def score(self, query, documents):
        if not documents: return np.empty(0,dtype=float)
        scores = []
        for start in range(0,len(documents),self.batch):
            docs = documents[start:start+self.batch]
            encoded = self.tokenizer([query]*len(docs),list(docs),padding=True,
                                     truncation=True,max_length=self.max_length,return_tensors='np')
            feed = {}
            for spec in self.inputs:
                value = encoded.get(spec.name)
                if value is None and spec.name=='token_type_ids':
                    value = np.zeros_like(encoded['input_ids'])
                if value is None: raise ValueError('Unsupported ONNX model input')
                dtype = {'tensor(int64)':np.int64, 'tensor(int32)':np.int32}.get(spec.type)
                if dtype is None: raise ValueError('Unsupported ONNX input dtype')
                feed[spec.name] = np.asarray(value,dtype=dtype)
            logits = np.asarray(self.session.run([self.output],feed)[0],dtype=float)
            if logits.shape not in ((len(docs),),(len(docs),1)) or not np.isfinite(logits).all():
                raise ValueError('Invalid ONNX reranker scores')
            scores.extend(logits.reshape(-1))
        return np.asarray(scores,dtype=float)


class HTTPReranker:
    """vLLM or DashScope native rerank, configured exclusively in .env.

    One pool per request: native scores must never be merged across batches.
    """
    def __init__(self, cfg):
        import httpx
        self.url = cfg['RAG_RERANK_API_URL'].rstrip('/')
        self.model_name = cfg.get('RAG_RERANK_API_MODEL', '')
        self.protocol = cfg.get('RAG_RERANK_API_PROTOCOL', 'vllm')
        if self.protocol not in ('vllm', 'dashscope'): raise ValueError('Invalid rerank API protocol')
        self.relative_scores = self.protocol == 'dashscope'
        self.max_documents = 500 if self.relative_scores else int(cfg.get('RAG_RERANK_API_MAX_DOCUMENTS', '1000'))
        self.timeout = float(cfg.get('RAG_RERANK_API_TIMEOUT', cfg.get('RAG_API_TIMEOUT', '60')))
        self.retries = int(cfg.get('RAG_RERANK_API_RETRIES', '1'))
        if not 0 <= self.retries <= 2: raise ValueError('Invalid rerank retry budget')
        key = cfg.get('RAG_RERANK_API_KEY', '')
        self.client = httpx.Client(timeout=self.timeout, trust_env=False,
            proxy=cfg.get('RAG_RERANK_API_PROXY') or None,
            headers={'Authorization': 'Bearer '+key} if key else {})
        self.identity = json.dumps(['http-reranker-v1', self.url, self.model_name], sort_keys=True)

    def score(self, query, documents):
        import httpx
        import time
        if not documents: return np.empty(0,dtype=float)
        if len(documents)>self.max_documents: raise ValueError('Rerank pool exceeds single-request limit')
        payload = {'query': query, 'documents': list(documents)}
        if self.model_name: payload['model'] = self.model_name
        if self.protocol == 'dashscope':
            payload = dict(model=self.model_name,input=dict(query=query,documents=list(documents)),
                           parameters=dict(top_n=len(documents)))
        for attempt in range(self.retries+1):
            try:
                response = self.client.post(self.url, json=payload)
                if response.status_code in (429,500,502,503,504) and attempt<self.retries:
                    try: delay=float(response.headers.get('Retry-After','1'))
                    except ValueError: delay=1
                    if not 0<=delay<=5: response.raise_for_status()
                    time.sleep(delay);continue
                response.raise_for_status()
                break
            except (httpx.TimeoutException,httpx.NetworkError):
                if attempt==self.retries: raise
        data=response.json()
        if data.get('code'): raise ValueError('Reranker API returned an error')
        rows = (data.get('output',{}) if self.relative_scores else data).get('results', [])
        scores = np.full(len(documents), -np.inf, dtype=float)
        seen=set()
        for row in rows:
            index = row['index']
            if type(index) is not int or not 0<=index<len(scores) or index in seen:
                raise ValueError('Invalid or duplicate reranker index')
            seen.add(index)
            scores[index] = float(row['relevance_score'])
        if not np.isfinite(scores).all(): raise ValueError('Invalid reranker API response')
        if self.relative_scores and (np.any(scores<0) or np.any(scores>1)):
            raise ValueError('Invalid reranker probability')
        return scores


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
