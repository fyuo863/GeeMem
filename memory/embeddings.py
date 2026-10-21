"""Embedding provider implementations used by the retrieval pipeline.

Providers expose the small interface required by ingestion and retrieval:
``documents(texts)`` for stored passages and ``queries(texts)`` for searches.
All configuration is supplied by the project ``.env`` loader.
"""

import hashlib
import json
import logging
import math
import random
import time
from threading import BoundedSemaphore
from pathlib import Path

import numpy as np

from .config import PROJECT_ROOT
from .provider_errors import EmbeddingUnavailable


class LocalEmbedder:
    def __init__(self, cfg):
        from sentence_transformers import SentenceTransformer

        path = Path(cfg.get('RAG_MODEL_PATH', 'data/models/bge-small-en-v1.5'))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not (path / 'manifest.json').is_file():
            raise ValueError('Download the embedding model with scripts/download_rag_model.py first')
        manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
        for name, expected in manifest['sha256'].items():
            file = (path / name).resolve()
            if not file.is_relative_to(path.resolve()):
                raise ValueError('Invalid model manifest path')
            with file.open('rb') as handle:
                if hashlib.file_digest(handle, 'sha256').hexdigest() != expected:
                    raise ValueError('Model snapshot checksum mismatch')
        prefix = cfg.get('RAG_QUERY_PREFIX', 'Represent this sentence for searching relevant passages: ')
        self.identity = json.dumps([manifest, prefix], sort_keys=True)
        self.prefix = prefix
        self.model = SentenceTransformer(str(path), device=cfg.get('RAG_DEVICE', 'cpu'),
                                         local_files_only=True, trust_remote_code=False)

    def documents(self, texts):
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)

    def queries(self, texts):
        return self.documents([self.prefix + t for t in texts])


class HTTPEmbedder:
    """OpenAI-compatible embedding endpoint, configured only in the project .env."""

    def __init__(self, cfg):
        import httpx

        self.url = cfg['RAG_EMBEDDING_API_URL'].rstrip('/')
        self.model_name = cfg.get('RAG_EMBEDDING_API_MODEL', '')
        self.prefix = cfg.get('RAG_QUERY_PREFIX', '')
        self.instruction = cfg.get('RAG_QUERY_INSTRUCTION', '')
        self.timeout = float(cfg.get('RAG_API_TIMEOUT', '120'))
        self.retries = int(cfg.get('RAG_EMBEDDING_API_RETRIES', '3'))
        self.retry_budget = float(cfg.get('RAG_EMBEDDING_RETRY_BUDGET', '90'))
        self.connect_timeout = float(cfg.get('RAG_EMBEDDING_CONNECT_TIMEOUT', '10'))
        self.read_timeout = float(cfg.get('RAG_EMBEDDING_READ_TIMEOUT', '30'))
        self.backoff = float(cfg.get('RAG_EMBEDDING_RETRY_BACKOFF', '1'))
        concurrency = int(cfg.get('RAG_EMBEDDING_CONCURRENCY', '4'))
        if (not 0 <= self.retries <= 10 or concurrency < 1 or
                any(not math.isfinite(x) or x <= 0 for x in
                    (self.timeout, self.retry_budget, self.connect_timeout, self.read_timeout, self.backoff))):
            raise ValueError('Invalid embedding retry/concurrency configuration')
        self.slots = BoundedSemaphore(concurrency)
        self.batch_size = int(cfg.get('RAG_EMBEDDING_BATCH_SIZE',
                                      '10' if self.model_name == 'text-embedding-v4' else '64'))
        dimension = cfg.get('RAG_EMBEDDING_DIMENSIONS', '')
        self.dimensions = int(dimension) if dimension else None
        key = cfg.get('RAG_EMBEDDING_API_KEY', '')
        if self.batch_size < 1 or (self.dimensions is not None and self.dimensions < 1):
            raise ValueError('Invalid embedding batch size or dimensions')
        if self.model_name == 'text-embedding-v4':
            if not key:
                raise ValueError('Set RAG_EMBEDDING_API_KEY in root .env for text-embedding-v4')
            if self.batch_size > 10 or self.dimensions not in (None, 64, 128, 256, 512, 768, 1024, 1536, 2048):
                raise ValueError('Invalid text-embedding-v4 batch size or dimensions')
        self.client = httpx.Client(timeout=self.timeout, trust_env=False,
                                   proxy=cfg.get('RAG_EMBEDDING_API_PROXY') or None,
                                   headers={'Authorization': 'Bearer ' + key} if key else {})
        self.identity = json.dumps(['http-embedding-v1', self.url, self.model_name,
                                    self.prefix, self.instruction], sort_keys=True)
        if self.dimensions is not None:
            self.identity = json.dumps([self.identity, 'dimensions', self.dimensions])

    def documents(self, texts):
        texts = list(texts)
        if not texts:
            raise ValueError('Embedding input must not be empty')
        batches = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            payload = {'input': batch, 'encoding_format': 'float'}
            if self.model_name:
                payload['model'] = self.model_name
            if self.dimensions is not None:
                payload['dimensions'] = self.dimensions
            response = self._request(payload)
            data = response.json().get('data', [])
            if (not isinstance(data, list) or any(not isinstance(item, dict) for item in data)
                    or any(type(item.get('index')) is not int for item in data)
                    or sorted(item['index'] for item in data) != list(range(len(batch)))):
                raise ValueError('Invalid embedding response indexes')
            ordered = sorted(data, key=lambda item: item['index'])
            vectors = self._vectors([item.get('embedding') for item in ordered], len(batch))
            if ((self.dimensions is not None and vectors.shape[1] != self.dimensions)
                    or (batches and vectors.shape[1] != batches[0].shape[1])):
                raise ValueError('Embedding dimension mismatch')
            batches.append(vectors)
        return np.concatenate(batches, axis=0)

    def _request(self, payload):
        import httpx

        deadline = time.monotonic() + self.retry_budget
        for attempt in range(self.retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.slots.acquire(timeout=max(0, remaining)):
                raise EmbeddingUnavailable('Embedding provider temporarily unavailable') from None
            delay = min(10.0, self.backoff * 2 ** attempt + random.uniform(0, self.backoff))
            try:
                # Four individually bounded HTTP phases stay within the remaining budget.
                phase = (deadline - time.monotonic()) / 4
                if phase <= 0:
                    raise EmbeddingUnavailable('Embedding provider temporarily unavailable')
                response = self.client.post(self.url, json=payload, timeout=httpx.Timeout(
                    min(self.timeout, self.read_timeout, phase),
                    connect=min(self.connect_timeout, phase), write=phase, pool=phase))
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in (408, 429, 500, 502, 503, 504):
                    raise EmbeddingUnavailable('Embedding provider rejected request') from None
                try:
                    delay = max(delay, min(10.0, max(0.0, float(exc.response.headers.get('Retry-After', '0')))))
                except ValueError:
                    pass
                exc.response.close()
                reason = 'http_' + str(exc.response.status_code)
            except httpx.TransportError as exc:
                reason = type(exc).__name__
            finally:
                self.slots.release()
            remaining = deadline - time.monotonic()
            if attempt == self.retries or remaining <= delay:
                raise EmbeddingUnavailable('Embedding provider temporarily unavailable') from None
            logging.getLogger(__name__).warning('embedding_retry attempt=%d reason=%s', attempt + 1, reason)
            time.sleep(delay)
        raise EmbeddingUnavailable('Embedding provider temporarily unavailable')

    @staticmethod
    def _vectors(value, count):
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != count or arr.shape[1] == 0 or not np.isfinite(arr).all():
            raise ValueError('Invalid embedding API response')
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError('Zero embedding vector')
        return arr / norms

    def queries(self, texts):
        if self.instruction:
            return self.documents([self.prefix + self.instruction + '\nQuery: ' + t for t in texts])
        return self.documents([self.prefix + t for t in texts])
