"""Evaluation must not count abstention as failed evidence recall or cache queries."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


scripts = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('multihop_benchmark', scripts/'test_multihop_full.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
sys.path.remove(str(scripts))


def test_abstention_and_missing_annotations_are_not_evidence_failures():
    base = dict(variant='baseline', category='temporal-reasoning', seconds=1, trace={}, hits=[{}])
    cases = [dict(base, gold=['a', 'b'], ranked_messages=['a'], abstention=False),
             dict(base, gold=[], ranked_messages=['x'], abstention=True),
             dict(base, gold=[], ranked_messages=['x'], abstention=False)]
    report = benchmark.summarize(cases, 3)['variants']['baseline']
    assert report['completed'] == 3 and report['scored'] == 1
    assert report['abstention'] == report['unannotated'] == 1
    assert report['metrics']['10']['recall'] == 0.5


def test_document_cache_reuses_only_documents_and_checks_model_identity(tmp_path):
    class Embedder:
        identity = 'model-a'
        calls = 0
        query_calls = 0

        def documents(self, texts):
            self.calls += 1
            return np.array([[1, 0] for _ in texts], dtype=np.float32)

        def queries(self, texts):
            self.query_calls += 1
            return np.array([[0, 1] for _ in texts], dtype=np.float32)

    inner = Embedder()
    cache = benchmark.DocumentCache(inner, tmp_path/'cache.sqlite3')
    assert cache.prefetch(['a', 'a', 'b'], 2) == 2
    assert cache.prefetch(['a'], 2) == 0
    assert inner.calls == 1
    assert cache.documents(['b', 'a']).tolist() == [[1, 0], [1, 0]]
    cache.queries(['a'])
    cache.queries(['a'])
    assert inner.query_calls == 2
    with pytest.raises(ValueError, match='not prefetched'):
        cache.documents(['unknown'])
    inner.identity = 'model-b'
    with pytest.raises(ValueError, match='identity mismatch'):
        benchmark.DocumentCache(inner, tmp_path/'cache.sqlite3')
    cache.db.close()
