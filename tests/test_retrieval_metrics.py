import pytest
from scripts.test_full_conversation import aggregate, metrics


def test_partial_evidence_and_cutoff():
    m = metrics(["a", "b"], ["x", "b", "y", "z", "w", "a"])
    assert m["hit@1"] == 0
    assert m["hit@5"] == 1
    assert m["recall@5"] == 0.5
    assert m["all_evidence@5"] == 0
    assert m["all_evidence@10"] == 1
    assert m["mrr@10"] == 0.5
    assert m["precision@5"] == 0.2


def test_failed_request_stays_in_denominator():
    rows = [dict(status=200, metrics=metrics(["a"], ["a"])),
            dict(status=502, metrics=metrics(["a"], []))]
    assert aggregate(rows)["metrics"]["hit@5"] == 0.5
    with pytest.raises(ValueError):
        metrics([], [])
