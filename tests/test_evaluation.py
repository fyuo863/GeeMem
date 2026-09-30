import pytest
from memory.evaluation import evidence_metrics


def test_macro_micro_hits_and_deduplication():
    cases=[dict(evidence=['a'],v=dict(hit_evidence=['a','a','irrelevant'])),
           dict(evidence=['b','c','d','d'],v=dict(hit_evidence=['b'])),
           dict(evidence=['e'],v=dict(hit_evidence=[])),
           dict(evidence=[],v=dict(hit_evidence=[]))]
    result=evidence_metrics(cases,'v')
    assert result['recall']==pytest.approx((1+1/3+0)/3)
    assert result['micro_recall']==2/5
    assert result['hit_rate']==2/3
    assert result['all_evidence_hit_rate']==1/3
    assert result['gold_evidence']==5 and result['matched_evidence']==2
    assert result['evaluated_questions']==3 and result['skipped_no_evidence']==1


def test_no_evidence_is_undefined_not_perfect():
    result=evidence_metrics([], 'v')
    assert result['recall'] is None and result['micro_recall'] is None
    assert result['hit_rate'] is None and result['all_evidence_hit_rate'] is None
    assert result['evaluated_questions']==0


def test_canonical_source_annotation_formats():
    from memory.evaluation import canonical_evidence
    assert canonical_evidence(['D8:6; D9:17','D30:05','D8:6'])==['D30:5','D8:6','D9:17']
    assert canonical_evidence(['D22:1 D22:2'])==['D22:1','D22:2']
    assert canonical_evidence([])==[]
