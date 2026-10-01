import importlib.util
import math
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('full_report',Path(__file__).resolve().parents[1]/'scripts/report_abc_full.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


def case(gold,before,after,missing=None):
    return dict(evidence=gold,missing_evidence=missing or [],
                baseline=dict(ranked_evidence=before),ABC=dict(ranked_evidence=after))


def test_duplicate_evidence_and_exclusions_have_known_metrics():
    cases=[case(['a','b'],['x','a','a','b'],['a','b']),case(['c'],['c','x'],['x']),
           case([],['x'],['x']),case(['missing'],['x'],['x'],['missing'])]
    m=report.metric(cases,'baseline',3)
    assert m['evaluated_questions']==2 and m['excluded_no_gold']==1 and m['excluded_missing_gold']==1
    assert m['recall']==.75 and m['micro_recall']==pytest.approx(2/3)
    assert m['hit_rate']==1 and m['all_evidence_hit_rate']==.5
    assert m['mrr']==.75 and m['precision']==pytest.approx(1/3)
    assert m['ndcg']==pytest.approx((1+(1/math.log2(3))/(1+1/math.log2(3)))/2)
    p=report.paired(cases,3)
    assert p==dict(hit_wins=0,hit_losses=1,evidence_wins=1,evidence_losses=1,evidence_ties=0)


def test_empty_gold_never_counts_as_success():
    result=report.metric([case([],['x'],[])],'ABC',10)
    assert result['evaluated_questions']==0 and result['hit_rate'] is None
    assert result['precision'] is None and result['ndcg'] is None
