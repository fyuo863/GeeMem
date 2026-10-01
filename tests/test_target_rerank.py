import numpy as np
import pytest
from memory.target_rerank import combine_target_scores, select_targets, target_penalty


def test_elliptical_reply_keeps_context_and_bad_scores_fail():
    assert combine_target_scores([8, 1], [-9, 9]).tolist() == [7.5, 1.5]
    with pytest.raises(ValueError): combine_target_scores([1], [float('nan')])
    with pytest.raises(ValueError): combine_target_scores([1,2], [1])


def test_factual_thanks_not_penalized_and_questions_depend_on_query():
    assert target_penalty('Where did Alice move?', {'content': 'Thanks, Alice!'}) > 0
    assert target_penalty('Where did Alice move?', {'content': 'Thanks, I moved to Paris.'}) == 0
    assert target_penalty('Where did Alice move?', {'content': 'Where did you move?'}) > 0
    assert target_penalty('What did Alice ask?', {'content': 'Where did you move?'}) == 0
    rows = [{'content': 'Where did you move?'}, {'content':'Paris.'}]
    order, scores = select_targets('Where did Alice move?', rows, [0,1], np.array([3.,2.8]))
    assert order == [1,0] and len(order) == 2
