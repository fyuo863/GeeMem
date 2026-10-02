import numpy as np
import pytest
from memory.fusion_qa import fuse_scores, answer_support


def row(text, speaker, index, session='s'):
    return dict(content=text,speaker=speaker,source_index=index,source_id=str(index),session_id=session)


def test_elliptical_answer_keeps_context_and_nonfinite_rejected():
    rows=[row('Yeah, it was great.', 'B', 1), row('I adopted a cat last week.', 'B', 2)]
    values=fuse_scores(rows,[0,1],[5.,1.],[-8.,6.])
    assert values[0] == 5 and values[1]>1.5
    with pytest.raises(ValueError):fuse_scores(rows,[0],[1.],[np.nan])


def test_reply_credit_is_one_hop_and_never_crosses_session_or_same_speaker():
    rows=[row('Where did you travel?', 'A', 0),row('I went to Paris yesterday.', 'B', 1),row('A third unrelated message here.', 'A', 2)]
    order,values=answer_support('Where did B travel?', rows,[0,2,1],np.array([5.,1.,2.]))
    assert values[1]==4.25 and values[2]==2.
    rows[1]['session_id']='other'
    assert answer_support('Where did B travel?',rows,[0,2,1],np.array([5.,1.,2.]))[1][1]==1.
    rows[1]['session_id']='s';rows[1]['speaker']='A'
    assert answer_support('Where did B travel?',rows,[0,2,1],np.array([5.,1.,2.]))[1][1]==1.
