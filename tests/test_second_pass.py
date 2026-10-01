from memory.second_pass import additional_candidates
from memory.vanilla import bm25


def test_adaptive_global_search_and_no_candidate_duplicates():
    rows = [dict(session_id='one', content=t) for t in ['surfing lessons', 'I can join next month', 'thanks']]
    rows += [dict(session_id='two', content='other topic')]
    # No strong seeds; retain original-query contextual route and find an elliptical reply.
    extra = additional_candidates('When did they agree on surfing?',rows,[0],[2.],bm25,limit=1)
    assert extra == [1]
    assert additional_candidates('A precise fact',rows,[0,1],[9.,2.],bm25) == []
    assert additional_candidates('surfing',rows,list(range(4)),[0.,0.,0.,0.],bm25) == []


def test_windows_never_cross_sessions():
    rows = [dict(session_id='a',content='surfing'), dict(session_id='b',content='tomorrow')]
    assert additional_candidates('When surfing?', rows, [0], [1.], bm25) == []
