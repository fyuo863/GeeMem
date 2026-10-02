from memory.multi_query import subqueries, merge_routes


def test_both_people_split_without_inventing_names_or_matching_substrings():
    rows=[dict(speaker='Alice'),dict(speaker='Bob')]
    queries=subqueries('What movies have both Alice and Bob seen?',rows)
    assert len(queries)==2
    assert 'Bob' not in queries[0][1] and 'Alice' not in queries[1][1]
    assert not subqueries('What did Alice watch?',rows)
    assert not subqueries('What did Alice and Bobby see?',rows)
    assert not subqueries('What did Alice tell Bob?',rows)


def test_rank_merge_preserves_original_route_and_deduplicates():
    order,scores=merge_routes([0,1,2],[[3,1],[4,1]])
    assert set(order)=={0,1,2,3,4} and len(order)==5
    assert order[0]==1
    assert all(scores[a]>=scores[b] for a,b in zip(order,order[1:]))
