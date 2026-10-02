from datetime import datetime, timezone
from memory.soft_recall import query_date, metadata_candidates
from memory.vanilla import bm25


def test_calendar_formats_and_invalid_dates():
    assert query_date('on 3 June, 2023')==(2023,6,3)
    assert query_date('October 3, 2023')==(2023,10,3)
    assert query_date('2023-10-03')==(2023,10,3)
    assert query_date('February 30, 2023')==(None,None,None)
    assert query_date('in 2023')==(2023,None,None)


def test_metadata_route_adds_weak_lexical_date_source_without_filtering():
    stamp=int(datetime(2023,6,3,tzinfo=timezone.utc).timestamp()*1000)
    rows=[dict(content='activity exercise',speaker='Bob',session_timestamp=None),
          dict(content='I help at the shelter.',speaker='Alice',session_timestamp=stamp)]
    route=metadata_candidates('What activity did Alice mention on June 3, 2023?',rows,[0,1],bm25)
    assert route[0]==1
    assert metadata_candidates('Something unrelated',rows,[0,1],bm25)==[]
