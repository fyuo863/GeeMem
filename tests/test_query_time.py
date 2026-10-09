from datetime import datetime, timezone
import pytest
from memory.query_time import resolve_query_time, expanded_query
from memory.aml_api import AMLSearch


def stamp(text): return int(datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp()*1000)


@pytest.mark.parametrize('query,expected',[
 ('Who did I hear last Friday?','2023-03-31'),
 ('What did I buy four weeks ago?','2023-03-08'),
 ('What happened yesterday?','2023-04-04'),
 ('What happened today?','2023-04-05'),
 ('发生在上周五的事','2023-03-31'),
 ('What happened 2020-02-29?','2020-02-29')])
def test_points(query,expected):
    result=resolve_query_time(query,stamp('2023-04-05'))
    assert result.start.isoformat()==expected and result.start==result.end


def test_intervals_and_calendar():
    target=resolve_query_time('during the past three months',stamp('2023-04-05'))
    assert (target.start.isoformat(),target.end.isoformat())==('2023-01-05','2023-04-05')
    target=resolve_query_time('last month',stamp('2024-03-31'))
    assert (target.start.isoformat(),target.end.isoformat())==('2024-02-01','2024-02-29')
    assert resolve_query_time('last Friday',stamp('2023-03-31')).start.isoformat()=='2023-03-24'


def test_no_invented_clock_or_ambiguous_constraint():
    for query in ['last Friday','between yesterday and today','not yesterday','before 2023-03-31','How many weeks ago did I buy it?']:
        assert resolve_query_time(query,None) is None
    assert resolve_query_time('between yesterday and today',stamp('2023-04-05')) is None
    assert resolve_query_time('on 2023-02-30',stamp('2023-04-05')) is None
    assert expanded_query('plain',None)=='plain'


def test_timezone_validation_and_no_boundary_shift():
    assert resolve_query_time('today',stamp('2023-04-05T20:00:00'),'Asia/Shanghai').start.isoformat()=='2023-04-06'
    with pytest.raises(ValueError): AMLSearch(user_id='u',query='q',top_k=5,reference_timezone='invalid/zone')
    with pytest.raises(ValueError): AMLSearch(user_id='u',query='q',top_k=5,reference_time=-1)
