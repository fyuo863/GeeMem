from types import SimpleNamespace
import pytest
from memory.state_evidence import StateEvidenceSelector, StateSelection


class Model:
    def __init__(self, selected=None, fail=False):
        self.selected = selected or []
        self.fail = fail
        self.calls = []

    def complete(self, instruction, payload, schema, timeout):
        self.calls.append((payload, timeout))
        if self.fail:
            raise RuntimeError('secret provider response')
        return StateSelection(selected=self.selected)


def run(model, query='Where does Mei live now?', hits=None, mode='on'):
    hits = hits if hits is not None else [
        dict(id='old', content='Mei lives in Paris.', score=.9),
        dict(id='new', content='Mei moved to Rome.', score=.8)]
    trace = {}
    result = StateEvidenceSelector({'RAG_STATE_EVIDENCE_MODE': mode}, model).select(
        SimpleNamespace(query=query, reference_time=123), hits, trace)
    return result, trace


def test_promote_without_rewriting_or_dropping():
    m = Model([dict(index=1, quote='moved to Rome')])
    result, trace = run(m)
    assert [h['id'] for h in result] == ['new', 'old']
    assert result[0]['content'] == 'Mei moved to Rome.'
    assert result[0]['score'] >= result[1]['score']
    assert m.calls[0][1] == 20
    assert m.calls[0][0]['reference_time'] == 123
    assert trace['state_evidence']['calls'] == 1


@pytest.mark.parametrize('selection', [[dict(index=9, quote='Rome')], [dict(index=1, quote='invented')]])
def test_invalid_selection_fails_open(selection):
    result, trace = run(Model(selection))
    assert result[0]['id'] == 'old'
    assert trace['state_evidence']['status'] == 'fallback'


def test_failure_log_does_not_include_data(caplog):
    with caplog.at_level('INFO'):
        result, trace = run(Model(fail=True))
    assert result[0]['id'] == 'old'
    assert 'secret' not in caplog.text and 'Mei' not in caplog.text
    assert trace['state_evidence']['status'] == 'fallback'


@pytest.mark.parametrize('query,mode', [('Who is Mei?', 'on'), ('now', 'off'), ('now '+ 'x'*1500, 'on')])
def test_no_unnecessary_calls(query, mode):
    m = Model()
    run(m, query=query, mode=mode)
    assert not m.calls


def test_bounds_and_no_truncated_evidence():
    m = Model()
    hits = [dict(id=str(i), content='x'*601 if i == 0 else 'safe evidence', score=1) for i in range(30)]
    run(m, hits=hits)
    assert len(m.calls[0][0]['candidates']) == 15
    assert all(c['index'] != 0 for c in m.calls[0][0]['candidates'])


def test_abstention_preserves_order():
    result, trace = run(Model())
    assert result[0]['id'] == 'old'
    assert trace['state_evidence']['status'] == 'abstained'


def test_context_cannot_outrank_current_primary():
    result, _ = run(Model([dict(index=0, quote='lives in Paris', role='context'),
                           dict(index=1, quote='moved to Rome', role='primary')]))
    assert [h['id'] for h in result] == ['new', 'old']


def test_backend_preserves_payload_protocol_and_topk():
    from memory.aml_api import AMLSearch
    from memory.vanilla import VanillaMemory
    backend = object.__new__(VanillaMemory)
    backend.state_evidence = StateEvidenceSelector({'RAG_STATE_EVIDENCE_MODE': 'on'}, Model())
    def search(payload, trace):
        assert payload.top_k == 16
        assert payload.model_copy(update={'query': 'subquery'}).user_id == 'u'
        return {'data': [dict(id=str(i), content='safe content', score=1) for i in range(16)]}
    backend.search_service = SimpleNamespace(search=search)
    request = AMLSearch(user_id='u', query='Where is Mei now?', top_k=2)
    assert len(backend.search(request)['data']) == 2
    assert request.top_k == 2
