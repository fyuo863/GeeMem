"""Transport and persistence regressions for the resumed Full failure."""
import json

import httpx
import numpy as np
import pytest

from memory.llm import LLM, LLMError
from memory.llm_transport import StreamingJSONClient
from memory.models import StrictModel
from memory.selector import MultiLabelJudge
from memory.write_gate import MEMORY_TYPE_CONFIG, MemoryTypeSelector


REAL_CLIENT = httpx.Client


class Chunks(httpx.SyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    def __iter__(self):
        for part in self.chunks:
            if isinstance(part, Exception):
                raise part
            yield part


class Answer(StrictModel):
    value: str


def sse(content, *, finish='stop', done=True, usage=True):
    frames = [{'choices': [{'index': 0, 'delta': {'content': content[:3]}, 'finish_reason': None}]},
              {'choices': [{'index': 0, 'delta': {'content': content[3:]}, 'finish_reason': None}]},
              {'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}]}]
    if usage:
        frames.append({'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': 5,
                                               'total_tokens': 15}})
    body = ': keepalive\n\n' + ''.join('data: '+json.dumps(x, ensure_ascii=False)+'\n\n' for x in frames)
    return (body + ('data: [DONE]\n\n' if done else '')).encode()


def install(monkeypatch, handler):
    monkeypatch.setattr('memory.llm.load_settings', lambda: {
        'LLM_API_KEY': 'synthetic-key', 'LLM_BASE_URL': 'https://unit.invalid/v1'})
    monkeypatch.setattr('memory.llm.httpx.Client',
        lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr('memory.selector.time.sleep', lambda _: None)


def test_fragmented_utf8_structured_output_preserves_schema(monkeypatch, caplog):
    seen = []
    def handler(request):
        payload = json.loads(request.content)
        seen.append(payload)
        data = sse(json.dumps({'value': '杭州'}, ensure_ascii=False))
        return httpx.Response(200, stream=Chunks([data[i:i+7] for i in range(0, len(data), 7)]),
            headers={'content-type': 'text/event-stream', 'x-request-id': 'synthetic-trace'})
    install(monkeypatch, handler)
    result = LLM(max_attempts=1).complete('classify', {'text': 'private-source'}, Answer)
    assert result.value == '杭州'
    assert len(seen) == 1 and seen[0]['stream'] is True
    assert seen[0]['response_format']['json_schema']['strict'] is True
    assert seen[0]['stream_options'] == {'include_usage': True}
    assert 'private-source' not in caplog.text and 'synthetic-key' not in caplog.text


@pytest.mark.parametrize('finish,done', [('stop', False), ('length', True),
                                          ('content_filter', True), (None, True)])
def test_incomplete_completion_never_returns_partial_json(monkeypatch, finish, done):
    install(monkeypatch, lambda _: httpx.Response(200,
        content=sse('{"value":"apparently complete"}', finish=finish, done=done)))
    with pytest.raises(LLMError) as caught:
        LLM(max_attempts=1).complete('classify', {}, Answer)
    assert isinstance(caught.value.__cause__, httpx.RemoteProtocolError)


@pytest.mark.parametrize('event', [
    {'choices': [{'index': 0, 'delta': {'refusal': 'cannot comply'}}]},
    {'choices': None}, {'choices': [None]}, {'choices': [{'delta': None}]},
    {'error': {'message': 'synthetic error'}},
])
def test_invalid_events_and_refusal_fail_closed(monkeypatch, event):
    body = ('data: '+json.dumps(event)+'\n\ndata: [DONE]\n\n').encode()
    install(monkeypatch, lambda _: httpx.Response(200, content=body))
    with pytest.raises(LLMError):
        LLM(max_attempts=1).complete('classify', {}, Answer)


@pytest.mark.parametrize('content', ['not json', '{"wrong":"field"}', '{}'])
def test_final_json_still_requires_original_schema(monkeypatch, content):
    install(monkeypatch, lambda _: httpx.Response(200, content=sse(content)))
    with pytest.raises(LLMError):
        LLM(max_attempts=1).complete('classify', {}, Answer)


@pytest.mark.parametrize('status', [401, 403, 429, 500, 502])
def test_upstream_http_status_is_preserved(monkeypatch, status):
    install(monkeypatch, lambda _: httpx.Response(status, json={'error': {'code': 'synthetic'}}))
    with pytest.raises(LLMError) as caught:
        LLM(max_attempts=1).complete('classify', {}, Answer)
    assert isinstance(caught.value.__cause__, httpx.HTTPStatusError)
    assert caught.value.__cause__.response.status_code == status


def test_budget_and_size_limits(monkeypatch):
    data = sse('{"value":"ok"}')
    for constant, value, error in [('_MAX_WIRE_BYTES', 8, httpx.RemoteProtocolError),
                                   ('_MAX_ELAPSED_SECONDS', -1, httpx.ReadTimeout)]:
        with monkeypatch.context() as patch:
            patch.setattr('memory.llm_transport.'+constant, value)
            with REAL_CLIENT(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=data))) as c:
                with pytest.raises(error):
                    StreamingJSONClient(c).post('https://unit.invalid', json={})


def test_multiline_sse_and_final_event_without_blank_line():
    chunk = {'choices': [{'index': 0, 'delta': {'content': '{"value":"ok"}'}, 'finish_reason': 'stop'}]}
    data = '\n'.join('data: '+line for line in json.dumps(chunk, indent=2).splitlines())
    data += '\n\ndata: [DONE]'
    with REAL_CLIENT(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=data))) as c:
        response = StreamingJSONClient(c).post('https://unit.invalid', json={})
    assert json.loads(response.json()['choices'][0]['message']['content']) == {'value': 'ok'}


def test_transport_retry_keeps_same_prompt_and_three_attempts(monkeypatch):
    payloads = []
    def handler(request):
        payloads.append(json.loads(request.content))
        raise httpx.RemoteProtocolError('synthetic disconnect', request=request)
    install(monkeypatch, handler)
    judge = MultiLabelJudge(MEMORY_TYPE_CONFIG, LLM(max_attempts=1))
    with pytest.raises(LLMError):
        judge.judge([{'content': 'synthetic source'}])
    assert len(payloads) == 3 and payloads[0] == payloads[1] == payloads[2]


def test_partial_stream_leaves_no_receipt_or_source_rows(monkeypatch, tmp_path):
    from memory.aml_api import AMLAdd
    from memory.vanilla import VanillaMemory
    from test_multilabel_judge import output
    calls = []
    result = output({'profile': [0]}).model_dump_json()
    def handler(request):
        calls.append(1)
        return httpx.Response(200, content=sse(result, done=False))
    install(monkeypatch, handler)
    class MustNotEmbed:
        identity = 'stream-failure-isolation'
        def documents(self, texts):
            pytest.fail('Partial classification must never reach embedding')
    backend = VanillaMemory({'RAG_MEMORY_DB': str(tmp_path/'isolated.sqlite3')}, MustNotEmbed(),
                            write_selector=MemoryTypeSelector(LLM(max_attempts=1)))
    request = AMLAdd(user_id='u', session_id='s', request_id='r',
                     messages=[{'role': 'user', 'content': 'synthetic source'}])
    with pytest.raises(LLMError):
        backend.add(request)
    assert len(calls) == 3
    with backend.connect() as db:
        for table in ('rag_requests', 'rag_sources', 'rag_memories', 'rag_write_decisions',
                      'rag_memory_routes', 'rag_memory_queue'):
            assert db.execute('SELECT count(*) FROM '+table).fetchone()[0] == 0



def test_transport_preserves_usage_and_provider_request_id():
    body = sse('{"value":"ok"}')
    with REAL_CLIENT(transport=httpx.MockTransport(lambda _: httpx.Response(
            200, content=body, headers={'content-type':'text/event-stream',
                                      'x-request-id':'synthetic-trace'}))) as raw:
        response = StreamingJSONClient(raw).post('https://unit.invalid/v1/chat/completions',
                                                 json={'model':'synthetic'})
    assert response.json()['usage'] == {'prompt_tokens':10, 'completion_tokens':5,
                                       'total_tokens':15}
    assert response.json()['choices'][0]['message']['content'] == '{"value":"ok"}'
    assert response.headers['x-request-id'] == 'synthetic-trace'
    assert response.headers['content-type'].startswith('application/json')
