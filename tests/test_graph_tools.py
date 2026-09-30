import json
import httpx
import pytest
from memory import config
from memory.models import AddRequest
from memory.conversation import Conversation, TurnDraft
from memory.graph_tools import GraphAction, GraphQueryAction, GraphQuery, query_graph
from memory.llm import LLM, LLMError


def request():
    return AddRequest(request_id='r',user_id='u',session_id='s',messages=[
        dict(role='user',content='I visited a shelter.',timestamp=0),
        dict(role='assistant',content='Interesting.',timestamp=1),
        dict(role='user',content='More talk.',timestamp=2),
        dict(role='assistant',content='Did you return there?',timestamp=3)])


def state():
    return Conversation(request()).accept(TurnDraft(entities=[dict(ref='s',name='shelter',
        kind='place',contacts=['A'])],relations=[dict(source='A',target='s',relation='visited')]),0)


def test_lookup_nodes_edges_old_evidence_and_pagination_are_read_only():
    s=state();before=s.graph().model_dump()
    result=query_graph(s,GraphQuery(tool='search_nodes',query='shelter',include_evidence=True))
    n=result['items'][0]
    assert n['owner_key'] is None and n['contact_keys']==[s.keys['A']]
    assert n['evidence'][0]['text']=='I visited a shelter.'
    assert query_graph(s,GraphQuery(tool='get_node',key='A'))['items'][0]['speaker_id']=='A'
    edges=query_graph(s,GraphQuery(tool='get_neighbors',key=n['key'],include_evidence=True))
    assert edges['items'][0]['relation']=='visited'
    assert query_graph(s,GraphQuery(tool='search_edges',query='VISIT',key='A'))['total']==1
    first=query_graph(s,GraphQuery(tool='search_nodes',limit=1))
    second=query_graph(s,GraphQuery(tool='search_nodes',offset=first['next_offset'],limit=1))
    assert first['items'][0]['key']!=second['items'][0]['key']
    assert 'error' in query_graph(s,GraphQuery(tool='get_neighbors',key='other-user-key'))
    result['items'][0]['contact_keys'].clear()
    assert s.graph().model_dump()==before
    assert query_graph(Conversation(request()),GraphQuery(tool='search_nodes',query='shelter'))['total']==0


def test_query_loop_feeds_results_to_model_without_mutating_graph():
    s=state();calls=[];events=[]
    class Agent(LLM):
        def complete(self,instruction,payload,schema):
            calls.append(payload)
            if len(calls)==1:
                return GraphAction(action='query',queries=[GraphQuery(tool='get_neighbors',key='A',include_evidence=True)])
            assert payload['graph_tool_history'][0]['result']['items'][0]['relation']=='visited'
            assert len(payload['context'])==2
            assert all(m['content']!='I visited a shelter.' for m in payload['context'])
            return GraphAction(action='finish',draft=TurnDraft(entities=[],relations=[]))
        def on_graph_tool(self,event):events.append(event)
    before=s.graph().model_dump()
    assert Agent().complete_turn(s,'extract',s.payload(3)).entities==[]
    assert len(events)==1 and s.graph().model_dump()==before
    assert 'known_entities' not in calls[0]


def test_tool_rounds_bounded_and_final_draft_forced():
    schemas=[]
    class Repeater(LLM):
        def complete(self,instruction,payload,schema):
            schemas.append(schema)
            if schema is TurnDraft:return TurnDraft(entities=[],relations=[])
            return GraphAction(action='query',queries=[GraphQuery(tool='search_nodes')])
    s=state();Repeater().complete_turn(s,'extract',s.payload(1))
    assert schemas==[GraphQueryAction,GraphAction,GraphAction,TurnDraft]


@pytest.mark.parametrize('action',[
    GraphAction(action='query'),
    GraphAction(action='finish'),
    GraphAction(action='finish',queries=[GraphQuery(tool='search_nodes')],draft=TurnDraft(entities=[],relations=[]))])
def test_invalid_actions_fail_closed(action):
    class Invalid(LLM):
        def complete(self,*args):return action
    s=state()
    with pytest.raises(LLMError):Invalid().complete_turn(s,'extract',s.payload(1))


def test_http_agent_requests_tool_and_uses_program_result(monkeypatch,tmp_path):
    original=httpx.Client;calls=[]
    monkeypatch.setattr(config,'PROJECT_ROOT',tmp_path)
    (tmp_path/'.env').write_text('LLM_API_KEY=test-key',encoding='utf-8')
    def respond(req):
        body=json.loads(req.content);payload=json.loads(body['messages'][1]['content']);calls.append(payload)
        if len(calls)==1:
            value=GraphAction(action='query',queries=[GraphQuery(tool='search_edges',query='visited',include_evidence=True)])
        else:
            assert payload['graph_tool_history'][0]['result']['items'][0]['evidence'][0]['message_index']==0
            value=GraphAction(action='finish',draft=TurnDraft(entities=[],relations=[]))
        return httpx.Response(200,json={'choices':[{'message':{'content':value.model_dump_json()}}]})
    monkeypatch.setattr(httpx,'Client',lambda **kw:original(transport=httpx.MockTransport(respond),**kw))
    s=state();LLM().complete_turn(s,'extract',s.payload(3))
    assert len(calls)==2


def test_first_network_action_schema_cannot_skip_graph_query():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        GraphQueryAction.model_validate(dict(action='finish',queries=[],draft=dict(entities=[],relations=[])))
    with pytest.raises(ValidationError):
        GraphQueryAction.model_validate(dict(action='query',queries=[],draft=None))
