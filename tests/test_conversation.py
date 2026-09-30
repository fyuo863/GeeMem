import json
import pytest
from pydantic import ValidationError
from memory.conversation import Conversation, TurnDraft
from memory.models import AddRequest
from memory.store import Store


def request():
    return AddRequest(request_id='r',user_id='u',session_id='s',messages=[
        dict(role='assistant',content='Hey John, my family supports me.',timestamp=0),
        dict(role='user',content='Thanks Maria. My family went to the concert.',timestamp=1),
        dict(role='assistant',content='John, your family enjoyed the concert.',timestamp=2)])


def family(owner, same_as=None, quote=None):
    return dict(ref='f',name='family',kind='group',owner=owner,same_as=same_as,match_quote=quote)


def initial():
    state=Conversation(request())
    return state.accept(TurnDraft(entities=[family('A')],
        relations=[dict(source='f',target='A',relation='支持')]),0)


def test_addressed_name_binds_other_speaker_without_identity_replacement():
    state=initial(); a=state.keys['A']; b=state.keys['B']
    assert state.nodes[b].name=='John' and state.nodes[a].name=='A'
    state=state.accept(TurnDraft(entities=[family('B')],relations=[]),1)
    assert state.keys['A']==a and state.nodes[a].name=='Maria'
    assert len([n for n in state.nodes.values() if n.kind=='person'])==2
    families=[n for n in state.nodes.values() if n.kind=='group']
    assert len(families)==2
    assert {n.owner_key for n in families}=={a,b}
    assert {tuple(n.speaker_tags) for n in families}=={(a,),(b,)}


def test_explicit_correspondence_unions_speaker_tags_and_program_positions(tmp_path):
    state=initial().accept(TurnDraft(entities=[family('B')],relations=[]),1)
    bfamily=next(n for n in state.nodes.values() if n.kind=='group' and n.owner_key==state.keys['B'])
    state=state.accept(TurnDraft(entities=[family('B',bfamily.key,'your family')],
        relations=[dict(source='f',target='B',relation='参与')]),2)
    merged=state.nodes[bfamily.key]
    assert merged.message_indices==[1,2]
    assert set(merged.speaker_tags)==set(state.keys.values())
    edge=next(e for e in state.edges.values() if e.relation=='参与')
    assert edge.message_indices==[2]
    assert edge.evidence[0].text==request().messages[2].content
    store=Store(tmp_path/'db');store.add(request(),state.graph())
    with store.connect() as db:
        row=db.execute('SELECT speaker_tags FROM nodes WHERE name=? AND owner_id IN (SELECT id FROM nodes WHERE name=?)',('family','John')).fetchone()
        assert set(json.loads(row[0]))==set(state.keys.values())
        assert not db.execute('PRAGMA foreign_key_check').fetchall()


def test_same_name_without_explicit_match_never_merges():
    state=initial()
    state=state.accept(TurnDraft(entities=[family('A')],relations=[]),2)
    assert len([n for n in state.nodes.values() if n.kind=='group'])==2


@pytest.mark.parametrize('owner,quote',[('B','your family'),('A','invented phrase')])
def test_wrong_owner_or_invented_match_rejected_without_mutation(owner,quote):
    state=initial(); key=next(n.key for n in state.nodes.values() if n.kind=='group')
    before=state.graph().model_dump()
    result=state.accept(TurnDraft(entities=[family(owner,key,quote)],relations=[]),2)
    assert state.graph().model_dump()==before
    assert len([n for n in result.nodes.values() if n.kind=='group'])==2
    assert result.nodes[key].message_indices==[0]
    assert any(a['action']=='decline_merge' for a in result.audit)


def test_consecutive_role_keeps_identity_and_foreign_name_is_rejected():
    req=request();req.messages[1].role='assistant';req.messages[2].role='user'
    state=Conversation(req)
    assert state.payload(0)['current']['speaker']==state.payload(1)['current']['speaker']=='A'
    assert state.payload(2)['current']['speaker']=='B'
    with pytest.raises(ValueError,match='verbatim'):
        state.bind_name('B','Unknown',0,state.keys['A'])


def test_model_cannot_assign_source_positions_or_speaker_tags():
    raw=dict(entities=[family('A')],relations=[])
    for field,value in [('message_indices',[99]),('speaker_tags',['B']),('evidence',[])]:
        modified=json.loads(json.dumps(raw));modified['entities'][0][field]=value
        with pytest.raises(ValidationError):TurnDraft.model_validate(modified)
    raw=dict(entities=[],relations=[dict(source='A',target='B',relation='询问',message_indices=[0])])
    with pytest.raises(ValidationError):TurnDraft.model_validate(raw)


def test_context_not_attributed_as_current_evidence():
    state=initial()
    state=state.accept(TurnDraft(entities=[],relations=[dict(source='A',target='B',relation='询问')]),2)
    edge=next(e for e in state.edges.values() if e.relation=='询问')
    assert edge.message_indices==[2] and edge.evidence[0].text==request().messages[2].content


def test_first_turn_reserves_named_counterpart_before_they_speak():
    req=request();req.messages=req.messages[:1]
    state=Conversation(req).accept(TurnDraft(entities=[],relations=[]),0)
    assert state.roles=={'A':'assistant','B':'user'}
    assert state.nodes[state.keys['B']].name=='John'
    assert state.nodes[state.keys['B']].message_indices==[0]


def test_unreferenced_duplicate_local_refs_are_preserved_but_ambiguous_edges_rejected():
    state=Conversation(request())
    entities=[family('A'),dict(ref='f',name='friends',kind='group',owner='A')]
    saved=state.accept(TurnDraft(entities=entities,relations=[]),0)
    assert len([n for n in saved.nodes.values() if n.kind=='group'])==2
    with pytest.raises(ValueError,match='Ambiguous'):
        state.accept(TurnDraft(entities=entities,relations=[dict(source='A',target='f',relation='支持')]),0)


def test_name_detection_ignores_third_person_and_keeps_current_program_binding():
    req=request();req.messages[0].content='A lot happened. My friend John is travelling.'
    state=Conversation(req).prepare(0)
    assert state.names=={}
    with pytest.raises(ValidationError):
        TurnDraft.model_validate(dict(self_name='John',entities=[],relations=[]))


def test_chinese_name_binding_and_no_rename_on_conflict():
    req=request();req.messages[0].content='你好，小林！我叫小王。'
    req.messages[1].content='你好，小李！'
    state=Conversation(req).prepare(0)
    assert state.names=={'A':'小王','B':'小林'}
    state=state.prepare(1)
    assert state.names['A']=='小王'
    assert any(a['action']=='unresolved_name' for a in state.audit)


def test_unused_duplicate_participant_is_discarded_but_referenced_one_rejected():
    state=initial()
    duplicate=dict(ref='p',name='John',kind='person',owner='B')
    saved=state.accept(TurnDraft(entities=[duplicate],relations=[]),0)
    assert len([n for n in saved.nodes.values() if n.kind=='person'])==2
    assert any(a['action']=='discard_duplicate_participant' for a in saved.audit)
    with pytest.raises(ValueError,match='Use speaker IDs'):
        state.accept(TurnDraft(entities=[duplicate],relations=[dict(source='A',target='p',relation='询问')]),0)
