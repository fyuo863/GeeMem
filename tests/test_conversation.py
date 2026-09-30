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


def test_contact_is_independent_of_owner_and_mentioning_speaker(tmp_path):
    req=request();req.messages[0].content='Hey John, you volunteered at the shelter.'
    state=Conversation(req).accept(TurnDraft(entities=[dict(ref='s',name='shelter',
        kind='place',owner=None,contacts=['B','B'])],relations=[]),0)
    node=next(n for n in state.nodes.values() if n.kind=='place')
    assert node.owner_key is None
    assert node.contact_keys==[state.keys['B']]
    assert node.speaker_tags==[state.keys['A']]
    assert node.message_indices==[0]
    store=Store(tmp_path/'db');store.add(req,state.graph())
    with store.connect() as db:
        row=db.execute("SELECT * FROM nodes WHERE name='shelter'").fetchone()
        assert row['owner_id'] is None
        assert json.loads(row['contact_keys'])==[state.keys['B']]
    assert state.payload(1)['known_entities'][0]['contacts']==['B']


def test_same_contact_does_not_merge_unowned_nodes():
    req=request();req.messages[0].content=req.messages[1].content='We visited the shelter.'
    entity=dict(ref='s',name='shelter',kind='place',contacts=['A'])
    state=Conversation(req).accept(TurnDraft(entities=[entity],relations=[]),0)
    key=next(n.key for n in state.nodes.values() if n.kind=='place')
    state=state.accept(TurnDraft(entities=[dict(entity,same_as=key,match_quote='shelter')],relations=[]),1)
    assert len([n for n in state.nodes.values() if n.kind=='place'])==2
    assert any(a['action']=='decline_merge' for a in state.audit)


def test_valid_merge_accumulates_contacts_and_persisted_updates(tmp_path):
    req=request();req.messages[0].content=req.messages[1].content='My family visited us.'
    state=Conversation(req).accept(TurnDraft(entities=[dict(family('A'),contacts=['A'])],relations=[]),0)
    key=next(n.key for n in state.nodes.values() if n.kind=='group')
    store=Store(tmp_path/'db');store.add(req,state.graph())
    state=state.accept(TurnDraft(entities=[dict(family('A',key,'family'),contacts=['B'])],relations=[]),1)
    assert set(state.nodes[key].contact_keys)==set(state.keys.values())
    second=req.model_copy(update={'request_id':'second'});store.add(second,state.graph())
    with store.connect() as db:
        row=db.execute("SELECT contact_keys FROM nodes WHERE name='family'").fetchone()
        assert set(json.loads(row[0]))==set(state.keys.values())


def test_invalid_contact_rejected_before_write(tmp_path):
    state=Conversation(request())
    with pytest.raises(ValueError,match='contacts'):
        state.accept(TurnDraft(entities=[dict(family('A'),contacts=['C'])],relations=[]),0)
    graph=initial().graph();graph.nodes[-1].contact_keys=['missing']
    store=Store(tmp_path/'db')
    with pytest.raises(ValueError,match='Unknown node contact'):store.add(request(),graph)
    with store.connect() as db:assert db.execute('SELECT count(*) FROM messages').fetchone()[0]==0
    graph.nodes[-1].contact_keys=[graph.nodes[-1].key]
    with pytest.raises(ValueError,match='itself'):graph.validate_references(3)


def test_legacy_contacts_migration_preserves_owner_and_evidence(tmp_path):
    store=Store(tmp_path/'db');store.add(request(),initial().graph())
    with store.connect() as db:
        before=[tuple(r) for r in db.execute('SELECT id,owner_id FROM nodes ORDER BY id')]
        evidence=[tuple(r) for r in db.execute('SELECT * FROM node_evidence ORDER BY node_id,message_id')]
        db.execute('ALTER TABLE nodes DROP COLUMN contact_keys')
    for _ in range(2):
        migrated=Store(tmp_path/'db')
        with migrated.connect() as db:
            assert before==[tuple(r) for r in db.execute('SELECT id,owner_id FROM nodes ORDER BY id')]
            assert evidence==[tuple(r) for r in db.execute('SELECT * FROM node_evidence ORDER BY node_id,message_id')]
            assert all(json.loads(r[0])==[] for r in db.execute('SELECT contact_keys FROM nodes'))
