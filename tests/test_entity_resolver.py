from memory.entity_resolver import EntityResolver
from memory.relationship import RelationshipBuilder, RelationshipCandidate, RelationshipExtraction
from memory.profile import ProfileBuilder, ProfileCandidate, ProfileExtraction, ProfileRetriever


class Fake:
    def __init__(self, result): self.result = result
    def complete(self, *args): return self.result


def test_explicit_nickname_is_shared_across_rounds(tmp_path):
    path = str(tmp_path / 'db')
    messages = [dict(message_index=0, source_id='s0', content='我的朋友王伟，大家都叫他老王。')]
    rel = RelationshipBuilder(path, Fake(RelationshipExtraction(relationships=[
        RelationshipCandidate(subject_name='王伟', aliases=['老王'], relation='friend',
                               object_name='我', message_indices=[0], confidence=.9)
    ])))
    rel.write('u', rel.extract(messages), messages)
    followup = [dict(message_index=0, source_id='s1', content='老王住在北京。')]
    prof = ProfileBuilder(path, Fake(ProfileExtraction(facts=[
        ProfileCandidate(subject_name='老王', attribute='residence', value='北京',
                         message_indices=[0], confidence=.8)
    ])))
    prof.write('u', prof.extract(followup), followup)
    row = ProfileRetriever(path).find('u', subject='老王')[0]
    assert row.subject_name == '王伟'
    with rel.connect() as db:
        assert db.execute('select count(*) from relationship_entities where user_id=?', ('u',)).fetchone()[0] == 2
        assert db.execute('select count(*) from entity_aliases where user_id=?', ('u',)).fetchone()[0] == 1


def test_ambiguous_alias_is_not_reassigned(tmp_path):
    path = str(tmp_path / 'db')
    resolver = EntityResolver()
    import sqlite3
    db = sqlite3.connect(path); db.row_factory = sqlite3.Row
    db.executescript('create table relationship_entities(id text primary key,user_id text,entity_key text,name text,kind text,aliases text)')
    with db:
        first = resolver.resolve(db, 'u', '王伟')
        second = resolver.resolve(db, 'u', '王玮')
        assert first['id'] != second['id']
        assert resolver.register_alias(db, 'u', first['id'], '老王')
        assert not resolver.register_alias(db, 'u', second['id'], '老王')
        assert resolver.resolve(db, 'u', '老王')['id'] == first['id']


def test_aliases_are_isolated_by_user(tmp_path):
    path = str(tmp_path / 'db')
    resolver = EntityResolver()
    import sqlite3
    db = sqlite3.connect(path); db.row_factory = sqlite3.Row
    db.executescript('create table relationship_entities(id text primary key,user_id text,entity_key text,name text,kind text,aliases text)')
    with db:
        a = resolver.resolve(db, 'a', '王伟')
        b = resolver.resolve(db, 'b', '李伟')
        resolver.register_alias(db, 'a', a['id'], '老王')
        assert resolver.resolve(db, 'b', '老王')['id'] != a['id']
