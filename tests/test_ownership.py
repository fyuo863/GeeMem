import sqlite3

import pytest

from memory.models import AddRequest, Graph, SearchRequest
from memory.store import Store


def request(rid="r", user="u"):
    return AddRequest(request_id=rid, user_id=user, session_id=rid, messages=[
        dict(role="user", content="My family attended a concert.", timestamp=0)])


def graph(owner):
    # Identical family key across writes: storage must still distinguish owners.
    return Graph.model_validate(dict(nodes=[
        dict(key="family", name="family", kind="group", owner_key=owner, message_indices=[0]),
        dict(key=owner, name=owner, kind="person", message_indices=[0]),
    ], edges=[dict(source="family", target=owner, relation="家人", message_indices=[0])]))


def test_same_key_distinct_owners_and_cross_session_merge(tmp_path):
    store = Store(tmp_path / "db")
    john = store.add(request(), graph("John"))
    maria = store.add(request("r2"), graph("Maria"))
    store.add(request("r3"), graph("John"))
    store.add(request(user="other"), graph("John"))
    with store.connect() as db:
        families = db.execute("SELECT n.id,o.name FROM nodes n JOIN nodes o ON n.owner_id=o.id WHERE n.user_id='u'").fetchall()
        assert {r["name"] for r in families} == {"John", "Maria"}
        assert len(families) == 2
        for family in families:
            count = db.execute("SELECT count(*) FROM node_evidence WHERE node_id=?", (family["id"],)).fetchone()[0]
            assert count == (2 if family["name"] == "John" else 1)
        assert not db.execute("PRAGMA foreign_key_check").fetchall()
    hits = store.search(SearchRequest(user_id="u", query="John", max_hops=1), ["John"])["data"]
    assert john["message_ids"][0] in {h["id"] for h in hits}
    assert maria["message_ids"][0] not in {h["id"] for h in hits}


@pytest.mark.parametrize("owner", ["missing", "family", "cycle"])
def test_invalid_ownership_rejected_before_write(tmp_path, owner):
    store = Store(tmp_path / "db")
    g = graph("John")
    g.nodes[0].owner_key = owner
    if owner == "cycle":
        g.nodes[0].owner_key = "John"
        g.nodes[1].owner_key = "family"
    with pytest.raises(ValueError):
        store.add(request(), g)
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM messages").fetchone()[0] == 0


def test_legacy_owner_migration_preserves_unknown_nodes(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY,user_id TEXT,node_key TEXT,name TEXT,kind TEXT,aliases TEXT,UNIQUE(user_id,node_key))")
        db.execute("INSERT INTO nodes VALUES ('old','u','family','family','group','[]')")
    store = Store(path)
    store.add(request(), graph("John"))
    Store(path)
    with store.connect() as db:
        assert db.execute("SELECT owner_id FROM nodes WHERE id='old'").fetchone()[0] is None
        assert db.execute("SELECT count(*) FROM nodes WHERE kind='group'").fetchone()[0] == 2
        assert not db.execute("PRAGMA foreign_key_check").fetchall()
