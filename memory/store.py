import hashlib
import json
import sqlite3
import unicodedata
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from .models import AddRequest, Graph, SearchRequest

class Conflict(Exception):
    pass

def normalize(value):
    return unicodedata.normalize("NFKC", value).casefold().strip()

class Store:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS requests (
                user_id TEXT, request_id TEXT, digest TEXT NOT NULL, result TEXT NOT NULL,
                PRIMARY KEY(user_id, request_id));
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL, session_id TEXT NOT NULL,
                role TEXT NOT NULL, content TEXT NOT NULL, timestamp INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS message_scope ON messages(user_id, session_id);
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL, node_key TEXT NOT NULL,
                name TEXT NOT NULL, kind TEXT NOT NULL, aliases TEXT NOT NULL,
                UNIQUE(user_id, node_key));
            CREATE TABLE IF NOT EXISTS node_evidence (
                node_id TEXT REFERENCES nodes(id), message_id TEXT REFERENCES messages(id),
                PRIMARY KEY(node_id, message_id));
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                source TEXT REFERENCES nodes(id), target TEXT REFERENCES nodes(id),
                relation TEXT NOT NULL, UNIQUE(user_id, source, target, relation));
            CREATE TABLE IF NOT EXISTS edge_evidence (
                edge_id TEXT REFERENCES edges(id), message_id TEXT REFERENCES messages(id),
                PRIMARY KEY(edge_id, message_id));
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def digest(request):
        return hashlib.sha256(request.model_dump_json().encode()).hexdigest()

    def existing(self, request, db=None):
        if db is None:
            with self.connect() as connection:
                return self.existing(request, connection)
        row = db.execute("SELECT * FROM requests WHERE user_id=? AND request_id=?",
                         (request.user_id, request.request_id)).fetchone()
        if row is None:
            return None
        if row["digest"] != self.digest(request):
            raise Conflict("request_id already used with a different payload")
        return dict(json.loads(row["result"]), deduplicated=True)

    def add(self, request: AddRequest, graph: Graph):
        graph.validate_references(len(request.messages))
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = self.existing(request, db)
            if previous:
                return previous
            ids = [str(uuid4()) for _ in request.messages]
            for mid, message in zip(ids, request.messages):
                db.execute("INSERT INTO messages VALUES (?,?,?,?,?,?)", (
                    mid, request.user_id, request.session_id, message.role,
                    message.content, message.timestamp))
            node_ids = {}
            for node in graph.nodes:
                key = normalize(node.key)
                row = db.execute("SELECT * FROM nodes WHERE user_id=? AND node_key=?",
                                 (request.user_id, key)).fetchone()
                nid = row["id"] if row else str(uuid4())
                aliases = set(node.aliases + [node.name])
                if row:
                    aliases.update(json.loads(row["aliases"]))
                    db.execute("UPDATE nodes SET aliases=? WHERE id=?",
                               (json.dumps(sorted(aliases), ensure_ascii=False), nid))
                else:
                    db.execute("INSERT INTO nodes VALUES (?,?,?,?,?,?)", (
                        nid, request.user_id, key, node.name, node.kind,
                        json.dumps(sorted(aliases), ensure_ascii=False)))
                node_ids[node.key] = nid
                db.executemany("INSERT OR IGNORE INTO node_evidence VALUES (?,?)",
                               [(nid, ids[i]) for i in node.message_indices])
            for edge in graph.edges:
                source, target = node_ids[edge.source], node_ids[edge.target]
                relation = normalize(edge.relation)
                row = db.execute("SELECT id FROM edges WHERE user_id=? AND source=? AND target=? AND relation=?",
                                 (request.user_id, source, target, relation)).fetchone()
                eid = row["id"] if row else str(uuid4())
                if not row:
                    db.execute("INSERT INTO edges VALUES (?,?,?,?,?)",
                               (eid, request.user_id, source, target, relation))
                db.executemany("INSERT OR IGNORE INTO edge_evidence VALUES (?,?)",
                               [(eid, ids[i]) for i in edge.message_indices])
            result = dict(request_id=request.request_id, message_ids=ids,
                          nodes=len(graph.nodes), edges=len(graph.edges), deduplicated=False)
            db.execute("INSERT INTO requests VALUES (?,?,?,?)", (
                request.user_id, request.request_id, self.digest(request), json.dumps(result)))
            return result

    def search(self, request: SearchRequest, keywords: list[str]):
        terms = {normalize(k) for k in keywords if normalize(k)}
        if not terms:
            return {"data": []}
        with self.connect() as db:
            sql = "SELECT * FROM messages WHERE user_id=?"
            args = [request.user_id]
            if request.session_id is not None:
                sql += " AND session_id=?"
                args.append(request.session_id)
            messages = {r["id"]: dict(r) for r in db.execute(sql, args)}
            node_evidence, edge_evidence = defaultdict(set), defaultdict(set)
            for row in db.execute("SELECT ne.* FROM node_evidence ne JOIN nodes n ON n.id=ne.node_id WHERE n.user_id=?", (request.user_id,)):
                if row["message_id"] in messages:
                    node_evidence[row["node_id"]].add(row["message_id"])
            for row in db.execute("SELECT ee.* FROM edge_evidence ee JOIN edges e ON e.id=ee.edge_id WHERE e.user_id=?", (request.user_id,)):
                if row["message_id"] in messages:
                    edge_evidence[row["edge_id"]].add(row["message_id"])
            nodes = [dict(r) for r in db.execute("SELECT * FROM nodes WHERE user_id=?", (request.user_id,))]
            edges = [dict(r) for r in db.execute("SELECT * FROM edges WHERE user_id=?", (request.user_id,)) if edge_evidence[r["id"]]]
        # Each keyword has its own bounded BFS. Incoming edges can be inspected for
        # questions about an object; stored source/target direction is never changed.
        adjacency = defaultdict(list)
        for edge in edges:
            adjacency[edge["source"]].append((edge["target"], edge["id"]))
            adjacency[edge["target"]].append((edge["source"], edge["id"]))
        scores = defaultdict(float)
        for term in terms:
            seeds, evidence = set(), {}
            for node in nodes:
                labels = [node["name"], node["node_key"], *json.loads(node["aliases"])]
                if node_evidence[node["id"]] and any(term in normalize(label) for label in labels):
                    seeds.add(node["id"])
            for edge in edges:
                if term in normalize(edge["relation"]):
                    seeds.update([edge["source"], edge["target"]])
                    for mid in edge_evidence[edge["id"]]:
                        evidence[mid] = 1.0
            queue = deque((nid, 0) for nid in sorted(seeds))
            visited = set(seeds)
            while queue:
                nid, depth = queue.popleft()
                weight = 1 / (depth + 1)
                for mid in node_evidence[nid]:
                    evidence[mid] = max(evidence.get(mid, 0), weight)
                if depth >= request.max_hops:
                    continue
                for other, eid in adjacency[nid]:
                    for mid in edge_evidence[eid]:
                        evidence[mid] = max(evidence.get(mid, 0), 1 / (depth + 2))
                    if other not in visited:
                        visited.add(other)
                        queue.append((other, depth + 1))
            for mid, weight in evidence.items():
                scores[mid] += weight / len(terms)
        ranked = sorted(scores, key=lambda mid: (-scores[mid], -messages[mid]["timestamp"], mid))
        data = []
        for mid in ranked[:request.limit]:
            message = messages[mid]
            created = datetime.fromtimestamp(message["timestamp"] / 1000, timezone.utc)
            data.append(dict(id=mid, content=message["content"], score=round(scores[mid], 6),
                             created_at=created.isoformat().replace("+00:00", "Z")))
        return {"data": data}
