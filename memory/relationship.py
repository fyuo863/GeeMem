"""Independent relationship memory builder and retriever.

This module deliberately does not import the legacy Graph/Store classes. It
stores semantic relationship assertions, entity identities, and source links
in its own tables. Model output contains names only; database IDs are bound by
the program from user scope and normalized names.
"""
from contextlib import closing
from dataclasses import dataclass
import json
import re
import sqlite3
from typing import Any

from pydantic import Field

from .llm import LLM
from .models import StrictModel
from .entity_resolver import EntityResolver


class RelationshipCandidate(StrictModel):
    subject_name: str = Field(min_length=1, max_length=256)
    subject_kind: str = Field(default='person', min_length=1, max_length=64)
    relation: str = Field(min_length=1, max_length=128)
    object_name: str = Field(min_length=1, max_length=256)
    object_kind: str = Field(default='person', min_length=1, max_length=64)
    message_indices: list[int] = Field(min_length=1, max_length=50)
    confidence: float = Field(ge=0, le=1)
    aliases: list[str] = Field(default_factory=list, max_length=20)
    object_aliases: list[str] = Field(default_factory=list, max_length=20)


class RelationshipExtraction(StrictModel):
    relationships: list[RelationshipCandidate] = Field(max_length=100)


class RelationshipRecord(StrictModel):
    id: str
    user_id: str
    subject_entity_id: str
    subject_name: str
    subject_kind: str
    relation: str
    object_entity_id: str
    object_name: str
    object_kind: str
    confidence: float
    status: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)


def normalize(value: str) -> str:
    return re.sub(r'\s+', ' ', value.strip().casefold())


def entity_key(kind: str, name: str) -> str:
    return f'{normalize(kind)}:{normalize(name)}'


_RELATION_ALIASES = {
    '朋友': 'friend', 'friend': 'friend', '好友': 'friend',
    '同事': 'colleague', 'colleague': 'colleague', 'colleague_of': 'colleague',
    '导师': 'mentor_of', 'mentor': 'mentor_of', 'mentor_of': 'mentor_of',
    '姐姐': 'sibling_of', '妹妹': 'sibling_of', '兄弟': 'sibling_of',
    '姐妹': 'sibling_of', 'sibling_of': 'sibling_of', 'sister_of': 'sibling_of',
    '父亲': 'parent_of', '母亲': 'parent_of', 'parent_of': 'parent_of',
}


def canonical_relation(value: str) -> str:
    return _RELATION_ALIASES.get(normalize(value), normalize(value))


def canonical_name(value: str) -> str:
    return '我' if normalize(value) in {'i', 'me', 'user', 'current speaker', '当前用户'} else value.strip()


def _id(prefix: str, *parts: str) -> str:
    import hashlib
    return prefix + '_' + hashlib.sha256('\x1f'.join(parts).encode('utf-8')).hexdigest()


@dataclass
class RelationshipBuilder:
    db_path: str
    llm: Any = None
    resolver: EntityResolver = None

    def __post_init__(self):
        self.llm = self.llm or LLM()
        self.resolver = self.resolver or EntityResolver()
        self.initialize()

    def connect(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        return db

    def initialize(self):
        with closing(self.connect()) as db, db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS relationship_entities(
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, entity_key TEXT NOT NULL,
                    name TEXT NOT NULL, kind TEXT NOT NULL, aliases TEXT NOT NULL,
                    UNIQUE(user_id,entity_key));
                CREATE TABLE IF NOT EXISTS relationship_assertions(
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                    subject_entity_id TEXT NOT NULL REFERENCES relationship_entities(id),
                    object_entity_id TEXT NOT NULL REFERENCES relationship_entities(id),
                    relation TEXT NOT NULL, relation_key TEXT NOT NULL,
                    confidence REAL NOT NULL, status TEXT NOT NULL,
                    UNIQUE(user_id,subject_entity_id,relation_key,object_entity_id));
                CREATE TABLE IF NOT EXISTS relationship_evidence(
                    relationship_id TEXT NOT NULL REFERENCES relationship_assertions(id),
                    source_id TEXT NOT NULL, message_index INTEGER NOT NULL,
                    content TEXT NOT NULL, source_kind TEXT NOT NULL,
                    PRIMARY KEY(relationship_id,source_id));
                CREATE INDEX IF NOT EXISTS relationship_lookup
                    ON relationship_assertions(user_id,relation_key);
            ''')

    def extract(self, builder_messages: list[dict], *, context: str = '') -> RelationshipExtraction:
        if not isinstance(builder_messages, list) or not builder_messages:
            raise ValueError('Relationship extraction requires builder messages')
        indexed = []
        for i, message in enumerate(builder_messages):
            if not isinstance(message, dict) or not message.get('content'):
                raise ValueError('Each builder message requires content')
            indexed.append(dict(message, route_index=i))
        instruction = (
            'Extract only explicit interpersonal or entity relationships from the supplied messages. '
            'Return names and kinds, never database IDs. If a message explicitly says a person is also called '
            'another name (for example “王伟，大家叫他老王”), put that nickname in aliases for the canonical '
            'person; otherwise return an empty aliases list. Do not infer a relationship from co-occurrence '
            'or a joint event: “I watched a film with Wang” does not prove friendship. '
            'A joint activity such as “I and Wang watched a film” is an event, not a relationship. '
            'For “Wang is my mentor/parent/sibling”, Wang is the subject and I/me is the object; '
            'preserve that direction. Use “I” for the current speaker when no canonical name is supplied. '
            'Preserve subject/object direction in the semantic assertion. Use message_index values from '
            'the supplied original message_index fields; cite every source message that supports the relation. '
            'A context message may explain a short answer but cannot create unsupported facts. '
            'Do not extract occupations, preferences or events as relationships. '
            'Use a stable concise relation such as friend, colleague, parent_of, mentor_of. '
            'Return an empty list when no explicit relationship is stated.')
        instruction += (' Use speaker names to resolve I. Never infer friendship from questions or '
                        'politeness. An unnamed partner is unknown, not a named person. '
                        'A nickname mentioned without explicit identity mapping must not create an alias merge.')
        payload = {'messages': indexed, 'context': context}
        result = self.llm.complete(instruction, payload, RelationshipExtraction)
        valid_indices = {m['message_index'] for m in indexed if isinstance(m.get('message_index'), int)}
        normalized = []
        for item in result.relationships:
            values = dict(subject_name=canonical_name(item.subject_name),
                          object_name=canonical_name(item.object_name),
                          relation=canonical_relation(item.relation))
            # Store symmetric friendship in one deterministic orientation.
            if values['relation'] in {'friend', 'colleague'} and values['object_name'] == '我' and values['subject_name'] != '我':
                values['subject_name'], values['object_name'] = values['object_name'], values['subject_name']
                values['object_aliases'] = list(item.aliases) + list(item.object_aliases)
                values['aliases'] = list(item.object_aliases)
            normalized.append(item.model_copy(update=values))
        result = result.model_copy(update={'relationships': normalized})
        for item in result.relationships:
            if item.subject_name.strip().casefold() == item.object_name.strip().casefold():
                raise ValueError('Relationship cannot connect an entity to itself')
            if any(type(i) is not int or i not in valid_indices for i in item.message_indices):
                raise ValueError('Relationship evidence index is not in builder messages')
        return result

    def write(self, user_id: str, extraction: RelationshipExtraction, builder_messages: list[dict]):
        messages = {m['message_index']: m for m in builder_messages}
        with closing(self.connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            records = []
            for item in extraction.relationships:
                subject = self._entity(db, user_id, item.subject_kind, item.subject_name, item.aliases,
                                       source_id=messages[item.message_indices[0]]['source_id'])
                target = self._entity(db, user_id, item.object_kind, item.object_name, item.object_aliases,
                                      source_id=messages[item.message_indices[0]]['source_id'])
                relation_key = normalize(item.relation)
                row = db.execute('SELECT * FROM relationship_assertions WHERE user_id=? AND '
                                 'subject_entity_id=? AND relation_key=? AND object_entity_id=?',
                                 (user_id, subject['id'], relation_key, target['id'])).fetchone()
                assertion_id = row['id'] if row else _id('rel', user_id, subject['id'], relation_key, target['id'])
                if row:
                    db.execute('UPDATE relationship_assertions SET confidence=MAX(confidence,?) WHERE id=?',
                               (item.confidence, assertion_id))
                else:
                    db.execute('INSERT INTO relationship_assertions VALUES (?,?,?,?,?,?,?,?)',
                               (assertion_id, user_id, subject['id'], target['id'], item.relation,
                                relation_key, item.confidence, 'active'))
                for index in sorted(set(item.message_indices)):
                    message = messages[index]
                    db.execute('INSERT OR IGNORE INTO relationship_evidence VALUES (?,?,?,?,?)',
                               (assertion_id, message['source_id'], index, message['content'],
                                message.get('source_kind', 'evidence')))
                records.append(assertion_id)
            return records

    def _entity(self, db, user_id, kind, name, aliases=(), source_id=None):
        return self.resolver.resolve(db, user_id, name, kind, aliases=aliases, source_id=source_id)


class RelationshipRetriever:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def find(self, user_id: str, *, subject: str | None = None,
             relation: str | None = None, object: str | None = None,
             status: str = 'active', limit: int = 20) -> list[RelationshipRecord]:
        if limit < 1 or limit > 100:
            raise ValueError('limit must be between 1 and 100')
        clauses, args = ['r.user_id=?', 'r.status=?'], [user_id, status]
        if subject:
            clauses.append('(normalize unavailable)')
        with closing(sqlite3.connect(self.db_path)) as db:
            db.row_factory = sqlite3.Row
            sql = '''SELECT r.*, s.name subject_name,s.kind subject_kind,
                     o.name object_name,o.kind object_kind
                     FROM relationship_assertions r
                     JOIN relationship_entities s ON s.id=r.subject_entity_id
                     JOIN relationship_entities o ON o.id=r.object_entity_id
                     WHERE '''
            real_clauses = ['r.user_id=?', 'r.status=?']; real_args = [user_id, status]
            if subject:
                real_clauses.append('(s.entity_key=? OR lower(s.name)=lower(?) OR EXISTS '
                                    '(SELECT 1 FROM entity_aliases a WHERE a.entity_id=s.id AND a.user_id=s.user_id '
                                    'AND a.normalized_alias=? AND a.status=\'active\'))')
                real_args.extend([entity_key('person', subject), subject, normalize(subject)])
            if relation:
                real_clauses.append('r.relation_key=?'); real_args.append(normalize(relation))
            if object:
                real_clauses.append('(o.entity_key=? OR lower(o.name)=lower(?) OR EXISTS '
                                    '(SELECT 1 FROM entity_aliases a WHERE a.entity_id=o.id AND a.user_id=o.user_id '
                                    'AND a.normalized_alias=? AND a.status=\'active\'))')
                real_args.extend([entity_key('person', object), object, normalize(object)])
            rows = db.execute(sql + ' AND '.join(real_clauses) +
                              ' ORDER BY r.confidence DESC LIMIT ?', [*real_args, limit]).fetchall()
            out = []
            for row in rows:
                evidence = [dict(e) for e in db.execute(
                    'SELECT source_id,message_index,content,source_kind FROM relationship_evidence WHERE relationship_id=?',
                    (row['id'],))]
                out.append(RelationshipRecord(id=row['id'], user_id=row['user_id'],
                    subject_entity_id=row['subject_entity_id'], subject_name=row['subject_name'],
                    subject_kind=row['subject_kind'], relation=row['relation'],
                    object_entity_id=row['object_entity_id'], object_name=row['object_name'],
                    object_kind=row['object_kind'], confidence=row['confidence'],
                    status=row['status'], evidence=evidence))
            return out

    def expand(self, user_id: str, entity: str, *, max_hops=2, limit=50):
        """Return bounded undirected neighborhood; relation direction stays in each record."""
        with closing(sqlite3.connect(self.db_path)) as db:
            row = db.execute('SELECT id FROM relationship_entities WHERE user_id=? AND '
                             '(entity_key=? OR lower(name)=lower(?))',
                             (user_id, entity_key('person', entity), entity)).fetchone()
        frontier = {row[0]} if row else set()
        seen, output = set(), []
        for _ in range(max(0, min(max_hops, 4)) + 1):
            next_frontier = set()
            for entity_id in frontier:
                records = self._find_entity(user_id, entity_id, limit)
                for record in records:
                    if record.id in seen: continue
                    seen.add(record.id); output.append(record)
                    next_frontier.add(record.object_entity_id)
                    next_frontier.add(record.subject_entity_id)
            frontier = next_frontier
        return output[:limit]

    def _find_entity(self, user_id, entity_id, limit):
        with closing(sqlite3.connect(self.db_path)) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute('''SELECT r.*, s.name subject_name,s.kind subject_kind,
                o.name object_name,o.kind object_kind FROM relationship_assertions r
                JOIN relationship_entities s ON s.id=r.subject_entity_id
                JOIN relationship_entities o ON o.id=r.object_entity_id
                WHERE r.user_id=? AND r.status=? AND
                (r.subject_entity_id=? OR r.object_entity_id=?)
                ORDER BY r.confidence DESC LIMIT ?''',
                (user_id, 'active', entity_id, entity_id, limit)).fetchall()
            return [self._record(db, row) for row in rows]

    @staticmethod
    def _record(db, row):
        evidence = [dict(e) for e in db.execute(
            'SELECT source_id,message_index,content,source_kind FROM relationship_evidence WHERE relationship_id=?',
            (row['id'],))]
        return RelationshipRecord(id=row['id'], user_id=row['user_id'],
            subject_entity_id=row['subject_entity_id'], subject_name=row['subject_name'],
            subject_kind=row['subject_kind'], relation=row['relation'],
            object_entity_id=row['object_entity_id'], object_name=row['object_name'],
            object_kind=row['object_kind'], confidence=row['confidence'],
            status=row['status'], evidence=evidence)

    def find_object(self, user_id, object, limit=20):
        return self.find(user_id, object=object, limit=limit)
