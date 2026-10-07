"""Independent profile facts sharing entity identity with relationship memory."""
from contextlib import closing
from dataclasses import dataclass
import json
import sqlite3
from typing import Any

from pydantic import Field
from .llm import LLM
from .models import StrictModel
from .relationship import canonical_name, entity_key, normalize, _id

_ATTRIBUTE_ALIASES = {
    '职业': 'occupation', '工作': 'occupation', 'occupation': 'occupation',
    '居住地': 'residence', '住址': 'residence', '地址': 'residence',
    'location': 'residence', 'residence': 'residence',
    '偏好': 'preference', '喜好': 'preference', '爱好': 'preference', '兴趣': 'preference', 'preference': 'preference',
    '健康限制': 'health_constraint', '过敏': 'health_constraint',
    'health_constraint': 'health_constraint',
}

def canonical_attribute(value: str) -> str:
    return _ATTRIBUTE_ALIASES.get(normalize(value), normalize(value))


class ProfileCandidate(StrictModel):
    subject_name: str = Field(min_length=1, max_length=256)
    subject_kind: str = Field(default='person', min_length=1, max_length=64)
    attribute: str = Field(min_length=1, max_length=128)
    value: str = Field(min_length=1, max_length=2000)
    certainty: str = Field(default='confirmed', pattern=r'^(confirmed|uncertain|planned|denied)$')
    message_indices: list[int] = Field(min_length=1, max_length=50)
    confidence: float = Field(ge=0, le=1)


class ProfileExtraction(StrictModel):
    facts: list[ProfileCandidate] = Field(max_length=100)


class ProfileFact(StrictModel):
    id: str
    user_id: str
    entity_id: str
    subject_name: str
    subject_kind: str
    attribute: str
    value: str
    certainty: str
    status: str
    confidence: float
    evidence: list[dict[str, Any]] = Field(default_factory=list)


@dataclass
class ProfileBuilder:
    db_path: str
    llm: Any = None

    def __post_init__(self):
        self.llm = self.llm or LLM()
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
                CREATE TABLE IF NOT EXISTS profile_facts(
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL REFERENCES relationship_entities(id),
                    attribute TEXT NOT NULL, value TEXT NOT NULL, normalized_value TEXT NOT NULL,
                    certainty TEXT NOT NULL, status TEXT NOT NULL, confidence REAL NOT NULL,
                    UNIQUE(user_id,entity_id,attribute,normalized_value,certainty));
                CREATE TABLE IF NOT EXISTS profile_evidence(
                    fact_id TEXT NOT NULL REFERENCES profile_facts(id), source_id TEXT NOT NULL,
                    message_index INTEGER NOT NULL, content TEXT NOT NULL, source_kind TEXT NOT NULL,
                    PRIMARY KEY(fact_id,source_id));
                CREATE INDEX IF NOT EXISTS profile_lookup ON profile_facts(user_id,attribute,status);
            ''')

    def extract(self, builder_messages: list[dict], *, context='') -> ProfileExtraction:
        if not isinstance(builder_messages, list) or not builder_messages:
            raise ValueError('Profile extraction requires builder messages')
        indexed = []
        for message in builder_messages:
            if not isinstance(message, dict) or not message.get('content'):
                raise ValueError('Each builder message requires content')
            indexed.append(message)
        instruction = (
            'Extract explicit person attributes from the supplied messages. Return names, not database IDs. '
            'Use “我” for the current speaker when no canonical name is given. Do not assign a friend’s '
            'attribute to the current user. Do not extract relationships or events. Never extract an activity, '
            'shared outing, or event as a profile fact. Preserve uncertainty, '
            'plans and negation in certainty. Use the original message_index values as evidence; do not '
            'invent indices. A context message can resolve a pronoun but cannot add an unsupported fact. '
            'Return an empty list when no explicit attribute is present.')
        result = self.llm.complete(instruction, {'messages': indexed, 'context': context}, ProfileExtraction)
        valid = {m.get('message_index') for m in indexed}
        normalized = []
        for item in result.facts:
            item = item.model_copy(update={'subject_name': canonical_name(item.subject_name),
                                           'attribute': canonical_attribute(item.attribute)})
            if item.attribute in {'activity', 'event', 'plan', 'event_type'}:
                continue
            if any(type(i) is not int or i not in valid for i in item.message_indices):
                raise ValueError('Profile evidence index is not in builder messages')
            normalized.append(item)
        return ProfileExtraction(facts=normalized)

    def write(self, user_id: str, extraction: ProfileExtraction, builder_messages: list[dict]):
        messages = {m['message_index']: m for m in builder_messages}
        with closing(self.connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            ids = []
            for item in extraction.facts:
                entity = self._entity(db, user_id, item.subject_kind, item.subject_name)
                attribute = normalize(item.attribute)
                value_key = normalize(item.value)
                fact_id = _id('fact', user_id, entity['id'], attribute, value_key, item.certainty)
                existing = db.execute('SELECT id FROM profile_facts WHERE id=?', (fact_id,)).fetchone()
                if existing:
                    db.execute('UPDATE profile_facts SET confidence=MAX(confidence,?) WHERE id=?',
                               (item.confidence, fact_id))
                else:
                    db.execute('INSERT INTO profile_facts VALUES (?,?,?,?,?,?,?,?,?)',
                               (fact_id, user_id, entity['id'], attribute, item.value, value_key,
                                item.certainty, 'active', item.confidence))
                for index in sorted(set(item.message_indices)):
                    message = messages[index]
                    db.execute('INSERT OR IGNORE INTO profile_evidence VALUES (?,?,?,?,?)',
                               (fact_id, message['source_id'], index, message['content'],
                                message.get('source_kind', 'evidence')))
                ids.append(fact_id)
            return ids

    def _entity(self, db, user_id, kind, name):
        key = entity_key(kind, name)
        row = db.execute('SELECT * FROM relationship_entities WHERE user_id=? AND entity_key=?',
                         (user_id, key)).fetchone()
        if row:
            aliases = set(json.loads(row['aliases'])) | {name}
            db.execute('UPDATE relationship_entities SET aliases=? WHERE id=?',
                       (json.dumps(sorted(aliases), ensure_ascii=False), row['id']))
            return row
        identifier = _id('ent', user_id, key)
        db.execute('INSERT INTO relationship_entities VALUES (?,?,?,?,?,?)',
                   (identifier, user_id, key, name, kind, json.dumps([name], ensure_ascii=False)))
        return db.execute('SELECT * FROM relationship_entities WHERE id=?', (identifier,)).fetchone()


class ProfileRetriever:
    def __init__(self, db_path: str): self.db_path = db_path

    def find(self, user_id: str, *, subject=None, attribute=None, value=None,
             status='active', limit=20):
        if not 1 <= limit <= 100: raise ValueError('limit must be between 1 and 100')
        clauses=['f.user_id=?','f.status=?']; args=[user_id,status]
        if subject:
            clauses.append('(e.entity_key=? OR lower(e.name)=lower(?))')
            args.extend([entity_key('person', subject), subject])
        if attribute: clauses.append('f.attribute=?'); args.append(normalize(attribute))
        if value: clauses.append('f.normalized_value=?'); args.append(normalize(value))
        with closing(sqlite3.connect(self.db_path)) as db:
            db.row_factory=sqlite3.Row
            rows=db.execute('''SELECT f.*,e.name subject_name,e.kind subject_kind FROM profile_facts f
                JOIN relationship_entities e ON e.id=f.entity_id WHERE '''+' AND '.join(clauses)+
                ' ORDER BY f.confidence DESC LIMIT ?', [*args,limit]).fetchall()
            out=[]
            for row in rows:
                evidence=[dict(e) for e in db.execute('SELECT source_id,message_index,content,source_kind FROM profile_evidence WHERE fact_id=?',(row['id'],))]
                out.append(ProfileFact(id=row['id'],user_id=row['user_id'],entity_id=row['entity_id'],subject_name=row['subject_name'],subject_kind=row['subject_kind'],attribute=row['attribute'],value=row['value'],certainty=row['certainty'],status=row['status'],confidence=row['confidence'],evidence=evidence))
            return out
