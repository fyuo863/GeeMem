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
from .entity_resolver import EntityResolver

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
    aliases: list[str] = Field(default_factory=list, max_length=20)


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
            'If a message explicitly maps a nickname to a canonical person, put the nickname in aliases; '
            'otherwise return an empty aliases list. '
            'Use “我” for the current speaker when no canonical name is given. Do not assign a friend’s '
            'attribute to the current user. Do not extract relationships or events. Never extract an activity, '
            'shared outing, or event as a profile fact. Preserve uncertainty, '
            'plans and negation in certainty. Use the original message_index values as evidence; do not '
            'invent indices. A context message can resolve a pronoun but cannot add an unsupported fact. '
            'Return an empty list when no explicit attribute is present.')
        instruction += (' Extract stable identity, current attributes, preferences and constraints only. '
            'Exclude greetings, compliments, transient emotions, goals, future plans, event histories, '
            'relationship assertions and descriptions of objects. A favorite subtype is more informative '
            'than a general interest: keep both when explicit. Travel memories never prove residence. '
            'Losing a job does not prove current employment. Use confirmed for explicit assertions; '
            'uncertain only if the source itself hedges. Do not infer occupation from an activity. '
            'Resolve I using speaker; never combine different speakers into one I. '
            'Avoid unknown people as profile subjects; leave unresolved identities for later resolution.')
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
                entity = self._entity(db, user_id, item.subject_kind, item.subject_name, item.aliases,
                                      source_id=messages[item.message_indices[0]]['source_id'])
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

    def _entity(self, db, user_id, kind, name, aliases=(), source_id=None):
        return self.resolver.resolve(db, user_id, name, kind, aliases=aliases, source_id=source_id)


class ProfileRetriever:
    def __init__(self, db_path: str): self.db_path = db_path

    def find(self, user_id: str, *, subject=None, attribute=None, value=None,
             status='active', limit=20):
        if not 1 <= limit <= 100: raise ValueError('limit must be between 1 and 100')
        clauses=['f.user_id=?','f.status=?']; args=[user_id,status]
        if subject:
            clauses.append('(e.entity_key=? OR lower(e.name)=lower(?) OR EXISTS '
                           '(SELECT 1 FROM entity_aliases a WHERE a.entity_id=e.id AND a.user_id=e.user_id '
                           'AND a.normalized_alias=? AND a.status=\'active\'))')
            args.extend([entity_key('person', subject), subject, normalize(subject)])
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
