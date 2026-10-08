"""Independent evidence-backed event memory."""
from contextlib import closing
from datetime import datetime, timezone
import hashlib, json, sqlite3, uuid
import re
from typing import Literal
from pydantic import Field, field_validator
from .llm import LLM
from .models import StrictModel

class EventCandidate(StrictModel):
    subject_name: str = Field(min_length=1, max_length=256)
    event_type: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=2000)
    status: Literal['planned','ongoing','completed','cancelled','uncertain'] = 'completed'
    certainty: Literal['confirmed','uncertain'] = 'confirmed'
    event_time_start: str | None = Field(default=None, max_length=64)
    event_time_end: str | None = Field(default=None, max_length=64)
    time_expression: str | None = Field(default=None, max_length=128)
    reference_timestamp: int | None = Field(default=None, ge=0)
    time_message_index: int | None = Field(default=None, ge=0, strict=True)
    time_precision: Literal['exact','day','month','year','relative','unknown'] = 'unknown'
    location: str | None = Field(default=None, max_length=256)
    participants: list[str] = Field(default_factory=list, max_length=50)
    previous_state: str | None = Field(default=None, max_length=1000)
    current_state: str | None = Field(default=None, max_length=1000)
    message_indices: list[int] = Field(min_length=1, max_length=50)
    confidence: float = Field(ge=0, le=1)

    @field_validator('event_time_start','event_time_end','time_expression','reference_timestamp',
                     'time_message_index','location','previous_state','current_state',mode='before')
    @classmethod
    def clean_null(cls, value):
        return None if isinstance(value,str) and value.strip().lower() in ('null','none','unknown','') else value

class EventExtraction(StrictModel):
    events: list[EventCandidate] = Field(max_length=100)

PROMPT = '''Extract explicit events, plans, progress updates, decisions and state changes.
Do not convert questions, assistant suggestions, stable attributes or reusable rules into events.
Keep subject and participants. A plan is planned, cancellation is cancelled, completed is completed.
Separate event time from source/message time. Use only explicit event dates or clear relative times;
otherwise use null and unknown. Preserve previous_state/current_state and source message_index.
Context resolves pronouns but cannot create unsupported events. Do not invent IDs or dates.'''
PROMPT += '''
For event_time_start, event_time_end and reference_timestamp return null: the program resolves dates.
Copy the exact time_expression from a cited message, and identify its time_message_index (or null).
Attach time to the correct action: a walk two weeks ago and a later design are separate events.
Include completed leisure activities as well as future plans. Do not omit a later activity
because earlier messages describe another task. Preserve uncertainty in perhaps/maybe.
Use JSON null, never the string "null". Resolve subjects from speaker names.'''


def resolve_event_time(start, end, precision, expression=None, reference_timestamp=None):
    """Normalize safe relative expressions without inventing day precision."""
    from .event_time import parse_time
    return parse_time(expression, reference_timestamp)


class EventBuilder:
    def __init__(self, db_path, llm=None):
        self.db_path=str(db_path); self.llm=llm
        with closing(self.connect()) as db, db:
            db.executescript('''CREATE TABLE IF NOT EXISTS event_records(
                id TEXT PRIMARY KEY,user_id TEXT NOT NULL,fingerprint TEXT NOT NULL,
                subject_name TEXT NOT NULL,event_type TEXT NOT NULL,description TEXT NOT NULL,
                status TEXT NOT NULL,certainty TEXT NOT NULL,event_time_start TEXT,event_time_end TEXT,
                time_precision TEXT NOT NULL,location TEXT,participants TEXT NOT NULL,
                previous_state TEXT,current_state TEXT,confidence REAL NOT NULL,version INTEGER NOT NULL,
                supersedes TEXT,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS event_lookup ON event_records(user_id,status,event_time_start);
                CREATE TABLE IF NOT EXISTS event_evidence(
                event_id TEXT NOT NULL REFERENCES event_records(id),source_id TEXT NOT NULL,
                message_index INTEGER NOT NULL,content TEXT NOT NULL,source_kind TEXT NOT NULL,
                source_timestamp INTEGER,PRIMARY KEY(event_id,source_id));''')
            cols={r[1] for r in db.execute('PRAGMA table_info(event_records)')}
            for name, definition in [('time_expression','TEXT'),('reference_timestamp','INTEGER')]:
                if name not in cols: db.execute(f'ALTER TABLE event_records ADD COLUMN {name} {definition}')
    def connect(self):
        db=sqlite3.connect(self.db_path,timeout=30); db.row_factory=sqlite3.Row; db.execute('PRAGMA foreign_keys=ON'); return db
    @staticmethod
    def validate(extraction,messages):
        by={m.get('message_index'):m for m in messages}
        if len(by)!=len(messages) or any(type(i) is not int or not m.get('source_id') or not m.get('content') for i,m in by.items()): raise ValueError('Invalid event source messages')
        for e in extraction.events:
            if any(i not in by for i in e.message_indices): raise ValueError('Unknown event evidence index')
            if not any(by[i].get('source_kind','evidence')=='evidence' for i in e.message_indices): raise ValueError('Event requires direct evidence')
        return by
    def extract(self,builder_messages):
        self.validate(EventExtraction(events=[]),builder_messages)
        out=(self.llm or LLM()).complete(PROMPT,{'messages':builder_messages},EventExtraction); self.validate(out,builder_messages); return out
    def write(self,user_id,extraction,builder_messages,*,supersedes=None):
        messages=self.validate(extraction,builder_messages); now=datetime.now(timezone.utc).isoformat(); ids=[]
        if supersedes and len(extraction.events)!=1: raise ValueError('Replacement requires one event')
        with closing(self.connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            for e in extraction.events:
                cited = [messages[i] for i in e.message_indices]
                matches = [m for m in cited if e.time_expression and e.time_expression.casefold() in m['content'].casefold()]
                if e.time_message_index is not None:
                    matches = [m for m in matches if m['message_index']==e.time_message_index]
                # Do not trust the model's reference timestamp. Ambiguous anchors stay unresolved.
                stamps = {m.get('timestamp') for m in matches}
                anchor = next(iter(stamps)) if len(stamps)==1 else None
                e = e.model_copy(update={'reference_timestamp':anchor})
                data=e.model_dump(exclude={'message_indices','confidence'}); fp=hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False).encode()).hexdigest(); old=None
                start,end,precision = resolve_event_time(None,None,e.time_precision,e.time_expression if matches else None,e.reference_timestamp)
                if supersedes:
                    old=db.execute('SELECT * FROM event_records WHERE id=? AND user_id=?',(supersedes,user_id)).fetchone()
                    if old is None or old['status']=='superseded': raise ValueError('Invalid replacement target')
                row=None if supersedes else db.execute("SELECT * FROM event_records WHERE user_id=? AND fingerprint=? AND status!='superseded'",(user_id,fp)).fetchone()
                if row: eid=row['id']
                else:
                    eid=str(uuid.uuid4()); db.execute('''INSERT INTO event_records
                      (id,user_id,fingerprint,subject_name,event_type,description,status,certainty,event_time_start,event_time_end,time_precision,location,participants,previous_state,current_state,confidence,version,supersedes,created_at,updated_at,time_expression,reference_timestamp)
                      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',(eid,user_id,fp,e.subject_name,e.event_type,e.description,e.status,e.certainty,start,end,precision,e.location,json.dumps(e.participants,ensure_ascii=False),e.previous_state,e.current_state,e.confidence,old['version']+1 if old else 1,supersedes,now,now,e.time_expression,e.reference_timestamp))
                    if old: db.execute("UPDATE event_records SET status='superseded',updated_at=? WHERE id=?",(now,old['id']))
                for i in set(e.message_indices):
                    m=messages[i]; db.execute('INSERT OR IGNORE INTO event_evidence VALUES (?,?,?,?,?,?)',(eid,m['source_id'],i,m['content'],m.get('source_kind','evidence'),m.get('timestamp')))
                ids.append(eid)
        return ids

class EventRetriever:
    def __init__(self,db_path): self.db_path=str(db_path)
    def timeline(self,user_id,subject=None,start=None,end=None,limit=100,status=None):
        with closing(sqlite3.connect(self.db_path)) as db:
            db.row_factory=sqlite3.Row; clauses=['user_id=?','status!=?']; args=[user_id,'superseded']
            if status: clauses[-1]='status=?'; args[-1]=status
            if subject: clauses.append('lower(subject_name)=lower(?)'); args.append(subject)
            rows=db.execute('SELECT * FROM event_records WHERE '+' AND '.join(clauses)+' ORDER BY CASE WHEN event_time_start IS NULL THEN 1 ELSE 0 END,event_time_start,created_at LIMIT ?',args+[limit]).fetchall(); out=[]
            for row in rows:
                if start and row['event_time_end'] and row['event_time_end']<start: continue
                if end and row['event_time_start'] and row['event_time_start']>end: continue
                item=dict(row); item['participants']=json.loads(item['participants']); item['evidence']=[dict(x) for x in db.execute('SELECT * FROM event_evidence WHERE event_id=? ORDER BY message_index',(row['id'],))]; out.append(item)
            return out
    def find(self,user_id,**kwargs): return self.timeline(user_id,kwargs.get('subject'),kwargs.get('start'),kwargs.get('end'),kwargs.get('limit',20),kwargs.get('status'))



