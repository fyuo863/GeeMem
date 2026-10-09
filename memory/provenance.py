"""Caller-supplied identity/time and program-assigned immutable source positions."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re


def source_id(user, request, index):
    return hashlib.sha256(json.dumps([user, request, index]).encode()).hexdigest()


def supplied_speaker(message):
    """Only explicit caller metadata/labels; never guess identity from pronouns."""
    if getattr(message,'speaker',None): return message.speaker
    match=re.match(r'^\[speaker: ([^\]\r\n]{1,256})\]\s',message.content)
    return match[1].strip() or None if match else None


def payload_digest(payload):
    # Preserve byte-compatible old digests when new optional metadata is absent.
    values = payload.model_dump()
    if values.get('session_timestamp') is None:
        values.pop('session_timestamp', None)
    for message in values['messages']:
        if message.get('speaker') is None:
            message.pop('speaker', None)
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def initialize(db):
    db.execute('''CREATE TABLE IF NOT EXISTS rag_sources (
        memory_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, source_index INTEGER NOT NULL,
        role TEXT, speaker TEXT, session_timestamp INTEGER, char_start INTEGER, char_end INTEGER)''')
    # Backfill only IDs/positions from legacy row order, never invent role/time/offsets.
    counts, positions = {}, {}
    for row in db.execute('SELECT id,user_id,session_id,request_id,message_index FROM rag_memories ORDER BY rowid'):
        scope = (row['user_id'], row['session_id'])
        key = (*scope, row['request_id'], row['message_index'])
        if key not in positions:
            positions[key] = counts.get(scope, 0)
            counts[scope] = positions[key] + 1
        db.execute('INSERT OR IGNORE INTO rag_sources(memory_id,source_id,source_index) VALUES (?,?,?)',
                   (row['id'], source_id(row['user_id'], row['request_id'], row['message_index']), positions[key]))


def store_sources(db, payload, chunker, tokenizer, size, overlap):
    offset = db.execute('SELECT COALESCE(MAX(s.source_index),-1)+1 FROM rag_sources s '
        'JOIN rag_memories m ON m.id=s.memory_id WHERE m.user_id=? AND m.session_id=?',
        (payload.user_id, payload.session_id)).fetchone()[0]
    for index, message in enumerate(payload.messages):
        spans = list(tokenizer.finditer(message.content))
        for chunk_index, text in enumerate(chunker(message.content, size, overlap)):
            start = 0 if len(spans) <= size else spans[chunk_index * (size-overlap)].start()
            assert message.content[start:start+len(text)] == text
            mid = hashlib.sha256(json.dumps([payload.user_id, payload.request_id, index, chunk_index]).encode()).hexdigest()
            db.execute('INSERT INTO rag_sources VALUES (?,?,?,?,?,?,?,?)',
                (mid, source_id(payload.user_id, payload.request_id, index), offset+index,
                 message.role, supplied_speaker(message), getattr(payload, 'session_timestamp', None), start, start+len(text)))


def metadata_text(row):
    row = dict(row)
    prefix = []
    if row.get('speaker'):
        prefix.append(row['speaker'] + ' said')
    stamp = row.get('timestamp') if row.get('timestamp') is not None else row.get('session_timestamp')
    if stamp is not None:
        date = datetime.fromtimestamp(stamp / 1000, timezone.utc)
        prefix.append('on ' + date.strftime('%d %B %Y'))
    text = (', '.join(prefix) + ': ' if prefix else '') + row['content']
    if stamp is not None:
        # Add an index-only interpretation of explicit relative phrases, not inferred events.
        references = []
        lower = row['content'].casefold()
        for phrase, days in [('yesterday', -1), ('tomorrow', 1)]:
            if re.search(r'\b' + phrase + r'\b', lower):
                try:
                    references.append(phrase + ' = ' + (date + timedelta(days=days)).strftime('%d %B %Y'))
                except OverflowError:
                    pass
        for phrase, months in [('last month', -1), ('next month', 1)]:
            if phrase in lower:
                absolute = date.year * 12 + date.month - 1 + months
                if 1 <= absolute//12 <= 9999:
                    references.append(phrase + ' = ' + datetime(absolute//12, absolute%12+1, 1).strftime('%B %Y'))
        if references:
            text += '\nTime references: ' + '; '.join(references)
    return text
