"""Additive time-mention index; raw memories remain the only evidence returned."""
from contextlib import closing
import json

from .annotator import TimeAnnotator
from .llm import LLM
from .provenance import source_id
from .temporal import TemporalNormalizer


class TemporalIndex:
    def __init__(self, connect, annotate=False, annotator=None):
        self.connect = connect
        self.annotator = annotator or (TimeAnnotator(LLM(max_attempts=2)) if annotate else None)
        with closing(connect()) as db, db:
            db.execute('''CREATE TABLE IF NOT EXISTS rag_time_mentions(
                user_id TEXT NOT NULL, source_id TEXT NOT NULL, version TEXT NOT NULL,
                mentions TEXT NOT NULL, status TEXT NOT NULL,
                PRIMARY KEY(user_id,source_id,version))''')

    def write(self, payload):
        calls = 0
        for i, message in enumerate(payload.messages):
            sid = source_id(payload.user_id, payload.request_id, i)
            version = TemporalNormalizer.VERSION + ('-llm' if self.annotator else '-rules')
            with closing(self.connect()) as db:
                if db.execute('SELECT 1 FROM rag_time_mentions WHERE user_id=? AND source_id=? AND version=?',
                              (payload.user_id,sid,version)).fetchone():
                    continue
            stamp = message.timestamp if message.timestamp is not None else getattr(payload,'session_timestamp',None)
            mentions = TemporalNormalizer.mentions(message.content, stamp)
            status = 'rules'
            # Small-model pressure budget: at most two calls per Add, one
            # source <= 6000 chars per call, two transport attempts per call.
            if self.annotator and mentions:
                status = 'budget_skipped'
                if calls < 2 and len(message.content) <= 6000:
                    calls += 1
                    try:
                        selected = self.annotator.annotate([dict(source_id=sid,content=message.content,timestamp=stamp)])
                        mentions = [dict(text=a['text'], char_start=a['start'],char_end=a['end'],
                                         label=a['label'],confidence=a['confidence'],
                                         **TemporalNormalizer.normalize(a['text'],stamp)) for a in selected]
                        status = 'annotated'
                    except Exception:
                        # Annotation is optional enrichment. Never lose source
                        # evidence or persist exception strings with secrets.
                        status = 'annotation_failed_rules_retained'
            with closing(self.connect()) as db, db:
                db.execute('INSERT OR IGNORE INTO rag_time_mentions VALUES (?,?,?,?,?)',
                           (payload.user_id,sid,version,json.dumps(mentions,ensure_ascii=False),status))

    def enrich(self, user_id, rows):
        with closing(self.connect()) as db:
            records = db.execute('SELECT source_id,version,mentions FROM rag_time_mentions WHERE user_id=?',
                                 (user_id,)).fetchall()
        by_id = {}
        for r in sorted(records, key=lambda r:r['version'].endswith('-llm')):
            by_id[r['source_id']] = json.loads(r['mentions'])
        enriched = []
        for row in rows:
            item = dict(row)
            start,end = item.get('char_start'),item.get('char_end')
            item['time_mentions'] = [m for m in by_id.get(item.get('source_id'),[]) if
                                    start is not None and end is not None and m['char_start']>=start and m['char_end']<=end]
            enriched.append(item)
        return enriched
