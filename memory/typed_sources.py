"""Typed source indices. Reuse original text and vectors without fact generation."""
from contextlib import closing
import json
import sqlite3
import numpy as np
from .provenance import source_id
from .retrieval import bm25
from .vector_writer import VectorWriter


TYPES = frozenset({"profile", "relationship", "rule", "event"})


class SourceBuilder:
    def __init__(self, db_path, memory_type):
        if memory_type not in TYPES: raise ValueError("Unsupported memory type")
        self.memory_type = memory_type
        self.table = memory_type + "_sources"
        self.db_path = str(db_path)
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute(f'''CREATE TABLE IF NOT EXISTS {self.table}(
                user_id TEXT NOT NULL, source_id TEXT NOT NULL,
                request_id TEXT NOT NULL, session_id TEXT NOT NULL,
                message_index INTEGER NOT NULL, context_source_ids TEXT NOT NULL,
                classification_score REAL,
                PRIMARY KEY(user_id,source_id))''')
            db.execute(f'CREATE INDEX IF NOT EXISTS {self.memory_type}_source_request ON {self.table}(user_id,request_id)')

    def write(self, payload, selected_indices, score=None):
        indices = sorted(set(selected_indices))
        if not indices or any(type(i) is not int or not 0 <= i < len(payload.messages) for i in indices):
            raise ValueError('Invalid typed source indices')
        if score is not None and not 0 <= score <= 1:
            raise ValueError('Invalid classification score')
        ids=[]
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute('BEGIN IMMEDIATE')
            for i in indices:
                sid=source_id(payload.user_id,payload.request_id,i)
                exists=db.execute('''SELECT 1 FROM rag_sources s JOIN rag_memories m ON m.id=s.memory_id
                    WHERE m.user_id=? AND m.request_id=? AND s.source_id=?''',
                    (payload.user_id,payload.request_id,sid)).fetchone()
                if exists is None:
                    raise ValueError('Typed source must be persisted before indexing')
                context=[source_id(payload.user_id,payload.request_id,j) for j in range(max(0,i-2),i)]
                db.execute(f'INSERT OR IGNORE INTO {self.table} VALUES (?,?,?,?,?,?,?)',
                    (payload.user_id,sid,payload.request_id,payload.session_id,i,json.dumps(context),score))
                ids.append(sid)
        return ids


class SourceRetriever:
    def __init__(self, db_path, memory_type, embedder=None):
        if memory_type not in TYPES: raise ValueError("Unsupported memory type")
        self.memory_type = memory_type
        self.table = memory_type + "_sources"
        self.db_path=str(db_path)
        self.embedder=embedder

    def search(self, user_id, query, *, limit=10, session_id=None, fallback=True):
        """Hybrid rank original chunks, adding a soft memory-type vote.

        fallback=False restricts to classified evidence. With fallback=True,
        all user sources compete so a classifier omission is not permanent loss.
        Timestamps are message timestamps, never claimed to be event dates.
        """
        if not query.strip() or not 1 <= limit <= 100:
            raise ValueError('Nonempty query and limit 1..100 required')
        with closing(sqlite3.connect(self.db_path)) as db:
            db.row_factory=sqlite3.Row
            sql=f'''SELECT m.*,s.source_id,s.speaker,s.role,e.context_source_ids,e.classification_score,
                e.source_id AS typed_source FROM rag_memories m JOIN rag_sources s ON s.memory_id=m.id
                LEFT JOIN {self.table} e ON e.user_id=m.user_id AND e.source_id=s.source_id WHERE m.user_id=?'''
            args=[user_id]
            if session_id is not None:
                sql+=' AND m.session_id=?';args.append(session_id)
            if not fallback:sql+=' AND e.source_id IS NOT NULL'
            rows=db.execute(sql+' ORDER BY m.rowid',args).fetchall()
            if not rows:return []
            lexical=bm25([r['content'] for r in rows],query)
            scores=np.zeros(len(rows))
            pool=set()
            for rank,i in enumerate(sorted((i for i in range(len(rows)) if lexical[i]>0),key=lambda i:-lexical[i]),1):
                scores[i]+=1/(60+rank);pool.add(i)
            if self.embedder is not None:
                vector=VectorWriter.vectors(self.embedder.queries([query]),1)[0]
                if any(r['dimension']!=len(vector) for r in rows):raise ValueError('Embedding dimension mismatch')
                dense=np.stack([np.frombuffer(r['vector'],dtype='<f4') for r in rows])@vector
                for rank,i in enumerate(sorted(range(len(rows)),key=lambda i:-dense[i]),1):
                    scores[i]+=1/(60+rank);pool.add(i)
            ranked=sorted(pool,key=lambda i:-scores[i])
            type_rank=0
            for i in ranked:
                if rows[i]['typed_source']:
                    type_rank+=1;scores[i]+=.25/(60+type_rank)
            out=[]
            for i in sorted(pool,key=lambda i:(-scores[i],i))[:limit]:
                r=rows[i]
                context=[]
                for sid in json.loads(r['context_source_ids'] or '[]'):
                    context.extend(dict(x) for x in db.execute('''SELECT m.id,m.content,s.source_id,s.speaker,m.timestamp
                        FROM rag_memories m JOIN rag_sources s ON s.memory_id=m.id
                        WHERE m.user_id=? AND s.source_id=? ORDER BY m.chunk_index''',(user_id,sid)))
                out.append(dict(id=r['id'],source_id=r['source_id'],content=r['content'],score=float(scores[i]),
                    speaker=r['speaker'],role=r['role'],message_timestamp=r['timestamp'],
                    session_id=r['session_id'],memory_type=self.memory_type if r['typed_source'] else 'fallback',
                    classification_score=r['classification_score'],context=context))
            return out
