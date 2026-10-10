"""Published source versions; ContextVar is copied into parallel search branches."""
from contextvars import ContextVar
from contextlib import closing

READ_VERSION = ContextVar('memory_read_version', default=None)


class ReadVersions:
    def __init__(self, backend):
        self.backend = backend
        with closing(backend.connect()) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS rag_publications '
                       '(version INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, '
                       'request_id TEXT NOT NULL, UNIQUE(user_id, request_id))')
            if not db.execute("SELECT 1 FROM rag_meta WHERE key='publication_migrated'").fetchone():
                tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                for r in db.execute('SELECT user_id,request_id FROM rag_requests ORDER BY rowid').fetchall():
                    pending = False
                    if 'rag_memory_queue' in tables:
                        q = db.execute('SELECT status FROM rag_memory_queue WHERE user_id=? AND request_id=?',tuple(r)).fetchone()
                        pending = q is not None and q[0] != 'completed'
                    if not pending:
                        db.execute('INSERT OR IGNORE INTO rag_publications(user_id,request_id) VALUES (?,?)',tuple(r))
                db.execute("INSERT INTO rag_meta VALUES ('publication_migrated','1')")

    def publish(self, user, request):
        with closing(self.backend.connect()) as db, db:
            db.execute('INSERT OR IGNORE INTO rag_publications(user_id,request_id) VALUES (?,?)',(user,request))

    def latest(self, user):
        with closing(self.backend.connect()) as db:
            return db.execute('SELECT COALESCE(MAX(version),0) FROM rag_publications WHERE user_id=?',(user,)).fetchone()[0]

    def visible(self, user, rows):
        current = READ_VERSION.get()
        version = current[1] if current and current[0] == user else self.latest(user)
        with closing(self.backend.connect()) as db:
            requests = {r[0] for r in db.execute('SELECT request_id FROM rag_publications WHERE user_id=? AND version<=?',(user,version))}
        return [r for r in rows if r['request_id'] in requests]
