"""Server-configured basic masking, not a complete authorization/PII classifier."""
import hashlib
import json
import re
from contextlib import closing


class DisclosurePolicy:
    PATTERNS = [
        ('credential', re.compile(r'(?i)\b(?:api[_ -]?key|password|passwd|secret|token)\s*[:=]\s*["\']?[^\s,;"\']+')),
        ('email', re.compile(r'(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b')),
        ('phone', re.compile(r'(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)|(?<!\d)(?:\+1[- .]?)?\(?\d{3}\)?[- .]\d{3}[- .]\d{4}(?!\d)')),
        ('identity', re.compile(r'(?<!\w)\d{17}[\dXx](?!\w)')),
    ]

    def __init__(self, cfg, backend):
        self.mode = cfg.get('RAG_DISCLOSURE_MODE','off')
        if self.mode not in ('off','mask'): raise ValueError('Invalid disclosure mode')
        self.backend=backend
        if self.mode=='mask':
            with closing(backend.connect()) as db,db:
                db.execute('CREATE TABLE IF NOT EXISTS rag_disclosure_views '
                           '(user_id TEXT, view_id TEXT, source_id TEXT, policy TEXT, PRIMARY KEY(user_id,view_id))')

    def mask(self, text):
        if self.mode!='mask': return text
        for kind,pattern in self.PATTERNS:
            text=pattern.sub('[REDACTED:'+kind+']',text)
        return text

    def rows(self, rows):
        return [dict(r,content=self.mask(r['content']),speaker=self.mask(r['speaker']) if r.get('speaker') else r.get('speaker')) for r in map(dict,rows)] if self.mode=='mask' else rows

    def result(self, user, result):
        if self.mode!='mask': return result
        hits=[]
        for raw in result['data']:
            h=dict(raw); h['content']=self.mask(h['content'])
            if '_retrieval_text' in h: h['_retrieval_text']=self.mask(h['_retrieval_text'])
            if 'context' in h:
                # Extra provenance context is not needed to disclose raw personal fields.
                h['context']=[]
            if '[REDACTED:' in h['content'] and not h['id'].startswith(('view_','bundle_')):
                source=h['id']
                h['id']='view_'+hashlib.sha256(json.dumps([user,source,'mask-v1',h['content']],ensure_ascii=False).encode()).hexdigest()
                with closing(self.backend.connect()) as db,db:
                    db.execute('INSERT OR IGNORE INTO rag_disclosure_views VALUES (?,?,?,?)',(user,h['id'],source,'mask-v1'))
            hits.append(h)
        return dict(result,data=hits)
