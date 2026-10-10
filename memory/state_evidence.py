"""Bounded, opt-in query-time state evidence selection; never rewrites memories."""
import logging
import re
import time
from typing import Literal
from pydantic import Field
from .models import StrictModel
from .multihop import Planner

logger = logging.getLogger(__name__)


class EvidenceChoice(StrictModel):
    index: int = Field(ge=0, strict=True)
    quote: str = Field(min_length=3, max_length=600)
    role: Literal['primary', 'context'] = 'primary'


class StateSelection(StrictModel):
    selected: list[EvidenceChoice] = Field(max_length=8)


PROMPT = '''Select source evidence needed for the question, not an answer.
For current values, select explicit effective corrections for the SAME person and
attribute. A later message is not automatically a newer fact. Future intentions,
questions, hearsay and another person's values do not replace current facts.
Use reference_time only when supplied; never invent today's date. If conflicting
claims cannot be resolved, select both. For rules select both the default and its
applicable exception. For historical questions preserve the requested historical
state. Select all necessary sources, not just a concluding sentence. Do not infer
missing facts. Return only candidate indices and short exact quotes from content.
If no reliable selection is possible return selected=[].'''

PROMPT += ''' Label each choice primary or context. Primary directly establishes
the requested value/constraint. An explicitly replaced old value or retrospective
old residence is context, NOT primary for a current-state question. Both unresolved
conflicting claims are primary. A default rule and its applicable exception are
both primary. Keep necessary relationship links primary for a chain question.'''


class StateEvidenceSelector:
    def __init__(self, cfg, model=None):
        self.mode = cfg.get('RAG_STATE_EVIDENCE_MODE', 'off')
        if self.mode not in ('off', 'on'):
            raise ValueError('Invalid state evidence mode')
        self.model = model or (Planner(cfg) if self.mode == 'on' else None)

    def applies(self, query):
        return self.mode == 'on' and len(query) <= 1500 and bool(re.search(
            r'现在|目前|当前|更正|冲突|规则|例外|最新|有效|\b(current|currently|now|latest|correction|conflict|rule|exception|still)\b',
            query, re.I))

    def select(self, payload, hits, trace):
        start = time.perf_counter()
        stats = dict(status='skipped', candidates=0, selected=0, calls=0)
        result = hits
        try:
            if not self.applies(payload.query) or not hits:
                return result
            # Skip oversized sources rather than truncate away a late correction.
            candidates = [(i, h) for i, h in enumerate(hits[:16]) if len(h['content']) <= 600]
            stats['candidates'] = len(candidates)
            if not candidates:
                return result
            stats['calls'] = 1
            decision = self.model.complete(PROMPT, dict(
                question=payload.query, reference_time=getattr(payload, 'reference_time', None),
                candidates=[dict(index=i, content=h['content'], created_at=h.get('created_at'))
                            for i, h in candidates]), StateSelection, timeout=20)
            allowed = dict(candidates)
            chosen = {}
            for item in decision.selected:
                if item.index not in allowed or item.quote not in allowed[item.index]['content']:
                    raise ValueError('Ungrounded selection')
                if item.index in chosen:
                    raise ValueError('Duplicate selection')
                chosen[item.index] = item.role
            # Program owns ordering. Within each tier preserve original relevance.
            order = {'primary': 0, 'context': 1}
            result = [h for i, h in sorted(enumerate(hits), key=lambda pair: order.get(chosen.get(pair[0]), 2))]
            if chosen:
                # Preserve descending score contract while retaining score scale.
                scores = sorted((h['score'] for h in hits), reverse=True)
                result = [dict(h, score=score) for h, score in zip(result, scores)]
            stats.update(status='selected' if chosen else 'abstained', selected=len(chosen))
        except Exception:
            # Fail open, without logging exception/provider bodies or source text.
            stats['status'] = 'fallback'
            result = hits
        finally:
            stats['seconds'] = round(time.perf_counter()-start, 4)
            trace['state_evidence'] = stats
            logger.info('state_evidence status=%s candidates=%d selected=%d calls=%d seconds=%.4f',
                        stats['status'], stats['candidates'], stats['selected'], stats['calls'], stats['seconds'])
        return result
