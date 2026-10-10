"""One small phase-selection task replaces the general review for plan changes."""
import re
from pydantic import Field
from .models import StrictModel

class Phases(StrictModel):
    original_plan: int = Field(ge=-1, le=15)
    change: int = Field(ge=-1, le=15)
    outcome: int = Field(ge=-1, le=15)

def matches(question):
    return len(question) <= 1200 and all(re.search(p, question, re.I) for p in (
        r'计划|\bplan\b', r'变化|改变|变更|\bchang(?:e|ed)\b', r'最终|结果|\b(outcome|finally|final)\b'))

def review(planner, question, evidence, timeout):
    from .multihop import Review, Support, Query
    candidates=[e for e in evidence if len(e['content']) <= 600][:16]
    selected=planner.complete('''Select one source index for EACH phase of the SAME person's SAME event:
original_plan: the original intention before the change;
change: cancellation, replacement, or revision;
outcome: what actually happened, not another plan.
Prefer a standalone original-plan source over a later cancellation mentioning it.
Use -1 for a missing phase. Do not select a different person's similar event.
Return indices only; do not write facts, quotes or an answer.''',
        {'question':question,'sources':[{'index':i,'content':e['content']} for i,e in enumerate(candidates)]}, Phases, timeout)
    supports={};missing=[]
    for phase,index in selected.model_dump().items():
        if index == -1:
            missing.append(phase)
        elif index >= len(candidates):
            raise ValueError('Stage selection outside candidate set')
        else:
            source=candidates[index]
            supports[source['id']]=Support(source_id=source['id'],quote=source['content'],needed_for=phase)
    queries=[]
    if missing:
        hint=missing[0].replace('_',' ')
        queries=[Query(query=question[:350]+'\nFind the '+hint+' evidence.', source_id='__question__',bridge=question[:80])]
    return Review(sufficient=not missing,supports=list(supports.values()),missing=', '.join(missing),queries=queries)
