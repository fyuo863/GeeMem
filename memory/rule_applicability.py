"""Bounded rule selection, separate from evidence sufficiency review."""
import re
from typing import Literal
from pydantic import Field
from .models import StrictModel

TRIGGER = re.compile(r'规则|例外|审批|报销|退款|\b(rule|exception|approval|refund|policy|require)\b', re.I)

class RuleVote(StrictModel):
    index: int
    status: Literal['applicable', 'inapplicable', 'unknown']
    quote: str = Field(max_length=600)

class RuleVotes(StrictModel):
    votes: list[RuleVote] = Field(max_length=12)

PROMPT = '''Classify each rule's applicability to the question, NOT topical similarity.
Return one vote per index. A specific exception is inapplicable when its condition
contradicts the stated scenario (ordinary commuting is not an emergency).
For a broad request to list all rules/exceptions, all relevant conditions can apply.
If the question leaves a necessary condition unspecified, mark unknown.
For inapplicable copy the exact condition-bearing quote from that candidate.
Unrelated facts/rules for a different activity are also inapplicable; quote their topic.
Do not reject general defaults that explain the exception. Do not infer unstated facts.'''

def select_rules(question, hits, planner, timeout):
    candidates = [(i, h) for i, h in enumerate(hits) if len(h['content']) <= 900][:12]
    if not TRIGGER.search(question) or not candidates or len(question) > 1500:
        return set(), {'status': 'skipped'}
    result = planner.complete(PROMPT, {'question': question, 'candidates': [
        {'index': i, 'content': h['content']} for i, h in candidates]}, RuleVotes, timeout)
    lookup = dict(candidates)
    indices = [v.index for v in result.votes]
    if len(set(indices)) != len(indices) or set(indices) != set(lookup):
        raise ValueError('Incomplete rule applicability votes')
    rejected = set()
    for v in result.votes:
        if v.status == 'inapplicable':
            if not v.quote.strip() or v.quote not in lookup[v.index]['content']:
                raise ValueError('Ungrounded applicability condition')
            rejected.add(lookup[v.index]['id'])
    return rejected, {'status': 'checked', 'votes': result.model_dump()['votes'],
                      'checked_ids': [h['id'] for _,h in candidates],
                      'rejected_ids': sorted(rejected)}
