"""Source coverage reporting and bounded structured-output repair."""
from typing import Literal
from pydantic import Field, create_model
from .models import StrictModel
from .llm import LLM


class Coverage(StrictModel):
    message_index: int = Field(strict=True)
    disposition: Literal['extracted', 'not_applicable', 'unresolved']
    reason: str


class AuditedLLM:
    def __init__(self, llm=None):
        self.llm = llm or LLM()
        self.audit = []

    def complete(self, instruction, payload, schema):
        audited = create_model('Audited'+schema.__name__, __base__=schema,
                               coverage=(list[Coverage], ...))
        expected = {m['message_index'] for m in payload['messages']}
        prompt = instruction + '''
Audit EVERY message separately in coverage, retaining its original message_index.
For each substantive fact of this memory type, include a record even when it is
not the classifier's focus. Use speaker names instead of merging speakers into I.
not_applicable means no facts of THIS type; unresolved means ambiguity remains.
Do not call a message extracted unless a returned record cites it. Preserve all
independent facts, especially later refinements and short confirmations.
Certainty describes the source's assertion, not your confidence: explicit facts
are confirmed; maybe/possibly are uncertain. Context-only messages explain facts.
Never treat a person's question as their own factual assertion.'''
        for attempt in range(2):
            result = self.llm.complete(prompt, payload, audited)
            data = result.model_dump()
            coverage = data.pop('coverage')
            indices = [c['message_index'] for c in coverage]
            cited = {i for values in data.values() if isinstance(values,list)
                     for item in values for i in item.get('message_indices',[])}
            valid = set(indices)==expected and len(indices)==len(expected) and cited <= expected
            valid = valid and all(c['disposition']!='extracted' or c['message_index'] in cited for c in coverage)
            self.audit.append(dict(attempt=attempt+1, valid=valid, coverage=coverage,
                                   candidate=data))
            if valid:
                return schema.model_validate(data)
            prompt += '\nRepair coverage: enumerate each input index once, and cite only supplied indices. Return the full extraction.'
        raise ValueError('Builder coverage/index validation failed after repair')
