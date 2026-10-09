from typing import Any
from pydantic import Field
from .llm import LLM
from .models import StrictModel

class AnnotationInput(StrictModel):
    source_id: str = Field(min_length=1, max_length=256)
    content: str = Field(min_length=1, max_length=32000)
    timestamp: int | None = Field(default=None, ge=0)

class SpanAnnotation(StrictModel):
    source_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=2000)
    label: str = Field(min_length=1, max_length=128)
    confidence: float = Field(ge=0, le=1)

class AnnotationBatch(StrictModel):
    annotations: list[SpanAnnotation] = Field(default_factory=list, max_length=500)

class LLMAnnotator:
    def __init__(self, labels: dict[str, str], *, instruction: str = '', llm=None):
        if not labels or len(labels) > 30 or any(not k.strip() or not v.strip() for k,v in labels.items()):
            raise ValueError('labels must be a nonempty mapping')
        self.labels, self.instruction, self.llm = dict(labels), instruction.strip(), llm or LLM()

    def annotate(self, sources: list[AnnotationInput | dict[str, Any]]) -> list[dict[str, Any]]:
        items = [AnnotationInput.model_validate(x) for x in sources]
        if len(items) > 100 or len({x.source_id for x in items}) != len(items):
            raise ValueError('sources must contain at most 100 unique source IDs')
        payload = {'sources': [x.model_dump() for x in items], 'labels': self.labels}
        prompt = ('Annotate supplied source texts. Return ONLY exact contiguous spans copied from a source, '
                  'with one configured label and confidence 0 to 1. Do not paraphrase, summarize, translate, '
                  'infer, or create facts. Do not annotate when no label applies. source_id must be copied exactly.\n'
                  + '\n'.join(f'- {k}: {v}' for k,v in self.labels.items()))
        if self.instruction: prompt += '\nAdditional rules:\n' + self.instruction
        result = AnnotationBatch.model_validate(self.llm.complete(prompt, payload, AnnotationBatch))
        by_id = {x.source_id: x for x in items}; seen=set(); out=[]
        for ann in result.annotations:
            source = by_id.get(ann.source_id)
            if source is None: raise ValueError(f'Unknown annotation source: {ann.source_id}')
            if ann.label not in self.labels: raise ValueError(f'Unknown annotation label: {ann.label}')
            pos=[]; start=0
            while True:
                index=source.content.find(ann.text,start)
                if index < 0: break
                pos.append(index); start=index+1
            if len(pos) != 1: raise ValueError('Annotation text must occur exactly once in its source')
            key=(ann.source_id,ann.text,ann.label)
            if key in seen: continue
            seen.add(key)
            out.append(dict(ann.model_dump(), start=pos[0], end=pos[0]+len(ann.text), source_timestamp=source.timestamp))
        return out

TIME_LABELS = {
    'absolute_date': 'An explicit calendar date, month, or year.',
    'relative_time': 'A relative expression such as yesterday, last week, or two months ago.',
    'time_range': 'An explicit duration or interval such as from June to August.',
    'temporal_relation': 'A relation such as before, after, during, later, or until.',
}

class TimeAnnotator(LLMAnnotator):
    def __init__(self, llm=None):
        super().__init__(TIME_LABELS, instruction=(
            'Annotate only the time expression itself, not the event around it. Keep punctuation exactly. '
            'A message timestamp is metadata, not a span. Do not annotate vague words such as recently '
            'unless they constrain an event.'), llm=llm)
