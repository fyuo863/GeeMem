"""Versioned semantic tags; no answers or benchmark labels enter extraction."""
import re
import unicodedata
from pydantic import Field
from .models import StrictModel
from .llm import LLM, LLMError


class TagList(StrictModel):
    tags: list[str] = Field(max_length=8)


class TagBatch(StrictModel):
    items: list[TagList] = Field(min_length=1, max_length=16)


def normalize_tags(values):
    result = set()
    for value in values:
        tag = re.sub(r"\s+", " ", unicodedata.normalize('NFKC', value).casefold()).strip(' .,:;')
        if tag and len(tag) <= 80:
            result.add(tag)
    return sorted(result)


class SemanticTagger:
    identity = 'semantic-tags-v1:gpt-4o-mini'
    def __init__(self):
        self.llm = LLM()
        if self.llm.model != 'gpt-4o-mini':
            raise ValueError('Tags require LLM_MODEL=gpt-4o-mini')

    def extract(self, texts, query=False):
        result = []
        for start in range(0, len(texts), 16):
            batch = texts[start:start+16]
            response = self.llm.complete(
                'Extract 3 to 8 short reusable retrieval tags per input text, in exactly input order. '
                'Use lowercase English singular nouns or short noun phrases. Include explicit named entities '
                'and broad semantic topics (e.g. education, family, health, travel, career, sport). '
                'Use consistent common labels, not sentences, dates, question words or generic chat tags. '
                'Only describe topics supported by each text. Do not infer facts, answer questions, '
                'resolve unknown pronouns or copy tags from another item. Empty tags are allowed when '
                'there is no substantive topic. For queries tag the information being requested, not an invented answer.',
                dict(kind='query' if query else 'memory', texts=batch), TagBatch)
            if len(response.items) != len(batch):
                raise LLMError('Tag response count mismatch')
            result.extend(normalize_tags(item.tags) for item in response.items)
        return result
