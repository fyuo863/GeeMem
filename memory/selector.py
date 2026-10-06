"""LLM-based option likelihood scoring.

This module is intentionally above the atomic retriever. It scores supplied
options for a question, but does not retrieve memories or generate a final
answer. Credentials and model settings come exclusively from the root .env.
"""

from typing import Any, Literal
import time
import httpx

from pydantic import Field

from .llm import LLM, LLMError
from .models import StrictModel


class OptionScore(StrictModel):
    option: str = Field(min_length=1, max_length=4000)
    score: float = Field(ge=0, le=1)


class SelectionResult(StrictModel):
    scores: list[OptionScore] = Field(min_length=1, max_length=100)


class ProfileGateResult(StrictModel):
    decision: Literal['profile_candidate', 'episodic_only', 'third_party', 'uncertain']
    profile_type: Literal['identity', 'residence', 'preference', 'constraint', 'occupation',
                          'background', 'relationship', 'none']
    stability: Literal['stable', 'temporary', 'unknown']
    confidence: float = Field(ge=0, le=1)
    evidence: str = Field(min_length=1, max_length=1000)
    reason: str = Field(min_length=1, max_length=1000)


class JudgeLabel(StrictModel):
    name: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=1000)
    examples: list[str] = Field(default_factory=list, max_length=20)


class JudgeConfig(StrictModel):
    name: str = Field(min_length=1, max_length=128)
    subject: str = Field(min_length=1, max_length=256)
    criteria: str = Field(min_length=1, max_length=4000)
    labels: list[JudgeLabel] = Field(min_length=2, max_length=20)
    require_evidence: bool = True
    max_attempts: int = Field(default=3, ge=1, le=5, strict=True)


class JudgeResult(StrictModel):
    label: str = Field(min_length=1, max_length=128)
    scores: list[OptionScore] = Field(min_length=2, max_length=20)
    confidence: float = Field(ge=0, le=1)
    evidence: str = Field(default='', max_length=1000)
    reason: str = Field(min_length=1, max_length=1000)


def _ground_quote(evidence: str, text: str) -> str:
    """Remove only quote wrappers; never rewrite or fuzzy-match source content."""
    if evidence.strip() and evidence in text:
        return evidence
    candidate = evidence.strip()
    pairs = {'"': '"', "'": "'", '“': '”', '‘': '’', '「': '」', '『': '』'}
    while candidate:
        if candidate in text:
            return candidate
        if len(candidate) < 2 or pairs.get(candidate[0]) != candidate[-1]:
            break
        candidate = candidate[1:-1].strip()
    raise ValueError('Judge evidence must be a nonblank quote copied from the judged text')


class OptionSelector:
    """Score each caller-supplied option for one question."""

    def __init__(self, llm=None):
        self.llm = llm or LLM()

    def score(self, question: str, options: list[str]) -> SelectionResult:
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Question must be nonblank')
        if not isinstance(options, list) or not 1 <= len(options) <= 100:
            raise ValueError('Options must contain between 1 and 100 items')
        cleaned = [option.strip() if isinstance(option, str) else option for option in options]
        if any(not isinstance(option, str) or not option for option in cleaned):
            raise ValueError('Options must be nonblank strings')
        if len(set(cleaned)) != len(cleaned):
            raise ValueError('Options must be unique')
        result = self.llm.complete(
            '''Score the supplied answer options for the question. Return one score
for every option, preserving the exact option text. A score is the option's
relative likelihood of being the correct answer given the question and supplied
context. Use 0 for clearly impossible and 1 for highly likely. Do not invent,
rewrite, merge, or omit options. Do not answer the question outside the schema.''',
            {'question': question.strip(), 'options': cleaned}, SelectionResult)
        returned = [item.option for item in result.scores]
        if returned != cleaned:
            raise ValueError('LLM selector must return options in the supplied order')
        return result


class ConfigurableJudge:
    """Generic configurable classifier for messages about any subject."""

    def __init__(self, config: JudgeConfig | dict[str, Any], llm=None):
        self.config = config if isinstance(config, JudgeConfig) else JudgeConfig.model_validate(config)
        names = [label.name for label in self.config.labels]
        if len(set(names)) != len(names):
            raise ValueError('Judge labels must be unique')
        # This layer owns retries; avoid multiplying transport and judge attempts.
        self.llm = llm or LLM(max_attempts=1)

    def judge(self, text: str, *, context: str = '', metadata: dict[str, Any] | None = None) -> JudgeResult:
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Judged text must be nonblank')
        if not isinstance(context, str) or (metadata is not None and not isinstance(metadata, dict)):
            raise ValueError('Invalid judge context or metadata')
        for attempt in range(self.config.max_attempts):
            try:
                return self._judge_once(text, context=context, metadata=metadata, retry=attempt > 0)
            except (LLMError, ValueError) as exc:
                cause = exc.__cause__
                permanent = isinstance(exc, LLMError) and str(exc) == 'LLM_API_KEY is not configured'
                if isinstance(cause, httpx.HTTPStatusError):
                    status = cause.response.status_code
                    permanent = status < 500 and status not in (408, 429)
                if permanent or attempt + 1 == self.config.max_attempts:
                    raise
                time.sleep(0.25 * 2 ** attempt)

    def _judge_once(self, text, *, context, metadata, retry=False):
        labels = [label.name for label in self.config.labels]
        label_text = '\n'.join(
            f"- {label.name}: {label.description}"
            + (f" Examples: {' | '.join(label.examples)}" if label.examples else '')
            for label in self.config.labels)
        retry_instruction = (' Previous attempt failed. Recheck schema, label order and exact source quotation.'
                             if retry else '')
        result = self.llm.complete(
            f'''You are the {self.config.name} judgement module for {self.config.subject}.
Classify the supplied text using exactly one configured label. Apply these criteria:
{self.config.criteria}
Configured labels:
{label_text}
Return one score for every label in the supplied order. Scores are relative
likelihoods from 0 to 1, and label must be one of the configured names. Do not
invent facts. If require_evidence is true, copy a nonblank contiguous substring
from text into evidence. Do not add quotation marks, explanations, or change
punctuation. If require_evidence is false, evidence may be empty.''' + retry_instruction,
            {'subject': self.config.subject, 'text': text.strip(), 'context': context.strip(),
             'metadata': metadata or {}, 'labels': labels,
             'require_evidence': self.config.require_evidence}, JudgeResult)
        if result.label not in labels:
            raise ValueError('Judge returned an unknown label')
        returned = [item.option for item in result.scores]
        if returned != labels:
            raise ValueError('Judge must score every configured label in order')
        if self.config.require_evidence:
            result = result.model_copy(update={'evidence': _ground_quote(result.evidence, text)})
        return result


class ProfileGate:
    """Decide whether one newly added conversation message belongs in a user profile."""

    def __init__(self, llm=None):
        self.llm = llm or LLM()

    def classify(self, message: str, *, role: str = 'user', speaker: str | None = None,
                 profile_context: str = '') -> ProfileGateResult:
        if not isinstance(message, str) or not message.strip():
            raise ValueError('Message must be nonblank')
        if role not in ('user', 'assistant', 'system', 'tool'):
            raise ValueError('Unsupported message role')
        result = self.llm.complete(
            '''Classify one newly added conversation message for a user-profile builder.
Use profile_candidate for a stable fact explicitly about the current user, including
identity, residence, occupation, background, preference, or constraint. Use
episodic_only for a one-time event that should remain ordinary memory. Use
third_party for facts about another person. Use uncertain for tentative or
ambiguous statements. Do not infer facts that are absent. evidence must be an
exact contiguous quote from the supplied message. profile_type=none for
episodic_only, third_party, or uncertain unless a stable type is explicit.''',
            {'message': message.strip(), 'role': role, 'speaker': speaker or '',
             'profile_context': profile_context.strip()}, ProfileGateResult)
        if result.evidence not in message:
            raise ValueError('Profile gate evidence must be copied from the message')
        if result.decision != 'profile_candidate' and result.profile_type != 'none':
            raise ValueError('Non-profile decisions must use profile_type=none')
        return result
