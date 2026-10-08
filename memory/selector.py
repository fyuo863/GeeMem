"""LLM-based option likelihood scoring.

This module is intentionally above the atomic retriever. It scores supplied
options for a question, but does not retrieve memories or generate a final
answer. Credentials and model settings come exclusively from the root .env.
"""

from typing import Any, Literal
import time
import httpx

from pydantic import Field, StrictInt, StrictBool

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


class MultiLabelConfig(JudgeConfig):
    """Same generic criteria/labels, with optional mutually exclusive solo labels."""
    require_evidence: bool = False
    exclusive_labels: list[str] = Field(default_factory=list, max_length=20)


class LabelAssessment(StrictModel):
    label: str = Field(min_length=1, max_length=128)
    selected: StrictBool
    score: float = Field(ge=0, le=1)
    message_indices: list[StrictInt] = Field(default_factory=list, max_length=200)
    reason: str = Field(min_length=1, max_length=1000)


class MultiLabelResult(StrictModel):
    assessments: list[LabelAssessment] = Field(min_length=2, max_length=20)

    @property
    def labels(self):
        return [item.label for item in self.assessments if item.selected]


class MultiLabelJudge(ConfigurableJudge):
    """Configurable multi-label routing; indices refer to program-indexed inputs."""

    def __init__(self, config, llm=None):
        config = config if isinstance(config, MultiLabelConfig) else MultiLabelConfig.model_validate(config)
        super().__init__(config, llm)
        names = {label.name for label in config.labels}
        if not set(config.exclusive_labels) <= names:
            raise ValueError('Unknown exclusive label')
        if config.require_evidence:
            raise ValueError('Multi-label judge uses message indices, not evidence quotations')

    def judge(self, messages: list[dict], *, context='') -> MultiLabelResult:
        if not isinstance(messages, list) or not 1 <= len(messages) <= 200:
            raise ValueError('Supply between 1 and 200 messages')
        indexed = []
        for i, message in enumerate(messages):
            if not isinstance(message, dict) or not isinstance(message.get('content'), str) or not message['content'].strip():
                raise ValueError('Each message requires nonblank content')
            indexed.append(dict(message, index=i))
        # Reuse the bounded retry policy, including permanent HTTP failures.
        return super().judge('indexed messages', context=context, metadata={'messages': indexed})

    def _judge_once(self, text, *, context, metadata, retry=False):
        labels = [label.name for label in self.config.labels]
        instruction = (
            'Classify the indexed messages using all applicable labels. Apply the configured criteria. '
            'Return one assessment for EVERY label in the supplied order. Scores are independent '
            'applicability scores in [0,1], not a distribution and need not sum to 1. '
            'Set selected=true for every applicable label; select at least one label. '
            'A selected exclusive label must be the ONLY selected label. '
            'For each selected label cite nonempty message_indices supporting it, including context '
            'messages required to interpret short answers. Unselected labels must have empty indices. '
            'Use only supplied integer indices, without duplicates. Consider roles, negation and '
            'uncertainty. Do not turn assistant suggestions or questions into user facts. '
            'You only route messages, do not extract facts or execute deletion requests. '
        )
        instruction += '\nConfigured criteria:\n' + self.config.criteria
        instruction += '\nConfigured labels:\n' + '\n'.join(
            f'{label.name}: {label.description}'
            + (f' Examples: {" | ".join(label.examples)}' if label.examples else '')
            for label in self.config.labels)
        instruction += '\nExclusive labels: ' + ', '.join(self.config.exclusive_labels) + '\n'
        if retry:
            instruction += 'Previous attempt failed: check label order, exclusivity and valid message indices. '
        result = self.llm.complete(instruction, {
            'name': self.config.name, 'subject': self.config.subject,
            'labels': labels,
            'exclusive_labels': self.config.exclusive_labels,
            'messages': metadata['messages'], 'context': context,
        }, MultiLabelResult)
        returned = [item.label for item in result.assessments]
        if len(returned) != len(labels) or set(returned) != set(labels):
            raise ValueError('Multi-label judge must return every label exactly once')
        # The model may put selected labels first. Order is presentation, not meaning.
        by_label = {item.label: item for item in result.assessments}
        result = result.model_copy(update={'assessments': [by_label[label] for label in labels]})
        if not result.labels:
            raise ValueError('Select at least one label')
        if set(result.labels).intersection(self.config.exclusive_labels) and len(result.labels) != 1:
            raise ValueError('Exclusive label cannot coexist with other labels')
        for item in result.assessments:
            indices = item.message_indices
            if item.selected != bool(indices):
                raise ValueError('Only selected labels must have nonempty source indices')
            if len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < len(metadata['messages']) for i in indices):
                raise ValueError('Invalid source indices')
        return result
