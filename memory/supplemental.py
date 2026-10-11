"""Bounded recall hints; existing review decides whether sources actually apply."""
import re


def retrieval_gap(missing):
    """Ignore explicit no-gap reports, not arbitrary negative statements.

    Full-clause matching deliberately preserves unknown/mixed descriptions. This
    guards prose-triggered retrieval; it does not adjudicate evidence sufficiency
    or override independently grounded queries/structured gaps.
    """
    text = (missing or '').strip()
    clauses = [s.strip() for s in re.split(r'[.;。；\n]+', text) if s.strip()]
    noun = r'(?:sources?|evidence|links?|gaps?|facts?|information|relationships?|steps?)'
    modifiers = r'(?:(?:relevant|intermediate|earlier|additional|further|necessary|required|retrieval|or|and)\s+)*'
    patterns = [
        r'(?:none|null|n/a|no|无|没有|无缺口|没有缺口)',
        rf'(?:there\s+(?:is|are)\s+)?no\s+{modifiers}{noun}\s+(?:(?:is|are)\s+)?(?:missing|needed|required)',
        rf'(?:there\s+(?:is|are)\s+)?no\s+{modifiers}missing\s+{modifiers}{noun}(?:\s+(?:found|identified|remain|remains))?',
        r'no\s+(?:retrieval\s+)?gaps?(?:\s+(?:found|identified|remain|remains))?',
        rf'all\s+(?:(?:relevant|necessary|required|needed|requested)\s+)*{noun}\s+(?:is|are|has been|have been)\s+(?:present|available|provided|found|supplied)',
        r'(?:没有|无|不存在)(?:任何|相关|必要|额外)?(?:缺失|缺少|遗漏)(?:的)?(?:中间|早期|或|和)*(?:来源|证据|关系|环节)',
        r'(?:没有|无|不存在)(?:任何|相关|必要|额外|中间|早期|或|和)*(?:来源|证据|关系|环节)(?:缺失|遗漏)',
        r'(?:所有|全部)(?:所需|相关|必要)?(?:证据|来源|关系|环节)(?:均|都)?(?:已齐全|齐全|已提供|已找到|已具备)',
    ]
    if clauses and all(any(re.fullmatch(p, c, re.I) for p in patterns) for c in clauses):
        return ''
    return text


def supplemental_query(question, missing=None):
    if len(question)>1200: return None
    if missing is not None:
        missing = retrieval_gap(missing)
        if not missing: return None
        base = supplemental_query(question)
        return (base[0], question+'\nMissing evidence: '+missing[:300]) if base and missing.strip() else None
    if re.search(r'规则|例外|审批|报销|退款|\b(rule|exception|approval|refund|policy)\b',question,re.I):
        return ('rule',question+'\n条件、适用范围、例外、限制、撤销。 Conditions, applicability, exceptions, restrictions, revocation.')
    if re.search(r'变化|过程|进展|最终|后来|取消|计划|\b(change|changed|progress|finally|cancelled|plan|planned|outcome)\b',question,re.I):
        return ('stage',question+'\n同一事项的计划、变更、取消、执行、结果。 Same matter: plan, change, cancellation, execution, outcome.')
    return None
