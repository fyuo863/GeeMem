"""Bounded recall hints; existing review decides whether sources actually apply."""
import re


def supplemental_query(question, missing=None):
    if len(question)>1200: return None
    if missing is not None:
        base = supplemental_query(question)
        return (base[0], question+'\nMissing evidence: '+missing[:300]) if base and missing.strip() else None
    if re.search(r'规则|例外|审批|报销|退款|\b(rule|exception|approval|refund|policy)\b',question,re.I):
        return ('rule',question+'\n条件、适用范围、例外、限制、撤销。 Conditions, applicability, exceptions, restrictions, revocation.')
    if re.search(r'变化|过程|进展|最终|后来|取消|计划|\b(change|changed|progress|finally|cancelled|plan|planned|outcome)\b',question,re.I):
        return ('stage',question+'\n同一事项的计划、变更、取消、执行、结果。 Same matter: plan, change, cancellation, execution, outcome.')
    return None
