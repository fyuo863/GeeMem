"""Grounded provenance, not a proof of semantic entailment or answer correctness."""
from dataclasses import dataclass, asdict
import re


@dataclass(frozen=True)
class ChainNode:
    hop: int
    source_id: str
    quote: str
    needed_for: str
    derived_from: list[str]


def build_chain(supports, sources, strategy, sufficient, links=(), missing='', root_anchor=''):
    selected = list(supports.values()) if hasattr(supports, 'values') else list(supports)
    nodes, errors, edges = [], [], []
    by_index = {}
    for index, support in enumerate(selected):
        sid = support.source_id
        if sid == '__question__' or sid not in sources:
            errors.append({'index': index, 'error': 'source_not_in_evidence'})
        elif not support.quote.strip() or support.quote not in sources[sid]:
            errors.append({'index': index, 'error': 'quote_not_grounded'})
        else:
            node = ChainNode(index+1, sid, support.quote, support.needed_for, [])
            nodes.append(node)
            by_index[index] = node
    for link in links:
        parent, child = by_index.get(link.parent), by_index.get(link.child)
        if (not parent or not child or link.parent >= link.child or
                link.bridge not in parent.quote or link.bridge not in child.quote):
            errors.append({'parent': link.parent, 'child': link.child, 'error': 'invalid_link'})
            continue
        if parent.source_id not in child.derived_from:
            child.derived_from.append(parent.source_id)
            edges.append(dict(parent=link.parent, child=link.child, bridge=link.bridge))
    # Direct/split facts need no artificial dependency edges. A single source may
    # contain an entire chain. Multi-source chain connectivity is necessary, never
    # sufficient to prove that the model's relationship interpretation is correct.
    connected = strategy != 'chain' or all(n.derived_from for n in nodes[1:])
    names = re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+){0,2}\b", root_anchor)
    # A possessive route anchor (Veda's brother) can be expressed as 'Veda has
    # a brother'. Accept its single explicit name, never invent an alias.
    anchor = names[0] if len(names) == 1 else root_anchor
    anchored = strategy != 'chain' or not anchor or bool(nodes and anchor in nodes[0].quote)
    complete = anchored and bool(nodes) and sufficient and not missing.strip() and not errors and connected
    gaps = ([missing] if missing.strip() else [])
    if not connected: gaps.append('unverified_dependency')
    if not anchored: gaps.append('missing_question_anchor')
    if not sufficient and not gaps: gaps.append('review_incomplete')
    if not nodes: gaps.append('no_grounded_support')
    return nodes, dict(status='complete' if complete else ('invalid' if errors else 'partial'),
        assurance='source_grounded_and_model_reviewed_not_semantically_proven',
        strategy=strategy, hop_count=len(nodes), errors=errors, missing_hops=gaps,
        nodes=[asdict(n) for n in nodes], links=edges)
