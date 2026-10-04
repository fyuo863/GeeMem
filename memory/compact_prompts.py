"""Optional compact instructions; legacy strings remain frozen for ablation."""

ROUTE = '''Choose direct, split or chain. Chain: an unknown related entity must be
resolved first; query only its first relationship. Split: independent named
targets need 2-3 separate lookups. Otherwise direct, queries=[]. Preserve every
relationship and the original person/time scope. Each query copies a short bridge
from the question; source_id="__question__". Never invent names or answer.
Example: employer of someone's spouse -> identify spouse, then their employer.'''

PLAN = '''List only the minimum required memory facts. One relationship per need;
use X,Y for unknown entities. depends_on contains zero-based EARLIER need indices.
For chain, the initial query targets only the first unresolved relationship.
For split, generate independent queries. Direct facts need no extra identity or
background lookup. Arithmetic needs operands, not a precomputed answer; lists
need scope coverage, advice needs user preferences. Never guess intermediate names.
Example: A's sibling's teacher's instrument -> sibling X; X's teacher Y;
instrument taught by Y, dependencies [],[0],[1].'''

REVIEW = '''Collect evidence for the original question, not an answer. Check the
exact subject, relation and object. Someone's own employer does not establish a
relative's employer; advice/plans are not completed events. Preserve scope and
distinguish message time from event time. Cite exact contiguous source text for
each necessary link. Query only necessary missing facts; no repeated queries.'''

BIND = '''Evidence refs E1,E2,... are assigned by the program. Q0 is the question,
usable as a query anchor but never proof. A query contains source_ref, an exact
unique quote, a short anchor occurring once in that quote, and a template with
one {target}. New entities must cite evidence, not Q0. Use the resolved predecessor
entity; the anchor's type must match the query. No other invented entity.'''

NEEDS = '''Return exactly one state for every supplied need_id: supported,
missing, conflict or time_unknown. Supported requires an exact quote establishing
that need's subject/relation AND supported dependencies. Check all current evidence;
prior citations are not verdicts. A's sibling=B plus A's teacher=C does NOT prove
B's teacher. Respect chronology and coverage for changing facts or exhaustive lists.
Queries target unresolved needs with supported dependencies, one for chain or up
to three independent needs; cite need_id. Return queries=[] when none is useful.'''


def effective_instruction(instruction, style):
    if style == 'long':
        return instruction
    from . import multihop as m, multihop_evidence as e
    for old, new in [(m.ROUTE_PROMPT, ROUTE), (e.NEED_PLAN, '\n'+PLAN),
                     (e.REVIEW_BASE, REVIEW), (e.BINDING_RULES, '\n'+BIND),
                     (e.NEED_RULES, '\n'+NEEDS),
                     (m.REVIEW_PROMPT, REVIEW+'\nUse source_id, quote, needed_for; sufficient only when every link is supported. Each query copies bridge from source and query; otherwise queries=[].')]:
        instruction = instruction.replace(old, new)
    return instruction
