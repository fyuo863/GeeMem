"""Read-only, bounded graph tools for the extraction agent's request-local graph."""
from typing import Literal
from pydantic import Field
from .models import StrictModel, Text
from .conversation import TurnDraft, norm, digest


class GraphQuery(StrictModel):
    tool: Literal['search_nodes', 'get_node', 'search_edges', 'get_neighbors']
    query: str = Field(default='', max_length=256)
    key: Text | None = None
    offset: int = Field(default=0, ge=0, le=10000)
    limit: int = Field(default=5, ge=1, le=10)
    include_evidence: bool = False


class GraphAction(StrictModel):
    action: Literal['query', 'finish']
    queries: list[GraphQuery] = Field(default_factory=list, max_length=2)
    draft: TurnDraft | None = None


class GraphQueryAction(GraphAction):
    action: Literal['query'] = 'query'
    queries: list[GraphQuery] = Field(min_length=1, max_length=2)
    draft: None = None


TOOL_INSTRUCTION = """
You can inspect the graph already constructed in THIS /add request using read-only
tools. Your FIRST action must query the graph before any final draft.
Return action=query with 1-2 queries and draft=null, or action=finish with
queries=[] and your final draft. Each tool round is executed by the program and
results are returned in graph_tool_history. At most 3 query rounds per attempt.
Tools: search_nodes (query matches name/aliases), get_node (key), search_edges
(query matches relation; optional key filters an endpoint), get_neighbors (key).
key accepts stable node keys or speaker IDs A/B. Empty search query lists items.
Results support offset/limit pagination and optional original evidence excerpts.
Use these tools to verify identity, owners, contacts and existing relationships,
especially before same_as or when pronouns refer to older information. Speaker
nodes expose speaker_id: use that ID in draft relations, never recreate that person.
Tool results are previous model extractions, NOT guaranteed truth. Check their
source evidence; never copy an old fact as if asserted in the CURRENT message.
Read-only provenance indices in results must NEVER be emitted in your draft.
Tools cannot mutate nodes, create edges, or read another request/user's graph.
"""


def query_graph(state, query: GraphQuery):
    """State is a per-extraction snapshot; no database, shared cache or mutation."""
    def evidence(indices):
        if not query.include_evidence:
            return []
        return [dict(message_index=i, speaker=state.labels[state.request.messages[i].role],
                     text=state.request.messages[i].content[:2000],
                     text_truncated=len(state.request.messages[i].content)>2000)
                for i in sorted(set(indices))[:2]]

    def node(n):
        return dict(key=n.key, name=n.name, kind=n.kind, aliases=list(n.aliases),
                    speaker_id=next((k for k,v in state.keys.items() if v==n.key),None),
                    owner_key=n.owner_key, contact_keys=list(n.contact_keys),
                    speaker_tags=list(n.speaker_tags), message_indices=list(n.message_indices),
                    evidence=evidence(n.message_indices),
                    evidence_total=len(set(n.message_indices)))

    def edge(e):
        return dict(key='edge:'+digest(e.source,e.target,norm(e.relation)),
                    source=e.source, target=e.target, relation=e.relation,
                    source_name=state.nodes[e.source].name, target_name=state.nodes[e.target].name,
                    message_indices=list(e.message_indices), evidence=evidence(e.message_indices),
                    evidence_total=len(set(e.message_indices)))

    key=state.keys.get(query.key,query.key)
    if query.tool in ('get_node','get_neighbors') and key not in state.nodes:
        return dict(error='Unknown node key', items=[], total=0, next_offset=None)
    if query.tool=='search_edges' and key is not None and key not in state.nodes:
        return dict(error='Unknown node key', items=[], total=0, next_offset=None)
    term=norm(query.query)
    if query.tool=='get_node':
        matches=[state.nodes[key]]; convert=node
    elif query.tool=='search_nodes':
        matches=[n for n in state.nodes.values() if any(term in norm(t) for t in [n.name,*n.aliases])];convert=node
    else:
        matches=[e for e in state.edges.values() if
                 (key is None or key in (e.source,e.target)) and
                 (query.tool=='get_neighbors' or term in norm(e.relation))];convert=edge
    stop=query.offset+query.limit
    return dict(items=[convert(x) for x in matches[query.offset:stop]], total=len(matches),
                next_offset=stop if stop<len(matches) else None,
                scope='current_add_request', evidence_limit_per_item=2)
