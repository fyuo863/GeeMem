"""Source-grounded extraction contracts and deterministic admission checks."""
from typing import Literal
from pydantic import Field
from .models import StrictModel, Text, Node, Edge, Graph, AddRequest


class Quote(StrictModel):
    message_index: int = Field(ge=0, strict=True)
    text: str = Field(min_length=1)


class GroundedNode(Node):
    kind: Literal["person", "group", "organization", "event", "activity", "place",
                  "object", "fact", "outcome", "emotion", "topic"]


class GroundedEdge(Edge):
    evidence: list[Quote] = Field(min_length=1, max_length=200)


class GroundedGraph(StrictModel):
    directed: Literal[False] = False
    nodes: list[GroundedNode] = Field(max_length=1000)
    edges: list[GroundedEdge] = Field(max_length=2000)


RELATION_RULES = """
Edges are UNDIRECTED associations, not subject-predicate-object statements. source
and target are just two unordered endpoint keys. A--B and B--A are the SAME edge.
Use concise relation labels such as 询问, 支持, 参与, 鼓励, 情绪关联, 结果关联.
Do not create inverse predicates or encode who acts on whom in endpoint order.
Who did what, temporal order and negation remain in the EXACT source quotes.
A speaker's family is its OWN group node, not the other conversation participant.
Every node has owner_key: the key of the entity it belongs to, or null if
ownership is unknown or not applicable. Ownership is not the storage user_id,
Use the JSON literal null, NEVER the string "null". Every non-null owner_key
MUST exactly match a key in nodes. People normally have owner_key=null; people
do not belong to the API caller. Check all owner references before responding.
the speaker mentioning something, or mere participation in an event.
Resolve possessives (my/our/his/her) using the actual speaker and context.
Never merge families, possessions or personal experiences of DIFFERENT owners.
Create a distinct node for each owner, with a unique owner-qualified key and
display name (e.g. group:alice_family, Alice's family, owner_key=person:alice).
Keep the owner node in the graph. Never use a list of different owners to paper
over distinct entities. Do not guess missing owners; use null instead.
Only assign ownership supported by the cited node messages and conversation.
Connect the entities actually mentioned in the evidence. Do not connect an event
to the person merely commenting on it. Capture all explicit questions as 询问
associations between the speakers, retaining each supporting question message.
Merge repeated unordered pairs with the same label and retain all their evidence.
"""


def admit_graph(graph: GroundedGraph, request: AddRequest) -> Graph:
    result = Graph.model_validate(graph.model_dump())
    keys = {n.key for n in result.nodes}
    for node in result.nodes:
        # Only repair an unambiguous null serialization mistake. Never erase
        # unknown real owners or turn a reference to an actual 'null' node into None.
        if node.owner_key is not None and node.owner_key not in keys and node.owner_key.casefold() == "null":
            node.owner_key = None
    result.validate_references(len(request.messages))
    merged = {}
    for edge in result.edges:
        if {q.message_index for q in edge.evidence} != set(edge.message_indices):
            raise ValueError("Every relation evidence index needs an exact source quote")
        for quote in edge.evidence:
            if quote.message_index >= len(request.messages) or quote.text not in request.messages[quote.message_index].content:
                raise ValueError(f"Evidence quote does not occur in message {quote.message_index} for edge {edge.source!r} / {edge.target!r}")
        edge.source, edge.target = sorted((edge.source, edge.target))
        key = (edge.source, edge.target, edge.relation)
        if key in merged:
            current = merged[key]
            current.message_indices = sorted(set(current.message_indices + edge.message_indices))
            seen = {(q.message_index, q.text) for q in current.evidence}
            current.evidence.extend(q for q in edge.evidence if (q.message_index, q.text) not in seen)
        else:
            merged[key] = edge.model_copy(deep=True)
    result.edges = list(merged.values())
    return result
