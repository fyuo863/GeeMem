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
Connect the entities actually mentioned in the evidence. Do not connect an event
to the person merely commenting on it. Capture all explicit questions as 询问
associations between the speakers, retaining each supporting question message.
Merge repeated unordered pairs with the same label and retain all their evidence.
"""


def admit_graph(graph: GroundedGraph, request: AddRequest) -> Graph:
    result = Graph.model_validate(graph.model_dump())
    result.validate_references(len(request.messages))
    merged = {}
    for edge in result.edges:
        if {q.message_index for q in edge.evidence} != set(edge.message_indices):
            raise ValueError("Every relation evidence index needs an exact source quote")
        for quote in edge.evidence:
            if quote.message_index >= len(request.messages) or quote.text not in request.messages[quote.message_index].content:
                raise ValueError("Evidence quote does not occur in the cited message")
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
