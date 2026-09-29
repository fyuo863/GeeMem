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
    nodes: list[GroundedNode] = Field(max_length=1000)
    edges: list[GroundedEdge] = Field(max_length=2000)


class GraphPatch(StrictModel):
    # Removal indices refer to the original candidate edge array, never a shifting list.
    remove_edge_indices: list[int] = Field(default_factory=list, max_length=2000)
    upsert_nodes: list[GroundedNode] = Field(default_factory=list, max_length=1000)
    add_edges: list[GroundedEdge] = Field(default_factory=list, max_length=2000)
    findings: list[str] = Field(default_factory=list, max_length=200)


RELATION_RULES = """
Use canonical directions: 询问 = questioner -> addressee; 支持 = supporter -> recipient;
参加 = participant -> event/activity; 鼓励 = encourager -> recipient;
导致 = event/activity/fact -> outcome/emotion/event/fact; 感受 = experiencer -> emotion;
感谢 = thankful actor -> thanked actor; 认可 = actor -> fact/topic/outcome.
Use these Chinese relation labels for these meanings, not inverse English predicates
such as experienced, receives_encouragement or resulted_in. For other meanings use
specific source-language labels with an unambiguous subject -> object direction.
A family or group belonging to a speaker is its OWN group node, not the other speaker.
Questions are speech acts, not evidence that their premises are true. Capture EVERY
explicit question and its questioner/addressee, including several questions along
the same edge. Merge identical triples, retaining all their evidence messages.
"""


def apply_patch(candidate: GroundedGraph, patch: GraphPatch) -> GroundedGraph:
    removed = set(patch.remove_edge_indices)
    if any(i < 0 or i >= len(candidate.edges) for i in removed):
        raise ValueError("Audit patch references an unknown edge")
    nodes = {node.key: node for node in candidate.nodes}
    if len(nodes) != len(candidate.nodes):
        raise ValueError("Duplicate candidate node key")
    for node in patch.upsert_nodes:
        nodes[node.key] = node
    edges = [edge for i, edge in enumerate(candidate.edges) if i not in removed]
    edges.extend(patch.add_edges)
    return GroundedGraph(nodes=list(nodes.values()), edges=edges)


def admit_graph(graph: GroundedGraph, request: AddRequest) -> Graph:
    result = Graph.model_validate(graph.model_dump())
    result.validate_references(len(request.messages))
    nodes = {node.key: node for node in result.nodes}
    actor = {"person", "group", "organization"}
    constraints = {
        "询问": (actor, actor), "支持": (actor, actor), "鼓励": (actor, actor),
        "参加": (actor, {"event", "activity"}),
        "导致": ({"event", "activity", "fact"}, {"outcome", "emotion", "event", "fact"}),
        "感受": (actor, {"emotion"}),
    }
    merged = {}
    for edge in result.edges:
        if edge.relation in {"experienced", "receives_encouragement", "resulted_in"}:
            raise ValueError("Noncanonical inverse or ambiguous relation")
        if edge.relation in constraints:
            source_kinds, target_kinds = constraints[edge.relation]
            if nodes[edge.source].kind not in source_kinds or nodes[edge.target].kind not in target_kinds:
                raise ValueError("Relation endpoint type violation")
        if {q.message_index for q in edge.evidence} != set(edge.message_indices):
            raise ValueError("Every relation evidence index needs an exact source quote")
        for quote in edge.evidence:
            if quote.message_index >= len(request.messages) or quote.text not in request.messages[quote.message_index].content:
                raise ValueError("Evidence quote does not occur in the cited message")
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
