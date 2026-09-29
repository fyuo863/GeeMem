"""Source-grounded extraction contracts and deterministic admission checks."""
from typing import Literal
from pydantic import Field, create_model
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


class IndexedEdge(StrictModel):
    source: Text
    target: Text
    relation: Text
    message_indices: list[int] = Field(min_length=1, max_length=200)


def indexed_schema(visible: list[int]):
    """Constrain generated references to the actual global IDs shown to the LLM."""
    index_type = Literal[tuple(visible)]
    node = create_model("IndexedNode", __base__=GroundedNode,
                        message_indices=(list[index_type], Field(min_length=1, max_length=200)))
    edge = create_model("IndexedEdge", __base__=IndexedEdge,
                        message_indices=(list[index_type], Field(min_length=1, max_length=200)))
    return create_model("IndexedGraph", __base__=StrictModel,
                        directed=(Literal[False], False),
                        nodes=(list[node], Field(max_length=1000)),
                        edges=(list[edge], Field(max_length=2000)))


def ground_from_indices(candidate, request: AddRequest) -> GroundedGraph:
    """Attach verbatim source, without asking the model to regenerate quotations.

    This validates provenance locations, NOT whether an edge is semantically
    entailed by its referenced message. Semantic accuracy needs separate evals.
    """
    data = candidate.model_dump()
    errors = [f"Invalid evidence index {i} on {item}" for item in [*data["nodes"], *data["edges"]]
              for i in item["message_indices"] if i < 0 or i >= len(request.messages)]
    if errors:
        raise ValueError("\n".join(errors[:50]))
    for edge in data["edges"]:
        edge["evidence"] = [dict(message_index=i, text=request.messages[i].content)
                            for i in sorted(set(edge["message_indices"]))]
    return GroundedGraph.model_validate(data)


RELATION_RULES = """
Edges are UNDIRECTED associations, not subject-predicate-object statements. source
and target are just two unordered endpoint keys. A--B and B--A are the SAME edge.
Use concise relation labels such as 询问, 支持, 参与, 鼓励, 情绪关联, 结果关联.
Do not create inverse predicates or encode who acts on whom in endpoint order.
Who did what, temporal order and negation remain in the EXACT source quotes.
A speaker's family is its OWN group node, not the other conversation participant.
Every node has owner_key: the key of the entity it belongs to, or null if
ownership is unknown or not applicable. Ownership is not the storage user_id,
the speaker mentioning something, or mere participation in an event.
Use the JSON literal null, NEVER the string "null". Every non-null owner_key
MUST exactly match a key in nodes. People normally have owner_key=null; people
do not belong to the API caller. Check all owner references before responding.
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


def admit_graph(graph: GroundedGraph, request: AddRequest, allowed_indices=None) -> Graph:
    result = Graph.model_validate(graph.model_dump())
    allowed = set(range(len(request.messages))) if allowed_indices is None else set(allowed_indices)
    errors = []
    keys = {n.key for n in result.nodes}
    if len(keys) != len(result.nodes):
        errors.append("Duplicate node keys")
    for node in result.nodes:
        # Only repair an unambiguous null serialization mistake. Never erase
        # unknown real owners or turn a reference to an actual 'null' node into None.
        if node.owner_key is not None and node.owner_key not in keys and node.owner_key.casefold() == "null":
            node.owner_key = None
    owners = {n.key: n.owner_key for n in result.nodes}
    for node in result.nodes:
        seen = {node.key}
        owner = node.owner_key
        while owner is not None:
            if owner not in keys:
                errors.append(f"Unknown node owner {owner!r} referenced by {node.key!r}")
                break
            if owner in seen:
                errors.append(f"Cyclic node ownership at {node.key!r}")
                break
            seen.add(owner)
            owner = owners[owner]
        if not set(node.message_indices) <= allowed:
            errors.append(f"Node {node.key!r} cites unavailable message indices {node.message_indices}")
    merged = {}
    for number, edge in enumerate(result.edges):
        label = f"edge[{number}] {edge.source!r} / {edge.target!r}"
        for endpoint in (edge.source, edge.target):
            if endpoint not in keys:
                errors.append(f"Unknown edge endpoint {endpoint!r} in {label}; define its node with evidence")
        if {q.message_index for q in edge.evidence} != set(edge.message_indices):
            errors.append(f"{label}: Every relation evidence index needs an exact source quote")
        for quote in edge.evidence:
            if quote.message_index in allowed and quote.text in request.messages[quote.message_index].content:
                continue
            # Relocate only an exact quote appearing in ONE available message.
            # Do not approximate, change casing, or choose among ambiguous matches.
            matches = [i for i in sorted(allowed) if quote.text in request.messages[i].content]
            if len(matches) == 1:
                quote.message_index = matches[0]
            else:
                errors.append(f"{label}: quote {quote.text!r} does not uniquely locate in source; "
                              f"cited={quote.message_index}, exact_matches={matches}. Copy an exact substring with its GLOBAL message_index.")
        edge.message_indices = sorted({q.message_index for q in edge.evidence})
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
    if errors:
        raise ValueError("Graph validation errors:\n" + "\n".join(errors[:50]) +
                         (f"\n... {len(errors)-50} additional errors" if len(errors)>50 else ""))
    result.validate_references(len(request.messages))
    return result


def merge_graphs(graphs: list[Graph], request: AddRequest) -> Graph:
    """Merge overlapping chunks without discarding evidence or guessing owners."""
    nodes = {}
    edges = []
    for graph in graphs:
        for node in graph.nodes:
            if node.key not in nodes:
                nodes[node.key] = node.model_copy(deep=True)
                continue
            existing = nodes[node.key]
            if (existing.kind, existing.owner_key) != (node.kind, node.owner_key):
                raise ValueError(f"Conflicting kind/ownership across chunks for {node.key!r}")
            existing.message_indices = sorted(set(existing.message_indices + node.message_indices))
            existing.aliases = sorted(set(existing.aliases + node.aliases + [node.name]) - {existing.name})[:30]
        edges.extend(graph.edges)
    # Each chunk was admitted already. Recheck the merged graph and merge
    # repeated undirected edges/quotes via the same deterministic admission path.
    combined = GroundedGraph.model_validate(dict(nodes=[n.model_dump() for n in nodes.values()],
                                                edges=[e.model_dump() for e in edges]))
    return admit_graph(combined, request)
