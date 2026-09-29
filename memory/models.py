from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing_extensions import Annotated

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

class Message(StrictModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: str = Field(min_length=1, max_length=32000)
    timestamp: int = Field(ge=0, le=253402300799999, strict=True)

class AddRequest(StrictModel):
    request_id: Text
    user_id: Text
    session_id: Text
    messages: list[Message] = Field(min_length=1, max_length=200)

class SearchRequest(StrictModel):
    user_id: Text
    query: str = Field(min_length=1, max_length=4000)
    session_id: Text | None = None
    limit: int = Field(default=10, ge=1, le=100)
    max_hops: int = Field(default=2, ge=0, le=4)

class Node(StrictModel):
    key: Text
    name: Text
    kind: Text
    aliases: list[Text] = Field(default_factory=list, max_length=30)
    message_indices: list[int] = Field(min_length=1, max_length=200)

class Edge(StrictModel):
    source: Text
    target: Text
    relation: Text
    message_indices: list[int] = Field(min_length=1, max_length=200)

class Graph(StrictModel):
    nodes: list[Node] = Field(max_length=1000)
    edges: list[Edge] = Field(max_length=2000)

    def validate_references(self, count: int) -> None:
        keys = {n.key for n in self.nodes}
        if len(keys) != len(self.nodes):
            raise ValueError("Duplicate node keys")
        for item in [*self.nodes, *self.edges]:
            if any(i < 0 or i >= count for i in item.message_indices):
                raise ValueError("Invalid evidence message index")
        if any(e.source not in keys or e.target not in keys for e in self.edges):
            raise ValueError("Unknown edge endpoint")

class Keywords(StrictModel):
    keywords: list[Text] = Field(min_length=1, max_length=20)

class MemoryHit(StrictModel):
    id: str
    content: str
    score: float
    created_at: str

class SearchResponse(StrictModel):
    data: list[MemoryHit]

class AddResponse(StrictModel):
    request_id: str
    message_ids: list[str]
    nodes: int
    edges: int
    deduplicated: bool = False
