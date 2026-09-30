"""Program-owned speaker identities, mention scopes and source positions."""
import copy
import hashlib
import json
import re
import unicodedata
from typing import Literal
from pydantic import Field
from .models import StrictModel, Text, Node, Edge, EvidenceQuote, Graph


class Mention(StrictModel):
    ref: Text
    name: Text
    kind: Literal['person', 'group', 'organization', 'event', 'activity', 'place',
                  'object', 'fact', 'outcome', 'emotion', 'topic']
    owner: Text | None = None
    same_as: Text | None = None
    match_quote: Text | None = None


class Relation(StrictModel):
    source: Text
    target: Text
    relation: Text


class TurnDraft(StrictModel):
    entities: list[Mention] = Field(max_length=100)
    relations: list[Relation] = Field(max_length=200)


def norm(text):
    return unicodedata.normalize('NFKC', text).casefold().strip()


def digest(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()


class Conversation:
    def __init__(self, request):
        self.request = request
        # Roles identify channels, not alternating positions. A repeated role
        # stays the same participant. Name discovery never changes node keys.
        roles = list(dict.fromkeys(m.role for m in request.messages))
        # Reserve the counterpart even when only the first turn has arrived.
        if 'user' in roles and 'assistant' not in roles:
            roles.append('assistant')
        elif 'assistant' in roles and 'user' not in roles:
            roles.append('user')
        self.labels = {role: chr(65+i) for i, role in enumerate(roles)}
        self.roles = {label: role for role, label in self.labels.items()}
        self.keys = {label: 'speaker:' + digest(request.session_id, role)
                     for role, label in self.labels.items()}
        self.names = {}
        self.nodes = {}
        self.edges = {}
        self.audit = []

    def payload(self, index):
        current = self.request.messages[index]
        return dict(current=dict(speaker=self.labels[current.role], role=current.role,
                                 content=current.content),
                    context=[dict(speaker=self.labels[m.role], role=m.role, content=m.content)
                             for m in self.request.messages[max(0, index-2):index]],
                    speakers=[dict(id=label, role=role, name=self.names.get(label))
                              for label, role in self.roles.items()],
                    known_entities=[dict(id=n.key, name=n.name, kind=n.kind,
                                         owner=next((label for label, key in self.keys.items() if key==n.owner_key), None),
                                         mentioned_by=[label for label,key in self.keys.items() if key in n.speaker_tags])
                                    for n in self.nodes.values() if n.key not in self.keys.values()])

    def speaker(self, label, index, tag):
        key = self.keys[label]
        if key not in self.nodes:
            self.nodes[key] = Node(key=key, name=self.names.get(label, label), kind='person',
                                  aliases=[label, self.roles[label]], speaker_tags=[], message_indices=[index])
        self.touch(self.nodes[key], index, tag)
        return key

    def touch(self, node, index, tag):
        node.message_indices = sorted(set(node.message_indices + [index]))
        node.speaker_tags = sorted(set(node.speaker_tags + [tag]))

    def bind_name(self, label, name, index, tag, self_introduction=False):
        if not name:
            return
        # Repeating a known name is not a new claim/evidence location. Ignore
        # placeholders instead of mistaking the 'A' in 'A lot' for a real name.
        if norm(name) in {norm(label), norm(self.roles[label])}:
            return
        if label in self.names and norm(self.names[label]) == norm(name):
            return
        content = self.request.messages[index].content
        if name not in content:
            raise ValueError('Name must occur verbatim in the CURRENT message')
        escaped = re.escape(name)
        if self_introduction:
            pattern = r"(?:my name is|i am|i'm|我是|我叫|我的名字是)\s*" + escaped + r"(?!\w)"
        else:
            pattern = (r"(?:(?:hey|hi|hello|thanks|thank you|谢谢你|谢谢|你好|嗨)[,，!！]?\s*" + escaped +
                       r"(?!\w)|(?:^|[.!?。！？])\s*" + escaped + r"\s*[,，!！?？])")
        if not re.search(pattern, content, flags=re.IGNORECASE):
            raise ValueError('New name needs an explicit self-introduction or direct address in current text; otherwise return null')
        if label in self.names and norm(self.names[label]) != norm(name):
            raise ValueError(f'Conflicting name for stable speaker {label}')
        if any(other != label and norm(value)==norm(name) for other,value in self.names.items()):
            raise ValueError('Same name cannot automatically merge two speakers')
        self.names[label] = name
        key = self.speaker(label, index, tag)
        self.nodes[key].name = name
        self.nodes[key].aliases = sorted(set(self.nodes[key].aliases + [name]))
        self.audit.append(dict(action='bind_name', speaker=label, name=name, message_index=index))

    def accept(self, draft: TurnDraft, index):
        # Model retries cannot partially rename, merge or attribute anything.
        state = copy.deepcopy(self)
        state._accept(draft, index)
        state.graph().validate_references(len(state.request.messages))
        return state

    def prepare(self, index):
        state = copy.deepcopy(self)
        state._identify(index)
        return state

    def _identify(self, index):
        current = self.request.messages[index]
        label = self.labels[current.role]
        tag = self.keys[label]
        self.speaker(label, index, tag)
        english = r"([A-Z][a-zA-Z'-]{1,30}(?: [A-Z][a-zA-Z'-]{1,30})?)"
        chinese = r"([\u4e00-\u9fff]{2,4})"
        intro = re.search(r"(?:[Mm]y name is|[Ii] am|I'm)\s+" + english, current.content)
        if not intro:
            intro = re.search(r"(?:我叫|我的名字是)\s*" + chinese + r"(?=[，。！、\s]|$)", current.content)
        address = re.search(r"(?:[Hh]ey|[Hh]i|[Hh]ello|[Tt]hanks|[Tt]hank you)[,，]?\s+" + english, current.content)
        if not address:
            address = re.search(r"(?:你好|嗨|谢谢你|谢谢)[，,]?\s*" + chinese + r"(?=[，。！、\s]|$)", current.content)
        if not address:
            address = re.match(english + r"\s*[,，!！]", current.content)
            if address and norm(address.group(1)) in {'hey','hi','hello','thanks','wow','yeah','yes','no','well','oh','great','awesome'}:
                address = None
        other = [x for x,role in self.roles.items() if x != label and role in ('user','assistant')]
        bindings = [(label,intro,True)]
        if current.role in ('user','assistant') and len(other)==1:
            bindings.append((other[0],address,False))
        for target,match,is_self in bindings:
            if match:
                try:
                    self.bind_name(target,match.group(1),index,tag,self_introduction=is_self)
                except ValueError as exc:
                    # An ambiguous/contradictory address must not rename or merge people.
                    self.audit.append(dict(action='unresolved_name',speaker=target,name=match.group(1),
                                           message_index=index,reason=str(exc)))

    def _accept(self, draft, index):
        self._identify(index)
        current = self.request.messages[index]
        label = self.labels[current.role]
        tag = self.keys[label]
        self.speaker(label, index, tag)
        refs = {}
        used_refs = {ref for r in draft.relations for ref in (r.source,r.target)}
        counts = {e.ref:sum(other.ref==e.ref for other in draft.entities) for e in draft.entities}
        for entity_index, entity in enumerate(draft.entities):
            local_ref = entity.ref
            if counts[entity.ref]>1 or entity.ref in self.roles:
                if entity.ref in used_refs:
                    raise ValueError(f'Ambiguous/reserved referenced entity ref {entity.ref}')
                # Unreferenced duplicates cannot misroute an edge. Retain every
                # isolated mention, giving it a program-owned local identifier.
                local_ref = None
            if entity.owner is not None and entity.owner not in self.roles:
                raise ValueError('Entity owner must be a supplied speaker ID or null')
            if entity.kind == 'person' and norm(entity.name) in {
                norm(x) for x in [*self.names.values(), *self.roles, *self.roles.values()]}:
                if entity.ref in used_refs:
                    raise ValueError('Use speaker IDs for participants; do not create duplicate participant nodes')
                self.audit.append(dict(action='discard_duplicate_participant', name=entity.name,
                                       message_index=index))
                continue
            owner = self.speaker(entity.owner, index, tag) if entity.owner else None
            existing = self.nodes.get(entity.same_as) if entity.same_as else None
            if entity.same_as:
                reason = None
                if existing is None or entity.same_as in self.keys.values():
                    reason = 'same_as must identify a known non-speaker entity'
                elif (existing.kind != entity.kind or norm(existing.name) != norm(entity.name)
                      or owner is None or existing.owner_key != owner):
                    reason = 'Merge requires matching kind, canonical name and explicit owner'
                elif (not entity.match_quote or entity.match_quote not in current.content
                      or norm(entity.name) not in norm(entity.match_quote)):
                    reason = 'Merge needs a CURRENT verbatim phrase containing the entity name'
                if reason:
                    self.audit.append(dict(action='decline_merge',target=entity.same_as,
                                           speaker=label,message_index=index,reason=reason))
                    existing = None
            if existing is not None:
                key = existing.key
                self.touch(existing, index, tag)
                self.audit.append(dict(action='merge_mention', node=key, speaker=label,
                                       message_index=index, quote=entity.match_quote))
            else:
                key = 'mention:' + digest(self.request.session_id, self.request.request_id,
                                          label, index, entity_index)
                self.nodes[key] = Node(key=key, name=entity.name, kind=entity.kind, owner_key=owner,
                                       speaker_tags=[tag], message_indices=[index])
                self.audit.append(dict(action='create_mention',node=key,speaker=label,message_index=index))
            if local_ref is not None:
                refs[local_ref] = key
        def endpoint(ref):
            if ref in self.roles:
                return self.speaker(ref, index, tag)
            if ref not in refs:
                raise ValueError(f'Unknown relation endpoint {ref}; use speaker IDs or current entity refs')
            return refs[ref]
        for relation in draft.relations:
            a, b = sorted((endpoint(relation.source), endpoint(relation.target)))
            identity = (a, b, norm(relation.relation))
            if identity not in self.edges:
                self.edges[identity] = Edge(source=a, target=b, relation=relation.relation,
                                            message_indices=[index], evidence=[])
            edge = self.edges[identity]
            edge.message_indices = sorted(set(edge.message_indices + [index]))
            if not any(q.message_index==index for q in edge.evidence):
                edge.evidence.append(EvidenceQuote(message_index=index, text=current.content))
            self.audit.append(dict(action='relation',source=a,target=b,relation=relation.relation,message_index=index))

    def graph(self):
        return Graph(nodes=list(self.nodes.values()), edges=list(self.edges.values()))


TURN_INSTRUCTION = """Extract ONLY facts and questions asserted in current.content.
context contains the preceding two messages for resolving pronouns, NOT facts to
extract again. Speaker IDs A/B/... are allocated by the program and authoritative.
Use these IDs directly as relation endpoints and entity owners. I/my refers to
current.speaker; you/your refers to the other speaker in a two-person exchange.
The role assistant is a channel, not an additional person. Do not invent another
participant. The program recognizes explicit greetings and self-introductions
and supplies speaker names. Do not assign names or replace speaker IDs.

Create entities for explicitly mentioned families, events, activities, objects
and third parties. Each entity has a LOCAL ref, name, kind and owner (speaker ID
or null). Mention attribution is added by the program, separate from ownership.
Give each entity a UNIQUE short local ref such as E0, E1. Never use null or a
speaker ID as ref. Existing entity IDs may appear only in same_as, not ref.
Use simple canonical entity names such as family, and the actual owner ID to
distinguish whose family. Do not substitute the listener for someone's family.
Only propose same_as for a known entity when kind, name and explicit owner agree
AND current.content clearly refers to that same entity. match_quote must be a
verbatim phrase from current containing its canonical name. If uncertain set
same_as and match_quote to null; distinct nodes are safer than false merges.
Both owners must be non-null for same_as. Do not merge unowned events or topics.

Relations are undirected; source/target are unordered speaker IDs or local refs.
Use 询问 between actual speakers for an explicit question. Do not turn a question
into a fact. Participation belongs to the actual participant, not a commentator.
Do not repeat old entities/relations simply because they are in known_entities.
Output no indices, timestamps, speaker tags or quotations for nodes/relations:
the program attaches the CURRENT original message position to every element.
"""
