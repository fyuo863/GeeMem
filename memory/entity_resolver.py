"""Shared, evidence-aware entity identity and alias resolution.

The resolver is deliberately deterministic.  A name is scoped by ``user_id``
and canonical name; ``kind`` is descriptive metadata and never part of the
identity key.  Aliases are only registered when the caller has explicit
evidence for the mapping.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Iterable

import hashlib
import re


def normalize(value: str) -> str:
    return re.sub(r'\s+', ' ', value.strip().casefold())


def canonical_name(value: str) -> str:
    return '我' if normalize(value) in {'i', 'me', 'user', 'current speaker', '当前用户'} else value.strip()


def _id(prefix: str, *parts: str) -> str:
    return prefix + '_' + hashlib.sha256('\x1f'.join(parts).encode('utf-8')).hexdigest()


class EntityResolver:
    def initialize(self, db: sqlite3.Connection) -> None:
        db.execute("""
            CREATE TABLE IF NOT EXISTS entity_aliases(
                user_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                alias TEXT NOT NULL,
                normalized_alias TEXT NOT NULL,
                alias_type TEXT NOT NULL,
                source_id TEXT,
                confidence REAL NOT NULL DEFAULT 1.0,
                status TEXT NOT NULL DEFAULT 'active',
                UNIQUE(user_id, normalized_alias)
            )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS entity_alias_lookup ON entity_aliases(user_id, normalized_alias)")

    def resolve(
        self,
        db: sqlite3.Connection,
        user_id: str,
        name: str,
        kind: str = "person",
        *,
        aliases: Iterable[str] = (),
        source_id: str | None = None,
        confidence: float = 1.0,
    ) -> sqlite3.Row:
        """Resolve a canonical name or known alias, creating an entity if needed."""
        self.initialize(db)
        display = canonical_name(name)
        norm = normalize(display)
        alias_row = db.execute(
            "SELECT e.* FROM entity_aliases a JOIN relationship_entities e ON e.id=a.entity_id "
            "WHERE a.user_id=? AND a.normalized_alias=? AND a.status='active'",
            (user_id, norm),
        ).fetchone()
        if alias_row:
            entity = alias_row
        else:
            # New identity semantics: kind does not split the same name.
            entity = db.execute(
                "SELECT * FROM relationship_entities WHERE user_id=? AND lower(name)=lower(?) "
                "ORDER BY id LIMIT 1", (user_id, display)
            ).fetchone()
            if entity is None:
                identifier = _id("ent", user_id, norm)
                db.execute(
                    "INSERT INTO relationship_entities VALUES (?,?,?,?,?,?)",
                    (identifier, user_id, norm, display, kind, json.dumps([display], ensure_ascii=False)),
                )
                entity = db.execute("SELECT * FROM relationship_entities WHERE id=?", (identifier,)).fetchone()

        for alias in aliases:
            alias = str(alias).strip()
            if alias and normalize(alias) != normalize(entity["name"]):
                self.register_alias(db, user_id, entity["id"], alias,
                                    source_id=source_id, confidence=confidence)
        return entity

    def register_alias(
        self,
        db: sqlite3.Connection,
        user_id: str,
        entity_id: str,
        alias: str,
        *,
        alias_type: str = "nickname",
        source_id: str | None = None,
        confidence: float = 1.0,
    ) -> bool:
        """Register an alias only if it is unclaimed or already points here."""
        value = alias.strip()
        if not value:
            return False
        norm = normalize(value)
        existing = db.execute(
            "SELECT entity_id FROM entity_aliases WHERE user_id=? AND normalized_alias=? AND status='active'",
            (user_id, norm),
        ).fetchone()
        if existing and existing[0] != entity_id:
            return False
        db.execute(
            "INSERT OR IGNORE INTO entity_aliases(user_id,entity_id,alias,normalized_alias,alias_type,source_id,confidence) "
            "VALUES (?,?,?,?,?,?,?)",
            (user_id, entity_id, value, norm, alias_type, source_id, confidence),
        )
        row = db.execute("SELECT aliases FROM relationship_entities WHERE id=?", (entity_id,)).fetchone()
        if row:
            values = set(json.loads(row[0])) | {value}
            db.execute("UPDATE relationship_entities SET aliases=? WHERE id=?",
                       (json.dumps(sorted(values), ensure_ascii=False), entity_id))
        return True

    def aliases_for(self, db: sqlite3.Connection, user_id: str, entity_id: str) -> list[str]:
        return [r[0] for r in db.execute(
            "SELECT alias FROM entity_aliases WHERE user_id=? AND entity_id=? AND status='active' ORDER BY alias",
            (user_id, entity_id),
        )]
