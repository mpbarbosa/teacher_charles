"""Tipos do domínio.

Dataclasses simples, não modelos de ORM: o esquema vive no SQL e estes tipos
só dão nome às linhas que saem dele.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Literal

CefrLevel = Literal["A2", "B1", "B2", "C1", "C2"]
Mode = Literal["conversation", "correction", "drill"]
Role = Literal["student", "teacher"]

CEFR_LEVELS: tuple[CefrLevel, ...] = ("A2", "B1", "B2", "C1", "C2")
MODES: tuple[Mode, ...] = ("conversation", "correction", "drill")


@dataclass(frozen=True, slots=True)
class Student:
    id: int
    name: str
    cefr_level: str
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Student:
        return cls(
            id=row["id"],
            name=row["name"],
            cefr_level=row["cefr_level"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


@dataclass(frozen=True, slots=True)
class Turn:
    id: int
    session_id: int
    role: str
    content: str
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Turn:
        return cls(
            id=row["id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
        )


@dataclass(frozen=True, slots=True)
class Correction:
    """Uma correção, como dado.

    ``rule_id`` aponta para um item de ``data/contrastive/``. Quando é ``None``,
    a correção veio só do modelo, sem regra curada por trás — distinção que a
    E4 precisa medir e que a E7 mostra no error log.
    """

    id: int
    turn_id: int
    span: str
    corrected: str
    error_type: str
    rule_id: str | None
    explanation: str
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Correction:
        return cls(
            id=row["id"],
            turn_id=row["turn_id"],
            span=row["span"],
            corrected=row["corrected"],
            error_type=row["error_type"],
            rule_id=row["rule_id"],
            explanation=row["explanation"],
            created_at=row["created_at"],
        )
