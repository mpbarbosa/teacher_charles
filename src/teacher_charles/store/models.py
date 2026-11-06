"""Tipos do domínio.

Dataclasses simples, não modelos de ORM: o esquema vive no SQL e estes tipos
só dão nome às linhas que saem dele.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Literal

CefrLevel = Literal["A2", "B1", "B2", "C1", "C2"]
LevelSource = Literal["self", "auto"]
Mode = Literal["conversation", "correction", "drill"]
Role = Literal["student", "teacher"]

CEFR_LEVELS: tuple[CefrLevel, ...] = ("A2", "B1", "B2", "C1", "C2")
LEVEL_SOURCES: tuple[LevelSource, ...] = ("self", "auto")
MODES: tuple[Mode, ...] = ("conversation", "correction", "drill")


@dataclass(frozen=True, slots=True)
class Student:
    """O perfil do aluno.

    ``cefr_level`` é o nível em vigor — o que muda o prompt da resposta — e
    anda sozinho a partir dos sinais de cada turno. ``self_declared_level``
    guarda o que o aluno disse quando chegou, e ``level_source`` diz qual dos
    dois está valendo. Sem essa separação não há como responder "por que o
    Charles mudou o jeito de falar comigo".
    """

    id: int
    name: str
    cefr_level: str
    created_at: str
    updated_at: str
    self_declared_level: str | None = None
    level_source: str = "self"
    seen_seq: int = 0

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Student:
        return cls(
            id=row["id"],
            name=row["name"],
            cefr_level=row["cefr_level"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            self_declared_level=row["self_declared_level"],
            level_source=row["level_source"],
            seen_seq=row["seen_seq"],
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


@dataclass(frozen=True, slots=True)
class LevelSignal:
    """Os sinais de nível de um turno, como linha.

    Guardados inclusive quando ``counted`` é falso — turno curto demais para
    ser evidência. Sem as linhas descartadas não dá para medir se o corte está
    no lugar certo, e um corte que ninguém mede é chute.
    """

    id: int
    turn_id: int
    words: int
    sentences: int
    long_words: int
    connectives: int
    structures: int
    errors: int
    score: float
    estimated_level: str
    counted: bool
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> LevelSignal:
        return cls(
            id=row["id"],
            turn_id=row["turn_id"],
            words=row["words"],
            sentences=row["sentences"],
            long_words=row["long_words"],
            connectives=row["connectives"],
            structures=row["structures"],
            errors=row["errors"],
            score=row["score"],
            estimated_level=row["estimated_level"],
            counted=bool(row["counted"]),
            created_at=row["created_at"],
        )


@dataclass(frozen=True, slots=True)
class LevelChange:
    """Uma mudança de nível, com o motivo."""

    id: int
    student_id: int
    from_level: str
    to_level: str
    source: str
    reason: str
    at_signal_id: int
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> LevelChange:
        return cls(
            id=row["id"],
            student_id=row["student_id"],
            from_level=row["from_level"],
            to_level=row["to_level"],
            source=row["source"],
            reason=row["reason"],
            at_signal_id=row["at_signal_id"],
            created_at=row["created_at"],
        )
