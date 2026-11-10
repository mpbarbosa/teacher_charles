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

# De onde veio a evidência de uma revisão (E13). Ver RuleReview.
ReviewSource = Literal["drill", "lapse"]
REVIEW_SOURCES: tuple[ReviewSource, ...] = ("drill", "lapse")


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
class Session:
    """Uma conversa, do primeiro "hi" ao fim.

    ``mode`` escolhe o prompt (``prompts/modes/``) e, no drill,
    ``drill_rule_id`` diz qual regra do error log do aluno está sendo treinada.
    A regra mora aqui, e não no turno, porque drill que troca de regra a cada
    mensagem não é drill — e porque bloco estável no prompt é prefixo cacheado.
    """

    id: int
    student_id: int
    mode: str
    started_at: str
    ended_at: str | None = None
    drill_rule_id: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Session:
        return cls(
            id=row["id"],
            student_id=row["student_id"],
            mode=row["mode"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            drill_rule_id=row["drill_rule_id"],
        )


@dataclass(frozen=True, slots=True)
class RuleCount:
    """Quantas vezes o aluno errou por uma regra.

    Matéria-prima do drill (E6) e linha do error log (E7).
    """

    rule_id: str
    times: int
    last_span: str
    last_correction: str
    last_at: str = ""

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> RuleCount:
        return cls(
            rule_id=row["rule_id"],
            times=row["times"],
            last_span=row["last_span"],
            last_correction=row["last_correction"],
            last_at=row["last_at"] if "last_at" in row.keys() else "",
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


@dataclass(frozen=True, slots=True)
class RuleSchedule:
    """O estado de uma regra na fila do drill (E13).

    ``box`` é o degrau na escada de intervalos de ``teaching/schedule.py``;
    ``streak`` são os acertos consecutivos, e é o que gradua a regra. Os dois
    andam juntos no acerto e os dois voltam a zero no erro, mas não são o mesmo
    número: ``box`` tem teto (o último degrau da escada) e ``streak`` não, então
    o aluno que continua acertando continua contando sem ganhar intervalo novo.

    **A ausência de linha é informação.** Regra que o aluno errou e nunca
    treinou não tem registro aqui, e é a mais urgente da fila — na frente de
    qualquer uma com ``due_at`` vencido. Por isso o repositório devolve ``None``
    em vez de um agendamento zerado.
    """

    student_id: int
    rule_id: str
    box: int
    streak: int
    reviews: int
    lapses: int
    due_at: str
    last_review_at: str
    created_at: str = ""

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> RuleSchedule:
        return cls(
            student_id=row["student_id"],
            rule_id=row["rule_id"],
            box=row["box"],
            streak=row["streak"],
            reviews=row["reviews"],
            lapses=row["lapses"],
            due_at=row["due_at"],
            last_review_at=row["last_review_at"],
            created_at=row["created_at"] if "created_at" in row.keys() else "",
        )


@dataclass(frozen=True, slots=True)
class RuleReview:
    """Uma revisão, como evento.

    Mesmo motivo de ``LevelSignal`` existir ao lado de ``LevelChange``: o estado
    derivado (``RuleSchedule``) responde "quando esta regra volta", e só as
    linhas cruas respondem "a escada de intervalos está no lugar certo?". Uma
    escada que ninguém mede é chute, e sem estas linhas não há como recalibrá-la
    sobre histórico real.

    ``source`` carrega a assimetria do sinal: ``'drill'`` é resposta a um
    exercício **sobre esta regra**, onde a ausência de correção é evidência de
    acerto porque o exercício construiu a oportunidade; ``'lapse'`` é erro pela
    regra em qualquer outro lugar. Não existe acerto fora do drill.
    """

    id: int
    student_id: int
    rule_id: str
    turn_id: int
    hit: bool
    source: str
    box_before: int
    box_after: int
    interval_days: float
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> RuleReview:
        return cls(
            id=row["id"],
            student_id=row["student_id"],
            rule_id=row["rule_id"],
            turn_id=row["turn_id"],
            hit=bool(row["hit"]),
            source=row["source"],
            box_before=row["box_before"],
            box_after=row["box_after"],
            interval_days=row["interval_days"],
            created_at=row["created_at"],
        )


@dataclass(frozen=True, slots=True)
class UnverifiedFinding:
    """Uma correção que o verificador barrou por não citar regra curada.

    Não é correção: nada daqui chegou ao aluno, e é justamente por isso que a
    linha existe. O que ela aponta é o avesso de ``Correction`` — um erro no
    texto real de um aluno que ``data/contrastive/`` não cobre, com o span que
    vira ``wrong:`` e a proposta do modelo que vira candidato a ``right:``.

    ``explanation`` é a gramática improvisada que foi descartada. Guardada
    como nota de curadoria, para dizer o que o modelo *achou* que era a regra;
    nunca para mostrar a alguém que está aprendendo.
    """

    id: int
    turn_id: int
    span: str
    span_key: str
    corrected: str
    error_type: str
    explanation: str
    candidates: str
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> UnverifiedFinding:
        return cls(
            id=row["id"],
            turn_id=row["turn_id"],
            span=row["span"],
            span_key=row["span_key"],
            corrected=row["corrected"],
            error_type=row["error_type"],
            explanation=row["explanation"],
            candidates=row["candidates"],
            created_at=row["created_at"],
        )


@dataclass(frozen=True, slots=True)
class GapCount:
    """Uma linha da fila de curadoria: um span sem cobertura, e quantas vezes.

    Agregada por ``span_key``, não por aluno: a base contrastiva é conteúdo de
    produto, revisado item por item, e não progresso pessoal — o span que dois
    perfis da máquina produziram é o *mesmo* buraco, e contá-lo duas vezes é o
    que faz a fila ordenar certo.

    ``last_candidates`` é o que separa os dois trabalhos de curadoria: span que
    apareceu ao lado de uma regra que casou costuma ser padrão a alargar no
    item que já existe; span sozinho costuma ser item novo.
    """

    span_key: str
    times: int
    last_span: str
    last_correction: str
    last_type: str
    last_explanation: str
    last_candidates: str
    last_at: str = ""

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> GapCount:
        return cls(
            span_key=row["span_key"],
            times=row["times"],
            last_span=row["last_span"],
            last_correction=row["last_correction"],
            last_type=row["last_type"],
            last_explanation=row["last_explanation"],
            last_candidates=row["last_candidates"],
            last_at=row["last_at"] if "last_at" in row.keys() else "",
        )
