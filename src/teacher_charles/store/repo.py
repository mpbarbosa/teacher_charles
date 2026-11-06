"""Acesso aos dados.

O que a E0 precisa para provar que o banco funciona de ponta a ponta, o que a
E2 usa para persistir histórico de conversa e o que a E4 usa para persistir
correção. Agregações do error log são da E7.
"""

from __future__ import annotations

import sqlite3

from .models import CEFR_LEVELS, MODES, Correction, Student, Turn


class StudentRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get_or_create(self, name: str, cefr_level: str = "B1") -> Student:
        if cefr_level not in CEFR_LEVELS:
            raise ValueError(f"nível CEFR inválido: {cefr_level!r} (use um de {CEFR_LEVELS})")

        row = self._conn.execute("SELECT * FROM student WHERE name = ?", (name,)).fetchone()
        if row is not None:
            return Student.from_row(row)

        self._conn.execute(
            "INSERT INTO student (name, cefr_level) VALUES (?, ?)",
            (name, cefr_level),
        )
        row = self._conn.execute("SELECT * FROM student WHERE name = ?", (name,)).fetchone()
        return Student.from_row(row)

    def set_level(self, student_id: int, cefr_level: str) -> Student:
        if cefr_level not in CEFR_LEVELS:
            raise ValueError(f"nível CEFR inválido: {cefr_level!r} (use um de {CEFR_LEVELS})")

        cur = self._conn.execute(
            "UPDATE student SET cefr_level = ?, updated_at = datetime('now') WHERE id = ?",
            (cefr_level, student_id),
        )
        if cur.rowcount == 0:
            raise LookupError(f"aluno {student_id} não existe")

        row = self._conn.execute("SELECT * FROM student WHERE id = ?", (student_id,)).fetchone()
        return Student.from_row(row)

    def count(self) -> int:
        return self._conn.execute("SELECT count(*) AS n FROM student").fetchone()["n"]


class SessionRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def start(self, student_id: int, mode: str = "conversation") -> int:
        if mode not in MODES:
            raise ValueError(f"modo inválido: {mode!r} (use um de {MODES})")
        cur = self._conn.execute(
            "INSERT INTO session (student_id, mode) VALUES (?, ?)",
            (student_id, mode),
        )
        return int(cur.lastrowid)

    def current_or_start(self, student_id: int, mode: str = "conversation") -> int:
        """A sessão aberta do aluno, ou uma nova.

        O aluno não deve precisar "criar uma sessão" para começar a falar;
        abrir a página basta.
        """
        row = self._conn.execute(
            """
            SELECT id FROM session
            WHERE student_id = ? AND ended_at IS NULL
            ORDER BY id DESC LIMIT 1
            """,
            (student_id,),
        ).fetchone()
        if row is not None:
            return int(row["id"])
        return self.start(student_id, mode)

    def end(self, session_id: int) -> None:
        self._conn.execute(
            "UPDATE session SET ended_at = datetime('now') WHERE id = ?",
            (session_id,),
        )


class TurnRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def append(self, session_id: int, role: str, content: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO turn (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        return int(cur.lastrowid)

    def history(self, session_id: int, limit: int = 50) -> list[Turn]:
        """Os ``limit`` turnos mais recentes, em ordem cronológica."""
        rows = self._conn.execute(
            "SELECT * FROM turn WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [Turn.from_row(r) for r in reversed(rows)]


class CorrectionRepo:
    """As correções, como dado contável.

    O princípio da Fase 1 em uma tabela: a correção que o aluno recebe não é só
    prosa na resposta do professor, é linha agregável. Sem isto não há error log
    (E7), drill a partir do erro real (E6) nem repetição espaçada (Fase 2).

    ``turn_id`` aponta para o turno **do aluno** — o turno que contém o erro —,
    não para a resposta do professor.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(
        self,
        turn_id: int,
        *,
        span: str,
        corrected: str,
        error_type: str,
        explanation: str,
        rule_id: str | None = None,
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO correction (turn_id, span, corrected, error_type, rule_id, explanation)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (turn_id, span, corrected, error_type, rule_id, explanation),
        )
        return int(cur.lastrowid)

    def for_turn(self, turn_id: int) -> list[Correction]:
        rows = self._conn.execute(
            "SELECT * FROM correction WHERE turn_id = ? ORDER BY id",
            (turn_id,),
        ).fetchall()
        return [Correction.from_row(row) for row in rows]

    def for_session(self, session_id: int) -> list[Correction]:
        rows = self._conn.execute(
            """
            SELECT c.* FROM correction c
            JOIN turn t ON t.id = c.turn_id
            WHERE t.session_id = ?
            ORDER BY c.id
            """,
            (session_id,),
        ).fetchall()
        return [Correction.from_row(row) for row in rows]

    def count(self) -> int:
        return self._conn.execute("SELECT count(*) AS n FROM correction").fetchone()["n"]

    def count_with_rule(self) -> int:
        """Quantas correções têm regra curada por trás.

        A razão entre este número e ``count()`` é o que a E4 mede: quanto da
        correção vem da base e quanto vem do modelo solto.
        """
        return self._conn.execute(
            "SELECT count(*) AS n FROM correction WHERE rule_id IS NOT NULL"
        ).fetchone()["n"]
