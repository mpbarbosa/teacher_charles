"""Acesso aos dados.

Só o que a E0 precisa para provar que o banco funciona de ponta a ponta, mais
o mínimo que a E2 vai usar para persistir histórico de conversa. Agregações do
error log são da E7.
"""

from __future__ import annotations

import sqlite3

from .models import CEFR_LEVELS, MODES, Student, Turn


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
