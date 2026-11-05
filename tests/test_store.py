"""Testes da fundação de persistência (E0)."""

from __future__ import annotations

import sqlite3

import pytest

from teacher_charles.store import db
from teacher_charles.store.repo import SessionRepo, StudentRepo, TurnRepo


@pytest.fixture
def conn(tmp_path):
    connection = db.init_db(tmp_path / "test.db")
    yield connection
    connection.close()


def test_migrations_apply_once(tmp_path):
    path = tmp_path / "m.db"
    first = db.connect(path)
    applied = db.migrate(first)
    assert "001_initial.sql" in applied

    # Rodar de novo não deve reaplicar nada.
    assert db.migrate(first) == []
    assert db.pending_migrations(first) == []
    first.close()

    # Nem numa conexão nova sobre o mesmo arquivo.
    second = db.init_db(path)
    assert db.pending_migrations(second) == []
    second.close()


def test_migration_is_atomic_on_failure(tmp_path, monkeypatch):
    """Um script que falha no meio não deve deixar tabelas pela metade.

    Isto é o que justifica o BEGIN ficar dentro do script em ``migrate``:
    ``executescript`` encerra transação aberta de fora, então sem esse cuidado
    o DDL parcial sobreviveria.
    """
    bad = tmp_path / "002_bad.sql"
    bad.write_text(
        "CREATE TABLE half_done (x INT);\nCREATE TABLE half_done (x INT);\n",
        encoding="utf-8",
    )

    conn = db.init_db(tmp_path / "atomic.db")
    monkeypatch.setattr(db, "pending_migrations", lambda _conn: [bad])

    with pytest.raises(sqlite3.OperationalError):
        db.migrate(conn)

    tables = {
        row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert "half_done" not in tables
    assert not conn.in_transaction
    conn.close()


def test_foreign_keys_are_enforced(conn):
    """O ON DELETE CASCADE do esquema só vale com o pragma ligado."""
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO session (student_id, mode) VALUES (9999, 'conversation')")


def test_cascade_removes_dependent_rows(conn):
    student = StudentRepo(conn).get_or_create("ana")
    session_id = SessionRepo(conn).start(student.id)
    TurnRepo(conn).append(session_id, "student", "I have 30 years")

    conn.execute("DELETE FROM student WHERE id = ?", (student.id,))

    assert conn.execute("SELECT count(*) AS n FROM session").fetchone()["n"] == 0
    assert conn.execute("SELECT count(*) AS n FROM turn").fetchone()["n"] == 0


def test_get_or_create_is_idempotent(conn):
    repo = StudentRepo(conn)
    first = repo.get_or_create("marcelo")
    second = repo.get_or_create("marcelo")

    assert first.id == second.id
    assert repo.count() == 1
    assert first.cefr_level == "B1"  # o default do produto é B1+


def test_set_level(conn):
    repo = StudentRepo(conn)
    student = repo.get_or_create("marcelo")

    updated = repo.set_level(student.id, "B2")
    assert updated.cefr_level == "B2"

    with pytest.raises(ValueError):
        repo.set_level(student.id, "A1")  # fora do público-alvo (decisão 1.1)

    with pytest.raises(LookupError):
        repo.set_level(4242, "B2")


def test_invalid_level_rejected_on_create(conn):
    with pytest.raises(ValueError):
        StudentRepo(conn).get_or_create("joao", cefr_level="Z9")


def test_invalid_mode_rejected(conn):
    student = StudentRepo(conn).get_or_create("ana")
    with pytest.raises(ValueError):
        SessionRepo(conn).start(student.id, mode="telepathy")


def test_history_is_chronological_and_capped(conn):
    student = StudentRepo(conn).get_or_create("ana")
    session_id = SessionRepo(conn).start(student.id)
    turns = TurnRepo(conn)

    for i in range(5):
        turns.append(session_id, "student", f"msg {i}")

    assert [t.content for t in turns.history(session_id)] == [f"msg {i}" for i in range(5)]
    # O limite pega os mais recentes, mas devolve em ordem de leitura.
    assert [t.content for t in turns.history(session_id, limit=2)] == ["msg 3", "msg 4"]


def test_correction_allows_null_rule_id(conn):
    """Correção sem regra curada é um caso legítimo, e a E4 precisa distingui-lo."""
    student = StudentRepo(conn).get_or_create("ana")
    session_id = SessionRepo(conn).start(student.id)
    turn_id = TurnRepo(conn).append(session_id, "student", "I have 30 years")

    conn.execute(
        """
        INSERT INTO correction (turn_id, span, corrected, error_type, rule_id, explanation)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (turn_id, "I have 30 years", "I am 30 years old", "calque", None, "Age uses 'to be'."),
    )
    row = conn.execute("SELECT * FROM correction").fetchone()
    assert row["rule_id"] is None
