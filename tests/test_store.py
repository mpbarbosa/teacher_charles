"""Testes da fundação de persistência (E0)."""

from __future__ import annotations

import sqlite3

import pytest

from teacher_charles.store import db
from teacher_charles.store.repo import (
    CorrectionRepo,
    LevelRepo,
    SessionRepo,
    StudentRepo,
    TurnRepo,
)


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


# --- Perfil e nível (E5) -------------------------------------------------- #


def test_new_student_records_the_declared_level(conn):
    student = StudentRepo(conn).get_or_create("ana", cefr_level="B2")
    assert student.cefr_level == "B2"
    assert student.self_declared_level == "B2"
    assert student.level_source == "self"
    assert student.seen_seq > 0


def test_auto_adjustment_does_not_touch_the_declared_level(conn):
    students = StudentRepo(conn)
    student = students.get_or_create("ana", cefr_level="B1")

    moved = students.set_level(student.id, "B2", source="auto")
    assert moved.cefr_level == "B2"
    assert moved.self_declared_level == "B1", "a referência externa tem que sobreviver"
    assert moved.level_source == "auto"


def test_the_student_redefines_the_starting_point(conn):
    students = StudentRepo(conn)
    student = students.get_or_create("ana", cefr_level="B1")
    students.set_level(student.id, "B2", source="auto")

    declared = students.set_level(student.id, "C1", source="self")
    assert (declared.cefr_level, declared.self_declared_level) == ("C1", "C1")
    assert declared.level_source == "self"


def test_invalid_level_source_is_rejected(conn):
    student = StudentRepo(conn).get_or_create("ana")
    with pytest.raises(ValueError):
        StudentRepo(conn).set_level(student.id, "B2", source="vibes")


def test_active_student_is_the_last_one_seen(conn):
    students = StudentRepo(conn)
    assert students.active() is None

    first = students.get_or_create("ana")
    second = students.get_or_create("joao")
    assert students.active().name == second.name

    students.touch(first.id)
    assert students.active().name == first.name


def test_set_level_on_a_missing_student(conn):
    with pytest.raises(LookupError):
        StudentRepo(conn).set_level(999, "B2")


def test_level_signals_are_recorded_and_queried(conn):
    students = StudentRepo(conn)
    student = students.get_or_create("ana")
    session_id = SessionRepo(conn).start(student.id)
    turns = TurnRepo(conn)
    levels = LevelRepo(conn)

    counted = turns.append(session_id, "student", "a long enough message about work")
    skipped = turns.append(session_id, "student", "ok")

    levels.record(
        counted,
        words=20,
        sentences=2,
        long_words=4,
        connectives=1,
        structures=1,
        errors=0,
        score=0.55,
        estimated_level="B1",
        counted=True,
    )
    levels.record(
        skipped,
        words=1,
        sentences=1,
        long_words=0,
        connectives=0,
        structures=0,
        errors=0,
        score=0.0,
        estimated_level="A2",
        counted=False,
    )

    # A linha descartada fica gravada, mas não entra na evidência.
    assert levels.signals_for_turn(skipped).counted is False
    assert levels.recent_levels(student.id) == ["B1"]


def test_level_changes_are_logged_per_student(conn):
    students = StudentRepo(conn)
    ana = students.get_or_create("ana")
    joao = students.get_or_create("joao")
    levels = LevelRepo(conn)

    levels.log_change(ana.id, "B1", "B2", source="auto", reason="4 turns in a row")
    levels.log_change(joao.id, "B1", "A2", source="self")

    changes = levels.changes(ana.id)
    assert [(c.from_level, c.to_level, c.source) for c in changes] == [("B1", "B2", "auto")]
    assert changes[0].reason == "4 turns in a row"
    assert [c.to_level for c in levels.changes(joao.id)] == ["A2"]


def test_level_signals_die_with_the_turn(conn):
    """ON DELETE CASCADE: sinal órfão é lixo que ninguém vai limpar depois."""
    student = StudentRepo(conn).get_or_create("ana")
    session_id = SessionRepo(conn).start(student.id)
    turn_id = TurnRepo(conn).append(session_id, "student", "something long enough here")
    LevelRepo(conn).record(
        turn_id,
        words=20,
        sentences=1,
        long_words=3,
        connectives=0,
        structures=0,
        errors=0,
        score=0.4,
        estimated_level="B1",
        counted=True,
    )

    conn.execute("DELETE FROM turn WHERE id = ?", (turn_id,))
    assert LevelRepo(conn).signals_for_turn(turn_id) is None


# --- Modos e drill (E6) --------------------------------------------------- #


def test_switching_mode_ends_the_session_and_opens_another(conn):
    students = StudentRepo(conn)
    sessions = SessionRepo(conn)
    student = students.get_or_create("ana")

    first = sessions.current_or_start(student.id)
    second = sessions.switch_mode(student.id, "drill", "calque.age-have")

    assert second != first
    assert sessions.get(first).ended_at is not None
    assert sessions.get(second).mode == "drill"
    assert sessions.get(second).drill_rule_id == "calque.age-have"
    assert sessions.current(student.id).id == second


def test_drilled_rules_are_the_most_recent_first(conn):
    students = StudentRepo(conn)
    sessions = SessionRepo(conn)
    student = students.get_or_create("ana")

    sessions.switch_mode(student.id, "drill", "calque.age-have")
    sessions.switch_mode(student.id, "conversation")
    sessions.switch_mode(student.id, "drill", "calque.make-question")

    assert sessions.drilled_rules(student.id) == [
        "calque.make-question",
        "calque.age-have",
    ]


def test_top_rules_counts_only_curated_corrections(conn):
    """Correção sem regra veio do modelo sozinho e não entra no error log agregado."""
    students = StudentRepo(conn)
    student = students.get_or_create("ana")
    session_id = SessionRepo(conn).start(student.id)
    turns = TurnRepo(conn)
    corrections = CorrectionRepo(conn)

    for span in ("I have 35 years", "she has 32 years"):
        turn_id = turns.append(session_id, "student", span)
        corrections.add(
            turn_id,
            span=span,
            corrected="is",
            error_type="calque",
            explanation="curada",
            rule_id="calque.age-have",
        )
    loose = turns.append(session_id, "student", "There have been changes")
    corrections.add(
        loose,
        span="There have been",
        corrected="There has been",
        error_type="grammar",
        explanation="inventada",
        rule_id=None,
    )

    top = CorrectionRepo(conn).top_rules(student.id)
    assert [(r.rule_id, r.times) for r in top] == [("calque.age-have", 2)]
    assert top[0].last_span == "she has 32 years", "o exemplo mais recente é o que o drill mostra"


def test_spans_for_rule_are_the_students_own_words(conn):
    students = StudentRepo(conn)
    student = students.get_or_create("ana")
    turns = TurnRepo(conn)
    corrections = CorrectionRepo(conn)

    # Duas sessões: o error log é do aluno, não da conversa de hoje.
    for index, span in enumerate(("I have 35 years", "my wife has 32 years")):
        session_id = SessionRepo(conn).start(student.id)
        turn_id = turns.append(session_id, "student", span)
        corrections.add(
            turn_id,
            span=span,
            corrected=f"fix {index}",
            error_type="calque",
            explanation="curada",
            rule_id="calque.age-have",
        )

    spans = corrections.spans_for_rule(student.id, "calque.age-have")
    assert spans == ["my wife has 32 years", "I have 35 years"]


# --- Agregação do error log (E7) ------------------------------------------ #


def test_progress_counts_rules_and_loose_corrections(conn):
    students = StudentRepo(conn)
    student = students.get_or_create("ana")
    session_id = SessionRepo(conn).start(student.id)
    turns = TurnRepo(conn)
    corrections = CorrectionRepo(conn)

    for rule in ("calque.age-have", "calque.age-have", "calque.make-question"):
        turn_id = turns.append(session_id, "student", "algo")
        corrections.add(
            turn_id,
            span="x",
            corrected="y",
            error_type="calque",
            explanation="curada",
            rule_id=rule,
        )
    loose = turns.append(session_id, "student", "algo")
    corrections.add(
        loose, span="x", corrected="y", error_type="grammar", explanation="solta", rule_id=None
    )

    assert corrections.progress(student.id) == {
        "corrections": 4,
        "rules": 2,
        "without_rule": 1,
    }


def test_progress_of_a_student_without_corrections(conn):
    student = StudentRepo(conn).get_or_create("ana")
    assert CorrectionRepo(conn).progress(student.id) == {
        "corrections": 0,
        "rules": 0,
        "without_rule": 0,
    }


def test_rule_history_is_most_recent_first(conn):
    student = StudentRepo(conn).get_or_create("ana")
    session_id = SessionRepo(conn).start(student.id)
    turns = TurnRepo(conn)
    corrections = CorrectionRepo(conn)

    ids = []
    for _ in range(3):
        turn_id = turns.append(session_id, "student", "I have 30 years")
        ids.append(
            corrections.add(
                turn_id,
                span="I have 30 years",
                corrected="I am 30",
                error_type="calque",
                explanation="curada",
                rule_id="calque.age-have",
            )
        )

    history = corrections.rule_history(student.id, "calque.age-have")
    assert len(history) == 3
    assert all(isinstance(when, str) and when for when in history)
    assert corrections.rule_history(student.id, "calque.inexistente") == []
