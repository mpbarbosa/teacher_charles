"""E12's tables and the ``rule_review`` rebuild (``006_read_aloud.sql``).

The rebuild is the part that can lose data, so it is tested against a database
that already has E13 reviews in it: every row must come through, and the
schema must now hold that a review points at a turn **or** a read-aloud
session, exactly one, with one review per session.
"""

from __future__ import annotations

import sqlite3

import pytest

from teacher_charles.store.db import MIGRATIONS_DIR, connect, migrate
from teacher_charles.store.repo import (
    ReadAloudRepo,
    ScheduleRepo,
    SessionRepo,
    StudentRepo,
    TurnRepo,
)
from teacher_charles.teaching.schedule import Scheduler


def apply_up_to(conn: sqlite3.Connection, last: str) -> None:
    """The migrations through ``last``, as an install from before E12 had them."""
    conn.execute(
        "CREATE TABLE schema_migrations (name TEXT PRIMARY KEY, "
        "applied_at TEXT NOT NULL DEFAULT (datetime('now')))"
    )
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        if path.name > last:
            break
        conn.executescript("BEGIN;\n" + path.read_text(encoding="utf-8") + "\nCOMMIT;")
        conn.execute("INSERT INTO schema_migrations (name) VALUES (?)", (path.name,))


@pytest.fixture
def db(tmp_path):
    conn = connect(tmp_path / "e12.db")
    migrate(conn)
    yield conn
    conn.close()


def test_the_rebuild_keeps_every_e13_review(tmp_path):
    conn = connect(tmp_path / "old.db")
    apply_up_to(conn, "005_unverified.sql")
    student = StudentRepo(conn).get_or_create("ana")
    session = SessionRepo(conn).start(student.id, mode="drill")
    turn = TurnRepo(conn).append(session, "student", "I have 30 years")
    conn.execute(
        """
        INSERT INTO rule_review (student_id, rule_id, turn_id, hit, source,
                                 box_before, box_after, interval_days)
        VALUES (?, 'calque.age-have', ?, 0, 'lapse', 0, 0, 0.5),
               (?, 'calque.age-have', ?, 1, 'drill', 0, 1, 1.0)
        """,
        (student.id, turn, student.id, turn),
    )
    before = [tuple(r) for r in conn.execute("SELECT * FROM rule_review ORDER BY id")]

    assert migrate(conn) == ["006_read_aloud.sql"]
    after = conn.execute("SELECT * FROM rule_review ORDER BY id").fetchall()
    assert [
        (
            r["id"],
            r["student_id"],
            r["rule_id"],
            r["turn_id"],
            r["hit"],
            r["source"],
            r["box_before"],
            r["box_after"],
            r["interval_days"],
            r["created_at"],
        )
        for r in after
    ] == before
    assert all(r["pronunciation_session_id"] is None for r in after)
    # And the next review still gets a fresh id after the copied ones.
    review = Scheduler(conn).review(student.id, "calque.age-have", turn, hit=True)
    assert ScheduleRepo(conn).reviews(student.id)[0].id == 3
    assert review.source == "drill"
    conn.close()


def test_a_pronunciation_review_has_a_session_and_no_turn(db):
    student = StudentRepo(db).get_or_create("ana")
    session = ReadAloudRepo(db).start(student.id, "pron.th-voiced", 0, "student")
    Scheduler(db).review(
        student.id,
        "pron.th-voiced",
        None,
        hit=False,
        source="pronunciation",
        pronunciation_session_id=session,
    )
    (review,) = ScheduleRepo(db).reviews(student.id)
    assert (review.turn_id, review.pronunciation_session_id) == (None, session)
    assert review.source == "pronunciation"


@pytest.mark.parametrize(
    ("turn", "session", "source"),
    [
        (None, None, "drill"),  # no evidence at all
        ("turn", "session", "drill"),  # two pieces of evidence
        ("turn", None, "pronunciation"),  # a pronunciation review on a turn
        (None, "session", "drill"),  # a drill review on a read-aloud session
    ],
)
def test_the_schema_holds_where_a_review_points(db, turn, session, source):
    student = StudentRepo(db).get_or_create("ana")
    turn_id = TurnRepo(db).append(SessionRepo(db).start(student.id), "student", "hi")
    session_id = ReadAloudRepo(db).start(student.id, "pron.th-voiced", 0, "student")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            """
            INSERT INTO rule_review (student_id, rule_id, turn_id, pronunciation_session_id,
                                     hit, source, box_before, box_after, interval_days)
            VALUES (?, 'pron.th-voiced', ?, ?, 1, ?, 0, 1, 1.0)
            """,
            (
                student.id,
                turn_id if turn else None,
                session_id if session else None,
                source,
            ),
        )


def test_one_review_per_session_is_held_by_the_schema(db):
    student = StudentRepo(db).get_or_create("ana")
    session = ReadAloudRepo(db).start(student.id, "pron.th-voiced", 0, "student")
    scheduler = Scheduler(db)
    kwargs = {"hit": True, "source": "pronunciation", "pronunciation_session_id": session}
    scheduler.review(student.id, "pron.th-voiced", None, **kwargs)
    with pytest.raises(sqlite3.IntegrityError):
        scheduler.review(student.id, "pron.th-voiced", None, **kwargs)


def test_a_session_ends_once(db):
    student = StudentRepo(db).get_or_create("ana")
    repo = ReadAloudRepo(db)
    session = repo.start(student.id, "pron.th-voiced", 0, "schedule")
    assert repo.end(session, "miss") is True
    assert repo.end(session, "hit") is False
    assert repo.get(session).outcome == "miss"


def test_a_silenced_verdict_cannot_be_stored(db):
    """Metodo §3: gates 1 and 2 record nothing, and the schema says so too."""
    student = StudentRepo(db).get_or_create("ana")
    repo = ReadAloudRepo(db)
    session = repo.get(repo.start(student.id, "pron.th-voiced", 0, "student"))
    with pytest.raises(sqlite3.IntegrityError):
        repo.add(
            session,
            pass_="unmonitored",
            verdict="silence",
            opportunities=4,
            errors=0,
            findings="[]",
            model="fake",
            duration_s=1.0,
            speech_from_s=None,
            speech_to_s=None,
        )


def test_the_guard_reads_any_named_take_of_either_pass(db):
    student = StudentRepo(db).get_or_create("ana")
    repo = ReadAloudRepo(db)
    session = repo.get(repo.start(student.id, "pron.th-voiced", 0, "student"))
    common = {
        "opportunities": 4,
        "findings": "[]",
        "model": "fake",
        "duration_s": 3.0,
        "speech_from_s": 1.0,
        "speech_to_s": 2.0,
    }
    repo.add(session, pass_="unmonitored", verdict="no-error", errors=0, **common)
    assert not repo.ever_named(student.id, "pron.th-voiced")
    repo.add(session, pass_="deliberate", verdict="correction", errors=2, **common)
    assert repo.ever_named(student.id, "pron.th-voiced")
    assert repo.has_deliberate(student.id, "pron.th-voiced")
    # The deliberate take is calibration: it is not a mistake on /progress.
    (count,) = repo.by_rule(student.id)
    assert (count.takes, count.named) == (1, 0)


def test_a_session_with_no_judged_take_is_not_practice_on_progress(db):
    """Opened by the schedule and walked away from: it reads as 0/0, which is nothing."""
    student = StudentRepo(db).get_or_create("ana")
    repo = ReadAloudRepo(db)
    repo.start(student.id, "pron.h-dropping", 0, "schedule")
    session = repo.get(repo.start(student.id, "pron.th-voiced", 0, "student"))
    repo.start(student.id, "pron.th-voiced", 0, "schedule")
    repo.add(
        session,
        pass_="unmonitored",
        verdict="correction",
        errors=1,
        opportunities=4,
        findings="[]",
        model="fake",
        duration_s=3.0,
        speech_from_s=1.0,
        speech_to_s=2.0,
    )
    (count,) = repo.by_rule(student.id)
    assert (count.rule_id, count.sessions, count.takes, count.named) == (
        "pron.th-voiced",
        1,
        1,
        1,
    )
