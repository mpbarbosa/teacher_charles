"""Testes do orquestrador de turno (E2). Nenhum usa modelo de verdade."""

from __future__ import annotations

import pytest

from teacher_charles.llm import FakeProvider, ProviderTimeout
from teacher_charles.prompts import PromptLibrary
from teacher_charles.store import db
from teacher_charles.store.repo import SessionRepo, StudentRepo, TurnRepo
from teacher_charles.teaching import Chunk, Done, Failed, TurnRunner


@pytest.fixture
def conn(tmp_path):
    connection = db.init_db(tmp_path / "turn.db")
    yield connection
    connection.close()


@pytest.fixture
def prompts(tmp_path):
    root = tmp_path / "prompts"
    root.mkdir()
    (root / "persona.md").write_text("You are Charles.", encoding="utf-8")
    return PromptLibrary(root)


@pytest.fixture
def session_id(conn):
    student = StudentRepo(conn).get_or_create("ana")
    return SessionRepo(conn).current_or_start(student.id)


def runner_for(provider, prompts, conn, **kwargs) -> TurnRunner:
    return TurnRunner(provider=provider, prompts=prompts, conn=conn, **kwargs)


async def collect(runner, session_id, text):
    return [event async for event in runner.run(session_id, text)]


# --- Caminho felizmente comum --------------------------------------------- #


async def test_turn_streams_then_reports_done(prompts, conn, session_id):
    provider = FakeProvider(replies="I am 30 years old.", chunk_size=5)
    events = await collect(runner_for(provider, prompts, conn), session_id, "I have 30 years")

    chunks = [e for e in events if isinstance(e, Chunk)]
    assert "".join(c.text for c in chunks) == "I am 30 years old."
    assert len(chunks) > 1, "o turno tem que repassar pedaços, não um blob"

    assert isinstance(events[-1], Done)
    done = events[-1]
    assert done.reply == "I am 30 years old."
    assert done.first_chunk_ms >= 0
    assert done.total_ms >= done.first_chunk_ms


async def test_turn_persists_both_sides(prompts, conn, session_id):
    provider = FakeProvider(replies="Nice!")
    await collect(runner_for(provider, prompts, conn), session_id, "I go to gym yesterday")

    turns = TurnRepo(conn).history(session_id)
    assert [(t.role, t.content) for t in turns] == [
        ("student", "I go to gym yesterday"),
        ("teacher", "Nice!"),
    ]


async def test_student_text_is_stripped(prompts, conn, session_id):
    await collect(runner_for(FakeProvider(), prompts, conn), session_id, "   hello   ")
    assert TurnRepo(conn).history(session_id)[0].content == "hello"


async def test_empty_student_text_is_rejected(prompts, conn, session_id):
    runner = runner_for(FakeProvider(), prompts, conn)
    with pytest.raises(ValueError):
        await collect(runner, session_id, "   ")
    assert TurnRepo(conn).history(session_id) == []


# --- O prompt ------------------------------------------------------------- #


async def test_persona_is_the_first_message(prompts, conn, session_id):
    provider = FakeProvider()
    await collect(runner_for(provider, prompts, conn), session_id, "hi")

    first = provider.last_call.messages[0]
    assert first.role == "system"
    assert first.content == "You are Charles."


async def test_history_becomes_user_and_assistant_turns(prompts, conn, session_id):
    provider = FakeProvider(replies=["first reply", "second reply"])
    runner = runner_for(provider, prompts, conn)

    await collect(runner, session_id, "turn one")
    await collect(runner, session_id, "turn two")

    roles_and_text = [(m.role, m.content) for m in provider.last_call.messages]
    assert roles_and_text == [
        ("system", "You are Charles."),
        ("user", "turn one"),
        ("assistant", "first reply"),
        ("user", "turn two"),
    ]


async def test_student_message_is_not_duplicated(prompts, conn, session_id):
    """A fala já está no histórico quando o prompt é montado; não deve entrar duas vezes."""
    provider = FakeProvider()
    await collect(runner_for(provider, prompts, conn), session_id, "only once")

    contents = [m.content for m in provider.last_call.messages]
    assert contents.count("only once") == 1


async def test_history_window_is_capped(prompts, conn, session_id):
    """Histórico longo atrasa a primeira palavra; a janela tem que limitar."""
    provider = FakeProvider(replies="ok")
    runner = runner_for(provider, prompts, conn, history_turns=4)

    for i in range(5):
        await collect(runner, session_id, f"message {i}")

    messages = provider.last_call.messages
    assert messages[0].role == "system"
    # 1 system + 4 do histórico
    assert len(messages) == 5
    assert messages[-1].content == "message 4"
    assert "message 0" not in [m.content for m in messages]


async def test_persona_edit_takes_effect_next_turn(prompts, conn, session_id):
    import os

    provider = FakeProvider(replies=["a", "b"])
    runner = runner_for(provider, prompts, conn)
    await collect(runner, session_id, "first")

    path = prompts.path_for("persona")
    path.write_text("You are someone else.", encoding="utf-8")
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 10))

    await collect(runner, session_id, "second")
    assert provider.last_call.messages[0].content == "You are someone else."


# --- Falhas --------------------------------------------------------------- #


async def test_provider_failure_keeps_student_text(prompts, conn, session_id):
    """Se a geração falha, o que o aluno disse não se perde."""
    provider = FakeProvider(error=ProviderTimeout("muito lento"))
    events = await collect(runner_for(provider, prompts, conn), session_id, "I have 30 years")

    assert isinstance(events[-1], Failed)
    assert "muito lento" in events[-1].message

    turns = TurnRepo(conn).history(session_id)
    assert [(t.role, t.content) for t in turns] == [("student", "I have 30 years")]


async def test_empty_reply_is_a_failure_not_a_silent_turn(prompts, conn, session_id):
    provider = FakeProvider(replies="   ")
    events = await collect(runner_for(provider, prompts, conn), session_id, "hi")

    assert isinstance(events[-1], Failed)
    # Resposta vazia não deve ser gravada como turno do professor.
    assert [t.role for t in TurnRepo(conn).history(session_id)] == ["student"]


async def test_blank_chunks_are_skipped(prompts, conn, session_id):
    """O Ollama manda um pedaço vazio junto do `done`; não deve virar evento."""
    provider = FakeProvider(replies="ab")
    events = await collect(runner_for(provider, prompts, conn), session_id, "hi")
    assert all(isinstance(e, Chunk) and e.text for e in events[:-1])
