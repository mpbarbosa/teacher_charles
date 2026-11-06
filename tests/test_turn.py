"""Testes do orquestrador de turno (E2). Nenhum usa modelo de verdade."""

from __future__ import annotations

import asyncio
import json

import pytest

from teacher_charles.config import REPO_ROOT
from teacher_charles.llm import FakeProvider, ProviderTimeout
from teacher_charles.llm.base import GenerationOptions
from teacher_charles.prompts import PromptLibrary
from teacher_charles.store import db
from teacher_charles.store.repo import CorrectionRepo, SessionRepo, StudentRepo, TurnRepo
from teacher_charles.teaching import Analyzed, Analyzer, Chunk, Done, Failed, TurnRunner
from teacher_charles.teaching.contrastive import load_base
from teacher_charles.teaching.turn import MAX_HINT_RULES


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


# --- O turno duplo (E4) --------------------------------------------------- #
#
# A análise roda em paralelo com a resposta. Sequencial, ela custaria segundos
# antes da primeira palavra do professor — nesta máquina, o turno inteiro do
# aluno. Quem informa a resposta é a parte determinística: a base contrastiva
# casa por padrão e entra no prompt como dica, sem esperar modelo nenhum.


def analyzer_and_provider(reply: str, **kwargs) -> tuple[Analyzer, FakeProvider]:
    provider = FakeProvider(replies=reply)
    analyzer = Analyzer(
        provider=provider,
        prompts=PromptLibrary(REPO_ROOT / "prompts"),
        base=load_base(REPO_ROOT / "data" / "contrastive"),
        **kwargs,
    )
    return analyzer, provider


def analyzer_for(reply: str, **kwargs) -> Analyzer:
    return analyzer_and_provider(reply, **kwargs)[0]


AGE_ERROR = json.dumps(
    {
        "errors": [
            {
                "span": "I have 30 years",
                "type": "grammar",
                "correction": "I am 30 years old",
                "rule_id": None,
                "explanation": "explicação do modelo, que a regra curada substitui",
            }
        ]
    }
)


class SlowFake(FakeProvider):
    """Devolve a mesma coisa, mas cedendo o event loop entre os pedaços.

    É o que um stream HTTP de verdade faz. Sem isso o turno roda inteiro sem
    nunca suspender, e a análise paralela só pegaria o loop no fim — o teste
    passaria a medir o escalonador em vez do desenho.
    """

    async def stream(self, messages, options=None):  # type: ignore[override]
        async for piece in super().stream(messages, options or GenerationOptions()):
            await asyncio.sleep(0)
            yield piece


async def test_analysis_event_comes_before_done(prompts, conn, session_id):
    runner = runner_for(
        FakeProvider(replies="You mean you are 30. What did you do?"),
        prompts,
        conn,
        analyzer=analyzer_for(AGE_ERROR),
    )
    events = await collect(runner, session_id, "I have 30 years")

    kinds = [type(e).__name__ for e in events]
    assert kinds.index("Analyzed") < kinds.index("Done")

    analyzed = next(e for e in events if isinstance(e, Analyzed))
    (finding,) = analyzed.analysis.findings
    # A regra curada vence a explicação do modelo.
    assert finding.rule_id == "calque.age-have"
    assert "age is something you are" in finding.explanation


async def test_analysis_waits_for_the_last_chunk_by_default(prompts, conn, session_id):
    """Conversa primeiro: a análise não pode disputar o modelo com a resposta.

    Medido (plano §5.2): disparadas juntas, o Ollama atende uma por vez e a
    primeira palavra do professor sai em 11,2s em vez de 0,7s. Este teste
    afirma a ordem, que é a parte que o código controla.
    """
    analyzer, analysis_provider = analyzer_and_provider(AGE_ERROR)
    runner = runner_for(
        SlowFake(replies="You are 30 years old. " * 6, chunk_size=4),
        prompts,
        conn,
        analyzer=analyzer,
    )

    chunks = 0
    async for event in runner.run(session_id, "I have 30 years"):
        if isinstance(event, Chunk):
            chunks += 1
            assert analysis_provider.call_count == 0, (
                "a análise começou antes de a resposta terminar"
            )

    assert chunks > 1
    assert analysis_provider.call_count == 1


async def test_parallel_mode_lands_the_analysis_mid_reply(prompts, conn, session_id):
    """O modo paralelo continua existindo: a premissa que o derrubou é desta
    máquina, não do desenho. Num provider que atenda dois pedidos ao mesmo
    tempo, disparar as duas juntas volta a ser o melhor arranjo."""
    runner = runner_for(
        SlowFake(replies="You are 30 years old. " * 6, chunk_size=4),
        prompts,
        conn,
        analyzer=analyzer_for(AGE_ERROR),
        analysis_after_reply=False,
    )
    events = await collect(runner, session_id, "I have 30 years")

    kinds = [type(e).__name__ for e in events]
    analyzed_at = kinds.index("Analyzed")
    last_chunk_at = max(i for i, k in enumerate(kinds) if k == "Chunk")
    assert analyzed_at < last_chunk_at, "a análise só chegou depois da resposta inteira"


async def test_corrections_are_persisted_against_the_student_turn(prompts, conn, session_id):
    runner = runner_for(
        FakeProvider(replies="Right!"), prompts, conn, analyzer=analyzer_for(AGE_ERROR)
    )
    events = await collect(runner, session_id, "I have 30 years")

    analyzed = next(e for e in events if isinstance(e, Analyzed))
    assert len(analyzed.correction_ids) == 1

    rows = CorrectionRepo(conn).for_turn(analyzed.student_turn_id)
    assert [(r.span, r.corrected, r.rule_id, r.error_type) for r in rows] == [
        ("I have 30 years", "I am 30 years old", "calque.age-have", "calque")
    ]
    assert CorrectionRepo(conn).count_with_rule() == 1

    # Aponta para o turno do aluno, que é o que contém o erro.
    student_turn = TurnRepo(conn).history(session_id)[0]
    assert analyzed.student_turn_id == student_turn.id


async def test_done_reports_what_the_analysis_cost(prompts, conn, session_id):
    runner = runner_for(FakeProvider(replies="ok"), prompts, conn, analyzer=analyzer_for(AGE_ERROR))
    events = await collect(runner, session_id, "I have 30 years")
    assert events[-1].analysis_ms > 0


async def test_degraded_analysis_still_closes_the_turn(prompts, conn, session_id):
    runner = runner_for(
        FakeProvider(replies="Tell me more."),
        prompts,
        conn,
        analyzer=analyzer_for("desculpe, não consigo gerar JSON"),
    )
    events = await collect(runner, session_id, "I have 30 years")

    assert isinstance(events[-1], Done)
    analyzed = next(e for e in events if isinstance(e, Analyzed))
    assert analyzed.analysis.degraded
    assert analyzed.correction_ids == ()
    assert CorrectionRepo(conn).count() == 0
    # A conversa não foi afetada.
    assert events[-1].reply == "Tell me more."


async def test_unexpected_analyzer_crash_does_not_break_the_turn(prompts, conn, session_id):
    class Exploding(Analyzer):
        async def analyze(self, student_text, cefr_level="B1"):  # type: ignore[override]
            raise RuntimeError("bug no analisador")

    analyzer = Exploding(
        provider=FakeProvider(),
        prompts=PromptLibrary(REPO_ROOT / "prompts"),
        base=load_base(REPO_ROOT / "data" / "contrastive"),
    )
    runner = runner_for(FakeProvider(replies="Hello!"), prompts, conn, analyzer=analyzer)
    events = await collect(runner, session_id, "I have 30 years")

    assert isinstance(events[-1], Done)
    assert not any(isinstance(e, Analyzed) for e in events)
    assert events[-1].reply == "Hello!"


async def test_failed_reply_leaves_no_pending_analysis(prompts, conn, session_id):
    runner = runner_for(
        FakeProvider(error=ProviderTimeout("muito lento")),
        prompts,
        conn,
        analyzer=analyzer_for(AGE_ERROR),
    )
    events = await collect(runner, session_id, "I have 30 years")

    assert isinstance(events[-1], Failed)
    assert not any(isinstance(e, Analyzed) for e in events)
    assert CorrectionRepo(conn).count() == 0


# --- A dica curada no prompt da resposta ---------------------------------- #


async def test_rule_hint_goes_last_in_the_prompt(prompts, conn, session_id):
    """No fim, e não logo após a persona, por causa do cache de prefixo.

    O Ollama cacheia o prefixo comum com a chamada anterior. Uma dica posta
    depois da persona empurraria o histórico inteiro para fora desse prefixo e
    faria o modelo reavaliar a conversa toda a cada turno.
    """
    provider = FakeProvider(replies="ok")
    runner = runner_for(provider, prompts, conn, analyzer=analyzer_for(AGE_ERROR))
    await collect(runner, session_id, "I have 30 years")

    messages = provider.calls_matching("You are Charles")[-1].messages
    assert messages[0].content == "You are Charles."
    assert messages[-1].role == "system"
    assert "age is something you are" in messages[-1].content
    assert '"I have 30 years"' in messages[-1].content
    # A fala do aluno continua sendo a última mensagem de usuário.
    assert [m.role for m in messages] == ["system", "user", "system"]


async def test_no_hint_when_the_base_matches_nothing(prompts, conn, session_id):
    provider = FakeProvider(replies="ok")
    runner = runner_for(provider, prompts, conn, analyzer=analyzer_for(AGE_ERROR))
    await collect(runner, session_id, "I am 30 years old and I like my job")

    messages = provider.calls_matching("You are Charles")[-1].messages
    assert messages[-1].role == "user"


async def test_hint_is_capped(prompts, conn, session_id):
    provider = FakeProvider(replies="ok")
    runner = runner_for(provider, prompts, conn, analyzer=analyzer_for(AGE_ERROR))
    await collect(
        runner,
        session_id,
        "I have 30 years, it depends of you, explain me this, people is nice",
    )

    hint = provider.calls_matching("You are Charles")[-1].messages[-1].content
    assert hint.count("\n- ") == MAX_HINT_RULES


async def test_no_analyzer_keeps_the_e2_turn(prompts, conn, session_id):
    """Sem analisador o turno volta a ser conversa e nada mais."""
    provider = FakeProvider(replies="Hello!")
    events = await collect(runner_for(provider, prompts, conn), session_id, "I have 30 years")

    assert not any(isinstance(e, Analyzed) for e in events)
    assert events[-1].analysis_ms == 0.0
    assert [m.role for m in provider.last_call.messages] == ["system", "user"]
