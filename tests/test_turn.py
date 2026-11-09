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
from teacher_charles.store.models import MODES
from teacher_charles.store.repo import CorrectionRepo, SessionRepo, StudentRepo, TurnRepo
from teacher_charles.teaching import (
    Analyzed,
    Analyzer,
    Chunk,
    Done,
    Failed,
    Leveled,
    Phonemized,
    Phonetics,
    PhoneticsUnavailable,
    TurnRunner,
)
from teacher_charles.teaching.capabilities import partial_name as app_partial_name
from teacher_charles.teaching.contrastive import load_base
from teacher_charles.teaching.level import AGREEING_TURNS, LevelTracker
from teacher_charles.teaching.level import partial_name as level_partial_name
from teacher_charles.teaching.modes import DrillPlanner
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


async def collect(runner, session_id, text, student=None):
    return [event async for event in runner.run(session_id, text, student)]


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
        async def analyze(  # type: ignore[override]
            self, student_text, cefr_level="B1", max_findings=None
        ):
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


async def test_rule_hint_rides_with_the_student_message(prompts, conn, session_id):
    """A dica vai no fim do prompt, **dentro** da fala do aluno.

    Como mensagem de sistema no fim da lista ela não ficava no fim do prompt:
    o template de chat junta as mensagens de sistema no topo. Medido na E7
    (plano §5.5) — o histórico inteiro parava de ser cacheado, o total
    cacheado ficava preso nos 674 tokens do prefixo estável e a primeira
    palavra ia de 2,8s para 9,1s em seis turnos. Dentro da fala do aluno, a
    dica fica onde se queria e o histórico volta ao cache.
    """
    provider = FakeProvider(replies="ok")
    runner = runner_for(provider, prompts, conn, analyzer=analyzer_for(AGE_ERROR))
    await collect(runner, session_id, "I have 30 years")

    messages = provider.calls_matching("You are Charles")[-1].messages
    assert messages[0].content == "You are Charles."
    # Nada de sistema depois da persona: é isso que o template moveria.
    assert [m.role for m in messages] == ["system", "user"]

    last = messages[-1]
    assert last.content.startswith("I have 30 years")
    assert "age is something you are" in last.content
    assert '"I have 30 years"' in last.content


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


# --- O ajuste por nível (E5) ---------------------------------------------- #
#
# Estes testes usam os prompts **de verdade**, de `prompts/`: o critério de
# aceite da E5 é sobre o produto, não sobre um partial inventado no tmp_path.


@pytest.fixture
def real_prompts():
    return PromptLibrary(REPO_ROOT / "prompts")


@pytest.fixture
def student(conn):
    return StudentRepo(conn).get_or_create("ana", cefr_level="B1")


def test_level_prompt_is_a_system_message_of_its_own(real_prompts, conn, session_id):
    """Depois da persona, nunca dentro dela.

    O Ollama cacheia o prefixo do prompt. Como mensagem separada, o ajuste de
    nível só invalida o cache quando o nível muda; interpolado na persona,
    invalidaria a cada turno e devolveria os segundos que o aquecimento
    economizou (plano §1.4).

    Terceira mensagem, não segunda: as capacidades do aplicativo entram antes,
    porque não mudam enquanto o servidor está de pé.
    """
    runner = runner_for(FakeProvider(), real_prompts, conn)
    messages = runner.build_messages(session_id, "hello there", "B2")

    assert messages[0].content == real_prompts.get("persona")
    assert messages[2].role == "system"
    assert messages[2].content == real_prompts.get(level_partial_name("B2"))
    # E nada do nível foi interpolado na persona.
    assert "B2" not in messages[0].content


def test_changing_the_level_changes_the_prompt(real_prompts, conn, session_id):
    """O critério de aceite da E5, na parte que não depende do modelo.

    Dois níveis têm que produzir prompts diferentes, com a persona idêntica nos
    dois: o que muda é o ajuste, não quem o Charles é.
    """
    runner = runner_for(FakeProvider(), real_prompts, conn)

    beginner = runner.build_messages(session_id, "hello there", "A2")
    advanced = runner.build_messages(session_id, "hello there", "C1")

    assert beginner[0] == advanced[0], "a persona não pode mudar com o nível"
    assert beginner[1] == advanced[1], "as capacidades do app não mudam com o nível"
    assert beginner[2] != advanced[2]
    assert "A2" in beginner[2].content and "C1" in advanced[2].content
    # E o ajuste de verdade chega ao modelo, não só ao build_messages.
    assert len({len(beginner), len(advanced)}) == 1


async def test_the_student_level_reaches_the_model(real_prompts, conn, session_id, student):
    provider = FakeProvider()
    runner = runner_for(provider, real_prompts, conn)

    await collect(runner, session_id, "I work with marketing here", student)

    system = provider.system_prompt_of_last_call()
    assert real_prompts.get(level_partial_name("B1")) in system
    assert real_prompts.get(level_partial_name("C1")) not in system


async def test_turn_without_student_falls_back_to_the_entry_level(real_prompts, conn, session_id):
    """Sem perfil, o turno é o da E2 com o registro de entrada do produto."""
    provider = FakeProvider()
    await collect(runner_for(provider, real_prompts, conn), session_id, "hello there")
    assert real_prompts.get(level_partial_name("B1")) in provider.system_prompt_of_last_call()


async def test_missing_level_partial_does_not_break_the_turn(prompts, conn, session_id, student):
    """`prompts` (o fixture do tmp_path) só tem a persona.

    Arquivo de prompt que falta não deveria derrubar a aula: o turno volta a
    ser o da E2.
    """
    provider = FakeProvider()
    events = await collect(runner_for(provider, prompts, conn), session_id, "hello", student)

    assert isinstance(events[-1], Done)
    assert provider.last_call.messages[0].content == "You are Charles."
    assert len([m for m in provider.last_call.messages if m.role == "system"]) == 1


async def test_level_event_comes_in_every_turn_with_a_profile(
    real_prompts, conn, session_id, student
):
    provider = FakeProvider()
    runner = runner_for(provider, real_prompts, conn, tracker=LevelTracker(conn))

    events = await collect(runner, session_id, "I work with marketing in a company here", student)

    leveled = [e for e in events if isinstance(e, Leveled)]
    assert len(leveled) == 1
    assert leveled[0].decision.level == "B1"
    assert leveled[0].decision.changed is False
    # Antes do Done: o Done fecha o turno.
    assert isinstance(events[-1], Done)


async def test_the_level_in_force_only_changes_the_next_turn(
    real_prompts, conn, session_id, student
):
    """Mudar o nível no meio do turno não refaz a resposta já gerada."""
    provider = FakeProvider(replies="ok")
    runner = runner_for(provider, real_prompts, conn, tracker=LevelTracker(conn))
    text = (
        "Although I have worked in logistics for almost ten years, I would rather "
        "understand how the current process works, which nobody has documented, "
        "before suggesting any change to the team."
    )

    changed = None
    for _ in range(AGREEING_TURNS):
        profile = StudentRepo(conn).get(student.id)
        events = await collect(runner, session_id, text, profile)
        for event in events:
            if isinstance(event, Leveled) and event.decision.changed:
                changed = event.decision

    assert changed is not None and changed.level == "B2"
    # O turno que mudou o nível ainda foi gerado com o prompt de B1...
    assert real_prompts.get(level_partial_name("B1")) in provider.system_prompt_of_last_call()
    # ...e o próximo já sai com o de B2.
    await collect(runner, session_id, "and what about you", StudentRepo(conn).get(student.id))
    assert real_prompts.get(level_partial_name("B2")) in provider.system_prompt_of_last_call()


async def test_level_tracking_failure_does_not_kill_the_turn(
    real_prompts, conn, session_id, student
):
    """A resposta do professor já saiu; nível quebrado não pode levá-la embora."""

    class Broken(LevelTracker):
        def observe(self, *args, **kwargs):
            raise RuntimeError("banco sumiu")

    runner = runner_for(FakeProvider(), real_prompts, conn, tracker=Broken(conn))
    events = await collect(runner, session_id, "I work with marketing here", student)

    assert not [e for e in events if isinstance(e, Leveled)]
    assert isinstance(events[-1], Done)


# --- Os modos (E6) --------------------------------------------------------- #


@pytest.fixture
def base():
    return load_base(REPO_ROOT / "data" / "contrastive")


def test_mode_prompt_comes_after_the_level(real_prompts, conn, session_id):
    """Nível antes de modo, por causa do cache de prefixo.

    O nível muda a cada muitos turnos; o modo muda quando o aluno quer. Pôr o
    modo antes faria trocar de modo invalidar também o ajuste de nível que já
    estava no cache.
    """
    runner = runner_for(FakeProvider(), real_prompts, conn)
    messages = runner.build_messages(session_id, "hello there", "B1", "correction")

    assert [m.role for m in messages[:4]] == ["system"] * 4
    assert messages[2].content == real_prompts.get(level_partial_name("B1"))
    assert messages[3].content == real_prompts.get("modes/correction")


def test_each_mode_sends_its_own_prompt(real_prompts, conn, session_id):
    runner = runner_for(FakeProvider(), real_prompts, conn)
    prompts_by_mode = {
        mode: "\n".join(
            m.content
            for m in runner.build_messages(session_id, "hello there", "B1", mode)
            if m.role == "system"
        )
        for mode in MODES
    }

    assert len(set(prompts_by_mode.values())) == len(MODES)
    for mode, text in prompts_by_mode.items():
        assert real_prompts.get(f"modes/{mode}") in text


async def test_the_session_mode_reaches_the_model(real_prompts, conn, student):
    """O modo vem da sessão, não de um parâmetro que a rota precise lembrar."""
    provider = FakeProvider()
    runner = runner_for(provider, real_prompts, conn)
    session_id = SessionRepo(conn).switch_mode(student.id, "correction")

    await collect(runner, session_id, "I have 30 years", student)

    system = provider.system_prompt_of_last_call()
    assert real_prompts.get("modes/correction") in system
    assert real_prompts.get("modes/drill") not in system


async def test_drill_block_carries_the_students_real_error(real_prompts, conn, student, base):
    """O critério de pronto da E6: o exercício é sobre erro de fato cometido."""
    session_id = SessionRepo(conn).current_or_start(student.id)
    turn_id = TurnRepo(conn).append(session_id, "student", "I have 35 years")
    CorrectionRepo(conn).add(
        turn_id,
        span="I have 35 years",
        corrected="I am 35 years old",
        error_type="calque",
        explanation="Age uses to be.",
        rule_id="calque.age-have",
    )

    planner = DrillPlanner(conn, base)
    target = planner.pick(student.id)
    drill_session = SessionRepo(conn).switch_mode(student.id, "drill", target.rule_id)

    provider = FakeProvider()
    runner = runner_for(provider, real_prompts, conn, planner=planner)
    await collect(runner, drill_session, "I am 35 years old?", student)

    system = provider.system_prompt_of_last_call()
    assert "calque.age-have" in system
    assert "I have 35 years" in system, "o drill tem que usar as palavras do aluno"


async def test_drill_without_an_error_log_says_so(real_prompts, conn, student, base):
    planner = DrillPlanner(conn, base)
    assert planner.pick(student.id) is None

    session_id = SessionRepo(conn).switch_mode(student.id, "drill", None)
    provider = FakeProvider()
    runner = runner_for(provider, real_prompts, conn, planner=planner)
    await collect(runner, session_id, "what are we doing today", student)

    assert "no rule to drill" in provider.system_prompt_of_last_call().lower()


async def test_correction_mode_keeps_more_findings(real_prompts, conn, student, base):
    """O teto de correções é do modo, e chega ao analisador."""
    errors = {
        "errors": [
            {
                "span": span,
                "type": "calque",
                "correction": correction,
                "rule_id": rule_id,
                "explanation": "ignorada: a curada vence",
            }
            for span, correction, rule_id in (
                ("I have 35 years", "I am 35 years old", "calque.age-have"),
                ("make a question", "ask a question", "calque.make-question"),
                ("depends of", "depends on", "calque.depend-of"),
            )
        ]
    }
    text = "I have 35 years, can I make a question? It depends of my boss."

    def run_in(mode):
        provider = FakeProvider(
            replies="Sure!",
            replies_by_system={"You find mistakes": json.dumps(errors)},
        )
        analyzer = Analyzer(provider=provider, prompts=real_prompts, base=base)
        runner = runner_for(provider, real_prompts, conn, analyzer=analyzer)
        session_id = SessionRepo(conn).switch_mode(student.id, mode)
        return runner, session_id

    runner, session_id = run_in("conversation")
    events = await collect(runner, session_id, text, student)
    in_conversation = [e for e in events if isinstance(e, Analyzed)][0]

    runner, session_id = run_in("correction")
    events = await collect(runner, session_id, text, student)
    in_correction = [e for e in events if isinstance(e, Analyzed)][0]

    assert len(in_conversation.analysis.findings) == 2
    assert len(in_correction.analysis.findings) == 3


async def test_the_drill_asks_for_an_exercise_at_the_end_of_the_prompt(
    real_prompts, conn, student, base
):
    """O pedido do drill fica no fim, depois da fala do aluno.

    Não é estética de prompt: medido num 3B (plano §5.4), com o pedido só no
    bloco do topo o modelo respondia à fala do aluno e esquecia de treinar.
    Com a linha no fim, ele dá exercício. O bloco de dados continua no topo,
    onde é estável e cacheável.
    """
    session_id = SessionRepo(conn).current_or_start(student.id)
    turn_id = TurnRepo(conn).append(session_id, "student", "I have 35 years")
    CorrectionRepo(conn).add(
        turn_id,
        span="I have 35 years",
        corrected="I am 35 years old",
        error_type="calque",
        explanation="Age uses to be.",
        rule_id="calque.age-have",
    )

    planner = DrillPlanner(conn, base)
    target = planner.pick(student.id)
    drill_session = SessionRepo(conn).switch_mode(student.id, "drill", target.rule_id)

    provider = FakeProvider()
    runner = runner_for(provider, real_prompts, conn, planner=planner)
    await collect(runner, drill_session, "I am 35 years old?", student)

    messages = provider.last_call.messages
    last = messages[-1]
    assert last.role == "user", "o pedido vai junto com a fala do aluno (plano §5.5)"
    assert "one short exercise" in last.content.lower()
    assert "calque.age-have" in last.content
    # E o bloco de dados continua lá em cima, onde o prefixo é cacheado.
    assert "Correct forms:" in messages[4].content


# --- O que o aplicativo sabe fazer --------------------------------------- #
#
# Estes testes usam os prompts **de verdade**: o que está sob teste é a verdade
# que o produto conta sobre si, não um partial inventado no tmp_path. A falha
# que os motivou é de uma sessão real — perguntado se tinha microfone para
# capturar a voz do aluno, o Charles disse que não tinha e sugeriu gravar por
# fora, com o botão de push-to-talk na tela ao lado da frase.


def prefix_text(runner, level="B1", mode="conversation") -> str:
    """O prefixo estável concatenado, com o espaço em branco normalizado.

    Os prompts são markdown com linhas de 79 colunas, então uma frase procurada
    aqui pode estar quebrada no meio. Normalizar faz o teste ser sobre o que o
    professor lê, e não sobre onde o parágrafo foi dobrado.
    """
    joined = "\n".join(m.content for m in runner.stable_prefix(level, mode))
    return " ".join(joined.split())


def test_capability_partial_is_picked_by_the_flag():
    assert app_partial_name(voice_enabled=True) == "partials/app/voice"
    assert app_partial_name(voice_enabled=False) == "partials/app/text"


def test_with_voice_off_charles_knows_there_is_no_microphone(real_prompts, conn):
    """A resposta honesta, sem inventar saída.

    Sem voz não há microfone nenhum, e o que o Charles não pode fazer é
    improvisar um contorno: ele não sabe o que mais existe na máquina do aluno.
    """
    runner = runner_for(FakeProvider(), real_prompts, conn, voice_enabled=False)
    text = prefix_text(runner)

    assert "no microphone" in text
    assert "Hold to talk" not in text, "não existe botão nenhum neste estado"


def test_with_voice_on_charles_knows_about_the_button(real_prompts, conn):
    """Com voz, a verdade é a oposta — e o botão tem nome na tela."""
    runner = runner_for(FakeProvider(), real_prompts, conn, voice_enabled=True)
    text = prefix_text(runner)

    # O mesmo rótulo do template (``web/templates/index.html``): mandar outro
    # nome faria o Charles apontar para um botão que o aluno não vê.
    assert "Hold to talk" in text
    assert "out loud" in text
    assert "no microphone" not in text


def test_neither_state_claims_to_hear_pronunciation(real_prompts, conn):
    """O avaliador de fonemas da Fase 2 não existe: a E9 é um portão aberto.

    Com voz ou sem, o que chega ao professor é texto — a transcrição, nunca o
    som. Deixá-lo agir como se ouvisse seria a mesma classe de erro que a E3 e
    a E4 existem para evitar do lado da gramática.
    """
    for voice_enabled in (False, True):
        runner = runner_for(FakeProvider(), real_prompts, conn, voice_enabled=voice_enabled)
        text = prefix_text(runner)
        assert "cannot hear pronunciation" in text
        assert "how *they* said it" in text


def test_the_persona_does_not_change_with_the_voice_flag(real_prompts, conn):
    """A persona não virou texto por execução.

    É o invariante do cache de prefixo: o Ollama cacheia o começo do prompt, e
    valor que varia dentro da persona jogaria esse cache fora a cada subida com
    configuração diferente. O que varia entra como mensagem própria.
    """
    off = runner_for(FakeProvider(), real_prompts, conn, voice_enabled=False).stable_prefix()
    on = runner_for(FakeProvider(), real_prompts, conn, voice_enabled=True).stable_prefix()

    assert off[0] == on[0], "a persona não pode mudar com a voz"
    assert off[0].content == real_prompts.get("persona")
    # Diferem numa mensagem só, e é a das capacidades.
    assert off[1] != on[1]
    assert [m.content for m in off[2:]] == [m.content for m in on[2:]]


def test_the_capabilities_are_the_second_system_message(real_prompts, conn, session_id):
    """Logo depois da persona, antes do nível e do modo.

    O critério é o mesmo de sempre: o cache é de prefixo, e isto é o mais
    estável do que sobra — a voz é decidida na subida do servidor, enquanto o
    nível muda a cada muitos turnos e o modo o aluno troca na tela.
    """
    runner = runner_for(FakeProvider(), real_prompts, conn, voice_enabled=True)
    messages = runner.build_messages(session_id, "hello there", "B1", "correction")

    assert messages[1].role == "system"
    assert messages[1].content == real_prompts.get("partials/app/voice")


@pytest.mark.parametrize(
    ("voice_enabled", "needle"),
    [(True, "Hold to talk"), (False, "no microphone")],
)
async def test_the_capabilities_reach_the_model(
    real_prompts, conn, session_id, voice_enabled, needle
):
    """E chegam ao modelo, não só ao ``build_messages``."""
    provider = FakeProvider()
    runner = runner_for(provider, real_prompts, conn, voice_enabled=voice_enabled)

    await collect(runner, session_id, "do you have a microphone?")

    assert needle in provider.system_prompt_of_last_call()


@pytest.mark.parametrize("voice_enabled", [False, True])
async def test_a_missing_capability_partial_does_not_break_the_turn(
    prompts, conn, session_id, voice_enabled
):
    """Mesma regra do nível e do modo: arquivo que falta não derruba a aula.

    O ``prompts`` do tmp_path tem só a persona.
    """
    provider = FakeProvider(replies="Sure.")
    runner = runner_for(provider, prompts, conn, voice_enabled=voice_enabled)

    assert runner.app_prompt() == ""
    assert [m.content for m in runner.stable_prefix()] == [prompts.get("persona")]

    events = await collect(runner, session_id, "hello there")
    assert isinstance(events[-1], Done)


# --- A pronúncia embaixo de cada fala ------------------------------------- #
#
# O G2P é determinístico e de graça (o espeak, não o modelo), então estes
# testes usam o de verdade: o que está sob teste é o arranjo do turno — quantos
# eventos, de quem, e em que ponto —, não a tabela do espeak, que
# ``test_phonetics.py`` cobre.


def test_sem_g2p_o_turno_nao_emite_fonetica(prompts, conn, session_id):
    """``None`` é estado de primeira classe: a aula roda inteira sem IPA."""
    provider = FakeProvider(replies="Nice.")
    runner = runner_for(provider, prompts, conn)

    events = asyncio.run(collect(runner, session_id, "I have 30 years"))

    assert not [e for e in events if isinstance(e, Phonemized)]


def test_o_turno_transcreve_os_dois_lados(prompts, conn, session_id):
    provider = FakeProvider(replies="I am 30 years old.")
    runner = runner_for(provider, prompts, conn, phonetics=Phonetics())

    events = asyncio.run(collect(runner, session_id, "I have 30 years"))
    spelled = [e for e in events if isinstance(e, Phonemized)]

    assert [e.role for e in spelled] == ["student", "teacher"]
    assert all(e.ipa for e in spelled)
    # IPA, e não a ortografia de volta.
    assert "30 years" not in spelled[0].ipa


def test_a_fonetica_do_aluno_vem_antes_do_primeiro_pedaco(prompts, conn, session_id):
    """O G2P não espera modelo nenhum, então a linha dele não espera também.

    Se ela saísse depois, a bolha do aluno ficaria segundos sem transcrição
    enquanto o professor escreve — e é a fala do aluno que ele quer conferir.
    """
    provider = FakeProvider(replies="I am 30 years old.", chunk_size=4)
    runner = runner_for(provider, prompts, conn, phonetics=Phonetics())

    events = asyncio.run(collect(runner, session_id, "I have 30 years"))
    kinds = [type(e) for e in events]

    assert kinds.index(Phonemized) < kinds.index(Chunk)


def test_a_fonetica_do_professor_cobre_a_resposta_inteira(prompts, conn, session_id):
    """Uma linha por bolha, não uma por sentença.

    Transcrever por sentença produziria uma linha remendada em que a última
    palavra de uma e a primeira da seguinte nunca se encontram — e aqui não há
    o que adiantar: o G2P já terminou antes de o quadro chegar à página.
    """
    provider = FakeProvider(replies="Good. How old are you?", chunk_size=3)
    runner = runner_for(provider, prompts, conn, phonetics=Phonetics())

    events = asyncio.run(collect(runner, session_id, "I have 30 years"))
    teacher = [e for e in events if isinstance(e, Phonemized) and e.role == "teacher"]

    assert len(teacher) == 1
    assert teacher[0].text == "Good. How old are you?"


def test_o_turno_nao_cai_com_o_g2p_quebrado(prompts, conn, session_id):
    """Pronúncia ausente é funcionalidade degradada; turno perdido seria aula parada."""

    class Broken(Phonetics):
        def _load(self):
            raise PhoneticsUnavailable("espeak não carregou")

    provider = FakeProvider(replies="Nice.")
    runner = runner_for(provider, prompts, conn, phonetics=Broken())

    events = asyncio.run(collect(runner, session_id, "I have 30 years"))

    assert not [e for e in events if isinstance(e, Phonemized)]
    assert isinstance(events[-1], Done)


def test_com_ipa_ligado_o_professor_sabe_que_a_linha_existe(real_prompts, conn):
    """Senão ele escreve os símbolos na prosa — e já foi medido escrevendo errado.

    O CLAUDE.md registra o caso: perguntado pelo IPA, ofereceu *pat* contra
    *bat* como contraste de vogal, que é a mesma nas duas.
    """
    runner = runner_for(FakeProvider(), real_prompts, conn, phonetics=Phonetics())
    text = prefix_text(runner)

    assert "Do not type IPA yourself" in text
    # E segue sem alegar que ouve: a E9 é um portão aberto.
    assert "not a reading of the student" in text


def test_sem_ipa_o_prefixo_nao_promete_linha_nenhuma(real_prompts, conn):
    runner = runner_for(FakeProvider(), real_prompts, conn)
    assert "Do not type IPA yourself" not in prefix_text(runner)


def test_a_fonetica_nao_acrescenta_mensagem_de_sistema(real_prompts, conn):
    """Vai junto com as capacidades do aplicativo, no mesmo bloco.

    As duas coisas são verdade sobre o produto, decididas na subida e imóveis
    enquanto o servidor está de pé. Mensagem a mais seria prefixo a mais sem
    estabilidade a mais.
    """
    com = runner_for(FakeProvider(), real_prompts, conn, phonetics=Phonetics())
    sem = runner_for(FakeProvider(), real_prompts, conn)

    assert len(com.stable_prefix()) == len(sem.stable_prefix())
