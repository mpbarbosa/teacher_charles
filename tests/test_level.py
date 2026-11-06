"""Testes do nível CEFR (E5). Nenhum usa modelo de verdade.

A rubrica é determinística, então ela é testável de ponta a ponta sem Ollama —
que é metade do motivo de ela ser determinística.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from teacher_charles.config import REPO_ROOT
from teacher_charles.prompts import PromptLibrary
from teacher_charles.store import db
from teacher_charles.store.models import CEFR_LEVELS
from teacher_charles.store.repo import LevelRepo, SessionRepo, StudentRepo, TurnRepo
from teacher_charles.teaching.level import (
    AGREEING_TURNS,
    MIN_WORDS,
    LevelTracker,
    Signals,
    decide,
    estimate,
    index_of,
    partial_name,
    step_toward,
)

SAMPLES_PATH = Path(__file__).parent / "fixtures" / "level_samples.yaml"


@pytest.fixture(scope="module")
def samples() -> dict[str, list[str]]:
    return yaml.safe_load(SAMPLES_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def conn(tmp_path):
    connection = db.init_db(tmp_path / "level.db")
    yield connection
    connection.close()


@pytest.fixture
def student(conn):
    return StudentRepo(conn).get_or_create("ana", cefr_level="B1")


@pytest.fixture
def session_id(conn, student):
    return SessionRepo(conn).current_or_start(student.id)


# --- Os sinais ------------------------------------------------------------- #


def test_counts_words_and_sentences():
    signals = Signals.of("I work here. I like it!")
    assert signals.words == 6
    assert signals.sentences == 2
    assert signals.mean_sentence_words == 3.0


def test_text_without_final_punctuation_is_one_sentence():
    assert Signals.of("just a line with no period").sentences == 1


def test_empty_text_has_no_signals():
    signals = Signals.of("")
    assert signals.words == 0
    assert signals.sentences == 0
    assert signals.mean_sentence_words == 0.0
    assert signals.long_word_ratio == 0.0


def test_long_words_are_counted_by_letters():
    signals = Signals.of("the manager approved it")
    # manager (7) e approved (8); "the" e "it" não.
    assert signals.long_words == 2


def test_connectives_are_the_advanced_ones():
    """ "and" e "but" não contam: um A2 usa os dois."""
    assert Signals.of("I was tired and I went home but I could not sleep").connectives == 0
    assert Signals.of("Although I was tired, I went out. However, I left early.").connectives == 2


def test_perfect_aspect_counts_even_contracted():
    """O aluno escreve "I've been", não "I have been".

    Ignorar a contração foi o maior buraco da primeira versão da rubrica: as
    amostras B1 e B2 vinham com zero estrutura.
    """
    assert Signals.of("I have worked there for years").structures >= 1
    assert Signals.of("I've worked there for years").structures >= 1


def test_conditional_and_relative_count_as_structures():
    assert Signals.of("I would have accepted if they had been clearer").structures >= 1
    assert Signals.of("the process, which nobody owns, is the problem").structures >= 1


def test_short_turn_is_not_evidence():
    """ "ok, thanks" tem a mesma métrica para um A2 e para um C1."""
    assert not Signals.of("ok, thanks!").counts_as_evidence
    assert not estimate("ok, thanks!").counted

    long_enough = " ".join(["word"] * MIN_WORDS)
    assert Signals.of(long_enough).counts_as_evidence


# --- A rubrica ------------------------------------------------------------- #


def test_errors_push_the_score_down():
    text = (
        "I have been working with suppliers for six years, and I think I need "
        "a change because I am not learning anything new."
    )
    clean = estimate(text, errors=0)
    with_errors = estimate(text, errors=3)

    assert with_errors.score < clean.score
    assert index_of(with_errors.level) <= index_of(clean.level)


def test_score_stays_in_range():
    for text in ("", "ok", "Although " * 50, "a " * 400):
        assert 0.0 <= estimate(text).score <= 1.0


def test_level_never_estimated_above_c1():
    """C2 é autodeclarado: a rubrica não tem sinal que o separe de C1.

    Medido: com um corte de C1->C2, três das oito amostras B2 iam para C2,
    errando por dois degraus. Ver docs/plano-fase-1.md §5.3.
    """
    saturated = (
        "Although the report, which had been written to avoid blame rather than "
        "to explain what happened, would have been more useful otherwise, "
        "nevertheless the conclusion is impossible to disguise, and no matter "
        "how many charts you add, whoever reads it ends up understanding that."
    )
    assert estimate(saturated).level == "C1"


# --- O portão sobre as amostras rotuladas ---------------------------------- #


def test_sample_suite_is_classified_well_enough(samples):
    """A rubrica tem que acertar o nível exato na maioria das amostras.

    O número medido é 25/32 (78%). O portão está em 70% para a suíte poder
    crescer sem ficar frágil, mas uma queda real quebra o teste.
    """
    total = exact = 0
    for level, texts in samples.items():
        for text in texts:
            total += 1
            exact += estimate(text).level == level

    assert total >= 32, "a suíte encolheu"
    assert exact / total >= 0.70, f"só {exact}/{total} acertos exatos"


def test_sample_suite_never_misses_by_two_levels(samples):
    """Errar um degrau é o esperado; dois é inaceitável.

    É o que a histerese consegue absorver: quatro turnos seguidos movem um
    degrau, então um erro de um degrau se corrige sozinho no uso. Um erro de
    dois degraus mandaria um C1 ouvir inglês de A2.
    """
    for level, texts in samples.items():
        for text in texts:
            guess = estimate(text)
            distance = abs(index_of(guess.level) - index_of(level))
            assert distance <= 1, f"{level} lido como {guess.level}: {text[:60]!r}"


def test_the_two_ends_of_the_scale_are_clean(samples):
    """O que a rubrica faz bem é separar as pontas, e isso não pode regredir."""
    for text in samples["A2"]:
        assert index_of(estimate(text).level) <= index_of("B1")
    for text in samples["C1"]:
        assert index_of(estimate(text).level) >= index_of("B2")


# --- A política de ajuste -------------------------------------------------- #


def test_one_turn_never_moves_the_level():
    assert decide("B1", ["C1"]) == ("B1", "")
    assert decide("B1", ["C1", "C1", "C1"]) == ("B1", "")


def test_agreeing_turns_move_one_step():
    level, reason = decide("B1", ["C1"] * AGREEING_TURNS)
    assert level == "B2", "o nível tem que andar um degrau, não pular para C1"
    assert "C1" in reason


def test_a_single_disagreeing_turn_resets():
    recent = ["B2", "B2", "B1", "B2", "B2"]
    assert decide("B1", recent) == ("B1", "")


def test_the_level_also_goes_down():
    level, _ = decide("B2", ["A2"] * AGREEING_TURNS)
    assert level == "B1"


def test_a_self_declared_c2_is_left_alone():
    """A rubrica não estima C2, então não tem autoridade para rebaixar um C2."""
    assert decide("C2", ["C1"] * AGREEING_TURNS) == ("C2", "")


def test_step_toward_never_jumps():
    assert step_toward("A2", "C1") == "B1"
    assert step_toward("C1", "A2") == "B2"
    assert step_toward("B1", "B1") == "B1"


def test_invalid_level_is_rejected():
    with pytest.raises(ValueError):
        index_of("Z9")


# --- O tracker, com banco -------------------------------------------------- #


def _say(conn, session_id, text: str) -> int:
    return TurnRepo(conn).append(session_id, "student", text)


def test_observe_records_the_signals(conn, student, session_id):
    tracker = LevelTracker(conn)
    text = "I work in a store in my city and I like my job."
    turn_id = _say(conn, session_id, text)
    decision = tracker.observe(student.id, turn_id, text)

    row = LevelRepo(conn).signals_for_turn(turn_id)
    assert row is not None
    assert row.words > MIN_WORDS
    assert row.estimated_level == decision.estimate.level
    assert row.counted is True


def test_short_turns_are_recorded_but_do_not_count(conn, student, session_id):
    """As linhas descartadas ficam no banco de propósito.

    Sem elas não dá para medir se o corte de MIN_WORDS está no lugar certo, e
    corte que ninguém mede é chute.
    """
    tracker = LevelTracker(conn)
    turn_id = _say(conn, session_id, "ok, thanks!")
    decision = tracker.observe(student.id, turn_id, "ok, thanks!")

    row = LevelRepo(conn).signals_for_turn(turn_id)
    assert row is not None and row.counted is False
    assert decision.changed is False
    assert LevelRepo(conn).recent_levels(student.id) == []


def test_level_moves_after_enough_agreeing_turns(conn, student, session_id):
    tracker = LevelTracker(conn)
    text = (
        "Although I have worked in logistics for almost ten years, I would rather "
        "understand how the current process works, which nobody has documented, "
        "before suggesting any change to the team."
    )

    decisions = []
    for _ in range(AGREEING_TURNS):
        decisions.append(tracker.observe(student.id, _say(conn, session_id, text), text))

    assert [d.changed for d in decisions] == [False] * (AGREEING_TURNS - 1) + [True]
    moved = decisions[-1]
    assert moved.previous == "B1"
    assert moved.level == "B2"
    assert moved.direction == 1

    # O perfil mudou, e o histórico diz por quê.
    assert StudentRepo(conn).get(student.id).cefr_level == "B2"
    assert StudentRepo(conn).get(student.id).level_source == "auto"
    changes = LevelRepo(conn).changes(student.id)
    assert [(c.from_level, c.to_level, c.source) for c in changes] == [("B1", "B2", "auto")]
    assert changes[0].reason


def test_auto_adjustment_keeps_the_self_declared_level(conn, student, session_id):
    """O autodeclarado é a referência externa; o ajuste não pode apagá-la."""
    tracker = LevelTracker(conn)
    text = (
        "Although I have worked in logistics for almost ten years, I would rather "
        "understand how the current process works, which nobody has documented, "
        "before suggesting any change to the team."
    )
    for _ in range(AGREEING_TURNS):
        tracker.observe(student.id, _say(conn, session_id, text), text)

    profile = StudentRepo(conn).get(student.id)
    assert profile.cefr_level == "B2"
    assert profile.self_declared_level == "B1"


def test_evidence_crosses_sessions(conn, student):
    """Começar conversa nova não volta a evidência de nível à estaca zero."""
    tracker = LevelTracker(conn)
    text = (
        "Although I have worked in logistics for almost ten years, I would rather "
        "understand how the current process works, which nobody has documented, "
        "before suggesting any change to the team."
    )
    sessions = SessionRepo(conn)

    for _ in range(AGREEING_TURNS):
        session_id = sessions.start(student.id)
        tracker.observe(student.id, _say(conn, session_id, text), text)

    assert StudentRepo(conn).get(student.id).cefr_level == "B2"


def test_only_curated_errors_penalize(conn, student, session_id):
    """A penalidade usa a contagem em que o projeto confia (plano §5.2)."""
    from teacher_charles.teaching.analyzer import Analysis, Finding

    text = "I have 35 years and I work with marketing in a company here."
    curated = Analysis(
        findings=(
            Finding(
                span="I have 35 years",
                error_type="calque",
                correction="I am 35 years old",
                explanation="Age uses to be.",
                rule_id="calque.age-have",
            ),
        )
    )
    loose = Analysis(
        findings=(
            Finding(
                span="I have 35 years",
                error_type="calque",
                correction="I am 35 years old",
                explanation="Age uses to be.",
            ),
        )
    )

    tracker = LevelTracker(conn)
    with_rule = tracker.observe(student.id, _say(conn, session_id, text), text, curated)
    without_rule = tracker.observe(student.id, _say(conn, session_id, text), text, loose)

    assert with_rule.estimate.signals.errors == 1
    assert without_rule.estimate.signals.errors == 0


# --- Os prompts por nível, do repo ---------------------------------------- #


def test_every_level_has_a_partial():
    library = PromptLibrary(REPO_ROOT / "prompts")
    for level in CEFR_LEVELS:
        assert library.exists(partial_name(level)), f"falta o partial de {level}"


def test_partials_are_distinct_and_english_only():
    library = PromptLibrary(REPO_ROOT / "prompts")
    texts = {level: library.get(partial_name(level)) for level in CEFR_LEVELS}

    for level, text in texts.items():
        assert text.strip(), f"o partial de {level} está vazio"
        assert level in text, f"o partial de {level} não diz de que nível é"
        # Decisão 1.1: nada que o modelo lê sobre o aluno vai em português.
        assert not any(word in text.lower() for word in (" aluno", " nível", "português", " frase"))

    assert len(set(texts.values())) == len(CEFR_LEVELS), "dois níveis com o mesmo prompt"


def test_the_window_closes_after_a_move(conn, student, session_id):
    """Os turnos que moveram o nível não podem movê-lo de novo no turno seguinte.

    Sem isto, quatro turnos bons levariam o aluno de A2 a C1 em seis: o quinto
    turno encontraria os mesmos quatro na janela e andaria outro degrau. Foi um
    bug de verdade — o script de medição simulava a janela zerada e o código
    não zerava.
    """
    tracker = LevelTracker(conn)
    text = (
        "Although I have worked in logistics for almost ten years, I would rather "
        "understand how the current process works, which nobody has documented, "
        "before suggesting any change to the team."
    )

    moves = []
    for _ in range(AGREEING_TURNS * 2):
        decision = tracker.observe(student.id, _say(conn, session_id, text), text)
        if decision.changed:
            moves.append((decision.previous, decision.level))

    assert moves == [("B1", "B2"), ("B2", "C1")], "um degrau a cada quatro turnos, não mais"


def test_declaring_the_level_also_closes_the_window(conn, student, session_id):
    """Declarar o nível zera a evidência acumulada.

    Senão o aluno escolhe B1, e o turno seguinte o devolve para B2 com base em
    turnos anteriores à escolha dele — que é o oposto de ter a última palavra.
    """
    tracker = LevelTracker(conn)
    text = (
        "Although I have worked in logistics for almost ten years, I would rather "
        "understand how the current process works, which nobody has documented, "
        "before suggesting any change to the team."
    )
    students = StudentRepo(conn)
    levels = LevelRepo(conn)

    for _ in range(AGREEING_TURNS - 1):
        tracker.observe(student.id, _say(conn, session_id, text), text)

    # O aluno discorda e declara A2 antes de a janela fechar.
    levels.log_change(
        student.id,
        "B1",
        "A2",
        source="self",
        reason="the student set the level",
        at_signal_id=levels.last_signal_id(student.id),
    )
    students.set_level(student.id, "A2", source="self")

    decision = tracker.observe(student.id, _say(conn, session_id, text), text)
    assert decision.changed is False
    assert students.get(student.id).cefr_level == "A2"
