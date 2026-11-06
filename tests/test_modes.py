"""Testes dos três modos e do drill (E6). Nenhum usa modelo de verdade.

O critério de pronto da E6 tem duas metades: os três modos funcionam, e o drill
usa erro **de fato cometido**. A segunda metade é a que estes testes vigiam de
perto — é fácil escrever um drill que parece certo e treina erro inventado.
"""

from __future__ import annotations

import pytest

from teacher_charles.config import REPO_ROOT
from teacher_charles.prompts import PromptLibrary
from teacher_charles.store import db
from teacher_charles.store.models import MODES
from teacher_charles.store.repo import CorrectionRepo, SessionRepo, StudentRepo, TurnRepo
from teacher_charles.teaching.contrastive import load_base
from teacher_charles.teaching.modes import (
    DEFAULT_MODE,
    MAX_FINDINGS_BY_MODE,
    DrillPlanner,
    DrillTarget,
    drill_block,
    drill_reminder,
    max_findings,
    partial_name,
)

AGE_RULE = "calque.age-have"
QUESTION_RULE = "calque.make-question"


@pytest.fixture(scope="module")
def base():
    return load_base(REPO_ROOT / "data" / "contrastive")


@pytest.fixture
def conn(tmp_path):
    connection = db.init_db(tmp_path / "modes.db")
    yield connection
    connection.close()


@pytest.fixture
def student(conn):
    return StudentRepo(conn).get_or_create("ana")


def record_error(conn, student_id: int, rule_id: str, span: str, corrected: str) -> int:
    """Um erro real no histórico do aluno: turno dele, correção com regra."""
    session_id = SessionRepo(conn).current_or_start(student_id)
    turn_id = TurnRepo(conn).append(session_id, "student", span)
    CorrectionRepo(conn).add(
        turn_id,
        span=span,
        corrected=corrected,
        error_type="calque",
        explanation="curada",
        rule_id=rule_id,
    )
    return turn_id


# --- O modo, como política ------------------------------------------------- #


def test_every_mode_has_a_prompt():
    library = PromptLibrary(REPO_ROOT / "prompts")
    for mode in MODES:
        assert library.exists(partial_name(mode)), f"falta o prompt do modo {mode}"


def test_mode_prompts_are_distinct_and_english_only():
    library = PromptLibrary(REPO_ROOT / "prompts")
    texts = {mode: library.get(partial_name(mode)) for mode in MODES}

    for mode, text in texts.items():
        assert text.strip(), f"o prompt do modo {mode} está vazio"
        assert mode in text.lower(), f"o prompt de {mode} não diz de que modo é"
        # Decisão 1.1: nada que o modelo lê sobre o aluno vai em português.
        assert not any(word in text.lower() for word in (" aluno", "português", " frase "))

    assert len(set(texts.values())) == len(MODES), "dois modos com o mesmo prompt"


def test_correction_mode_reports_more_errors_than_conversation():
    """Não é só prosa diferente: o teto de correções por turno muda.

    No modo de correção o aluno pediu os erros. Devolver dois de cinco seria
    esconder dele o que ele veio buscar.
    """
    assert max_findings("correction") > max_findings("conversation")
    assert max_findings("drill") == MAX_FINDINGS_BY_MODE["drill"]
    assert max_findings("modo-que-não-existe") == MAX_FINDINGS_BY_MODE[DEFAULT_MODE]


# --- O bloco do drill ------------------------------------------------------ #


def test_drill_block_carries_the_students_own_words(base):
    target = DrillTarget(
        item=base.get(AGE_RULE),
        times=3,
        student_examples=("I have 35 years", "my wife has 32 years"),
    )
    block = drill_block(target)

    assert AGE_RULE in block
    assert "I have 35 years" in block
    assert "my wife has 32 years" in block
    assert "3 times" in block
    # E as formas certas vêm da base curada, não do modelo.
    assert any(right in block for right in base.get(AGE_RULE).right)


def test_drill_block_without_a_rule_says_so(base):
    """Sem erro registrado o produto diz a verdade.

    Inventar exercício genérico aqui seria exatamente o que o critério da E6
    proíbe — e o aluno nem saberia que o "seu erro" não era dele.
    """
    block = drill_block(None)
    assert "no rule to drill" in block.lower()
    assert "talk" in block.lower()


# --- A escolha da regra ---------------------------------------------------- #


def test_planner_picks_the_most_frequent_real_error(conn, student, base):
    record_error(conn, student.id, QUESTION_RULE, "make a question", "ask a question")
    for _ in range(2):
        record_error(conn, student.id, AGE_RULE, "I have 35 years", "I am 35")

    target = DrillPlanner(conn, base).pick(student.id)

    assert target is not None
    assert target.rule_id == AGE_RULE
    assert target.times == 2
    assert "I have 35 years" in target.student_examples


def test_planner_ignores_corrections_without_a_curated_rule(conn, student, base):
    """Correção sem regra veio do modelo sozinho (plano §5.2).

    Treinar em cima dela seria ensinar o erro com mais repetição.
    """
    session_id = SessionRepo(conn).current_or_start(student.id)
    turn_id = TurnRepo(conn).append(session_id, "student", "There have been many changes")
    CorrectionRepo(conn).add(
        turn_id,
        span="There have been",
        corrected="There has been",
        error_type="grammar",
        explanation="inventada pelo modelo",
        rule_id=None,
    )

    assert DrillPlanner(conn, base).pick(student.id) is None


def test_planner_ignores_rules_that_left_the_base(conn, student, base):
    """Regra renomeada na curadoria continua no histórico e não pode virar drill."""
    record_error(conn, student.id, "calque.regra-que-sumiu", "whatever", "whatever else")
    assert DrillPlanner(conn, base).pick(student.id) is None


def test_planner_has_nothing_to_drill_for_a_new_student(conn, student, base):
    assert DrillPlanner(conn, base).pick(student.id) is None


def test_the_error_log_is_per_student(conn, base):
    """O erro de um aluno não vira exercício de outro."""
    students = StudentRepo(conn)
    ana = students.get_or_create("ana")
    joao = students.get_or_create("joao")
    record_error(conn, ana.id, AGE_RULE, "I have 35 years", "I am 35")

    planner = DrillPlanner(conn, base)
    assert planner.pick(ana.id) is not None
    assert planner.pick(joao.id) is None


def test_rotation_avoids_the_last_drilled_rules(conn, student, base):
    for _ in range(3):
        record_error(conn, student.id, AGE_RULE, "I have 35 years", "I am 35")
    record_error(conn, student.id, QUESTION_RULE, "make a question", "ask a question")

    planner = DrillPlanner(conn, base)
    sessions = SessionRepo(conn)

    first = planner.pick(student.id)
    assert first is not None and first.rule_id == AGE_RULE
    sessions.switch_mode(student.id, "drill", first.rule_id)

    second = planner.pick(student.id)
    assert second is not None
    assert second.rule_id == QUESTION_RULE, "a mesma regra duas vezes seguidas não é rodízio"


def test_rotation_repeats_rather_than_drilling_nothing(conn, student, base):
    """Rodízio é preferência, não impedimento.

    Com uma regra só no error log, repeti-la é melhor que não treinar nada.
    """
    record_error(conn, student.id, AGE_RULE, "I have 35 years", "I am 35")
    planner = DrillPlanner(conn, base)
    sessions = SessionRepo(conn)

    for _ in range(3):
        target = planner.pick(student.id)
        assert target is not None and target.rule_id == AGE_RULE
        sessions.switch_mode(student.id, "drill", target.rule_id)


def test_target_for_resolves_the_rule_of_an_ongoing_session(conn, student, base):
    """Uma regra por sessão: o turno resolve a regra já escolhida, não escolhe outra."""
    record_error(conn, student.id, AGE_RULE, "I have 35 years", "I am 35")
    planner = DrillPlanner(conn, base)

    target = planner.target_for(student.id, AGE_RULE)
    assert target is not None
    assert target.student_examples == ("I have 35 years",)
    assert planner.target_for(student.id, "calque.inexistente") is None


# --- O pedido no fim do prompt --------------------------------------------- #


def test_drill_reminder_names_the_rule(base):
    target = DrillTarget(item=base.get(AGE_RULE), times=2, student_examples=("I have 35 years",))
    reminder = drill_reminder(target)

    assert AGE_RULE in reminder
    assert "one short exercise" in reminder.lower()


def test_drill_reminder_is_empty_without_a_rule():
    """Sem regra não há o que pedir: o bloco do topo já explica a situação."""
    assert drill_reminder(None) == ""


def test_drill_prompt_forbids_portuguese_exercises():
    """Decisão 1.1 tem que estar explícita justamente aqui.

    Medido (plano §5.4): a primeira versão do `drill.md` pedia ao professor
    "ask them to translate a situation", e um 7B obedeceu escrevendo *"can you
    translate this situation: 'Ela tem 28 anos.'"* — português na tela do
    aluno. Todo verbo no prompt é uma permissão, e esta foi retirada.
    """
    text = PromptLibrary(REPO_ROOT / "prompts").get(partial_name("drill")).lower()

    assert "portuguese" in text, "o drill é o modo onde traduzir é tentador; diga que não"
    assert "never" in text
