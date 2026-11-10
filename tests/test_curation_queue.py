"""A fila de curadoria: o que o verificador barrou, guardado em vez de contado.

Até aqui `Analysis.unverified` dizia *quantas* correções foram barradas por não
citarem regra de `data/contrastive/`, e o span ia para `log.debug`. A medida de
quanto a base não cobre existia; a lista não. Estes testes cobrem as quatro
camadas que a lista atravessa, e a garantia que separa as duas pontas:

- a chave de agregação (`contrastive.span_key`) é a normalização do matcher;
- `UnverifiedFindingRepo` soma o mesmo buraco vindo de turnos diferentes;
- o `TurnRunner` grava o barrado e **não** grava correção nenhuma por ele;
- `Analysis.as_dict` não expõe o barrado — material de curadoria não é
  material de aula, e é a mesma gramática improvisada que a E4 acabou de
  recusar;
- e o script de leitura, que é por onde um humano vê a fila.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from teacher_charles.config import REPO_ROOT, Settings
from teacher_charles.llm import FakeProvider
from teacher_charles.prompts import PromptLibrary
from teacher_charles.store import db
from teacher_charles.store.repo import (
    CorrectionRepo,
    SessionRepo,
    StudentRepo,
    UnverifiedFindingRepo,
)
from teacher_charles.teaching import Analyzer, TurnRunner
from teacher_charles.teaching.contrastive import load_base, span_key


def make_settings(**overrides) -> Settings:
    """Sem o ``.env`` do repositório, mesma razão do conftest."""
    return Settings(_env_file=None, **overrides)


BASE = load_base(REPO_ROOT / "data" / "contrastive")
PROMPTS = PromptLibrary(REPO_ROOT / "prompts")

# O modelo acha um erro que a base não cobre. O decalque da idade no começo da
# frase é o que faz a análise rodar: sem candidato nenhum ela nem chamaria o
# modelo, que é o ponto cego documentado na migração 005.
UNCOVERED = json.dumps(
    {
        "errors": [
            {
                "span": "There have been",
                "type": "grammar",
                "correction": "There has been",
                "explanation": "gramática que o modelo improvisou",
            }
        ]
    }
)
STUDENT_TEXT = "I have 30 years. There have been many changes"


@pytest.fixture
def conn(tmp_path):
    connection = db.init_db(tmp_path / "curation.db")
    yield connection
    connection.close()


@pytest.fixture
def prompts(tmp_path):
    root = tmp_path / "prompts"
    root.mkdir()
    (root / "persona.md").write_text("You are Charles.", encoding="utf-8")
    return PromptLibrary(root)


@pytest.fixture
def turn_id(conn):
    student = StudentRepo(conn).get_or_create("ana")
    session_id = SessionRepo(conn).current_or_start(student.id)
    from teacher_charles.store.repo import TurnRepo

    return TurnRepo(conn).append(session_id, "student", STUDENT_TEXT)


def analyzer_for(reply: str, **kwargs) -> Analyzer:
    return Analyzer(provider=FakeProvider(replies=reply), prompts=PROMPTS, base=BASE, **kwargs)


# --- A chave de agregação -------------------------------------------------- #


def test_span_key_folds_case_and_typography():
    """A mesma normalização do matcher, que é o ponto de ela morar lá.

    O apóstrofo curvo é o caso que importa: o aluno que digita no celular e o
    que digita no teclado produzem o mesmo erro, e dois grupos na fila fariam
    o buraco parecer metade do que é.
    """
    assert span_key("I Don’t Have") == span_key("i don't have")


def test_span_key_collapses_whitespace():
    """Aqui o comprimento pode mudar: ninguém aponta offset para esta string.

    É o que separa ``span_key`` de ``normalize`` — aquela preserva o
    comprimento porque os offsets da correção apontam para o texto original.
    """
    assert span_key("  I  have   30 years\n") == "i have 30 years"


# --- O repositório --------------------------------------------------------- #


def test_the_same_hole_from_two_turns_is_one_line_in_the_queue(conn, turn_id):
    repo = UnverifiedFindingRepo(conn)
    for span in ("There have been", "there  HAVE been"):
        repo.add(
            turn_id,
            span=span,
            span_key=span_key(span),
            corrected="There has been",
            error_type="grammar",
            explanation="improvisada",
        )

    (gap,) = repo.backlog()
    assert gap.times == 2
    assert repo.count() == 2
    # Dez turnos com o mesmo erro são um item de YAML a escrever, não dez.
    assert repo.distinct_spans() == 1
    # O span mais recente, com a grafia do aluno: é o que vira `wrong:`.
    assert gap.last_span == "there  HAVE been"


def test_the_queue_is_ordered_by_frequency(conn, turn_id):
    repo = UnverifiedFindingRepo(conn)
    for span, times in (("rare one", 1), ("common one", 3)):
        for _ in range(times):
            repo.add(
                turn_id,
                span=span,
                span_key=span_key(span),
                corrected="fixed",
                error_type="grammar",
                explanation="improvisada",
            )

    assert [gap.last_span for gap in repo.backlog()] == ["common one", "rare one"]


def test_the_queue_keeps_the_rules_that_matched_alongside(conn, turn_id):
    """Span ao lado de regra que casou é padrão a alargar; sozinho, item novo.

    É a distinção que decide qual dos dois trabalhos de curadoria o buraco
    pede, e sem a coluna ela se perderia junto com o turno.
    """
    repo = UnverifiedFindingRepo(conn)
    repo.add(
        turn_id,
        span="There have been",
        span_key=span_key("There have been"),
        corrected="There has been",
        error_type="grammar",
        explanation="improvisada",
        candidates="calque.age-have",
    )

    (gap,) = repo.backlog()
    assert gap.last_candidates == "calque.age-have"


def test_the_queue_is_not_split_by_student(conn):
    """A base contrastiva é conteúdo de produto, não progresso pessoal.

    Dois perfis da máquina que produzem o mesmo span produziram o *mesmo*
    buraco. Agregar por aluno faria a fila ordenar errado — é a única
    agregação do repositório que não é do aluno, e é de propósito.
    """
    from teacher_charles.store.repo import TurnRepo

    students, turns = StudentRepo(conn), TurnRepo(conn)
    sessions = SessionRepo(conn)
    repo = UnverifiedFindingRepo(conn)
    for name in ("ana", "bruno"):
        student = students.get_or_create(name)
        session_id = sessions.current_or_start(student.id)
        turn = turns.append(session_id, "student", STUDENT_TEXT)
        repo.add(
            turn,
            span="There have been",
            span_key=span_key("There have been"),
            corrected="There has been",
            error_type="grammar",
            explanation="improvisada",
        )

    (gap,) = repo.backlog()
    assert gap.times == 2


# --- O analisador ---------------------------------------------------------- #


async def test_the_rejected_finding_is_kept_and_not_just_counted():
    analysis = await analyzer_for(UNCOVERED).analyze(STUDENT_TEXT)

    assert analysis.findings == (), "nada sem regra curada chega ao aluno"
    (rejected,) = analysis.rejected
    assert rejected.span == "There have been"
    assert rejected.correction == "There has been"
    assert rejected.rule_id is None
    # O contador continua valendo o que sempre valeu, agora derivado da lista.
    assert analysis.unverified == len(analysis.rejected) == 1


async def test_the_rejected_finding_never_reaches_the_student_payload():
    """A garantia que separa as duas pontas desta mudança.

    ``as_dict`` alimenta o SSE. O que está em ``rejected`` é exatamente a
    gramática que o verificador acabou de barrar, então vazar aqui desfaria a
    política da E4 por outro caminho — e em silêncio, porque o aluno veria
    uma explicação plausível.
    """
    analysis = await analyzer_for(UNCOVERED).analyze(STUDENT_TEXT)
    payload = json.dumps(analysis.as_dict())

    assert analysis.rejected, "sem achado barrado este teste não prova nada"
    assert "gramática que o modelo improvisou" not in payload
    assert "There has been" not in payload
    # O contador pode ir: ele é medida, não conteúdo.
    assert analysis.as_dict()["unverified"] == 1


# --- O turno --------------------------------------------------------------- #


async def test_the_turn_files_the_rejected_finding_in_the_queue(prompts, conn):
    student = StudentRepo(conn).get_or_create("ana")
    session_id = SessionRepo(conn).current_or_start(student.id)
    runner = TurnRunner(
        provider=FakeProvider(replies="Tell me more."),
        prompts=prompts,
        conn=conn,
        analyzer=analyzer_for(UNCOVERED),
    )

    [event async for event in runner.run(session_id, STUDENT_TEXT)]

    (gap,) = UnverifiedFindingRepo(conn).backlog()
    assert gap.last_span == "There have been"
    assert gap.times == 1
    # O decalque da idade casou e é o que fez a análise rodar: guardado ao
    # lado do buraco porque muda o trabalho de curadoria.
    assert "calque.age-have" in gap.last_candidates
    # E a política não mudou: o barrado não virou correção para o aluno.
    assert CorrectionRepo(conn).count() == 0


async def test_a_turn_with_nothing_rejected_files_nothing(prompts, conn):
    """A fila não cresce com turno limpo — senão não seria fila de nada."""
    student = StudentRepo(conn).get_or_create("ana")
    session_id = SessionRepo(conn).current_or_start(student.id)
    runner = TurnRunner(
        provider=FakeProvider(replies="Nice."),
        prompts=prompts,
        conn=conn,
        analyzer=analyzer_for(json.dumps({"errors": []})),
    )

    [event async for event in runner.run(session_id, STUDENT_TEXT)]

    assert UnverifiedFindingRepo(conn).count() == 0


async def test_a_broken_queue_write_does_not_cost_the_lesson(prompts, conn, monkeypatch):
    """Curadoria nunca derruba o turno.

    O aluno já recebeu a aula quando esta escrita acontece; trocá-la por uma
    linha de telemetria de curadoria seria a inversão exata do que a tabela
    serve.
    """
    student = StudentRepo(conn).get_or_create("ana")
    session_id = SessionRepo(conn).current_or_start(student.id)
    runner = TurnRunner(
        provider=FakeProvider(replies="Tell me more."),
        prompts=prompts,
        conn=conn,
        analyzer=analyzer_for(UNCOVERED),
    )
    monkeypatch.setattr(
        runner._gaps, "add", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disco cheio"))
    )

    events = [event async for event in runner.run(session_id, STUDENT_TEXT)]

    assert type(events[-1]).__name__ == "Done"
    assert UnverifiedFindingRepo(conn).count() == 0


# --- O script -------------------------------------------------------------- #


@pytest.fixture(scope="module")
def curation_queue():
    script = Path(__file__).resolve().parent.parent / "scripts" / "curation_queue.py"
    spec = importlib.util.spec_from_file_location("curation_queue", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["curation_queue"] = module
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        del sys.modules["curation_queue"]


def test_the_script_reads_the_queue(curation_queue, conn, turn_id, tmp_path, monkeypatch, capsys):
    UnverifiedFindingRepo(conn).add(
        turn_id,
        span="There have been",
        span_key=span_key("There have been"),
        corrected="There has been",
        error_type="grammar",
        explanation="improvisada",
        candidates="calque.age-have",
    )
    path = tmp_path / "curation.db"
    monkeypatch.setattr(
        curation_queue, "get_settings", lambda: make_settings(db_path=path), raising=True
    )

    assert curation_queue.main([]) == 0
    out = capsys.readouterr().out
    assert "There have been" in out
    assert "calque.age-have" in out


def test_the_yaml_skeleton_carries_the_category_into_the_id(curation_queue, conn, turn_id):
    """O id tem que começar pela categoria, senão a carga da base barra.

    O esqueleto existe para encurtar a escrita do item, e um esqueleto que não
    carrega nem o que o validador exige não encurta nada.
    """
    UnverifiedFindingRepo(conn).add(
        turn_id,
        span="I lost the bus",
        span_key=span_key("I lost the bus"),
        corrected="I missed the bus",
        error_type="calque",
        explanation="improvisada",
    )

    rendered = curation_queue.render_yaml(UnverifiedFindingRepo(conn).backlog())

    (item,) = yaml.safe_load(rendered)["items"]
    assert item["id"] == "calque.TODO"
    assert item["category"] == "calque"
    assert item["wrong"] == ["I lost the bus"]
    assert item["right"] == ["I missed the bus"]


def test_an_unmappable_type_does_not_guess_the_category(curation_queue, conn, turn_id):
    """Chutar categoria faria o item barrar na carga com mensagem inútil."""
    UnverifiedFindingRepo(conn).add(
        turn_id,
        span="whatever",
        span_key=span_key("whatever"),
        corrected="whichever",
        error_type="other",
        explanation="improvisada",
    )

    rendered = curation_queue.render_yaml(UnverifiedFindingRepo(conn).backlog())

    (item,) = yaml.safe_load(rendered)["items"]
    assert item["id"] == "TODO.TODO"
    assert item["category"] == "TODO"


def test_the_skeleton_survives_a_span_with_both_quote_kinds(curation_queue, conn, turn_id):
    """O esqueleto existe para ser colado, e um YAML que não faz parse não serve.

    `repr` do Python escaparia com barra invertida e o arquivo quebraria na
    carga — num span que o aluno é perfeitamente capaz de digitar.
    """
    span = 'don\'t say "hi"'
    UnverifiedFindingRepo(conn).add(
        turn_id,
        span=span,
        span_key=span_key(span),
        corrected='do not say "hello"',
        error_type="grammar",
        explanation="improvisada",
    )

    rendered = curation_queue.render_yaml(UnverifiedFindingRepo(conn).backlog())

    (item,) = yaml.safe_load(rendered)["items"]
    assert item["wrong"] == [span]


def test_an_empty_queue_says_so_instead_of_printing_a_header(curation_queue):
    assert "Fila vazia" in curation_queue.render_table([])
