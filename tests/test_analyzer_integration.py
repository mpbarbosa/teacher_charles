"""Teste de integração da correção estruturada (E4), contra o Ollama local.

É o critério de pronto da E4 no caminho real: o modelo de verdade produz JSON,
o JSON vira correção verificada e a correção vira linha no banco, com a conversa
saindo em streaming ao mesmo tempo.

O que este arquivo afirma é o **contrato**, não a qualidade do modelo: span que
existe na fala do aluno, regra citada vinda da base, turno que fecha. A
qualidade é medida à parte, com número, por ``scripts/measure_analysis.py``, e
registrada no plano §5.2 — cobrar aqui a correção perfeita de um 3B seria testar
o modelo, não o código.

Pula sozinho sem Ollama. Rode só os de integração com::

    uv run pytest -m integration
"""

from __future__ import annotations

import pytest

from teacher_charles.config import REPO_ROOT
from teacher_charles.llm.base import GenerationOptions
from teacher_charles.prompts import PromptLibrary
from teacher_charles.store import db
from teacher_charles.store.repo import CorrectionRepo, SessionRepo, StudentRepo
from teacher_charles.teaching import Analyzed, Analyzer, Chunk, Done, TurnRunner
from teacher_charles.teaching.contrastive import load_base

pytestmark = pytest.mark.integration

AGE_SENTENCE = "Hi Charles, I have 35 years and I work with marketing."


@pytest.fixture
def prompts() -> PromptLibrary:
    return PromptLibrary(REPO_ROOT / "prompts")


@pytest.fixture
def base():
    return load_base(REPO_ROOT / "data" / "contrastive")


async def test_real_analysis_returns_a_verified_correction(ollama_provider, prompts, base):
    analyzer = Analyzer(provider=ollama_provider, prompts=prompts, base=base)
    analysis = await analyzer.analyze(AGE_SENTENCE)

    print(
        f"\n  modelo={ollama_provider.model}  {analysis.elapsed_ms / 1000:.1f}s  "
        f"tentativas={analysis.attempts}  achados={len(analysis.findings)}"
    )
    for finding in analysis.findings:
        print(f"    {finding.span!r} => {finding.correction!r}  [{finding.rule_id}]")

    assert not analysis.degraded, f"o modelo não entregou JSON: {analysis.detail}"
    assert analysis.findings, "nenhuma correção para uma frase com erro conhecido"

    for finding in analysis.findings:
        # A garantia que vale: o trecho é literal da fala do aluno.
        assert finding.span in AGE_SENTENCE
        assert finding.correction and finding.explanation

    # A base casou `calque.age-have` nesta frase, então a regra tem que ser
    # citada e a explicação tem que ser a curada — não a do modelo.
    assert "calque.age-have" in analysis.candidates
    cited = next(f for f in analysis.findings if f.rule_id == "calque.age-have")
    assert cited.explanation == base.get("calque.age-have").explanation


async def test_real_analysis_of_clean_english_is_held_back_by_the_policy(
    ollama_provider, prompts, base
):
    """Inglês correto: o que o modelo inventar não passa da política.

    Medido num 3B (plano §5.2), ele "corrige" frase correta em 2 de 3 casos, e
    às vezes ensinando erro. Nenhuma dessas correções citava regra curada, e é
    por isso que a política existe. O que este teste cobra é o código: toda
    correção que sobra cita regra que existe na base, e nada escapa com span
    inventado.
    """
    text = "There have been many changes since the new manager arrived."
    analyzer = Analyzer(provider=ollama_provider, prompts=prompts, base=base)
    analysis = await analyzer.analyze(text)

    print(
        f"\n  {ollama_provider.model}: {len(analysis.findings)} achado(s), "
        f"{analysis.unverified} barrada(s) por não citar regra"
    )
    for finding in analysis.findings:
        print(f"    {finding.span!r} => {finding.correction!r}  [{finding.rule_id}]")

    assert analysis.candidates == (), "a base não deveria levantar regra aqui"
    for finding in analysis.findings:
        assert finding.rule_id in base, "correção sem regra curada passou da política"
        assert finding.span in text
        assert finding.correction.strip().lower() != finding.span.strip().lower()


async def test_real_double_turn_streams_and_persists(ollama_provider, prompts, base, tmp_path):
    """Conversa e correção no mesmo turno, com o modelo de verdade.

    O número que interessa está no print: a análise roda em paralelo, então o
    turno completo tem que custar bem menos que a soma das duas etapas. O custo
    medido está no plano §5.2.
    """
    conn = db.init_db(tmp_path / "integration.db")
    try:
        student = StudentRepo(conn).get_or_create("integration")
        session_id = SessionRepo(conn).start(student.id)
        runner = TurnRunner(
            provider=ollama_provider,
            prompts=prompts,
            conn=conn,
            analyzer=Analyzer(provider=ollama_provider, prompts=prompts, base=base),
        )

        events = [event async for event in runner.run(session_id, AGE_SENTENCE)]
        kinds = [type(e).__name__ for e in events]

        chunks = [e for e in events if isinstance(e, Chunk)]
        done = events[-1]
        assert isinstance(done, Done), f"o turno não fechou: {events[-1]}"
        assert len(chunks) > 5, f"esperava streaming, vieram {len(chunks)} pedaço(s)"

        analyzed = next((e for e in events if isinstance(e, Analyzed)), None)
        assert analyzed is not None, "o turno não produziu análise"
        assert kinds.index("Analyzed") < kinds.index("Done")

        print(
            f"\n  modelo={ollama_provider.model}\n"
            f"  primeira palavra: {done.first_chunk_ms / 1000:.1f}s\n"
            f"  turno completo:   {done.total_ms / 1000:.1f}s\n"
            f"  análise:          {done.analysis_ms / 1000:.1f}s "
            f"({len(analyzed.analysis.findings)} correção(ões))\n"
            f"  resposta: {done.reply[:120]!r}"
        )

        # A correção virou dado, que é o ponto da E4.
        rows = CorrectionRepo(conn).for_turn(analyzed.student_turn_id)
        assert len(rows) == len(analyzed.analysis.findings)
        if rows:
            assert rows[0].span in AGE_SENTENCE
    finally:
        conn.close()


# --- E5: o nível muda a resposta de verdade ------------------------------- #


async def test_the_level_changes_the_real_reply(ollama_provider, prompts, base, tmp_path):
    """O critério de aceite da E5 no caminho real.

    A MESMA fala do aluno, mandada com o partial de A2 e com o de C1, tem que
    voltar diferente. O que se afirma aqui é o contrato — o ajuste chega ao
    modelo e muda a saída —, não a qualidade do ajuste: o quanto ele muda é
    medido por ``scripts/measure_level.py --reply`` e registrado no plano §5.3.
    """
    from teacher_charles.store.repo import StudentRepo as Students
    from teacher_charles.teaching import Chunk as Piece

    conn = db.init_db(tmp_path / "level.db")
    try:
        students = Students(conn)
        sessions = SessionRepo(conn)
        runner = TurnRunner(
            provider=ollama_provider,
            prompts=prompts,
            conn=conn,
            options=GenerationOptions(temperature=0.0, think=False),
        )

        replies = {}
        for level in ("A2", "C1"):
            student = students.get_or_create(f"aluno-{level}", cefr_level=level)
            session_id = sessions.start(student.id)
            pieces = [
                event.text
                async for event in runner.run(session_id, AGE_SENTENCE, student)
                if isinstance(event, Piece)
            ]
            replies[level] = "".join(pieces).strip()
            print(f"\n  [{level}] {replies[level]}")

        for level, reply in replies.items():
            assert reply, f"o modelo não respondeu nada em {level}"
        assert replies["A2"] != replies["C1"], (
            "a mesma fala com níveis diferentes voltou idêntica: "
            "o ajuste por nível não chegou ao modelo"
        )
    finally:
        conn.close()
