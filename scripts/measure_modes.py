#!/usr/bin/env python3
"""Mede os três modos contra o modelo de verdade.

O critério de pronto da E6 é "os três modos funcionam na UI e o drill usa erros
de fato cometidos, não genéricos". A segunda metade é verificável sem modelo —
e os testes a verificam. A primeira precisa de modelo: *comportamento de fato
distinto* (plano §4) não se prova lendo três arquivos de prompt.

    uv run python scripts/measure_modes.py
        A mesma fala do aluno nos três modos, com o mesmo erro plantado no
        error log. Imprime cada resposta e o que dá para contar nela: tamanho,
        quantas correções a prosa traz, se o drill pede uma tarefa ao aluno.

    uv run python scripts/measure_modes.py --turns 3
        Um drill de verdade, com várias trocas: é onde se vê se o modelo
        mantém uma regra só ou se dispersa.

Precisa de Ollama de pé com o modelo de `TC_LLM_MODEL` baixado.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from teacher_charles.config import get_settings  # noqa: E402
from teacher_charles.llm import build_provider, default_options  # noqa: E402
from teacher_charles.prompts import PromptLibrary  # noqa: E402
from teacher_charles.store.db import init_db  # noqa: E402
from teacher_charles.store.models import MODES  # noqa: E402
from teacher_charles.store.repo import (  # noqa: E402
    CorrectionRepo,
    SessionRepo,
    StudentRepo,
    TurnRepo,
)
from teacher_charles.teaching import Analysis, Analyzer, Chunk, TurnRunner  # noqa: E402
from teacher_charles.teaching.contrastive import load_base  # noqa: E402
from teacher_charles.teaching.modes import DrillPlanner  # noqa: E402

# A fala do aluno, e o erro que ela comete. O mesmo nos três modos: o que muda
# é o modo, não a entrada.
PROBE = "I have 35 years and I work with marketing. Can I make a question about the report?"

SEEDED_ERRORS = (
    ("I have 35 years", "I am 35 years old", "calque.age-have"),
    ("make a question", "ask a question", "calque.make-question"),
    ("I have 40 years", "I am 40 years old", "calque.age-have"),
)

# Sinais contáveis na prosa do professor. Grosseiros de propósito — servem para
# comparar modos entre si, não para virar nota.
_QUOTED = re.compile(r'"[^"]{3,}"')
_QUESTION = re.compile(r"\?")


class HintOnlyAnalyzer(Analyzer):
    """O analisador do produto, sem a chamada de análise.

    A dica curada que vai no prompt da resposta sai do casamento por padrão e é
    de graça — e tirá-la daqui mediria um produto que não existe, porque no
    produto ela está sempre lá. Já a etapa de análise custa uma chamada ao
    modelo e não muda a prosa do professor: aqui se mede a prosa. A qualidade
    da correção estruturada é medida em ``scripts/measure_analysis.py``.
    """

    async def analyze(self, student_text, cefr_level="B1", max_findings=None) -> Analysis:
        return Analysis(skipped=True)


def seed(conn, student_id: int) -> None:
    """Planta no banco o error log que o drill vai usar."""
    sessions = SessionRepo(conn)
    turns = TurnRepo(conn)
    corrections = CorrectionRepo(conn)
    session_id = sessions.start(student_id)
    for span, corrected, rule_id in SEEDED_ERRORS:
        turn_id = turns.append(session_id, "student", span)
        corrections.add(
            turn_id,
            span=span,
            corrected=corrected,
            error_type="calque",
            explanation="curada",
            rule_id=rule_id,
        )
    sessions.end(session_id)


def describe(reply: str) -> str:
    words = reply.split()
    sentences = [s for s in re.split(r"[.!?]+", reply) if s.strip()]
    return (
        f"{len(words)} palavras · {len(words) / max(len(sentences), 1):.1f} por frase · "
        f"{len(_QUOTED.findall(reply))} trechos entre aspas · "
        f"{len(_QUESTION.findall(reply))} perguntas"
    )


async def run(turns_per_mode: int) -> int:
    settings = get_settings()
    prompts = PromptLibrary(settings.prompts_dir)
    base = load_base(settings.data_dir / "contrastive")
    provider = build_provider(settings)

    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "modes.db")
        student = StudentRepo(conn).get_or_create("medicao")
        seed(conn, student.id)

        planner = DrillPlanner(conn, base)
        sessions = SessionRepo(conn)
        runner = TurnRunner(
            provider=provider,
            prompts=prompts,
            conn=conn,
            options=default_options(settings),
            planner=planner,
            analyzer=HintOnlyAnalyzer(provider=provider, prompts=prompts, base=base),
        )

        print(f"modelo: {provider.model}")
        print(f"error log plantado: {len(SEEDED_ERRORS)} correções com regra curada")
        print(f"fala do aluno: {PROBE!r}\n")

        try:
            for mode in MODES:
                target = planner.pick(student.id) if mode == "drill" else None
                session_id = sessions.switch_mode(
                    student.id, mode, target.rule_id if target is not None else None
                )
                if mode == "drill":
                    rule = target.rule_id if target is not None else "nenhuma"
                    print(f"--- {mode} (treinando {rule}) ---")
                else:
                    print(f"--- {mode} ---")

                for turn in range(turns_per_mode):
                    text = PROBE if turn == 0 else "I am 35 years old. Is this correct?"
                    pieces = [
                        event.text
                        async for event in runner.run(session_id, text, student)
                        if isinstance(event, Chunk)
                    ]
                    reply = "".join(pieces).strip()
                    if turn:
                        print(f"  aluno: {text}")
                    print(f"  {describe(reply)}")
                    print("  " + reply.replace("\n", "\n  "))
                print()
        finally:
            await provider.aclose()
            conn.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--turns",
        type=int,
        default=1,
        help="quantos turnos por modo (default: 1; use 3 para ver o drill andar)",
    )
    args = parser.parse_args()
    return asyncio.run(run(max(args.turns, 1)))


if __name__ == "__main__":
    raise SystemExit(main())
