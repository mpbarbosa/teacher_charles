#!/usr/bin/env python3
"""Mede a estimativa de nível e o efeito do nível na resposta.

Duas perguntas diferentes, dois modos — e o primeiro não precisa de modelo
nenhum, porque a rubrica é determinística:

    uv run python scripts/measure_level.py
        Qualidade da estimativa: matriz de confusão sobre
        `tests/fixtures/level_samples.yaml`, acerto exato, acerto a um degrau e
        os números de cada amostra. É o que se usa para recalibrar as
        constantes de `teaching/level.py`. Roda em milissegundos, sem Ollama.

    uv run python scripts/measure_level.py --reply
        O critério de aceite da E5 no caminho real: a MESMA fala do aluno
        mandada ao modelo com cada nível, para ver se o registro da resposta
        muda de fato. Precisa de Ollama de pé com o modelo de `TC_LLM_MODEL`.

    uv run python scripts/measure_level.py --hysteresis
        Quantos turnos a política leva para mover o nível de um aluno cujas
        amostras são todas do mesmo nível. Mostra o custo da inércia: ela é o
        que impede o nível de oscilar, e também o que atrasa o acerto.

O resultado fica registrado no plano §5.3.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from teacher_charles.config import get_settings  # noqa: E402
from teacher_charles.llm import build_provider, default_options  # noqa: E402
from teacher_charles.llm.base import Message  # noqa: E402
from teacher_charles.prompts import PromptLibrary  # noqa: E402
from teacher_charles.store.models import CEFR_LEVELS  # noqa: E402
from teacher_charles.teaching.level import (  # noqa: E402
    AGREEING_TURNS,
    MIN_WORDS,
    decide,
    estimate,
    index_of,
    partial_name,
)

SUITE_PATH = REPO_ROOT / "tests" / "fixtures" / "level_samples.yaml"

# A fala usada no modo --reply. Tem erro conhecido de propósito: o que interessa
# é ver o registro da correção mudar com o nível, não só a prosa.
PROBE = "I have 35 years and I work with marketing. Yesterday I go to a meeting with my boss."


def load_suite() -> dict[str, list[str]]:
    return yaml.safe_load(SUITE_PATH.read_text(encoding="utf-8"))


def pct(part: int, whole: int) -> str:
    if whole == 0:
        return "—"
    return f"{100 * part / whole:.0f}% ({part}/{whole})"


# --- modo 1: qualidade da estimativa --------------------------------------- #


def measure_quality(verbose: bool = True) -> int:
    suite = load_suite()
    labels = [level for level in CEFR_LEVELS if level in suite]
    confusion = {true: dict.fromkeys(CEFR_LEVELS, 0) for true in labels}
    exact = adjacent = total = 0
    scores: dict[str, list[float]] = {level: [] for level in labels}

    for level in labels:
        if verbose:
            print(f"\n{level}")
        for text in suite[level]:
            guess = estimate(text)
            total += 1
            distance = abs(index_of(guess.level) - index_of(level))
            exact += distance == 0
            adjacent += distance <= 1
            confusion[level][guess.level] += 1
            scores[level].append(guess.score)
            if verbose:
                mark = " " if distance == 0 else ("~" if distance == 1 else "!")
                print(f"  {mark} {guess.summary}")
                print(f"      {text[:88]}")

    print("\n--- matriz de confusão (linha = rótulo, coluna = estimativa) ---")
    print("        " + "".join(f"{level:>5}" for level in CEFR_LEVELS))
    for level in labels:
        row = "".join(f"{confusion[level][guess]:>5}" for guess in CEFR_LEVELS)
        print(f"  {level:<5}" + row)

    print("\n--- score por nível ---")
    for level in labels:
        values = scores[level]
        print(
            f"  {level}: mediana {statistics.median(values):.2f} · "
            f"min {min(values):.2f} · max {max(values):.2f}"
        )

    print("\n--- resumo ---")
    print(f"  acerto exato:        {pct(exact, total)}")
    print(f"  acerto a um degrau:  {pct(adjacent, total)}")
    if adjacent < total:
        print("  ATENÇÃO: há amostra errada por dois degraus — a histerese não cobre isso.")
    return 0


# --- modo 2: a inércia da política ----------------------------------------- #


def measure_hysteresis() -> int:
    """Quantos turnos até o nível andar, por nível de partida.

    Mede o preço da inércia: ela existe para o nível não oscilar a cada
    mensagem, e o preço é demorar a acertar quem foi classificado errado na
    entrada.
    """
    suite = load_suite()
    print(f"janela: {AGREEING_TURNS} turnos contados · mínimo de {MIN_WORDS} palavras por turno\n")

    for declared in CEFR_LEVELS:
        for real in (level for level in CEFR_LEVELS if level in suite):
            current = declared
            recent: list[str] = []
            path = [current]
            turn = 0
            for turn in range(1, 25):
                text = suite[real][(turn - 1) % len(suite[real])]
                guess = estimate(text)
                if guess.counted:
                    recent.insert(0, guess.level)
                moved, _ = decide(current, recent)
                if moved != current:
                    current = moved
                    path.append(f"{current}@{turn}")
                    recent = []
                if current == real:
                    break
            arrow = " -> ".join(path)
            if current != real:
                verdict = "não chegou"
            elif len(path) == 1:
                verdict = "já estava lá"
            else:
                verdict = f"chegou em {turn} turnos"
            print(f"  declarado {declared}, escreve como {real}: {arrow}  ({verdict})")
    return 0


# --- modo 3: o efeito do nível na resposta --------------------------------- #


async def measure_reply(levels: list[str]) -> int:
    settings = get_settings()
    prompts = PromptLibrary(settings.prompts_dir)
    provider = build_provider(settings)
    options = default_options(settings)

    print(f"modelo: {provider.model}")
    print(f"fala do aluno: {PROBE!r}\n")

    try:
        for level in levels:
            name = partial_name(level)
            messages = [Message(role="system", content=prompts.get("persona"))]
            if prompts.exists(name):
                messages.append(Message(role="system", content=prompts.get(name)))
            messages.append(Message(role="user", content=PROBE))

            started = asyncio.get_running_loop().time()
            reply = await provider.complete(messages, options)
            elapsed = asyncio.get_running_loop().time() - started

            words = reply.split()
            sentences = [
                s for s in reply.replace("!", ".").replace("?", ".").split(".") if s.strip()
            ]
            print(f"--- {level} ({elapsed:.1f}s) ---")
            print(f"  {len(words)} palavras · {len(words) / max(len(sentences), 1):.1f} por frase")
            print("  " + reply.strip().replace("\n", "\n  "))
            print()
    finally:
        await provider.aclose()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reply", action="store_true", help="manda a mesma fala em cada nível")
    parser.add_argument(
        "--hysteresis", action="store_true", help="quantos turnos até o nível andar"
    )
    parser.add_argument("--quiet", action="store_true", help="só o resumo, sem amostra por amostra")
    parser.add_argument(
        "--levels",
        default="A2,C1",
        help="níveis do modo --reply (default: A2,C1)",
    )
    args = parser.parse_args()

    if args.reply:
        levels = [level.strip().upper() for level in args.levels.split(",") if level.strip()]
        unknown = [level for level in levels if level not in CEFR_LEVELS]
        if unknown:
            parser.error(f"nível desconhecido: {unknown}")
        return asyncio.run(measure_reply(levels))
    if args.hysteresis:
        return measure_hysteresis()
    return measure_quality(verbose=not args.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
