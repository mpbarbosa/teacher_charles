#!/usr/bin/env python3
"""Mede a correção estruturada contra o modelo de verdade.

A E4 pede número, não impressão: "a precisão medida fica registrada neste doc"
(plano §5, E4). Este script produz esse número. A suíte de frases é a mesma que
`tests/test_analyzer.py` usa como portão determinístico —
`tests/fixtures/known_errors.yaml` — mas aqui ela passa pelo modelo.

Dois modos, porque são duas perguntas diferentes:

    uv run python scripts/measure_analysis.py
        Qualidade da análise: em quantas frases com erro conhecido a correção
        sai, quantas citam a regra certa, quantas frases corretas são corrigidas
        por engano, quantas vezes o JSON degrada. A latência medida aqui é a
        otimista — chamadas de análise em sequência mantêm o prompt dela em
        cache.

    uv run python scripts/measure_analysis.py --double-turn 5
        O custo real do turno duplo: conversa e análise disputando o mesmo
        modelo, como no produto. É aqui que o cache de prefixo do Ollama
        aparece, porque as duas etapas têm prompts diferentes e uma joga a
        outra fora do cache.

Precisa de Ollama de pé com o modelo de `TC_LLM_MODEL` baixado.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from teacher_charles.config import get_settings  # noqa: E402
from teacher_charles.llm import build_provider, default_options  # noqa: E402
from teacher_charles.prompts import PromptLibrary  # noqa: E402
from teacher_charles.store.db import init_db  # noqa: E402
from teacher_charles.store.repo import SessionRepo, StudentRepo  # noqa: E402
from teacher_charles.teaching import Analyzed, Analyzer, Chunk, Done, TurnRunner  # noqa: E402
from teacher_charles.teaching.contrastive import load_base  # noqa: E402

SUITE_PATH = REPO_ROOT / "tests" / "fixtures" / "known_errors.yaml"


def load_suite() -> dict:
    return yaml.safe_load(SUITE_PATH.read_text(encoding="utf-8"))


def pct(part: int, whole: int) -> str:
    if whole == 0:
        return "—"
    return f"{100 * part / whole:.0f}% ({part}/{whole})"


def summarize(label: str, values: list[float]) -> str:
    if not values:
        return f"{label}: —"
    return (
        f"{label}: mediana {statistics.median(values) / 1000:.1f}s · "
        f"média {statistics.fmean(values) / 1000:.1f}s · "
        f"pior {max(values) / 1000:.1f}s"
    )


# --- modo 1: qualidade da análise ------------------------------------------ #


async def measure_quality(limit: int | None, allow_unverified: bool = False) -> int:
    settings = get_settings()
    suite = load_suite()
    base = load_base(settings.data_dir / "contrastive")
    prompts = PromptLibrary(settings.prompts_dir)
    provider = build_provider(settings)
    analyzer = Analyzer(
        provider=provider,
        prompts=prompts,
        base=base,
        require_curated_rule=settings.require_curated_rule and not allow_unverified,
    )

    wrong = suite["errors"][:limit]
    clean = suite["clean"][:limit]

    print(f"modelo: {settings.llm_model} · provider: {settings.provider}")
    print(f"base contrastiva: {len(base)} itens")
    policy = settings.require_curated_rule and not allow_unverified
    print(f"exige regra curada: {policy}")
    print(f"suíte: {len(wrong)} frases com erro, {len(clean)} corretas\n")

    # Uma chamada de aquecimento, descartada: a primeira paga a carga do modelo
    # do disco (~15s nesta máquina) e contaminaria a mediana.
    print("aquecendo...", flush=True)
    await analyzer.analyze("I have 30 years")

    produced = 0
    skipped = 0
    cited_expected = 0
    cited_any = 0
    degraded = 0
    retried = 0
    dropped_wrong = 0
    dropped_clean = 0
    latencies: list[float] = []

    print("\n--- frases com erro conhecido ---")
    for case in wrong:
        expected = set(case["rules"])
        analysis = await analyzer.analyze(case["text"])
        latencies.append(analysis.elapsed_ms)
        if analysis.degraded:
            degraded += 1
        if analysis.attempts > 1:
            retried += 1
        if analysis.skipped:
            skipped += 1

        got_rules = set(analysis.rule_ids)
        if analysis.findings:
            produced += 1
        if got_rules:
            cited_any += 1
        hit = bool(expected & got_rules)
        if hit:
            cited_expected += 1

        mark = "ok  " if hit else ("~   " if analysis.findings else "MISS")
        print(f"{mark} {analysis.elapsed_ms / 1000:5.1f}s  {case['text'][:52]!r}")
        print(f"        esperado: {sorted(expected)}  citado: {sorted(got_rules) or '—'}")
        for finding in analysis.findings:
            print(f"        -> {finding.span!r} => {finding.correction!r}")
        dropped_wrong += analysis.unverified
        if analysis.unverified:
            print(f"        {analysis.unverified} sem regra, barrada(s) pela política")
        if analysis.degraded:
            print(f"        degradada: {analysis.detail}")

    false_positives = 0
    fp_with_rule = 0
    print("\n--- frases corretas (não deveriam ser corrigidas) ---")
    for text in clean:
        analysis = await analyzer.analyze(text)
        latencies.append(analysis.elapsed_ms)
        if analysis.degraded:
            degraded += 1
        dropped_clean += analysis.unverified
        if analysis.skipped:
            skipped += 1
        if analysis.findings:
            false_positives += 1
            if analysis.rule_ids:
                fp_with_rule += 1
            print(f"FP  {text[:52]!r}")
            for finding in analysis.findings:
                print(
                    f"        -> {finding.span!r} => {finding.correction!r} "
                    f"[{finding.rule_id or 'sem regra'}]"
                )
        else:
            if analysis.skipped:
                note = "  (análise dispensada: sem candidato)"
            elif analysis.unverified:
                note = f"  ({analysis.unverified} barrada(s))"
            else:
                note = ""
            print(f"ok  {text[:52]!r}{note}")

    total = len(wrong) + len(clean)
    print("\n=== resultado ===")
    print(f"correção produzida em frase com erro: {pct(produced, len(wrong))}")
    print(f"  com regra curada citada:            {pct(cited_any, len(wrong))}")
    print(f"  com a regra ESPERADA citada:        {pct(cited_expected, len(wrong))}")
    print(f"frase correta corrigida por engano:   {pct(false_positives, len(clean))}")
    print(f"  dessas, citando regra:              {pct(fp_with_rule, len(clean))}")
    print(
        f"correção sem regra, barrada:          {dropped_wrong} em frase com erro, "
        f"{dropped_clean} em frase correta"
    )
    print(f"análise dispensada (sem candidato):   {pct(skipped, total)}")
    print(f"JSON exigiu retentativa:              {pct(retried, total)}")
    print(f"análise degradada (sem correção):     {pct(degraded, total)}")
    print(summarize("latência da análise isolada", latencies))

    await provider.aclose()
    return 0


# --- modo 2: o custo do turno duplo ---------------------------------------- #


async def measure_double_turn(turns: int, parallel: bool = False) -> int:
    settings = get_settings()
    suite = load_suite()
    base = load_base(settings.data_dir / "contrastive")
    prompts = PromptLibrary(settings.prompts_dir)
    provider = build_provider(settings)
    analyzer = Analyzer(provider=provider, prompts=prompts, base=base)

    # Banco temporário: medir não deve escrever no progresso do usuário.
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "measure.db")
        student = StudentRepo(conn).get_or_create("measurement")
        session_id = SessionRepo(conn).start(student.id)
        runner = TurnRunner(
            provider=provider,
            prompts=prompts,
            conn=conn,
            options=default_options(settings),
            analyzer=analyzer,
            analysis_after_reply=not parallel,
        )

        ordem = "análise em paralelo com a resposta" if parallel else "conversa primeiro"
        print(f"modelo: {settings.llm_model} · {turns} turno(s) · {ordem}\n")

        first: list[float] = []
        totals: list[float] = []
        analysis_ms: list[float] = []

        texts = [case["text"] for case in suite["errors"][:turns]]
        for index, text in enumerate(texts, start=1):
            corrections = 0
            async for event in runner.run(session_id, text):
                if isinstance(event, Analyzed):
                    corrections = len(event.analysis.findings)
                elif isinstance(event, Done):
                    first.append(event.first_chunk_ms)
                    totals.append(event.total_ms)
                    analysis_ms.append(event.analysis_ms)
                    print(
                        f"turno {index}: primeira palavra {event.first_chunk_ms / 1000:.1f}s · "
                        f"turno {event.total_ms / 1000:.1f}s · "
                        f"análise {event.analysis_ms / 1000:.1f}s · "
                        f"{corrections} correção(ões)"
                    )
                elif not isinstance(event, Chunk):
                    print(f"turno {index}: {event}")

        print("\n=== resultado ===")
        # O primeiro turno paga a carga do modelo; separar é o que torna o
        # número comparável com a medição da E2 (plano §1.4).
        print(summarize("primeira palavra (todos)", first))
        print(summarize("primeira palavra (sem o 1º)", first[1:]))
        print(summarize("turno completo", totals))
        print(summarize("análise (em paralelo)", analysis_ms))
        conn.close()

    await provider.aclose()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--double-turn",
        type=int,
        metavar="N",
        help="mede N turnos completos (conversa + análise no mesmo modelo)",
    )
    parser.add_argument("--limit", type=int, help="usa só as N primeiras frases de cada lista")
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="desliga a exigência de regra curada: mostra o que o modelo propõe sozinho",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="com --double-turn: dispara a análise junto com a resposta, em vez de depois",
    )
    args = parser.parse_args()

    if args.double_turn:
        return asyncio.run(measure_double_turn(args.double_turn, parallel=args.parallel))
    return asyncio.run(measure_quality(args.limit, allow_unverified=args.allow_unverified))


if __name__ == "__main__":
    raise SystemExit(main())
