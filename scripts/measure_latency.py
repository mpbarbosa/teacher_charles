#!/usr/bin/env python3
"""Mede a latência do turno, e tenta explicá-la.

A E7 pede "latência medida e registrada", e há duas dívidas nominais esperando
por este script:

- §1.4: *"7,4s no primeiro turno mesmo com aquecimento, contra 0,7s nos
  seguintes. Não está explicado."*
- §5.2: *"a primeira palavra degrada turno a turno (0,1s → 3,8s → 6,1s → 7,9s
  → 10,8s em 5 turnos)"*, com a hipótese de que cada chamada de análise joga o
  prompt da conversa fora do cache de prefixo do Ollama.

As duas são sobre **por quê**, e cronômetro sozinho não responde: é preciso o
que o servidor conta de cada chamada — quanto do prompt estava em cache, quanto
foi avaliado agora, quanto custou carregar o modelo. Daí
``ChatProvider.last_metrics`` existir.

    uv run python scripts/measure_latency.py
        Uma sessão realista: turnos com erro e turnos corretos, análise ligada,
        como no produto. Por turno: primeira palavra, turno completo, análise e
        as métricas do servidor. É aqui que a degradação aparece — ou não.

    uv run python scripts/measure_latency.py --no-analysis
        A mesma sessão sem a etapa de análise. A diferença entre as duas é o
        preço da análise, incluindo o que ela faz com o cache.

    uv run python scripts/measure_latency.py --cold
        Descarrega o modelo e mede o primeiro turno com e sem aquecimento,
        decompondo em carga de modelo, avaliação de prompt e geração. É a
        resposta para a dívida da §1.4.

    uv run python scripts/measure_latency.py --analysis-model MODELO
        A sessão com a análise num segundo modelo. Se a hipótese da §5.2 estiver
        certa, cada modelo mantém o próprio cache e a conversa para de ser
        despejada.

Precisa de Ollama de pé com o modelo de `TC_LLM_MODEL` baixado.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import tempfile
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import httpx  # noqa: E402

from teacher_charles.config import get_settings  # noqa: E402
from teacher_charles.llm import OllamaProvider, default_options  # noqa: E402
from teacher_charles.llm.base import (  # noqa: E402
    DEFAULT_OPTIONS,
    GenerationMetrics,
    GenerationOptions,
    Message,
)
from teacher_charles.prompts import PromptLibrary  # noqa: E402
from teacher_charles.store.db import init_db  # noqa: E402
from teacher_charles.store.repo import SessionRepo, StudentRepo  # noqa: E402
from teacher_charles.teaching import (  # noqa: E402
    Analysis,
    Analyzer,
    Chunk,
    Done,
    TurnRunner,
)
from teacher_charles.teaching.contrastive import load_base  # noqa: E402

# Uma sessão realista: erro e acerto alternados, porque os dois custam coisas
# diferentes — turno sem candidato na base nem chama a análise (§5.2).
SESSION = [
    "Hi Charles, I have 35 years and I work with marketing in a big company.",
    "Last weekend I went to the beach with my family and we had a great time.",
    "Can I make a question about the report you sent me yesterday?",
    "I have been working here since 2019, and I still like the job.",
    "My manager is very comprehensive about my schedule when my son is sick.",
    "The weather has been strange this year, with rain almost every afternoon.",
]


class HintOnlyAnalyzer(Analyzer):
    """O analisador sem a chamada ao modelo.

    Separa duas causas que andavam juntas na suspeita da §5.2: a **chamada** de
    análise, que ocupa o servidor entre dois turnos de conversa, e a **dica
    curada**, que entra no fim do prompt e muda a cada turno. Desligar a
    análise inteira (``--no-analysis``) tira as duas ao mesmo tempo e não diz
    qual delas custa. Esta tira só a chamada.
    """

    async def analyze(self, student_text, cefr_level="B1", max_findings=None):
        return Analysis(skipped=True)


class RecordingProvider(OllamaProvider):
    """O provider do produto, anotando as métricas de cada chamada concluída.

    Subclasse no script, e não gancho no provider: medição não deveria
    atravessar o código que serve o aluno. Vale para chamadas em série, que é
    como o produto roda por default (``analysis_after_reply=True``).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.recorded: list[GenerationMetrics] = []

    async def stream(
        self,
        messages: Sequence[Message],
        options: GenerationOptions = DEFAULT_OPTIONS,
    ) -> AsyncIterator[str]:
        async for piece in super().stream(messages, options):
            yield piece
        if self.last_metrics is not None:
            self.recorded.append(self.last_metrics)

    def take(self) -> list[GenerationMetrics]:
        taken, self.recorded = self.recorded, []
        return taken


def describe(metrics: GenerationMetrics) -> str:
    ratio = metrics.cache_hit_ratio
    cached = "—" if ratio is None else f"{100 * ratio:.0f}%"
    speed = metrics.output_tokens_per_s
    parts = [
        f"prompt {metrics.prompt_tokens or 0} tok ({cached} em cache, "
        f"{(metrics.prompt_eval_ms or 0) / 1000:.1f}s)",
        f"saída {metrics.output_tokens or 0} tok"
        + ("" if speed is None else f" a {speed:.1f} tok/s"),
    ]
    if metrics.load_ms and metrics.load_ms >= 50:
        parts.append(f"carga {metrics.load_ms / 1000:.1f}s")
    return " · ".join(parts)


def summarize(label: str, values: list[float]) -> str:
    if not values:
        return f"{label}: —"
    return (
        f"{label}: mediana {statistics.median(values) / 1000:.1f}s · pior {max(values) / 1000:.1f}s"
    )


async def unload(base_url: str, model: str) -> None:
    """Tira o modelo da memória, para medir o custo de carregá-lo."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        await client.post(
            f"{base_url}/api/generate",
            json={"model": model, "keep_alive": 0},
        )
    # O Ollama descarrega de forma assíncrona; um instante de folga evita medir
    # a descarga como se fosse carga.
    await asyncio.sleep(2)


def build(settings, provider, analyzer, conn):
    return TurnRunner(
        provider=provider,
        prompts=PromptLibrary(settings.prompts_dir),
        conn=conn,
        options=default_options(settings),
        analyzer=analyzer,
        analysis_after_reply=settings.analysis_after_reply,
    )


# --- modo 1: a sessão ------------------------------------------------------ #


async def measure_session(
    with_analysis: bool,
    analysis_model: str | None,
    warm: bool,
    hint_only: bool = False,
) -> int:
    settings = get_settings()
    prompts = PromptLibrary(settings.prompts_dir)
    base = load_base(settings.data_dir / "contrastive")

    chat = RecordingProvider(
        model=settings.llm_model,
        base_url=settings.ollama_url,
        request_timeout_s=settings.request_timeout_s,
    )
    analysis_provider = None
    if with_analysis:
        analysis_provider = (
            RecordingProvider(
                model=analysis_model,
                base_url=settings.ollama_url,
                request_timeout_s=settings.request_timeout_s,
            )
            if analysis_model
            else chat
        )
    make_analyzer = HintOnlyAnalyzer if hint_only else Analyzer
    analyzer = (
        make_analyzer(
            provider=analysis_provider,
            prompts=prompts,
            base=base,
            require_curated_rule=settings.require_curated_rule,
        )
        if analysis_provider is not None
        else None
    )

    print(f"modelo da conversa: {chat.model}")
    if analyzer is None:
        print("análise: desligada (e sem dica curada no prompt)")
    elif hint_only:
        print("análise: só a dica curada, sem chamada ao modelo")
    else:
        print(f"análise: ligada ({analysis_provider.model})")

    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "latency.db")
        student = StudentRepo(conn).get_or_create("medicao")
        session_id = SessionRepo(conn).start(student.id)
        runner = build(settings, chat, analyzer, conn)

        try:
            if warm:
                # O mesmo prefixo que o turno manda, pelo mesmo caminho do app:
                # aquecer outra coisa mediria outra coisa (plano §5.5).
                started = asyncio.get_running_loop().time()
                await chat.complete(
                    [*runner.stable_prefix(), Message(role="user", content="Hello.")],
                    GenerationOptions(temperature=0.0, max_tokens=1, think=False),
                )
                warmed = chat.take()[-1]
                print(
                    f"aquecimento: {asyncio.get_running_loop().time() - started:.1f}s · "
                    f"{describe(warmed)}"
                )
            print()

            first_words: list[float] = []
            totals: list[float] = []
            analyses: list[float] = []

            for index, text in enumerate(SESSION, start=1):
                done: Done | None = None
                async for event in runner.run(session_id, text, student):
                    if isinstance(event, Done):
                        done = event
                    elif not isinstance(event, Chunk):
                        continue
                if done is None:
                    print(f"{index}. FALHOU: {text[:50]}")
                    continue

                first_words.append(done.first_chunk_ms)
                totals.append(done.total_ms)
                if done.analysis_ms:
                    analyses.append(done.analysis_ms)

                calls = chat.take()
                conversation = calls[0] if calls else None
                print(
                    f"{index}. primeira palavra {done.first_chunk_ms / 1000:.1f}s · "
                    f"turno {done.total_ms / 1000:.1f}s · "
                    f"análise {done.analysis_ms / 1000:.1f}s"
                )
                if conversation is not None:
                    print(f"   conversa: {describe(conversation)}")
                for extra in calls[1:]:
                    print(f"   análise:  {describe(extra)}")
                if analysis_provider is not None and analysis_provider is not chat:
                    for extra in analysis_provider.take():
                        print(f"   análise:  {describe(extra)}")

            print()
            print(summarize("primeira palavra", first_words))
            print(summarize("turno completo  ", totals))
            print(summarize("análise         ", analyses))
            if len(first_words) >= 4:
                early = statistics.median(first_words[:2])
                late = statistics.median(first_words[-2:])
                print(
                    f"\nprimeira palavra, 2 primeiros turnos vs 2 últimos: "
                    f"{early / 1000:.1f}s -> {late / 1000:.1f}s"
                )
        finally:
            await chat.aclose()
            if analysis_provider is not None and analysis_provider is not chat:
                await analysis_provider.aclose()
            conn.close()
    return 0


# --- modo 2: o primeiro turno ---------------------------------------------- #


async def measure_cold() -> int:
    """Decompõe o primeiro turno, com e sem aquecimento.

    A dívida da §1.4: 7,4s no primeiro turno mesmo aquecendo. Carga de modelo?
    Avaliação de prompt? Geração? O servidor sabe, e agora nós perguntamos.
    """
    settings = get_settings()
    prompts = PromptLibrary(settings.prompts_dir)
    persona = prompts.get("persona")
    options = default_options(settings)

    for warm in (False, True):
        chat = RecordingProvider(
            model=settings.llm_model,
            base_url=settings.ollama_url,
            request_timeout_s=settings.request_timeout_s,
        )
        try:
            await unload(settings.ollama_url, chat.model)
            label = "com aquecimento" if warm else "sem aquecimento"
            print(f"--- primeiro turno, modelo descarregado, {label} ---")

            if warm:
                started = asyncio.get_running_loop().time()
                await chat.complete(
                    [
                        Message(role="system", content=persona),
                        Message(role="user", content="Hello."),
                    ],
                    GenerationOptions(temperature=0.0, max_tokens=1, think=False),
                )
                elapsed = asyncio.get_running_loop().time() - started
                metrics = chat.take()[-1]
                print(f"  aquecimento: {elapsed:.1f}s · {describe(metrics)}")

            messages = [
                Message(role="system", content=persona),
                Message(role="user", content=SESSION[0]),
            ]  # turno cru, sem nível nem modo: aqui se mede carga, não prefixo
            started = asyncio.get_running_loop().time()
            first_word = None
            async for piece in chat.stream(messages, options):
                if piece and first_word is None:
                    first_word = asyncio.get_running_loop().time() - started
            total = asyncio.get_running_loop().time() - started
            metrics = chat.take()[-1]
            print(f"  primeira palavra {first_word or 0:.1f}s · turno {total:.1f}s")
            print(f"  {describe(metrics)}")
            print()
        finally:
            await chat.aclose()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-analysis", action="store_true", help="sessão sem a análise")
    parser.add_argument("--cold", action="store_true", help="decompõe o primeiro turno")
    parser.add_argument("--analysis-model", default=None, help="roda a análise em outro modelo")
    parser.add_argument("--no-warmup", action="store_true", help="não aquece antes da sessão")
    parser.add_argument(
        "--hint-only",
        action="store_true",
        help="mantém a dica curada no prompt e tira só a chamada de análise",
    )
    args = parser.parse_args()

    if args.cold:
        return asyncio.run(measure_cold())
    return asyncio.run(
        measure_session(
            with_analysis=not args.no_analysis,
            analysis_model=args.analysis_model,
            warm=not args.no_warmup,
            hint_only=args.hint_only,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
