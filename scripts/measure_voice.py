#!/usr/bin/env python3
"""Mede a voz da Fase 1.5 nesta máquina, antes de confiar nela.

O plano §1.3 diz que a Fase 1.5 **começa por uma medição, não por
integração**: a 7-8 tok/s a conta de 1-2s por turno é apertada antes de o
áudio entrar. Este script faz essa conta com números desta máquina.

    uv run python scripts/measure_voice.py --tts
        Carga e fator de tempo real do Kokoro, sentença por sentença. O número
        que decide a Fase 1.5 é o RTF: acima de 1,0 a síntese não acompanha a
        reprodução e a voz engasga entre sentenças.

    uv run python scripts/measure_voice.py --stt
        Carga e latência do Whisper em falas de 3, 6 e 10 segundos. O áudio é
        gerado pelo Kokoro — ver a ressalva em "O que esta medição NÃO diz".

    uv run python scripts/measure_voice.py --stt --audio fala.wav --say "o que eu disse"
        O mesmo, com áudio de verdade. É assim que se mede o que importa:
        sotaque brasileiro, microfone real, cozinha ao fundo.

    uv run python scripts/measure_voice.py --record 6
        Grava 6 segundos do microfone desta máquina e mede em cima deles.

    uv run python scripts/measure_voice.py --budget
        Junta tudo: STT + primeira sentença do professor + TTS dessa sentença,
        contra a meta de 1-2s. Precisa de Ollama de pé.

O resultado fica registrado no plano §5.6.

## O que esta medição NÃO diz

Sem ``--audio`` nem ``--record``, o áudio do modo ``--stt`` é fala
**sintética**: limpa, sem sotaque e sem ruído. Isso mede latência honestamente
(o Whisper leva o mesmo tempo em áudio difícil) e mede exatidão no caso mais
fácil que existe. A taxa de erro que um aluno brasileiro falando num
microfone de notebook vai ver é **maior**, e este script não a conhece. Por
isso ``--record`` existe, e por isso o número sintético é reportado como piso,
nunca como expectativa.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from teacher_charles.config import Settings  # noqa: E402
from teacher_charles.llm.base import Message  # noqa: E402
from teacher_charles.voice.base import Speech  # noqa: E402
from teacher_charles.voice.segment import SentenceAssembler, split_sentences  # noqa: E402

# Prosa de professor de verdade: a resposta que a persona pede, com a correção
# entre aspas e o markdown que ela usa. Medir numa frase de laboratório
# ("the quick brown fox") mediria outra coisa.
REPLY = (
    'You said "I have 30 years." In English, age uses *to be*: "I am 30 years old." '
    "That is one of the most common mistakes Brazilian learners make, and it is an easy "
    "one to fix. So tell me — what did you do on your birthday?"
)

# Falas de aluno de três tamanhos. A primeira é o turno curto típico; a última
# é um aluno que se empolgou.
UTTERANCES = (
    "I work in marketing here in Brazil.",
    "Yesterday I go to a meeting with my boss and we discuss the new project.",
    "I pretend to travel to Canada next year because my sister lives in Toronto, "
    "and I want to practice my English before I go there.",
)

_NORMALISE = re.compile(r"[^\w\s]")


def human_ms(ms: float) -> str:
    return f"{ms / 1000:.2f}s"


def normalise(text: str) -> list[str]:
    return _NORMALISE.sub(" ", text.lower()).split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER por distância de edição em palavras. 0,0 é transcrição exata."""
    ref, hyp = normalise(reference), normalise(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ref_word != hyp_word),
                )
            )
        previous = current
    return previous[-1] / len(ref)


def row(*cells: str) -> str:
    widths = (34, 11, 11, 9, 10)
    return "  ".join(cell.ljust(w) for cell, w in zip(cells, widths, strict=False)).rstrip()


# --- TTS -------------------------------------------------------------------- #


async def measure_tts(settings: Settings, models: list[str], threads: list[int]) -> int:
    """Carga e fator de tempo real do Kokoro, sentença por sentença."""
    from teacher_charles.voice.kokoro import KokoroTTS

    sentences = split_sentences(REPLY)
    print(f"\n TTS — Kokoro-82M na CPU, voz {settings.tts_voice}")
    print(f" Resposta de professor com {len(sentences)} sentenças:\n")
    for index, sentence in enumerate(sentences, start=1):
        print(f"   {index}. {sentence}")

    for model in models:
        path = settings.resolved_voice_dir / model
        if not path.is_file():
            print(f"\n  {model}: ausente em {path}")
            print("    Rode `uv run python scripts/fetch_voice_models.py --all`")
            continue

        for thread_count in threads:
            label = f"{model}  {thread_count or 'auto'} threads"
            engine = KokoroTTS(
                model_path=path,
                voices_path=settings.tts_voices_path,
                voice=settings.tts_voice,
                speed=settings.tts_speed,
                cpu_threads=thread_count,
            )

            started = time.monotonic()
            await engine.warm_up()
            load_s = time.monotonic() - started

            print(f"\n  {label}  (carga + aquecimento: {load_s:.2f}s)")
            print("  " + row("sentença", "síntese", "áudio", "RTF", "chars"))

            results: list[Speech] = []
            for sentence in sentences:
                speech = await engine.speak(sentence)
                results.append(speech)
                rtf = speech.real_time_factor
                print(
                    "  "
                    + row(
                        sentence[:32] + ("…" if len(sentence) > 32 else ""),
                        human_ms(speech.elapsed_ms),
                        f"{speech.duration_s:.2f}s",
                        f"{rtf:.2f}" if rtf else "—",
                        str(len(sentence)),
                    )
                )

            first = results[0]
            rtfs = [s.real_time_factor for s in results if s.real_time_factor]
            print(
                f"    primeira sentença: {human_ms(first.elapsed_ms)} para "
                f"{first.duration_s:.2f}s de áudio"
            )
            print(
                f"    RTF mediano: {statistics.median(rtfs):.2f}  "
                f"pior: {max(rtfs):.2f}  "
                f"total de áudio: {sum(s.duration_s for s in results):.1f}s"
            )
            await engine.aclose()

    print(
        "\n  Leitura: RTF < 1,0 significa que a síntese da sentença seguinte cabe\n"
        "  dentro da reprodução da atual — a voz não engasga. É o requisito\n"
        "  do TTS por sentença.\n"
    )
    return 0


# --- STT -------------------------------------------------------------------- #


async def synthesize_probes(settings: Settings) -> list[tuple[str, bytes]]:
    """Gera o áudio das falas de aluno com o Kokoro. Ver a ressalva no topo."""
    from teacher_charles.voice.kokoro import KokoroTTS

    engine = KokoroTTS(
        model_path=settings.tts_model_path,
        voices_path=settings.tts_voices_path,
        voice=settings.tts_voice,
        speed=settings.tts_speed,
    )
    probes = []
    try:
        for text in UTTERANCES:
            # Uma chamada por fala inteira: aqui o Kokoro é instrumento de
            # medição, não o caminho do produto.
            speech = await engine.speak(text)
            probes.append((text, speech.as_wav()))
    finally:
        await engine.aclose()
    return probes


def record(seconds: int) -> bytes:
    """Grava do microfone desta máquina com o ``arecord`` do alsa-utils."""
    print(f"\n  Gravando {seconds}s — fale agora.", flush=True)
    try:
        done = subprocess.run(
            ["arecord", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1", "-d", str(seconds)],
            capture_output=True,
            check=True,
            timeout=seconds + 15,
        )
    except FileNotFoundError:
        print("  `arecord` não existe. Instale alsa-utils, ou use --audio arquivo.wav")
        raise SystemExit(1) from None
    except subprocess.CalledProcessError as exc:
        print(f"  a gravação falhou: {exc.stderr.decode(errors='replace')[:200]}")
        raise SystemExit(1) from exc
    print(f"  pronto, {len(done.stdout) / 1024:.0f} KB")
    return done.stdout


async def measure_stt(
    settings: Settings,
    models: list[str],
    probes: list[tuple[str, bytes]],
    synthetic: bool,
) -> int:
    """Carga e latência do Whisper sobre cada fala."""
    from teacher_charles.voice.whisper import WhisperSTT

    print(f"\n STT — faster-whisper em {settings.stt_compute_type}, feixe {settings.stt_beam_size}")
    if synthetic:
        print(" Áudio sintético (Kokoro): mede latência de verdade, exatidão no caso fácil.")

    for model in models:
        engine = WhisperSTT(
            model=model,
            compute_type=settings.stt_compute_type,
            beam_size=settings.stt_beam_size,
            no_speech_threshold=settings.stt_no_speech_threshold,
            cpu_threads=settings.stt_cpu_threads,
        )

        started = time.monotonic()
        await engine.warm_up()
        load_s = time.monotonic() - started
        if not engine.ready:
            print(f"\n  {model}: não carregou — {engine.last_error}")
            continue

        print(f"\n  {model}  (carga: {load_s:.2f}s)")
        print("  " + row("fala", "transcrição", "áudio", "RTF", "WER"))

        wers = []
        for reference, audio in probes:
            result = await engine.transcribe(audio)
            wer = word_error_rate(reference, result.text)
            wers.append(wer)
            rtf = result.real_time_factor
            print(
                "  "
                + row(
                    reference[:32] + ("…" if len(reference) > 32 else ""),
                    human_ms(result.elapsed_ms),
                    f"{result.duration_s:.2f}s",
                    f"{rtf:.2f}" if rtf else "—",
                    f"{wer:.0%}",
                )
            )
            if wer > 0:
                print(f"      ouviu: {result.text}")

        print(f"    WER médio: {statistics.mean(wers):.0%}" if wers else "    sem falas")
        await engine.aclose()

    print(
        "\n  Leitura: o STT roda uma vez por turno, depois de o aluno largar o botão,\n"
        "  e o aluno está parado esperando. É latência pura — não há streaming\n"
        "  para esconder nada aqui.\n"
    )
    return 0


# --- orçamento do turno ----------------------------------------------------- #


async def measure_budget(settings: Settings) -> int:
    """STT + primeira sentença do professor + TTS dela, contra a meta de 1-2s.

    Passa pelo ``TurnRunner`` de verdade, e não por uma chamada direta ao
    provider, porque o que o aluno espera é o turno **do produto**: persona,
    ajuste de nível, prompt de modo, dica curada e histórico, com o prefixo
    estável aquecido como a subida aquece. Um atalho aqui mediria um turno que
    não existe — foi exatamente o erro que a E7 encontrou no aquecimento
    (plano §5.5).
    """
    import tempfile

    from teacher_charles.llm import build_provider, default_options
    from teacher_charles.prompts import PromptLibrary
    from teacher_charles.store.db import init_db
    from teacher_charles.store.repo import SessionRepo, StudentRepo
    from teacher_charles.teaching import Analyzer, Chunk, TurnRunner
    from teacher_charles.teaching.contrastive import load_base
    from teacher_charles.voice.kokoro import KokoroTTS
    from teacher_charles.voice.whisper import WhisperSTT

    prompts = PromptLibrary(settings.prompts_dir)
    provider = build_provider(settings)
    stt = WhisperSTT(
        model=settings.stt_model,
        compute_type=settings.stt_compute_type,
        beam_size=settings.stt_beam_size,
        cpu_threads=settings.stt_cpu_threads,
    )
    tts = KokoroTTS(
        model_path=settings.tts_model_path,
        voices_path=settings.tts_voices_path,
        voice=settings.tts_voice,
        speed=settings.tts_speed,
        cpu_threads=settings.tts_cpu_threads,
    )

    # Banco descartável: a medição não deveria sujar o progresso de ninguém.
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "measure.db")
        student = StudentRepo(conn).get_or_create("measure-voice")
        session_id = SessionRepo(conn).current_or_start(student.id)
        # O analisador entra, mas a análise **não** está no caminho medido: ela
        # roda depois do último pedaço (plano §5.2). O que ele traz para cá é a
        # dica curada, que vai no prompt da resposta e custa tempo de verdade —
        # medido na E7 em 1,7s (plano §5.5). Tirá-lo mediria um turno mais
        # rápido que o do produto.
        analyzer = Analyzer(
            provider=provider,
            prompts=prompts,
            base=load_base(settings.data_dir / "contrastive"),
            require_curated_rule=settings.require_curated_rule,
        )
        runner = TurnRunner(
            provider=provider,
            prompts=prompts,
            conn=conn,
            options=default_options(settings),
            analyzer=analyzer,
            analysis_after_reply=settings.analysis_after_reply,
        )

        print("\n Orçamento do turno de voz — o aluno falou, quando ele ouve a resposta?")
        print(
            f" LLM: {settings.llm_model} · STT: {settings.stt_model} · TTS: {settings.tts_model}\n"
        )

        try:
            print("  aquecendo os três, com o prefixo que o turno manda...", end="", flush=True)
            await tts.warm_up()
            await stt.warm_up()
            await provider.complete(
                [
                    *runner.stable_prefix(student.cefr_level),
                    Message(role="user", content="Hello."),
                ],
                default_options(settings, temperature=0.0, max_tokens=1, think=False),
            )
            print(" ok\n")

            # A fala do aluno, gravada como o navegador gravaria.
            spoken = await tts.speak(UTTERANCES[1])
            audio = spoken.as_wav()

            turn_started = time.monotonic()
            transcript = await stt.transcribe(audio)
            after_stt = time.monotonic()

            # Primeira sentença do professor: é ela que fecha a espera do aluno.
            assembler = SentenceAssembler()
            first_sentence: str | None = None
            first_token_at: float | None = None
            async for event in runner.run(session_id, transcript.text, student):
                if not isinstance(event, Chunk):
                    continue
                if first_token_at is None and event.text.strip():
                    first_token_at = time.monotonic()
                ready = assembler.push(event.text)
                if ready:
                    first_sentence = ready[0]
                    break
            if first_sentence is None:
                first_sentence = assembler.flush() or ""
            after_llm = time.monotonic()

            speech = await tts.speak(first_sentence)
            after_tts = time.monotonic()

            stt_s = after_stt - turn_started
            ttft_s = (first_token_at or after_stt) - after_stt
            llm_s = after_llm - after_stt
            tts_s = after_tts - after_llm
            total_s = after_tts - turn_started

            print(f'  o aluno disse:  "{UTTERANCES[1]}"')
            print(f'  o STT ouviu:    "{transcript.text}"')
            print(f'  o professor:    "{first_sentence}"\n')
            print(f"  STT ({transcript.duration_s:.1f}s de áudio)      {stt_s:6.2f}s")
            print(f"  LLM até o 1º token              {ttft_s:6.2f}s")
            print(f"  LLM até fechar a 1ª sentença    {llm_s:6.2f}s")
            print(f"  TTS da 1ª sentença              {tts_s:6.2f}s")
            print(f"  {'-' * 40}")
            print(f"  até o aluno OUVIR a 1ª palavra  {total_s:6.2f}s   (meta da Fase 1.5: 1-2s)")
            print(f"\n  Daí em diante o aluno ouve {speech.duration_s:.1f}s de fala, e a síntese")
            print(
                f"  da sentença seguinte corre dentro dela (RTF {speech.real_time_factor:.2f}).\n"
            )
        finally:
            await provider.aclose()
            await stt.aclose()
            await tts.aclose()
            conn.close()
    return 0


# --- main ------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tts", action="store_true", help="só o TTS")
    parser.add_argument("--stt", action="store_true", help="só o STT")
    parser.add_argument("--budget", action="store_true", help="o turno inteiro (precisa de Ollama)")
    parser.add_argument(
        "--tts-models",
        default="kokoro-v1.0.int8.onnx,kokoro-v1.0.onnx",
        help="modelos de TTS a comparar",
    )
    parser.add_argument(
        "--tts-threads",
        default="0",
        help="contagens de thread a comparar no TTS (0 = runtime decide)",
    )
    parser.add_argument(
        "--stt-models",
        default="tiny.en,base.en,small.en",
        help="modelos de STT a comparar",
    )
    parser.add_argument("--audio", type=Path, default=None, help="arquivo de áudio para o STT")
    parser.add_argument("--say", default="", help="o que o --audio/--record diz, para calcular WER")
    parser.add_argument("--record", type=int, default=0, help="grava N segundos do microfone")
    args = parser.parse_args()

    settings = Settings()
    nothing_asked = not (args.tts or args.stt or args.budget)

    async def run() -> int:
        if args.budget:
            return await measure_budget(settings)

        code = 0
        if args.tts or nothing_asked:
            code |= await measure_tts(
                settings,
                [m.strip() for m in args.tts_models.split(",") if m.strip()],
                [int(t) for t in args.tts_threads.split(",") if t.strip()],
            )

        if args.stt or nothing_asked:
            models = [m.strip() for m in args.stt_models.split(",") if m.strip()]
            if args.record:
                probes = [(args.say, record(args.record))]
                synthetic = False
            elif args.audio:
                probes = [(args.say, args.audio.read_bytes())]
                synthetic = False
            else:
                probes = await synthesize_probes(settings)
                synthetic = True
            if not synthetic and not args.say:
                print("\n  Sem --say não há referência, então o WER sai como 100%.")
            code |= await measure_stt(settings, models, probes, synthetic)
        return code

    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
