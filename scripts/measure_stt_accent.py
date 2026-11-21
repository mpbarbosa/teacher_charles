#!/usr/bin/env python3
"""Measures Whisper on a Brazilian voice, which is E14's first condition.

In a free turn the expected text the pronunciation evaluator aligns against
can only be Whisper's transcript (plan 2 §5, E14). A transcript that heard the
wrong word makes the evaluator correct a sentence the student never said — so
before anything is built on it, plan 1.5 §2.3's debt has to be paid: how often
does Whisper hear a Brazilian speaker's words right?

    uv run python scripts/measure_stt_accent.py --dir ../../../gravacoes
    uv run python scripts/measure_stt_accent.py --models tiny.en,base.en,small.en
    uv run python scripts/measure_stt_accent.py --evaluate

`--evaluate` is the question E14 actually asks, one layer past the WER: with
the transcript as the expected text, does the evaluator say what it says with
the phrase that was really read? Each take goes through gate 1, the E11
recognizer and `Evaluator.evaluate_text` twice — against the phrase read and
against each model's transcript — and every rule where the two disagree is
sorted into a **false correction** (the transcript makes a correction the
truth does not) or a **miss** (the truth makes one the transcript does not).
The first is the failure plan 2 §6 names; the second is silence, which plan
§3.3 accepts.

The takes are the ones `measure_phonemes.py --record` left on disk: every WAV
whose report (a `*.json` next to it, matched by **md5**, not by name — a name
is reused when a phrase is recorded again) says what was read in `dito`. That
is the reference, and it is a reference by construction: the speaker was
reading it. The WAV and the reports are personal data and are not versioned;
the numbers this prints are ours, and go to plan 2 §5.

Two passes come out separately, because they answer different questions:

- **correct** — the phrase read as well as the speaker can. Every word Whisper
  gets wrong here is a word the evaluator would align against and the student
  never said: the false-correction risk.
- **deliberate** (`-errado`) — the curated error said on purpose. Here a
  transcript that **keeps** the intended word is what the evaluator needs (the
  error is then measured against the right phonemes); one that writes the
  mispronounced word down hides the error, which is a miss and not a false
  correction.

Each take goes through the app's own `WhisperSTT` with the options in
`Settings` — VAD, language, beam — so the number is the product's, not a
laboratory's.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from measure_voice import normalise, word_error_rate  # noqa: E402

from teacher_charles.config import Settings  # noqa: E402
from teacher_charles.voice.whisper import WhisperSTT  # noqa: E402

# Takes whose report does not say what the audio says, by md5 prefix, as
# `pronunciation.yaml` names its takes. Each was checked against the report's
# own phoneme reading, which is independent of Whisper: an exclusion here is a
# label that was wrong, never a transcription that was inconvenient.
MISLABELLED = {
    "93f54488": "not the phrase: the recognizer heard PER 3.08 of something else",
    "5bdb9776": "labelled 'I'm ready to go'; the recognizer heard the red car is ready, "
    "symbol for symbol",
    "b7f563f0": "labelled 'the car is ready'; the recognizer heard the r(e)d car is ready",
}


@dataclass(frozen=True)
class Take:
    path: Path
    said: str
    deliberate: bool
    warnings: int
    md5: str


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def find_takes(directory: Path) -> tuple[list[Take], list[Path]]:
    """Every WAV with a report of the same bytes; the rest come back apart."""
    reports: dict[str, dict] = {}
    for report in directory.glob("*.json"):
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("md5") and data.get("dito"):
            reports[data["md5"]] = data

    takes, orphans = [], []
    for wav in sorted(directory.glob("*.wav")):
        # The `.bruto.wav` is the uncut original of a take; the cut one is
        # what the report measured.
        if wav.name.endswith(".bruto.wav"):
            continue
        md5 = md5_of(wav)
        data = reports.get(md5)
        if data is None:
            orphans.append(wav)
            continue
        takes.append(
            Take(
                path=wav,
                said=data["dito"],
                deliberate="-errado" in wav.stem,
                warnings=len(data.get("avisos") or []),
                md5=md5,
            )
        )
    return takes, orphans


def word_diff(reference: str, hypothesis: str) -> str:
    """The words that differ, as `ref→hyp`, in reading order."""
    import difflib

    ref, hyp = normalise(reference), normalise(hypothesis)
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ref, b=hyp).get_opcodes():
        if op == "equal":
            continue
        out.append(f"{' '.join(ref[i1:i2]) or '∅'}→{' '.join(hyp[j1:j2]) or '∅'}")
    return ", ".join(out)


async def transcribe_all(model: str, takes: list[Take], settings: Settings) -> list[str]:
    stt = WhisperSTT(
        model=model,
        compute_type=settings.stt_compute_type,
        beam_size=settings.stt_beam_size,
        no_speech_threshold=settings.stt_no_speech_threshold,
        cpu_threads=settings.stt_cpu_threads,
    )
    out = []
    for take in takes:
        result = await stt.transcribe(take.path.read_bytes())
        out.append(result.text)
    return out


def report(model: str, takes: list[Take], texts: list[str]) -> None:
    print(f"\n## {model}\n")
    print("| pass | take | read | heard | WER | warned |")
    print("|---|---|---|---|---|---|")
    for take, text in zip(takes, texts, strict=True):
        wer = word_error_rate(take.said, text)
        diff = word_diff(take.said, text)
        print(
            f"| {'deliberate' if take.deliberate else 'correct'} | {take.path.stem} "
            f"| {take.said} | {text} | {wer:.0%}{' (' + diff + ')' if diff else ''} "
            f"| {take.warnings or ''} |"
        )

    for label, wanted in (("correct", False), ("deliberate", True)):
        pairs = [(t, x) for t, x in zip(takes, texts, strict=True) if t.deliberate is wanted]
        if not pairs:
            continue
        wers = [word_error_rate(t.said, x) for t, x in pairs]
        exact = sum(1 for w in wers if w == 0)
        words = sum(len(normalise(t.said)) for t, _ in pairs)
        wrong = sum(
            round(w * len(normalise(t.said))) for (t, _), w in zip(pairs, wers, strict=True)
        )
        print(
            f"\n{label}: {exact} of {len(pairs)} takes transcribed exactly; "
            f"{wrong} word errors in {words} words read "
            f"(pooled WER {wrong / words:.1%}, median per take {statistics.median(wers):.0%})"
        )


def pcm_of(path: Path) -> bytes:
    with wave.open(str(path)) as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            sys.exit(f"{path}: not 16 kHz mono 16-bit, which is what the app hears")
        return wav.readframes(wav.getnframes())


@dataclass(frozen=True)
class Heard:
    phonemes: tuple[str, ...]
    usable: bool


async def hear_all(takes: list[Take]) -> tuple[list[Heard], frozenset[str]]:
    """Gate 1 and the E11 recognizer, as `judge_take` runs them."""
    import numpy as np

    from teacher_charles.voice.base import STT_SAMPLE_RATE, Speech
    from teacher_charles.voice.phonemes import Wav2Vec2Phonemes
    from teacher_charles.voice.signal import diagnose, samples_from_pcm

    ear = Wav2Vec2Phonemes()
    out = []
    for take in takes:
        pcm = pcm_of(take.path)
        capture = diagnose(samples_from_pcm(pcm, np), np)
        heard = await ear.recognize(Speech(pcm=pcm, sample_rate=STT_SAMPLE_RATE, elapsed_ms=0.0))
        out.append(Heard(tuple(heard.phonemes), capture.usable))
    return out, await ear.vocabulary()


@dataclass(frozen=True)
class Verdict:
    kind: str  # correction | no-error | silence | not judged
    words: frozenset[str] = frozenset()
    gate: int | None = None

    def __str__(self) -> str:
        if self.kind == "correction":
            return f"correction ({' '.join(sorted(self.words))})"
        return f"silence g{self.gate}" if self.kind == "silence" else self.kind


NOT_JUDGED = Verdict("not judged")


def verdicts(evaluator, text: str, heard: Heard, vocabulary, gate_1: bool) -> dict[str, Verdict]:
    """Rule → its verdict over ``text``, for one take."""
    return {
        j.rule_id: Verdict(j.verdict, frozenset(f.word for f in j.findings), j.gate)
        for j in evaluator.evaluate_text(
            text, heard.phonemes, vocabulary, signal_usable=heard.usable or not gate_1
        )
    }


def classify(truth: Verdict, via: Verdict) -> str:
    """What reading the transcript instead of the phrase did to one rule's verdict.

    A correction is false when it names a word the truth does not: that is a
    word the student said right, or never said. Fewer words than the truth
    is a partial miss — the student still reads about a real error.
    """
    if truth == via:
        return "agree"
    if via.kind == "correction":
        if truth.kind != "correction" or via.words - truth.words:
            return "FALSE CORRECTION"
        return "partial"
    if truth.kind == "correction":
        return "miss"
    return "no correction either way"


def evaluate(
    models: list[str], takes: list[Take], texts: dict[str, list[str]], settings: Settings
) -> None:
    from teacher_charles.teaching.phonetics import Phonetics
    from teacher_charles.teaching.pronunciation import Evaluator, load_base

    evaluator = Evaluator(
        load_base(settings.data_dir / "contrastive" / "pronunciation.yaml"),
        Phonetics(with_stress=False).transcribe,
    )
    heard, vocabulary = asyncio.run(hear_all(takes))
    for gate_1 in (True, False):
        state = "on (the product)" if gate_1 else "off"
        print(f"\n## The evaluator over the transcript, gate 1 {state}\n")
        for model in models:
            tally: dict[str, int] = {}
            lines = []
            for take, h, text in zip(takes, heard, texts[model], strict=True):
                truth = verdicts(evaluator, take.said, h, vocabulary, gate_1)
                via = verdicts(evaluator, text, h, vocabulary, gate_1)
                for rule in sorted(truth.keys() | via.keys()):
                    t, v = truth.get(rule, NOT_JUDGED), via.get(rule, NOT_JUDGED)
                    kind = classify(t, v)
                    tally[kind] = tally.get(kind, 0) + 1
                    if t.kind == "correction":
                        tally["corrections on the truth"] = (
                            tally.get("corrections on the truth", 0) + 1
                        )
                    if kind not in ("agree", "no correction either way"):
                        lines.append(
                            f"  {take.path.stem}: {kind}: {rule.removeprefix('pron.')} "
                            f"truth={t} transcript={v}"
                        )
            print(f"{model}: " + ", ".join(f"{k} {n}" for k, n in sorted(tally.items())))
            for line in lines:
                print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dir", type=Path, default=REPO_ROOT / "gravacoes")
    parser.add_argument("--models", default="tiny.en,base.en,small.en")
    parser.add_argument(
        "--evaluate", action="store_true", help="run the E11 evaluator over each transcript"
    )
    args = parser.parse_args()

    found, orphans = find_takes(args.dir)
    takes = [t for t in found if t.md5[:8] not in MISLABELLED]
    if not takes:
        sys.exit(f"no take with a report in {args.dir}")
    print(
        f"{len(takes)} takes with a report in {args.dir} "
        f"({sum(t.deliberate for t in takes)} deliberate); "
        f"{len(orphans)} WAV with no report of the same bytes, left out"
    )
    for take in found:
        if take.md5[:8] in MISLABELLED:
            print(f"  left out {take.path.stem} ({take.md5[:8]}): {MISLABELLED[take.md5[:8]]}")

    settings = Settings(_env_file=None)
    models = [m.strip() for m in args.models.split(",")]
    texts = {}
    for model in models:
        texts[model] = asyncio.run(transcribe_all(model, takes, settings))
        report(model, takes, texts[model])
    if args.evaluate:
        evaluate(models, takes, texts, settings)


if __name__ == "__main__":
    main()
