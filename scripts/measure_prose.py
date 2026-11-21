#!/usr/bin/env python3
"""Measures the teacher's prose, so two models can be compared on it.

The 4B default rests on decision 1.3's bet that a 4B writes a better teacher
than a 3B, and none of the house scripts measured that: `measure_analysis.py`
reads the structured correction and `measure_latency.py` the clock. This one
reads what the student reads — Charles's reply — on a fixed battery of student
turns, through the product's own prompt (persona, app capabilities, level,
mode, drill data, curated hint inside the student's message).

    uv run python scripts/measure_prose.py run --model qwen2.5:3b-instruct-q4_K_M
    uv run python scripts/measure_prose.py run --model qwen3:4b-instruct
        Every probe, `--runs` times each (default 3), written as JSON lines to
        `--out` (default: `prose-<model>.jsonl` in the working directory).

    uv run python scripts/measure_prose.py sheet A.jsonl B.jsonl
        Pairs the two runs probe by probe, shuffles which model is "A", and
        writes a blind judging sheet (`--sheet`) and the key (`--key`)
        separately — whoever judges reads the sheet and never the key.

    uv run python scripts/measure_prose.py score --key KEY verdicts.json ...
        Unblinds one or more judges' verdicts against the key and prints
        the tally per model, per dimension and per probe kind.

## What the code counts, and what it leaves to a judge

Counted, because a counter cannot be talked into it: length against the
persona's "two or three sentences", whether the reply ends on a question (the
conversation goes on), Portuguese in the reply (decision 1.1), formatting the
persona never asked for (a `>` block, a list, a heading), IPA typed into the
prose (the IPA line under the bubble does that), and — on a probe with a known
error — whether the right form appears in the reply at all.

Judged, because only reading tells: whether a correction is *right* and does
not keep the student's error inside the model sentence (the 7B did, §5.4 of
the Phase 1 plan), whether a clean sentence got an invented correction,
whether a grammar or sound fact is false, and whether the reply sounds like a
teacher talking to a person. The sheet is blind and the order is shuffled per
pair, because a judge who knows which model wrote what is measuring the label.

## What this does not measure

The clock. Prose does not depend on it, which is why this can run on a busy
machine where `measure_latency.py` cannot — except that a turn which times out
is a reply the student never read, so a timeout is recorded, not retried.

Needs Ollama up with each model already pulled.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from teacher_charles.config import Settings  # noqa: E402
from teacher_charles.llm import build_provider, default_options  # noqa: E402
from teacher_charles.prompts import PromptLibrary  # noqa: E402
from teacher_charles.store.db import init_db  # noqa: E402
from teacher_charles.store.repo import (  # noqa: E402
    CorrectionRepo,
    SessionRepo,
    StudentRepo,
    TurnRepo,
)
from teacher_charles.teaching import Analysis, Analyzer, Chunk, Done, Failed, TurnRunner  # noqa: E402
from teacher_charles.teaching.contrastive import load_base  # noqa: E402
from teacher_charles.teaching.ipa_symbols import load_symbols  # noqa: E402
from teacher_charles.teaching.modes import DrillPlanner  # noqa: E402
from teacher_charles.teaching.phonetics import build_phonetics  # noqa: E402
from teacher_charles.teaching.sound_facts import build_sound_facts  # noqa: E402

# --------------------------------------------------------------------------- #
# The battery
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Probe:
    """One student turn, and what a good teacher does with it.

    ``good`` is written for the judge, in the persona's own terms — the rubric
    is the spec the model was given, not a taste brought from outside.
    ``expect`` is the counter's half: any of these substrings in the reply
    (case-insensitive) means the right form reached the student at all.
    """

    id: str
    kind: str
    level: str
    text: str
    good: str
    mode: str = "conversation"
    history: tuple[tuple[str, str], ...] = ()
    expect: tuple[str, ...] = ()


PROBES: tuple[Probe, ...] = (
    # --- clean English: nothing to fix, keep talking ----------------------- #
    Probe(
        "clean-a2-football",
        "clean",
        "A2",
        "I like football. I play on Sunday with my friends.",
        "No correction. Very simple English for an A2, short reply, a natural question back.",
    ),
    Probe(
        "clean-b1-beach",
        "clean",
        "B1",
        "Yesterday I went to the beach with my family. The water was cold but we had fun.",
        "No correction. Reacts to the beach story and asks a follow-up question.",
    ),
    Probe(
        "clean-b2-jobs",
        "clean",
        "B2",
        "I've been thinking about changing jobs, but I'm not sure it's the right moment.",
        "No correction. Engages with the dilemma (reasons, feelings) and asks about it.",
    ),
    Probe(
        "clean-c1-remote",
        "clean",
        "C1",
        "Honestly, I find remote work isolating, even though it saves me two hours of "
        "commuting every day.",
        "No correction. Richer C1 register, engages with the trade-off, asks an opinion question.",
    ),
    # --- an error the curated base matches: the hint rides along ----------- #
    Probe(
        "curated-b1-age",
        "curated",
        "B1",
        "I have 30 years and I work in a bank.",
        "Corrects the age calque to 'I am 30 (years old)', one short reason, then asks about "
        "the bank job.",
        expect=("i am 30", "i'm 30"),
    ),
    Probe(
        "curated-b1-bus",
        "curated",
        "B1",
        "I lost the bus this morning, so I arrived late at work.",
        "Corrects 'lost the bus' to 'missed the bus', briefly, and continues.",
        expect=("missed the bus", "miss the bus", "missed it"),
    ),
    Probe(
        "curated-b2-realize",
        "curated",
        "B2",
        "We realized a meeting with the clients yesterday.",
        "Corrects the false friend: 'held / had a meeting' (realize = perceive), then asks "
        "about the meeting.",
        expect=("held a meeting", "had a meeting", "hold a meeting", "have a meeting"),
    ),
    Probe(
        "curated-b1-since",
        "curated",
        "B1",
        "I work here since 2019 and I like my team.",
        "Corrects to 'I have worked / have been working here since 2019', briefly, continues.",
        expect=("have worked", "have been working", "'ve worked", "'ve been working"),
    ),
    Probe(
        "curated-b1-two-errors",
        "curated",
        "B1",
        "I have 35 years and I am living here since 2010.",
        "Two errors: picks one (or at most two) that matters, corrects it correctly — the model "
        "sentence must not keep the other error inside it — and keeps talking.",
        expect=("i am 35", "i'm 35", "have lived", "have been living", "'ve lived"),
    ),
    # --- an error the base does not cover: no hint, the model is alone ----- #
    Probe(
        "uncurated-b1-sympathetic",
        "uncurated",
        "B1",
        "My boss is very sympathetic, she always helps me.",
        "False friend not in the base ('simpático' = nice/friendly). A good reply suggests "
        "'nice' or 'friendly' without inventing a wrong rule; letting it pass is acceptable, "
        "a false explanation is not.",
        expect=("nice", "friendly", "kind"),
    ),
    Probe(
        "uncurated-c1-actually",
        "uncurated",
        "C1",
        "In my opinion, the government should invest more in education, because actually "
        "the schools are in a terrible state.",
        "'actually' here means 'atualmente' (currently/these days) — a false friend the base "
        "covers but the matcher did not raise. A good reply engages with the opinion and may "
        "suggest 'currently/these days'; no invented rule.",
        expect=("currently", "these days", "at the moment", "nowadays", "right now"),
    ),
    # --- not about an error ------------------------------------------------- #
    Probe(
        "portuguese-b1",
        "portuguese",
        "B1",
        "Desculpa, não entendi. Pode explicar de novo?",
        "Answers only in simple English, reassures, invites the student to try again in "
        "English. No Portuguese at all.",
    ),
    # Added after the first measurement, when the 3B became the default: one
    # Portuguese turn was enough to see the failure (3 of 3), not to see a fix.
    Probe(
        "portuguese-b1-news",
        "portuguese",
        "B1",
        "Hoje eu fui ao médico e estou muito cansado.",
        "Answers only in simple English: sympathy for the tiring day, and an invitation to "
        "say it in English. No Portuguese at all.",
    ),
    Probe(
        "portuguese-b2-how-say",
        "portuguese",
        "B2",
        "Como se diz 'saudade' em inglês?",
        "Answers only in English: there is no single word; 'miss' / 'longing' / 'I miss you' "
        "come close. May quote the word 'saudade' itself, nothing else in Portuguese.",
    ),
    Probe(
        "mixed-b1-feira",
        "mixed",
        "B1",
        "Yesterday I went to the feira with my mother and we bought many vegetables.",
        "English with one Portuguese word: a natural reply in English, may offer 'street "
        "market' / 'farmers' market' for 'feira'. Not treated as a Portuguese turn.",
        expect=("market",),
    ),
    Probe(
        "grammar-question-b1",
        "grammar-question",
        "B1",
        "What is the difference between 'make' and 'do'?",
        "A short, accurate answer (make = create/produce; do = activities, tasks, work) with an "
        "example or two. No invented rule, no long list.",
    ),
    Probe(
        "sound-question-b2",
        "sound-question",
        "B2",
        "Is the 'th' in 'think' the same sound as the 'th' in 'this'?",
        "True answer: no — 'think' is voiceless, 'this' is voiced (vocal cords vibrate). Any "
        "false sound fact is a fault; saying both words and not inventing is acceptable.",
    ),
    # Added with the sound note: the first sound question was answered wrongly
    # six times out of six by both models, and one pair cannot show a fix.
    Probe(
        "sound-question-b1-ship",
        "sound-question",
        "B1",
        "What is the difference between 'ship' and 'sheep'?",
        "True answer: the vowel — short in 'ship' (as in 'sit'), long in 'sheep' (as in "
        "'see'); the first and last sounds are the same. Any false sound fact is a fault.",
    ),
    Probe(
        "sound-question-b1-rose",
        "sound-question",
        "B1",
        "Is the 's' in 'rose' the same sound as the 's' in 'rice'?",
        "True answer: no — 'rose' ends in a /z/ sound (voiced, like 'zoo'), 'rice' in an /s/ "
        "sound (voiceless, like 'see'). Any false sound fact is a fault.",
    ),
    Probe(
        "sound-question-b2-thought",
        "sound-question",
        "B2",
        "Do 'thought' and 'taught' sound the same?",
        "True answer (American English): no — they share the vowel and the end, but 'thought' "
        "starts with the 'th' of 'think' and 'taught' with a plain 't'. Any false sound fact "
        "is a fault.",
    ),
    Probe(
        "capability-b1",
        "capability",
        "B1",
        "Can you listen to my pronunciation and tell me if it is good?",
        "Honest: this is a text chat, there is no microphone, he cannot hear or score "
        "pronunciation. No workaround (no 'record a file'), and keeps the conversation going.",
    ),
    Probe(
        "feelings-b1",
        "feelings",
        "B1",
        "I failed my English exam today. I feel so stupid.",
        "Empathy first, encouragement, no correction (the English is fine), a gentle question.",
    ),
    # --- the student gets right what they got wrong a turn ago -------------- #
    Probe(
        "praise-b1-age",
        "praise",
        "B1",
        "I am 30 years old now, and on my birthday I went to a restaurant with my wife.",
        "The student just got the age structure right after being corrected on it: say so "
        "briefly (the persona: 'it matters more than the corrections'), then ask about the "
        "restaurant.",
        history=(
            ("student", "I have 30 years. My birthday was last week."),
            (
                "teacher",
                'You said "I have 30 years." In English, age uses *to be*: "I am 30 years old." '
                "Happy belated birthday! What did you do to celebrate?",
            ),
        ),
    ),
    # --- the other two modes ------------------------------------------------ #
    Probe(
        "correction-mode-b1",
        "correction-mode",
        "B1",
        "Yesterday I go to the supermarket and I buyed many fruits because I want to make a salad.",
        "Correction mode: corrects 'go' → 'went' and 'buyed' → 'bought' (and may mention "
        "'many fruits' → 'a lot of fruit'), each with a short reason, ranked; notes what was "
        "right; ends with one short question. Every corrected form must itself be right.",
        mode="correction",
        expect=("went", "bought"),
    ),
    Probe(
        "drill-b1-age",
        "drill",
        "B1",
        "OK, I'm ready. Let's practise.",
        "Drill on the age calque: ONE short item in English (fix / complete / answer), ideally "
        "using the student's own sentences ('I have 35 years'), no Portuguese to translate, no "
        "list of exercises, no lecture up front.",
        mode="drill",
    ),
)

# The drill needs an error log to choose from — the student's own words, as
# `measure_modes.py` plants them.
SEEDED_ERRORS = (
    ("I have 35 years", "I am 35 years old", "calque.age-have"),
    ("I have 40 years", "I am 40 years old", "calque.age-have"),
)


# --------------------------------------------------------------------------- #
# What the code counts
# --------------------------------------------------------------------------- #

_SENTENCE_END = re.compile(r"[.!?]+(?:[\"')\]]*)(?=\s|$)")
# Portuguese the student could not miss: accented letters English never uses,
# and function words that are not English words. Coarse on purpose — a hit is
# a reply to read, not a verdict.
_PORTUGUESE = re.compile(
    r"[ãõçâêô]|\b(?:você|não|obrigad[oa]|tudo bem|está|então|porque|muito|também|"
    r"pode|desculp[ae]|entendi|explicar|de novo)\b",
    re.IGNORECASE,
)
_BLOCKQUOTE = re.compile(r"^\s*>", re.MULTILINE)
_LIST = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
_HEADING = re.compile(r"^\s*#{1,6}\s", re.MULTILINE)
# IPA between slashes or brackets, with at least one symbol ASCII lacks.
_IPA = re.compile(r"[/\[][^/\]\n]*[ɪʊəɛæʌɔɑθðʃʒŋˈˌː][^/\]\n]*[/\]]")


_TRAILING = re.compile(r"[^\w?.!]+$")
# Nothing in the persona forbids an emoji, and nothing asks for one: counted
# because one model sprinkles them and the other never does.
_EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿]")


def sentences(reply: str) -> list[str]:
    parts = _SENTENCE_END.split(reply.replace("\n", " "))
    return [p.strip() for p in parts if re.search(r"[A-Za-z]", p)]


def count(reply: str, probe: Probe) -> dict[str, object]:
    lowered = reply.lower()
    said = sentences(reply)
    words = reply.split()
    # What trails the last word is not the end of the sentence: closing quotes,
    # markdown, and the emoji the 4B puts after its question ("…today? 😊").
    tail = _TRAILING.sub("", reply)
    return {
        "words": len(words),
        "sentences": len(said),
        "words_per_sentence": round(len(words) / max(len(said), 1), 1),
        # The persona: "two or three sentences is the right size for a reply".
        # Correction mode asks for more on purpose, so it is not held to it.
        "over_length": probe.mode != "correction" and len(said) > 4,
        "ends_on_question": tail.endswith("?"),
        "portuguese": bool(_PORTUGUESE.search(reply)),
        "blockquote": bool(_BLOCKQUOTE.search(reply)),
        "list": bool(_LIST.search(reply)),
        "heading": bool(_HEADING.search(reply)),
        "typed_ipa": bool(_IPA.search(reply)),
        "emoji": bool(_EMOJI.search(reply)),
        "right_form_shown": (
            any(form in lowered for form in probe.expect) if probe.expect else None
        ),
    }


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


class HintOnlyAnalyzer(Analyzer):
    """The product's analyzer without the analysis call — `measure_modes.py`'s.

    The curated hint in the reply's prompt comes from pattern matching and is
    free, and without it this would measure a product that does not exist. The
    analysis call costs a model call and does not change the prose.
    """

    async def analyze(self, student_text, cefr_level="B1", max_findings=None) -> Analysis:
        return Analysis(skipped=True)


@dataclass
class Record:
    model: str
    probe: str
    kind: str
    level: str
    mode: str
    run: int
    text: str
    reply: str
    failed: str = ""
    total_ms: float = 0.0
    first_chunk_ms: float = 0.0
    counts: dict[str, object] = field(default_factory=dict)


def seed_error_log(conn, student_id: int) -> None:
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
            explanation="curated",
            rule_id=rule_id,
        )
    sessions.end(session_id)


async def run_one(runner, conn, base, probe: Probe, run: int) -> tuple[str, str, Done | None]:
    """One probe on a fresh student, so no run leaks history into the next."""
    students = StudentRepo(conn)
    sessions = SessionRepo(conn)
    turns = TurnRepo(conn)
    student = students.get_or_create(f"{probe.id}-{run}", probe.level)

    rule = None
    if probe.mode == "drill":
        seed_error_log(conn, student.id)
        target = DrillPlanner(conn, base).pick(student.id)
        rule = target.rule_id if target is not None else None
    session_id = sessions.switch_mode(student.id, probe.mode, rule)
    for role, content in probe.history:
        turns.append(session_id, role, content)

    pieces: list[str] = []
    done: Done | None = None
    failed = ""
    async for event in runner.run(session_id, probe.text, student):
        if isinstance(event, Chunk):
            pieces.append(event.text)
        elif isinstance(event, Done):
            done = event
        elif isinstance(event, Failed):
            failed = event.message
    return "".join(pieces).strip(), failed, done


async def run(model: str, runs: int, out: Path, only: set[str]) -> int:
    settings = Settings(llm_model=model)
    prompts = PromptLibrary(settings.prompts_dir)
    base = load_base(settings.data_dir / "contrastive")
    provider = build_provider(settings)
    probes = [p for p in PROBES if not only or p.id in only]

    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "prose.db")
        phonetics = build_phonetics(settings)
        symbols = load_symbols(settings.data_dir / "phonetics" / "symbols.yaml")
        runner = TurnRunner(
            provider=provider,
            prompts=prompts,
            conn=conn,
            options=default_options(settings),
            analyzer=HintOnlyAnalyzer(provider=provider, prompts=prompts, base=base),
            planner=DrillPlanner(conn, base),
            # The default install: text only, the IPA line on, no Structure
            # button — the app-capabilities message the student's Charles gets.
            voice_enabled=False,
            phonetics=phonetics,
            syntax_enabled=False,
            sound_facts=build_sound_facts(phonetics, symbols),
        )
        print(f"model: {provider.model} · {len(probes)} probes × {runs} runs → {out}")
        try:
            with out.open("w", encoding="utf-8") as sink:
                for probe in probes:
                    for n in range(1, runs + 1):
                        started = time.perf_counter()
                        reply, failed, done = await run_one(runner, conn, base, probe, n)
                        record = Record(
                            model=provider.model,
                            probe=probe.id,
                            kind=probe.kind,
                            level=probe.level,
                            mode=probe.mode,
                            run=n,
                            text=probe.text,
                            reply=reply,
                            failed=failed,
                            total_ms=round(
                                done.total_ms if done else (time.perf_counter() - started) * 1e3
                            ),
                            first_chunk_ms=round(done.first_chunk_ms) if done else 0.0,
                            counts=count(reply, probe),
                        )
                        sink.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                        sink.flush()
                        flag = f" FAILED: {failed}" if failed else ""
                        print(
                            f"  {probe.id} #{n}: {record.counts['words']} words, "
                            f"{record.total_ms / 1000:.1f}s{flag}",
                            flush=True,
                        )
        finally:
            await provider.aclose()
            conn.close()
    return 0


# --------------------------------------------------------------------------- #
# sheet
# --------------------------------------------------------------------------- #

RUBRIC = """\
# Blind judging sheet — Charles's replies

Charles is an English teacher in a local app; the student is a Brazilian adult
learner. Each item below is ONE student turn and TWO replies to it, labelled A
and B. The labels are shuffled per item: A in one item has nothing to do with A
in the next. Judge each item on its own.

Charles's spec (his system prompt, summarised):
- Speak only English, even if the student writes Portuguese.
- A real conversation with a person, not a grammar checker. Two or three
  sentences per reply; no long paragraphs, no lists of rules.
- Correct at most one mistake that matters per turn (two is a lot), and only
  one that matters: the right version, why in one short sentence, then keep
  the conversation going. Let small slips pass. If the English is fine, do not
  look for something to fix.
- When the student gets right something they usually get wrong, say so.
- Do not invent grammar rules or facts about sounds. "We'd usually say it this
  way" is better than a confident wrong reason.
- It is a text chat: no microphone, he cannot hear or score pronunciation.
- Level: A2/B1 get short sentences and common words; C1 can get richer English.
- Correction mode: correct all the mistakes that matter, ranked, each with a
  short reason, say what was right, end with one short question.
- Drill mode: one short exercise at a time on one rule, in English, using the
  student's own sentences, no lecture up front.

For EACH reply (A and B), mark every FAULT that applies (a list, may be empty):
- "portuguese"          any Portuguese in the reply
- "invented_correction" corrects something that was not wrong
- "wrong_correction"    a "corrected" form that is still wrong, or keeps the
                        student's error inside the model sentence
- "missed_error"        a mistake that matters (named in "what a good reply
                        does") passed without any correction — not a fault in
                        conversation mode for an uncurated slip, use judgment
- "false_fact"          a false statement about grammar, meaning or sounds
- "capability_lie"      claims to hear/record/score audio, or offers a
                        workaround for it
- "too_long"            clearly longer than the spec asks for this mode
- "off_level"           vocabulary or sentences clearly wrong for the level
- "ignores_student"     does not engage with what the student actually said
- "unnatural"           stilted, robotic or broken English from the teacher
- "format"              block quotes, headings, bullet lists the spec did not ask for

Then, per item, say which reply is better on each dimension — "A", "B" or
"tie":
- "accuracy"   — is what he teaches true and right?
- "pedagogy"   — does he follow the spec: the right correction, briefly, the
                 right amount, at the right level, in the right mode?
- "conversation" — does it sound like a warm teacher talking to this person,
                 engaging with what they said and keeping it going?
- "overall"    — which reply would you rather a student read?

Use "tie" honestly when they are equally good or equally bad. Length alone is
not quality.

Answer with ONE JSON object and nothing else:

{"items": [{"item": "<item id>", "faults": {"A": [...], "B": [...]},
            "accuracy": "A|B|tie", "pedagogy": "A|B|tie",
            "conversation": "A|B|tie", "overall": "A|B|tie",
            "why": "<one sentence>"}, ...]}

One entry per item, every item.

"""


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sheet(first: Path, second: Path, sheet_path: Path, key_path: Path, seed: int) -> int:
    by_probe = {p.id: p for p in PROBES}
    left = {(r["probe"], r["run"]): r for r in load(first)}
    right = {(r["probe"], r["run"]): r for r in load(second)}
    shared = sorted(set(left) & set(right), key=lambda k: (list(by_probe).index(k[0]), k[1]))
    rng = random.Random(seed)
    # Items go out in a shuffled order too, so a judge cannot read a probe's
    # three runs side by side and learn which label "sounds like" which model.
    rng.shuffle(shared)

    lines = [RUBRIC]
    key: dict[str, dict[str, str]] = {}
    for number, (probe_id, run_no) in enumerate(shared, start=1):
        probe = by_probe[probe_id]
        pair = [left[(probe_id, run_no)], right[(probe_id, run_no)]]
        rng.shuffle(pair)
        item = f"item-{number:02d}"
        key[item] = {"probe": probe_id, "run": run_no, "A": pair[0]["model"], "B": pair[1]["model"]}
        lines.append(f"## {item}\n")
        lines.append(f"- level: {probe.level} · mode: {probe.mode}")
        for role, content in probe.history:
            lines.append(f"- earlier, {role}: {content}")
        lines.append(f"- student: {probe.text}")
        lines.append(f"- what a good reply does: {probe.good}\n")
        for label, record in zip("AB", pair, strict=True):
            body = record["reply"] or f"(no reply — the turn failed: {record['failed']})"
            quoted = "\n".join(f"    {line}" for line in body.splitlines())
            lines.append(f"**Reply {label}:**\n\n{quoted}\n")
    sheet_path.write_text("\n".join(lines), encoding="utf-8")
    key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(f"{len(shared)} items → {sheet_path} (key: {key_path})")
    return 0


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #

DIMENSIONS = ("accuracy", "pedagogy", "conversation", "overall")


def score(key_path: Path, verdict_paths: list[Path]) -> int:
    key = json.loads(key_path.read_text(encoding="utf-8"))
    models = sorted({v for entry in key.values() for v in (entry["A"], entry["B"])})
    by_probe = {p.id: p for p in PROBES}

    for path in verdict_paths:
        raw = path.read_text(encoding="utf-8")
        items = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])["items"]
        wins: dict[str, Counter] = {d: Counter() for d in DIMENSIONS}
        by_kind: dict[str, Counter] = defaultdict(Counter)
        faults: dict[str, Counter] = {m: Counter() for m in models}
        replies_with_fault: Counter = Counter()
        seen = 0
        for verdict in items:
            entry = key.get(verdict["item"])
            if entry is None:
                continue
            seen += 1
            for dim in DIMENSIONS:
                choice = verdict.get(dim, "tie")
                winner = entry[choice] if choice in ("A", "B") else "tie"
                wins[dim][winner] += 1
                if dim == "overall":
                    by_kind[by_probe[entry["probe"]].kind][winner] += 1
            for label in "AB":
                listed = verdict.get("faults", {}).get(label, [])
                faults[entry[label]].update(listed)
                if listed:
                    replies_with_fault[entry[label]] += 1

        print(f"=== {path.name}: {seen} of {len(key)} items ===")
        for dim in DIMENSIONS:
            row = " · ".join(f"{m} {wins[dim][m]}" for m in models)
            print(f"  {dim:<13} {row} · tie {wins[dim]['tie']}")
        print("  overall by probe kind:")
        for kind, tally in sorted(by_kind.items()):
            row = " · ".join(f"{m} {tally[m]}" for m in models)
            print(f"    {kind:<17} {row} · tie {tally['tie']}")
        print("  replies with at least one fault:")
        for m in models:
            print(f"    {m}: {replies_with_fault[m]} of {seen}")
        print("  faults:")
        for m in models:
            listed = ", ".join(f"{name} {n}" for name, n in faults[m].most_common()) or "none"
            print(f"    {m}: {listed}")
        print()
    return 0


def counts_table(paths: list[Path]) -> int:
    """The counter's half, per model: no judge involved."""
    for path in paths:
        records = load(path)
        model = records[0]["model"] if records else path.name
        n = len(records)
        # Recounted from the reply, not read from the record: a fix to the
        # counter applies to a run already on disk without re-running a model.
        by_probe = {p.id: p for p in PROBES}
        c = [count(r["reply"], by_probe[r["probe"]]) for r in records]
        shown = [x["right_form_shown"] for x in c if x["right_form_shown"] is not None]
        conv = [x for r, x in zip(records, c, strict=True) if r["mode"] == "conversation"]
        words = sorted(x["words"] for x in c)
        print(f"=== {model}: {n} replies ===")
        print(f"  failed turns            {sum(1 for r in records if r['failed'])}")
        print(f"  words, median · max     {words[n // 2]} · {words[-1]}")
        print(f"  over 4 sentences        {sum(x['over_length'] for x in c)}")
        print(
            f"  ends on a question      {sum(x['ends_on_question'] for x in conv)} of "
            f"{len(conv)} (conversation mode)"
        )
        print(f"  right form shown        {sum(shown)} of {len(shown)}")
        for flag in ("portuguese", "blockquote", "list", "heading", "typed_ipa", "emoji"):
            print(f"  {flag:<23} {sum(x[flag] for x in c)}")
        total = sorted(r["total_ms"] for r in records)
        print(f"  turn, median            {total[n // 2] / 1000:.1f}s")
        print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run every probe against one model")
    p_run.add_argument("--model", required=True)
    p_run.add_argument("--runs", type=int, default=3)
    p_run.add_argument("--out", type=Path)
    p_run.add_argument("--only", nargs="*", default=[], help="probe ids to run")

    p_sheet = sub.add_parser("sheet", help="write a blind judging sheet for two runs")
    p_sheet.add_argument("first", type=Path)
    p_sheet.add_argument("second", type=Path)
    p_sheet.add_argument("--sheet", type=Path, default=Path("prose-sheet.md"))
    p_sheet.add_argument("--key", type=Path, default=Path("prose-key.json"))
    p_sheet.add_argument("--seed", type=int, default=1)

    p_score = sub.add_parser("score", help="unblind judges' verdicts against the key")
    p_score.add_argument("--key", type=Path, required=True)
    p_score.add_argument("verdicts", type=Path, nargs="+")

    p_counts = sub.add_parser("counts", help="the counted metrics per model")
    p_counts.add_argument("runs", type=Path, nargs="+")

    args = parser.parse_args()
    if args.command == "run":
        out = args.out or Path(f"prose-{args.model.replace(':', '_').replace('/', '_')}.jsonl")
        return asyncio.run(run(args.model, max(args.runs, 1), out, set(args.only)))
    if args.command == "sheet":
        return sheet(args.first, args.second, args.sheet, args.key, args.seed)
    if args.command == "score":
        return score(args.key, args.verdicts)
    return counts_table(args.runs)


if __name__ == "__main__":
    raise SystemExit(main())
