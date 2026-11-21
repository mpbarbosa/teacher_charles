"""The read-aloud drill (E12): what three takes of one phrase are worth.

E11 judges one take. This module decides what a **session** of them says,
because plan §5's read-aloud paragraphs measured that one take is not a
finding: two things that looked like habits in a practice session's first
takes did not come back, and what recurred was what repeated. So the student
reads the curated phrase three times and is told about the session, not the
take, and the schedule gets **one** review from the tally:

* named in 2 or more judged takes — a **miss**;
* named in exactly one — **once**: shown as "once in three", and it reviews
  nothing, because once is not a habit in either direction;
* named in none of at least two judged takes — a **hit**, but only under
  metodo §2.4's guard: the rule must have been named at least once, in some
  take, **for this student**. Otherwise the silence may be the evaluator not
  resolving this rule in this voice at all, and a rule cannot be graduated by
  silence — the same guard as E13's "an analysis that did not run graduates
  nobody", one layer down;
* anything else — **none**: too few takes, or never heard in this voice.

The threshold of 2 of 3 is reasoning over 18 takes of operations no rule cites
(plan §6), not a measurement. ``MISS_AT`` and ``HIT_MIN_TAKES`` are where it
moves when E10's clean takes measure it.

**The deliberate take** is metodo §2.4's step: said wrong on purpose, once per
rule and never scheduled twice, after the pair and before the drill takes, so
the session ends on the correct production. It is never tallied. What it buys
is the guard above: a deliberate take the alignment names is proof that this
stack hears this rule in this voice.

Nothing here touches the database or audio: the route reads the takes, this
module reads them back as a card, and the card is the student's words. It
speaks in takes ("2 of 3"), never in a score (plan §4).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from ..store.models import PronunciationSignal, RuleCount
from .pronunciation import Judgement, PronunciationBase, PronunciationItem
from .schedule import Scheduler

# Judged takes a session asks for.
TAKES = 3
# Named in this many judged takes is a miss. Reasoning, not measurement (plan §6).
MISS_AT = 2
# A hit needs at least this many judged takes with nothing named.
HIT_MIN_TAKES = 2

Outcome = Literal["miss", "hit", "once", "none"]


@dataclass(frozen=True, slots=True)
class Tally:
    """What the judged takes of one session add up to."""

    rule_id: str
    takes: int
    named: int
    # (word, expected, heard) → in how many takes it was named.
    by_word: tuple[tuple[tuple[str, str, str | None], int], ...]
    outcome: Outcome
    # Why 'none': "few" (fewer than HIT_MIN_TAKES judged) or "unheard"
    # (the guard: never named for this student).
    reason: str | None = None

    @property
    def review_hit(self) -> bool | None:
        """What the schedule hears: ``True``, ``False``, or nothing at all."""
        if self.outcome == "miss":
            return False
        if self.outcome == "hit":
            return True
        return None


def findings_of(signal: PronunciationSignal) -> list[tuple[str, str, str | None]]:
    return [(f["word"], f["expected"], f.get("heard")) for f in json.loads(signal.findings)]


def tally(rule_id: str, takes: Sequence[PronunciationSignal], *, ever_named: bool) -> Tally:
    """Plan §5's rule over one session's unmonitored takes.

    ``ever_named`` is metodo §2.4's guard: has any take, in any session and of
    either pass, ever named this rule for this student.
    """
    drill = [take for take in takes if take.pass_ == "unmonitored"]
    named = sum(1 for take in drill if take.errors > 0)
    words: Counter[tuple[str, str, str | None]] = Counter()
    for take in drill:
        # Counted once per take: "in 2 of 3 takes" is about takes, and a word
        # named twice in one take (none of today's phrases repeats a word with
        # its target, but a phrase may) is still one take.
        words.update(set(findings_of(take)))
    by_word = tuple(sorted(words.items(), key=lambda kv: (-kv[1], kv[0][0])))

    if named >= MISS_AT:
        outcome: Outcome = "miss"
        reason = None
    elif named == 1:
        outcome, reason = "once", None
    elif len(drill) < HIT_MIN_TAKES:
        outcome, reason = "none", "few"
    elif not ever_named:
        outcome, reason = "none", "unheard"
    else:
        outcome, reason = "hit", None
    return Tally(rule_id, len(drill), named, by_word, outcome, reason)


# --- what the student reads -------------------------------------------------- #


def _what_happened(word: str, expected: str, heard: str | None) -> str:
    """One finding in metodo §4's words: the phoneme by name, in its word.

    Consonant identity is resolved (metodo §4), and all three graded rules are
    consonants; a vowel rule would have to be worded by class, and none is
    admitted yet.
    """
    if heard is None:
        return f"in *{word}*, the /{expected}/ was missing"
    return f"in *{word}*, /{expected}/ came out as /{heard}/"


def _takes(n: int) -> str:
    return f"{n} take" if n == 1 else f"{n} takes"


def looked_for(item: PronunciationItem) -> str:
    """What the rule's signature looks for, in a phrase the card can use."""
    op = item.signature[0]
    if op[0] == "del":
        return f"a missing /{op[1]}/"
    if op[0] == "sub":
        return f"/{op[1]}/ coming out as /{op[2]}/"
    return f"an extra /{op[1]}/"


def card(item: PronunciationItem, result: Tally, *, due_in_days: float | None = None) -> dict:
    """The card over the session. Sentences, not numbers to interpret."""
    lines: list[str] = []
    if result.outcome == "miss":
        for (word, expected, heard), count in result.by_word:
            if count >= MISS_AT:
                lines.append(
                    _what_happened(word, expected, heard).capitalize()
                    + f" in {count} of {result.takes} takes."
                )
        if not lines:
            # Named in two takes, but in different words: still the same habit.
            lines.append(
                f"{looked_for(item).capitalize()} was heard in "
                f"{result.named} of {result.takes} takes."
            )
        lines.append(item.explanation)
    elif result.outcome == "once":
        (word, expected, heard), _ = result.by_word[0]
        lines.append(
            f"Once in {_takes(result.takes)}: {_what_happened(word, expected, heard)}. "
            "Once is not a habit, so this session does not move the rule either way."
        )
    elif result.outcome == "hit":
        lines.append(
            f"None of your {_takes(result.takes)} had {looked_for(item)}. "
            "That counts: the rule moves up your schedule."
        )
    elif result.reason == "unheard":
        lines.append(
            f"None of your {_takes(result.takes)} had {looked_for(item)}. "
            "But it has never once been heard in your voice, so a clean read cannot "
            "count yet: it may only mean this rule is hard to hear in your voice."
        )
    else:
        lines.append(
            f"Only {_takes(result.takes)} could be judged, which is not enough to say "
            "anything. Try again somewhere quieter."
        )
    if due_in_days is not None:
        lines.append(f"It comes back {_when(due_in_days)}.")
    return {
        "rule_id": item.id,
        "outcome": result.outcome,
        "takes": result.takes,
        "named": result.named,
        "lines": lines,
        "right": item.right,
    }


def _when(days: float) -> str:
    if days < 1:
        return "in half a day"
    if days == 1:
        return "tomorrow"
    return f"in {days:g} days"


def deliberate_note(item: PronunciationItem, judgement: Judgement) -> str:
    """What the take said wrong on purpose showed. It is calibration, not a mistake."""
    if judgement.verdict == "silence":
        return judgement.advice or ""
    if judgement.findings:
        first = judgement.findings[0]
        return (
            f"Heard: {_what_happened(first.word, first.expected, first.heard)}. "
            "That is the wrong side, so now your clean reads can count. "
            "Next, read it the right way."
        )
    return (
        f"That did not come out as {looked_for(item)}. This rule may be hard to hear "
        "in your voice, so clean reads will not count until a take names it. "
        "Next, read it the right way."
    )


# --- the contrast pair -------------------------------------------------------- #


def wrong_side(phonemes: str, item: PronunciationItem) -> str | None:
    """The phrase's phonemes with the curated swap applied (metodo §2.2).

    The same construction as ``--discriminate``: the swap at the phoneme level,
    so the two sides differ in the contrast and nothing else. ``None`` when the
    signature is an insertion, which has no single place to apply, or when the
    phrase does not carry the target at all.
    """
    out = phonemes
    for op in item.signature:
        if op[0] == "sub":
            out = out.replace(op[1], op[2])
        elif op[0] == "del":
            out = out.replace(op[1], "")
        else:
            return None
    return out if out != phonemes else None


# --- which rule ---------------------------------------------------------------- #


def queue(
    base: PronunciationBase,
    scheduler: Scheduler,
    student_id: int,
    *,
    errors: dict[str, int],
    recent: list[str],
    at: datetime | None = None,
) -> list[str]:
    """The graded rules in the schedule's order, the most urgent first.

    E13's ``Scheduler.queue``, unchanged, over a different candidate list: every
    graded rule, with the takes that named it as the frequency tiebreak and
    read-aloud's own sessions as the rotation.
    """
    candidates = [
        RuleCount(rule_id=item.id, times=errors.get(item.id, 0), last_span="", last_correction="")
        for item in base.graded()
    ]
    return [q.rule_id for q in scheduler.queue(student_id, candidates, at=at, recent=recent)]


def as_take_payload(judgement: Judgement, *, recorded: bool, judged: int) -> dict[str, Any]:
    """One take's answer. The verdict is in it, but the page shows the session."""
    payload = judgement.as_dict()
    payload["recorded"] = recorded
    payload["judged"] = judged
    payload["needed"] = TAKES
    return payload


__all__ = [
    "HIT_MIN_TAKES",
    "MISS_AT",
    "TAKES",
    "Tally",
    "as_take_payload",
    "card",
    "deliberate_note",
    "looked_for",
    "queue",
    "tally",
    "wrong_side",
]
