"""Pronunciation after a conversation turn (E14): what the student reads under their bubble.

The evaluator is E11's, run over the student's own words instead of a curated
phrase (``Evaluator.evaluate_text``). Those words are Whisper's transcript,
and that is the whole risk of this surface: a transcript that heard another
word makes the evaluator correct a sentence nobody said. Plan 2 §5 measured it
before this module existed, and the three things that hold it are not here,
because none of them is wording:

- the route judges a take only against **its own transcript, unedited** — an
  edit means the audio and the text no longer say the same thing;
- gate 1 runs on the audio first, as in the drill;
- gate 2 asks whether the heard phonemes *are* the transcript, which over
  free speech is an independent check on Whisper.

What is here is the other half of plan §3.2: only a correction that cites a
curated rule reaches the student, and ``no-error`` and silence show nothing.
A clean free turn is not a pass — §6's asymmetry, and the reason this records
nothing either: what a free turn's take is worth to the schedule is a decision
this deliverable does not take (plan 2 §5, after the table).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .pronunciation import Judgement
from .read_aloud import what_happened


def notes(judgements: Sequence[Judgement]) -> list[dict[str, Any]]:
    """One note per rule the take broke, in the base's order, and nothing else.

    A word named twice in one rule (the alignment can book two operations in
    it) is said once: the student reads about words, not operations.
    """
    out = []
    for judgement in judgements:
        if judgement.verdict != "correction":
            continue
        seen: set[tuple[str, str, str | None]] = set()
        said = []
        for finding in judgement.findings:
            key = (finding.word, finding.expected, finding.heard)
            if key in seen:
                continue
            seen.add(key)
            said.append(what_happened(*key))
        sentence = "; ".join(said)
        out.append(
            {
                "rule_id": judgement.rule_id,
                # Not `.capitalize()`, which lowercases the rest: the word
                # may be *I*.
                "heard": sentence[:1].upper() + sentence[1:] + ".",
                "explanation": judgement.explanation,
            }
        )
    return out


__all__ = ["notes"]
