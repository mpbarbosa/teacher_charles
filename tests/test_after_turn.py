"""What the student reads under their bubble after a spoken turn (E14).

Only curated corrections, each said once per word, and nothing at all for
``no-error`` or silence: a clean free turn is not a pass.
"""

from __future__ import annotations

from teacher_charles.teaching.after_turn import notes
from teacher_charles.teaching.pronunciation import Finding, Judgement


def correction(rule_id: str, *findings: Finding) -> Judgement:
    return Judgement(
        rule_id, "correction", opportunities=4, findings=findings, explanation="Why it is so."
    )


def test_only_a_correction_reaches_the_student():
    judged = [
        Judgement("pron.h-dropping", "no-error", opportunities=4),
        Judgement("pron.ing-alveolar", "silence", gate=2),
        correction("pron.th-voiced", Finding("other", "ð", "d")),
    ]
    assert notes(judged) == [
        {
            "rule_id": "pron.th-voiced",
            "heard": "In *other*, /ð/ came out as /d/.",
            "explanation": "Why it is so.",
        }
    ]


def test_a_clean_turn_says_nothing():
    assert notes([Judgement("pron.th-voiced", "no-error", opportunities=2)]) == []


def test_the_same_word_twice_is_said_once_and_a_deletion_is_worded_as_missing():
    judged = [
        correction(
            "pron.h-dropping",
            Finding("house", "h", None),
            Finding("house", "h", None),
            Finding("here", "h", None),
        )
    ]
    assert notes(judged)[0]["heard"] == (
        "In *house*, the /h/ was missing; in *here*, the /h/ was missing."
    )
