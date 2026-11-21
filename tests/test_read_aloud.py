"""The read-aloud tally and card (E12), with no database and no audio.

Plan §5's rule for a session of three takes is what is held here: named in
two or more judged takes is a miss, named once reviews nothing, and named in
none is a hit **only** if the rule has ever been named for this student —
metodo §2.4's guard against graduating by silence.
"""

from __future__ import annotations

import json
import re

import pytest

from teacher_charles.config import REPO_ROOT
from teacher_charles.store.models import PronunciationSignal
from teacher_charles.teaching import read_aloud
from teacher_charles.teaching.pronunciation import Finding, Judgement, load_base
from teacher_charles.teaching.read_aloud import card, deliberate_note, tally, wrong_side

BASE = load_base(REPO_ROOT / "data" / "contrastive" / "pronunciation.yaml")
TH = BASE.get("pron.th-voiced")
H = BASE.get("pron.h-dropping")
ING = BASE.get("pron.ing-alveolar")

OTHER = {"word": "other", "expected": "ð", "heard": "d"}
BROTHER = {"word": "brother", "expected": "ð", "heard": "d"}


def take(findings=(), pass_="unmonitored", n=[0]) -> PronunciationSignal:  # noqa: B006
    n[0] += 1
    return PronunciationSignal(
        id=n[0],
        session_id=1,
        student_id=1,
        rule_id=TH.id,
        phrase=0,
        pass_=pass_,
        verdict="correction" if findings else "no-error",
        opportunities=4,
        errors=len(findings),
        findings=json.dumps(list(findings)),
        model="fake",
        duration_s=3.0,
        speech_from_s=1.0,
        speech_to_s=2.5,
        created_at="",
    )


# --- the tally ------------------------------------------------------------- #


def test_named_in_two_of_three_is_a_miss():
    result = tally(TH.id, [take([OTHER]), take(), take([OTHER])], ever_named=True)
    assert (result.outcome, result.review_hit) == ("miss", False)
    assert result.named == 2 and result.takes == 3


def test_named_once_reviews_nothing():
    """Plan §5: one take is not a finding. Once moves the rule neither way."""
    result = tally(TH.id, [take([OTHER]), take(), take()], ever_named=True)
    assert (result.outcome, result.review_hit) == ("once", None)


def test_clean_takes_are_a_hit_once_the_rule_was_heard_in_this_voice():
    result = tally(TH.id, [take(), take(), take()], ever_named=True)
    assert (result.outcome, result.review_hit) == ("hit", True)


def test_clean_takes_graduate_nothing_when_the_rule_was_never_named():
    """Metodo §2.4: the silence may be the evaluator not hearing this rule here."""
    result = tally(TH.id, [take(), take(), take()], ever_named=False)
    assert (result.outcome, result.reason, result.review_hit) == ("none", "unheard", None)


def test_one_clean_take_is_too_few_for_a_hit():
    result = tally(TH.id, [take()], ever_named=True)
    assert (result.outcome, result.reason) == ("none", "few")


def test_the_deliberate_take_is_never_tallied():
    """It is said wrong on purpose: counting it would book a miss nobody made."""
    result = tally(
        TH.id,
        [take([OTHER, BROTHER], pass_="deliberate"), take(), take(), take()],
        ever_named=True,
    )
    assert result.outcome == "hit"
    assert result.takes == 3


def test_a_word_is_counted_once_per_take():
    result = tally(TH.id, [take([OTHER, OTHER]), take([OTHER])], ever_named=True)
    assert dict(result.by_word)[("other", "ð", "d")] == 2


# --- the card --------------------------------------------------------------- #


def test_the_card_speaks_in_takes_and_names_the_word():
    result = tally(TH.id, [take([OTHER]), take(), take([OTHER])], ever_named=True)
    lines = card(TH, result, due_in_days=0.5)["lines"]
    assert lines[0] == "In *other*, /ð/ came out as /d/ in 2 of 3 takes."
    assert TH.explanation in lines
    assert lines[-1] == "It comes back in half a day."


def test_a_deletion_is_worded_as_missing():
    finding = {"word": "house", "expected": "h", "heard": None}
    result = tally(H.id, [take([finding]), take([finding])], ever_named=True)
    assert card(H, result)["lines"][0] == "In *house*, the /h/ was missing in 2 of 2 takes."


@pytest.mark.parametrize(
    ("takes", "named"),
    [
        ([[OTHER], [], [OTHER]], True),
        ([[OTHER], [], []], True),
        ([[], [], []], True),
        ([[], [], []], False),
        ([[]], True),
    ],
)
def test_no_card_carries_a_score(takes, named):
    """Plan §4: no percentage, no 0–100. The only numbers are counts of takes."""
    result = tally(TH.id, [take(f) for f in takes], ever_named=named)
    text = " ".join(card(TH, result, due_in_days=1.0)["lines"])
    assert "%" not in text
    for number in re.findall(r"\d+(?:\.\d+)?", text):
        assert number in {"1", "2", "3"}, (number, text)


def test_the_deliberate_note_says_what_came_out():
    named = Judgement(TH.id, "correction", opportunities=4, findings=(Finding("other", "ð", "d"),))
    assert "now your clean reads can count" in deliberate_note(TH, named)
    clean = Judgement(TH.id, "no-error", opportunities=4)
    assert "will not count until a take names it" in deliberate_note(TH, clean)


# --- the contrast pair ------------------------------------------------------ #


def test_the_wrong_side_is_the_swap_and_nothing_else():
    assert wrong_side("ðeɪ bɹˈɔːt ðɪ ˈʌðɚ bɹˈʌðɚ", TH) == "deɪ bɹˈɔːt dɪ ˈʌdɚ bɹˈʌdɚ"
    assert wrong_side("hiː hɐz ɐ hˈaʊs hˈɪɹ", H) == "iː ɐz ɐ ˈaʊs ˈɪɹ"
    assert wrong_side("ˈaɪɐm spˈiːkɪŋ ænd ɹˈiːdɪŋ", ING) == "ˈaɪɐm spˈiːkɪn ænd ɹˈiːdɪn"


def test_a_phrase_without_the_target_has_no_wrong_side():
    assert wrong_side("ðə ɹˈɛd kˈɑːɹ", H) is None


def test_the_constants_are_the_plans():
    assert (read_aloud.TAKES, read_aloud.MISS_AT, read_aloud.HIT_MIN_TAKES) == (3, 2, 2)
