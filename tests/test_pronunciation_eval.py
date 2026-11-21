"""The evaluator of E11: gates 2 and 3 of ``docs/metodo-fonetico.md`` §3.

It runs on symbols, so the suite needs neither the 318 MB recognizer nor numpy:
the heard side of every case below is what the recognizer wrote into a real
E9 report (``gravacoes/<rule>-<md5>.json``, local and unversioned), copied
here as the derived observation it is. The expected side is espeak, live, the
same as in production; ``ENGLISH_PHONEMES`` stands in for the recognizer's
vocabulary, and the first test holds it to the real one's output.
"""

from __future__ import annotations

import pytest

from teacher_charles.config import REPO_ROOT
from teacher_charles.teaching.phonetics import Phonetics
from teacher_charles.teaching.pronunciation import (
    PHRASE_HIT_MIN,
    Evaluator,
    align,
    expected_for,
    heard_where_expected,
    load_base,
    tokenize,
)
from teacher_charles.voice import ENGLISH_PHONEMES

BASE = load_base(REPO_ROOT / "data" / "contrastive" / "pronunciation.yaml")


@pytest.fixture(scope="module")
def g2p():
    phonetics = Phonetics(with_stress=False)
    if not phonetics.transcribe("the"):
        pytest.fail(f"espeak did not load: {phonetics.last_error}")
    return phonetics.transcribe


@pytest.fixture(scope="module")
def evaluator(g2p):
    return Evaluator(BASE, g2p)


def heard(text: str) -> list[str]:
    return text.split()


# The `esperado` of each rule's E9 reports, which the recognizer's real
# vocabulary produced. The fake vocabulary has to cut the phrase the same way,
# or every test below measures a different expected text from production's.
E9_EXPECTED = {
    "pron.th-voiced": "ð eɪ b ɹ ɔː t ð ɪ ʌ ð ɚ b ɹ ʌ ð ɚ",
    "pron.h-dropping": "h iː h ɐ z ɐ h aʊ s h ɪɹ",
    "pron.ing-alveolar": "aɪ ɐ m s p iː k ɪ ŋ æ n d ɹ iː d ɪ ŋ",
}


@pytest.mark.parametrize("rule_id", sorted(E9_EXPECTED))
def test_the_expected_text_is_the_one_e9_measured(rule_id, g2p):
    expected = expected_for(BASE.get(rule_id).phrases[0].text, g2p, ENGLISH_PHONEMES)
    assert " ".join(expected.phonemes) == E9_EXPECTED[rule_id]


def test_joined_words_keep_both_names(g2p):
    """espeak reads *I am* as one word, ``aɪɐm``; the error is still in *speaking*."""
    expected = expected_for("I am speaking and reading", g2p, ENGLISH_PHONEMES)
    assert expected.words[:3] == ("I am", "I am", "I am")
    assert expected.words[expected.phonemes.index("ŋ")] == "speaking"


def test_tokenize_takes_the_longest_symbol_and_marks_the_unknown():
    assert tokenize("aɪɐm", ENGLISH_PHONEMES) == ["aɪ", "ɐ", "m"]
    assert tokenize("q", frozenset({"a"})) == ["?q"]


def test_align_keeps_where_each_edit_is():
    ops = align(["ð", "ɪ", "ʌ", "ð", "ɚ"], ["ð", "ɪ", "ʌ", "d", "ɚ"])
    assert [(op.kind, op.expected, op.heard, op.at) for op in ops] == [("sub", "ð", "d", 3)]


# --- gate 3: the verdicts on real E9 takes --------------------------------- #


def test_a_clean_take_with_the_error_is_a_correction_in_its_word(evaluator):
    """``b704dc72``: the unmonitored take that named /ð/→/d/ in *other*."""
    judgement = evaluator.evaluate(
        "pron.th-voiced",
        heard("ð eɪ b ɹ oʊ t ð ɚ ʌ d ɚ b ɹ ʌ ð ɚ"),
        ENGLISH_PHONEMES,
    )
    assert judgement.verdict == "correction"
    assert [(f.word, f.expected, f.heard) for f in judgement.findings] == [("other", "ð", "d")]
    assert judgement.opportunities == 4
    assert judgement.explanation == BASE.get("pron.th-voiced").explanation


def test_a_clean_take_without_it_is_no_error_and_not_a_success(evaluator):
    """``3ba35169``: all four /ð/ right, and two divergences no rule cites."""
    judgement = evaluator.evaluate(
        "pron.th-voiced",
        heard("ð eɪ b ɹ ɑː t ð ə ʌ ð ɚ b ɹ ʌ ð ɚ"),
        ENGLISH_PHONEMES,
    )
    assert judgement.verdict == "no-error"
    assert judgement.opportunities == 4
    assert judgement.findings == ()


def test_operations_outside_the_signature_never_reach_the_student(evaluator):
    """Plan §3.2: ``/ɔː/→/oʊ/`` is in the b704dc72 alignment and in no rule."""
    judgement = evaluator.evaluate(
        "pron.th-voiced",
        heard("ð eɪ b ɹ oʊ t ð ɚ ʌ d ɚ b ɹ ʌ ð ɚ"),
        ENGLISH_PHONEMES,
    )
    assert all(f.expected == "ð" for f in judgement.findings)


def test_the_deliberate_take_names_each_word(evaluator):
    """``eaa0d509``: /ð/ said as /d/ on purpose, in *other* and *brother*."""
    judgement = evaluator.evaluate(
        "pron.th-voiced",
        heard("ð eɪ b ɹ ɑː t d uː d ɚ b ɹ uː d ɚ"),
        ENGLISH_PHONEMES,
    )
    assert judgement.verdict == "correction"
    assert {f.word for f in judgement.findings} == {"other", "brother"}


def test_a_deletion_is_named_with_no_heard_side(evaluator):
    """A dropped /h/ in two words: the signature ``[del, h]`` on a take that is the phrase."""
    judgement = evaluator.evaluate(
        "pron.h-dropping",
        heard("iː h ɐ z ɐ aʊ s h ɪɹ"),
        ENGLISH_PHONEMES,
    )
    assert judgement.verdict == "correction"
    assert [(f.word, f.heard) for f in judgement.findings] == [("He", None), ("house", None)]


# --- gate 2: is this the phrase? ------------------------------------------- #


def test_another_sentence_read_in_its_place_is_silence(evaluator):
    """The measured reason gate 2 is not a PER ceiling.

    *The red car is ready* against `pron.th-voiced` scores PER 0.875 — inside
    a ceiling that keeps a real deliberate take — and the alignment then books
    a /ð/→/d/ nobody said. Under the hit floor it is refused instead.
    """
    judgement = evaluator.evaluate(
        "pron.th-voiced", heard("ð ə ɹ ɛ d k ɑːɹ ɪ z ɹ ɛ d i"), ENGLISH_PHONEMES
    )
    assert judgement.verdict == "silence"
    assert judgement.gate == 2
    assert judgement.findings == ()


def test_the_floor_costs_two_takes_that_were_mostly_noise(evaluator):
    """The price of the floor, held in place so it is not paid by surprise.

    ``c3e16004`` (h-dropping, deliberate) and ``365e3cfb`` (ing-alveolar,
    unmonitored) are the clean takes that admitted their rules in E10, and the
    recognizer heard most of each as something else: 0.18 and 0.47 of the
    phrase where it was expected. Silence on them is what refusing every one
    of 751 wrong-phrase pairings costs.
    """
    for rule_id, take in (
        ("pron.h-dropping", "æ z ɑːɹ oʊ z ɪ h iː"),
        ("pron.ing-alveolar", "aɪ ɛ m s p ɪ k ə n ɪ n v eː d ɪ n"),
    ):
        judgement = evaluator.evaluate(rule_id, heard(take), ENGLISH_PHONEMES)
        assert (judgement.verdict, judgement.gate) == ("silence", 2), rule_id


def test_the_floor_sits_between_the_measured_tails(g2p):
    """0.50 was the best any unrelated phrase managed; the floor is above it."""
    expected = expected_for("I know the road", g2p, ENGLISH_PHONEMES).phonemes
    unrelated = heard("ð ə ɹ ɛ d k ɑːɹ ɪ z ɹ ɛ d i")
    assert heard_where_expected(expected, align(expected, unrelated)) < PHRASE_HIT_MIN
    assert PHRASE_HIT_MIN > 0.50


# --- gate 1 arrives as data ------------------------------------------------ #


def test_a_take_gate_one_refused_is_silence_whatever_was_heard(evaluator):
    judgement = evaluator.evaluate(
        "pron.th-voiced",
        heard("ð eɪ b ɹ oʊ t ð ɚ ʌ d ɚ b ɹ ʌ ð ɚ"),
        ENGLISH_PHONEMES,
        signal_usable=False,
        signal_advice="Your voice does not stand out from the room.",
    )
    assert (judgement.verdict, judgement.gate) == ("silence", 1)
    assert judgement.advice.startswith("Your voice")


# --- what the payload carries ---------------------------------------------- #


def test_the_payload_has_no_score_and_no_transcription(evaluator):
    """Plan §4 rules out a score; plan §3.2 keeps the uncurated operations away."""
    payload = evaluator.evaluate(
        "pron.th-voiced",
        heard("ð eɪ b ɹ oʊ t ð ɚ ʌ d ɚ b ɹ ʌ ð ɚ"),
        ENGLISH_PHONEMES,
    ).as_dict()
    assert set(payload) == {
        "rule_id",
        "verdict",
        "gate",
        "advice",
        "opportunities",
        "errors",
        "findings",
        "explanation",
    }
    assert payload["errors"] == 1


def test_only_a_graded_rule_gives_a_verdict(g2p):
    item = BASE.get("pron.th-voiced")
    demoted = item.model_copy(
        update={"discrimination": item.discrimination.model_copy(update={"status": "heard-only"})}
    )
    from teacher_charles.teaching.pronunciation import PronunciationBase

    with pytest.raises(ValueError, match="gives no verdict"):
        Evaluator(PronunciationBase([demoted]), g2p).evaluate(
            "pron.th-voiced", heard("ð"), ENGLISH_PHONEMES
        )


# --- the free turn (E14) --------------------------------------------------- #
#
# `evaluate_text` is `evaluate` over the student's own words. The two takes
# below are the ones `scripts/measure_stt_accent.py --evaluate` found where a
# transcript and the phrase read gave different verdicts, so each test pins a
# measured case, not an invented one.

# ``eaa0d509``: *They brought the other brother* said wrong on purpose.
OTHER_AND_BROTHER_AS_D = heard("ð eɪ b ɹ ɑː t d uː d ɚ b ɹ uː d ɚ")
# ``1d64e713``: *Sport is good for stress* with the epenthesis, warned by gate 1.
STRESS_WITH_EPENTHESIS = heard("ɪ s p ɑː t ɪ z ɡ uː d f ɔːɹ ɪ s t ɹ ɛ s")


def test_a_free_text_judges_only_the_rules_it_gives_an_opportunity_for(evaluator):
    udder = heard("ð eɪ b ɹ oʊ t ð ɚ ʌ d ɚ b ɹ ʌ ð ɚ")
    judged = evaluator.evaluate_text("They brought the other brother", udder, ENGLISH_PHONEMES)
    assert [j.rule_id for j in judged] == ["pron.th-voiced"]
    (judgement,) = judged
    assert judgement.verdict == "correction"
    assert [(f.word, f.expected, f.heard) for f in judgement.findings] == [("other", "ð", "d")]


def test_a_text_with_no_target_is_not_judged_at_all(evaluator):
    """No opportunity is no observation, which is not a pass."""
    cats = heard("aɪ l aɪ k k æ t s")
    assert evaluator.evaluate_text("I like cats", cats, ENGLISH_PHONEMES) == ()


def test_the_words_a_transcript_kept_are_the_words_it_can_correct(evaluator):
    """base.en heard the deliberate take as *They brought the Buddha brother*.

    The truth names *other* and *brother*; the transcript has no /ð/ left in
    *Buddha*, so it can only name *brother*. A partial correction, about a
    real error — the measured shape of what a transcript does to a
    mispronounced word: it writes another word down, and the error goes with it.
    """
    truth = evaluator.evaluate_text(
        "They brought the other brother", OTHER_AND_BROTHER_AS_D, ENGLISH_PHONEMES
    )
    via = evaluator.evaluate_text(
        "They brought the Buddha brother", OTHER_AND_BROTHER_AS_D, ENGLISH_PHONEMES
    )
    assert {f.word for f in truth[0].findings} == {"other", "brother"}
    assert [f.word for f in via[0].findings] == ["brother"]


def test_the_one_false_correction_measured_is_the_one_gate_one_stands_between(evaluator):
    """tiny.en heard *for stress* as *for his dress*, and *his* has an /h/.

    With the signal taken as usable the alignment books the /h/ nobody said,
    and h-dropping corrects a word the student never spoke. The take carried
    a level warning, so in the product gate 1 makes it silence — which is why
    the route runs gate 1 even though the transcript already exists.
    """
    text = "This spot is good for his dress."
    unguarded = evaluator.evaluate_text(text, STRESS_WITH_EPENTHESIS, ENGLISH_PHONEMES)
    guarded = evaluator.evaluate_text(
        text, STRESS_WITH_EPENTHESIS, ENGLISH_PHONEMES, signal_usable=False
    )
    assert {j.rule_id: j.verdict for j in unguarded} == {
        "pron.th-voiced": "no-error",
        "pron.h-dropping": "correction",
    }
    assert [f.word for f in unguarded[1].findings] == ["his"]
    assert {(j.verdict, j.gate) for j in guarded} == {("silence", 1)}


def test_a_transcript_of_other_words_is_silence_not_a_correction(evaluator):
    """Gate 2 over free speech checks Whisper: the phonemes are not the text.

    The heard side is *They brought the udder brother*; the text is *He has a
    house here*. Every /h/ of the text is missing from what was heard, which
    is exactly h-dropping's signature — and gate 2 has to stop it first.
    """
    judged = evaluator.evaluate_text(
        "He has a house here", heard("ð eɪ b ɹ oʊ t ð ɚ ʌ d ɚ b ɹ ʌ ð ɚ"), ENGLISH_PHONEMES
    )
    assert [(j.rule_id, j.verdict, j.gate) for j in judged] == [("pron.h-dropping", "silence", 2)]


def test_a_symbol_the_ear_cannot_emit_is_silence_and_not_a_broken_base(evaluator):
    """Over a curated phrase that is a 503; over the student's words, a sentence not judged."""
    vocabulary = ENGLISH_PHONEMES - {"s"}
    judged = evaluator.evaluate_text(
        "He has a house here", heard("h iː h æ z ɐ h aʊ s h ɪɹ"), vocabulary
    )
    assert [(j.verdict, j.gate) for j in judged] == [("silence", 2)]
