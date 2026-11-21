"""The sound note: the teacher is handed espeak's sounds, not left to recall them.

Measured before it existed (``docs/plano-fase-1.md`` §5.5): asked whether the
*th* of *think* is the sound of the *th* of *this*, both models said yes, six
times out of six. The facts below are espeak's — the same G2P as the IPA line —
so these tests ask espeak, as ``test_ipa_symbols.py`` does.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from teacher_charles.llm import FakeProvider
from teacher_charles.prompts import PromptLibrary
from teacher_charles.store import db
from teacher_charles.store.repo import SessionRepo, StudentRepo
from teacher_charles.teaching import TurnRunner
from teacher_charles.teaching.ipa_symbols import load_symbols
from teacher_charles.teaching.phonetics import Phonetics
from teacher_charles.teaching.sound_facts import (
    VOICING_PAIRS,
    VOWEL_PAIRS,
    SoundFacts,
    build_sound_facts,
    named_words,
)

TABLE_PATH = Path(__file__).resolve().parents[1] / "data" / "phonetics" / "symbols.yaml"

THINK_THIS = "Is the 'th' in 'think' the same sound as the 'th' in 'this'?"


@pytest.fixture(scope="module")
def table():
    return load_symbols(TABLE_PATH)


@pytest.fixture(scope="module")
def facts(table):
    return SoundFacts(Phonetics(), table)


# --- which words ------------------------------------------------------------ #


@pytest.mark.parametrize(
    ("text", "words"),
    [
        # A quoted spelling is not a word: espeak would read *th* as letter names.
        (THINK_THIS, ["think", "this"]),
        ("Is the 's' in 'rose' the same sound as the 's' in 'rice'?", ["rose", "rice"]),
        ("Do 'thought' and 'taught' sound the same?", ["thought", "taught"]),
        ('What is the difference between "ship" and "sheep"?', ["ship", "sheep"]),
        ("What's the difference between ship and sheep?", ["ship", "sheep"]),
        ("How do I pronounce comfortable?", ["comfortable"]),
        ("How do I pronounce the word Wednesday?", ["Wednesday"]),
        # An apostrophe inside a word opens no quote.
        ("OK, I'm ready. Let's practise.", []),
        ("I have 30 years and I work in a bank.", []),
    ],
)
def test_named_words(text, words):
    assert named_words(text) == words


# --- what the note says ----------------------------------------------------- #


def test_think_and_this_differ_in_voicing(facts):
    """The measured failure, held by the fact that fixes it."""
    note = facts.note(THINK_THIS)

    assert '"think": θ ɪ ŋ k' in note
    assert '"this": ð ɪ s' in note
    assert 'θ (as in "think") in "think" where "this" has ð' in note
    assert "ð is voiced (the vocal cords vibrate) and θ is not" in note
    assert "the same ɪ in both" in note


def test_ship_and_sheep_differ_in_the_vowel_only(facts):
    note = facts.note("What is the difference between 'ship' and 'sheep'?")

    assert "the same ʃ in both" in note
    assert "the same p in both" in note
    assert 'ɪ (as in "sit") in "ship" where "sheep" has iː (as in "see")' in note
    assert "ɪ is short and relaxed, iː is longer and tenser" in note


def test_a_long_replacement_is_still_split_sound_by_sound(facts):
    """*oʊ z* against *aɪ s* is two pairs, and the second is a voicing pair."""
    note = facts.note("Is the 's' in 'rose' the same sound as the 's' in 'rice'?")

    assert 'z (as in "zoo") in "rose" where "rice" has s' in note
    assert "z is voiced" in note


def test_a_pair_with_no_curated_fact_is_only_called_different(facts):
    note = facts.note("Do 'thought' and 'taught' sound the same?")

    assert 'θ (as in "think") in "thought" where "taught" has t' in note
    assert "voiced" not in note
    assert "the same ɔː t in both" in note


def test_one_word_gets_its_sounds_and_no_comparison(facts):
    note = facts.note("How do I pronounce comfortable?")

    assert '"comfortable":' in note
    assert "Comparing" not in note


def test_the_note_tells_him_not_to_type_ipa(facts):
    """The partial says it, and the 4B typed IPA eight times anyway."""
    assert "instead of typing IPA yourself" in facts.note(THINK_THIS)


# --- when it speaks --------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        # A difference question with no sound word: about meaning, not sound.
        "What is the difference between 'make' and 'do'?",
        "What is the difference between 'actually' and 'currently'?",
        # Sound words and no word named: nothing to read.
        "Can you listen to my pronunciation and tell me if it is good?",
        # An ordinary turn.
        "I have 30 years and I work in a bank.",
    ],
)
def test_no_note(facts, text):
    assert facts.note(text) == ""


def test_a_minimal_pair_is_a_sound_question_without_saying_so(facts):
    """*Ship* and *sheep* are one sound apart, so the difference is a sound."""
    assert facts.note("What's the difference between ship and sheep?")


def test_the_curated_pairs_are_symbols_of_the_table(table):
    """A pair naming a symbol the table lacks would never be found by segment()."""
    known = {s.symbol for s in table.symbols}
    for pair in (*VOICING_PAIRS, *VOWEL_PAIRS):
        assert set(pair) <= known, pair


def test_no_builder_without_the_g2p_or_the_table(table):
    assert build_sound_facts(None, table) is None
    assert build_sound_facts(Phonetics(), None) is None


# --- where the note goes ---------------------------------------------------- #


@pytest.fixture
def conn(tmp_path):
    connection = db.init_db(tmp_path / "sound.db")
    yield connection
    connection.close()


@pytest.fixture
def prompts(tmp_path):
    root = tmp_path / "prompts"
    root.mkdir()
    (root / "persona.md").write_text("You are Charles.", encoding="utf-8")
    return PromptLibrary(root)


async def _messages(prompts, conn, text, sound_facts):
    student = StudentRepo(conn).get_or_create("ana")
    session_id = SessionRepo(conn).current_or_start(student.id)
    provider = FakeProvider(replies="ok")
    runner = TurnRunner(provider=provider, prompts=prompts, conn=conn, sound_facts=sound_facts)
    async for _ in runner.run(session_id, text, student):
        pass
    return provider.calls_matching("You are Charles")[-1].messages


async def test_the_note_rides_with_the_student_message(prompts, conn, facts):
    messages = await _messages(prompts, conn, THINK_THIS, facts)

    assert [m.role for m in messages] == ["system", "user"]
    assert messages[0].content == "You are Charles."
    last = messages[-1].content
    assert last.startswith(THINK_THIS)
    assert "ð is voiced" in last


async def test_without_the_builder_the_turn_is_what_it_was(prompts, conn):
    messages = await _messages(prompts, conn, THINK_THIS, None)

    assert messages[-1].content == THINK_THIS
