"""Word timings for the lesson reader, mapped from what an STT heard.

No audio and no model: ``word_starts`` takes words and timed words, so every
case the alignment has to survive is written here as data. Each one is a
thing Whisper actually did to Kokoro's reading of the Mon Mothma lesson.
"""

from __future__ import annotations

from teacher_charles.voice.align import word_starts
from teacher_charles.voice.base import TimedWord as T


def test_heard_as_written_takes_the_heard_starts():
    words = ["I", "came", "here."]
    heard = [T("I", 0.0, 0.2), T("came", 0.2, 0.5), T("here.", 0.5, 0.9)]
    assert word_starts(words, heard, 1.0) == [0.0, 0.2, 0.5]


def test_a_word_heard_in_two_pieces_starts_with_the_first():
    words = ["memories", "that", "pre-date", "my"]
    heard = [
        T("memories", 3.02, 3.4),
        T("that", 3.4, 3.74),
        T("pre", 3.74, 3.9),
        T("-date", 3.9, 4.12),
        T("my", 4.12, 4.34),
    ]
    assert word_starts(words, heard, 5.0) == [3.02, 3.4, 3.74, 4.12]


def test_a_dash_nobody_says_sits_between_its_neighbours():
    words = ["yesterday", "—", "WHAT"]
    heard = [T("yesterday", 0.48, 1.0), T("WHAT", 1.5, 1.7)]
    starts = word_starts(words, heard, 2.0)
    assert starts[0] == 0.48 and starts[2] == 1.5
    assert 0.48 < starts[1] < 1.5


def test_a_misheard_name_still_lands_on_its_letters():
    """Whisper writes 'Gorman'; five letters of seven are enough to place it."""
    words = ["on", "the", "Ghorman", "Plaza."]
    heard = [
        T("on", 2.5, 2.88),
        T("the", 2.88, 3.0),
        T("Gorman", 3.0, 3.26),
        T("Plaza.", 3.26, 3.52),
    ]
    assert word_starts(words, heard, 4.0) == [2.5, 2.88, 3.0, 3.26]


def test_shouting_and_curly_apostrophes_are_the_same_letters():
    words = ["I’ve", "SPENT"]
    heard = [T("I've", 0.1, 0.3), T("spent", 0.3, 0.6)]
    assert word_starts(words, heard, 1.0) == [0.1, 0.3]


def test_a_word_never_starts_before_the_one_before_it():
    words = ["the", "truth", "the", "end"]
    heard = [T("the", 0.0, 0.1), T("truth", 0.1, 0.4), T("end", 0.6, 0.9)]
    starts = word_starts(words, heard, 1.0)
    assert starts == sorted(starts)
    assert starts[1] == 0.1 and starts[3] == 0.6


def test_overlapping_heard_words_still_play_forward():
    """Whisper's word spans can overlap; a word matched by a late letter must not jump back."""
    words = ["xb", "c"]
    # "b" is the second letter of a word heard from 0.0 to 1.0, so it sits at
    # 0.5 -- after "c", which the STT placed at 0.2.
    heard = [T("ab", 0.0, 1.0), T("c", 0.2, 0.3)]
    starts = word_starts(words, heard, 1.0)
    assert starts == sorted(starts)


def test_words_after_the_last_heard_one_spread_to_the_end():
    words = ["one", "two", "three"]
    heard = [T("one", 0.0, 0.3)]
    assert word_starts(words, heard, 0.9) == [0.0, 0.3, 0.6]


def test_nothing_heard_is_no_timings():
    """Empty, not invented: the page then lights the sentence alone."""
    assert word_starts(["I", "came"], [], 1.0) == []
