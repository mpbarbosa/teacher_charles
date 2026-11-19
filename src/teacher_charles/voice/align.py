"""When each word of a sentence is spoken, from what an STT heard.

The lesson reader lights the word being said. Kokoro returns audio and no
timings, so the timings come from listening to the synthesised audio
(``SpeechToText.word_times``) — and what the STT hears is not the text:

- it splits and joins words: *pre-date* comes back as ``pre`` + ``-date``;
- it drops punctuation that stands alone: the ``—`` between two clauses;
- it mishears names: *Ghorman* may come back *Gorman*.

So the match is made **letter by letter**, not word by word. Both sides are
reduced to lowercase letters and digits, ``difflib`` finds the longest common
runs, and each word of the text takes the time of its first letter the STT
heard. A word with no heard letter at all (a lone dash, a name heard as
something else entirely) takes the time between its neighbours. Words keep
their order: a start never goes back in time.

Pure function, no audio and no model, so it is tested with no weight on disk.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from .base import TimedWord

_NOT_LETTER = re.compile(r"[^0-9a-z]+")


def _letters(word: str) -> str:
    # The typographic apostrophe and plain one both go: "I’ve" and "I've"
    # are the same four letters to the matcher.
    return _NOT_LETTER.sub("", word.lower())


def word_starts(words: list[str], heard: list[TimedWord], duration: float) -> list[float]:
    """The second at which each of ``words`` starts, or ``[]`` with nothing heard.

    ``words`` are the sentence's words as the page shows them (split on
    whitespace); ``duration`` is the audio's length, the ceiling for a word
    the STT never placed.
    """
    if not words or not heard:
        return []

    # The heard side as one string of letters, each with the moment it is
    # said: a word's span is shared out evenly among its letters.
    heard_letters: list[str] = []
    heard_times: list[float] = []
    for w in heard:
        letters = _letters(w.word)
        span = max(w.end - w.start, 0.0)
        for k, letter in enumerate(letters):
            heard_letters.append(letter)
            heard_times.append(w.start + span * k / len(letters))

    owner: list[int] = []  # text letter -> index of its word
    text_letters: list[str] = []
    for i, w in enumerate(words):
        for letter in _letters(w):
            text_letters.append(letter)
            owner.append(i)

    starts: list[float | None] = [None] * len(words)
    matcher = SequenceMatcher(None, text_letters, heard_letters, autojunk=False)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            i = owner[block.a + k]
            t = heard_times[block.b + k]
            if starts[i] is None or t < starts[i]:
                starts[i] = t

    return _fill(starts, duration)


def _fill(starts: list[float | None], duration: float) -> list[float]:
    """Unplaced words between their neighbours, and every start in order."""
    out: list[float] = []
    n = len(starts)
    i = 0
    while i < n:
        if starts[i] is not None:
            out.append(starts[i])
            i += 1
            continue
        # A run of unplaced words, spread evenly between the last placed
        # start and the next one (or the end of the audio).
        j = i
        while j < n and starts[j] is None:
            j += 1
        before = out[-1] if out else 0.0
        after = starts[j] if j < n else duration
        after = max(after, before)
        gap = (after - before) / (j - i + 1)
        for k in range(j - i):
            out.append(before + gap * (k + 1))
        i = j
    # The matcher can place a later word earlier than the one before it (a
    # repeated word matched out of turn); the page plays forward, so clamp.
    for k in range(1, len(out)):
        out[k] = max(out[k], out[k - 1])
    return [round(t, 3) for t in out]
