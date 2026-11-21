"""Where a student reading a text aloud has got to, from what an STT heard.

The rehearsal's "Follow me" lights the word the student is saying. The page
sends the last few seconds of the microphone, the timing Whisper hears them,
and this module finds where in the text those heard words end. It is
``align.py``'s problem turned around: there the audio is a known sentence and
the question is *when* each word starts; here the audio is a stretch of the
student's reading and the question is *which* word of the text was said last.

## Letter by letter, from one sure match outward

What the STT hears is not the text, for the reasons ``align.py`` lists — split
and joined words, dropped punctuation, misheard names — and here also because
the speaker is a learner: a word slurred, a word repeated, a word skipped. So
the match is made over letters, by ``difflib``, as it is there.

But the text is much longer than what was heard (a window of forty words, or
the whole text while the light is lost), and over a long text a few letters
in common are chance: *andthe*, *ofthe*, *ing*. Taking the furthest match
would let one such chance pull the light paragraphs ahead. So the answer is
built from one **anchor**, the longest run of letters the two sides share,
which must be long enough to be two or three words and not chance; then the
runs after it are followed only while each starts close to where the last
one ended, on both sides. The light is the word holding the last letter of
that chain: the most recent thing heard that is also surely in the text.

Pure function, no audio and no model, so it is tested with no weight on disk.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

_NOT_LETTER = re.compile(r"[^0-9a-z]+")

# The anchor's least length, in letters: about two short words. Shorter, and
# a window of forty words offers chance matches of that size.
MIN_ANCHOR = 10

# How far, in letters on either side, the next run may start from the end of
# the last one and still continue the chain: a misheard word or two.
MAX_GAP = 14

# A run shorter than this after the anchor is passed over, not followed: two
# letters in common is what any two words share.
MIN_RUN = 3


def _letters(word: str) -> str:
    return _NOT_LETTER.sub("", word.lower())


def last_heard(words: list[str], heard: list[str]) -> int | None:
    """The index in ``words`` of the last word surely heard, or ``None``.

    ``words`` are the text's words as the page shows them; ``heard`` are the
    words the STT heard, in order. ``None`` when nothing heard is surely in
    the text — silence, speech that is not this text, or too little of it.
    """
    owner: list[int] = []  # text letter -> index of its word
    text_letters: list[str] = []
    for i, w in enumerate(words):
        for letter in _letters(w):
            text_letters.append(letter)
            owner.append(i)
    heard_letters = list("".join(_letters(w) for w in heard))
    if not text_letters or not heard_letters:
        return None

    matcher = SequenceMatcher(None, text_letters, heard_letters, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size]
    if not blocks:
        return None
    # A phrase the text says twice is matched at its first place in `words`
    # (difflib's tie-break), which is why the page sends only a few words
    # behind the light: the earlier copy is then usually out of the window.
    anchor = max(range(len(blocks)), key=lambda k: blocks[k].size)
    if blocks[anchor].size < MIN_ANCHOR:
        return None

    # The matching blocks come in order on both sides, so the chain is a walk
    # forward from the anchor.
    last = blocks[anchor]
    for block in blocks[anchor + 1 :]:
        if block.size < MIN_RUN:
            continue
        if block.a - (last.a + last.size) > MAX_GAP or block.b - (last.b + last.size) > MAX_GAP:
            break
        last = block
    return owner[last.a + last.size - 1]
