"""The noun espeak reads as a verb, in "record by record".

English has a set of words whose stress tells noun from verb: a ˈrecord, to
reˈcord; an ˈobject, to obˈject. espeak has both readings and picks one from
the word before, not from the grammar: after "then", "now", "we" or "to" it
reads the verb. In "then record by record" that splits one phrase in two —
``ɹᵻkˈɔːɹd baɪ ɹˈɛkɚd`` — and the student reads, and hears, the same word with
two stresses in a row. Changing what "then" does would be wrong more often than
right: *then record the result* is a verb.

A word repeated around one of these prepositions is a noun on both sides,
whatever comes before it: step by step, record after record, word for word,
contract upon contract. espeak already reads the second one as a noun, because
a preposition precedes it, so the fix is to give the first one the second one's
reading. "to" is left out on purpose: after "to" espeak reads the verb, so the
second word is the wrong model, and *refuse to refuse* is a verb on both sides
anyway.

## Why this is its own module

Two G2P calls read the same sentence, and they must agree: the IPA line
(``teaching/phonetics.py``) and Kokoro, which runs its own espeak over the text
it speaks (``voice/kokoro.py``). The promise of the IPA line is that the symbol
the student reads is the sound the voice makes, so a fix on one side only would
have the screen teaching against the loudspeaker. ``voice/`` does not import
``teaching/``, so the fix lives below both. It is text in, text out, stdlib
only, and it knows nothing about espeak beyond the ``read`` callable each side
hands it — which is also what keeps each side's readings comparable with its
own sentence.
"""

from __future__ import annotations

import re
from collections.abc import Callable

_REDUPLICATING = ("by", "after", "upon", "for")
_REDUPLICATION = re.compile(
    r"\b([^\W\d_]+)\s+(" + "|".join(_REDUPLICATING) + r")\s+\1\b",
    re.IGNORECASE,
)

# The punctuation espeak keeps around a word (``preserve_punctuation``), so a
# token's sounds can be compared without it and the punctuation put back.
_TOKEN = re.compile(r"^([\"“‘(\[]*)(.*?)([.,;:!?…\"”’)\]]*)$")


def reduplications(text: str) -> list[tuple[str, str]]:
    """The ``(word, preposition)`` of each *W by W* in ``text``, lower case, once each."""
    seen: list[tuple[str, str]] = []
    for match in _REDUPLICATION.finditer(text):
        found = (match.group(1).lower(), match.group(2).lower())
        if found not in seen:
            seen.append(found)
    return seen


def settle_reduplications(ipa: str, readings: list[str]) -> str:
    """``ipa`` with the verb reading of each *W by W*'s first word made a noun.

    ``readings`` is espeak's reading of ``to W by W`` for each phrase, which
    puts the word after a verb context and after the preposition in one
    utterance: its last three IPA words are the verb reading, the
    preposition, and the noun reading. Where the sentence's IPA has those
    three in a row, the first becomes the noun. Anything else is left as
    espeak gave it — a word read the same both ways (``subject``), a reading
    that came back joined to a neighbour or split, a sentence where the first
    word was already a noun ("the record by record count"). The fix only ever
    swaps one of espeak's own readings for another; it never writes a sound.
    """
    tokens = ipa.split(" ")
    cores = [_TOKEN.match(t).group(2) for t in tokens]  # type: ignore[union-attr]
    for reading in readings:
        parts = reading.split()
        if len(parts) != 4:
            continue
        verb, prep, noun = (_TOKEN.match(t).group(2) for t in parts[1:])  # type: ignore[union-attr]
        for i in range(len(tokens) - 2):
            if cores[i : i + 3] == [verb, prep, noun]:
                lead, _, trail = _TOKEN.match(tokens[i]).groups()  # type: ignore[union-attr]
                tokens[i] = f"{lead}{noun}{trail}"
                cores[i] = noun
    return " ".join(tokens)


def read_nouns_as_nouns(text: str, ipa: str, read: Callable[[str], str]) -> str:
    """``ipa``, espeak's reading of ``text``, with every *W by W* read as nouns.

    ``read`` is the same G2P that produced ``ipa``, called once per phrase and
    only when ``text`` has one — so a sentence without a reduplication costs a
    regular expression and nothing else.
    """
    phrases = reduplications(text)
    if not phrases:
        return ipa
    return settle_reduplications(ipa, [read(f"to {w} {p} {w}") for w, p in phrases])


__all__ = ["read_nouns_as_nouns", "reduplications", "settle_reduplications"]
