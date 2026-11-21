"""What the words the student asked about sound like — from espeak, not the model.

The persona has said it since the *pat*/*bat* episode: *do not invent facts
about sounds.* Measured, the sentence does not hold. Asked whether the *th* of
*think* is the sound of the *th* of *this*, both ``qwen2.5:3b-instruct`` and
``qwen3:4b-instruct`` said yes, six answers out of six, and the 4B wrote IPA
to prove it (``docs/plano-fase-1.md`` §5.5). They are different sounds — *θ*
and *ð* — and the IPA line under the same bubble said so.

This is the rule the grammar follows, one layer down, and the cure is the one
the grammar got. The curated hint gives the model a reviewed explanation
instead of letting it improvise one. Here the app gives it the words'
sounds, as espeak reads them — the same G2P behind the IPA line and the
voice, so the screen, the loudspeaker and the teacher cannot disagree — and
says where the two words differ. The note rides inside the student's message,
next to the point of generation, for the reason every note does
(``TurnRunner._with_notes``).

## What it knows, and what it does not

The sound units are the curated table's (``data/phonetics/symbols.yaml``), so
*tʃ* is one sound and not two letters. The comparison is an alignment of two
words' units — which are shared, which one replaces which — and it is a
matter of the table, not of judgement. On top of that, only two kinds of fact
are curated: a consonant pair that differs in voicing alone (*θ*/*ð*, *s*/*z*,
…), and the two vowel pairs that Brazilian learners merge (*ɪ*/*iː*, *ʊ*/*uː*).
Anything else is named as *different* and left there. It is the reading of the
written word in American English, not of how anybody said it.

## When it speaks

When the message asks about sound — *sound*, *pronounce*, *rhyme* — and names
words, in quotes or after *pronounce*. And for *the difference between X and
Y* with no sound word in it only when X and Y are a minimal pair, one sound
apart. *Ship* and *sheep* are a question about sounds even when the student
does not say so, while *make* and *do* are a question about meaning, and a
note about their vowels would push the teacher off it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from .ipa_symbols import SymbolTable
from .phonetics import Phonetics

# Same mouth, same place; one makes the vocal cords vibrate and one does not.
# Voiceless first.
VOICING_PAIRS: tuple[tuple[str, str], ...] = (
    ("p", "b"),
    ("t", "d"),
    ("k", "ɡ"),
    ("f", "v"),
    ("θ", "ð"),
    ("s", "z"),
    ("ʃ", "ʒ"),
    ("tʃ", "dʒ"),
)

# The vowel pairs Portuguese has one sound for, short first.
VOWEL_PAIRS: tuple[tuple[str, str], ...] = (
    ("ɪ", "iː"),
    ("ʊ", "uː"),
)

MAX_WORDS = 4

_ASKS_ABOUT_SOUND = re.compile(
    r"\b(?:sounds?|sounding|pronounc\w*|pronunciation|rhym\w*|homophones?)\b",
    re.IGNORECASE,
)
_ASKS_FOR_DIFFERENCE = re.compile(r"\b(?:difference|differ|same|different)\b", re.IGNORECASE)

# Quoted spans: what the student points at. Guards on both sides keep the
# apostrophe inside *I'm* or *Let's* from opening one.
_QUOTED = re.compile(r"(?<!\w)[\"'“‘*]([^\"'“”‘’*\n]{1,40})[\"'”’*](?!\w)")
_AFTER_PRONOUNCE = re.compile(r"\bpronounc\w*\s+(?:the\s+word\s+)?([A-Za-z]+)", re.IGNORECASE)
_BETWEEN = re.compile(r"\bbetween\s+([A-Za-z]+)\s+and\s+([A-Za-z]+)", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_VOWEL_LETTER = re.compile(r"[aeiouy]", re.IGNORECASE)

# Words the student uses to ask, not words they ask about.
_NOT_TARGETS = frozenset({"the", "and", "word", "words", "sound", "sounds", "letter"})


def named_words(text: str) -> list[str]:
    """The words the message points at, in order, at most ``MAX_WORDS``.

    Quoted words first, then the word after *pronounce* and the two after
    *between*. A quoted letter group with no vowel — *'th'*, *'s'* — is a
    spelling, not a word, and espeak would read it as letter names, so it is
    dropped: *the 'th' in 'think'* names *think*.
    """
    found: list[str] = []
    for span in _QUOTED.findall(text):
        found.extend(_WORD.findall(span))
    for match in _AFTER_PRONOUNCE.finditer(text):
        found.append(match.group(1))
    for match in _BETWEEN.finditer(text):
        found.extend(match.groups())

    words: list[str] = []
    for word in found:
        lowered = word.lower()
        if len(word) < 2 or not _VOWEL_LETTER.search(word) or lowered in _NOT_TARGETS:
            continue
        if lowered not in (w.lower() for w in words):
            words.append(word)
    return words[:MAX_WORDS]


@dataclass(frozen=True)
class Reading:
    word: str
    units: tuple[str, ...]


def _pair_fact(a: str, b: str) -> str:
    """The curated fact about two single sounds, or an empty string."""
    for voiceless, voiced in VOICING_PAIRS:
        if {a, b} == {voiceless, voiced}:
            return (
                f"{voiceless} and {voiced} are made the same way in the mouth; "
                f"{voiced} is voiced (the vocal cords vibrate) and {voiceless} is not."
            )
    for short, long_ in VOWEL_PAIRS:
        if {a, b} == {short, long_}:
            return f"{short} is short and relaxed, {long_} is longer and tenser."
    return ""


class SoundFacts:
    """Builds the note from espeak's reading and the curated symbol table."""

    def __init__(self, phonetics: Phonetics, symbols: SymbolTable) -> None:
        self._phonetics = phonetics
        self._symbols = symbols
        self._examples = {s.symbol: s.example for s in symbols.symbols}

    def read(self, word: str) -> Reading | None:
        """espeak's sounds for one word, or ``None`` when there are none."""
        ipa = self._phonetics.transcribe(word)
        units, _unknown = self._symbols.segment(ipa)
        return Reading(word, tuple(units)) if units else None

    def _as_in(self, unit: str) -> str:
        example = self._examples.get(unit)
        return f'{unit} (as in "{example}")' if example else unit

    def _compare(self, first: Reading, second: Reading) -> list[str]:
        lines = []
        matcher = SequenceMatcher(a=first.units, b=second.units, autojunk=False)
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            left = " ".join(first.units[i1:i2])
            right = " ".join(second.units[j1:j2])
            if op == "equal":
                lines.append(f"- the same {left} in both.")
            elif op == "replace":
                # Same length on both sides: one sound for one sound, each pair
                # on its own line, so *oʊ z* against *aɪ s* still says that z
                # and s differ only in voicing.
                if i2 - i1 == j2 - j1:
                    for a, b in zip(first.units[i1:i2], second.units[j1:j2], strict=True):
                        line = (
                            f'- different: {self._as_in(a)} in "{first.word}" where '
                            f'"{second.word}" has {self._as_in(b)}.'
                        )
                        fact = _pair_fact(a, b)
                        lines.append(f"{line} {fact}" if fact else line)
                else:
                    lines.append(
                        f'- different: {left} in "{first.word}" where "{second.word}" has {right}.'
                    )
            elif op == "delete":
                lines.append(f'- only "{first.word}" has {left}.')
            else:
                lines.append(f'- only "{second.word}" has {right}.')
        return lines

    def is_minimal_pair(self, first: Reading, second: Reading) -> bool:
        """Two words one sound apart: same length, one unit swapped."""
        if len(first.units) != len(second.units):
            return False
        return sum(a != b for a, b in zip(first.units, second.units, strict=True)) == 1

    def note(self, student_text: str) -> str:
        """The note for this turn, or an empty string."""
        words = named_words(student_text)
        if not words:
            return ""
        readings = [r for r in (self.read(w) for w in words) if r is not None]
        if not readings:
            return ""

        if not _ASKS_ABOUT_SOUND.search(student_text):
            # No sound word: only a minimal pair, asked about as a difference,
            # is a sound question — see the module's docstring.
            if not (
                len(readings) >= 2
                and _ASKS_FOR_DIFFERENCE.search(student_text)
                and self.is_minimal_pair(readings[0], readings[1])
            ):
                return ""

        lines = [
            "The student is asking how words sound. This is how the app's "
            "pronunciation dictionary reads them (espeak, American English — the "
            "same reading as the IPA line under each message and as the voice). "
            "Say nothing about these sounds that this does not support. If it "
            "does not answer the question, say the words and leave it there.",
        ]
        for reading in readings:
            lines.append(f'- "{reading.word}": {" ".join(reading.units)}')
        if len(readings) >= 2:
            first, second = readings[0], readings[1]
            lines.append(f'Comparing "{first.word}" and "{second.word}", sound by sound:')
            lines.extend(self._compare(first, second))
        lines.append(
            "Explain it in plain words; point the student at the IPA line instead "
            "of typing IPA yourself."
        )
        return "\n".join(lines)


def build_sound_facts(
    phonetics: Phonetics | None, symbols: SymbolTable | None
) -> SoundFacts | None:
    """The note's builder, or ``None`` without the G2P or without the table."""
    if phonetics is None or symbols is None:
        return None
    return SoundFacts(phonetics, symbols)


__all__ = [
    "MAX_WORDS",
    "VOICING_PAIRS",
    "VOWEL_PAIRS",
    "Reading",
    "SoundFacts",
    "build_sound_facts",
    "named_words",
]
