"""The IPA symbols a text uses, each with a word to hear it in.

## Why a table, and not the characters of the IPA line

Because a sound is not a character. espeak writes the vowel of *face* as
``eɪ`` and the first sound of *church* as ``tʃ``: two characters, one sound.
A chart built from characters would list ``e`` and ``ʃ`` for those words and
teach the student two sounds the voice never made. So the units are curated in
``data/phonetics/symbols.yaml`` and an IPA string is cut longest-first against
them, which is what makes ``tʃ`` one tile and ``t`` another.

## Why the example word is curated and its IPA is not

The example is a choice — *teacher* for ``ɚ``, *button* for ``ʔ`` — and a
choice is data, reviewed and tested item by item like the contrastive base.
Its IPA is not a choice: it is espeak's, computed when the page is served,
for the rule ``teaching/phonetics.py`` states — the symbol on screen is the
symbol the voice pronounces. ``tests/test_ipa_symbols.py`` holds each example
to that: espeak's reading of the word, cut by this module, has to contain the
symbol it illustrates.

Text in, text out, no audio, which is why this lives beside ``phonetics.py``
in ``teaching/``. Playing the example is the page's job, through the same
``/api/tts`` the word speakers use.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

KINDS = ("vowel", "diphthong", "consonant")

# Not sounds: stress marks, and what espeak keeps of the punctuation.
_SKIPPED = set("ˈˌ.,;:!?—–-'\"()…")


class SymbolError(ValueError):
    """The symbol table is malformed: the chart is left out, not guessed."""


@dataclass(frozen=True)
class Symbol:
    symbol: str
    kind: str
    example: str


@dataclass(frozen=True)
class SymbolTable:
    symbols: tuple[Symbol, ...]

    def __len__(self) -> int:
        return len(self.symbols)

    def segment(self, ipa: str) -> tuple[list[str], list[str]]:
        """``ipa`` cut into the table's symbols, and what matched none.

        Longest match first, so ``tʃ`` is one unit and not ``t`` + ``ʃ``.
        Whitespace, stress marks and punctuation are skipped. Anything else
        that matches no symbol is returned in the second list, one character
        at a time, rather than dropped: an unknown sound is a gap in the
        table, and the test over the lessons' own text is what finds it.
        """
        by_length = sorted((s.symbol for s in self.symbols), key=len, reverse=True)
        units: list[str] = []
        unknown: list[str] = []
        i = 0
        while i < len(ipa):
            ch = ipa[i]
            if ch.isspace() or ch in _SKIPPED:
                i += 1
                continue
            match = next((s for s in by_length if ipa.startswith(s, i)), None)
            if match is None:
                unknown.append(ch)
                i += 1
            else:
                units.append(match)
                i += len(match)
        return units, unknown

    def chart(self, ipa_texts: list[str]) -> list[dict[str, object]]:
        """The symbols ``ipa_texts`` use, grouped by kind, in the table's order.

        Each group is ``{"kind", "symbols": [{"symbol", "example", "count"}]}``;
        a kind the texts never use is left out, and so is an empty chart's
        every group. ``count`` is how often the symbol occurs, for a title.
        """
        counts: Counter[str] = Counter()
        for text in ipa_texts:
            counts.update(self.segment(text)[0])
        groups = []
        for kind in KINDS:
            rows = [
                {"symbol": s.symbol, "example": s.example, "count": counts[s.symbol]}
                for s in self.symbols
                if s.kind == kind and counts[s.symbol]
            ]
            if rows:
                groups.append({"kind": kind, "symbols": rows})
        return groups


def load_symbols(path: Path) -> SymbolTable:
    """The table in ``path`` (``data/phonetics/symbols.yaml``), checked."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SymbolError(f"{path}: {exc}") from exc
    items = raw.get("symbols") if isinstance(raw, dict) else None
    if not isinstance(items, list) or not items:
        raise SymbolError(f"{path}: no 'symbols' list")
    symbols: list[Symbol] = []
    seen: set[str] = set()
    for n, item in enumerate(items):
        if not isinstance(item, dict):
            raise SymbolError(f"{path}: item {n} is not a mapping")
        symbol, kind, example = item.get("symbol"), item.get("kind"), item.get("example")
        if not (isinstance(symbol, str) and symbol.strip()):
            raise SymbolError(f"{path}: item {n} has no symbol")
        if kind not in KINDS:
            raise SymbolError(f"{path}: {symbol!r} has kind {kind!r}, not one of {KINDS}")
        if not (isinstance(example, str) and example.strip()):
            raise SymbolError(f"{path}: {symbol!r} has no example word")
        if symbol in seen:
            raise SymbolError(f"{path}: {symbol!r} is listed twice")
        seen.add(symbol)
        symbols.append(Symbol(symbol=symbol, kind=kind, example=example.strip()))
    return SymbolTable(tuple(symbols))
