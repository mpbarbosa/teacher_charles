"""What a word means, in English, and how it goes into Portuguese.

The card that opens when the pointer rests on a word of a lesson's full text:
the word's senses, each with an English definition, an example and the
Portuguese words for it — the shape of a bilingual learner's dictionary entry.
It is the fourth sibling of the IPA line, the syntax view and the correction:
a fact about the written words, read from a table and never from the model.

## Why a lexicon, and not the model

For grammar's reason. A 4B model asked what *chamber* means in Portuguese will
answer, fluently, and nothing downstream can tell a right gloss from an
invented one. ``data/vocab/items.yaml`` already measured it on *fodder*: one
answer in four got the figurative sense wrong, and one invented an example that
is not English.

## Why WordNet, and not the dictionary the card is modelled on

Two open lexicons linked sense by sense, through the Interlingual Index:

- **Open English WordNet 2024** (CC BY 4.0) for the definitions, the examples
  and the irregular forms — *went* finds *go*, *children* finds *child*. Its
  older ancestor in the Open Multilingual Wordnet has no irregular forms.
- **OpenWN-PT 1.0** (CC BY-SA 4.0) for the Portuguese. Measured against the
  English: 52,540 of 120,630 senses have Portuguese words, against 43,808 in
  the OMW 1.4 copy of the same project — and the difference falls on senses
  that matter here (the *legislative assembly* sense of *chamber* is
  *câmara* in one and missing in the other).

A commercial dictionary is better written. It is also somebody else's text,
and this repository is public: the same line as ``corpus/README.md``. WordNet is
downloaded to the student's machine by ``scripts/fetch_dictionary.py`` and
never versioned — it is re-downloadable, hence a cache, like the voice weights.

## What the card is not

It does not know which sense the text means. WordNet orders a word's senses by
how often they were tagged in a corpus decades ago, and the card shows that
order. *This chamber* in a speech to the Senate is the fourth sense of
*chamber*, under *an enclosed space* and *a room where a judge transacts
business*. Picking the sense from the sentence is word-sense disambiguation, a
model's job and a model's error rate, so ``NOTE`` travels with every result and
the page prints it — the absence of a better guess is not a verdict.

Coverage is open-class words only: WordNet has nouns, verbs, adjectives and
adverbs, so *the*, *of* and *we* get no entry. Proper nouns of a fiction —
*Ghorman*, *Palpatine* — get none either. Both are answered as a miss, and the
miss says why.

## Why its own SQLite file, and not WordNet's

The fetch script uses the ``wn`` package to download and join the two lexicons,
then writes the few columns the card reads into a file of ours: 270 MB of
``wn`` working data becomes a lookup a fraction of that size, read here with
the stdlib's ``sqlite3`` — no ``wn`` at runtime, and none of its process-wide
configuration. Opening it read-only matters for a second reason: ``wn`` creates
an empty database when the file is missing, and a status check that writes a
file is the side effect ``voice/base.Loadable`` exists to forbid.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings

log = logging.getLogger(__name__)

# Bumped when the tables below change, so a file built by an older script is
# refused with a reason instead of failing on a missing column mid-lookup.
SCHEMA_VERSION = "1"

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE word (id INTEGER PRIMARY KEY, lemma TEXT NOT NULL, pos TEXT NOT NULL);
-- Every spelling that leads to a word: the lemma itself and its irregular
-- forms, lower-cased. Regular inflections are not stored; ``base_forms``
-- strips them at lookup time, as WordNet's own morphy does.
CREATE TABLE form (form TEXT NOT NULL, word_id INTEGER NOT NULL REFERENCES word(id));
CREATE TABLE sense (
    word_id INTEGER NOT NULL REFERENCES word(id),
    rank INTEGER NOT NULL,
    definition TEXT NOT NULL,
    example TEXT,
    -- A JSON list of Portuguese words, possibly empty.
    portuguese TEXT NOT NULL
);
CREATE INDEX form_by_form ON form(form);
CREATE INDEX sense_by_word ON sense(word_id, rank);
"""

# How much of an entry the card shows. A bilingual dictionary prints every
# sense; a card under the pointer has room for a few, and says how many more.
MAX_SENSES = 5
MAX_PORTUGUESE = 4

# Longer than any English word worth a card, short enough that the route
# cannot be handed a paragraph.
MAX_WORD_CHARS = 64

# WordNet's parts of speech, in the words the student reads. ``s`` is WordNet's
# "satellite adjective", a bookkeeping distinction no learner needs.
PARTS_OF_SPEECH = {
    "n": "noun",
    "v": "verb",
    "a": "adjective",
    "s": "adjective",
    "r": "adverb",
}

# The order a word's parts of speech go on the card: WordNet's own (noun,
# verb, adjective, adverb). Not the file's row order, which follows the
# lexicon's ids and so the alphabet of their pos letter — that put *garden*
# the adjective ("the usual type") above *garden* the noun.
POS_ORDER = {"n": 0, "v": 1, "a": 2, "s": 2, "r": 3}

# Goes with every result and the page prints it, for ``syntax.NOTE``'s reason.
NOTE = "Senses in the dictionary's order: the card does not know which one this text means."

# What a miss says instead. Most misses are not gaps but WordNet's scope, and
# a card that only said "not found" would read as a broken lookup.
MISS = (
    "No entry. WordNet lists nouns, verbs, adjectives and adverbs, so small words "
    "such as \u201cthe\u201d, \u201cof\u201d or \u201cwe\u201d, contractions and names have none."
)

# Credit for both lexicons, on every card. OpenWN-PT's licence asks for it, and
# a gloss the student cannot trace to a source is the thing this module avoids.
SOURCES = (
    {"name": "Open English WordNet 2024", "url": "https://en-word.net/", "licence": "CC BY 4.0"},
    {
        "name": "OpenWN-PT",
        "url": "https://github.com/own-pt/openWordnet-PT",
        "licence": "CC BY-SA 4.0",
    },
)

# WordNet's detachment rules (morphy), per part of speech: the suffix a regular
# inflection adds and what the base ends in instead. Irregular forms are not
# rules, they are rows in ``form`` — that is what OEWN brings and OMW lacks.
SUFFIXES: dict[str, tuple[tuple[str, str], ...]] = {
    "n": (
        ("s", ""),
        ("ses", "s"),
        ("xes", "x"),
        ("zes", "z"),
        ("ches", "ch"),
        ("shes", "sh"),
        ("men", "man"),
        ("ies", "y"),
    ),
    "v": (
        ("s", ""),
        ("ies", "y"),
        ("es", "e"),
        ("es", ""),
        ("ed", "e"),
        ("ed", ""),
        ("ing", "e"),
        ("ing", ""),
    ),
    "a": (("er", ""), ("est", ""), ("er", "e"), ("est", "e")),
}

# Suffixes after which English doubles a final consonant: *stopped*, *bigger*.
# morphy leaves those to its exception lists; one extra candidate covers them.
DOUBLING = frozenset({"ed", "ing", "er", "est"})

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})
_EDGES = re.compile(r"^[^\w']+|[^\w']+$")


class DictionaryUnavailable(Exception):
    """The lookup cannot be used: file missing, unreadable, or from an older build."""


# --- What a lookup returns -------------------------------------------------- #


@dataclass(frozen=True)
class Sense:
    definition: str
    example: str | None
    portuguese: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "definition": self.definition,
            "example": self.example,
            "portuguese": list(self.portuguese),
        }


@dataclass(frozen=True)
class Headword:
    """One block of the card: a lemma in one part of speech, and its senses."""

    lemma: str
    pos: str
    senses: tuple[Sense, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "lemma": self.lemma,
            "pos": PARTS_OF_SPEECH.get(self.pos, self.pos),
            "senses": [s.as_dict() for s in self.senses],
        }


@dataclass(frozen=True)
class Entry:
    """What the card shows for one word under the pointer.

    ``word`` is the word as looked up (normalised), and the headwords may be
    other spellings of it: *voices* is shown as *voice*. ``more`` counts the
    senses cut by ``MAX_SENSES``. No headwords is a miss, and a miss is an
    answer — the page draws it — not an error.
    """

    word: str
    headwords: tuple[Headword, ...] = ()
    more: int = 0

    @property
    def found(self) -> bool:
        return bool(self.headwords)

    def as_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "found": self.found,
            "headwords": [h.as_dict() for h in self.headwords],
            "more": self.more,
            "note": NOTE if self.found else MISS,
            "sources": list(SOURCES),
        }


# --- From the word on the page to the spellings worth looking up ------------ #


def normalise(token: str) -> str:
    """The word under the pointer, as the lookup wants it.

    The page's words carry their punctuation (*chamber,*), and a speech typed
    by a person carries curly apostrophes. Lower-cased, because the lexicon's
    lemmas are matched without case: *Senate* at the start of a sentence is
    still *senate*. A possessive loses its ending — *nation's* is *nation* —
    while a contraction keeps its apostrophe and, having no entry, is a miss.
    """
    return _as_written(token).lower()


def _as_written(token: str) -> str:
    """``normalise`` without the lower-casing, for ordering by case."""
    word = _EDGES.sub("", token.translate(_APOSTROPHES)).strip("'")
    if word.lower().endswith("'s"):
        word = word[:-2]
    return word


def base_forms(word: str) -> list[tuple[str, str]]:
    """Every (base, part of speech) a regular inflection could have come from.

    Candidates, not answers: *singer* yields *sing* as an adjective, and the
    lookup throws away the ones the lexicon does not have in that part of
    speech. Order follows ``SUFFIXES``, so the answers come back in morphy's
    order too.
    """
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for pos, rules in SUFFIXES.items():
        for suffix, ending in rules:
            if not word.endswith(suffix) or len(word) <= len(suffix):
                continue
            stem = word[: -len(suffix)]
            bases = [stem + ending]
            if suffix in DOUBLING and len(stem) > 2 and stem[-1] == stem[-2]:
                bases.append(stem[:-1])
            for base in bases:
                if len(base) > 1 and (base, pos) not in seen:
                    seen.add((base, pos))
                    out.append((base, pos))
    return out


def spellings(word: str) -> list[str]:
    """The written variants of a word to try as they are: *pre-war*, *pre war*.

    WordNet writes compounds with a space and a few with the hyphen, and a
    student's text may write them either way.
    """
    out = [word]
    if "-" in word:
        out += [word.replace("-", " "), word.replace("-", "")]
    return out


# --- The lookup ------------------------------------------------------------- #


class Dictionary:
    """The fetched lookup file, opened when somebody first hovers a word.

    ``installed`` asks the filesystem and nothing more, and ``ready`` never
    opens the file: the page, ``/health`` and the route ask them on every
    request. ``lookup`` is the one place that raises, with a typed error,
    because its caller is a route that must tell the page why the card is
    empty.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn: sqlite3.Connection | None = None
        self._meta: dict[str, str] = {}
        self._last_error: str | None = None
        # One connection, read from executor threads: sqlite3's own check
        # is off, and this lock is what makes that safe.
        self._lock = threading.Lock()

    # --- state ------------------------------------------------------------ #

    @property
    def installed(self) -> bool:
        """Is the lookup file on disk? Opens nothing."""
        return self.path.is_file()

    @property
    def ready(self) -> bool:
        """Is the file open? Forces no open."""
        return self._conn is not None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def meta(self) -> dict[str, str]:
        """What the fetch script recorded about the build; empty until open."""
        return dict(self._meta)

    # --- open ------------------------------------------------------------- #

    def _open(self) -> sqlite3.Connection:
        if not self.installed:
            raise DictionaryUnavailable(
                f"no dictionary at {self.path}; "
                "run `uv run --with wn==1.1.1 python scripts/fetch_dictionary.py`"
            )
        try:
            # Read-only by URI: a file that vanished between the check and
            # here must fail, not be recreated empty.
            conn = sqlite3.connect(
                f"{self.path.resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False
            )
            meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        except sqlite3.Error as exc:
            raise DictionaryUnavailable(f"dictionary at {self.path} is unreadable: {exc}") from exc
        if meta.get("schema") != SCHEMA_VERSION:
            conn.close()
            raise DictionaryUnavailable(
                f"dictionary at {self.path} was built by another version of the fetch "
                f"script (schema {meta.get('schema')!r}, this code reads {SCHEMA_VERSION!r}); "
                "run `uv run --with wn==1.1.1 python scripts/fetch_dictionary.py` again"
            )
        self._meta = meta
        return conn

    def _ensure(self) -> sqlite3.Connection:
        if self._conn is None:
            try:
                self._conn = self._open()
            except DictionaryUnavailable as exc:
                self._last_error = str(exc)
                raise
            self._last_error = None
        return self._conn

    async def warm_up(self) -> None:
        """Opens the file before the first hover. Never propagates."""
        try:
            await asyncio.to_thread(self._ensure_locked)
        except DictionaryUnavailable as exc:
            log.warning("word cards unavailable, the lesson goes on without them: %s", exc)

    def _ensure_locked(self) -> None:
        with self._lock:
            self._ensure()

    # --- lookup ----------------------------------------------------------- #

    def lookup(self, token: str) -> Entry:
        """The card for one word of the page, or a miss. Raises only when unusable."""
        word = normalise(token)
        if not word:
            return Entry(word="")
        with self._lock:
            conn = self._ensure()
            ids = self._word_ids(conn, word, _as_written(token))
            return self._entry(conn, word, ids)

    def _word_ids(self, conn: sqlite3.Connection, word: str, written: str) -> list[int]:
        """The lexicon's words for this spelling: as written first, then inflected.

        As written covers the lemma and the irregular forms in one query — that
        is why both are rows of ``form``. Among those, a lemma in the case the
        text used goes first: *nation's* in lower case is the noun before the
        proper name *Nation*, and *Senate* with a capital mid-sentence is the
        institution. The inflected ones must exist in the part of speech the
        rule belongs to: *-ing* strips to a verb only.

        A word the text writes in lower case drops every lemma with a capital
        in it. That is evidence, not taste: running text does not write a
        name or a symbol in lower case, so *as* is never *AS* (American Samoa)
        or *As* (arsenic), and those blocks would push the real senses off the
        card. A capital proves less — it may only open the sentence — so a
        capitalised word keeps both readings, its own case first.
        """
        exact: list[tuple[int, str]] = []
        for spelling in spellings(word):
            exact += conn.execute(
                "SELECT w.id, w.lemma, w.pos FROM form f JOIN word w ON w.id = f.word_id "
                "WHERE f.form = ? ORDER BY w.id",
                (spelling,),
            ).fetchall()

        # Case first — the text's own case, then lower case (a capital may
        # only open the sentence), then any other — and then WordNet's order
        # of parts of speech. Stable, so within those the lexicon's order
        # stands.
        def rank(row: tuple[int, str, str]) -> tuple[int, int]:
            lemma, pos = row[1], row[2]
            case = 0 if lemma == written else 1 if lemma == lemma.lower() else 2
            return case, POS_ORDER.get(pos, len(POS_ORDER))

        exact.sort(key=rank)
        found = [(word_id, lemma) for word_id, lemma, _ in exact]
        for base, pos in base_forms(word):
            found += conn.execute(
                "SELECT w.id, w.lemma FROM form f JOIN word w ON w.id = f.word_id "
                "WHERE f.form = ? AND w.lemma = f.form COLLATE NOCASE "
                "AND (w.pos = ? OR (? = 'a' AND w.pos = 's')) ORDER BY w.id",
                (base, pos, pos),
            ).fetchall()
        if written == written.lower():
            found = [row for row in found if row[1] == row[1].lower()]
        return list(dict.fromkeys(word_id for word_id, _ in found))

    def _entry(self, conn: sqlite3.Connection, word: str, ids: list[int]) -> Entry:
        headwords: list[Headword] = []
        shown = total = 0
        for word_id in ids:
            lemma, pos = conn.execute(
                "SELECT lemma, pos FROM word WHERE id = ?", (word_id,)
            ).fetchone()
            rows = conn.execute(
                "SELECT definition, example, portuguese FROM sense WHERE word_id = ? ORDER BY rank",
                (word_id,),
            ).fetchall()
            total += len(rows)
            room = MAX_SENSES - shown
            if room <= 0 or not rows:
                continue
            senses = tuple(
                Sense(definition, example, tuple(json.loads(pt))[:MAX_PORTUGUESE])
                for definition, example, pt in rows[:room]
            )
            shown += len(senses)
            # Two WordNet words that print the same — *better* as an adjective
            # and as a satellite — are one block on the card.
            if headwords and (headwords[-1].lemma, PARTS_OF_SPEECH.get(headwords[-1].pos)) == (
                lemma,
                PARTS_OF_SPEECH.get(pos),
            ):
                last = headwords.pop()
                senses = last.senses + senses
            headwords.append(Headword(lemma=lemma, pos=pos, senses=senses))
        return Entry(word=word, headwords=tuple(headwords), more=total - shown)


# --- Writing the file (the fetch script's half) ----------------------------- #


@dataclass(frozen=True)
class BuiltWord:
    """One lexicon word as the fetch script hands it over."""

    lemma: str
    pos: str
    forms: tuple[str, ...] = ()
    senses: tuple[Sense, ...] = field(default_factory=tuple)


def write_lookup(path: Path, words: Iterable[BuiltWord], meta: dict[str, str]) -> int:
    """Writes the lookup file the card reads, and returns how many words it holds.

    Here and not in the script so that the tables are defined in one place,
    next to the queries that read them, and a test can build a small file
    through the same code. The file is written beside its final name and moved
    over it at the end: an interrupted build must never leave a file that
    ``installed`` would take for a finished one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.unlink(missing_ok=True)
    conn = sqlite3.connect(part)
    count = 0
    try:
        conn.executescript(SCHEMA)
        for word in words:
            if not word.senses:
                continue
            cur = conn.execute(
                "INSERT INTO word (lemma, pos) VALUES (?, ?)", (word.lemma, word.pos)
            )
            word_id = cur.lastrowid
            forms = dict.fromkeys(f.lower() for f in (word.lemma, *word.forms))
            conn.executemany(
                "INSERT INTO form (form, word_id) VALUES (?, ?)", [(f, word_id) for f in forms]
            )
            conn.executemany(
                "INSERT INTO sense (word_id, rank, definition, example, portuguese) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        word_id,
                        rank,
                        s.definition,
                        s.example,
                        json.dumps(list(s.portuguese), ensure_ascii=False),
                    )
                    for rank, s in enumerate(word.senses)
                ],
            )
            count += 1
        conn.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            [("schema", SCHEMA_VERSION), ("words", str(count)), *meta.items()],
        )
        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()
    os.replace(part, path)
    return count


def build_dictionary(settings: Settings) -> Dictionary | None:
    """``None`` with the cards off; a ``Dictionary`` otherwise, file or not.

    Building opens nothing. Whether the file is there decides whether the
    card is on screen (``on_screen``), and ``/health`` says why when it is not.
    """
    if not settings.dictionary_enabled:
        return None
    return Dictionary(settings.resolved_dictionary_path)


def on_screen(dictionary: Dictionary | None) -> bool:
    """Is the hover card on the page? Asked for *and* fetched.

    One function, for ``syntax.on_screen``'s reason: the template, the page's
    script and ``/health`` must agree, and three conditions written three
    times is how they stop agreeing.
    """
    return dictionary is not None and dictionary.installed
