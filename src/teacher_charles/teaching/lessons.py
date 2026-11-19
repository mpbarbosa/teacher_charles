"""Lessons: a curated reading of an authentic text, laid out by linguistic level.

A lesson is the third kind of curated content, next to ``contrastive.py`` and
``vocabulary.py``, and it follows the project's principle one step further:
**the analysis the student reads is data a curator wrote and a test checked,
not prose the model improvised.** Charles never sees a lesson. The page shows
it as it is, and the only thing computed at serve time is the IPA, which comes
from espeak like every other IPA on screen (``teaching/phonetics.py``).

## The source text is linked, never copied

A lesson is *about* somebody else's text — a speech, an article, a scene. The
repository is public, so the rule in ``corpus/README.md`` applies whole:
derive into ``data/``, never copy into it. ``Source`` therefore has a URL and
a word count and **no field for the text**; ``extra="forbid"`` refuses a
``text:`` key the way the vocabulary base refuses an ``ipa:`` key. The student
reads the original where its owner published it.

### Unless this machine holds a copy

Whoever runs the project may own a copy of the text, the way they own the
subtitles in ``corpus/subtitles/``. If ``corpus/lessons/<slug>.txt`` exists,
the page shows it whole above the analysis (``read_local_text``). That is the
same rule, not an exception to it: the file is outside version control, so a
clone shows the link and only the machine that holds the copy shows the text.
It is read per request and never cached into the ``Lesson``, so dropping the
file in needs no restart, and it never enters ``/api/lessons/<slug>`` — that
payload is the curated lesson, and the copy is not ours to serve.

What a lesson may carry from the source is the short phrase an analysis has to
point at — *heavy heart*, *took place* against *happened*. Those live in
``Point.cites`` and nowhere else, and two limits hold them to quotation and
keep them from becoming a copy by instalments:

- each citation is at most ``MAX_CITE_WORDS`` words, and
- all citations together are at most ``CITE_BUDGET`` of the source's length.

Both are checked at load time, so a lesson that drifts towards reproducing its
source fails to load instead of being published. ``tests/test_lessons.py``
deletes each guard in turn to prove a test is holding it.

## Why the aspects are a fixed list

``Aspect`` is the six levels a reading is analysed at, in teaching order: from
the word (lexicon, vocabulary) through the sentence (grammar, syntax) to
meaning (semantics) and sound (phonetics). A free-text heading would let two
lessons call the same level by two names, and the page orders sections by
this list, not by the order the curator happened to type.

## What this module does not do

It does not decide which lesson a student sees, and it does not track whether
a lesson was read. Every lesson is on the shelf for every student; the CEFR
label is shown, not enforced, because reading above one's level with the
analysis beside it is the point of a lesson, while a *word* above one's level
is a lost lesson (``VocabularyBase.at_or_below``). Practice prompts hand a
starter sentence to the chat input and stop there: what the student writes
then goes through the ordinary turn, correction and all.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from ..config import get_settings
from ..store.models import CefrLevel
from .contrastive import RULE_ID_PATTERN
from .vocabulary import ITEM_ID_PATTERN as VOCAB_ID_PATTERN

LESSON_ID_PATTERN = r"^lesson\.[a-z0-9]+(?:-[a-z0-9]+)*$"

# A citation longer than this is no longer pointing at a phrase; it is
# reproducing a sentence.
MAX_CITE_WORDS = 5

# All citations together may not exceed this fraction of the source. A tenth
# of a 269-word speech is 26 words — room for the phrases an analysis names,
# and no room for the passage around them.
CITE_BUDGET = 0.10

Aspect = Literal["lexicon", "vocabulary", "grammar", "syntax", "semantics", "phonetics"]

# Teaching order: word, sentence, meaning, sound. The page sorts by this.
ASPECTS: tuple[Aspect, ...] = (
    "lexicon",
    "vocabulary",
    "grammar",
    "syntax",
    "semantics",
    "phonetics",
)


class LessonError(ValueError):
    """The shelf did not load. Carries every problem, not just the first."""


def word_count(text: str) -> int:
    """Words as a reader counts them: runs of letters, digits and apostrophes."""
    return len(re.findall(r"[A-Za-z0-9’']+", text))


class Source(BaseModel):
    """Where the analysed text lives — a pointer, never the text itself."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # The piece the student opens: headline, outlet, byline, date.
    title: str = Field(min_length=1)
    publication: str = Field(min_length=1)
    author: str = Field(min_length=1)
    published: date
    url: str = Field(pattern=r"^https://\S+$")

    # What the piece is about, when that is a work of its own (a show, a
    # speech): shown under the title so the student knows what they will read.
    work: str | None = None

    # Length of the analysed text, in words. It is the denominator of the
    # citation budget, so it is required: without it the budget means nothing.
    words: int = Field(gt=0)


# YouTube's video ids: eleven characters from the URL-safe base64 alphabet.
YOUTUBE_ID_PATTERN = r"^[A-Za-z0-9_-]{11}$"


class Video(BaseModel):
    """The work performed, embedded on the page from the channel that owns it.

    An id, never a URL: the address is built at render time, so a pasted
    link's tracking parameters (``?si=``, ``&t=``) cannot reach the repository.
    The player is the owner's, on YouTube's privacy-enhanced host — nothing of
    the recording is copied here either.

    ``channel`` is identity, kept apart from the id: it is what was checked
    when the id was filed (YouTube's oEmbed ``author_name``; that endpoint
    answers 400 for an id that does not exist, so a checker could stand on
    it). A re-upload by anyone else is not a source.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    youtube_id: str = Field(pattern=YOUTUBE_ID_PATTERN)
    title: str = Field(min_length=1)
    channel: str = Field(min_length=1)

    @computed_field
    @property
    def embed_url(self) -> str:
        return f"https://www.youtube-nocookie.com/embed/{self.youtube_id}"

    @computed_field
    @property
    def watch_url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.youtube_id}"


class Point(BaseModel):
    """One observation about the text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    heading: str = Field(min_length=1)
    # The curator's own analysis. Goes to the student, so in English.
    body: str = Field(min_length=1)
    # Phrases from the source this point is about, shown as quoted. The only
    # place source wording may sit, and the place the budget counts.
    cites: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _cites_are_short(self) -> Point:
        for cite in self.cites:
            n = word_count(cite)
            if n == 0:
                raise ValueError("an empty citation")
            if n > MAX_CITE_WORDS:
                raise ValueError(f"citation {cite!r} has {n} words; the limit is {MAX_CITE_WORDS}")
        return self


class Section(BaseModel):
    """The analysis at one linguistic level."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    aspect: Aspect
    title: str = Field(min_length=1)
    points: list[Point] = Field(min_length=1)


class WordNote(BaseModel):
    """A word from the text worth learning, with what a Brazilian reader needs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    word: str = Field(min_length=1)
    gloss: str = Field(min_length=1)
    # The trap or nuance — a false friend, an unusual sense. Optional.
    note: str | None = None
    # Bridges to the other two bases, so the lesson and the chat teach the
    # same thing about the same word. Checked against the bases in tests.
    vocab_id: str | None = Field(default=None, pattern=VOCAB_ID_PATTERN)
    rule_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _rule_ids_look_like_rule_ids(self) -> WordNote:
        for rule_id in self.rule_ids:
            if not re.match(RULE_ID_PATTERN, rule_id):
                raise ValueError(f"{self.word}: {rule_id!r} is not shaped like a rule_id")
        return self


class SoundNote(BaseModel):
    """A pronunciation trap. No IPA field: the transcription comes from espeak."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    word: str = Field(min_length=1)
    trap: str = Field(min_length=1)


class Question(BaseModel):
    """A self-check question with one right option."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt: str = Field(min_length=1)
    options: list[str] = Field(min_length=2, max_length=4)
    # Index into ``options``.
    answer: int = Field(ge=0)
    why: str = Field(min_length=1)

    @model_validator(mode="after")
    def _answer_is_an_option(self) -> Question:
        if self.answer >= len(self.options):
            raise ValueError(f"answer {self.answer} but only {len(self.options)} options")
        if len(set(self.options)) != len(self.options):
            raise ValueError("two options are the same")
        return self


class Practice(BaseModel):
    """A writing task the student takes to Charles."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task: str = Field(min_length=1)
    # Goes into the chat input, unsent. The student finishes the sentence.
    starter: str = Field(min_length=1, max_length=200)


class Lesson(BaseModel):
    """A curated lesson. One per YAML file in ``data/lessons/``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=LESSON_ID_PATTERN)
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    cefr: CefrLevel
    source: Source
    # The work performed, when an official recording exists. Optional.
    video: Video | None = None
    sections: list[Section] = Field(min_length=1)
    words: list[WordNote] = Field(default_factory=list)
    sounds: list[SoundNote] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    practice: list[Practice] = Field(default_factory=list)
    # Curation note. Never reaches the student.
    notes: str | None = None

    @property
    def slug(self) -> str:
        return self.id.removeprefix("lesson.")

    @property
    def cited_words(self) -> int:
        return sum(word_count(c) for s in self.sections for p in s.points for c in p.cites)

    def ordered_sections(self) -> list[Section]:
        return sorted(self.sections, key=lambda s: ASPECTS.index(s.aspect))

    @model_validator(mode="after")
    def _one_section_per_aspect(self) -> Lesson:
        aspects = [s.aspect for s in self.sections]
        repeated = sorted({a for a in aspects if aspects.count(a) > 1})
        if repeated:
            raise ValueError(f"more than one section for {', '.join(repeated)}")
        return self

    @model_validator(mode="after")
    def _citations_fit_the_budget(self) -> Lesson:
        """The whole lesson may quote at most ``CITE_BUDGET`` of its source.

        Per-citation length alone would let a lesson rebuild the source five
        words at a time; this is the guard that stops that.
        """
        budget = int(self.source.words * CITE_BUDGET)
        if self.cited_words > budget:
            raise ValueError(
                f"{self.cited_words} cited words; the budget for a "
                f"{self.source.words}-word source is {budget}"
            )
        return self


class LessonShelf:
    """The loaded lessons, by slug, in file order."""

    def __init__(self, lessons: Sequence[Lesson]) -> None:
        self._lessons = tuple(lessons)
        self._by_slug = {lesson.slug: lesson for lesson in self._lessons}
        if len(self._by_slug) != len(self._lessons):
            raise LessonError("two lessons share an id")

    def __len__(self) -> int:
        return len(self._lessons)

    def __iter__(self) -> Iterator[Lesson]:
        return iter(self._lessons)

    def get(self, slug: str) -> Lesson | None:
        return self._by_slug.get(slug)


# --- the local copy of the source ------------------------------------------ #

# A copy whose length is this far from ``Source.words`` is probably not the
# analysed text — a different cut of the speech, or another file under the
# right name. The page says so instead of refusing: the reader can tell.
LOCAL_TEXT_TOLERANCE = 0.10


class LocalText(BaseModel):
    """The source text as this machine holds it, split into paragraphs."""

    model_config = ConfigDict(frozen=True)

    paragraphs: tuple[str, ...]
    words: int
    # True when the copy's length is within ``LOCAL_TEXT_TOLERANCE`` of the
    # length the lesson was analysed against.
    matches: bool


def local_text_path(corpus_dir: Path, lesson: Lesson) -> Path:
    """Where a copy of ``lesson``'s source would sit. The slug is pattern-checked."""
    return corpus_dir / "lessons" / f"{lesson.slug}.txt"


def read_local_text(corpus_dir: Path, lesson: Lesson) -> LocalText | None:
    """The source text from ``corpus/lessons/<slug>.txt``, or ``None``.

    ``None`` covers every way of not having it — no file, an empty file, a
    file that is not UTF-8 — because each of them leaves the page exactly as a
    fresh clone shows it: the link, and nothing lost.
    """
    path = local_text_path(corpus_dir, lesson)
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return None
    # A blank line separates paragraphs; the line breaks inside one are the
    # copy's wrapping, not the speaker's.
    blocks = re.split(r"\n\s*\n", raw.replace("\r\n", "\n").strip())
    paragraphs = tuple(" ".join(b.split()) for b in blocks if b.strip())
    if not paragraphs:
        return None
    words = sum(word_count(p) for p in paragraphs)
    expected = lesson.source.words
    return LocalText(
        paragraphs=paragraphs,
        words=words,
        matches=abs(words - expected) <= expected * LOCAL_TEXT_TOLERANCE,
    )


# A run of a passage: its text, and whether it is the phrase the point cites.
Run = tuple[str, bool]


@dataclass(frozen=True)
class Passages:
    """Where in the copy a point's citations sit.

    ``passages`` are the copy's sentences that hold at least one citation, in
    the order the text has them, each cut into runs so the page can mark the
    cited phrase inside it. ``unplaced`` are the citations no sentence holds —
    a copy that is a different version of the text — which the page still
    shows as bare quotes, so a point never loses what it is about.
    """

    passages: tuple[tuple[Run, ...], ...]
    unplaced: tuple[str, ...]


def _fold(text: str) -> str:
    """Lower case with typographic apostrophes as plain ones, length unchanged.

    The curator types *we've*; the copy says *WE’VE*. Each replacement is one
    character for one, so an offset in the folded text is an offset in the
    original, which is what lets the match mark the original's own letters.
    """
    return text.lower().replace("’", "'").replace("‘", "'")


def place_cites(sentences: Sequence[str], cites: Sequence[str]) -> Passages:
    """The sentences of ``sentences`` that ``cites`` quote, with the quotes marked.

    ## Why this is computed per request and never stored

    The sentences come from this machine's copy (``read_local_text``), so the
    passage beside a point is the copy, shown on the page that already shows
    the copy whole — not a longer citation. Nothing of it reaches the
    ``Lesson`` or ``/api/lessons``, and a clone without the copy gets the bare
    quotes it always had. ``MAX_CITE_WORDS`` and ``CITE_BUDGET`` still hold
    the YAML, which is the only thing versioned.

    A citation matches as whole words, ignoring case — the speech shouts its
    climax in capitals — and every sentence that holds it is a passage: the
    curator who cites a word that recurs is pointing at the recurrence.
    """
    folded = [_fold(s) for s in sentences]
    spans: dict[int, list[tuple[int, int]]] = {}
    unplaced: list[str] = []
    for cite in cites:
        needle = _fold(" ".join(cite.split()))
        if not needle:
            continue
        pattern = re.compile(r"(?<![\w'])" + re.escape(needle) + r"(?![\w'])")
        found = False
        for i, text in enumerate(folded):
            for m in pattern.finditer(text):
                spans.setdefault(i, []).append(m.span())
                found = True
        if not found:
            unplaced.append(cite)
    passages = []
    for i in sorted(spans):
        sentence = sentences[i]
        runs: list[Run] = []
        at = 0
        for start, end in sorted(spans[i]):
            if start < at:  # overlaps the previous mark: extend it
                if end > at:
                    text, _ = runs.pop()
                    runs.append((text + sentence[at:end], True))
                    at = end
                continue
            if start > at:
                runs.append((sentence[at:start], False))
            runs.append((sentence[start:end], True))
            at = end
        if at < len(sentence):
            runs.append((sentence[at:], False))
        passages.append(tuple(runs))
    return Passages(passages=tuple(passages), unplaced=tuple(unplaced))


# --- loading --------------------------------------------------------------- #


def load_lessons(directory: Path) -> list[Lesson]:
    """Validate and return every lesson in ``directory``, in file order.

    Collects the problems of every file before failing, like the other two
    bases: whoever is curating wants the whole list of what to fix.
    """
    if not directory.is_dir():
        raise LessonError(f"no lesson directory at {directory}")

    lessons: list[Lesson] = []
    problems: list[str] = []

    for path in sorted(directory.glob("*.yaml")):
        source = path.name
        try:
            document: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            problems.append(f"{source}: invalid YAML — {exc}")
            continue
        if not isinstance(document, dict):
            problems.append(f"{source}: the top of the file must be a mapping")
            continue
        try:
            lesson = Lesson.model_validate(document)
        except Exception as exc:  # pydantic.ValidationError and ValueError
            problems.append(f"{source}: {exc}")
            continue
        # The file name is the URL: a lesson whose id disagrees with its file
        # would be served at one address and logged under another.
        if lesson.slug != path.stem:
            problems.append(f"{source}: id {lesson.id!r} must be 'lesson.{path.stem}'")
            continue
        lessons.append(lesson)

    if problems:
        raise LessonError(
            f"{len(problems)} problem(s) in the lessons at {directory}:\n  - "
            + "\n  - ".join(problems)
        )
    return lessons


def load_shelf(directory: Path | None = None) -> LessonShelf:
    """Load the shelf from ``directory`` (default: ``data/lessons/``)."""
    if directory is None:
        directory = get_settings().data_dir / "lessons"
    return LessonShelf(load_lessons(directory))


@lru_cache(maxsize=4)
def get_shelf(directory: Path | None = None) -> LessonShelf:
    """The process's shelf, loaded once — same policy as the other bases."""
    return load_shelf(directory)


__all__ = [
    "ASPECTS",
    "CITE_BUDGET",
    "LESSON_ID_PATTERN",
    "LOCAL_TEXT_TOLERANCE",
    "MAX_CITE_WORDS",
    "Aspect",
    "Lesson",
    "LessonError",
    "LessonShelf",
    "LocalText",
    "Point",
    "Practice",
    "Question",
    "Section",
    "SoundNote",
    "Source",
    "WordNote",
    "get_shelf",
    "load_lessons",
    "load_shelf",
    "local_text_path",
    "read_local_text",
    "word_count",
]
