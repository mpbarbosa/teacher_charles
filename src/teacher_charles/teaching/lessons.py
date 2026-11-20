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

### A text nobody published

The text may also be the student's own — a pitch or a talk they wrote and
want to say well. It has no URL to link, so the source is an ``OwnText``
instead of a ``Source``, and the copy in ``corpus/`` is then the only place the
text exists. Nothing else changes: the budget still holds, because the lesson
is public even when the text is private.

### Rehearsing it

With a copy, the page can rehearse it aloud: blocks, a pace, a stopwatch, a
run-through. The YAML says only what the curator knows about the *structure*
— each block's title and target time, in ``rehearsal`` — and the cue words
shown while the text is hidden stay with the copy, for the reason
``RehearsalBlock`` gives. ``rehearsal_plan`` joins the two per request.

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
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    computed_field,
    model_validator,
)

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

    # The default, so a lesson written before ``OwnText`` existed needs no edit.
    kind: Literal["published"] = "published"

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


class OwnText(BaseModel):
    """A text the person running the project wrote, which nobody published.

    A pitch, a talk, a presentation to rehearse. It has no address, outlet or
    byline, and inventing them to fill ``Source`` would be a citation to
    nothing — so this is a second kind of source, not ``Source`` with its
    fields made optional. ``extra="forbid"`` is the guard: an own text with a
    ``url:`` is refused, and a published source keeps every field required.

    Everything else is the same rule. The text sits in
    ``corpus/lessons/<slug>.txt``, outside version control, and the citation
    budget is counted against ``words`` as for any source: a lesson about your
    own text is still public, and a clone must not be able to rebuild it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["own"]
    # What the text is, generically: the page says whose it is.
    title: str = Field(min_length=1)
    words: int = Field(gt=0)


def _source_kind(value: Any) -> str:
    """Which kind of source ``value`` is; a source that does not say is published."""
    if isinstance(value, dict):
        return str(value.get("kind", "published"))
    return getattr(value, "kind", "published")


AnySource = Annotated[
    Annotated[Source, Tag("published")] | Annotated[OwnText, Tag("own")],
    Discriminator(_source_kind),
]


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


class LevelReading(BaseModel):
    """Why the lesson carries the CEFR label it does.

    The label alone is a claim; this is the evidence for it. A text's level is not one
    number: the surface (sentence length, word length) can read a step below
    what its vocabulary and figures of speech ask of the reader, and
    ``teaching/level.py`` sees only the surface, so the curator says which
    measures were taken and which way each one pulls.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # One or two sentences: the verdict and the reason it is not a step either side.
    summary: str = Field(min_length=1)
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


# The longest block a rehearsal plan may time. A target past ten minutes is a
# typo for seconds, not a block: a whole run-through of the first plan is 4:45.
MAX_BLOCK_SECONDS = 600


class RehearsalBlock(BaseModel):
    """One block of a timed rehearsal: one paragraph of the copy, said aloud.

    ## Why there is no ``keywords`` field

    The cue words shown while the text is hidden are words *of the text*, so
    they are source wording, and the YAML has one place for that: ``cites``,
    under the budget. A second, unbudgeted channel would let a plan carry the
    text's key phrases past every guard above. So the cue words live with the
    copy (``corpus/lessons/<slug>.cues.txt``, ``read_local_cues``), and without
    that file the page cues each sentence by its first words. ``extra="forbid"``
    refuses a ``keywords:`` key, the way ``Source`` refuses ``text:``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=1, max_length=60)
    # Seconds the block should take said aloud. Absent: the block is rehearsed
    # but not timed (phrases held in reserve, a case to practise on), and the
    # run-through skips it.
    target: int | None = Field(default=None, gt=0, le=MAX_BLOCK_SECONDS)


class Lesson(BaseModel):
    """A curated lesson. One per YAML file in ``data/lessons/``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=LESSON_ID_PATTERN)
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    cefr: CefrLevel
    source: AnySource
    # The work performed, when an official recording exists. Optional.
    video: Video | None = None
    sections: list[Section] = Field(min_length=1)
    # Why ``cefr`` is what it is. Optional; its citations count in the budget.
    level: LevelReading | None = None
    words: list[WordNote] = Field(default_factory=list)
    sounds: list[SoundNote] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    practice: list[Practice] = Field(default_factory=list)
    # How the copy is rehearsed: one block per paragraph, in order, each with a
    # title and a target. Optional — without it every paragraph is a block of
    # its own, untimed (``rehearsal_plan``).
    rehearsal: list[RehearsalBlock] | None = None
    # Curation note. Never reaches the student.
    notes: str | None = None

    @property
    def slug(self) -> str:
        return self.id.removeprefix("lesson.")

    @property
    def cited_words(self) -> int:
        points = [p for s in self.sections for p in s.points]
        if self.level:
            points += self.level.points
        return sum(word_count(c) for p in points for c in p.cites)

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
    def _a_rehearsal_plan_times_something(self) -> Lesson:
        """A plan that times nothing is the default plan with titles on it.

        It would also leave the run-through with no block to run, which is the
        one thing a plan is for.
        """
        if self.rehearsal is not None and not any(b.target for b in self.rehearsal):
            raise ValueError("a rehearsal plan with no target; leave `rehearsal` out instead")
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


# --- rehearsing the copy --------------------------------------------------- #

# Without cue words of its own, a block shows each sentence by this many of
# its first words: the opening is what a speaker forgets, and the rest of a
# sentence they know usually follows from it.
CUE_OPENING_WORDS = 3


def local_cues_path(corpus_dir: Path, lesson: Lesson) -> Path:
    """Where the cue words for ``lesson``'s copy would sit, beside the copy."""
    return corpus_dir / "lessons" / f"{lesson.slug}.cues.txt"


def read_local_cues(corpus_dir: Path, lesson: Lesson) -> tuple[tuple[str, ...], ...] | None:
    """The cue words from ``corpus/lessons/<slug>.cues.txt``, or ``None``.

    One line per paragraph of the copy, in order, the phrases on a line
    separated by ``·`` or ``;``. An empty line is a paragraph with no cues of
    its own, and so is every paragraph past the last line: trailing blank
    lines are what an editor trims first, so they cannot carry meaning. The
    file sits beside the copy and outside version control for the
    copy's reason (``RehearsalBlock`` says why it is not in the YAML), and it
    is read per request, like the copy. ``None`` covers every way of not
    having it, as in ``read_local_text``.
    """
    try:
        raw = local_cues_path(corpus_dir, lesson).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return None
    # Only trailing blank lines go: a leading one is the first paragraph's.
    lines = raw.replace("\r\n", "\n").rstrip().split("\n")
    cues = tuple(
        tuple(phrase for part in re.split(r"[·;]", line) if (phrase := " ".join(part.split())))
        for line in lines
    )
    return cues if any(cues) else None


def sentence_openings(sentences: Sequence[str]) -> tuple[str, ...]:
    """Each sentence by its first ``CUE_OPENING_WORDS`` words, ``…`` if it goes on."""
    out = []
    for sentence in sentences:
        words = sentence.split()
        if words:
            more = "…" if len(words) > CUE_OPENING_WORDS else ""
            out.append(" ".join(words[:CUE_OPENING_WORDS]) + more)
    return tuple(out)


@dataclass(frozen=True)
class Block:
    """One paragraph of the copy, as the rehearsal shows it."""

    paragraph: int
    title: str
    # Seconds; ``None`` is untimed.
    target: int | None
    words: int
    cues: tuple[str, ...]
    # True when the cues came from the cue file, False when they are the
    # sentence openings the page made up.
    own_cues: bool


@dataclass(frozen=True)
class RehearsalPlan:
    """The copy cut into blocks, with what the lesson says about each.

    ## Why a plan that does not fit is set aside, not stretched

    A plan is aligned with the copy by position: block *k* is paragraph *k*.
    A copy with one paragraph more or less shifts every block after the
    difference, and the page would time paragraph 4 against paragraph 3's
    target and show it the wrong cue words — a wrong number on screen, which
    is worse than no number. So when the counts differ the plan is set aside
    whole, the blocks fall back to plain paragraphs, and the page says why,
    as it says when the copy's length is off. A cue file with more lines than
    the copy has paragraphs is set aside the same way; a shorter one is not
    (``read_local_cues`` says why).
    """

    blocks: tuple[Block, ...]
    # The lesson has a plan, and the copy fits it.
    planned: bool
    # The lesson has a plan, and the copy does not fit it.
    plan_set_aside: bool
    # A cue file exists, and has more lines than the copy has paragraphs. A
    # shorter one fits: its last paragraphs simply have no cues of their own.
    cues_set_aside: bool

    @property
    def timed(self) -> tuple[Block, ...]:
        return tuple(b for b in self.blocks if b.target)

    @property
    def target(self) -> int:
        return sum(b.target or 0 for b in self.blocks)


def rehearsal_plan(
    lesson: Lesson,
    paragraphs: Sequence[Sequence[str]],
    cues: Sequence[Sequence[str]] | None = None,
) -> RehearsalPlan:
    """The rehearsal of ``lesson`` over its copy.

    ``paragraphs`` is the copy as the page shows it — each paragraph cut into
    sentences — and ``cues`` what ``read_local_cues`` returned. Computed per
    request from the copy, like ``place_cites``, and never stored: the cue
    words are the copy's, so nothing of this reaches ``/api/lessons``.
    """
    plan = lesson.rehearsal
    plan_fits = plan is not None and len(plan) == len(paragraphs)
    cues_fit = cues is not None and len(cues) <= len(paragraphs)
    blocks = []
    for i, sentences in enumerate(paragraphs):
        planned = plan[i] if plan is not None and plan_fits else None
        own = tuple(cues[i]) if cues is not None and cues_fit and i < len(cues) else ()
        blocks.append(
            Block(
                paragraph=i,
                title=planned.title if planned else f"Paragraph {i + 1}",
                target=planned.target if planned else None,
                words=sum(word_count(s) for s in sentences),
                cues=own or sentence_openings(sentences),
                own_cues=bool(own),
            )
        )
    return RehearsalPlan(
        blocks=tuple(blocks),
        planned=plan_fits,
        plan_set_aside=plan is not None and not plan_fits,
        cues_set_aside=cues is not None and not cues_fit,
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
    "CUE_OPENING_WORDS",
    "LESSON_ID_PATTERN",
    "LOCAL_TEXT_TOLERANCE",
    "MAX_BLOCK_SECONDS",
    "MAX_CITE_WORDS",
    "AnySource",
    "Aspect",
    "Block",
    "Lesson",
    "LessonError",
    "LessonShelf",
    "LocalText",
    "OwnText",
    "Point",
    "Practice",
    "Question",
    "RehearsalBlock",
    "RehearsalPlan",
    "Section",
    "SoundNote",
    "Source",
    "WordNote",
    "get_shelf",
    "load_lessons",
    "load_shelf",
    "local_cues_path",
    "local_text_path",
    "read_local_cues",
    "read_local_text",
    "rehearsal_plan",
    "sentence_openings",
    "word_count",
]
