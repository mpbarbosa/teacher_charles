"""Reading authentic English off a subtitle file.

`corpus/` holds dialogue somebody else wrote (see `corpus/README.md`). This
module is the only thing that knows its file format: SubRip in, cues and
utterances out. Text goes in, text comes out -- there is no audio anywhere
here, which is the same reason `teaching/phonetics.py` is not in `voice/`.

## Why this is not in `teaching/`

Because assembling cues into sentences needs a definition of "end of
sentence", and the project already has exactly one: `voice/segment.py`, whose
three exceptions -- abbreviation, decimal, initial -- were curated against the
prose the persona actually produces. A second copy in `teaching/` would drift
from it, which is the mistake the IPA line exists to avoid one layer down.

And `teaching/` may not import `voice/` (`teaching/turn.py`, and the layer
table in CLAUDE.md). So the reader lives out here with `prompts.py`, the other
module whose job is to turn a file on disk into strings the product can use,
and it is free to reuse the one boundary oracle instead of writing a second.

`min_chars=0` when it calls that oracle, and the zero is the point. The
12-character floor in `voice/segment.py` is a *cost* rule -- a synthesis call
for "Oh!" wastes a request and sounds clipped. Nothing is being synthesised
here, and a two-word line is a real thing a speaker said, so the floor would
only glue one speaker's line onto the next speaker's.

## A cue is not an utterance, and that is the whole difficulty

A cue is a box of text sized to fit on screen for a readable number of
milliseconds. It breaks where the screen runs out, not where the sentence
does: one sentence routinely spans three cues, and one cue routinely holds the
ends and beginnings of two. So `parse_srt` reports the file faithfully --
every cue, with its timing -- and `utterances` does the reassembly, which is
lossy and heuristic and kept deliberately separate from the parse.

## What is thrown away, and why that is not tidying

Subtitle files carry text nobody said: `[door slams]`, a music note around a
lyric, a position override from the authoring tool. For a *reading* of the
episode that text is content; for *this* project it is poison, because the
product's whole premise is that what it shows the student is English a person
would say. `kind` records the verdict per cue rather than dropping the row, so
a caller that disagrees can still see everything.

Two limits worth stating, because neither is fixable from the file alone.
A speaker label (`MONICA:`) exists only in subtitles authored for the deaf and
hard of hearing; most releases have none, and then `speaker` is `None`
everywhere and the dash convention is all there is. And a sentence assembled
from several cues gets the span of **those cues**, not of itself -- the file
does not say where inside a cue a sentence ended.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .voice.segment import SentenceAssembler, split_sentences

# How far apart two cues can be and still be read as one continuing sentence.
# Dialogue cues follow each other inside a few hundred milliseconds; a gap of
# seconds is a scene change, and a sentence does not survive one. Without this
# the last cue before a cut would swallow the first cue after it whenever the
# cut landed mid-sentence, which in a sitcom is most cuts.
MAX_CONTINUATION_GAP_MS = 2_000

CueKind = Literal["speech", "music", "sound"]

_TIMECODE = re.compile(
    r"(\d{1,3}):([0-5]\d):([0-5]\d)[,.](\d{1,3})"
    r"\s*-->\s*"
    r"(\d{1,3}):([0-5]\d):([0-5]\d)[,.](\d{1,3})"
)
_INDEX = re.compile(r"\d{1,6}")

# `<i>`, `</i>`, `<font color="#ffffff">`: HTML-ish markup SubRip inherited.
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
# `{\an8}`, `{\pos(192,230)}`: ASS override blocks that leak into .srt files.
# The backslash is required so a brace in real dialogue survives.
_OVERRIDE = re.compile(r"\{\\[^}]*\}")
# A whole bracketed aside: `[door slams]`, `(laughs)`. Stripped from speech,
# and a cue that is nothing else is `sound`.
_BRACKETED = re.compile(r"\[[^\]]*\]|\([^)]*\)")
_MUSIC = re.compile(r"[♪♫♬♩]")
# `MONICA:` -- all caps, because SDH labels are, and because requiring caps is
# what keeps `I said: listen` from being read as a speaker named "I said".
_SPEAKER = re.compile(r"^([A-Z][A-Z0-9 .'’-]{0,24}):\s*(.*)$", re.DOTALL)
# A dash opening a line, which is how one cue holds two speakers.
_DASH = re.compile(r"^[-‐-―]\s*")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class Cue:
    """One numbered block of the file, reported as the file has it."""

    index: int
    start_ms: int
    end_ms: int
    text: str
    kind: CueKind
    # The dash-separated parts of the cue, in order. One element when the cue
    # holds one speaker's words, which is the common case.
    parts: tuple[str, ...]

    @property
    def is_speech(self) -> bool:
        return self.kind == "speech"


@dataclass(frozen=True, slots=True)
class Utterance:
    """One sentence somebody said, reassembled across cue boundaries."""

    text: str
    start_ms: int
    end_ms: int
    speaker: str | None
    # Which cues this sentence was built from. More than one is normal.
    cue_indexes: tuple[int, ...]

    @property
    def spans_cues(self) -> bool:
        """True when the span is the cues' and not this sentence's own."""
        return len(self.cue_indexes) > 1


def _to_ms(hours: str, minutes: str, seconds: str, fraction: str) -> int:
    # SubRip writes milliseconds, but truncated files write two digits and
    # some tools write one. `.5` is 500 ms, not 5.
    millis = int(fraction.ljust(3, "0")[:3])
    return ((int(hours) * 60 + int(minutes)) * 60 + int(seconds)) * 1000 + millis


def _clean(raw: str) -> str:
    # Both substitute the empty string, not a space. `<b>down</b>.` with a
    # space would come out as `down .`, and a stray space before the period
    # is a token the analyser would see and the IPA line would transcribe.
    # Inline markup implies no whitespace; a bracketed aside does, and that
    # one is handled in `_classify`.
    text = _OVERRIDE.sub("", raw)
    text = _TAG.sub("", text)
    return _WHITESPACE.sub(" ", text).strip()


def _strip_asides(text: str) -> str:
    """The text without its bracketed asides.

    Called in three places, which is the point: `_classify` decides the cue's
    kind with it, the cue's `parts` are cleaned with it, and a run is cleaned
    with it again after joining. Missing any one of the three leaks `[laughs]`
    into the corpus, and the leak is invisible in a test that asserts on
    `Cue.text` -- `parts` is what the utterances are built from.

    The third call is not redundant. An aside split across two cues has an
    unbalanced bracket in each of them, so neither cue can recognise it; the
    pair only closes once the run is joined.
    """
    return _WHITESPACE.sub(" ", _BRACKETED.sub(" ", text)).strip()


def _classify(text: str) -> tuple[CueKind, str]:
    """The verdict for a cue, and what is left of it as speech."""
    if _MUSIC.search(text):
        return "music", ""

    spoken = _strip_asides(text)
    # Nothing but the aside: `[door slams]`. An unclosed bracket leaves its
    # text behind, which is the safe direction -- keeping a line that may be
    # speech beats dropping one that is.
    if not spoken:
        return "sound", ""
    return "speech", spoken


def _split_parts(text: str) -> tuple[str, ...]:
    """The cue's dash-separated parts, dashes removed.

    One rule, no special cases: a dash opens a part, and a line without one
    continues the part above it. That covers the three shapes a cue actually
    comes in -- two dashes for two speakers, a single leading dash for a
    sentence the previous cue started, and a dash only on the second line
    where the release omitted the first speaker's.

    The last shape is why the rule is written this way rather than counting
    dashes first. Counting, and treating a lone dash as a continuation
    marker, turned `A said this` / `- B said that` into one part with a
    literal dash sitting in the middle of it -- a hyphen in the text where a
    speaker change belongs.
    """
    parts: list[str] = []
    for line in (line.strip() for line in text.split("\n")):
        if not line:
            continue
        if _DASH.match(line):
            parts.append(_DASH.sub("", line).strip())
        elif parts:
            parts[-1] = f"{parts[-1]} {line}".strip()
        else:
            parts.append(line)
    return tuple(_WHITESPACE.sub(" ", part).strip() for part in parts if part.strip())


def _speaker_of(text: str) -> tuple[str | None, str]:
    match = _SPEAKER.match(text)
    if match is None:
        return None, text
    label, rest = match.group(1).strip(), match.group(2).strip()
    # `NARRATOR:` with nothing after it keeps the label and returns no speech,
    # so the caller drops it. That is the right reading: a label alone is the
    # SDH convention for naming who is about to speak, with the words in the
    # next cue. Returning the raw `NARRATOR:` instead would put a line ending
    # in a colon into the corpus, which is not a thing anybody said.
    return label, rest


def _cue_headers(lines: list[str]) -> Iterator[tuple[int, int, re.Match[str]]]:
    """Every timecode line, with the line index and any number above it."""
    for i, line in enumerate(lines):
        match = _TIMECODE.search(line)
        if match is None:
            continue
        number = 0
        if i and _INDEX.fullmatch(lines[i - 1].strip()):
            number = int(lines[i - 1].strip())
        yield i, number, match


def parse_srt(text: str) -> list[Cue]:
    """Every cue in the file, in file order.

    Tolerant on purpose: a byte-order mark, CRLF, a comma or a dot before the
    milliseconds, a missing cue number, a blank line inside a cue's text. The
    files in `corpus/` come from whatever tool whoever ripped them happened to
    use, and a reader that rejects one of these costs an afternoon to diagnose
    for no gain.

    Anchors on the timecode line rather than on blank-line-separated blocks,
    because the timecode is the one line whose shape is unambiguous. A stray
    blank line inside a cue therefore keeps its text instead of truncating it.
    """
    lines = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n").split("\n")

    headers = list(_cue_headers(lines))
    cues: list[Cue] = []

    for position, (line_index, number, match) in enumerate(headers):
        stop = headers[position + 1][0] if position + 1 < len(headers) else len(lines)
        # The next cue's number sits on the line above its timecode; it is not
        # this cue's text.
        if stop > line_index + 1 and _INDEX.fullmatch(lines[stop - 1].strip()):
            stop -= 1

        body = "\n".join(lines[line_index + 1 : stop]).strip("\n")
        cleaned = _clean(body)
        if not cleaned:
            continue

        kind, spoken = _classify(cleaned)
        # Keep the line structure for the dash test, but on cleaned text.
        structured = "\n".join(_clean(line) for line in body.split("\n") if _clean(line))
        parts = (
            tuple(
                stripped for part in _split_parts(structured) if (stripped := _strip_asides(part))
            )
            if kind == "speech"
            else ()
        )

        cues.append(
            Cue(
                index=number or len(cues) + 1,
                start_ms=_to_ms(*match.group(1, 2, 3, 4)),
                end_ms=_to_ms(*match.group(5, 6, 7, 8)),
                text=spoken if kind == "speech" else cleaned,
                kind=kind,
                parts=parts,
            )
        )

    return cues


def read_srt(path: str | Path) -> list[Cue]:
    """`parse_srt` over a file. UTF-8, which is what the import normalises to."""
    return parse_srt(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class _Segment:
    text: str
    start_ms: int
    end_ms: int
    speaker: str | None
    cue_index: int


def _segments(cues: Iterable[Cue]) -> Iterator[_Segment]:
    for cue in cues:
        if not cue.is_speech:
            continue
        for part in cue.parts:
            speaker, text = _speaker_of(part)
            if text:
                yield _Segment(text, cue.start_ms, cue.end_ms, speaker, cue.index)


def _ends_sentence(text: str) -> bool:
    """Does this text stop at a sentence boundary the project recognises?

    Asked of the assembler rather than of a regex here, so that the three
    exceptions stay in one place: `Mr.`, `3.` and `J.` all leave text pending
    and so all answer no. `3.` answering no is the undecidable case from
    `voice/segment.py` -- it may be the start of `3.5` -- and no is the right
    answer for a cue, because a cue that ends on a digit and a dot is nearly
    always a number the next cue finishes.
    """
    assembler = SentenceAssembler(min_chars=0)
    assembler.push(text)
    return not assembler.pending


_NUMBER_TAIL = re.compile(r"\d\.$")


def _join(run: list[_Segment]) -> str:
    """The run's text, joined without inventing a boundary.

    A space is wrong in exactly one place. `_ends_sentence` refuses to close a
    cue ending in a digit and a dot, because `3.` is probably `3.5` continuing
    in the next cue -- and then joining with a space produces `3. 5`, where
    the dot is followed by whitespace and so *is* a boundary by the oracle's
    own rule. The merge would be undone by the join. A cue break falling
    inside a number is a break with no space in it.
    """
    text = run[0].text
    for segment in run[1:]:
        glue = "" if _NUMBER_TAIL.search(text) and segment.text[:1].isdigit() else " "
        text = f"{text}{glue}{segment.text}"
    return text


def utterances(
    cues: Iterable[Cue], *, max_gap_ms: int = MAX_CONTINUATION_GAP_MS
) -> list[Utterance]:
    """Sentences, reassembled across the cue boundaries that interrupt them.

    Two consecutive segments join when all three hold: the same speaker (or
    neither names one), no sentence boundary at the end of the first, and less
    than `max_gap_ms` of silence between them. Anything else closes the run.

    A run then splits into sentences, and each becomes an `Utterance` carrying
    the run's span -- `spans_cues` says when that span is wider than the
    sentence, which the file gives no way to narrow.
    """
    runs: list[list[_Segment]] = []
    for segment in _segments(cues):
        if runs:
            previous = runs[-1][-1]
            same_speaker = previous.speaker == segment.speaker or segment.speaker is None
            near = segment.start_ms - previous.end_ms <= max_gap_ms
            if same_speaker and near and not _ends_sentence(previous.text):
                runs[-1].append(segment)
                continue
        runs.append([segment])

    out: list[Utterance] = []
    for run in runs:
        joined = _strip_asides(_join(run))
        indexes = tuple(dict.fromkeys(segment.cue_index for segment in run))
        speaker = next((s.speaker for s in run if s.speaker), None)
        sentences = split_sentences(joined, min_chars=0) or [joined]
        for sentence in sentences:
            if not sentence.strip():
                continue
            out.append(
                Utterance(
                    text=sentence.strip(),
                    start_ms=run[0].start_ms,
                    end_ms=run[-1].end_ms,
                    speaker=speaker,
                    cue_indexes=indexes,
                )
            )
    return out
