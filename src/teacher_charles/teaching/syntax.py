"""How a sentence is built: its parts, and the part of speech of each word.

The student asks for it — a button under a message — and gets the sentence
split into subject, verb, objects and the phrases around them, with every word
labelled. It is the sibling of the IPA line (``phonetics.py``): a fact about
the written words, produced by a tool and never by the model.

## Why a parser, and not the model

Because the model improvises. The same reason the correction cites a curated
``rule_id`` and the IPA comes from espeak: a 4B model asked to label
constituents will hand back a confident table, and nothing downstream can tell
a right label from a wrong one. E4's verifier cannot help — it accepts a finding
only when a curated rule backs it, and there is no curated rule for "this word
is the subject".

A parser is not a table the way espeak is. spaCy's small pipeline is a
statistical model too, and it does misread sentences. What it gives that the
chat model does not is **the same answer every time**, a known accuracy, and no
prose: it returns labels from a closed set, and everything the student reads is
written here, in ``WORD_ROLES`` and ``PARTS_OF_SPEECH``, not generated.

## What is shown, and what is deliberately left out

Shown: the coarse part of speech and the dependency relation, which is what
the small pipeline is good at.

Left out: person, number and tense. Measured while choosing the parser, on
*"because I was tired"*: the pipeline tagged *was* as third person. A panel
that says *I was* is third person teaches a mistake, and there is no way to
know which of its morphology calls are the wrong ones.

Left out too: meaning. *in a sewer* is "prepositional phrase", not "place".
The parser knows the shape of the phrase and nothing about what it refers to.

## What the breakdown is not

It is not a verdict. *"I didn't went to the party"* parses cleanly — subject,
auxiliary, negation, verb — because the structure is fine and the mistake is in
the verb form. A tidy breakdown under a wrong sentence must not read as
approval, which is why ``NOTE`` travels with every result and the page prints
it. Same family as §6's asymmetry: the absence of a complaint is not a pass.

## Why it is off the turn's path

Nothing here runs unless the student presses the button. That keeps the path to
the first word untouched (it is already the tightest budget in the product) and
is why this module yields no ``TurnEvent``: ``TurnRunner`` only learns, as a
``bool``, whether the button is on screen, so the teacher can point at it.

## Why it is an extra

Weight. spaCy and its pipeline take a 92 MB environment to 359 MB, against the
21 MB the IPA line cost. See the ``syntax`` extra in ``pyproject.toml``.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import re
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from ..config import Settings

log = logging.getLogger(__name__)

# The pipeline the `syntax` extra installs. It is a Python package, so its name
# is also what `find_spec` looks for.
DEFAULT_MODEL = "en_core_web_sm"

# A breakdown is a pure function of the text, and the student reopens the same
# bubble. Parsing costs ~3 ms, so this is about not redoing silly work.
CACHE_SIZE = 256

# Goes with every result, and the page prints it under the breakdown. It is
# data and not a string in the template so that a test can hold it in place:
# without it, a clean breakdown under a wrong sentence reads as approval.
NOTE = (
    "This shows how the sentence is built, not whether it is correct. "
    "It comes from a parser, not from Charles, and it can misread long or "
    "unusual sentences."
)

# --- The words the student reads ------------------------------------------- #
#
# Closed tables, on purpose. The parser emits a tag; the English on screen is
# chosen here. A tag missing from a table shows as "other (tag)" instead of
# being guessed at.

PARTS_OF_SPEECH = {
    "ADJ": "adjective",
    "ADP": "preposition",
    "ADV": "adverb",
    "AUX": "auxiliary verb",
    "CCONJ": "coordinating conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "noun",
    "NUM": "number",
    "PART": "particle",
    "PRON": "pronoun",
    "PROPN": "proper noun",
    "SCONJ": "subordinating conjunction",
    "SYM": "symbol",
    "VERB": "verb",
    "X": "other",
}

# What a word does for the word it hangs from.
WORD_ROLES = {
    "nsubj": "subject",
    "nsubjpass": "subject (passive)",
    "csubj": "clause as subject",
    "csubjpass": "clause as subject (passive)",
    "expl": "dummy subject",
    "aux": "auxiliary",
    "auxpass": "passive auxiliary",
    "neg": "negation",
    "dobj": "direct object",
    "dative": "indirect object",
    "attr": "subject complement",
    "acomp": "subject complement",
    "oprd": "object complement",
    # Not "preposition": that is its part of speech, printed right beside it.
    "prep": "head of the prepositional phrase",
    "pobj": "object of the preposition",
    "pcomp": "complement of the preposition",
    "agent": "agent (by-phrase)",
    "det": "determiner",
    "predet": "pre-determiner",
    "poss": "possessive",
    "case": "possessive marker",
    "amod": "adjective modifier",
    "advmod": "adverbial modifier",
    "npadvmod": "adverbial (noun phrase)",
    "nummod": "number modifier",
    "quantmod": "quantifier modifier",
    "compound": "part of a compound",
    "nmod": "noun modifier",
    "appos": "apposition",
    "relcl": "relative clause",
    "acl": "clause describing a noun",
    "advcl": "adverbial clause",
    "ccomp": "complement clause",
    "xcomp": "complement clause",
    "mark": "subordinator",
    "cc": "coordinator",
    "conj": "coordinated element",
    "preconj": "pre-coordinator",
    "prt": "particle of a phrasal verb",
    "intj": "interjection",
    "parataxis": "loosely attached clause",
}

# What a whole chunk does in the clause. Only where the phrase-level name
# differs from the word-level one; the rest falls through to ``WORD_ROLES``.
PART_ROLES = {
    "prep": "prepositional phrase",
    "advmod": "adverbial",
    "npadvmod": "adverbial",
    "conj": "coordinated part",
    "prt": "particle",
}

# Pieces that fuse with the main verb when they sit next to it: *has been
# painting*, *didn't go*, *will stay*. Apart from it — *did* in *What did you
# do?* — they stay a part of their own, because that separation is exactly
# what the student needs to see in a question.
VERB_GROUP = frozenset({"aux", "auxpass", "neg", "prt"})

VERBAL = frozenset({"VERB", "AUX"})


class SyntaxUnavailable(Exception):
    """The parser cannot be used: extra not installed, pipeline that will not load.

    React by type, never by substring — same rule as provider, voice and
    phonetics errors.
    """


# --- The result ------------------------------------------------------------ #


@dataclass(frozen=True)
class Word:
    text: str
    # Part of speech, in the English of ``PARTS_OF_SPEECH``.
    pos: str
    # What it does for ``head``, in the English of ``WORD_ROLES``.
    role: str
    # The word it hangs from; empty for the head of the sentence.
    head: str

    def as_dict(self) -> dict[str, str]:
        return {"text": self.text, "pos": self.pos, "role": self.role, "head": self.head}


@dataclass(frozen=True)
class Part:
    """One chunk of the clause: *I*, *'m*, *in a sewer*."""

    text: str
    role: str
    words: tuple[Word, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "role": self.role,
            "words": [word.as_dict() for word in self.words],
        }


@dataclass(frozen=True)
class Sentence:
    text: str
    parts: tuple[Part, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"text": self.text, "parts": [part.as_dict() for part in self.parts]}


@dataclass(frozen=True)
class Breakdown:
    sentences: tuple[Sentence, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "sentences": [sentence.as_dict() for sentence in self.sentences],
            "note": NOTE,
        }


# --- From the parser's tree to the breakdown ------------------------------- #
#
# Everything below reads tokens by attribute (``pos_``, ``tag_``, ``dep_``,
# ``head``, ``children``, ``subtree``, ``i``, ``whitespace_``) and imports
# nothing from spaCy. That is what lets the tests drive it with hand-built
# trees, with the extra not installed.


def _is_noise(token: Any) -> bool:
    """Punctuation and whitespace: part of the text, not of the analysis."""
    return token.pos_ in ("PUNCT", "SPACE") or token.dep_ == "punct"


def part_of_speech(token: Any) -> str:
    """The part of speech as a student's grammar book would name it.

    The parser's coarse tag is almost that, and the exceptions below are the
    places where it would contradict what the breakdown says one column over.
    """
    pos, tag, dep = token.pos_, token.tag_, token.dep_

    if tag == "MD":
        return "modal verb"
    if tag == "TO" and pos == "PART":
        return "infinitive marker"
    if dep == "neg":
        return "negative particle"
    if tag in ("PRP$", "WP$"):
        return "possessive determiner"
    # *off* in *turn the light off* comes back as a preposition with nothing
    # after it; the fine tag knows it is the particle of the phrasal verb.
    if tag == "RP":
        return "particle"
    # *be* as the only verb of its clause is tagged AUX, and "auxiliary verb"
    # under *'m* in *I'm in a sewer* would be wrong: there is no other verb for
    # it to help.
    if pos == "AUX" and dep not in ("aux", "auxpass"):
        return "verb"
    # The fine tag IN covers both readings, and the coarse one sometimes picks
    # the one the tree did not: *since Monday* came back as a subordinating
    # conjunction heading a prepositional phrase. The tree decides.
    if tag == "IN":
        if dep == "mark":
            return "subordinating conjunction"
        if dep in ("prep", "agent", "dative"):
            return "preposition"

    return PARTS_OF_SPEECH.get(pos, f"other ({pos})")


def word_role(token: Any) -> str:
    if token.dep_ == "ROOT":
        return "main verb" if token.pos_ in VERBAL else "head of the phrase"
    if token.tag_ == "TO" and token.pos_ == "PART":
        return "infinitive marker"
    return WORD_ROLES.get(token.dep_, f"other ({token.dep_})")


def _word(token: Any) -> Word:
    return Word(
        text=token.text,
        pos=part_of_speech(token),
        role=word_role(token),
        head="" if token.dep_ == "ROOT" else token.head.text,
    )


def _text(tokens: list[Any]) -> str:
    return "".join(token.text + token.whitespace_ for token in tokens).strip()


def _part(tokens: list[Any], role: str) -> Part | None:
    words = tuple(_word(token) for token in tokens if not _is_noise(token))
    if not words:
        return None
    # Punctuation is dropped from the edges only: a comma inside the chunk is
    # part of how it reads.
    while tokens and _is_noise(tokens[0]):
        tokens = tokens[1:]
    while tokens and _is_noise(tokens[-1]):
        tokens = tokens[:-1]
    return Part(text=_text(tokens), role=role, words=words)


def _clause_parts(root: Any) -> list[Part]:
    """The root's verb, and one chunk for each thing hanging from it."""
    # (tokens, role, fuses with the verb)
    units: list[tuple[list[Any], str, bool]] = [([root], "verb", True)]
    for child in root.children:
        if _is_noise(child):
            continue
        tokens = sorted(child.subtree, key=lambda token: token.i)
        role = PART_ROLES.get(child.dep_) or word_role(child)
        units.append((tokens, role, child.dep_ in VERB_GROUP))
    units.sort(key=lambda unit: unit[0][0].i)

    merged: list[tuple[list[Any], str, bool]] = []
    for tokens, role, fuses in units:
        previous = merged[-1] if merged else None
        # Only neighbours fuse. In a question the subject sits between *did*
        # and *do* as a part of its own, and that alone keeps them apart.
        if previous and previous[2] and fuses:
            merged[-1] = (previous[0] + tokens, "verb", True)
        else:
            merged.append((tokens, role, fuses))

    parts = (_part(tokens, role) for tokens, role, _ in merged)
    return [part for part in parts if part is not None]


def _sentence(sent: Any) -> Sentence | None:
    root = sent.root
    if root.pos_ in VERBAL:
        parts = _clause_parts(root)
    else:
        # *Hello!*, *Good morning.* — no verb, so no clause to cut into
        # subject and predicate. One chunk, words labelled, and no pretending.
        whole = _part(list(sent), "phrase without a verb")
        parts = [whole] if whole is not None else []
    if not parts:
        return None
    return Sentence(text=sent.text.strip(), parts=tuple(parts))


def analyse(doc: Any) -> Breakdown:
    """The breakdown of a parsed text, sentence by sentence."""
    sentences = (_sentence(sent) for sent in doc.sents)
    return Breakdown(sentences=tuple(s for s in sentences if s is not None))


# --- The parser ------------------------------------------------------------ #


class Syntax:
    """spaCy's pipeline, loaded when somebody needs it.

    Lazy, and it never brings anything down: the warm-up swallows the error
    and ``/health`` reports it, same policy as the voice and the IPA line.
    ``parse`` is the one place that raises, and it raises a typed error,
    because its caller is a route that has to tell the student *why* the panel
    did not open.

    ``ready`` never triggers a load, and neither does ``installed`` — that one
    only asks the import system whether the two packages exist. It is what the
    page and the prompt use to decide whether the button is on screen: a
    button that can never work is worse than no button.
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self._nlp: Any | None = None
        self._last_error: str | None = None
        # A spaCy pipeline is not promised to be reentrant, and the warm-up
        # and the route both run it from executor threads.
        self._lock = threading.Lock()
        self._cache = lru_cache(maxsize=CACHE_SIZE)(self._parse)

    # --- state ------------------------------------------------------------ #

    @property
    def installed(self) -> bool:
        """Are spaCy and the pipeline importable? Imports neither."""
        try:
            return all(importlib.util.find_spec(name) is not None for name in ("spacy", self.model))
        except (ImportError, ValueError):
            # A dotted or malformed pipeline name makes `find_spec` raise
            # instead of answering no.
            return False

    @property
    def ready(self) -> bool:
        """Is the pipeline loaded? Forces no load."""
        return self._nlp is not None

    @property
    def last_error(self) -> str | None:
        """What the last load attempt complained about, if it did."""
        return self._last_error

    # --- load ------------------------------------------------------------- #

    def _load(self) -> Any:
        """The pipeline, imported here and not at the top of the module.

        Importing spaCy costs ~3 s and loading the pipeline ~0.5 s, measured
        on this machine. ``import teacher_charles.app`` must not pay that to
        run a test that never opens the panel.
        """
        try:
            import spacy
        except ImportError as exc:
            raise SyntaxUnavailable(
                f"spaCy is not installed ({exc}); run `uv sync --extra syntax`"
            ) from exc

        try:
            # Named entities are not shown, so the component is not loaded.
            return spacy.load(self.model, exclude=["ner"])
        except Exception as exc:  # noqa: BLE001 - spaCy fails in several ways
            raise SyntaxUnavailable(f"pipeline {self.model!r} did not load: {exc}") from exc

    def _ensure(self) -> Any:
        if self._nlp is None:
            self._nlp = self._load()
            self._last_error = None
        return self._nlp

    def _ensure_locked(self) -> None:
        with self._lock:
            self._ensure()

    async def warm_up(self) -> None:
        """Loads the pipeline before the student presses the button.

        In a thread, unlike the IPA warm-up: 3.5 s of import and load on the
        event loop would freeze the server while it is taking its first
        requests. Never propagates — ``/health`` reports, and the lesson does
        not depend on this panel.
        """
        try:
            await asyncio.to_thread(self._ensure_locked)
        except SyntaxUnavailable as exc:
            self._last_error = str(exc)
            log.warning("syntax view unavailable, the lesson goes on without it: %s", exc)

    # --- parse ------------------------------------------------------------ #

    def _parse(self, text: str) -> Breakdown:
        with self._lock:
            nlp = self._ensure()
            return analyse(nlp(text))

    def parse(self, text: str) -> Breakdown:
        """The breakdown of this text. Blocking — call it from a thread.

        Raises ``SyntaxUnavailable`` when the parser cannot load. Blank text
        is an empty breakdown, not an error: there is no structure in nothing.
        """
        clean = text.strip()
        if not clean:
            return Breakdown(sentences=())
        try:
            return self._cache(clean)
        except SyntaxUnavailable as exc:
            self._last_error = str(exc)
            raise


def build_syntax(settings: Settings) -> Syntax | None:
    """The parser the configuration asks for, or ``None`` with the view off.

    ``None`` is a first-class state, as in ``build_voice`` and
    ``build_phonetics``. Building loads nothing.
    """
    if not settings.syntax_enabled:
        return None
    return Syntax(model=settings.syntax_model)


def on_screen(syntax: Syntax | None) -> bool:
    """Is the Structure button on the student's screen?

    One function for the page and for the prompt, so the teacher never points
    at a button the page did not draw. Asked for but not installed is *off
    screen*: ``/health`` says why, and the student is not handed a control
    that answers 503 every time.
    """
    return syntax is not None and syntax.installed


# What asking for a breakdown looks like in the student's own words. A pattern
# and not a question to the model, for the reason the curated hint is one:
# deterministic, free, and debuggable. It errs towards matching — a false hit
# costs one mention of a button, a miss costs an improvised analysis.
_ASKS_FOR_BREAKDOWN = re.compile(
    r"\b(?:"
    r"syntax|syntactic(?:al)?|pars(?:e|ing)|parts? of speech"
    r"|(?:sentence|grammatical) structure"
    r"|structure of (?:this|the|that|my) (?:sentence|phrase)"
    r"|anal[yi][sz](?:e|is|ing) (?:(?:of )?(?:this|the|that|my) )?(?:sentence|phrase)"
    r"|break (?:(?:this|that|my|the) (?:sentence|phrase)|it|this|that) down"
    r"|subject (?:and|or) (?:the )?(?:verb|predicate|object)"
    r")\b",
    re.IGNORECASE,
)

# What the teacher is told, at the end of the prompt, when the student asks.
#
# It is here and not only in the app-capabilities partial because of what the
# partial measured on its own (qwen2.5 3B, three runs each way): without it the
# model answered in two sentences of prose; *with* it, it typed a breakdown
# three times out of three — "Verb phrase: am in", "Object: a sewer", "Present
# Continuous Tense". Describing the panel at the top of the prompt primed the
# very thing it forbade. E4 and E6 had already measured the cure: an
# instruction counts when it sits next to the point of generation.
REQUEST_NOTE = (
    "The student is asking how a sentence is built. Do not analyse it "
    "yourself: no list of parts, no parts of speech, no tense names. Tell them "
    "to send the sentence as a message of its own and press the Structure "
    "button under it — the app shows the breakdown there. Then go on with the "
    "conversation."
)


def asks_for_breakdown(text: str) -> bool:
    """Is the student asking for the structure of a sentence?"""
    return _ASKS_FOR_BREAKDOWN.search(text) is not None


def request_note(syntax_on_screen: bool, student_text: str) -> str:
    """The note for this turn, or an empty string.

    Empty without the button: sending the student to a control the page did
    not draw is the microphone mistake again (``capabilities.py``).
    """
    if syntax_on_screen and asks_for_breakdown(student_text):
        return REQUEST_NOTE
    return ""


def partial_name(syntax_on_screen: bool) -> str | None:
    """The partial that tells the teacher the Structure button exists.

    ``None`` when there is nothing to tell. Same reason as
    ``capabilities.partial_name``: a model that does not know what the product
    does answers from its chatbot priors — here, by typing an analysis of its
    own, which is the improvised grammar this module exists to replace.
    """
    return "partials/app/syntax" if syntax_on_screen else None


__all__ = [
    "CACHE_SIZE",
    "DEFAULT_MODEL",
    "NOTE",
    "Breakdown",
    "Part",
    "Sentence",
    "Syntax",
    "SyntaxUnavailable",
    "Word",
    "REQUEST_NOTE",
    "analyse",
    "asks_for_breakdown",
    "build_syntax",
    "on_screen",
    "partial_name",
    "request_note",
]
