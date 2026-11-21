"""The pronunciation base and its evaluator: what Portuguese bends in a student's English.

E10 and E11 of ``docs/plano-fase-2.md``. The first half holds the curated items
of ``data/contrastive/pronunciation.yaml``; what an item carries is
``docs/metodo-fonetico.md`` §1 and §7 read as a schema. The second half,
``Evaluator``, is gates 2 and 3 of metodo §3: it takes the phonemes a
recognizer heard and returns a verdict that cites the rule, or silence. It
takes text and symbols, never audio, so ``teaching/`` still imports nothing
from ``voice/``; gate 1 runs there and arrives here as two values.

**Pronunciation items are not text items, and they never meet the text
base.** ``ContrastiveBase`` matches a student's *typed* sentence, and the
grammar verifier (``analyzer._Verifier``) accepts any finding whose
``rule_id`` the base holds. A pronunciation item in that base would let the
model cite ``pron.th-voiced`` against a written sentence and have it pass as
curated, which is the improvised correction E4 exists to refuse. So
``contrastive.load_items`` skips this file, the text model no longer admits
the category, and this module loads it alone.

**The id keeps the ``pron.`` prefix the measurement already wrote.** The E9
reports on disk name their rule as ``pron.th-voiced``, ``--rescore`` matches
them by that id, and so do both plans. A different prefix here would split one
rule into two names across the evidence that admitted it.

**An item is admitted by evidence, and the evidence is in the item.**
``discrimination`` records the synthetic pair (plan §2.3) and the human takes
(plan §3.5) by the md5 prefix of each report, so a status can be re-checked
when a threshold moves (metodo §8). The validators hold the plan §3.7 rule in
place: ``graded`` needs a human take that is both clean and named, and a
graded item the synthetic pair missed has to say why.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..config import get_settings
from ..store.models import CefrLevel
from .contrastive import PRONUNCIATION_FILE
from .phonetics import ipa_owners

PRONUNCIATION_ID_PATTERN = r"^pron\.[a-z0-9]+(?:-[a-z0-9]+)*$"

# Metodo §7's sorting of an item by its evidence. ``rejected`` is a status an
# item can have, but not one it can have *here*: a rejected item is not in the
# base, so the field does not admit it.
Status = Literal["graded", "named-not-clean", "heard-only"]

Pass = Literal["unmonitored", "deliberate"]


class PronunciationError(ValueError):
    """The pronunciation base did not load. Carries every problem, not the first."""


def _check_operation(op: Sequence[str]) -> tuple[str, ...]:
    """An alignment operation, as ``scripts/measure_phonemes.py::ops`` tags it."""
    shape = {"sub": 3, "del": 2, "ins": 2}
    if not op or op[0] not in shape:
        raise ValueError(f"operation {list(op)!r}: the kind must be one of sub, del, ins")
    if len(op) != shape[op[0]]:
        raise ValueError(f"operation {list(op)!r}: {op[0]} takes {shape[op[0]] - 1} phoneme(s)")
    if any(not phoneme for phoneme in op[1:]):
        raise ValueError(f"operation {list(op)!r}: empty phoneme")
    if op[0] == "sub" and op[1] == op[2]:
        raise ValueError(f"operation {list(op)!r}: a substitution must change the phoneme")
    return tuple(op)


class Phrase(BaseModel):
    """What the student reads aloud. It loads the contrast at least twice (metodo §4)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)


class Equivalence(BaseModel):
    """Something correct English also does in this phrase, which the expected text spells otherwise.

    It is per item and per word, not a global table: the three sets metodo §1
    names came from one speaker, and a second voice may turn one of them into a
    real deviation (metodo §10).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    word: str = Field(min_length=1)
    expected: list[str] = Field(min_length=1)
    heard: list[str] = Field(min_length=1)
    takes: list[str] = Field(min_length=1)
    why: str = Field(min_length=1)

    @model_validator(mode="after")
    def _changes_something(self) -> Equivalence:
        if self.expected == self.heard:
            raise ValueError(f"equivalence in {self.word!r}: expected and heard are the same")
        return self


class Take(BaseModel):
    """One human report of plan §3.5, named by the md5 prefix it was written under."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    take: str = Field(pattern=r"^[0-9a-f]{8}$")
    pass_: Pass = Field(alias="pass")
    clean: bool
    named: bool
    per: float = Field(ge=0)


class Synthetic(BaseModel):
    """The ``--discriminate`` pair of plan §2.3, for this rule."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    named: bool
    heard: str | None = None


class Discrimination(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Status
    synthetic: Synthetic
    human: list[Take] = Field(min_length=1)
    # Required when the synthetic pair missed a graded rule: plan §3.7 admits
    # such an item only with the reason written down.
    reason: str | None = None

    @model_validator(mode="after")
    def _status_follows_the_takes(self) -> Discrimination:
        named = [take for take in self.human if take.named]
        clean_named = [take for take in named if take.clean]
        if self.status == "graded" and not clean_named:
            raise ValueError("graded needs a human take that is both clean and named (plan §3.7)")
        if self.status == "named-not-clean" and (not named or clean_named):
            raise ValueError("named-not-clean needs a named take and no clean named one")
        if self.status == "heard-only" and named:
            raise ValueError("heard-only cannot have a named take")
        if self.status == "graded" and not self.synthetic.named and not self.reason:
            raise ValueError(
                "a graded item the synthetic pair missed must give its reason (plan §3.7)"
            )
        return self


class PronunciationItem(BaseModel):
    """One curated rule of ``pronunciation.yaml``.

    ``extra="forbid"`` for ``ContrastiveItem``'s reason: a misspelt field in a
    hand-curated file is a curation error, and failing at load beats ignoring a
    signature the curator believed was in force.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=PRONUNCIATION_ID_PATTERN)

    # The phoneme whose occurrences in the phrase are the opportunities
    # (metodo §6: the signal row counts them, it is not a boolean).
    target: str = Field(min_length=1)

    phrases: list[Phrase] = Field(min_length=1)

    # The minimal pair the student hears before reading (metodo §2.2), in
    # spelling: the measurement script synthesizes it from the phonemes.
    right: str = Field(min_length=1)
    wrong: str = Field(min_length=1)

    # The set of alignment operations that count as this error (plan §3.4).
    signature: list[tuple[str, ...]] = Field(min_length=1)

    equivalences: list[Equivalence] = Field(default_factory=list)

    discrimination: Discrimination

    # Reaches the student, in English (plan decision 1.1), worded by metodo
    # §4's line: name the phoneme where the stack resolves identity, the class
    # where it does not.
    explanation: str = Field(min_length=1)

    cefr: CefrLevel = "B1"

    # For whoever edits the YAML. Never reaches the student.
    notes: str | None = None

    @field_validator("signature", mode="before")
    @classmethod
    def _operations(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [_check_operation(op) if isinstance(op, list | tuple) else op for op in value]

    @model_validator(mode="after")
    def _signature_starts_at_the_target(self) -> PronunciationItem:
        for op in self.signature:
            if op[0] in ("sub", "del") and op[1] != self.target:
                raise ValueError(
                    f"{self.id}: operation {list(op)!r} does not start at "
                    f"the target {self.target!r}"
                )
        return self


class PronunciationBase:
    """The base loaded and indexed by id."""

    def __init__(self, items: Sequence[PronunciationItem]) -> None:
        self._items = tuple(items)
        self._by_id = {item.id: item for item in self._items}
        if len(self._by_id) != len(self._items):
            raise PronunciationError("there are repeated ids in the pronunciation base")

    @property
    def items(self) -> tuple[PronunciationItem, ...]:
        return self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[PronunciationItem]:
        return iter(self._items)

    def __contains__(self, rule_id: object) -> bool:
        return rule_id in self._by_id

    def get(self, rule_id: str) -> PronunciationItem:
        try:
            return self._by_id[rule_id]
        except KeyError as exc:
            raise LookupError(f"no item with id {rule_id!r} in the pronunciation base") from exc

    def graded(self) -> tuple[PronunciationItem, ...]:
        """The items that may give a verdict (metodo §7: all six steps)."""
        return tuple(item for item in self._items if item.discrimination.status == "graded")


def load_items(path: Path) -> list[PronunciationItem]:
    """Validate and return every item in ``path``, collecting all the problems first."""
    if not path.is_file():
        raise PronunciationError(f"the pronunciation base does not exist at {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PronunciationError(f"{path.name}: invalid YAML — {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("items"), list):
        raise PronunciationError(f"{path.name}: the top must be a mapping with an 'items' list")

    items: list[PronunciationItem] = []
    problems: list[str] = []
    seen: dict[str, str] = {}
    for index, entry in enumerate(document["items"]):
        label = f"{path.name}[{index}]"
        if not isinstance(entry, dict):
            problems.append(f"{label}: an item must be a mapping")
            continue
        try:
            item = PronunciationItem.model_validate(entry)
        except Exception as exc:  # pydantic.ValidationError and ValueError
            problems.append(f"{label} ({entry.get('id', '<no id>')}): {exc}")
            continue
        if item.id in seen:
            problems.append(f"{label} ({item.id}): repeated id, already defined in {seen[item.id]}")
            continue
        seen[item.id] = label
        items.append(item)

    if problems:
        raise PronunciationError(
            f"{len(problems)} problem(s) in the pronunciation base at {path}:\n  - "
            + "\n  - ".join(problems)
        )
    return items


def load_base(path: Path | None = None) -> PronunciationBase:
    """Load ``path`` (default: ``data/contrastive/pronunciation.yaml``)."""
    if path is None:
        path = get_settings().data_dir / "contrastive" / PRONUNCIATION_FILE
    return PronunciationBase(load_items(path))


@lru_cache(maxsize=4)
def get_base(path: Path | None = None) -> PronunciationBase:
    """The process's base, loaded once. Editing it means restarting, as for the text base."""
    return load_base(path)


# --- schema ---------------------------------------------------------------- #

SCHEMA_RELATIVE_PATH = Path("schema") / "pronunciation-item.schema.json"


def json_schema() -> dict[str, Any]:
    """The JSON Schema of an item, derived from the model, as ``contrastive.json_schema``."""
    schema = PronunciationItem.model_json_schema(by_alias=True)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://teacher-charles.local/schema/pronunciation-item.schema.json",
        **schema,
    }


def schema_json_text() -> str:
    return json.dumps(json_schema(), indent=2, ensure_ascii=False) + "\n"


# --- the evaluator (E11) --------------------------------------------------- #

# Gate 2's floor: the share of the phrase's phonemes heard where they were
# expected (not substituted, not deleted). It asks whether the take *is the
# phrase*, not whether it was said well, which is why an aggregate is allowed
# here when plan §3.3 forbids one for quality (metodo §3).
#
# It is not the phoneme error rate, and that was measured, not assumed. Reading
# *the red car is ready* against `pron.th-voiced` scores PER 0.875 — inside
# any ceiling that keeps a real deliberate take (0.818) — and the alignment
# then books a /ð/→/d/ that nobody said. Over the 64 takes on disk, each scored
# against every unrelated phrase (751 pairs), the share heard where expected
# never passed 0.50; at 0.55 none of the 751 gets through.
#
# The cost is two clean takes of the right phrase that the recognizer heard as
# mostly noise, 0.18 and 0.47 — and they are the takes that admitted
# `pron.h-dropping` and `pron.ing-alveolar` in E10. Silence on them is the
# price of never correcting a sentence the student did not read; plan §3.3's
# rule is that silence is an answer, and the false correction is the worst one.
PHRASE_HIT_MIN = 0.55

# The classes whose identity this stack does not resolve, on the heard side of
# a signature operation (plan §3.4; the script's ``PHONEME_CLASSES``). None of
# the three admitted rules has a vowel on its heard side, so today this widens
# nothing, and it is here so the first epenthesis item does not have to know.
PHONEME_CLASSES: tuple[frozenset[str], ...] = (frozenset({"ɪ", "i", "ɐ", "ə"}),)

# Punctuation espeak keeps in its output, which is no phoneme.
_IPA_PUNCTUATION = str.maketrans("", "", '.,;:!?…—–"“”‘’()')

Verdict = Literal["correction", "no-error", "silence"]


class PronunciationUnavailable(RuntimeError):
    """The expected text could not be built: espeak did not answer."""


def tokenize(ipa: str, vocabulary: frozenset[str]) -> list[str]:
    """Cut one IPA word into the recognizer's symbols, longest match first.

    ``aɪ`` before ``a``: a two-character symbol split in two would read as an
    error on every take. A character outside the inventory comes back marked
    with ``?`` so it can be refused, never dropped in silence.
    """
    out: list[str] = []
    longest = max((len(token) for token in vocabulary), default=1)
    i = 0
    while i < len(ipa):
        for size in range(min(longest, len(ipa) - i), 0, -1):
            if ipa[i : i + size] in vocabulary:
                out.append(ipa[i : i + size])
                i += size
                break
        else:
            out.append("?" + ipa[i])
            i += 1
    return out


@dataclass(frozen=True, slots=True)
class Expected:
    """The phrase as the recognizer should hear it, each symbol with its word."""

    phonemes: tuple[str, ...]
    words: tuple[str, ...]


def expected_for(text: str, g2p: Callable[[str], str], vocabulary: frozenset[str]) -> Expected:
    """espeak's reading of ``text``, tokenized, with the written word each symbol belongs to.

    espeak joins words the way a speaker runs them (*I am* is ``aɪɐm``), so the
    sentence's IPA words are fitted back to the written ones by
    ``phonetics.ipa_owners`` — the same fit the IPA line uses — and a joined
    pair is labelled with both words.
    """
    sentence = g2p(text).translate(_IPA_PUNCTUATION).split()
    if not sentence:
        raise PronunciationUnavailable(f"espeak gave no reading for {text!r}")
    written = [word.strip(".,;:!?\"'") for word in text.split()]
    per_word = [g2p(word).translate(_IPA_PUNCTUATION).replace(" ", "") for word in written]
    owners = ipa_owners(written, sentence, per_word)
    labels = [
        " ".join(written[i] for i, owned in enumerate(owners) if j in owned) or ""
        for j in range(len(sentence))
    ]
    phonemes: list[str] = []
    words: list[str] = []
    for ipa_word, label in zip(sentence, labels, strict=True):
        for token in tokenize(ipa_word, vocabulary):
            phonemes.append(token)
            words.append(label)
    return Expected(phonemes=tuple(phonemes), words=tuple(words))


@dataclass(frozen=True, slots=True)
class Operation:
    """One edit of the alignment. ``at`` indexes the expected side; an insertion sits before it."""

    kind: Literal["sub", "del", "ins"]
    expected: str | None
    heard: str | None
    at: int

    @property
    def tag(self) -> tuple[str, ...]:
        if self.kind == "sub":
            return ("sub", self.expected or "", self.heard or "")
        if self.kind == "del":
            return ("del", self.expected or "")
        return ("ins", self.heard or "")


def align(expected: Sequence[str], heard: Sequence[str]) -> list[Operation]:
    """Levenshtein with a backtrace that keeps *where* each edit is.

    The script's ``align`` keeps only which edits there were, which was enough
    to measure; a verdict has to say which word the error was in, so this one
    keeps the position. The tie-breaking order is the script's, so the same
    take aligns the same way in both.
    """
    n, m = len(expected), len(heard)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if expected[i - 1] == heard[j - 1]:
                cost[i][j] = cost[i - 1][j - 1]
            else:
                cost[i][j] = 1 + min(cost[i - 1][j - 1], cost[i - 1][j], cost[i][j - 1])
    ops: list[Operation] = []
    i, j = n, m
    while i > 0 or j > 0:
        same = i > 0 and j > 0 and expected[i - 1] == heard[j - 1]
        if same and cost[i][j] == cost[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + 1:
            ops.append(Operation("sub", expected[i - 1], heard[j - 1], i - 1))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            ops.append(Operation("del", expected[i - 1], None, i - 1))
            i -= 1
        else:
            ops.append(Operation("ins", None, heard[j - 1], i))
            j -= 1
    ops.reverse()
    return ops


def heard_where_expected(expected: Sequence[str], ops: Sequence[Operation]) -> float:
    """The share of ``expected`` that came out unchanged: gate 2's measure."""
    missed = {op.at for op in ops if op.kind in ("sub", "del")}
    return 1 - len(missed) / max(len(expected), 1)


def _class_of(phoneme: str) -> frozenset[str]:
    for klass in PHONEME_CLASSES:
        if phoneme in klass:
            return klass
    return frozenset({phoneme})


def widen(signature: Sequence[tuple[str, ...]]) -> set[tuple[str, ...]]:
    """The signature with its heard side opened to the phoneme's class (plan §3.4)."""
    wider: set[tuple[str, ...]] = set(signature)
    for op in signature:
        if op[0] == "sub":
            wider |= {("sub", op[1], alt) for alt in _class_of(op[2])}
        elif op[0] == "ins":
            wider |= {("ins", alt) for alt in _class_of(op[1])}
    return wider


@dataclass(frozen=True, slots=True)
class Finding:
    """One curated operation, in the word it happened in."""

    word: str
    expected: str
    heard: str | None

    def as_dict(self) -> dict[str, Any]:
        return {"word": self.word, "expected": self.expected, "heard": self.heard}


@dataclass(frozen=True, slots=True)
class Judgement:
    """What one take earned. Never a score: metodo §4 has three verdicts and no fourth.

    ``no-error`` is not ``success``. Metodo §4's success also needs the rule to
    have been named at least once *for this student* (§2.4's guard against
    graduating by silence), and that is history, which E12 persists and this
    request does not have. So the evaluator says what it saw, and the schedule
    decides what it was worth.
    """

    rule_id: str
    verdict: Verdict
    # For silence: which gate (1, 2), and the sentence the student reads.
    gate: int | None = None
    advice: str | None = None
    opportunities: int = 0
    findings: tuple[Finding, ...] = field(default_factory=tuple)
    explanation: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """The payload of ``POST /api/pronunciation``.

        It carries **no PER and no heard transcription**: the operations outside
        the signature are exactly what plan §3.2 keeps from the student, and a
        number is the score plan §4 rules out.
        """
        return {
            "rule_id": self.rule_id,
            "verdict": self.verdict,
            "gate": self.gate,
            "advice": self.advice,
            "opportunities": self.opportunities,
            "errors": len(self.findings),
            "findings": [finding.as_dict() for finding in self.findings],
            "explanation": self.explanation,
        }


GATE_2_ADVICE = "I could not match that to the sentence. Read it once more, just as it is written."


class Evaluator:
    """Gates 2 and 3 of metodo §3, for the graded items of a base."""

    def __init__(self, base: PronunciationBase, g2p: Callable[[str], str]) -> None:
        self._base = base
        self._g2p = g2p

    def item(self, rule_id: str) -> PronunciationItem:
        return self._base.get(rule_id)

    def expected(self, rule_id: str, phrase: int, vocabulary: frozenset[str]) -> Expected:
        item = self._base.get(rule_id)
        return expected_for(item.phrases[phrase].text, self._g2p, vocabulary)

    def evaluate(
        self,
        rule_id: str,
        heard: Sequence[str],
        vocabulary: frozenset[str],
        *,
        phrase: int = 0,
        signal_usable: bool = True,
        signal_advice: str | None = None,
    ) -> Judgement:
        """The verdict on one take of ``phrase`` of ``rule_id``.

        ``signal_usable`` and ``signal_advice`` are gate 1, run in ``voice/``
        before the recognizer: a take it refused or warned about is silence
        here, whatever the phonemes say (metodo §7).
        """
        item = self._base.get(rule_id)
        if item.discrimination.status != "graded":
            raise ValueError(f"{rule_id} is {item.discrimination.status}: it gives no verdict")
        if not signal_usable:
            return Judgement(rule_id, "silence", gate=1, advice=signal_advice)

        expected = expected_for(item.phrases[phrase].text, self._g2p, vocabulary)
        if any(token.startswith("?") for token in expected.phonemes):
            raise PronunciationUnavailable(
                f"{rule_id}: the phrase has symbols the recognizer cannot emit: "
                f"{[t for t in expected.phonemes if t.startswith('?')]}"
            )

        ops = align(expected.phonemes, heard)
        if heard_where_expected(expected.phonemes, ops) < PHRASE_HIT_MIN:
            return Judgement(rule_id, "silence", gate=2, advice=GATE_2_ADVICE)

        signature = widen(item.signature)
        findings = tuple(
            Finding(
                word=expected.words[op.at] if op.at < len(expected.words) else "",
                expected=op.expected or "",
                heard=op.heard,
            )
            for op in ops
            if op.tag in signature
        )
        opportunities = sum(1 for token in expected.phonemes if token == item.target)
        if findings:
            return Judgement(
                rule_id,
                "correction",
                opportunities=opportunities,
                findings=findings,
                explanation=item.explanation,
            )
        return Judgement(rule_id, "no-error", opportunities=opportunities)


__all__ = [
    "PHRASE_HIT_MIN",
    "Discrimination",
    "Evaluator",
    "Expected",
    "Finding",
    "Judgement",
    "Operation",
    "PronunciationUnavailable",
    "align",
    "expected_for",
    "heard_where_expected",
    "tokenize",
    "widen",
    "Equivalence",
    "PRONUNCIATION_ID_PATTERN",
    "Phrase",
    "PronunciationBase",
    "PronunciationError",
    "PronunciationItem",
    "SCHEMA_RELATIVE_PATH",
    "Synthetic",
    "Take",
    "get_base",
    "json_schema",
    "load_base",
    "load_items",
    "schema_json_text",
]
