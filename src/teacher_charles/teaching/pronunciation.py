"""The pronunciation base: the sounds Portuguese bends in a student's English.

E10 of ``docs/plano-fase-2.md``. This module holds the curated items of
``data/contrastive/pronunciation.yaml`` and nothing that judges audio yet: the
evaluator that intersects an alignment with an item's signature is E11, and it
joins this module then. What an item carries is ``docs/metodo-fonetico.md``
§1 and §7 read as a schema.

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
from collections.abc import Iterator, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..config import get_settings
from ..store.models import CefrLevel
from .contrastive import PRONUNCIATION_FILE

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


__all__ = [
    "Discrimination",
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
