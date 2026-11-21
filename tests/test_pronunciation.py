"""The pronunciation base (E10).

Three halves, for three different failures:

**The loader**, on synthetic files in ``tmp_path``: an item outside the schema,
or one whose status its own takes do not support, is a code bug and fails the
load.

**The wall between the two bases.** The text base matches typed sentences and
the grammar verifier accepts any ``rule_id`` it holds
(``analyzer._Verifier._rule_for``: ``claimed in self.base``). A pronunciation
id in there would pass a sound off as a curated grammar correction.

**The real base, item by item**, against espeak: the phrase loads the target
at least twice, the pair differs in the target, and every equivalence is
something the phrase actually contains. A phrase that drifts from its rule
breaks the suite when it is written.
"""

from __future__ import annotations

import json

import pytest
import yaml

from teacher_charles.config import REPO_ROOT
from teacher_charles.teaching import contrastive
from teacher_charles.teaching.phonetics import Phonetics
from teacher_charles.teaching.pronunciation import (
    SCHEMA_RELATIVE_PATH,
    PronunciationError,
    load_base,
    load_items,
    schema_json_text,
)

DATA = REPO_ROOT / "data" / "contrastive"
REAL_FILE = DATA / contrastive.PRONUNCIATION_FILE

# The three rules plan §3.7 admitted. Growing this set is E10's work; it grows
# by a clean human take, and this line is where a new admission is noticed.
ADMITTED = {"pron.th-voiced", "pron.h-dropping", "pron.ing-alveolar"}


def write(path, items) -> None:
    path.write_text(yaml.safe_dump({"items": items}, allow_unicode=True), encoding="utf-8")


def valid_item(**overrides) -> dict:
    item = {
        "id": "pron.th-voiced",
        "target": "ð",
        "phrases": [{"text": "They brought the other brother"}],
        "right": "the other brother",
        "wrong": "de udder brudder",
        "signature": [["sub", "ð", "d"]],
        "discrimination": {
            "status": "graded",
            "synthetic": {"named": True},
            "human": [
                {
                    "take": "b704dc72",
                    "pass": "unmonitored",
                    "clean": True,
                    "named": True,
                    "per": 0.188,
                }
            ],
        },
        "explanation": "English <th> in other is /ð/.",
    }
    item.update(overrides)
    return item


def discrimination(status="graded", synthetic=True, reason=None, **take) -> dict:
    row = {"take": "b704dc72", "pass": "unmonitored", "clean": True, "named": True, "per": 0.1}
    row.update(take)
    out = {"status": status, "synthetic": {"named": synthetic}, "human": [row]}
    if reason is not None:
        out["reason"] = reason
    return out


def refused(tmp_path, item, needle: str) -> None:
    path = tmp_path / "pronunciation.yaml"
    write(path, [item])
    with pytest.raises(PronunciationError) as caught:
        load_items(path)
    assert needle in str(caught.value), str(caught.value)


# --- the loader ------------------------------------------------------------ #


def test_a_valid_item_loads(tmp_path):
    path = tmp_path / "pronunciation.yaml"
    write(path, [valid_item()])
    (item,) = load_items(path)
    assert item.signature == [("sub", "ð", "d")]
    assert item.discrimination.human[0].pass_ == "unmonitored"


def test_an_unknown_field_is_refused(tmp_path):
    refused(tmp_path, valid_item(signatures=[["sub", "ð", "d"]]), "signatures")


def test_the_id_keeps_the_prefix_the_reports_use(tmp_path):
    """``pron.``, not ``pronunciation.``: the E9 reports and ``--rescore`` say ``pron.``."""
    refused(tmp_path, valid_item(id="pronunciation.th-voiced"), "pattern")


def test_graded_needs_a_take_both_clean_and_named(tmp_path):
    """Plan §3.7: a naming on a warned take does not admit a rule."""
    item = valid_item(discrimination=discrimination(clean=False))
    refused(tmp_path, item, "both clean and named")


def test_named_not_clean_cannot_have_a_clean_naming(tmp_path):
    item = valid_item(discrimination=discrimination(status="named-not-clean"))
    refused(tmp_path, item, "no clean named one")


def test_heard_only_cannot_have_a_naming(tmp_path):
    item = valid_item(discrimination=discrimination(status="heard-only"))
    refused(tmp_path, item, "heard-only")


def test_a_graded_rule_the_synthesis_missed_must_say_why(tmp_path):
    """Plan §3.7's clause for ``pron.ing-alveolar``: admitted with the reason written."""
    refused(tmp_path, valid_item(discrimination=discrimination(synthetic=False)), "reason")
    path = tmp_path / "ok.yaml"
    write(path, [valid_item(discrimination=discrimination(synthetic=False, reason="Kokoro."))])
    assert load_items(path)


def test_the_signature_starts_at_the_target(tmp_path):
    refused(tmp_path, valid_item(signature=[["sub", "θ", "t"]]), "does not start at the target")


def test_a_malformed_operation_is_refused(tmp_path):
    refused(tmp_path, valid_item(signature=[["sub", "ð"]]), "takes 2 phoneme(s)")
    refused(tmp_path, valid_item(signature=[["swap", "ð", "d"]]), "sub, del, ins")


def test_a_take_is_named_by_its_md5_prefix(tmp_path):
    refused(tmp_path, valid_item(discrimination=discrimination(take="th-voiced")), "take")


def test_a_repeated_id_is_refused(tmp_path):
    path = tmp_path / "pronunciation.yaml"
    write(path, [valid_item(), valid_item()])
    with pytest.raises(PronunciationError, match="repeated id"):
        load_items(path)


# --- the wall between the bases ------------------------------------------- #


def test_the_text_base_refuses_a_pronunciation_category(tmp_path):
    """Put in a text file, a pronunciation item does not load as text."""
    item = {
        "id": "pronunciation.th-voiced",
        "category": "pronunciation",
        "pt": "o português não tem /ð/",
        "wrong": ["de udder"],
        "right": ["the other"],
        "explanation": "English <th> is /ð/.",
    }
    write(tmp_path / "grammar.yaml", [item])
    with pytest.raises(contrastive.ContrastiveError):
        contrastive.load_items(tmp_path)


def test_the_text_loader_never_reads_the_pronunciation_file(tmp_path):
    """The file shares the directory, and the text loader passes it by."""
    text_item = {
        "id": "calque.age-have",
        "category": "calque",
        "pt": "Eu tenho 30 anos",
        "wrong": ["I have 30 years"],
        "right": ["I am 30 years old"],
        "explanation": "Age is something you are.",
    }
    write(tmp_path / "calques.yaml", [text_item])
    write(tmp_path / contrastive.PRONUNCIATION_FILE, [valid_item()])
    assert [item.id for item in contrastive.load_items(tmp_path)] == ["calque.age-have"]


def test_no_pronunciation_rule_is_citable_against_typed_text():
    """The verifier keeps a model's cited id when ``claimed in self.base``."""
    text_base = contrastive.load_base(DATA)
    for item in load_base(REAL_FILE):
        assert item.id not in text_base, item.id


# --- the real base, item by item ------------------------------------------- #

REAL = load_base(REAL_FILE)


@pytest.fixture(scope="module")
def g2p():
    """espeak with no stress marks, which is how the E9 reports wrote the expected text."""
    phonetics = Phonetics(with_stress=False)
    if not phonetics.transcribe("the"):
        pytest.fail(f"espeak did not load: {phonetics.last_error}")
    return phonetics.transcribe


def test_the_real_base_is_the_three_admitted_rules():
    assert {item.id for item in REAL} == ADMITTED
    assert {item.id for item in REAL.graded()} == ADMITTED


@pytest.mark.parametrize("item", REAL.items, ids=lambda item: item.id)
def test_each_phrase_loads_the_target_at_least_twice(item, g2p):
    """Metodo §4: one opportunity cannot tell a clean production from a lucky alignment."""
    for phrase in item.phrases:
        count = g2p(phrase.text).count(item.target)
        assert count >= 2, f"{phrase.text!r} has {item.target} {count} time(s)"


@pytest.mark.parametrize("item", REAL.items, ids=lambda item: item.id)
def test_the_pair_differs_in_the_target(item, g2p):
    """The ``right`` side carries the target and the ``wrong`` side has lost it (metodo §2.2)."""
    assert item.target in g2p(item.right), item.right
    assert item.target not in g2p(item.wrong), item.wrong


@pytest.mark.parametrize("item", REAL.items, ids=lambda item: item.id)
def test_each_equivalence_is_in_the_phrase(item, g2p):
    for equivalence in item.equivalences:
        phrases = [p.text for p in item.phrases if equivalence.word in p.text.lower().split()]
        assert phrases, f"{equivalence.word!r} is in none of the phrases"
        spelled = "".join(equivalence.expected)
        assert any(spelled in g2p(text).replace(" ", "") for text in phrases), spelled


@pytest.mark.parametrize("item", REAL.items, ids=lambda item: item.id)
def test_each_equivalence_cites_a_take_of_the_item(item):
    takes = {take.take for take in item.discrimination.human}
    for equivalence in item.equivalences:
        assert set(equivalence.takes) <= takes, equivalence.word


# --- the versioned schema -------------------------------------------------- #


def test_the_versioned_schema_matches_the_model():
    path = REPO_ROOT / "data" / SCHEMA_RELATIVE_PATH
    assert path.read_text(encoding="utf-8") == schema_json_text(), (
        "schema out of sync: run `uv run python scripts/dump_contrastive_schema.py`"
    )


def test_the_versioned_schema_is_json_schema():
    schema = json.loads((REPO_ROOT / "data" / SCHEMA_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert schema["$schema"].startswith("https://json-schema.org/")
    assert {"id", "target", "signature", "discrimination", "explanation"} <= set(schema["required"])
