"""The IPA symbol table, its cut, and the chart under a lesson's full text."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from teacher_charles.app import create_app
from teacher_charles.llm.fake import FakeProvider
from teacher_charles.teaching.ipa_symbols import (
    Symbol,
    SymbolError,
    SymbolTable,
    load_symbols,
)
from teacher_charles.teaching.phonetics import Phonetics
from test_lessons import PLACEHOLDER, SLUG, put_copy, settings_for

DATA = Path(__file__).resolve().parents[1] / "data"
TABLE_PATH = DATA / "phonetics" / "symbols.yaml"


@pytest.fixture(scope="module")
def table() -> SymbolTable:
    return load_symbols(TABLE_PATH)


@pytest.fixture(scope="module")
def phonetics() -> Phonetics:
    return Phonetics()


# --- the curated table ------------------------------------------------------ #


def test_every_example_word_carries_its_own_symbol(table, phonetics):
    """The example is a claim about espeak, and espeak is asked.

    An example whose IPA lacks the symbol would play the student a word
    without the sound the tile names — the failure the table exists to
    prevent, one layer down.
    """
    wrong = []
    for s in table.symbols:
        units, _ = table.segment(phonetics.transcribe(s.example))
        if s.symbol not in units:
            wrong.append(f"{s.symbol} in {s.example!r}: {phonetics.transcribe(s.example)}")
    assert not wrong, wrong


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)


def test_the_lessons_own_english_has_no_sound_the_table_lacks(table, phonetics):
    """Every string the versioned lessons carry, read by espeak, cuts clean.

    A sound outside the table would be counted nowhere and drawn nowhere:
    the chart would quietly leave it out. The lessons are the English this
    product shows, and they are in the repository, where a copy of a source
    text is not.
    """
    texts = [
        s
        for path in sorted((DATA / "lessons").glob("*.yaml"))
        for s in _strings(yaml.safe_load(path.read_text(encoding="utf-8")))
        if s.isascii() and len(s.split()) > 2
    ]
    assert len(texts) > 20
    unknown = {}
    for text in texts:
        _, missing = table.segment(phonetics.transcribe(text))
        for ch in missing:
            unknown.setdefault(ch, text[:60])
    assert not unknown, unknown


def test_the_table_lists_each_symbol_once_and_only_known_kinds(table):
    symbols = [s.symbol for s in table.symbols]
    assert len(symbols) == len(set(symbols))
    assert {s.kind for s in table.symbols} == {"vowel", "diphthong", "consonant"}


# --- the cut ---------------------------------------------------------------- #


def test_a_two_character_sound_is_one_unit(table):
    assert table.segment("tʃˈɜːtʃ") == (["tʃ", "ɜː", "tʃ"], [])
    assert table.segment("fˈeɪs") == (["f", "eɪ", "s"], [])


def test_stress_and_punctuation_are_not_sounds(table):
    assert table.segment("ˌɔn ðə, ˈwɛt.") == (["ɔ", "n", "ð", "ə", "w", "ɛ", "t"], [])


def test_a_character_outside_the_table_is_reported_not_dropped(table):
    units, unknown = table.segment("pʁa")
    assert units == ["p"]
    # "a" alone is no sound of English here: it only rides in "aɪ" and "aʊ".
    assert unknown == ["ʁ", "a"]


def test_the_syllabic_mark_stays_with_its_n(table):
    assert table.segment("bˈʌʔn̩") == (["b", "ʌ", "ʔ", "n̩"], [])


# --- the chart -------------------------------------------------------------- #


def test_the_chart_lists_only_what_the_text_uses_in_the_tables_order():
    small = SymbolTable(
        (
            Symbol("iː", "vowel", "see"),
            Symbol("ɪ", "vowel", "sit"),
            Symbol("eɪ", "diphthong", "face"),
            Symbol("s", "consonant", "see"),
            Symbol("t", "consonant", "tea"),
        )
    )
    chart = small.chart(["sˈɪt", "sˈiː sˈɪt"])
    assert [g["kind"] for g in chart] == ["vowel", "consonant"]  # no diphthong used
    assert chart[0]["symbols"] == [
        {"symbol": "iː", "example": "see", "count": 1},
        {"symbol": "ɪ", "example": "sit", "count": 2},
    ]
    assert [r["symbol"] for r in chart[1]["symbols"]] == ["s", "t"]


# --- the loader refuses what it cannot use ---------------------------------- #


@pytest.mark.parametrize(
    "items, complaint",
    [
        ([{"symbol": "p", "kind": "plosive", "example": "pen"}], "kind"),
        ([{"symbol": "p", "kind": "consonant"}], "example"),
        (
            [
                {"symbol": "p", "kind": "consonant", "example": "pen"},
                {"symbol": "p", "kind": "consonant", "example": "pin"},
            ],
            "twice",
        ),
        ([], "no 'symbols'"),
    ],
)
def test_a_malformed_table_is_refused(tmp_path, items, complaint):
    path = tmp_path / "symbols.yaml"
    path.write_text(yaml.safe_dump({"symbols": items}), encoding="utf-8")
    with pytest.raises(SymbolError, match=complaint):
        load_symbols(path)


# --- on the page ------------------------------------------------------------ #


def _page(tmp_path, **overrides) -> str:
    settings = settings_for(tmp_path, **overrides)
    put_copy(settings.resolved_corpus_dir, PLACEHOLDER)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        return c.get(f"/lessons/{SLUG}").text


def test_with_voice_each_sound_is_a_speaker_for_its_example(tmp_path):
    page = _page(tmp_path, phonetics_enabled=True, voice_enabled=True)
    assert 'class="symbol-chart"' in page
    # "First paragraph..." has the vowel of "bird" in "First".
    assert 'class="say symbol" data-say="bird"' in page
    assert '<span class="glyph">ɜː</span>' in page
    # The example's IPA is espeak's, made for the page.
    assert "/bˈɜːd/" in page


def test_without_voice_the_chart_shows_but_plays_nothing(tmp_path):
    page = _page(tmp_path, phonetics_enabled=True)
    assert 'class="symbol-chart"' in page
    assert 'class="symbol"' in page
    assert 'data-say="bird"' not in page


def test_without_ipa_there_is_no_chart(tmp_path):
    assert 'class="symbol-chart"' not in _page(tmp_path, voice_enabled=True)
