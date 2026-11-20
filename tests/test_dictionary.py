"""Tests for the word cards: the lookup, its file, the route and the page.

A small lookup file is built through ``write_lookup`` — the same code the
fetch script uses — so nothing here needs WordNet or the network.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from teacher_charles.app import create_app
from teacher_charles.config import PACKAGE_ROOT, Settings
from teacher_charles.llm import FakeProvider
from teacher_charles.teaching.dictionary import (
    MAX_SENSES,
    MISS,
    NOTE,
    BuiltWord,
    Dictionary,
    DictionaryUnavailable,
    Sense,
    base_forms,
    normalise,
    write_lookup,
)
from teacher_charles.teaching.lessons import load_shelf, local_text_path

SLUG = "mon-mothma-senate-speech"


def sense(definition: str, *portuguese: str, example: str | None = None) -> Sense:
    return Sense(definition=definition, example=example, portuguese=portuguese)


WORDS = [
    BuiltWord(
        "chamber",
        "n",
        senses=(
            sense("a natural or artificial enclosed space", "estância"),
            sense("an enclosed volume in the body", "câmara", example="the chambers of his heart"),
            sense("a room where a judge transacts business", "câmara", "sala"),
            sense("a deliberative or legislative assembly", "câmara"),
            sense("a room used primarily for sleeping", "dormitório", "quarto"),
            sense("a sixth sense, to be cut by the card"),
        ),
    ),
    BuiltWord("chamber", "v", senses=(sense("place in a chamber"),)),
    BuiltWord("voice", "n", senses=(sense("the sound made by the vocal folds", "voz"),)),
    BuiltWord("voice", "v", senses=(sense("give voice to", "exprimir"),)),
    BuiltWord("go", "v", forms=("went", "gone"), senses=(sense("move", "ir"),)),
    BuiltWord("stop", "v", senses=(sense("come to a halt", "parar"),)),
    BuiltWord("nation", "n", senses=(sense("a politically organized people", "nação"),)),
    BuiltWord("Nation", "n", senses=(sense("a United States prohibitionist"),)),
    BuiltWord("as", "r", senses=(sense("to the same degree", "tanto quanto"),)),
    BuiltWord("AS", "n", senses=(sense("a United States territory", "Samoa Americana"),)),
    BuiltWord("As", "n", senses=(sense("a very poisonous metallic element", "arsênico"),)),
    BuiltWord("silenced", "s", senses=(sense("reduced to silence"),)),
    BuiltWord("silence", "v", senses=(sense("cause to be quiet", "silenciar"),)),
    BuiltWord("pre-war", "a", senses=(sense("existing before a war", "pré-guerra"),)),
    # The lexicon's ids sort the adjective first; the card must not.
    BuiltWord("garden", "a", senses=(sense("the usual or familiar type"),)),
    BuiltWord("garden", "n", senses=(sense("a plot of ground for plants", "jardim"),)),
    # A word with no senses is not written at all.
    BuiltWord("empty", "n"),
]


@pytest.fixture
def lookup_path(tmp_path) -> Path:
    path = tmp_path / "cache" / "wordnet-en-pt.sqlite"
    write_lookup(path, WORDS, meta={"english": "test", "portuguese": "test"})
    return path


@pytest.fixture
def dictionary(lookup_path) -> Dictionary:
    return Dictionary(lookup_path)


def blocks(entry) -> list[tuple[str, str]]:
    return [(h.lemma, h.pos) for h in entry.headwords]


# --- the word on the page --------------------------------------------------- #


@pytest.mark.parametrize(
    ("token", "word"),
    [
        ("chamber,", "chamber"),
        ("Voices.", "voices"),
        ("“Tyranny”", "tyranny"),
        ("nation’s", "nation"),
        ("Senators'", "senators"),
        ("we've", "we've"),
        ("pre-war", "pre-war"),
        ("—", ""),
    ],
)
def test_the_word_loses_its_punctuation_case_and_possessive(token, word):
    assert normalise(token) == word


def test_base_forms_follow_morphy_and_undo_a_doubled_consonant():
    assert ("voice", "n") in base_forms("voices")
    assert ("silence", "v") in base_forms("silenced")
    assert ("stop", "v") in base_forms("stopped")
    assert ("big", "a") in base_forms("bigger")
    # -ing only ever strips to a verb.
    assert {pos for _, pos in base_forms("unfolding")} == {"v"}


# --- the lookup ------------------------------------------------------------- #


def test_a_word_is_found_with_its_senses_in_the_lexicons_order(dictionary):
    entry = dictionary.lookup("chamber,")
    assert entry.found
    assert blocks(entry) == [("chamber", "n")]
    first = entry.headwords[0]
    assert [s.definition for s in first.senses][:2] == [
        "a natural or artificial enclosed space",
        "an enclosed volume in the body",
    ]
    assert first.senses[1].example == "the chambers of his heart"
    assert first.senses[2].portuguese == ("câmara", "sala")


def test_the_card_shows_at_most_max_senses_and_counts_the_rest(dictionary):
    entry = dictionary.lookup("chamber")
    assert sum(len(h.senses) for h in entry.headwords) == MAX_SENSES
    # The sixth noun sense and the verb's one.
    assert entry.more == 2


def test_a_regular_inflection_finds_its_lemma_in_each_part_of_speech(dictionary):
    assert blocks(dictionary.lookup("Voices")) == [("voice", "n"), ("voice", "v")]
    assert blocks(dictionary.lookup("stopped")) == [("stop", "v")]


def test_an_irregular_form_finds_its_lemma_through_the_forms_table(dictionary):
    assert blocks(dictionary.lookup("went")) == [("go", "v")]


def test_the_word_as_written_comes_before_its_inflected_reading(dictionary):
    # Both are right, and the order is morphy's: the word as given first.
    assert blocks(dictionary.lookup("silenced")) == [("silenced", "s"), ("silence", "v")]


def test_a_capital_in_the_text_puts_its_own_case_first_and_keeps_both(dictionary):
    assert blocks(dictionary.lookup("Nation")) == [("Nation", "n"), ("nation", "n")]
    assert blocks(dictionary.lookup("As")) == [("As", "n"), ("as", "r"), ("AS", "n")]


def test_a_lower_case_word_is_never_a_name_or_a_symbol(dictionary):
    assert blocks(dictionary.lookup("nation’s")) == [("nation", "n")]
    assert blocks(dictionary.lookup("as")) == [("as", "r")]


def test_parts_of_speech_come_in_wordnets_order_not_the_files(dictionary):
    assert blocks(dictionary.lookup("garden")) == [("garden", "n"), ("garden", "a")]


def test_a_hyphenated_word_is_found(dictionary):
    assert blocks(dictionary.lookup("pre-war")) == [("pre-war", "a")]


def test_a_word_outside_the_lexicon_is_a_miss_that_says_why(dictionary):
    for token in ("the", "we've", "Palpatine", "empty"):
        entry = dictionary.lookup(token)
        assert not entry.found
        assert entry.as_dict()["note"] == MISS


def test_every_found_entry_carries_the_note_and_its_sources(dictionary):
    body = dictionary.lookup("chamber").as_dict()
    assert body["note"] == NOTE
    assert {s["licence"] for s in body["sources"]} == {"CC BY 4.0", "CC BY-SA 4.0"}
    assert body["headwords"][0]["pos"] == "noun"


def test_a_satellite_adjective_reads_as_an_adjective(dictionary):
    assert dictionary.lookup("silenced").as_dict()["headwords"][0]["pos"] == "adjective"


# --- the file --------------------------------------------------------------- #


def test_a_missing_file_is_not_installed_and_is_never_created(tmp_path):
    path = tmp_path / "nowhere" / "wordnet-en-pt.sqlite"
    missing = Dictionary(path)
    assert not missing.installed and not missing.ready
    with pytest.raises(DictionaryUnavailable, match="fetch_dictionary"):
        missing.lookup("chamber")
    assert not path.exists()
    assert "fetch_dictionary" in missing.last_error


def test_installed_and_ready_open_nothing(dictionary):
    assert dictionary.installed
    assert not dictionary.ready
    dictionary.lookup("chamber")
    assert dictionary.ready
    assert dictionary.meta["english"] == "test"


def test_a_file_from_another_schema_is_refused_with_the_way_out(lookup_path):
    conn = sqlite3.connect(lookup_path)
    conn.execute("UPDATE meta SET value = '0' WHERE key = 'schema'")
    conn.commit()
    conn.close()
    with pytest.raises(DictionaryUnavailable, match="again"):
        Dictionary(lookup_path).lookup("chamber")


def test_an_interrupted_build_leaves_no_file_that_looks_finished(tmp_path):
    path = tmp_path / "wordnet-en-pt.sqlite"

    def broken():
        yield WORDS[0]
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        write_lookup(path, broken(), meta={})
    assert not Dictionary(path).installed


def test_the_lookup_is_opened_read_only(dictionary, lookup_path):
    dictionary.lookup("chamber")
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        dictionary._conn.execute("DELETE FROM word")


# --- the route and the page ------------------------------------------------- #


def settings_for(tmp_path, **overrides) -> Settings:
    base = {
        "db_path": tmp_path / "words.db",
        "provider": "fake",
        "llm_warmup": False,
        "phonetics_enabled": False,
        "corpus_dir": tmp_path / "corpus",
        "dictionary_path": tmp_path / "cache" / "wordnet-en-pt.sqlite",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def put_copy(settings: Settings, text: str = "The chamber voices went.\n") -> None:
    lesson = load_shelf(settings.data_dir / "lessons").get(SLUG)
    path = local_text_path(settings.resolved_corpus_dir, lesson)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def client(tmp_path, lookup_path):
    settings = settings_for(tmp_path, dictionary_path=lookup_path)
    put_copy(settings)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        yield c


def test_the_route_returns_the_card(client):
    response = client.post("/api/define", json={"word": "Voices,"})
    assert response.status_code == 200
    body = response.json()
    assert body["word"] == "voices"
    assert [(h["lemma"], h["pos"]) for h in body["headwords"]] == [
        ("voice", "noun"),
        ("voice", "verb"),
    ]
    assert body["headwords"][0]["senses"][0]["portuguese"] == ["voz"]
    # The IPA is off in these settings, and off means no IPA, not a guess.
    assert body["headwords"][0]["ipa"] == ""


def test_the_headword_ipa_is_espeaks(tmp_path, lookup_path, monkeypatch):
    settings = settings_for(tmp_path, dictionary_path=lookup_path, phonetics_enabled=True)
    from teacher_charles.teaching.phonetics import Phonetics

    monkeypatch.setattr(Phonetics, "transcribe", lambda self, text: f"ipa({text})")
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        body = c.post("/api/define", json={"word": "went"}).json()
    assert body["headwords"][0]["ipa"] == "ipa(go)"


def test_a_miss_is_an_answer_not_an_error(client):
    response = client.post("/api/define", json={"word": "the"})
    assert response.status_code == 200
    assert response.json()["found"] is False


def test_the_route_refuses_empty_and_oversized_words(client):
    assert client.post("/api/define", json={"word": ""}).status_code == 422
    assert client.post("/api/define", json={"word": "a" * 65}).status_code == 422


def test_the_route_records_nothing(client):
    conn = client.app.state.db
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]

    def counts() -> list[int]:
        return [conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables]

    before = counts()
    client.post("/api/define", json={"word": "chamber"})
    assert counts() == before


def test_the_page_loads_the_card_when_the_file_is_there(client):
    page = client.get(f"/lessons/{SLUG}").text
    assert "/static/word-card.js" in page
    assert "Rest the pointer on a word" in page


def test_without_the_file_the_page_has_no_card_and_health_says_how(tmp_path):
    settings = settings_for(tmp_path)
    put_copy(settings)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        page = c.get(f"/lessons/{SLUG}").text
        assert "/static/word-card.js" not in page
        assert "Rest the pointer on a word" not in page
        assert c.post("/api/define", json={"word": "chamber"}).status_code == 503
        health = c.get("/health").json()["dictionary"]
    assert health["installed"] is False and health["ok"] is False
    assert "fetch_dictionary" in health["detail"]
    # The lesson is whole without the card: the status does not depend on it.
    assert not settings.dictionary_path.exists()


def test_switched_off_there_is_no_card_and_the_route_says_how(tmp_path, lookup_path):
    settings = settings_for(tmp_path, dictionary_path=lookup_path, dictionary_enabled=False)
    put_copy(settings)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        assert "/static/word-card.js" not in c.get(f"/lessons/{SLUG}").text
        response = c.post("/api/define", json={"word": "chamber"})
        assert response.status_code == 503
        assert "TC_DICTIONARY_ENABLED" in response.json()["how"]
        assert c.get("/health").json()["dictionary"] == {"ok": True, "enabled": False}


def test_without_a_full_text_the_quotes_beside_the_points_still_get_the_card(tmp_path, lookup_path):
    settings = settings_for(tmp_path, dictionary_path=lookup_path)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        page = c.get(f"/lessons/{SLUG}").text
    # No copy: no full text, but the points still quote the speech, bare.
    assert 'id="full-text"' not in page
    assert 'class="from-text"' in page
    assert "/static/word-card.js" in page


def test_the_card_script_cuts_the_point_quotes_into_words():
    # The quotes are cut in the browser, so the server's markup stays the
    # runs and marks the lesson tests read back; this holds the selectors
    # that join the two halves.
    script = (PACKAGE_ROOT / "web" / "static" / "word-card.js").read_text(encoding="utf-8")
    assert '".point .from-text"' in script
    assert ".from-text .w" in script


def test_the_conversation_loads_the_card_when_the_file_is_there(tmp_path, lookup_path):
    settings = settings_for(tmp_path, dictionary_path=lookup_path)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        page = c.get("/").text
    assert "/static/word-card.js" in page
    # After app.js: the transcript it hooks into is on the page by then.
    assert page.index("/static/app.js") < page.index("/static/word-card.js")


def test_without_the_file_the_conversation_has_no_card(tmp_path):
    with TestClient(create_app(settings_for(tmp_path), provider=FakeProvider())) as c:
        assert "/static/word-card.js" not in c.get("/").text


def test_the_card_reads_the_messages_from_the_caret_and_never_rewrites_them():
    # The conversation is an aria-live region: cutting a finished bubble into
    # spans would read to a screen reader as new content, and it would say
    # Charles's answer twice. The words of a message are found from the
    # caret instead, and the bubble's text is never replaced.
    script = (PACKAGE_ROOT / "web" / "static" / "word-card.js").read_text(encoding="utf-8")
    assert '".turn .what"' in script
    assert "caretPositionFromPoint" in script
    assert "cutIntoWords(what" not in script
    assert "quotes.forEach(cutIntoWords)" in script


def test_the_dictionary_does_not_decide_the_health_status(tmp_path, lookup_path):
    with_file = settings_for(tmp_path, dictionary_path=lookup_path)
    without = settings_for(tmp_path, dictionary_path=tmp_path / "absent.sqlite")
    with TestClient(create_app(with_file, provider=FakeProvider())) as a:
        status_with = a.get("/health").json()["status"]
    with TestClient(create_app(without, provider=FakeProvider())) as b:
        status_without = b.get("/health").json()["status"]
    assert status_with == status_without


def test_the_lookup_default_lives_in_the_cache_not_the_repo_or_student_data(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    path = Settings(_env_file=None).resolved_dictionary_path
    assert path.is_relative_to(tmp_path / "cache")
