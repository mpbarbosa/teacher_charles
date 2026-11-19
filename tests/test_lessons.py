"""Tests for the lesson shelf: the curated lessons, their guards, and the routes."""

from __future__ import annotations

import copy
from html import escape
from html.parser import HTMLParser
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from teacher_charles.app import TIMED_SPEECH_KEPT, create_app
from teacher_charles.config import REPO_ROOT, Settings
from teacher_charles.llm import FakeProvider
from teacher_charles.teaching import lessons as lessons_module
from teacher_charles.teaching.contrastive import load_base as load_contrastive
from teacher_charles.teaching.lessons import (
    ASPECTS,
    CITE_BUDGET,
    MAX_CITE_WORDS,
    Lesson,
    LessonError,
    load_lessons,
    load_shelf,
    local_text_path,
    read_local_text,
    word_count,
)
from teacher_charles.teaching.vocabulary import load_base as load_vocabulary
from teacher_charles.voice.base import TimedWord, VoiceBadAudio

LESSONS_DIR = REPO_ROOT / "data" / "lessons"
SLUG = "mon-mothma-senate-speech"


def valid_lesson(**overrides) -> dict:
    lesson = {
        "id": "lesson.sample",
        "title": "A sample",
        "summary": "A lesson for tests.",
        "cefr": "B2",
        "source": {
            "title": "A speech",
            "publication": "A paper",
            "author": "Someone",
            "published": "2025-01-01",
            "url": "https://example.com/speech",
            "words": 100,
        },
        "sections": [
            {
                "aspect": "syntax",
                "title": "Syntax",
                "points": [{"heading": "h", "body": "b", "cites": ["two words"]}],
            },
            {"aspect": "lexicon", "title": "Lexicon", "points": [{"heading": "h", "body": "b"}]},
        ],
    }
    lesson.update(overrides)
    return lesson


def write_lesson(directory: Path, lesson: dict, name: str = "sample") -> None:
    directory.mkdir(exist_ok=True)
    (directory / f"{name}.yaml").write_text(yaml.safe_dump(lesson), encoding="utf-8")


# --- the shipped shelf ----------------------------------------------------- #


def test_the_shipped_lessons_load():
    shelf = load_shelf(LESSONS_DIR)
    assert len(shelf) >= 1
    assert shelf.get(SLUG) is not None


def test_the_speech_lesson_covers_all_six_aspects():
    lesson = load_shelf(LESSONS_DIR).get(SLUG)
    assert [s.aspect for s in lesson.ordered_sections()] == list(ASPECTS)
    assert lesson.words and lesson.sounds and lesson.questions and lesson.practice


def test_the_speech_lesson_stays_inside_its_citation_budget():
    lesson = load_shelf(LESSONS_DIR).get(SLUG)
    assert lesson.cited_words <= int(lesson.source.words * CITE_BUDGET)


def test_lesson_bridges_resolve_in_the_other_bases():
    """A ``vocab_id`` or ``rule_id`` that names nothing would be a link to an
    empty page — the lesson and the chat would teach the word differently."""
    vocabulary = load_vocabulary(REPO_ROOT / "data" / "vocab")
    contrastive = load_contrastive(REPO_ROOT / "data" / "contrastive")
    rule_ids = {item.id for item in contrastive}
    for lesson in load_shelf(LESSONS_DIR):
        for note in lesson.words:
            if note.vocab_id:
                assert vocabulary.by_id(note.vocab_id).word == note.word
            for rule_id in note.rule_ids:
                assert rule_id in rule_ids, f"{lesson.id}: {rule_id} is not in the base"


def test_shipped_questions_point_at_a_real_option():
    for lesson in load_shelf(LESSONS_DIR):
        for q in lesson.questions:
            assert 0 <= q.answer < len(q.options)


# --- the guards that keep a lesson from copying its source ---------------- #


def test_the_source_has_no_place_for_the_text():
    lesson = valid_lesson()
    lesson["source"]["text"] = "The whole speech, pasted."
    with pytest.raises(ValueError, match="text"):
        Lesson.model_validate(lesson)


def test_a_citation_longer_than_the_limit_is_refused():
    lesson = valid_lesson()
    too_long = " ".join(["word"] * (MAX_CITE_WORDS + 1))
    lesson["sections"][0]["points"][0]["cites"] = [too_long]
    with pytest.raises(ValueError, match="the limit is"):
        Lesson.model_validate(lesson)


def test_a_citation_at_the_limit_is_accepted():
    lesson = valid_lesson()
    lesson["sections"][0]["points"][0]["cites"] = [" ".join(["word"] * MAX_CITE_WORDS)]
    Lesson.model_validate(lesson)


def test_short_citations_cannot_add_up_to_a_copy():
    """Five words at a time, a lesson could still rebuild its source. The
    budget is the guard against that, and this test is the one holding it."""
    lesson = valid_lesson()
    budget = int(lesson["source"]["words"] * CITE_BUDGET)
    lesson["sections"][0]["points"][0]["cites"] = ["one two"] * (budget // 2 + 1)
    with pytest.raises(ValueError, match="budget"):
        Lesson.model_validate(lesson)


def test_word_count_counts_contractions_as_one_word():
    assert word_count("we've helped create") == 3


# --- the rest of the shape ------------------------------------------------- #


def test_an_answer_outside_the_options_is_refused():
    lesson = valid_lesson(
        questions=[{"prompt": "p", "options": ["a", "b"], "answer": 2, "why": "w"}]
    )
    with pytest.raises(ValueError, match="only 2 options"):
        Lesson.model_validate(lesson)


def test_two_sections_for_one_aspect_are_refused():
    lesson = valid_lesson()
    lesson["sections"].append(copy.deepcopy(lesson["sections"][0]))
    with pytest.raises(ValueError, match="more than one section"):
        Lesson.model_validate(lesson)


def test_sections_are_served_in_teaching_order():
    lesson = Lesson.model_validate(valid_lesson())
    assert [s.aspect for s in lesson.ordered_sections()] == ["lexicon", "syntax"]


def test_the_id_must_match_the_file_name(tmp_path):
    write_lesson(tmp_path / "lessons", valid_lesson(), name="other")
    with pytest.raises(LessonError, match="lesson.other"):
        load_lessons(tmp_path / "lessons")


def test_every_problem_is_reported_at_once(tmp_path):
    directory = tmp_path / "lessons"
    write_lesson(directory, valid_lesson(cefr="Z9"), name="sample")
    (directory / "broken.yaml").write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(LessonError) as raised:
        load_lessons(directory)
    assert "2 problem(s)" in str(raised.value)


def test_a_missing_directory_is_an_error(tmp_path):
    with pytest.raises(LessonError):
        load_lessons(tmp_path / "nowhere")


# --- routes ----------------------------------------------------------------- #


def settings_for(tmp_path, **overrides) -> Settings:
    base = {
        "db_path": tmp_path / "lessons.db",
        "provider": "fake",
        "llm_warmup": False,
        "phonetics_enabled": False,
        # An empty corpus by default: the developer's own copies must not
        # decide what a route test sees.
        "corpus_dir": tmp_path / "corpus",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(settings_for(tmp_path), provider=FakeProvider())) as c:
        yield c


def test_the_shelf_page_lists_the_lesson(client):
    response = client.get("/lessons")
    assert response.status_code == 200
    assert f'href="/lessons/{SLUG}"' in response.text


def test_the_lesson_page_links_the_source_and_shows_every_section(client):
    lesson = load_shelf(LESSONS_DIR).get(SLUG)
    response = client.get(f"/lessons/{SLUG}")
    assert response.status_code == 200
    assert lesson.source.url in response.text
    for aspect in ASPECTS:
        assert f'id="{aspect}"' in response.text


def test_practice_hands_the_starter_to_the_chat_unsent(client):
    response = client.get(f"/lessons/{SLUG}")
    assert 'href="/?draft=Tired%20as%20I%20was%2C%20"' in response.text


def test_an_unknown_lesson_is_404(client):
    assert client.get("/lessons/nope").status_code == 404
    assert client.get("/api/lessons/nope").status_code == 404


def test_the_api_serves_the_lesson_without_curation_notes(client):
    body = client.get(f"/api/lessons/{SLUG}").json()
    assert body["slug"] == SLUG
    assert "notes" not in body
    assert [s["aspect"] for s in body["sections"]] == list(ASPECTS)
    # IPA off: the key is there and empty, the page draws nothing.
    assert all(note["ipa"] == "" for note in body["words"] + body["sounds"])


def test_the_api_lists_cards(client):
    cards = client.get("/api/lessons").json()["lessons"]
    assert any(card["slug"] == SLUG for card in cards)


def test_the_conversation_page_links_the_shelf(client):
    assert 'href="/lessons"' in client.get("/").text


def test_health_reports_the_shelf(client):
    body = client.get("/health").json()
    assert body["lessons"]["ok"] is True
    assert body["lessons"]["items"] >= 1


def test_a_broken_shelf_shows_in_health_and_leaves_the_page_empty(tmp_path, monkeypatch):
    def broken(_directory=None):
        raise LessonError("1 problem(s): broken on purpose")

    monkeypatch.setattr("teacher_charles.app.load_shelf", broken)
    with TestClient(create_app(settings_for(tmp_path), provider=FakeProvider())) as c:
        lessons = c.get("/health").json()["lessons"]
        assert lessons["ok"] is False
        assert "broken on purpose" in lessons["detail"]
        assert "No lessons on the shelf yet." in c.get("/lessons").text


def test_the_ipa_comes_from_espeak_when_it_is_on(tmp_path):
    """The lesson's IPA is the conversation's IPA: same espeak, same symbols."""
    settings = settings_for(tmp_path, phonetics_enabled=True)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        body = c.get(f"/api/lessons/{SLUG}").json()
    truth = next(note for note in body["sounds"] if note["word"] == "truth")
    if not truth["ipa"]:
        pytest.skip("espeak did not load on this machine")
    assert "θ" in truth["ipa"]


# --- the local copy of the source ------------------------------------------ #

# Placeholder prose, ours: the tests must not need the real speech, and the
# repo must not carry it.
PLACEHOLDER = "First paragraph of a stand-in text.\n\nSecond one, wrapped\nacross two lines.\n"


class _Reading(HTMLParser):
    """The full text as the page draws it, read back as structure.

    ``text``: paragraphs -> sentences -> ``{"s": data-s, "words": [...],
    "ipa_of": [...]}``. ``ipa``: one ``{data-s: [IPA words]}`` per IPA line.
    Asserting on this rather than on exact markup keeps the tests about what
    the reader relies on -- ids that line up, words that rejoin to the text --
    and not about whitespace in the template.
    """

    def __init__(self) -> None:
        super().__init__()
        self.text: list[list[dict]] = []
        self.ipa: list[dict[str, list[str]]] = []
        self._stack: list[tuple[str, dict]] = []
        self._in_reading = False
        self._word: dict | None = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if a.get("id") == "full-text-reading":
            self._in_reading = True
        if not self._in_reading:
            return
        cls = a.get("class", "")
        if tag == "p":
            self.text.append([])
        elif tag == "div" and "ipa" in cls.split():
            self.ipa.append({})
        elif tag == "span" and cls == "sentence":
            self.text[-1].append({"s": a["data-s"], "words": [], "ipa_of": []})
        elif tag == "span" and "data-s" in a and self.ipa:
            self.ipa[-1][a["data-s"]] = []
            self._ipa_s = a["data-s"]
        elif tag == "span" and cls == "w":
            self._word = {"attrs": a, "text": ""}
        self._stack.append((tag, a))

    def handle_data(self, data):
        if self._word is not None:
            self._word["text"] += data

    def handle_endtag(self, tag):
        if not self._in_reading or not self._stack:
            return
        opened, a = self._stack.pop()
        if opened == "div" and a.get("id") == "full-text-reading":
            self._in_reading = False
        if opened == "span" and a.get("class") == "w" and self._word is not None:
            w = self._word
            self._word = None
            if "data-t" in w["attrs"]:
                self.ipa[-1][self._ipa_s].append(w["text"])
            else:
                sentence = self.text[-1][-1]
                assert int(w["attrs"]["data-w"]) == len(sentence["words"])
                sentence["words"].append(w["text"])
                if "data-ipa" in w["attrs"]:
                    sentence["ipa_of"].append([int(t) for t in w["attrs"]["data-ipa"].split()])


def reading_of(page: str) -> _Reading:
    parser = _Reading()
    parser.feed(page)
    return parser


def sentences_of(page: str) -> list[list[str]]:
    """Each paragraph's sentences, as the words on the page rejoin them."""
    return [[" ".join(s["words"]) for s in p] for p in reading_of(page).text]


def shipped_lesson() -> Lesson:
    return load_shelf(LESSONS_DIR).get(SLUG)


def put_copy(corpus_dir: Path, text: str, lesson: Lesson | None = None) -> Path:
    path = local_text_path(corpus_dir, lesson or shipped_lesson())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_no_copy_means_no_text(tmp_path):
    assert read_local_text(tmp_path, shipped_lesson()) is None


def test_an_empty_copy_means_no_text(tmp_path):
    put_copy(tmp_path, "\n  \n")
    assert read_local_text(tmp_path, shipped_lesson()) is None


def test_a_copy_that_is_not_utf8_means_no_text(tmp_path):
    path = local_text_path(tmp_path, shipped_lesson())
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xff\xfe\x00broken")
    assert read_local_text(tmp_path, shipped_lesson()) is None


def test_the_copy_lives_under_corpus_lessons_by_slug(tmp_path):
    assert local_text_path(tmp_path, shipped_lesson()) == tmp_path / "lessons" / f"{SLUG}.txt"


def test_paragraphs_split_on_blank_lines_and_unwrap(tmp_path):
    put_copy(tmp_path, "\ufeff" + PLACEHOLDER.replace("\n", "\r\n"))
    text = read_local_text(tmp_path, shipped_lesson())
    assert text.paragraphs == (
        "First paragraph of a stand-in text.",
        "Second one, wrapped across two lines.",
    )
    assert text.words == 13


def test_a_copy_of_the_wrong_length_is_flagged_not_refused(tmp_path):
    lesson = shipped_lesson()
    put_copy(tmp_path, PLACEHOLDER)
    assert read_local_text(tmp_path, lesson).matches is False
    put_copy(tmp_path, " ".join(["word"] * lesson.source.words))
    assert read_local_text(tmp_path, lesson).matches is True


def test_the_page_shows_the_copy_when_the_machine_holds_one(tmp_path):
    settings = settings_for(tmp_path)
    put_copy(settings.resolved_corpus_dir, PLACEHOLDER)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        page = c.get(f"/lessons/{SLUG}").text
    assert 'id="full-text"' in page
    assert sentences_of(page)[1] == ["Second one, wrapped across two lines."]
    assert 'href="#full-text"' in page
    # The link stays: the copy is a convenience, the source is still credited.
    assert shipped_lesson().source.url in page


def test_without_a_copy_the_page_is_what_a_clone_shows(client):
    page = client.get(f"/lessons/{SLUG}").text
    assert 'id="full-text"' not in page
    assert "Read the text first:" in page


def test_the_copy_never_reaches_the_api(tmp_path):
    """/api/lessons is the curated lesson; somebody else's text is not ours to serve."""
    settings = settings_for(tmp_path)
    put_copy(settings.resolved_corpus_dir, PLACEHOLDER)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        body = c.get(f"/api/lessons/{SLUG}").text
    assert "stand-in text" not in body


def test_the_copy_is_read_per_request_not_cached(tmp_path):
    settings = settings_for(tmp_path)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        assert 'id="full-text"' not in c.get(f"/lessons/{SLUG}").text
        put_copy(settings.resolved_corpus_dir, PLACEHOLDER)
        assert 'id="full-text"' in c.get(f"/lessons/{SLUG}").text


def test_the_corpus_lessons_folder_is_not_versioned():
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "corpus/*" in ignored
    assert not any(line.startswith("!corpus/lessons") for line in ignored)


def test_the_section_links_are_separate_words(client):
    """The anchors used to be glued together, so without CSS they read as one word."""
    page = client.get(f"/lessons/{SLUG}").text
    assert "</a><a" not in page.split('class="aspects"')[1].split("</nav>")[0]


def test_the_module_exports_what_it_promises():
    for name in lessons_module.__all__:
        assert hasattr(lessons_module, name)


def test_the_stylesheet_is_revalidated_not_cached_blind(client):
    # Without this the browser kept a stale style.css and the lesson's links
    # came out in its default blue on the dark background.
    response = client.get("/static/style.css")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"


# --- the speaker beside each word ------------------------------------------ #


def test_without_voice_the_words_have_no_speaker(client):
    """/api/tts is 503 with voice off, so a button would only ever fail."""
    body = client.get(f"/lessons/{SLUG}").text
    assert 'class="say"' not in body
    assert "lesson.js" not in body


def test_with_voice_every_word_and_sound_has_a_speaker(tmp_path):
    lesson = load_shelf(LESSONS_DIR).get(SLUG)
    settings = settings_for(tmp_path, voice_enabled=True)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        body = c.get(f"/lessons/{SLUG}").text
        assert "/static/lesson.js" in body
        for note in [*lesson.words, *lesson.sounds]:
            assert f'data-say="{escape(note.word)}"' in body
        # The button's text goes to the same route the chat's voice uses.
        word = lesson.words[0].word
        spoken = c.post("/api/tts", json={"text": word})
        assert spoken.status_code == 200
        assert spoken.headers["content-type"] == "audio/wav"


def test_the_shelf_page_has_no_speaker_even_with_voice(tmp_path):
    settings = settings_for(tmp_path, voice_enabled=True)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        assert "lesson.js" not in c.get("/lessons").text


# --- the microphone beside the speaker -------------------------------------- #


def test_without_voice_the_words_have_no_microphone(client):
    """/api/stt is 503 with voice off, the speaker's reason again."""
    body = client.get(f"/lessons/{SLUG}").text
    assert 'class="record"' not in body
    assert 'class="howto"' not in body


def test_with_voice_every_speaker_has_a_microphone_beside_it(tmp_path):
    lesson = load_shelf(LESSONS_DIR).get(SLUG)
    settings = settings_for(tmp_path, voice_enabled=True)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        body = c.get(f"/lessons/{SLUG}").text
    for note in [*lesson.words, *lesson.sounds]:
        word = escape(note.word)
        speaker = body.index(f'data-say="{word}"')
        microphone = body.index(f'class="record" data-word="{word}"')
        # Beside it: the next button after the speaker, not somewhere on the page.
        assert speaker < microphone
        # The one "<button" between them is the microphone's own opening tag.
        assert body[speaker:microphone].count("<button") == 1


def test_the_page_says_the_take_is_not_a_pronunciation_score(tmp_path):
    """Whisper understanding a word is recognition, and a student reading
    "understood" under a word would take it as a pass. E10/E11 are what would
    judge pronunciation, and they do not exist — so the page says so, once per
    table, where the student reads it before pressing the button."""
    lesson = load_shelf(LESSONS_DIR).get(SLUG)
    settings = settings_for(tmp_path, voice_enabled=True)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        body = c.get(f"/lessons/{SLUG}").text
    tables = bool(lesson.words) + bool(lesson.sounds)
    assert body.count('class="howto"') == tables
    assert body.count("not a pronunciation score") == tables


def test_a_practice_take_records_no_turn(tmp_path):
    """The microphone posts to /api/stt, which writes nothing: a take is
    practice, not evidence — the reason /api/syntax records nothing either."""
    settings = settings_for(tmp_path, voice_enabled=True)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        before = c.app.state.db.execute("SELECT COUNT(*) FROM turn").fetchone()[0]
        heard = c.post("/api/stt", content=b"RIFF fake take")
        assert heard.status_code == 200
        after = c.app.state.db.execute("SELECT COUNT(*) FROM turn").fetchone()[0]
    assert after == before


# --- the reader beside the full text ---------------------------------------- #


def test_with_voice_the_full_text_has_a_reader_in_its_own_voice(tmp_path):
    settings = settings_for(tmp_path, voice_enabled=True, tts_reader_voice="bf_isabella")
    put_copy(settings.resolved_corpus_dir, PLACEHOLDER)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        page = c.get(f"/lessons/{SLUG}").text
    assert 'class="read-aloud"' in page
    # The reader's voice is the setting, not Charles's.
    assert 'data-voice="bf_isabella"' in page
    assert f'data-voice="{settings.tts_voice}"' not in page
    # The paragraphs stay plain: the script reads them from the page.
    assert sentences_of(page)[1] == ["Second one, wrapped across two lines."]


def test_the_reader_is_a_female_voice_by_default():
    """Kokoro's names carry the sex in the second letter: bf_ is British female."""
    voice = Settings(_env_file=None).tts_reader_voice
    assert voice[1] == "f"
    assert voice != Settings(_env_file=None).tts_voice


def test_without_voice_the_full_text_has_no_reader(tmp_path):
    """/api/tts is 503 with voice off, so the button would only ever fail."""
    settings = settings_for(tmp_path)
    put_copy(settings.resolved_corpus_dir, PLACEHOLDER)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        page = c.get(f"/lessons/{SLUG}").text
    assert 'id="full-text"' in page
    assert "read-aloud" not in page


def test_without_a_copy_there_is_nothing_to_read(tmp_path):
    settings = settings_for(tmp_path, voice_enabled=True)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        assert "read-aloud" not in c.get(f"/lessons/{SLUG}").text


# --- the phonetics of the full text ----------------------------------------- #


def test_with_ipa_the_full_text_has_a_phonetics_button_and_hidden_lines(tmp_path):
    settings = settings_for(tmp_path, phonetics_enabled=True)
    put_copy(settings.resolved_corpus_dir, PLACEHOLDER)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        page = c.get(f"/lessons/{SLUG}").text
    assert 'class="show-ipa"' in page
    # The layout switch (after the paragraph / between the lines) waits,
    # hidden, until the IPA is showing (full-text.js).
    assert 'class="ipa-layout"' in page
    assert "/static/full-text.js" in page
    # One line per paragraph, hidden until asked for, and the paragraph's own
    # markup untouched.
    assert page.count('<div class="ipa" hidden>') == 2
    assert sentences_of(page)[1] == ["Second one, wrapped across two lines."]
    # espeak's reading, not the English: "wrapped" is /ɹ/ and a stress mark.
    assert "ɹˈæpt" in page


def test_without_ipa_the_full_text_has_no_phonetics(tmp_path):
    """The line comes from espeak; with it off there is nothing to show."""
    settings = settings_for(tmp_path)
    put_copy(settings.resolved_corpus_dir, PLACEHOLDER)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        page = c.get(f"/lessons/{SLUG}").text
    assert 'id="full-text"' in page
    assert "show-ipa" not in page
    assert "ipa-layout" not in page
    assert "full-text.js" not in page
    assert '<div class="ipa" hidden>' not in page


def test_the_phonetics_never_reach_the_api(tmp_path):
    """The IPA of somebody else's text is that text, one step removed."""
    settings = settings_for(tmp_path, phonetics_enabled=True)
    put_copy(settings.resolved_corpus_dir, PLACEHOLDER)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        body = c.get(f"/api/lessons/{SLUG}").text
    assert "ɹˈæpt" not in body


# --- the reader and the IPA, sentence by sentence --------------------------- #

TWO_SENTENCES = "I came here as a child. I have almost no memories that pre-date my arrival.\n"


def test_the_copy_is_cut_into_sentences_that_join_back_to_it(tmp_path):
    settings = settings_for(tmp_path)
    put_copy(settings.resolved_corpus_dir, TWO_SENTENCES)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        page = c.get(f"/lessons/{SLUG}").text
    reading = reading_of(page)
    assert [s["s"] for s in reading.text[0]] == ["0.0", "0.1"]
    assert sentences_of(page) == [
        ["I came here as a child.", "I have almost no memories that pre-date my arrival."]
    ]
    # Words are the whitespace split, the split /api/tts times: "pre-date" is one.
    assert "pre-date" in reading.text[0][1]["words"]


def test_each_sentence_and_its_ipa_share_an_id(tmp_path):
    """The reader lights both by `data-s`; a mismatch would light the wrong IPA."""
    settings = settings_for(tmp_path, phonetics_enabled=True)
    put_copy(settings.resolved_corpus_dir, TWO_SENTENCES)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        page = c.get(f"/lessons/{SLUG}").text
    reading = reading_of(page)
    ipa = reading.ipa[0]
    assert list(ipa) == ["0.0", "0.1"]
    # Each IPA span is that sentence's own: "child" in the first, "arrival" in
    # the second.
    assert any("tʃˈaɪld" in t for t in ipa["0.0"]) and not any("ɚɹˈaɪvəl" in t for t in ipa["0.0"])
    assert any("ɚɹˈaɪvəl" in t for t in ipa["0.1"]) and not any("tʃˈaɪld" in t for t in ipa["0.1"])


def test_each_word_names_the_ipa_words_it_is_part_of(tmp_path):
    """espeak runs "on the" into one IPA word; both written words must light it."""
    settings = settings_for(tmp_path, phonetics_enabled=True)
    put_copy(settings.resolved_corpus_dir, "The truth was lost on the plaza.\n")
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        page = c.get(f"/lessons/{SLUG}").text
    reading = reading_of(page)
    sentence = reading.text[0][0]
    tokens = reading.ipa[0]["0.0"]
    owner = dict(zip(sentence["words"], sentence["ipa_of"], strict=True))
    if len(tokens) == len(sentence["words"]):
        pytest.skip("this espeak does not join 'on the'")
    assert owner["on"] == owner["the"] and len(owner["on"]) == 1
    assert tokens[owner["on"][0]].replace("ˈ", "").replace("ˌ", "") == "ɔnðə"
    # Every word lights something, and no IPA word is left unlit.
    assert all(sentence["ipa_of"])
    assert sorted({t for o in sentence["ipa_of"] for t in o}) == list(range(len(tokens)))


def test_a_short_exclamation_rides_with_the_next_sentence(tmp_path):
    """The chat's segmenter keeps "Yes!" from being a call, and a highlight, of its own."""
    settings = settings_for(tmp_path)
    put_copy(settings.resolved_corpus_dir, "It was genocide. Yes! It was genocide.\n")
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        page = c.get(f"/lessons/{SLUG}").text
    assert sentences_of(page) == [["It was genocide.", "Yes! It was genocide."]]


# --- word timings for the reader -------------------------------------------- #


def test_tts_keeps_the_audio_and_times_it_on_request(tmp_path):
    settings = settings_for(tmp_path, voice_enabled=True)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        stt = c.app.state.voice.stt
        stt.heard = [
            TimedWord("I", 0.0, 0.2),
            TimedWord("came", 0.2, 0.5),
            TimedWord("here.", 0.5, 0.9),
        ]
        spoken = c.post("/api/tts", json={"text": "I came here.", "timings": True})
        assert spoken.status_code == 200
        assert spoken.headers["content-type"] == "audio/wav"
        # The audio comes back before anything is timed: the Whisper pass must
        # not stand between the synthesis and the sound.
        assert stt.timed == []
        words = c.get(f"/api/tts/{spoken.headers['x-speech-id']}/words")
        assert words.status_code == 200
        assert words.json() == {"starts": [0.0, 0.2, 0.5]}
        # The STT listened to the audio the route made for that request.
        assert len(stt.timed) == 1
        assert stt.timed[0].pcm and spoken.content.endswith(stt.timed[0].pcm)


def test_timings_are_handed_out_once(tmp_path):
    """The server holds a sentence only between its audio and its timings."""
    settings = settings_for(tmp_path, voice_enabled=True)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        spoken = c.post("/api/tts", json={"text": "I came here.", "timings": True})
        url = f"/api/tts/{spoken.headers['x-speech-id']}/words"
        assert c.get(url).status_code == 200
        assert c.get(url).status_code == 404
        assert c.get("/api/tts/not-an-id/words").status_code == 404


def test_the_server_keeps_only_a_few_untimed_sentences(tmp_path):
    """A page that never asks for its timings must not grow the server."""
    settings = settings_for(tmp_path, voice_enabled=True)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        ids = [
            c.post("/api/tts", json={"text": f"Sentence {i}.", "timings": True}).headers[
                "x-speech-id"
            ]
            for i in range(TIMED_SPEECH_KEPT + 3)
        ]
        assert len(c.app.state.timed_speech) == TIMED_SPEECH_KEPT
        # The oldest went first; the newest is still there.
        assert c.get(f"/api/tts/{ids[0]}/words").status_code == 404
        assert c.get(f"/api/tts/{ids[-1]}/words").status_code == 200


def test_tts_keeps_nothing_unless_asked(tmp_path):
    """The chat's voice calls /api/tts on every sentence and lights nothing."""
    settings = settings_for(tmp_path, voice_enabled=True)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        plain = c.post("/api/tts", json={"text": "I came here."})
        assert plain.status_code == 200
        assert "x-speech-id" not in plain.headers
        assert len(c.app.state.timed_speech) == 0
        assert c.app.state.voice.stt.timed == []


def test_a_failed_timing_is_no_timings_not_an_error(tmp_path):
    """No timings is a sentence lit whole, never a reader that falls silent."""
    settings = settings_for(tmp_path, voice_enabled=True)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        c.app.state.voice.stt.error = VoiceBadAudio("nothing to hear")
        spoken = c.post("/api/tts", json={"text": "I came here.", "timings": True})
        assert spoken.status_code == 200
        words = c.get(f"/api/tts/{spoken.headers['x-speech-id']}/words")
        assert words.status_code == 200
        assert words.json() == {"starts": []}


def test_without_voice_there_are_no_timings(tmp_path):
    settings = settings_for(tmp_path)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        assert c.get("/api/tts/whatever/words").status_code == 503


# --- starting the reader from a sentence ------------------------------------ #


def test_with_voice_the_page_says_a_sentence_can_be_clicked(tmp_path):
    settings = settings_for(tmp_path, voice_enabled=True)
    put_copy(settings.resolved_corpus_dir, PLACEHOLDER)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        page = c.get(f"/lessons/{SLUG}").text
    assert 'class="read-hint"' in page
    assert "Click any sentence to hear it from there." in page


def test_without_voice_no_sentence_promises_a_reading(tmp_path):
    """No voice, no reader: a hint to click would be a promise nothing keeps."""
    settings = settings_for(tmp_path)
    put_copy(settings.resolved_corpus_dir, PLACEHOLDER)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        page = c.get(f"/lessons/{SLUG}").text
    assert 'id="full-text"' in page
    assert "read-hint" not in page
