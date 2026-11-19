"""Tests for the lesson shelf: the curated lessons, their guards, and the routes."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from teacher_charles.app import create_app
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
    assert "<p>Second one, wrapped across two lines.</p>" in page
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
