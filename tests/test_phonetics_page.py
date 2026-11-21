"""The phonetics page: a box for a sentence, and its IPA under it.

espeak is a base dependency (see ``test_phonetics.py``), so these run with no
weights downloaded and are not integration tests. A test that needs espeak to
have loaded skips itself on a machine where it did not, as the lesson's IPA
test does.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from teacher_charles.app import PHONETICS_MAX_CHARS, create_app
from teacher_charles.config import Settings
from teacher_charles.llm import FakeProvider
from teacher_charles.teaching.phonetics import Phonetics, PhoneticsUnavailable, ipa_pairs
from teacher_charles.voice.base import TimedWord


def settings_for(tmp_path, **overrides) -> Settings:
    base = {
        "db_path": tmp_path / "phonetics.db",
        "provider": "fake",
        "llm_warmup": False,
        "phonetics_enabled": True,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(settings_for(tmp_path), provider=FakeProvider())) as handle:
        yield handle


def spelled(client, text: str) -> dict:
    response = client.post("/api/phonetics", json={"text": text})
    if response.status_code == 503:
        pytest.skip(f"espeak did not load on this machine: {response.json()['detail']}")
    assert response.status_code == 200
    return response.json()


# --- ipa_pairs ------------------------------------------------------------- #


def test_pairs_are_one_to_one_when_espeak_keeps_words_apart():
    pairs = ipa_pairs(["I", "came."], ["aɪ", "kˈeɪm."], [[0], [1]])
    assert pairs == [{"text": "I", "ipa": "aɪ"}, {"text": "came.", "ipa": "kˈeɪm."}]


def test_words_espeak_runs_together_are_one_pair():
    """'on the' over ɔnðə once, not ɔnðə under each of the two words."""
    words = ["lost", "on", "the", "plaza."]
    tokens = ["lˈɔst", "ɔnðə", "plˈɑːzə."]
    pairs = ipa_pairs(words, tokens, [[0], [1], [1], [2]])
    assert pairs == [
        {"text": "lost", "ipa": "lˈɔst"},
        {"text": "on the", "ipa": "ɔnðə"},
        {"text": "plaza.", "ipa": "plˈɑːzə."},
    ]


def test_one_word_over_several_ipa_words_keeps_them_all():
    pairs = ipa_pairs(["don't"], ["dˈoʊnt", "x"], [[0, 1]])
    assert pairs == [{"text": "don't", "ipa": "dˈoʊnt x"}]


def test_a_word_that_owns_nothing_gets_no_guess():
    assert ipa_pairs(["a", "b"], [], [[], []]) == [
        {"text": "a", "ipa": ""},
        {"text": "b", "ipa": ""},
    ]


# --- the page -------------------------------------------------------------- #


def test_the_page_has_the_box_and_the_script(client):
    page = client.get("/phonetics")
    assert page.status_code == 200
    assert 'id="phonetics-text"' in page.text
    assert "/static/phonetics.js" in page.text
    # The sentence goes in a POST body: the form must not submit it as a GET.
    assert 'method="get"' not in page.text.lower()


def test_the_conversation_links_to_the_page(client):
    assert 'href="/phonetics"' in client.get("/").text


def test_the_conversation_links_to_the_ipa_chart(client):
    page = client.get("/").text
    assert 'href="https://www.ipachart.com/"' in page
    # The class the Phonetics switch hides it by, with the IPA lines.
    assert 'class="button ipa-chart"' in page


def test_with_a_voice_the_page_asks_for_listen_buttons(tmp_path):
    """phonetics.js draws Listen only where /api/tts can answer."""
    settings = settings_for(tmp_path, voice_enabled=True)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        page = c.get("/phonetics").text
        assert 'data-voice="on"' in page
        # What a Listen button posts is a sentence the route returns, so the
        # route the button calls has to take it.
        [[sentence]] = spelled(c, "I thought it through.")["lines"]
        spoken = c.post("/api/tts", json={"text": sentence["text"]})
    assert spoken.status_code == 200
    assert spoken.headers["content-type"] == "audio/wav"


def test_the_word_estimate_loads_before_the_scripts_that_use_it(tmp_path):
    """phonetics.js and lesson.js both read ``window.TCWordStarts`` at once.

    Deferred scripts run in document order, so word-starts.js has to come
    first on both pages: after them, the lesson reader would stop at its
    first line and the phonetics page would light no word.
    """
    settings = settings_for(tmp_path, voice_enabled=True)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        phonetics = c.get("/phonetics").text
        # data/lessons/ is versioned, so there is always a lesson to open.
        slug = c.get("/api/lessons").json()["lessons"][0]["slug"]
        lesson = c.get(f"/lessons/{slug}").text
    assert phonetics.index("/static/word-starts.js") < phonetics.index("/static/phonetics.js")
    assert lesson.index("/static/word-starts.js") < lesson.index("/static/lesson.js")


def test_with_a_voice_the_page_has_the_lessons_speed_bar(tmp_path):
    """One bar, one script and one remembered setting for both pages."""
    settings = settings_for(tmp_path, voice_enabled=True, tts_speed=0.85)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        phonetics = c.get("/phonetics").text
        slug = c.get("/api/lessons").json()["lessons"][0]["slug"]
        lesson = c.get(f"/lessons/{slug}").text
        # What the bar sends, the route the Listen button calls takes.
        spoken = c.post("/api/tts", json={"text": "I thought it through.", "speed": 0.7})
        assert c.app.state.voice.tts.last_call.speed == 0.7
    assert spoken.status_code == 200
    bar = 'class="speech-speed" data-default-speed="0.85"'
    assert bar in phonetics
    assert bar in lesson


def test_the_speed_bar_loads_before_the_scripts_that_read_it(tmp_path):
    """phonetics.js and lesson.js read ``window.TCSpeechSpeed``.

    lesson.js reads it once, at load: after it, the lesson's speakers would
    keep the server's speed and 1x whatever the bar said.
    """
    settings = settings_for(tmp_path, voice_enabled=True)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        phonetics = c.get("/phonetics").text
        slug = c.get("/api/lessons").json()["lessons"][0]["slug"]
        lesson = c.get(f"/lessons/{slug}").text
    assert phonetics.index("/static/speech-speed.js") < phonetics.index("/static/phonetics.js")
    assert lesson.index("/static/speech-speed.js") < lesson.index("/static/lesson.js")


def test_without_a_voice_there_is_no_speed_bar(client):
    page = client.get("/phonetics").text
    assert 'class="speech-speed"' not in page
    assert "speech-speed.js" not in page


def test_without_a_voice_the_word_estimate_is_not_loaded(client):
    assert "word-starts.js" not in client.get("/phonetics").text


def test_the_timings_count_the_words_the_pairs_hold(tmp_path):
    """The page lights cell k for timing k, so the two must count alike.

    phonetics.js maps word k to its cell by splitting each pair's text on
    spaces; /api/tts/{id}/words gives one start per whitespace-split word of
    the sentence. A sentence where espeak joins two words ("on the") is the
    case where a count by pair would be off by one from then on.
    """
    settings = settings_for(tmp_path, voice_enabled=True)
    with TestClient(create_app(settings, provider=FakeProvider())) as c:
        [[sentence]] = spelled(c, "We met on the plaza.")["lines"]
        words = sentence["text"].split()
        c.app.state.voice.stt.heard = [
            TimedWord(w, 0.3 * i, 0.3 * i + 0.25) for i, w in enumerate(words)
        ]
        spoken = c.post("/api/tts", json={"text": sentence["text"], "timings": True})
        starts = c.get(f"/api/tts/{spoken.headers['x-speech-id']}/words").json()["starts"]
    in_cells = sum(len(pair["text"].split(" ")) for pair in sentence["pairs"])
    if len(sentence["pairs"]) == len(words):
        pytest.skip("espeak joined no words here; the case under test did not arise")
    assert len(starts) == len(words) == in_cells


def test_without_a_voice_there_are_no_listen_buttons(client):
    """Voice is off by default; a button that always answers 503 is worse than none."""
    assert "data-voice" not in client.get("/phonetics").text


def test_with_the_ipa_off_the_page_says_how_and_the_route_answers_503(tmp_path):
    settings = settings_for(tmp_path, phonetics_enabled=False)
    with TestClient(create_app(settings, provider=FakeProvider())) as off:
        page = off.get("/phonetics").text
        # No IPA on screen, so no chart to look its symbols up in.
        assert "ipachart.com" not in off.get("/").text
        assert 'id="phonetics-text"' not in page
        assert "TC_PHONETICS_ENABLED" in page
        response = off.post("/api/phonetics", json={"text": "Hello."})
    assert response.status_code == 503
    assert "TC_PHONETICS_ENABLED" in response.json()["how"]


# --- the route ------------------------------------------------------------- #


def test_the_route_returns_the_sentence_its_ipa_and_its_pairs(client):
    body = spelled(client, "I thought it through.")
    assert body["language"] == "en-us"
    [[sentence]] = body["lines"]
    assert sentence["text"] == "I thought it through."
    assert "θ" in sentence["ipa"]
    # The pairs rejoin to the sentence, and their IPA to the sentence's IPA:
    # nothing is dropped or invented between the two views.
    assert " ".join(p["text"] for p in sentence["pairs"]) == sentence["text"]
    assert " ".join(p["ipa"] for p in sentence["pairs"]) == sentence["ipa"]


def test_the_ipa_is_the_conversations_ipa(client):
    """Same espeak as the line under every chat bubble."""
    body = spelled(client, "Though it was tough.")
    assert body["lines"][0][0]["ipa"] == Phonetics().transcribe("Though it was tough.")


def test_lines_and_sentences_come_back_apart(client):
    body = spelled(client, "Hello there. How are you?\n\nFine.")
    assert [[s["text"] for s in line] for line in body["lines"]] == [
        ["Hello there.", "How are you?"],
        ["Fine."],
    ]


def test_a_blank_text_is_nothing_to_transcribe(client):
    response = client.post("/api/phonetics", json={"text": "   \n "})
    assert response.status_code == 200
    assert response.json()["lines"] == []


def test_a_long_run_on_gets_its_ipa_without_pairs(client):
    text = " ".join(["word"] * 100)
    [[sentence]] = spelled(client, text)["lines"]
    assert sentence["ipa"]
    assert sentence["pairs"] == []


def test_the_route_refuses_empty_and_oversized_text(client):
    assert client.post("/api/phonetics", json={"text": ""}).status_code == 422
    too_long = "a" * (PHONETICS_MAX_CHARS + 1)
    assert client.post("/api/phonetics", json={"text": too_long}).status_code == 422


def test_espeak_that_does_not_load_is_a_503_not_an_empty_answer(tmp_path, monkeypatch):
    """An empty IPA because espeak never came up must not read as 'no sounds'."""

    def broken(self):
        raise PhoneticsUnavailable("espeak broken on purpose")

    monkeypatch.setattr(Phonetics, "_load", broken)
    with TestClient(create_app(settings_for(tmp_path), provider=FakeProvider())) as c:
        response = c.post("/api/phonetics", json={"text": "Hello."})
    assert response.status_code == 503
    assert "broken on purpose" in response.json()["detail"]


def test_the_route_records_nothing(client):
    """Looking up how a sentence sounds is not a turn, a correction or a review."""
    conn = client.app.state.db
    tables = ("turn", "correction", "rule_review", "level_signal")

    def counts() -> list[int]:
        return [conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables]

    before = counts()
    client.post("/api/phonetics", json={"text": "I have 30 years."})
    assert counts() == before
    assert client.app.state.provider.calls == []
