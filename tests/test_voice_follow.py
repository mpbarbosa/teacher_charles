"""Where a student reading aloud has got to, from what an STT heard.

No audio and no model: ``last_heard`` takes the text's words and the heard
words, so each thing a reading does to the match is written here as data.
"""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from teacher_charles.app import create_app
from teacher_charles.config import Settings
from teacher_charles.llm import FakeProvider
from teacher_charles.voice import FakeSTT, FakeTTS, Voice, VoiceBadAudio
from teacher_charles.voice.base import TimedWord
from teacher_charles.voice.follow import MIN_ANCHOR, last_heard

TEXT = (
    "Hi, I'm a software engineer with twenty years of building systems that "
    "other people depend on. Most of that time I worked on payments, where a "
    "bug costs money by the minute. What I want next is a team that ships "
    "carefully and talks openly about what went wrong."
).split()


def at(heard: str, words: list[str] = TEXT) -> str | None:
    """The word the light lands on, as text, for a readable assertion."""
    k = last_heard(words, heard.split())
    return None if k is None else f"{k}:{words[k]}"


def test_the_light_lands_on_the_last_word_heard():
    assert at("of building systems that other people") == "13:people"


def test_punctuation_and_case_do_not_matter():
    assert at("hi im a software engineer") == "4:engineer"


def test_a_misheard_word_in_the_middle_does_not_stop_the_chain():
    # "payments" heard as "pavements": the run after it still continues.
    assert at("I worked on pavements where a bug costs") == "27:costs"


def test_words_heard_together_or_apart_are_the_same_letters():
    assert at("soft ware engineer with twenty") == "6:twenty"


def test_silence_places_nothing():
    assert last_heard(TEXT, []) is None


def test_speech_that_is_not_the_text_places_nothing():
    # Whisper's favourite hallucination over a quiet room.
    assert at("Thanks for watching!") is None


def test_a_few_letters_in_common_are_not_enough_to_move_the_light():
    # "the time" is in the text, but two short words are chance, not a place.
    assert len("thetime") < MIN_ANCHOR
    assert at("the time") is None


def test_a_chance_match_far_ahead_does_not_pull_the_light():
    # The anchor is in the first sentence; "wrong" also ends the text, but a
    # run that far from the anchor is not followed.
    assert at("building systems that other people wrong") == "13:people"


def test_a_skipped_word_is_jumped_over():
    assert at("what I want next is team that ships") == "40:ships"


def test_the_window_can_start_mid_text():
    window = TEXT[20:45]
    k = last_heard(window, "where a bug costs money".split())
    assert window[k] == "money"


def test_no_text_places_nothing():
    assert last_heard([], ["hello", "there"]) is None


# --- the route ------------------------------------------------------------- #


def settings_for(tmp_path, **overrides) -> Settings:
    base = {
        "db_path": tmp_path / "follow.db",
        "provider": "fake",
        "llm_warmup": False,
        "voice_enabled": True,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def pcm(seconds: float = 1.0) -> str:
    return base64.b64encode(b"\x00\x01" * int(16_000 * seconds)).decode()


def heard(text: str) -> list[TimedWord]:
    return [TimedWord(w, i * 0.3, i * 0.3 + 0.25) for i, w in enumerate(text.split())]


@pytest.fixture
def follow(tmp_path):
    """A client whose timing STT hears what the test says it heard."""
    app = create_app(settings_for(tmp_path), provider=FakeProvider())
    stt = FakeSTT(heard=heard("of building systems that other people"))
    with TestClient(app) as handle:
        app.state.voice = Voice(stt=stt, tts=FakeTTS())
        yield handle, stt


def test_the_route_answers_with_the_index_and_no_transcript(follow):
    client, stt = follow
    response = client.post("/api/follow", json={"audio": pcm(), "words": TEXT})
    assert response.status_code == 200
    assert response.json() == {"at": 13, "heard": 6}
    # What Whisper heard stays in the request: no word of it comes back.
    assert "people" not in response.text
    # The audio went to the STT as 16 kHz PCM, the rate the page sends.
    assert stt.timed[-1].sample_rate == 16_000
    assert len(stt.timed[-1].pcm) == 32_000


def test_the_route_asks_the_timing_whisper_not_the_stt(tmp_path):
    app = create_app(settings_for(tmp_path), provider=FakeProvider())
    stt, timer = FakeSTT(), FakeSTT(heard=heard("hi im a software engineer"))
    with TestClient(app) as client:
        app.state.voice = Voice(stt=stt, tts=FakeTTS(), timer=timer)
        body = client.post("/api/follow", json={"audio": pcm(), "words": TEXT}).json()
    assert body["at"] == 4
    assert timer.timed and not stt.timed


def test_the_route_records_nothing(follow):
    client, _ = follow
    before = client.app.state.db.execute("SELECT COUNT(*) FROM turn").fetchone()[0]
    client.post("/api/follow", json={"audio": pcm(), "words": TEXT})
    after = client.app.state.db.execute("SELECT COUNT(*) FROM turn").fetchone()[0]
    assert after == before


def test_audio_that_is_not_base64_is_422(follow):
    client, _ = follow
    response = client.post("/api/follow", json={"audio": "not base64!", "words": TEXT})
    assert response.status_code == 422


def test_an_odd_byte_count_is_not_16_bit_pcm(follow):
    client, _ = follow
    odd = base64.b64encode(b"\x00\x01\x02").decode()
    response = client.post("/api/follow", json={"audio": odd, "words": TEXT})
    assert response.status_code == 422


def test_audio_past_the_ceiling_is_refused_before_it_is_heard(follow):
    client, stt = follow
    response = client.post("/api/follow", json={"audio": pcm(9.0), "words": TEXT})
    assert response.status_code == 422
    assert not stt.timed


def test_an_stt_failure_is_422_not_500(tmp_path):
    app = create_app(settings_for(tmp_path), provider=FakeProvider())
    with TestClient(app) as client:
        app.state.voice = Voice(stt=FakeSTT(error=VoiceBadAudio("no")), tts=FakeTTS())
        response = client.post("/api/follow", json={"audio": pcm(), "words": TEXT})
    assert response.status_code == 422


def test_without_a_voice_the_route_is_503(tmp_path):
    app = create_app(settings_for(tmp_path, voice_enabled=False), provider=FakeProvider())
    with TestClient(app) as client:
        response = client.post("/api/follow", json={"audio": pcm(), "words": TEXT})
    assert response.status_code == 503
