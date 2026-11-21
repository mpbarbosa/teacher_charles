"""``POST /api/pronunciation/turn`` (E14), on the fake ears.

The route judges the take ``/api/stt`` transcribed against that transcript,
and only when the turn the server recorded **is** the transcript unedited.
What is tested is that condition and its edges: the take id comes back from
``/api/stt`` only with pronunciation on, an edited turn is 409 and the
recognizer is never asked, a take is judged once, gate 1 runs on the audio
first, and the route **records nothing**.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from fastapi.testclient import TestClient  # noqa: E402

from teacher_charles.app import HEARD_TAKES_KEPT, create_app  # noqa: E402
from teacher_charles.config import Settings  # noqa: E402
from teacher_charles.llm import FakeProvider  # noqa: E402
from teacher_charles.voice import FakePhonemes, FakeSTT, FakeTTS, Voice  # noqa: E402

SAID = "They brought the other brother."
# What b704dc72 heard: /ð/→/d/ in *other*.
OTHER_AS_UDDER = tuple("ð eɪ b ɹ oʊ t ð ɚ ʌ d ɚ b ɹ ʌ ð ɚ".split())


def pcm(levels: list[float]) -> bytes:
    """PCM of white-noise blocks: speech-shaped when the middle stands out.

    ``FakeSTT.decode`` takes the body as PCM already, which is what lets
    gate 1 run on it here as it runs on the decoded webm in production.
    """
    rng = np.random.default_rng(3)
    samples = np.concatenate([rng.normal(0.0, lv, size=1600) for lv in levels])
    return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()


SPEECH = pcm([0.002] * 10 + [0.05] * 15 + [0.002] * 10)
ROOM = pcm([0.009] * 10 + [0.025] * 55 + [0.050] * 5)


def settings_for(tmp_path, **overrides) -> Settings:
    base = {
        "db_path": tmp_path / "turn.db",
        "provider": "fake",
        "llm_warmup": False,
        "voice_enabled": True,
        "pronunciation_enabled": True,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


@pytest.fixture
def client(tmp_path):
    app = create_app(settings_for(tmp_path), provider=FakeProvider(replies="Nice."))
    with TestClient(app) as handle:
        app.state.voice = Voice(stt=FakeSTT(SAID), tts=FakeTTS())
        app.state.phonemes = FakePhonemes(heard=OTHER_AS_UDDER)
        yield handle


def heard(client, audio: bytes = SPEECH) -> dict:
    response = client.post("/api/stt", content=audio, headers={"Content-Type": "audio/webm"})
    assert response.status_code == 200
    return response.json()


def send(client, text: str) -> None:
    response = client.post("/api/chat", json={"message": text})
    assert response.status_code == 200


def judge(client, take_id: str):
    return client.post("/api/pronunciation/turn", json={"take_id": take_id})


def test_the_unedited_transcript_is_judged_against_its_own_audio(client):
    take = heard(client)
    send(client, take["text"])
    response = judge(client, take["take_id"])
    assert response.status_code == 200
    body = response.json()
    assert body["notes"] == [
        {
            "rule_id": "pron.th-voiced",
            "heard": "In *other*, /ð/ came out as /d/.",
            "explanation": client.app.state.pronunciation.item("pron.th-voiced").explanation,
        }
    ]
    assert body["judged"] == [{"rule_id": "pron.th-voiced", "verdict": "correction", "gate": None}]
    assert "per" not in body and "transcript" not in body and "text" not in body
    assert response.headers["cache-control"] == "no-store"


def test_an_edited_turn_is_409_and_the_ear_is_never_asked(client):
    """The edit is the student fixing what Whisper heard; the audio no longer says it."""
    take = heard(client)
    send(client, "They brought the other brothers.")
    response = judge(client, take["take_id"])
    assert response.status_code == 409
    assert client.app.state.phonemes.calls == []


def test_the_check_reads_the_recorded_turn_not_the_request(client):
    """A turn typed after the take, with no edit to the take: still not its turn."""
    take = heard(client)
    send(client, take["text"])
    send(client, "And then we went home.")
    assert judge(client, take["take_id"]).status_code == 409


def test_a_take_is_judged_once(client):
    take = heard(client)
    send(client, take["text"])
    assert judge(client, take["take_id"]).status_code == 200
    assert judge(client, take["take_id"]).status_code == 404


def test_only_the_last_few_takes_are_held(client):
    first = heard(client)
    for _ in range(HEARD_TAKES_KEPT):
        heard(client)
    send(client, first["text"])
    assert judge(client, first["take_id"]).status_code == 404


def test_gate_one_refuses_a_room_before_the_recognizer_is_asked(client):
    take = heard(client, ROOM)
    send(client, take["text"])
    body = judge(client, take["take_id"]).json()
    assert body["notes"] == []
    assert body["judged"] == [{"rule_id": "pron.th-voiced", "verdict": "silence", "gate": 1}]
    assert client.app.state.phonemes.calls == []


def test_it_records_nothing(client):
    take = heard(client)
    send(client, take["text"])
    db = client.app.state.db
    tables = [
        row[0]
        for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        if not row[0].startswith("sqlite_") and row[0] != "schema_version"
    ]
    before = {t: db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    judge(client, take["take_id"])
    after = {t: db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    assert after == before


def test_with_pronunciation_off_stt_holds_nothing_and_the_route_is_503(tmp_path):
    app = create_app(
        settings_for(tmp_path, pronunciation_enabled=False), provider=FakeProvider(replies="Hi.")
    )
    with TestClient(app) as handle:
        app.state.voice = Voice(stt=FakeSTT(SAID), tts=FakeTTS())
        body = heard(handle)
        assert "take_id" not in body
        assert len(handle.app.state.heard_takes) == 0
        assert judge(handle, "anything").status_code == 503


def test_an_empty_transcript_holds_no_take(client):
    client.app.state.voice = Voice(stt=FakeSTT(""), tts=FakeTTS())
    assert "take_id" not in heard(client)
