"""``POST /api/pronunciation`` (E11), on the fake recognizer.

The route is the three gates in order and nothing else, so what is tested is
the order and the edges: off is 503, an unknown rule 404, gate 1 is decided on
the audio before the recognizer is asked, and a take that passes all three
comes back as a verdict with a ``rule_id`` and no transcription. And it
**records nothing**: E12 persists, E11 does not.
"""

from __future__ import annotations

import base64

import pytest

np = pytest.importorskip("numpy")

from fastapi.testclient import TestClient  # noqa: E402

from teacher_charles.app import create_app  # noqa: E402
from teacher_charles.config import Settings  # noqa: E402
from teacher_charles.llm import FakeProvider  # noqa: E402
from teacher_charles.voice import FakePhonemes  # noqa: E402

# What b704dc72 heard: /ð/→/d/ in *other*.
OTHER_AS_UDDER = tuple("ð eɪ b ɹ oʊ t ð ɚ ʌ d ɚ b ɹ ʌ ð ɚ".split())


def settings_for(tmp_path, **overrides) -> Settings:
    base = {
        "db_path": tmp_path / "pron.db",
        "provider": "fake",
        "llm_warmup": False,
        "pronunciation_enabled": True,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def audio(levels: list[float]) -> str:
    """Base64 PCM of white-noise blocks: speech-shaped when the middle stands out."""
    rng = np.random.default_rng(3)
    samples = np.concatenate([rng.normal(0.0, lv, size=1600) for lv in levels])
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
    return base64.b64encode(pcm).decode()


SPEECH = audio([0.002] * 10 + [0.05] * 15 + [0.002] * 10)
ROOM = audio([0.009] * 10 + [0.025] * 55 + [0.050] * 5)


@pytest.fixture
def client(tmp_path):
    app = create_app(settings_for(tmp_path), provider=FakeProvider(replies="Hello."))
    with TestClient(app) as handle:
        app.state.phonemes = FakePhonemes(heard=OTHER_AS_UDDER)
        yield handle


def post(client, **body):
    payload = {"rule_id": "pron.th-voiced", "audio": SPEECH} | body
    return client.post("/api/pronunciation", json=payload)


def test_off_by_default_is_503_with_the_way_on(tmp_path):
    app = create_app(
        settings_for(tmp_path, pronunciation_enabled=False), provider=FakeProvider(replies="Hi.")
    )
    with TestClient(app) as handle:
        response = post(handle)
    assert response.status_code == 503
    assert "TC_PRONUNCIATION_ENABLED" in response.json()["how"]


def test_an_unknown_rule_is_404(client):
    assert post(client, rule_id="pron.nonexistent").status_code == 404


def test_audio_that_is_not_base64_is_422(client):
    assert post(client, audio="not base64!!").status_code == 422


def test_a_take_through_all_three_gates_is_a_correction(client):
    response = post(client)
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "correction"
    assert body["rule_id"] == "pron.th-voiced"
    assert body["findings"] == [{"word": "other", "expected": "ð", "heard": "d"}]
    assert "per" not in body and "heard" not in body
    assert response.headers["cache-control"] == "no-store"


def test_gate_one_refuses_before_the_recognizer_is_asked(client):
    """A room is never handed to the recognizer, so it cannot become a verdict."""
    body = post(client, audio=ROOM).json()
    assert (body["verdict"], body["gate"]) == ("silence", 1)
    assert "room" in body["advice"]
    assert client.app.state.phonemes.calls == []


def test_it_records_nothing(client):
    db = client.app.state.db
    tables = [
        row[0]
        for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        if not row[0].startswith("sqlite_") and row[0] != "schema_version"
    ]
    before = {t: db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    post(client)
    post(client, audio=ROOM)
    after = {t: db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    assert after == before


def test_health_reports_the_recognizer(client):
    body = client.get("/health").json()
    assert body["pronunciation"]["enabled"] is True
    assert body["pronunciation"]["model"] == "fake-phonemes"
