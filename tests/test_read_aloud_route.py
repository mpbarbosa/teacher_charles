"""The read-aloud drill's routes (E12), on the fake recognizer and the fake voice.

What is held here is what the session is worth and what it records: a judged
take becomes one ``pronunciation_signal``, a silenced one becomes nothing, a
finished session gives the schedule **one** review with no turn behind it, and
the deliberate take is once per rule and before the drill. The heard sides
are real E9 takes (``tests/test_pronunciation_eval.py`` says which).
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

# b704dc72: /ð/→/d/ in *other*. 3ba35169: all four /ð/ right.
# eaa0d509: /ð/ said as /d/ on purpose. And *the red car is ready*, not the phrase.
UDDER = tuple("ð eɪ b ɹ oʊ t ð ɚ ʌ d ɚ b ɹ ʌ ð ɚ".split())
CLEAN = tuple("ð eɪ b ɹ ɑː t ð ə ʌ ð ɚ b ɹ ʌ ð ɚ".split())
ON_PURPOSE = tuple("ð eɪ b ɹ ɑː t d uː d ɚ b ɹ uː d ɚ".split())
ANOTHER = tuple("ð ə ɹ ɛ d k ɑːɹ ɪ z ɹ ɛ d i".split())

TH = "pron.th-voiced"


def audio(levels: list[float]) -> str:
    rng = np.random.default_rng(3)
    samples = np.concatenate([rng.normal(0.0, lv, size=1600) for lv in levels])
    return base64.b64encode((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()).decode()


SPEECH = audio([0.002] * 10 + [0.05] * 15 + [0.002] * 10)
ROOM = audio([0.009] * 10 + [0.025] * 55 + [0.050] * 5)


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        _env_file=None,
        db_path=tmp_path / "ra.db",
        provider="fake",
        llm_warmup=False,
        pronunciation_enabled=True,
        voice_enabled=True,
    )
    app = create_app(settings, provider=FakeProvider(replies="Hello."))
    with TestClient(app) as handle:
        yield handle


def hears(client, *takes) -> None:
    client.app.state.phonemes = FakePhonemes(heard=list(takes))


def start(client, **body) -> dict:
    response = client.post("/api/read-aloud/start", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def take(client, session, deliberate=False, sound=SPEECH):
    return client.post(
        "/api/read-aloud/take",
        json={"session_id": session["session_id"], "deliberate": deliberate, "audio": sound},
    )


def finish(client, session):
    return client.post("/api/read-aloud/finish", json={"session_id": session["session_id"]})


def count(client, table: str) -> int:
    return client.app.state.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def reviews(client) -> list:
    return client.app.state.db.execute("SELECT * FROM rule_review ORDER BY id").fetchall()


# --- opening a session ---------------------------------------------------- #


def test_off_by_default_the_page_says_how_and_the_routes_are_503(tmp_path):
    app = create_app(
        Settings(_env_file=None, db_path=tmp_path / "off.db", provider="fake", llm_warmup=False),
        provider=FakeProvider(replies="Hi."),
    )
    with TestClient(app) as handle:
        assert "TC_PRONUNCIATION_ENABLED" in handle.get("/read-aloud").text
        assert handle.post("/api/read-aloud/start", json={}).status_code == 503


def test_the_student_picks_a_rule_and_owes_the_deliberate_take(client):
    session = start(client, rule_id=TH)
    assert (session["rule_id"], session["picked_by"]) == (TH, "student")
    assert session["text"] == "They brought the other brother"
    assert session["deliberate"] is True and session["takes"] == 3


def test_the_schedule_picks_when_the_student_does_not(client):
    session = start(client)
    assert session["picked_by"] == "schedule"
    assert session["rule_id"].startswith("pron.")


def test_an_unknown_rule_is_404(client):
    assert client.post("/api/read-aloud/start", json={"rule_id": "pron.nope"}).status_code == 404


# --- takes ------------------------------------------------------------------ #


def test_a_judged_take_is_one_signal_with_its_numbers(client):
    hears(client, UDDER)
    session = start(client, rule_id=TH)
    body = take(client, session).json()
    assert (body["recorded"], body["judged"], body["needed"]) == (True, 1, 3)
    row = client.app.state.db.execute("SELECT * FROM pronunciation_signal").fetchone()
    assert (row["pass"], row["verdict"], row["opportunities"], row["errors"]) == (
        "unmonitored",
        "correction",
        4,
        1,
    )
    assert row["model"] == "fake-phonemes" and row["duration_s"] > 0
    assert "heard" not in body or body.get("heard") is None


def test_a_silenced_take_records_nothing_and_is_read_again(client):
    """Metodo §3: gates 1 and 2 record nothing, and a silenced take is not one of three."""
    hears(client, ANOTHER)
    session = start(client, rule_id=TH)
    room = take(client, session, sound=ROOM).json()
    assert (room["verdict"], room["gate"], room["recorded"], room["judged"]) == (
        "silence",
        1,
        False,
        0,
    )
    other_phrase = take(client, session).json()
    assert (other_phrase["gate"], other_phrase["recorded"]) == (2, False)
    assert count(client, "pronunciation_signal") == 0


def test_three_judged_takes_and_no_more(client):
    hears(client, CLEAN)
    session = start(client, rule_id=TH)
    for _ in range(3):
        assert take(client, session).json()["recorded"]
    assert take(client, session).status_code == 409


def test_the_deliberate_take_comes_first_and_once(client):
    hears(client, ON_PURPOSE, CLEAN)
    session = start(client, rule_id=TH)
    body = take(client, session, deliberate=True).json()
    assert body["recorded"] and "now your clean reads can count" in body["note"]
    assert take(client, session, deliberate=True).status_code == 409
    assert start(client, rule_id=TH)["deliberate"] is False


def test_the_deliberate_take_cannot_follow_the_drill(client):
    """The session ends on the correct production (metodo §2.4)."""
    hears(client, CLEAN)
    session = start(client, rule_id=TH)
    take(client, session)
    assert take(client, session, deliberate=True).status_code == 409


def test_another_students_session_is_not_found(client):
    hears(client, CLEAN)
    session = start(client, rule_id=TH)
    client.post("/api/profile", json={"name": "bruno"})
    assert take(client, session).status_code == 404


# --- finishing ------------------------------------------------------------- #


def test_two_named_takes_are_one_miss_with_no_turn(client):
    hears(client, UDDER, CLEAN, UDDER)
    session = start(client, rule_id=TH)
    for _ in range(3):
        take(client, session)
    body = finish(client, session).json()
    assert body["card"]["outcome"] == "miss"
    assert body["card"]["lines"][0] == "In *other*, /ð/ came out as /d/ in 2 of 3 takes."
    (review,) = reviews(client)
    assert (review["source"], review["hit"], review["turn_id"]) == ("pronunciation", 0, None)
    assert review["pronunciation_session_id"] == session["session_id"]
    assert count(client, "turn") == 0


def test_a_second_finish_reviews_nothing(client):
    hears(client, UDDER)
    session = start(client, rule_id=TH)
    take(client, session)
    take(client, session)
    assert finish(client, session).status_code == 200
    assert finish(client, session).status_code == 409
    assert len(reviews(client)) == 1


def test_clean_takes_with_no_naming_ever_graduate_nothing(client):
    """Metodo §2.4's guard: the silence may be the evaluator, not the student."""
    hears(client, CLEAN)
    session = start(client, rule_id=TH)
    for _ in range(3):
        take(client, session)
    body = finish(client, session).json()
    assert (body["card"]["outcome"], body["review"]) == ("none", None)
    assert reviews(client) == []


def test_clean_takes_after_a_named_deliberate_take_are_a_hit(client):
    hears(client, ON_PURPOSE, CLEAN)
    session = start(client, rule_id=TH)
    take(client, session, deliberate=True)
    for _ in range(3):
        take(client, session)
    body = finish(client, session).json()
    assert body["card"]["outcome"] == "hit"
    assert body["review"]["hit"] is True
    (review,) = reviews(client)
    assert review["source"] == "pronunciation"


def test_once_in_three_reviews_nothing(client):
    hears(client, UDDER, CLEAN)
    session = start(client, rule_id=TH)
    for _ in range(3):
        take(client, session)
    body = finish(client, session).json()
    assert body["card"]["outcome"] == "once" and body["review"] is None
    assert reviews(client) == []


def test_a_rule_just_missed_is_not_picked_again_today(client):
    """The schedule's half-day rung, through read-aloud: the next pick is another rule."""
    hears(client, UDDER)
    session = start(client, rule_id=TH)
    take(client, session)
    take(client, session)
    finish(client, session)
    assert start(client)["rule_id"] != TH


# --- what the rest of the app sees --------------------------------------- #


def test_progress_shows_the_rule_in_takes(client):
    hears(client, UDDER, CLEAN, UDDER)
    session = start(client, rule_id=TH)
    for _ in range(3):
        take(client, session)
    finish(client, session)
    (rule,) = client.get("/api/progress").json()["read_aloud"]
    assert (rule["rule_id"], rule["sessions"], rule["takes"], rule["named"]) == (TH, 1, 3, 2)
    assert rule["last_outcome"] == "miss"
    page = client.get("/progress").text
    assert "Read aloud" in page and "2/3" in page


def test_the_text_drill_never_sees_a_pronunciation_rule(client):
    """The E10 invariant one layer up: a sound reviewed here is not a grammar drill."""
    hears(client, UDDER)
    session = start(client, rule_id=TH)
    take(client, session)
    take(client, session)
    finish(client, session)
    progress = client.get("/api/progress").json()
    assert all(not r["rule_id"].startswith("pron.") for r in progress["rules"])
    assert client.get("/api/mode").json()["drillable"] == 0


def test_the_pair_swaps_the_target_and_nothing_else(client):
    tts = client.app.state.voice.tts

    async def phonemize(text):
        return "ðeɪ bɹˈɔːt ðɪ ˈʌðɚ bɹˈʌðɚ"

    tts.phonemize = phonemize
    right = client.get(f"/api/read-aloud/pair/{TH}/right")
    wrong = client.get(f"/api/read-aloud/pair/{TH}/wrong")
    assert right.status_code == wrong.status_code == 200
    assert right.headers["content-type"] == "audio/wav"
    assert tts.spoken[-2:] == ["ðeɪ bɹˈɔːt ðɪ ˈʌðɚ bɹˈʌðɚ", "deɪ bɹˈɔːt dɪ ˈʌdɚ bɹˈʌdɚ"]


def test_the_page_lists_the_graded_rules(client):
    page = client.get("/read-aloud").text
    for rule_id in ("pron.th-voiced", "pron.h-dropping", "pron.ing-alveolar"):
        assert rule_id in page
