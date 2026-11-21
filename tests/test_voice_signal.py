"""Gate 1 of a pronunciation take (``voice/signal.py``).

The profiles are the measured ones from ``tests/test_measure_phonemes.py``,
whose script this gate was ported from: the two must agree, so the same
signals give the same verdicts in both. numpy comes with the ``voice`` extra,
so these skip where it is absent, as that file's do.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from teacher_charles.voice.signal import diagnose  # noqa: E402

BLOCK = 1600


def blocks(levels: list[float], seed: int = 7):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0.0, lv, size=BLOCK).astype(np.float32) for lv in levels])


def test_a_short_sentence_in_a_long_take_is_usable():
    verdict = diagnose(blocks([0.002] * 30 + [0.05] * 8 + [0.002] * 61), np)
    assert verdict.usable, verdict


def test_a_room_with_no_pause_is_refused():
    verdict = diagnose(blocks([0.009] * 10 + [0.025] * 55 + [0.050] * 5), np)
    assert verdict.refusal and "room" in verdict.refusal


def test_one_click_is_not_speech():
    verdict = diagnose(blocks([0.009] * 10 + [0.025] * 55 + [0.200] + [0.050] * 4), np)
    assert verdict.refusal is not None


def test_silence_is_refused():
    assert diagnose(np.zeros(16_000, dtype=np.float32), np).refusal.startswith("That was silence")


def test_a_quiet_take_is_warned_and_unusable():
    verdict = diagnose(blocks([0.0005] * 30 + [0.006] * 10 + [0.0005] * 30), np)
    assert verdict.refusal is None
    assert not verdict.usable
    assert "too quiet" in verdict.advice


def test_speech_at_the_start_warns_of_a_lost_first_phoneme():
    verdict = diagnose(blocks([0.05] * 10 + [0.002] * 40), np)
    assert any("as the recording started" in w for w in verdict.warnings)


def test_speech_at_the_end_warns_of_a_lost_last_phoneme():
    """The shape that names `[del, h]` for free: the take stops inside the phrase."""
    verdict = diagnose(blocks([0.002] * 40 + [0.05] * 10), np)
    assert any("stopped before you had finished" in w for w in verdict.warnings)


def test_every_message_is_capture_advice_in_english():
    """Metodo §3: about the room and the microphone, never about pronunciation."""
    takes = [
        np.zeros(16_000, dtype=np.float32),
        blocks([0.009] * 10 + [0.025] * 55 + [0.050] * 5),
        blocks([0.0005] * 30 + [0.006] * 10 + [0.0005] * 30),
        blocks([0.05] * 10 + [0.002] * 40),
    ]
    for take in takes:
        advice = diagnose(take, np).advice
        assert advice and advice.isascii(), advice
        assert "pronounc" not in advice.lower()
