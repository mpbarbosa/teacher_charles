"""Gate 1 of a pronunciation take: is this audio an utterance at all?

``docs/metodo-fonetico.md`` §3 names three gates, and this is the one that
lives in ``voice/``: it takes samples and knows nothing about phrases, rules
or the student. The questions and their thresholds are the five of plan §3.5,
measured and revised in ``scripts/measure_phonemes.py`` (``diagnose_signal``
and the ``warn_if_*`` family), which stays the place they were measured; the
numbers here are the same numbers, and a change to one belongs in both.

**What it says is capture advice, never pronunciation.** A student told they
mispronounced a phoneme when the microphone was listening to the room gets the
failure ``POST /api/stt`` was built to avoid. So every message here is about
the room, the microphone or the timing of the take, in English, because the
student reads it (plan decision 1.1).

**A warning silences the verdict as surely as a refusal does** (metodo §7: a
verdict never comes from a take the diagnostic warned about). They are kept
apart because they say different things to the student: a refusal is "that was
not speech", a warning is "that was speech, recorded in a way that can fake an
error". The head-cut and truncated warnings exist for the second reason: a
phoneme the take cut off reads as a deletion, which is exactly what
``pron.h-dropping``'s signature is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SAMPLE_RATE = 16_000
BLOCK_SECONDS = 0.1

RMS_MIN = 0.001
CLIPPED_FRACTION_MAX = 0.005
LOW_FREQ_FRACTION_MAX = 0.50
# The loudest half-second against the floor: see the script's comment on the
# same constants for why it is not the 90th percentile (PR #101).
SPEECH_CONTRAST_MIN = 6.0
SPEECH_CONTRAST_BLOCKS = 5
RMS_LOW = 0.010
PEAK_LOW_DBFS = -20.0
# How close to either edge the speech may come before the take may have lost a
# phoneme there.
EDGE_SECONDS = 0.1
TAIL_SECONDS = 0.15


@dataclass(frozen=True, slots=True)
class SignalVerdict:
    """What gate 1 found. ``usable`` is the only thing the evaluator needs."""

    refusal: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    speech_from_s: float | None = None
    speech_to_s: float | None = None

    @property
    def usable(self) -> bool:
        return self.refusal is None and not self.warnings

    @property
    def advice(self) -> str | None:
        """The one sentence the student reads, or ``None`` for a clean take."""
        if self.refusal is not None:
            return self.refusal
        return self.warnings[0] if self.warnings else None


def samples_from_pcm(pcm: bytes, np: Any) -> Any:
    """16-bit little-endian mono PCM as float32 in [-1, 1]."""
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def block_levels(samples: Any, np: Any, rate: int = SAMPLE_RATE) -> tuple[Any, Any, Any]:
    """RMS and peak per 100 ms block, and which blocks are speech (4× the floor)."""
    block = int(rate * BLOCK_SECONDS)
    if len(samples) < block * 3:
        return None, None, None
    count = len(samples) // block
    shaped = samples[: count * block].reshape(count, block)
    rms = np.sqrt((shaped**2).mean(axis=1))
    peak = np.abs(shaped).max(axis=1)
    floor = float(np.percentile(rms, 10))
    loud = np.flatnonzero(rms >= max(4 * floor, 0.002))
    return rms, peak, loud


def _refusal(samples: Any, np: Any, rate: int) -> str | None:
    if len(samples) == 0:
        return "Nothing was recorded. Check that the microphone is on and try again."

    rms = float(np.sqrt((samples**2).mean()))
    if rms < RMS_MIN:
        return "That was silence. Check that the microphone is on, then read it again."

    head = samples[: int(0.3 * rate)]
    tail = samples[int(0.3 * rate) :]
    if len(tail) and abs(float(head.mean())) > max(0.02, 2 * float(np.sqrt((tail**2).mean()))):
        return "The microphone clicked as it opened. Wait a moment before you start reading."

    if float((np.abs(samples) >= 0.999).mean()) > CLIPPED_FRACTION_MAX:
        return "The microphone is overloaded. Move it a little further away and read it again."

    window = 2048
    if len(samples) >= window:
        freqs = np.fft.rfftfreq(window, 1 / rate)
        spectrum = np.zeros(window // 2 + 1)
        taper = np.hanning(window)
        for start in range(0, len(samples) - window, window):
            spectrum += np.abs(np.fft.rfft(samples[start : start + window] * taper))
        total = spectrum.sum()
        if total and float(spectrum[freqs < 100].sum() / total) > LOW_FREQ_FRACTION_MAX:
            return (
                "The recording is mostly a low hum — breath on the microphone or a "
                "fan. Turn the microphone away from your mouth a little and try again."
            )

    rms_blocks, _, loud = block_levels(samples, np, rate)
    if rms_blocks is not None:
        low = max(float(np.percentile(rms_blocks, 10)), 1e-12)
        top = np.sort(rms_blocks)[::-1][:SPEECH_CONTRAST_BLOCKS]
        if len(loud) == 0 or float(top[-1]) / low < SPEECH_CONTRAST_MIN:
            return (
                "Your voice does not stand out from the room. Read it closer to the "
                "microphone, somewhere quieter, with a moment of silence before and after."
            )
    return None


def _warnings(samples: Any, np: Any, rate: int, span: tuple[float, float]) -> list[str]:
    out: list[str] = []
    rms_blocks, peak_blocks, loud = block_levels(samples, np, rate)
    if rms_blocks is not None and len(loud):
        rms = float(np.median(rms_blocks[loud]))
        peak_dbfs = 20 * np.log10(max(float(np.median(peak_blocks[loud])), 1e-12))
        if rms < RMS_LOW or peak_dbfs < PEAK_LOW_DBFS:
            out.append(
                "You were too quiet to judge. Read it again a little closer to the microphone."
            )
    if span[0] <= EDGE_SECONDS:
        out.append("You started reading as the recording started. Wait a moment, then read it.")
    if span[1] >= len(samples) / rate - TAIL_SECONDS:
        out.append("The recording stopped before you had finished. Let it run a moment longer.")
    return out


def diagnose(samples: Any, np: Any, rate: int = SAMPLE_RATE) -> SignalVerdict:
    """Gate 1: refuse what is not speech, warn about what can fake an error."""
    refusal = _refusal(samples, np, rate)
    if refusal is not None:
        return SignalVerdict(refusal=refusal)
    _, _, loud = block_levels(samples, np, rate)
    span = (float(loud[0]) * BLOCK_SECONDS, float(loud[-1] + 1) * BLOCK_SECONDS)
    return SignalVerdict(
        warnings=tuple(_warnings(samples, np, rate, span)),
        speech_from_s=span[0],
        speech_to_s=span[1],
    )


__all__ = ["SAMPLE_RATE", "SignalVerdict", "block_levels", "diagnose", "samples_from_pcm"]
