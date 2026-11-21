"""E11 with the real engines: Kokoro reads a phrase, wav2vec2 hears it, the gates judge it.

The floor of plan §2.2, as a product test: correct synthetic speech of each
graded phrase must come through all three gates with no error. Under
``-m integration``, and skipped when the Kokoro weights or the 318 MB
recognizer are not on disk — nobody downloads them to get the suite green.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from teacher_charles.config import REPO_ROOT, Settings  # noqa: E402
from teacher_charles.teaching.phonetics import Phonetics  # noqa: E402
from teacher_charles.teaching.pronunciation import Evaluator, load_base  # noqa: E402
from teacher_charles.voice.base import STT_SAMPLE_RATE, Speech  # noqa: E402
from teacher_charles.voice.signal import diagnose, samples_from_pcm  # noqa: E402

pytestmark = pytest.mark.integration

BASE = load_base(REPO_ROOT / "data" / "contrastive" / "pronunciation.yaml")


@pytest.fixture
async def recognizer():
    from huggingface_hub import try_to_load_from_cache

    from teacher_charles.voice.phonemes import HF_MODEL_FILE, HF_REPO, Wav2Vec2Phonemes

    if not isinstance(try_to_load_from_cache(HF_REPO, HF_MODEL_FILE), str):
        pytest.skip("the phoneme recognizer is not in the Hugging Face cache (318 MB)")
    engine = Wav2Vec2Phonemes()
    await engine.warm_up()
    if not engine.ready:
        pytest.skip(f"the recognizer did not load: {engine.last_error}")
    return engine


@pytest.fixture
def tts():
    from teacher_charles.voice.kokoro import KokoroTTS

    config = Settings(_env_file=None)
    if not config.tts_model_path.is_file() or not config.tts_voices_path.is_file():
        pytest.skip("Kokoro weights absent — run `uv run python scripts/fetch_voice_models.py`")
    return KokoroTTS(model_path=config.tts_model_path, voices_path=config.tts_voices_path)


def at_16k(speech: Speech) -> Speech:
    samples = np.frombuffer(speech.pcm, dtype="<i2").astype(np.float32)
    count = int(len(samples) * STT_SAMPLE_RATE / speech.sample_rate)
    resampled = np.interp(
        np.arange(count) * speech.sample_rate / STT_SAMPLE_RATE, np.arange(len(samples)), samples
    )
    # Half a second of quiet either side, as a student's take has, so gate 1's
    # edge warnings do not fire on a synthetic phrase that starts at sample 0.
    quiet = np.zeros(STT_SAMPLE_RATE // 2)
    pcm = np.concatenate([quiet, resampled, quiet]).astype("<i2").tobytes()
    return Speech(pcm=pcm, sample_rate=STT_SAMPLE_RATE, elapsed_ms=0.0)


@pytest.mark.parametrize("item", BASE.graded(), ids=lambda item: item.id)
async def test_correct_speech_is_no_error(item, recognizer, tts):
    speech = at_16k(await tts.speak(item.phrases[0].text))
    signal = diagnose(samples_from_pcm(speech.pcm, np), np)
    heard = await recognizer.recognize(speech)
    judgement = Evaluator(BASE, Phonetics(with_stress=False).transcribe).evaluate(
        item.id,
        heard.phonemes,
        await recognizer.vocabulary(),
        signal_usable=signal.usable,
        signal_advice=signal.advice,
    )
    assert judgement.verdict == "no-error", (judgement, heard.phonemes)
