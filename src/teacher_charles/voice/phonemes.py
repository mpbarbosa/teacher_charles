"""The phoneme recognizer of E11: wav2vec2 trained on espeak IPA, in ONNX on the CPU.

It is the model ``scripts/measure_phonemes.py`` measured in E8 and E9, loaded
the same way, so the numbers in plan §2 and §3.5 describe this engine and not
a cousin of it. ``docs/plano-fase-2.md`` §2.0 gives the three reasons for the
choice, and all three are about budget: no new wheel (``onnxruntime`` and
``huggingface_hub`` already come with the ``voice`` extra), int8 like the other
two engines, and an output alphabet that is the espeak IPA the expected text
is written in, so no map between two inventories sits between the student and
the verdict.

**Greedy CTC, no beam and no phoneme language model.** A decoder that knows
which sequences are plausible English repairs the pronunciation before the
evaluator sees it, which is the defect the initial analysis §3 points at in
Whisper (*tink* heard as *think*).

**One at a time, in a thread**, for ``whisper.py``'s reason: the session is
native and synchronous, it would freeze the event loop, and two on this CPU at
once only slow both.

**318 MB from the Hub, into the Hub's cache.** CLAUDE.md, "Where the data
lives": the recognizer follows Whisper, and never ``XDG_DATA_HOME``. The weight
downloads on first load, which is the warm-up; ``ready`` never triggers it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from .base import (
    STT_SAMPLE_RATE,
    HeardPhonemes,
    PhonemeRecognizer,
    Speech,
    VoiceBadAudio,
    VoiceModelNotFound,
    VoiceUnavailable,
)

log = logging.getLogger(__name__)

HF_REPO = "onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX"
HF_MODEL_FILE = "onnx/model_int8.onnx"

# A read-aloud phrase is a breath long; the ceiling keeps a stuck microphone
# from holding the CPU. Thirty seconds at 16 kHz.
MAX_SECONDS = 30.0


def _special(token: str) -> bool:
    return token.startswith("<") and token.endswith(">")


class Wav2Vec2Phonemes(PhonemeRecognizer):
    def __init__(self, threads: int = 0) -> None:
        self.model = f"{HF_REPO}#{HF_MODEL_FILE}"
        self._threads = threads
        self._session: Any | None = None
        self._id_to_token: dict[int, str] = {}
        self._vocab: frozenset[str] | None = None
        self._pad_id = 0
        self._input_name = ""
        self._lock = asyncio.Lock()

    # --- load ------------------------------------------------------------- #

    def _vocab_raw(self) -> dict[str, int]:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise VoiceUnavailable(
                "huggingface_hub is not installed. Run `uv sync --extra voice`."
            ) from exc
        try:
            path = hf_hub_download(HF_REPO, "vocab.json")
        except Exception as exc:  # noqa: BLE001 - network and missing repo look alike
            raise VoiceModelNotFound(f"the phoneme vocabulary did not download: {exc}") from exc
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def _load(self) -> None:
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise VoiceUnavailable(
                "onnxruntime is not installed. Run `uv sync --extra voice`."
            ) from exc
        raw = self._vocab_raw()
        try:
            model_path = hf_hub_download(HF_REPO, HF_MODEL_FILE)
        except Exception as exc:  # noqa: BLE001
            raise VoiceModelNotFound(
                f"the phoneme recognizer ({self.model}) did not download: {exc}. "
                "It is 318 MB, fetched once into the Hugging Face cache."
            ) from exc
        options = ort.SessionOptions()
        if self._threads:
            options.intra_op_num_threads = self._threads
        session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._id_to_token = {index: token for token, index in raw.items()}
        self._vocab = frozenset(token for token in raw if not _special(token))
        self._pad_id = raw.get("<pad>", 0)
        self._input_name = session.get_inputs()[0].name
        self._session = session

    async def _ready(self) -> None:
        if self._session is None:
            await asyncio.to_thread(self._load)

    async def warm_up(self) -> None:
        try:
            async with self._lock:
                await self._ready()
            self._last_error = None
            log.info("phoneme recognizer %s loaded", self.model)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a warm-up never brings startup down
            self._last_error = str(exc)
            log.warning("phoneme recognizer warm-up failed (carrying on): %s", exc)

    @property
    def ready(self) -> bool:
        return self._session is not None

    async def vocabulary(self) -> frozenset[str]:
        async with self._lock:
            if self._vocab is None:
                raw = await asyncio.to_thread(self._vocab_raw)
                self._vocab = frozenset(token for token in raw if not _special(token))
            return self._vocab

    # --- recognition ------------------------------------------------------ #

    def _recognize(self, speech: Speech) -> HeardPhonemes:
        import numpy as np

        audio = np.frombuffer(speech.pcm, dtype="<i2").astype(np.float32) / 32768.0
        # Zero mean and unit variance: the model's `do_normalize`. Without it
        # the output is not a bit worse, it is garbage.
        audio = (audio - audio.mean()) / (audio.std() + 1e-7)
        started = time.perf_counter()
        try:
            logits = self._session.run(None, {self._input_name: audio[None, :]})[0]
        except Exception as exc:  # noqa: BLE001 - onnxruntime raises its own types
            raise VoiceBadAudio(f"the phoneme recognizer could not read this audio: {exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000
        out: list[str] = []
        previous = -1
        for token_id in logits[0].argmax(axis=-1).tolist():
            if token_id != previous and token_id != self._pad_id:
                token = self._id_to_token.get(token_id, "")
                if token and not _special(token):
                    out.append(token)
            previous = token_id
        return HeardPhonemes(
            phonemes=tuple(out), duration_s=speech.duration_s, elapsed_ms=elapsed_ms
        )

    async def recognize(self, speech: Speech) -> HeardPhonemes:
        if speech.sample_rate != STT_SAMPLE_RATE:
            raise VoiceBadAudio(
                f"the recognizer takes {STT_SAMPLE_RATE} Hz, not {speech.sample_rate}"
            )
        if speech.duration_s > MAX_SECONDS:
            raise VoiceBadAudio(f"a take of {speech.duration_s:.0f} s is longer than a phrase")
        if not speech.pcm:
            return HeardPhonemes(phonemes=(), duration_s=0.0, elapsed_ms=0.0)
        async with self._lock:
            await self._ready()
            return await asyncio.to_thread(self._recognize, speech)

    async def aclose(self) -> None:
        self._session = None


__all__ = ["HF_MODEL_FILE", "HF_REPO", "Wav2Vec2Phonemes"]
