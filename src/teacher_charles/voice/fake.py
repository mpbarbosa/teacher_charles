"""Motores de voz falsos, para a suíte rodar sem nenhum peso baixado.

Mesma razão do ``llm/fake.py``: a estrutura do produto é o que se testa, e um
teste que carrega o Whisper mede o Whisper. Aqui são ~450 MB de pesos e dois
runtimes nativos — dependência que não pode estar no caminho de
``pytest -m "not integration"``.

Os dois registram o que lhes foi pedido (``calls``, ``last_call``), pelo mesmo
motivo do ``FakeProvider``: o que interessa testar na voz é quase sempre **o
que** foi mandado sintetizar, e em quantos pedaços.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import (
    HeardPhonemes,
    PhonemeRecognizer,
    Speech,
    SpeechToText,
    TextToSpeech,
    TimedWord,
    Transcript,
    VoiceError,
)

# Taxa do Kokoro. O falso usa a mesma para o código de borda que monta WAV ver
# o mesmo número que veria em produção.
FAKE_SAMPLE_RATE = 24_000

# Fala sintetizada rende ~14 caracteres por segundo de áudio em ritmo normal.
# O falso respeita isso para que `duration_s` e o fator de tempo real saiam
# com ordem de grandeza plausível em quem consome.
CHARS_PER_SECOND = 14.0


class FakeSTT(SpeechToText):
    """Devolve transcrições roteirizadas e guarda os áudios que recebeu.

    Com vários ``transcripts``, cada chamada consome o próximo; o último se
    repete. Áudio abaixo do mínimo devolve texto vazio, como o motor real.
    """

    def __init__(
        self,
        transcripts: list[str] | str = "I have 30 years.",
        model: str = "fake-stt",
        error: Exception | None = None,
        heard: list[TimedWord] | None = None,
    ) -> None:
        self.model = model
        # What `word_times` reports, for a test of the reader's timings.
        self.heard = list(heard or [])
        self.timed: list[Speech] = []
        self.transcripts = [transcripts] if isinstance(transcripts, str) else list(transcripts)
        if not self.transcripts:
            raise ValueError("FakeSTT precisa de ao menos uma transcrição")
        self.error = error
        self.calls: list[bytes] = []
        # The takes handed back as PCM, for the after-turn route (E14).
        self.decoded: list[bytes] = []
        self.warmed = False
        self.closed = False

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def last_call(self) -> bytes:
        if not self.calls:
            raise AssertionError("o STT nunca foi chamado")
        return self.calls[-1]

    async def transcribe(self, audio: bytes) -> Transcript:
        index = len(self.calls)
        self.calls.append(audio)
        if self.error is not None:
            raise self.error
        text = self.transcripts[min(index, len(self.transcripts) - 1)]
        return Transcript(
            text=text,
            duration_s=max(len(audio) / 16_000 / 2, 0.1),
            elapsed_ms=1.0,
            confidence=0.9,
        )

    async def decode(self, audio: bytes) -> Speech:
        """The bytes taken as 16 kHz PCM already, which is what a test sends."""
        self.decoded.append(audio)
        if self.error is not None:
            raise self.error
        return Speech(pcm=audio[: len(audio) // 2 * 2], sample_rate=16_000, elapsed_ms=1.0)

    async def word_times(self, speech: Speech) -> list[TimedWord]:
        self.timed.append(speech)
        if self.error is not None:
            raise self.error
        return list(self.heard)

    async def warm_up(self) -> None:
        self.warmed = True

    async def aclose(self) -> None:
        self.closed = True


@dataclass(slots=True)
class SpokenCall:
    text: str
    voice: str
    # As asked for: None is "the engine's own", not 1.0.
    speed: float | None = None


class FakeTTS(TextToSpeech):
    """Devolve silêncio com a duração plausível do texto pedido.

    Silêncio, e não um tom: o que se testa na borda é quantos bytes
    atravessaram, com que cabeçalho e em que ordem — nunca o conteúdo da onda.
    """

    def __init__(
        self,
        voice: str = "fake-voice",
        model: str = "fake-tts",
        error: Exception | None = None,
        available: tuple[str, ...] = ("fake-voice", "other-voice"),
    ) -> None:
        self.voice = voice
        self.model = model
        self.error = error
        self._available = available
        self.calls: list[SpokenCall] = []
        self.warmed = False
        self.closed = False

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def last_call(self) -> SpokenCall:
        if not self.calls:
            raise AssertionError("o TTS nunca foi chamado")
        return self.calls[-1]

    @property
    def spoken(self) -> list[str]:
        """Só os textos, na ordem — o que a maioria das asserções quer."""
        return [call.text for call in self.calls]

    async def speak(
        self, text: str, voice: str | None = None, speed: float | None = None
    ) -> Speech:
        trimmed = text.strip()
        if not trimmed:
            raise VoiceError("não há texto para falar")
        self.calls.append(SpokenCall(text=trimmed, voice=voice or self.voice, speed=speed))
        if self.error is not None:
            raise self.error

        # Slower speech lasts longer, as Kokoro's does: a test can tell the
        # speed reached the engine by the length of what came back.
        seconds = len(trimmed) / CHARS_PER_SECOND / (speed or 1.0)
        frames = max(int(seconds * FAKE_SAMPLE_RATE), 1)
        return Speech(
            pcm=b"\x00\x00" * frames,
            sample_rate=FAKE_SAMPLE_RATE,
            elapsed_ms=1.0,
        )

    def voices(self) -> tuple[str, ...]:
        return self._available

    async def phonemize(self, text: str) -> str:
        """The text itself, lower-cased: enough for a test to see a swap applied."""
        if self.error is not None:
            raise self.error
        return text.strip().lower()

    async def speak_phonemes(self, phonemes: str) -> Speech:
        """Recorded as a call with the phonemes as its text, so tests can read the pair."""
        return await self.speak(phonemes)

    async def warm_up(self) -> None:
        self.warmed = True

    async def aclose(self) -> None:
        self.closed = True


# The English part of the real recognizer's inventory (its `vocab.json` has 392
# symbols across the languages it was trained on). Enough to tokenize the
# expected text of every curated phrase the way the real engine would: the
# two-character symbols are what matter, because a split `aɪ` is two errors.
ENGLISH_PHONEMES: frozenset[str] = frozenset(
    "aɪ aɪə aɪɚ aʊ b d dʒ eɪ f h i iə iː j k l m n oʊ oːɹ p s t tʃ v w z "
    "æ ð ŋ ɐ ɑː ɑːɹ ɔ ɔɪ ɔː ɔːɹ ə əl ɚ ɛ ɛɹ ɜː ɡ ɪ ɪɹ ɹ ʃ ʊ ʊɹ ʌ ʒ θ "
    "a o e ɾ uː u x ʔ".split()
)


class FakePhonemes(PhonemeRecognizer):
    """Scripted phonemes, and a record of every take it was handed.

    With several ``heard`` sequences each call takes the next; the last one
    repeats, as in ``FakeSTT``.
    """

    def __init__(
        self,
        heard: list[tuple[str, ...]] | tuple[str, ...] = (),
        vocabulary: frozenset[str] = ENGLISH_PHONEMES,
        error: Exception | None = None,
        model: str = "fake-phonemes",
    ) -> None:
        self.model = model
        self._heard = list(heard) if isinstance(heard, list) else [heard]
        self._vocabulary = vocabulary
        self._error = error
        self.calls: list[Speech] = []

    async def recognize(self, speech: Speech) -> HeardPhonemes:
        self.calls.append(speech)
        if self._error is not None:
            raise self._error
        heard = self._heard.pop(0) if len(self._heard) > 1 else self._heard[0]
        return HeardPhonemes(phonemes=tuple(heard), duration_s=speech.duration_s, elapsed_ms=1.0)

    async def vocabulary(self) -> frozenset[str]:
        return self._vocabulary
