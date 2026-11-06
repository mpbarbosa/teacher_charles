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

from .base import Speech, SpeechToText, TextToSpeech, Transcript, VoiceError

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
    ) -> None:
        self.model = model
        self.transcripts = [transcripts] if isinstance(transcripts, str) else list(transcripts)
        if not self.transcripts:
            raise ValueError("FakeSTT precisa de ao menos uma transcrição")
        self.error = error
        self.calls: list[bytes] = []
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

    async def warm_up(self) -> None:
        self.warmed = True

    async def aclose(self) -> None:
        self.closed = True


@dataclass(slots=True)
class SpokenCall:
    text: str
    voice: str


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

    async def speak(self, text: str, voice: str | None = None) -> Speech:
        trimmed = text.strip()
        if not trimmed:
            raise VoiceError("não há texto para falar")
        self.calls.append(SpokenCall(text=trimmed, voice=voice or self.voice))
        if self.error is not None:
            raise self.error

        frames = max(int(len(trimmed) / CHARS_PER_SECOND * FAKE_SAMPLE_RATE), 1)
        return Speech(
            pcm=b"\x00\x00" * frames,
            sample_rate=FAKE_SAMPLE_RATE,
            elapsed_ms=1.0,
        )

    def voices(self) -> tuple[str, ...]:
        return self._available

    async def warm_up(self) -> None:
        self.warmed = True

    async def aclose(self) -> None:
        self.closed = True
