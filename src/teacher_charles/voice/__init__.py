"""As duas pontas da Fase 1.5: a orelha e a boca.

A aplicação pede os motores à ``build_voice`` e conversa só pelos contratos de
``base``. Nada fora deste pacote sabe se a transcrição vem do
``faster-whisper`` ou de um dublê, nem se a voz é Kokoro ou silêncio.

É o mesmo arranjo do ``llm/``, e pela mesma razão: ``teaching/`` é agnóstico
quanto à origem do texto desde a E2 (ver o cabeçalho de ``teaching/turn.py``),
então a voz entra como **duas pontas** — STT antes do turno, TTS depois — sem
mexer no meio. Este pacote é essas duas pontas, e nada mais: ele não sabe o
que é um nível, um modo ou uma correção.

``config`` nunca importa ``voice``, igual a ``llm``: a dependência corre num
sentido só.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings
from .base import (
    STT_SAMPLE_RATE,
    Speech,
    SpeechToText,
    TextToSpeech,
    Transcript,
    VoiceBadAudio,
    VoiceError,
    VoiceModelNotFound,
    VoiceUnavailable,
)
from .fake import FakeSTT, FakeTTS
from .segment import MIN_SENTENCE_CHARS, SentenceAssembler, for_speech, split_sentences

__all__ = [
    "MIN_SENTENCE_CHARS",
    "STT_SAMPLE_RATE",
    "FakeSTT",
    "FakeTTS",
    "SentenceAssembler",
    "Speech",
    "SpeechToText",
    "TextToSpeech",
    "Transcript",
    "Voice",
    "VoiceBadAudio",
    "VoiceError",
    "VoiceModelNotFound",
    "VoiceUnavailable",
    "build_voice",
    "for_speech",
    "split_sentences",
]


@dataclass(frozen=True, slots=True)
class Voice:
    """Os dois motores, juntos.

    Andam em par porque a voz só é voz com os dois: ouvir sem falar é um campo
    de texto com microfone, e falar sem ouvir é um leitor de tela. Quem quiser
    só um lado usa o atributo.
    """

    stt: SpeechToText
    tts: TextToSpeech

    async def warm_up(self) -> None:
        """Aquece os dois. Nenhum dos dois propaga erro — ver ``base``."""
        await self.stt.warm_up()
        await self.tts.warm_up()

    async def aclose(self) -> None:
        await self.stt.aclose()
        await self.tts.aclose()


def build_voice(settings: Settings) -> Voice | None:
    """Os motores que a configuração pede, ou ``None`` se a voz está desligada.

    ``None`` é estado de primeira classe, não falha: o chat por texto da Fase 1
    funciona inteiro sem voz, e é assim que ele segue funcionando numa máquina
    que não baixou os pesos. As rotas de voz respondem 503 nesse caso, e o
    ``/health`` diz por quê.

    Importa os motores reais **dentro** da função, de propósito: o extra
    ``voice`` não é obrigatório, e um ``import`` no topo faria
    ``import teacher_charles.app`` falhar numa instalação sem ele.
    """
    if not settings.voice_enabled:
        return None

    if settings.provider == "fake":
        # Mesma licença do `TC_PROVIDER=fake`: deixa trabalhar na interface de
        # voz sem carregar 450 MB de pesos.
        return Voice(stt=FakeSTT(), tts=FakeTTS())

    from .kokoro import KokoroTTS
    from .whisper import WhisperSTT

    return Voice(
        stt=WhisperSTT(
            model=settings.stt_model,
            compute_type=settings.stt_compute_type,
            beam_size=settings.stt_beam_size,
            no_speech_threshold=settings.stt_no_speech_threshold,
            cpu_threads=settings.stt_cpu_threads,
        ),
        tts=KokoroTTS(
            model_path=settings.tts_model_path,
            voices_path=settings.tts_voices_path,
            voice=settings.tts_voice,
            speed=settings.tts_speed,
            cpu_threads=settings.tts_cpu_threads,
        ),
    )
