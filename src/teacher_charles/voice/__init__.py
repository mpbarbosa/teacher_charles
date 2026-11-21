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
    HeardPhonemes,
    PhonemeRecognizer,
    Speech,
    SpeechToText,
    TextToSpeech,
    Transcript,
    VoiceBadAudio,
    VoiceError,
    VoiceModelNotFound,
    VoiceUnavailable,
)
from .fake import ENGLISH_PHONEMES, FakePhonemes, FakeSTT, FakeTTS
from .segment import MIN_SENTENCE_CHARS, SentenceAssembler, for_speech, split_sentences

__all__ = [
    "ENGLISH_PHONEMES",
    "MIN_SENTENCE_CHARS",
    "STT_SAMPLE_RATE",
    "FakePhonemes",
    "FakeSTT",
    "HeardPhonemes",
    "PhonemeRecognizer",
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
    "build_phonemes",
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
    # The engine that times the words of synthesised speech for the lesson
    # reader, when it is not ``stt`` itself. A second, smaller Whisper: the
    # timing runs beside the synthesis on the same CPU, and with base.en the
    # two together needed about as much CPU as the audio lasts, so the reader
    # stalled between sentences. ``None`` means ``stt`` times the words.
    timer: SpeechToText | None = None

    @property
    def word_timer(self) -> SpeechToText:
        return self.timer or self.stt

    async def warm_up(self) -> None:
        """Aquece os dois. Nenhum dos dois propaga erro — ver ``base``."""
        await self.stt.warm_up()
        await self.tts.warm_up()
        if self.timer is not None:
            await self.timer.warm_up()

    async def aclose(self) -> None:
        await self.stt.aclose()
        await self.tts.aclose()
        if self.timer is not None:
            await self.timer.aclose()


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

    timer = None
    if settings.stt_timing_model and settings.stt_timing_model != settings.stt_model:
        timer = WhisperSTT(
            model=settings.stt_timing_model,
            compute_type=settings.stt_compute_type,
            beam_size=1,
            cpu_threads=settings.stt_cpu_threads,
        )

    return Voice(
        timer=timer,
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


def build_phonemes(settings: Settings) -> PhonemeRecognizer | None:
    """The phoneme recognizer of E11, or ``None`` with pronunciation off (the default).

    Separate from ``build_voice`` because it is a separate ear with a
    separate weight: a student can talk to Charles without the 318 MB, and the
    read-aloud drill needs only this and a microphone. It needs the ``voice``
    extra installed all the same, for ``onnxruntime`` and ``huggingface_hub``;
    no wheel is added for it (plan §2.0), so there is no extra of its own.
    """
    if not settings.pronunciation_enabled:
        return None
    if settings.provider == "fake":
        return FakePhonemes()

    from .phonemes import Wav2Vec2Phonemes

    return Wav2Vec2Phonemes(threads=settings.pronunciation_threads)
