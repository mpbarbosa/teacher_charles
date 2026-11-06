"""O contrato que todo motor de voz cumpre.

Espelha de propósito o ``llm/base.py``: dois protocolos abstratos, erros
tipados, e nada fora deste pacote sabendo se a transcrição vem do
``faster-whisper``, de uma API de nuvem ou de um dublê de teste. A razão é a
mesma que lá — ``teaching/`` tem que poder ser testado sem nenhum peso
baixado.

## Por que áudio é PCM aqui dentro, e WAV só na borda

Tudo o que atravessa este pacote é **PCM 16-bit mono little-endian** mais a
taxa de amostragem. O WAV é montado uma vez, no ``as_wav``, quando o áudio vai
sair pela rota. Guardar o cabeçalho junto obrigaria todo mundo que concatena
dois pedaços de fala a saber desmontá-lo — e concatenar sentença a sentença é
exatamente o que o TTS incremental faz.

16-bit em vez do float32 que o Kokoro devolve: é metade dos bytes na rede, e o
``<audio>`` do navegador não toca WAV float sem ajuda.
"""

from __future__ import annotations

import abc
import io
import wave
from dataclasses import dataclass

# O Whisper reamostra tudo para 16 kHz internamente; mandar mais que isso é
# pagar banda por informação que o modelo descarta.
STT_SAMPLE_RATE = 16_000


class VoiceError(Exception):
    """Base de tudo o que pode dar errado num motor de voz.

    Reaja pelo tipo, nunca por substring da mensagem — mesma regra dos erros
    de provider do LLM.
    """


class VoiceUnavailable(VoiceError):
    """O motor não está utilizável: extra não instalado, peso ausente."""


class VoiceModelNotFound(VoiceError):
    """O modelo ou a voz nomeada na configuração não existe."""


class VoiceBadAudio(VoiceError):
    """O áudio que chegou não decodifica, ou não tem fala nenhuma."""


@dataclass(frozen=True, slots=True)
class Transcript:
    """O que o aluno disse, e quanto custou descobrir isso."""

    text: str
    duration_s: float
    elapsed_ms: float
    language: str = "en"
    # Probabilidade média que o modelo atribuiu ao que transcreveu, quando o
    # motor reporta. ``None`` é resposta honesta de quem não mede isso.
    confidence: float | None = None

    @property
    def real_time_factor(self) -> float | None:
        """Tempo de máquina por segundo de áudio. Abaixo de 1 é mais rápido que a fala."""
        if self.duration_s <= 0:
            return None
        return (self.elapsed_ms / 1000) / self.duration_s

    def as_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "duration_s": round(self.duration_s, 3),
            "elapsed_ms": round(self.elapsed_ms),
            "language": self.language,
            "confidence": None if self.confidence is None else round(self.confidence, 3),
        }


@dataclass(frozen=True, slots=True)
class Speech:
    """Um trecho de fala sintetizada, em PCM 16-bit mono."""

    pcm: bytes
    sample_rate: int
    elapsed_ms: float

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / 2 / self.sample_rate

    @property
    def real_time_factor(self) -> float | None:
        """Quanto de máquina por segundo de fala gerada.

        É **o** número da Fase 1.5: acima de 1 o aluno ouve a voz engasgar,
        porque a síntese não acompanha a reprodução.
        """
        if self.duration_s <= 0:
            return None
        return (self.elapsed_ms / 1000) / self.duration_s

    def as_wav(self) -> bytes:
        """O mesmo áudio com cabeçalho RIFF, para o ``<audio>`` do navegador."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(self.sample_rate)
            out.writeframes(self.pcm)
        return buffer.getvalue()


class Loadable:
    """O estado de carga que o ``/health`` lê sem provocar a carga.

    Os dois motores carregam preguiçosamente, e o aquecimento engole o erro de
    propósito (derrubar a subida por causa da voz seria pior). Sem isto o erro
    sumiria: o dev veria a rota responder 422 e não saberia que o peso não
    está no disco. ``ready`` e ``last_error`` são o que o ``/health`` mostra —
    e ``ready`` nunca dispara carga, porque um health check não pode ter efeito
    colateral de centenas de megabytes.
    """

    _last_error: str | None = None

    @property
    def ready(self) -> bool:
        """O motor pode atender agora? Default ``True``: quem não carrega nada."""
        return True

    @property
    def last_error(self) -> str | None:
        """O que o último aquecimento reclamou, se reclamou."""
        return self._last_error


class SpeechToText(Loadable, abc.ABC):
    """A orelha. Recebe os bytes que o navegador gravou, devolve texto."""

    model: str

    @abc.abstractmethod
    async def transcribe(self, audio: bytes) -> Transcript:
        """Transcreve o áudio inteiro.

        ``audio`` são os bytes do arquivo como o navegador os gravou
        (webm/opus, ogg, wav): quem decodifica é o motor, não quem chama.
        """

    async def warm_up(self) -> None:
        """Carrega o modelo antes de o aluno precisar dele.

        Nunca propaga erro, pela mesma razão que o aquecimento do LLM não
        propaga: o ``/health`` já reporta motor inacessível, e derrubar a
        subida por causa do aquecimento seria pior.

        Default no-op, e não abstrato: um motor sem carga preguiçosa não tem o
        que aquecer, e obrigá-lo a escrever um método vazio não compra nada.
        """
        return None

    async def aclose(self) -> None:
        """Libera o que o motor segurar. Idempotente. Default: nada a soltar."""
        return None


class TextToSpeech(Loadable, abc.ABC):
    """A boca. Recebe uma sentença, devolve PCM."""

    voice: str
    model: str

    @abc.abstractmethod
    async def speak(self, text: str, voice: str | None = None) -> Speech:
        """Sintetiza **um** trecho curto.

        Uma sentença por chamada é escolha de desenho, não limitação: é o que
        permite o professor começar a falar antes de a resposta terminar. Ver
        ``voice/segment.py``.
        """

    @abc.abstractmethod
    def voices(self) -> tuple[str, ...]:
        """As vozes que este motor tem. Tupla vazia se ele não sabe dizer."""

    async def warm_up(self) -> None:
        """Mesma regra do STT: carrega cedo, nunca propaga erro."""
        return None

    async def aclose(self) -> None:
        """Libera o que o motor segurar. Idempotente. Default: nada a soltar."""
        return None
