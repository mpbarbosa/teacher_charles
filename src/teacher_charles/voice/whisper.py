"""STT com ``faster-whisper`` na CPU.

``small.en`` em int8, como o plano §4 escolheu. O sufixo ``.en`` é consequência
direta da decisão 1.1: produto monolíngue permite o modelo só de inglês, que é
menor e mais exato na única língua que interessa aqui.

## O VAD do Silero entra por aqui

O ``faster-whisper`` embute o Silero VAD como filtro de entrada
(``vad_filter=True``), e isso resolve dois problemas de uma vez num botão de
push-to-talk: corta o silêncio que o aluno gravou antes e depois de falar — o
que é tempo de transcrição pago por nada — e tira o combustível da alucinação
em silêncio, que no Whisper costuma sair como ``"Thank you."`` ou
``"Thanks for watching!"``.

Não é o VAD de turn-taking que o plano §4 descreve: aqui quem decide que o
aluno terminou é o dedo dele no botão. Usar o mesmo VAD para ouvir o aluno sem
botão é a melhoria natural, e esta é a dependência que ela já encontra pronta.

## Por que tudo corre numa thread, e um por vez

O ``ctranslate2`` é código nativo síncrono: chamá-lo direto travaria o loop do
asyncio e pararia o streaming de **todos** os turnos abertos. Daí o
``to_thread``. E daí também o lock: a transcrição é limitada por CPU, e duas
ao mesmo tempo nesta máquina de 14 threads não dobram a vazão, só pioram as
duas — a mesma razão que fez a análise do LLM esperar a resposta terminar
(plano §5.2).
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
import time
from typing import Any

from .base import SpeechToText, Transcript, VoiceBadAudio, VoiceModelNotFound, VoiceUnavailable

log = logging.getLogger(__name__)

# Áudio mais curto que isto não é fala: é um clique no botão. Transcrever
# devolveria alucinação, e devolver string vazia é a resposta correta.
MIN_AUDIO_BYTES = 1024

# O aluno fala uma ou duas frases por turno. Um minuto é folga larga, e o teto
# existe para um upload grande não prender a CPU por minutos.
MAX_AUDIO_BYTES = 8 * 1024 * 1024


class WhisperSTT(SpeechToText):
    """A orelha do Charles."""

    def __init__(
        self,
        model: str = "small.en",
        compute_type: str = "int8",
        beam_size: int = 1,
        no_speech_threshold: float = 0.6,
        cpu_threads: int = 0,
        download_root: str | None = None,
    ) -> None:
        self.model = model
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._no_speech_threshold = no_speech_threshold
        self._cpu_threads = cpu_threads
        self._download_root = download_root
        self._engine: Any | None = None
        # Serializa carga e inferência: ver o cabeçalho do módulo.
        self._lock = asyncio.Lock()

    # --- carga ------------------------------------------------------------ #

    def _load(self) -> Any:
        """Carrega o modelo. Síncrono de propósito: sempre chamado em thread."""
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - depende do extra
            raise VoiceUnavailable(
                "faster-whisper não está instalado. Rode `uv sync --extra voice`."
            ) from exc

        try:
            return WhisperModel(
                self.model,
                device="cpu",
                compute_type=self._compute_type,
                cpu_threads=self._cpu_threads,
                download_root=self._download_root,
            )
        except (OSError, ValueError) as exc:
            # Nome de modelo errado e rede fora chegam aqui iguais; a mensagem
            # tem que dizer os dois, porque o dev não sabe qual é.
            raise VoiceModelNotFound(
                f"modelo de STT {self.model!r} não carregou: {exc}. "
                f"Ele é baixado na primeira vez — confira o nome e a rede."
            ) from exc

    async def _engine_ready(self) -> Any:
        if self._engine is None:
            self._engine = await asyncio.to_thread(self._load)
        return self._engine

    async def warm_up(self) -> None:
        try:
            async with self._lock:
                await self._engine_ready()
            self._last_error = None
            log.info("STT %s (%s) carregado", self.model, self._compute_type)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - aquecimento nunca derruba a subida
            # Guardado para o /health poder dizer POR QUE a voz não atende.
            self._last_error = str(exc)
            log.warning("aquecimento do STT falhou (segue assim mesmo): %s", exc)

    # --- transcrição ------------------------------------------------------ #

    def _transcribe(self, engine: Any, audio: bytes) -> Transcript:
        started = time.monotonic()
        try:
            segments, info = engine.transcribe(
                io.BytesIO(audio),
                language="en",
                beam_size=self._beam_size,
                no_speech_threshold=self._no_speech_threshold,
                # Sem isto o modelo usa a transcrição anterior como contexto e
                # pode entrar em laço repetindo a última frase. Cada turno do
                # aluno é independente; não há contexto para condicionar.
                condition_on_previous_text=False,
                # Silero VAD: corta silêncio e tira o combustível da
                # alucinação. Ver o cabeçalho do módulo.
                vad_filter=True,
            )
            # `transcribe` é lazy: nada roda até o gerador ser consumido, e é
            # aqui dentro da thread que ele tem que ser consumido.
            found = list(segments)
        except (RuntimeError, ValueError, OSError) as exc:
            raise VoiceBadAudio(f"o áudio não pôde ser transcrito: {exc}") from exc

        elapsed_ms = (time.monotonic() - started) * 1000
        text = " ".join(segment.text.strip() for segment in found).strip()

        # Média das probabilidades por segmento. `avg_logprob` é log, então
        # exponenciar devolve algo que se lê como probabilidade.
        probabilities = [
            math.exp(segment.avg_logprob)
            for segment in found
            if getattr(segment, "avg_logprob", None) is not None
        ]
        confidence = sum(probabilities) / len(probabilities) if probabilities else None

        return Transcript(
            text=text,
            duration_s=float(getattr(info, "duration", 0.0) or 0.0),
            elapsed_ms=elapsed_ms,
            language=getattr(info, "language", "en") or "en",
            confidence=confidence,
        )

    async def transcribe(self, audio: bytes) -> Transcript:
        if len(audio) > MAX_AUDIO_BYTES:
            raise VoiceBadAudio(
                f"áudio grande demais: {len(audio)} bytes, teto de {MAX_AUDIO_BYTES}"
            )
        if len(audio) < MIN_AUDIO_BYTES:
            # Clique no botão sem fala. Transcrever devolveria alucinação.
            return Transcript(text="", duration_s=0.0, elapsed_ms=0.0)

        async with self._lock:
            engine = await self._engine_ready()
            return await asyncio.to_thread(self._transcribe, engine, audio)

    @property
    def ready(self) -> bool:
        return self._engine is not None

    async def aclose(self) -> None:
        # O ctranslate2 libera o modelo quando o objeto é coletado; soltar a
        # referência é tudo o que há para fazer.
        self._engine = None
