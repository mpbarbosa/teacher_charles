"""TTS com Kokoro-82M em ONNX, na CPU.

82 milhões de parâmetros contra os bilhões do LLM: aqui o orçamento da §1.3 não
aperta, e o que importa é outra coisa — o **fator de tempo real**. Acima de 1,0
a síntese não acompanha a reprodução e o aluno ouve a voz engasgar entre
sentenças, por mais rápido que o modelo de linguagem seja.

## Uma sentença por chamada

``speak`` sintetiza um trecho curto, e é o ``voice/segment.py`` que decide o
que é um trecho. Não é limitação do Kokoro — ele aceita texto longo e tem
streaming próprio — é o desenho que faz a voz caber nesta máquina: a primeira
sentença do professor é sintetizada enquanto o modelo escreve a segunda.

## Threads vêm de fora

O ``create_session`` do ``kokoro-onnx`` não expõe ``SessionOptions``, então a
sessão é montada aqui e entregue pronta pelo ``from_session``. Só isso dá o
controle de threads, e controle de threads é o que separa "TTS e LLM dividem
14 threads" de "os dois pedem as 14 e os dois ficam mais lentos".
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from .base import Speech, TextToSpeech, VoiceError, VoiceModelNotFound, VoiceUnavailable

log = logging.getLogger(__name__)

# Trecho maior que isto não é sentença, é parágrafo — e sintetizar parágrafo
# numa chamada só é justamente o que o TTS incremental existe para não fazer.
# O teto corta antes de o Kokoro estourar o limite de fonemas dele (510).
MAX_SPEECH_CHARS = 600


class KokoroTTS(TextToSpeech):
    """A boca do Charles."""

    def __init__(
        self,
        model_path: Path | str,
        voices_path: Path | str,
        voice: str = "bm_george",
        speed: float = 1.0,
        cpu_threads: int = 0,
    ) -> None:
        self._model_path = Path(model_path)
        self._voices_path = Path(voices_path)
        self.model = self._model_path.name
        self.voice = voice
        self._speed = speed
        self._cpu_threads = cpu_threads
        self._engine: Any | None = None
        self._np: Any | None = None
        # Mesma razão do STT: código nativo síncrono, limitado por CPU.
        self._lock = asyncio.Lock()

    # --- carga ------------------------------------------------------------ #

    def _load(self) -> Any:
        """Carrega modelo e vozes. Síncrono: sempre chamado em thread.

        Os arquivos são conferidos **antes** do import. É a checagem barata, e
        descobrir que falta um peso não deveria custar o carregamento de dois
        runtimes nativos. Também é o que mantém este caminho testável sem o
        extra ``voice`` instalado.
        """
        missing = [p for p in (self._model_path, self._voices_path) if not p.is_file()]
        if missing:
            raise VoiceModelNotFound(
                "pesos de TTS ausentes: "
                + ", ".join(str(p) for p in missing)
                + ". Rode `uv run python scripts/fetch_voice_models.py`."
            )

        try:
            import numpy as np
            import onnxruntime as rt
            from kokoro_onnx import Kokoro
        except ImportError as exc:  # pragma: no cover - depende do extra
            raise VoiceUnavailable(
                "kokoro-onnx não está instalado. Rode `uv sync --extra voice`."
            ) from exc

        # O numpy só é usado na conversão de amostras, e o import no topo do
        # módulo obrigaria quem não tem o extra a tê-lo só para importar este
        # arquivo. Guardado no objeto para o `_speak` não reimportar por
        # sentença.
        self._np = np

        options = rt.SessionOptions()
        if self._cpu_threads > 0:
            options.intra_op_num_threads = self._cpu_threads

        try:
            session = rt.InferenceSession(
                str(self._model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            engine = Kokoro.from_session(session, str(self._voices_path))
        except Exception as exc:  # noqa: BLE001 - o onnxruntime levanta de tudo
            raise VoiceModelNotFound(f"TTS {self._model_path.name!r} não carregou: {exc}") from exc

        available = set(engine.get_voices())
        if self.voice not in available:
            raise VoiceModelNotFound(
                f"voz {self.voice!r} não existe neste arquivo. "
                f"Há {len(available)}; por exemplo: {', '.join(sorted(available)[:6])}"
            )
        return engine

    async def _engine_ready(self) -> Any:
        if self._engine is None:
            self._engine = await asyncio.to_thread(self._load)
        return self._engine

    async def warm_up(self) -> None:
        """Carrega **e** sintetiza uma sentença curta.

        A primeira chamada ao onnxruntime paga alocação de arenas e o primeiro
        traçado do grafo; só carregar deixaria esse custo para a primeira
        sentença que o aluno espera ouvir. É a mesma lição do aquecimento do
        LLM (plano §1.4): aquecer com o que o turno de fato manda.
        """
        try:
            async with self._lock:
                engine = await self._engine_ready()
                await asyncio.to_thread(self._speak, engine, "Hello there.", self.voice)
            self._last_error = None
            log.info("TTS %s carregado, voz %s", self.model, self.voice)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - aquecimento nunca derruba a subida
            # Guardado para o /health poder dizer POR QUE a voz não atende.
            self._last_error = str(exc)
            log.warning("aquecimento do TTS falhou (segue assim mesmo): %s", exc)

    # --- síntese ---------------------------------------------------------- #

    def _speak(self, engine: Any, text: str, voice: str) -> Speech:
        started = time.monotonic()
        try:
            samples, sample_rate = engine.create(
                text,
                voice=voice,
                speed=self._speed,
                lang="en-us",
            )
        except Exception as exc:  # noqa: BLE001 - phonemizer e onnx levantam de tudo
            raise VoiceError(f"a síntese falhou para {text[:40]!r}: {exc}") from exc

        # O Kokoro devolve float32 em [-1, 1]; a borda fala PCM 16-bit. O
        # clip não é zelo: um pico acima de 1.0 viraria estalo ao truncar.
        pcm = (self._np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        return Speech(
            pcm=pcm,
            sample_rate=int(sample_rate),
            elapsed_ms=(time.monotonic() - started) * 1000,
        )

    async def speak(self, text: str, voice: str | None = None) -> Speech:
        trimmed = text.strip()
        if not trimmed:
            raise VoiceError("não há texto para falar")
        if len(trimmed) > MAX_SPEECH_CHARS:
            raise VoiceError(
                f"trecho longo demais para uma chamada: {len(trimmed)} caracteres, "
                f"teto de {MAX_SPEECH_CHARS}. Divida com voice/segment.py."
            )

        async with self._lock:
            engine = await self._engine_ready()
            return await asyncio.to_thread(self._speak, engine, trimmed, voice or self.voice)

    def voices(self) -> tuple[str, ...]:
        """As vozes do arquivo carregado, ou vazio se ele ainda não carregou.

        Não força a carga de propósito: isto é chamado pelo ``/health``, e um
        health check não deveria ter efeito colateral de 300 MB.
        """
        if self._engine is None:
            return ()
        return tuple(self._engine.get_voices())

    @property
    def ready(self) -> bool:
        return self._engine is not None

    async def aclose(self) -> None:
        self._engine = None
