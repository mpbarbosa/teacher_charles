"""Provider local via Ollama. É o default do projeto.

Formato confirmado contra um Ollama 0.35 de verdade: ``POST /api/chat`` com
``stream: true`` devolve NDJSON, uma linha por pedaço, cada uma com
``message.content``, e a última com ``done: true`` mais as estatísticas.
Modelo inexistente devolve 404 com ``{"error": "model '...' not found"}``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any, ClassVar

import httpx

from .base import (
    DEFAULT_OPTIONS,
    ChatProvider,
    GenerationOptions,
    Message,
    ModelNotFound,
    ProviderBadResponse,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
)

# Conectar num servidor local é instantâneo ou não vai acontecer; gerar texto
# num 4B em CPU é lento por natureza. Daí dois prazos diferentes.
CONNECT_TIMEOUT_S = 5.0


def _timeout(read_timeout_s: float) -> httpx.Timeout:
    return httpx.Timeout(read_timeout_s, connect=CONNECT_TIMEOUT_S)


async def list_models(base_url: str, timeout_s: float = 5.0) -> list[str]:
    """Nomes dos modelos baixados, ordenados.

    Vive aqui, e não em ``health.py``, para o conhecimento sobre o formato da
    API do Ollama ficar num só lugar.
    """
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.get(url)
            response.raise_for_status()
            payload = response.json()
    except httpx.TimeoutException as exc:
        raise ProviderTimeout(f"Ollama não respondeu em {timeout_s}s: {exc}") from exc
    except httpx.HTTPError as exc:
        raise ProviderUnavailable(f"Ollama não respondeu em {base_url}: {exc}") from exc
    except ValueError as exc:
        raise ProviderBadResponse(f"Ollama respondeu algo que não é JSON: {exc}") from exc

    models = payload.get("models", [])
    return sorted(m["name"] for m in models if isinstance(m, dict) and "name" in m)


class OllamaProvider(ChatProvider):
    name: ClassVar[str] = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        request_timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(model)
        self.base_url = base_url.rstrip("/")
        self.request_timeout_s = request_timeout_s
        # Cliente de vida longa: reaproveitar a conexão tira dezenas de ms de
        # cada turno, e numa conversa isso aparece.
        self._client = client or httpx.AsyncClient(timeout=_timeout(request_timeout_s))
        self._owns_client = client is None

    def _payload(
        self,
        messages: Sequence[Message],
        options: GenerationOptions,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        opts: dict[str, Any] = {"temperature": options.temperature}
        if options.max_tokens is not None:
            opts["num_predict"] = options.max_tokens
        if options.stop:
            opts["stop"] = list(options.stop)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.as_dict() for m in messages],
            "stream": stream,
            "options": opts,
        }
        # Só manda o campo quando há decisão explícita, para não sobrescrever o
        # default do servidor. Verificado: um modelo que não raciocina aceita
        # `think: false` sem erro, então isto é seguro de mandar.
        if options.think is not None:
            payload["think"] = options.think
        return payload

    async def stream(
        self,
        messages: Sequence[Message],
        options: GenerationOptions = DEFAULT_OPTIONS,
    ) -> AsyncIterator[str]:
        if not messages:
            raise ValueError("stream() precisa de ao menos uma mensagem")

        url = f"{self.base_url}/api/chat"
        payload = self._payload(messages, options, stream=True)

        try:
            async with self._client.stream("POST", url, json=payload) as response:
                if response.status_code >= 400:
                    # Num streaming o corpo ainda não foi lido; sem este aread()
                    # não há como dizer de que erro se trata.
                    raw = await response.aread()
                    raise self._error_for(response.status_code, raw)

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    chunk, done = self._parse_line(line)
                    if chunk:
                        yield chunk
                    if done:
                        return
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(
                f"modelo {self.model!r} não respondeu em {self.request_timeout_s}s. "
                f"Num modelo grande em CPU isso é esperado; baixe TC_LLM_MODEL "
                f"ou suba TC_REQUEST_TIMEOUT_S."
            ) from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise ProviderUnavailable(
                f"não deu para falar com o Ollama em {self.base_url}: {exc}. O serviço está de pé?"
            ) from exc

    def _error_for(self, status_code: int, raw: bytes) -> ProviderError:
        try:
            detail = json.loads(raw).get("error", "")
        except (ValueError, AttributeError):
            detail = raw.decode("utf-8", errors="replace")[:200]

        if status_code == 404:
            return ModelNotFound(
                f"o Ollama não conhece o modelo {self.model!r} ({detail}). "
                f"Rode: ollama pull {self.model}"
            )
        return ProviderError(f"Ollama devolveu HTTP {status_code}: {detail}")

    def _parse_line(self, line: str) -> tuple[str, bool]:
        """Devolve ``(texto, terminou)`` para uma linha do NDJSON."""
        try:
            data = json.loads(line)
        except ValueError as exc:
            raise ProviderBadResponse(f"linha do Ollama não é JSON: {line[:120]!r}") from exc

        if not isinstance(data, dict):
            raise ProviderBadResponse(f"linha do Ollama não é um objeto: {line[:120]!r}")

        # Um erro pode chegar no meio do stream, com HTTP 200 já enviado.
        if "error" in data:
            raise ProviderError(f"Ollama interrompeu a geração: {data['error']}")

        message = data.get("message") or {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        return content, bool(data.get("done"))

    async def aclose(self) -> None:
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()
