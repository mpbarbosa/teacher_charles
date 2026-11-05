"""Provider de nuvem, para APIs compatíveis com OpenAI.

Existe por causa da recomendação §6 da análise inicial: *local por padrão, com
nuvem opcional*. Fica atrás de ``TC_PROVIDER=openai_compat`` e nunca é usado
sem escolha explícita — o default do produto é não mandar as conversas do aluno
para fora da máquina.

Serve qualquer servidor que fale o dialeto ``/chat/completions``, inclusive
llama.cpp e vLLM locais, que é o caminho se o Ollama sair do caminho mais tarde.
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

CONNECT_TIMEOUT_S = 10.0
_SSE_DATA_PREFIX = "data:"
_SSE_DONE = "[DONE]"


class OpenAICompatProvider(ChatProvider):
    name: ClassVar[str] = "openai_compat"

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        request_timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(model)
        self.base_url = base_url.rstrip("/")
        self.request_timeout_s = request_timeout_s
        self._api_key = api_key
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(request_timeout_s, connect=CONNECT_TIMEOUT_S),
            headers=headers,
        )
        self._owns_client = client is None

    def _payload(
        self,
        messages: Sequence[Message],
        options: GenerationOptions,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.as_dict() for m in messages],
            "stream": stream,
            "temperature": options.temperature,
        }
        if options.max_tokens is not None:
            payload["max_tokens"] = options.max_tokens
        if options.stop:
            payload["stop"] = list(options.stop)
        # `think` é específico do Ollama; aqui não tem equivalente portável.
        # O contrato diz que opção não suportada é ignorada, não é erro.
        return payload

    async def stream(
        self,
        messages: Sequence[Message],
        options: GenerationOptions = DEFAULT_OPTIONS,
    ) -> AsyncIterator[str]:
        if not messages:
            raise ValueError("stream() precisa de ao menos uma mensagem")

        url = f"{self.base_url}/chat/completions"
        payload = self._payload(messages, options, stream=True)

        try:
            async with self._client.stream("POST", url, json=payload) as response:
                if response.status_code >= 400:
                    raw = await response.aread()
                    raise self._error_for(response.status_code, raw)

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith(_SSE_DATA_PREFIX):
                        # Comentários de keep-alive e linhas em branco do SSE.
                        continue
                    data = line[len(_SSE_DATA_PREFIX) :].strip()
                    if data == _SSE_DONE:
                        return
                    chunk = self._parse_event(data)
                    if chunk:
                        yield chunk
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(
                f"{self.base_url} não respondeu em {self.request_timeout_s}s"
            ) from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise ProviderUnavailable(f"não deu para falar com {self.base_url}: {exc}") from exc

    def _error_for(self, status_code: int, raw: bytes) -> ProviderError:
        try:
            body = json.loads(raw)
            error = body.get("error", body)
            detail = error.get("message", "") if isinstance(error, dict) else str(error)
        except (ValueError, AttributeError):
            detail = raw.decode("utf-8", errors="replace")[:200]

        if status_code == 404:
            return ModelNotFound(f"{self.base_url} não conhece o modelo {self.model!r}: {detail}")
        if status_code in (401, 403):
            return ProviderError(
                f"{self.base_url} recusou a credencial (HTTP {status_code}): {detail}. "
                f"Confira TC_OPENAI_API_KEY."
            )
        if status_code == 429:
            return ProviderError(f"{self.base_url} aplicou rate limit: {detail}")
        return ProviderError(f"{self.base_url} devolveu HTTP {status_code}: {detail}")

    def _parse_event(self, data: str) -> str:
        try:
            payload = json.loads(data)
        except ValueError as exc:
            raise ProviderBadResponse(f"evento SSE não é JSON: {data[:120]!r}") from exc

        if isinstance(payload, dict) and "error" in payload:
            raise ProviderError(f"a API interrompeu a geração: {payload['error']}")

        choices = payload.get("choices") or []
        if not choices:
            return ""
        delta = choices[0].get("delta") or {}
        return delta.get("content") or ""

    async def aclose(self) -> None:
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()
