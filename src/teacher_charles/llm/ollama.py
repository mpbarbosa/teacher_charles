"""Provider local via Ollama. É o default do projeto.

Formato confirmado contra um Ollama 0.35 de verdade: ``POST /api/chat`` com
``stream: true`` devolve NDJSON, uma linha por pedaço, cada uma com
``message.content``, e a última com ``done: true`` mais as estatísticas.
Modelo inexistente devolve 404 com ``{"error": "model '...' not found"}``.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from typing import Any, ClassVar

import httpx

from .base import (
    DEFAULT_OPTIONS,
    ChatProvider,
    GenerationMetrics,
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

# ``/api/show`` reads a manifest off disk and loads no model, so it gets the
# short deadline: a server that cannot answer it in this long is not going to
# be saved by the question.
SHOW_TIMEOUT_S = 5.0

# The request's ``think`` field reaches a model in exactly one way: the chat
# template reads it, as the variable ``.Think``. The word boundary is what
# keeps out the two lookalikes every thinking template carries -- ``.Thinking``
# (a past message's stored reasoning) and ``.IsThinkSet`` (whether the request
# had the field at all, true for ``false`` as much as for ``true``).
_READS_THINK_FLAG = re.compile(r"\.Think\b")

# A template that follows the assistant header with ``<think>`` has started
# the reasoning before the model writes a token, so the model reasons on every
# turn. That is the only evidence of reasoning ``/api/show`` gives. The
# capability list is not: ``qwen3:4b-instruct`` lists ``thinking`` and answers
# a one-word question in two tokens, with the field sent or left out.
_OPENS_THINK_BLOCK = re.compile(r"assistant\s*<think>")

log = logging.getLogger(__name__)


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
        # Whether ``think: false`` is safe to send to this model. ``None``
        # until the server has been asked; see ``_think_off_is_obeyed``.
        self._think_off_obeyed: bool | None = None

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
        # Sent only on an explicit decision, so the server's default is not
        # overridden. By the time the options get here ``stream`` has already
        # dropped a ``False`` the model would not obey.
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
        if options.think is False and not await self._think_off_is_obeyed():
            options = replace(options, think=None)
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

    async def _think_off_is_obeyed(self) -> bool:
        """Whether ``think: false`` turns this model's reasoning off.

        It is asked, and not assumed, because the field does two things in
        Ollama and only one of them is up to the model. It is offered to the
        chat template, which may use it to tell the model not to reason; and
        it stops the server from splitting reasoning out of the reply into
        ``message.thinking``. A template that never reads the flag leaves the
        model reasoning exactly as before -- and now nothing separates it.

        Measured on Ollama 0.35 with ``qwen3:4b``, which is the thinking-only
        Qwen3-4B-Thinking-2507 and whose template opens ``<think>`` whatever
        the request says: ``think: false`` returned the whole chain of thought,
        a bare ``</think>`` and then the answer, all in ``message.content`` --
        179 tokens against 247 with ``think: true``, so nothing was saved
        either. The student read every word of it. With the field left out,
        the same model's content was the answer alone.

        So ``false`` goes on the wire only for a template that reads it. For
        every other model the field is left out, which is the one setting
        under which the server hides whatever reasoning there is. That costs
        nothing on a model that does not reason, and on one that cannot stop
        it trades a leak for a wait. A model whose template is not a Go
        template at all lands on the same cautious side: ``qwen3:4b-instruct``
        shows a Jinja one, has no ``.Think`` to find, and measured the same
        two-token answer with the field sent or left out.

        The warning is for the one case the template proves: it opens the
        ``<think>`` block itself, so the operator asked for reasoning off and
        is not getting it.

        A failed lookup is not remembered, so a server that was down at the
        first call is asked again at the next; it answers "no" meanwhile
        because that is the side that cannot leak.
        """
        if self._think_off_obeyed is not None:
            return self._think_off_obeyed

        try:
            response = await self._client.post(
                f"{self.base_url}/api/show",
                json={"model": self.model},
                timeout=SHOW_TIMEOUT_S,
            )
            response.raise_for_status()
            shown = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            # No typed error from here: the chat request that follows hits the
            # same server and says what is wrong in the terms the caller knows.
            log.debug("could not ask Ollama how %s takes `think`: %s", self.model, exc)
            return False

        if not isinstance(shown, dict):
            return False

        template = shown.get("template")
        if not isinstance(template, str):
            template = ""
        obeyed = bool(_READS_THINK_FLAG.search(template))
        self._think_off_obeyed = obeyed

        if obeyed:
            return True
        if _OPENS_THINK_BLOCK.search(template):
            log.warning(
                "model %s reasons before answering and its template ignores "
                "`think: false`, so the reasoning cannot be turned off. It is "
                "hidden from the reply instead, and the wait for the first "
                "word includes it. A non-thinking model is the only way to get "
                "that time back.",
                self.model,
            )
        else:
            log.debug("`think: false` is not sent to %s: its template does not read it", self.model)
        return False

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

    @staticmethod
    def _ms(value: object) -> float | None:
        """Nanossegundos do Ollama em milissegundos."""
        if isinstance(value, (int, float)):
            return value / 1_000_000
        return None

    def _record_metrics(self, data: dict[str, Any]) -> None:
        """Guarda o que a linha final do Ollama contou sobre a chamada.

        `prompt_eval_count` é o que foi avaliado agora; o que estava em cache
        o Ollama reporta em `prompt_cache_hit_count` nas versões que o têm, e
        em `prompt_eval_cached_count` nas que não. Lemos os dois nomes porque
        o campo mudou entre versões e perder a métrica por causa disso seria
        perder justamente o número que a E7 foi medir.
        """
        cached = data.get("prompt_cache_hit_count")
        if cached is None:
            cached = data.get("prompt_eval_cached_count")

        self.last_metrics = GenerationMetrics(
            load_ms=self._ms(data.get("load_duration")),
            prompt_tokens=data.get("prompt_eval_count"),
            prompt_cached=cached if isinstance(cached, int) else None,
            prompt_eval_ms=self._ms(data.get("prompt_eval_duration")),
            output_tokens=data.get("eval_count"),
            eval_ms=self._ms(data.get("eval_duration")),
            total_ms=self._ms(data.get("total_duration")),
        )

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

        # ``message.thinking`` is deliberately not read. It is where the
        # server puts a model's reasoning, and nothing above this layer has a
        # use for it: the student must never see it, and the analyzer wants
        # the JSON that comes after.
        message = data.get("message") or {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        done = bool(data.get("done"))
        if done:
            self._record_metrics(data)
        return content, done

    async def aclose(self) -> None:
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()
