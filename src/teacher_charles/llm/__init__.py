"""Providers de modelo.

A aplicação pede um provider à ``build_provider`` e conversa só pelo contrato
de ``base.ChatProvider``. Nada fora deste pacote deve saber se o modelo está no
Ollama, num llama.cpp ou numa API de nuvem.
"""

from __future__ import annotations

from ..config import Settings
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
from .fake import FakeProvider
from .ollama import OllamaProvider, list_models
from .openai_compat import OpenAICompatProvider

__all__ = [
    "DEFAULT_OPTIONS",
    "ChatProvider",
    "FakeProvider",
    "GenerationOptions",
    "Message",
    "ModelNotFound",
    "OllamaProvider",
    "OpenAICompatProvider",
    "ProviderBadResponse",
    "ProviderError",
    "ProviderTimeout",
    "ProviderUnavailable",
    "build_provider",
    "default_options",
    "list_models",
]


class UnknownProvider(ProviderError):
    """``TC_PROVIDER`` nomeia um provider que não existe."""


def default_options(settings: Settings, **overrides: object) -> GenerationOptions:
    """Opções de geração com o que vier da configuração já aplicado.

    Fica aqui, e não em ``Settings``, para ``config`` não precisar importar
    ``llm`` — a dependência corre num sentido só.
    """
    values: dict[str, object] = {"think": settings.llm_think}
    values.update(overrides)
    return GenerationOptions(**values)  # type: ignore[arg-type]


def build_provider(settings: Settings) -> ChatProvider:
    """O provider que as configurações pedem.

    Falha cedo e com instrução: um provider mal configurado deve estourar aqui,
    na subida, e não no meio do primeiro turno do aluno.
    """
    if settings.provider == "ollama":
        return OllamaProvider(
            model=settings.llm_model,
            base_url=settings.ollama_url,
            request_timeout_s=settings.request_timeout_s,
        )

    if settings.provider == "openai_compat":
        if not settings.openai_api_key:
            raise UnknownProvider(
                "TC_PROVIDER=openai_compat exige TC_OPENAI_API_KEY. "
                "Para rodar tudo local, use TC_PROVIDER=ollama."
            )
        return OpenAICompatProvider(
            model=settings.llm_model,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            request_timeout_s=settings.request_timeout_s,
        )

    if settings.provider == "fake":
        # Permitido de propósito: deixa subir o app inteiro sem modelo, o que
        # é útil para trabalhar na interface.
        return FakeProvider(model="fake-model")

    raise UnknownProvider(
        f"provider desconhecido: {settings.provider!r}. "
        f"Use 'ollama' (local), 'openai_compat' (nuvem) ou 'fake' (sem modelo)."
    )
