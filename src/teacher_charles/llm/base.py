"""O contrato que todo provider de modelo cumpre.

O resto da aplicação fala com esta interface e nunca com o Ollama ou com uma
API de nuvem diretamente. É o que sustenta a recomendação §6 da análise
inicial — *local por padrão, com nuvem opcional* — e o que permite a
`teaching/` ser testada sem nenhum modelo baixado.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    """Parâmetros de geração, iguais para todo provider.

    Cada implementação traduz para o formato da sua API. O que um provider não
    suportar, ele ignora — nunca falha por causa de uma opção.
    """

    temperature: float = 0.7
    max_tokens: int | None = None
    stop: tuple[str, ...] = ()

    # Modelos de raciocínio (Qwen3, entre outros) pensam antes de responder,
    # o que custa segundos. Conversa precisa responder em 1-2s, então isto
    # existe para poder desligar. None = não manda o campo, deixa o default
    # do servidor valer.
    think: bool | None = None


DEFAULT_OPTIONS = GenerationOptions()


# --- Erros ----------------------------------------------------------------- #
#
# Tipados de propósito: a camada de cima reage de formas diferentes a "não
# consigo falar com o servidor" e "o modelo pedido não existe", e distinguir
# isso por substring na mensagem de erro é frágil.


class ProviderError(RuntimeError):
    """Qualquer falha ao falar com o modelo."""


class ProviderUnavailable(ProviderError):
    """Não deu para alcançar o servidor do modelo."""


class ModelNotFound(ProviderError):
    """O servidor respondeu, mas não conhece o modelo pedido."""


class ProviderTimeout(ProviderError):
    """O servidor aceitou a requisição e não respondeu no prazo."""


class ProviderBadResponse(ProviderError):
    """O servidor respondeu algo que não dá para interpretar."""


# --- O contrato ------------------------------------------------------------ #


class ChatProvider(abc.ABC):
    """Base dos providers.

    Só ``stream`` é abstrato: ``complete`` é derivado dele, para não haver duas
    implementações que possam divergir.
    """

    name: ClassVar[str] = "base"

    def __init__(self, model: str) -> None:
        self.model = model

    @abc.abstractmethod
    def stream(
        self,
        messages: Sequence[Message],
        options: GenerationOptions = DEFAULT_OPTIONS,
    ) -> AsyncIterator[str]:
        """Produz a resposta em pedaços, na ordem em que o modelo a gera."""
        raise NotImplementedError

    async def complete(
        self,
        messages: Sequence[Message],
        options: GenerationOptions = DEFAULT_OPTIONS,
    ) -> str:
        """A resposta inteira. Útil para a etapa de análise, que precisa do
        JSON completo antes de poder validar qualquer coisa."""
        return "".join([chunk async for chunk in self.stream(messages, options)])

    async def aclose(self) -> None:
        """Libera recursos (conexões). Idempotente.

        Não é abstrato: um provider sem recursos a liberar — o fake, por
        exemplo — não deve ser obrigado a escrever um corpo vazio.
        """
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"
