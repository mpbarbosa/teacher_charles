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


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """O que o servidor contou sobre a chamada que acabou de terminar.

    Existe por causa da E7: sem ``prompt_cached`` não dá para dizer se o
    primeiro turno custou 7s por carga de modelo, por avaliação de prompt ou
    por evicção de cache — e "não está explicado" era a dívida que a §1.4
    deixou em aberto.

    Tudo é opcional porque é informação de servidor: um provider que não
    reporte métrica nenhuma devolve o objeto vazio em vez de inventar número.
    Durações em milissegundos.
    """

    load_ms: float | None = None
    prompt_tokens: int | None = None
    prompt_cached: int | None = None
    prompt_eval_ms: float | None = None
    output_tokens: int | None = None
    eval_ms: float | None = None
    total_ms: float | None = None

    @property
    def cache_hit_ratio(self) -> float | None:
        """Quanto do prompt veio do cache de prefixo. É o número da §5.5."""
        if not self.prompt_tokens or self.prompt_cached is None:
            return None
        return self.prompt_cached / self.prompt_tokens

    @property
    def output_tokens_per_s(self) -> float | None:
        if not self.output_tokens or not self.eval_ms:
            return None
        return self.output_tokens / (self.eval_ms / 1000)

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "load_ms": self.load_ms,
            "prompt_tokens": self.prompt_tokens,
            "prompt_cached": self.prompt_cached,
            "prompt_eval_ms": self.prompt_eval_ms,
            "output_tokens": self.output_tokens,
            "eval_ms": self.eval_ms,
            "total_ms": self.total_ms,
        }


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
        # Métricas da última chamada **concluída**, quando o servidor as
        # reporta. É best-effort e deliberadamente não é por requisição: com
        # duas chamadas ao mesmo tempo (conversa e análise), a última a
        # terminar sobrescreve. Serve para medir em série, que é o que
        # `scripts/measure_latency.py` faz; não serve para contabilidade.
        self.last_metrics: GenerationMetrics | None = None

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
