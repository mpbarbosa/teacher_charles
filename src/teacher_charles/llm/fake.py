"""Provider falso, para os testes rodarem sem nenhum modelo baixado.

Isto não é um detalhe de conveniência: a estrutura pedagógica é o produto, e
ela tem que ser testável de forma determinística. Um teste que depende de um
modelo de verdade mede o modelo, não o código.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import ClassVar

from .base import DEFAULT_OPTIONS, ChatProvider, GenerationOptions, Message


@dataclass(slots=True)
class RecordedCall:
    messages: list[Message]
    options: GenerationOptions


class FakeProvider(ChatProvider):
    """Devolve respostas roteirizadas e guarda o que lhe foi pedido.

    Com ``replies`` de vários itens, cada chamada consome o próximo; a última
    resposta se repete se houver mais chamadas que respostas.
    """

    name: ClassVar[str] = "fake"

    def __init__(
        self,
        replies: Sequence[str] | str = "Looks good!",
        model: str = "fake-model",
        chunk_size: int = 4,
        error: Exception | None = None,
    ) -> None:
        super().__init__(model)
        self.replies: list[str] = [replies] if isinstance(replies, str) else list(replies)
        if not self.replies:
            raise ValueError("FakeProvider precisa de ao menos uma resposta")
        self.chunk_size = chunk_size
        # Para exercitar o tratamento de erro da camada de cima.
        self.error = error
        self.calls: list[RecordedCall] = []
        self.closed = False

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def last_call(self) -> RecordedCall:
        if not self.calls:
            raise AssertionError("o provider nunca foi chamado")
        return self.calls[-1]

    def system_prompt_of_last_call(self) -> str:
        return "\n".join(m.content for m in self.last_call.messages if m.role == "system")

    async def stream(
        self,
        messages: Sequence[Message],
        options: GenerationOptions = DEFAULT_OPTIONS,
    ) -> AsyncIterator[str]:
        if not messages:
            raise ValueError("stream() precisa de ao menos uma mensagem")

        index = len(self.calls)
        self.calls.append(RecordedCall(messages=list(messages), options=options))

        if self.error is not None:
            raise self.error

        reply = self.replies[min(index, len(self.replies) - 1)]
        # Fatiado de propósito: quem consome tem que lidar com pedaços, igual
        # ao provider de verdade.
        for start in range(0, len(reply), self.chunk_size):
            yield reply[start : start + self.chunk_size]

    async def aclose(self) -> None:
        self.closed = True
