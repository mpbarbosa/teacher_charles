"""Um turno da conversa, de ponta a ponta.

Este módulo é **agnóstico quanto à origem do texto**. Recebe uma string e
devolve eventos; não sabe se ela veio de um campo de texto, de uma transcrição
do Whisper ou de um teste. É o que permite a voz da Fase 1.5 acoplar duas
pontas (STT na entrada, TTS na saída) sem reescrever o meio.

Na E2 o turno é simples: persona + histórico -> resposta em streaming. A etapa
de análise que extrai correções estruturadas é da E4, e entra aqui como um
segundo passo, sem mudar esta interface.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from ..llm.base import ChatProvider, GenerationOptions, Message, ProviderError
from ..prompts import PromptLibrary
from ..store.repo import TurnRepo

# Quantos turnos do histórico vão no prompt.
#
# Não é só sobre a janela de contexto: nesta classe de hardware a avaliação do
# prompt custa tempo real (medido: ~54 tok/s num 3B), então histórico longo
# atrasa a primeira palavra. 20 turnos são ~10 trocas, o bastante para a
# conversa ter fio sem pagar por tudo o que já foi dito.
DEFAULT_HISTORY_TURNS = 20


@dataclass(frozen=True, slots=True)
class Chunk:
    """Um pedaço de texto do professor, para repassar na hora."""

    text: str
    kind: Literal["chunk"] = "chunk"


@dataclass(frozen=True, slots=True)
class Done:
    """Fim do turno, com o que foi gravado e quanto tempo levou."""

    reply: str
    student_turn_id: int
    teacher_turn_id: int
    first_chunk_ms: float
    total_ms: float
    kind: Literal["done"] = "done"


@dataclass(frozen=True, slots=True)
class Failed:
    """O turno não completou. A fala do aluno já está gravada."""

    message: str
    student_turn_id: int
    kind: Literal["failed"] = "failed"


TurnEvent = Chunk | Done | Failed


class TurnRunner:
    def __init__(
        self,
        provider: ChatProvider,
        prompts: PromptLibrary,
        conn: sqlite3.Connection,
        options: GenerationOptions | None = None,
        history_turns: int = DEFAULT_HISTORY_TURNS,
    ) -> None:
        self._provider = provider
        self._prompts = prompts
        self._turns = TurnRepo(conn)
        self._options = options or GenerationOptions()
        self._history_turns = history_turns

    def build_messages(self, session_id: int, student_text: str) -> list[Message]:
        """Persona, histórico, e por fim a mensagem nova.

        A persona vem primeiro e inalterada de propósito: o Ollama cacheia o
        prefixo do prompt (``prompt_eval_cached_count``), então um system prompt
        estável é avaliado uma vez e não a cada turno. Interpolar valores que
        mudam aqui jogaria esse cache fora.
        """
        messages = [Message(role="system", content=self._prompts.get("persona"))]

        # O histórico já inclui a fala do aluno recém-gravada, então a
        # mensagem nova não é acrescentada duas vezes.
        for turn in self._turns.history(session_id, limit=self._history_turns):
            role = "user" if turn.role == "student" else "assistant"
            messages.append(Message(role=role, content=turn.content))

        return messages

    async def run(self, session_id: int, student_text: str) -> AsyncIterator[TurnEvent]:
        text = student_text.strip()
        if not text:
            raise ValueError("a fala do aluno está vazia")

        # Grava a fala do aluno ANTES de chamar o modelo. Se a geração falhar,
        # o que o aluno disse não se perde.
        student_turn_id = self._turns.append(session_id, "student", text)

        messages = self.build_messages(session_id, text)

        pieces: list[str] = []
        started = time.monotonic()
        first_chunk_ms: float | None = None

        try:
            async for piece in self._provider.stream(messages, self._options):
                if not piece:
                    continue
                if first_chunk_ms is None:
                    first_chunk_ms = (time.monotonic() - started) * 1000
                pieces.append(piece)
                # Repassa na hora. Acumular aqui anularia o streaming, que é o
                # que torna 7-8 tok/s tolerável para o aluno.
                yield Chunk(text=piece)
        except ProviderError as exc:
            yield Failed(message=str(exc), student_turn_id=student_turn_id)
            return

        reply = "".join(pieces)
        if not reply.strip():
            yield Failed(
                message="o modelo não devolveu texto nenhum",
                student_turn_id=student_turn_id,
            )
            return

        teacher_turn_id = self._turns.append(session_id, "teacher", reply)

        yield Done(
            reply=reply,
            student_turn_id=student_turn_id,
            teacher_turn_id=teacher_turn_id,
            first_chunk_ms=first_chunk_ms or 0.0,
            total_ms=(time.monotonic() - started) * 1000,
        )
