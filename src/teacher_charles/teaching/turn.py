"""Um turno da conversa, de ponta a ponta.

Este módulo é **agnóstico quanto à origem do texto**. Recebe uma string e
devolve eventos; não sabe se ela veio de um campo de texto, de uma transcrição
do Whisper ou de um teste. É o que permite a voz da Fase 1.5 acoplar duas
pontas (STT na entrada, TTS na saída) sem reescrever o meio.

## O turno duplo da E4, e a ordem que a medição escolheu

O plano §2 descreve duas etapas por turno: análise e resposta. Quem informa a
resposta não é a análise, é a parte determinística:

- A **base contrastiva** casa por padrão, em microssegundos, e vai no prompt da
  resposta como dica. É daí que sai a explicação curada na prosa do professor —
  sem esperar modelo nenhum.
- A **análise** produz o dado estruturado: ``span``, ``correction``,
  ``rule_id``, ``explanation``, persistidos em ``correction``. Ela aparece ao
  aluno como anotação do que ele disse, não como texto do professor.

A primeira versão disparava as duas juntas, e a medição derrubou isso: o Ollama
atende um pedido por vez neste modelo, então "paralelo" aqui é ficção — a
análise só adiava a resposta. Medido (plano §5.2): **11,2s até a primeira
palavra**, contra 0,7s de um turno só de conversa.

Então a ordem default é **conversa primeiro, análise depois do último pedaço**:
o aluno lê a resposta enquanto a correção é calculada, e a anotação aparece
sobre o que ele escreveu alguns segundos depois. A primeira palavra volta ao
tempo da E2.

O modo paralelo continua existindo (``analysis_after_reply=False``) porque a
premissa que o derrubou é desta máquina, não do desenho: num provider de nuvem,
ou num servidor que atenda dois pedidos ao mesmo tempo, disparar as duas juntas
volta a ser o melhor arranjo. É configuração, não reescrita.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal

from ..llm.base import ChatProvider, GenerationOptions, Message, ProviderError
from ..prompts import PromptLibrary
from ..store.repo import CorrectionRepo, TurnRepo
from .analyzer import Analysis, Analyzer

# Quantos turnos do histórico vão no prompt.
#
# Não é só sobre a janela de contexto: nesta classe de hardware a avaliação do
# prompt custa tempo real (medido: ~54 tok/s num 3B), então histórico longo
# atrasa a primeira palavra. 20 turnos são ~10 trocas, o bastante para a
# conversa ter fio sem pagar por tudo o que já foi dito.
DEFAULT_HISTORY_TURNS = 20

# Quantas regras da base entram na dica da resposta. A persona pede uma
# correção por turno, duas no máximo; mais que três candidatos só encheria o
# prompt.
MAX_HINT_RULES = 3


@dataclass(frozen=True, slots=True)
class Chunk:
    """Um pedaço de texto do professor, para repassar na hora."""

    text: str
    kind: Literal["chunk"] = "chunk"


@dataclass(frozen=True, slots=True)
class Analyzed:
    """A análise do que o aluno disse, com o que foi gravado.

    Chega antes do ``Done`` e, em geral, antes de a resposta terminar.
    """

    analysis: Analysis
    student_turn_id: int
    correction_ids: tuple[int, ...] = ()
    kind: Literal["analysis"] = "analysis"


@dataclass(frozen=True, slots=True)
class Done:
    """Fim do turno, com o que foi gravado e quanto tempo levou."""

    reply: str
    student_turn_id: int
    teacher_turn_id: int
    first_chunk_ms: float
    total_ms: float
    analysis_ms: float = 0.0
    kind: Literal["done"] = "done"


@dataclass(frozen=True, slots=True)
class Failed:
    """O turno não completou. A fala do aluno já está gravada."""

    message: str
    student_turn_id: int
    kind: Literal["failed"] = "failed"


TurnEvent = Chunk | Analyzed | Done | Failed


async def _settle(task: asyncio.Task[Analysis]) -> None:
    """Espera a tarefa terminar sem nunca propagar o erro dela.

    ``await task`` levantaria aqui o que a análise levantou lá, e isso mataria
    o turno do aluno por causa de uma correção. ``asyncio.wait`` espera sem
    repassar; quem lê o resultado depois é o ``_collect``, que trata.
    """
    await asyncio.wait({task})


async def _discard(task: asyncio.Task[Analysis]) -> None:
    """Cancela a análise e recolhe o que sobrou.

    Usado quando a resposta falhou e a correção perdeu o destino. O
    ``exception()`` no fim existe só para o asyncio não reclamar de exceção
    não recolhida quando a tarefa havia falhado antes do cancelamento.
    """
    if task.done():
        if not task.cancelled():
            task.exception()
        return
    task.cancel()
    await asyncio.wait({task})
    if not task.cancelled():
        task.exception()


@dataclass(slots=True)
class _Pending:
    """A análise em andamento, e o que já foi feito com ela."""

    task: asyncio.Task[Analysis]
    emitted: bool = False
    elapsed_ms: float = field(default=0.0)


class TurnRunner:
    def __init__(
        self,
        provider: ChatProvider,
        prompts: PromptLibrary,
        conn: sqlite3.Connection,
        options: GenerationOptions | None = None,
        history_turns: int = DEFAULT_HISTORY_TURNS,
        analyzer: Analyzer | None = None,
        analysis_after_reply: bool = True,
    ) -> None:
        self._provider = provider
        self._prompts = prompts
        self._turns = TurnRepo(conn)
        self._corrections = CorrectionRepo(conn)
        self._options = options or GenerationOptions()
        self._history_turns = history_turns
        # Sem analisador o turno volta a ser o da E2: conversa e nada mais. É
        # assim que os testes de conversa seguem simples.
        self._analyzer = analyzer
        self._analysis_after_reply = analysis_after_reply

    # --- prompt da resposta ---------------------------------------------- #

    def rule_hint(self, student_text: str) -> str:
        """A dica curada que vai no prompt da resposta, ou string vazia.

        Determinística e de graça: vem do casamento por padrão, não de uma
        chamada ao modelo. É isto que faz a explicação na prosa do professor sair
        da regra revisada em vez da improvisação do 4B — o problema nº 1 da
        análise inicial.
        """
        if self._analyzer is None:
            return ""

        matches = self._analyzer.base.match(student_text)
        if not matches:
            return ""

        lines = []
        for match in matches[:MAX_HINT_RULES]:
            lines.append(f'- "{match.span}" — {match.item.explanation}')

        return (
            "The student may have made these mistakes. If you correct one, use this "
            "explanation instead of your own:\n" + "\n".join(lines)
        )

    def build_messages(self, session_id: int, student_text: str) -> list[Message]:
        """Persona, histórico, a mensagem nova e, por fim, a dica do turno.

        A persona vem primeiro e inalterada de propósito: o Ollama cacheia o
        prefixo do prompt (``prompt_eval_cached_count``), então um system prompt
        estável é avaliado uma vez e não a cada turno. Interpolar valores que
        mudam aqui jogaria esse cache fora.

        A dica das regras vai no **fim**, depois da fala do aluno, pelo mesmo
        motivo: o cache é de prefixo. Posta logo após a persona, ela empurraria
        todo o histórico para fora do prefixo comum com o turno anterior e faria
        o modelo reavaliar a conversa inteira a cada turno.
        """
        messages = [Message(role="system", content=self._prompts.get("persona"))]

        # O histórico já inclui a fala do aluno recém-gravada, então a
        # mensagem nova não é acrescentada duas vezes.
        for turn in self._turns.history(session_id, limit=self._history_turns):
            role = "user" if turn.role == "student" else "assistant"
            messages.append(Message(role=role, content=turn.content))

        hint = self.rule_hint(student_text)
        if hint:
            messages.append(Message(role="system", content=hint))

        return messages

    # --- análise ---------------------------------------------------------- #

    def _start_analysis(self, student_text: str) -> _Pending | None:
        if self._analyzer is None:
            return None
        return _Pending(task=asyncio.create_task(self._analyzer.analyze(student_text)))

    def _persist(self, student_turn_id: int, analysis: Analysis) -> tuple[int, ...]:
        return tuple(
            self._corrections.add(
                student_turn_id,
                span=finding.span,
                corrected=finding.correction,
                error_type=finding.error_type,
                explanation=finding.explanation,
                rule_id=finding.rule_id,
            )
            for finding in analysis.findings
        )

    def _collect(self, pending: _Pending, student_turn_id: int) -> Analyzed | None:
        """Grava a análise concluída e devolve o evento. Nunca levanta."""
        pending.emitted = True
        try:
            analysis = pending.task.result()
        except asyncio.CancelledError:
            return None
        except Exception:  # noqa: BLE001 - análise nunca derruba o turno
            # O Analyzer já degrada por conta própria; isto cobre o inesperado.
            return None

        pending.elapsed_ms = analysis.elapsed_ms
        return Analyzed(
            analysis=analysis,
            student_turn_id=student_turn_id,
            correction_ids=self._persist(student_turn_id, analysis),
        )

    # --- o turno ---------------------------------------------------------- #

    async def run(self, session_id: int, student_text: str) -> AsyncIterator[TurnEvent]:
        text = student_text.strip()
        if not text:
            raise ValueError("a fala do aluno está vazia")

        # Grava a fala do aluno ANTES de chamar o modelo. Se a geração falhar,
        # o que o aluno disse não se perde.
        student_turn_id = self._turns.append(session_id, "student", text)

        messages = self.build_messages(session_id, text)

        # Quando as duas etapas disputam o mesmo modelo, começar a análise
        # aqui atrasa a primeira palavra do professor em segundos (medido:
        # 11,2s contra 0,7s). Default: ela espera o fim da resposta.
        pending = None if self._analysis_after_reply else self._start_analysis(text)

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

                # A correção aparece assim que fica pronta, mesmo com o
                # professor ainda digitando: ela anota o que o aluno disse, não
                # interrompe o que o professor está dizendo.
                if pending is not None and not pending.emitted and pending.task.done():
                    event = self._collect(pending, student_turn_id)
                    if event is not None:
                        yield event
        except ProviderError as exc:
            # A resposta morreu; a análise perde o destino. Cancelar evita
            # tarefa pendurada depois do fim do turno.
            if pending is not None:
                await _discard(pending.task)
            yield Failed(message=str(exc), student_turn_id=student_turn_id)
            return

        reply = "".join(pieces)
        if not reply.strip():
            if pending is not None:
                await _discard(pending.task)
            yield Failed(
                message="o modelo não devolveu texto nenhum",
                student_turn_id=student_turn_id,
            )
            return

        teacher_turn_id = self._turns.append(session_id, "teacher", reply)

        # A resposta saiu inteira; agora a análise tem o modelo para si. O
        # aluno está lendo, então este tempo não é tela parada.
        if pending is None:
            pending = self._start_analysis(text)

        # O Analyzer tem timeout próprio, então esperar aqui não pendura o
        # turno.
        if pending is not None and not pending.emitted:
            await _settle(pending.task)
            event = self._collect(pending, student_turn_id)
            if event is not None:
                yield event

        yield Done(
            reply=reply,
            student_turn_id=student_turn_id,
            teacher_turn_id=teacher_turn_id,
            first_chunk_ms=first_chunk_ms or 0.0,
            total_ms=(time.monotonic() - started) * 1000,
            analysis_ms=pending.elapsed_ms if pending is not None else 0.0,
        )
