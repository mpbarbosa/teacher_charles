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
import logging
import sqlite3
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal

from ..llm.base import ChatProvider, GenerationOptions, Message, ProviderError
from ..prompts import PromptLibrary
from ..store.models import Student
from ..store.repo import CorrectionRepo, SessionRepo, TurnRepo
from .analyzer import Analysis, Analyzer
from .level import LevelDecision, LevelTracker
from .level import partial_name as level_partial_name
from .modes import (
    DEFAULT_MODE,
    DrillPlanner,
    DrillTarget,
    drill_block,
    drill_reminder,
    max_findings,
)
from .modes import partial_name as mode_partial_name

log = logging.getLogger(__name__)

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

# Nível usado quando o turno não traz aluno nenhum — testes de conversa, e o
# script de medição. É o nível de entrada do produto (plano §4).
DEFAULT_LEVEL = "B1"


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
class Leveled:
    """O que este turno disse sobre o nível do aluno.

    Sai em todo turno com perfil, mesmo quando nada muda: a UI mostra o nível
    em vigor, e um evento só na mudança deixaria a tela defasada depois de um
    reinício.
    """

    decision: LevelDecision
    kind: Literal["level"] = "level"


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


TurnEvent = Chunk | Analyzed | Leveled | Done | Failed


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
        tracker: LevelTracker | None = None,
        planner: DrillPlanner | None = None,
    ) -> None:
        self._provider = provider
        self._prompts = prompts
        self._turns = TurnRepo(conn)
        self._sessions = SessionRepo(conn)
        self._corrections = CorrectionRepo(conn)
        self._options = options or GenerationOptions()
        self._history_turns = history_turns
        # Sem analisador o turno volta a ser o da E2: conversa e nada mais. É
        # assim que os testes de conversa seguem simples.
        self._analyzer = analyzer
        self._analysis_after_reply = analysis_after_reply
        # Sem tracker o turno não acompanha nível: é assim que os testes de
        # conversa seguem simples, e é o que o script de medição usa.
        self._tracker = tracker
        # Sem planner o modo drill ainda funciona: o bloco diz que não há regra
        # para treinar, que é a verdade quando a base não carregou.
        self._planner = planner

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

    def level_prompt(self, level: str) -> str:
        """O ajuste de registro deste nível, ou string vazia.

        Partial ausente não é erro: o turno segue com a persona sozinha, que é
        o comportamento da E2. Um arquivo de prompt que falta não deveria
        derrubar a aula do aluno.
        """
        return self._optional_prompt(level_partial_name(level))

    def mode_prompt(self, mode: str) -> str:
        """O comportamento deste modo, ou string vazia (mesma regra do nível)."""
        return self._optional_prompt(mode_partial_name(mode))

    def _optional_prompt(self, name: str) -> str:
        return self._prompts.get(name) if self._prompts.exists(name) else ""

    def stable_prefix(
        self,
        level: str = DEFAULT_LEVEL,
        mode: str = DEFAULT_MODE,
    ) -> list[Message]:
        """As mensagens de sistema que não mudam de turno para turno.

        Persona, ajuste de nível e comportamento do modo — nesta ordem, da que
        muda menos para a que muda mais. É **este** prefixo que o Ollama
        mantém em cache, e é por isso que ele existe como método: o
        aquecimento aquece exatamente o que o turno manda.

        Medido na E7 (plano §5.5): quando o aquecimento mandava só a persona e
        o turno mandava persona + nível + modo, o primeiro turno chegava com
        53% do prompt em cache e pagava 11,6s de avaliação. O aquecimento não
        tinha envelhecido mal por descuido — ele duplicava uma lista que a E5 e
        a E6 depois mudaram. Duplicata some; sobra uma lista só.
        """
        messages = [Message(role="system", content=self._prompts.get("persona"))]

        adjustment = self.level_prompt(level)
        if adjustment:
            messages.append(Message(role="system", content=adjustment))

        behaviour = self.mode_prompt(mode)
        if behaviour:
            messages.append(Message(role="system", content=behaviour))

        return messages

    def build_messages(
        self,
        session_id: int,
        student_text: str,
        level: str = DEFAULT_LEVEL,
        mode: str = DEFAULT_MODE,
        drill: DrillTarget | None = None,
    ) -> list[Message]:
        """Persona, nível, modo, dados do drill, histórico e, por fim, os pedidos.

        A ordem tem duas metades e dois critérios. No começo, **estabilidade**:
        o que muda menos vem antes, porque o cache é de prefixo. No fim,
        **eficácia**: a dica curada e o pedido do drill ficam depois da fala do
        aluno, porque instrução longe do ponto de geração se dilui — medido na
        E4 com a dica e de novo na E6 com o drill (plano §5.2 e §5.4).

        A persona vem primeiro e inalterada de propósito: o Ollama cacheia o
        prefixo do prompt (``prompt_eval_cached_count``), então um system prompt
        estável é avaliado uma vez e não a cada turno. Interpolar valores que
        mudam aqui jogaria esse cache fora.

        O ajuste de nível é a **segunda** mensagem, nunca texto interpolado na
        persona (plano §1.4). Como segunda mensagem ele só invalida o cache
        quando o nível de fato muda — o que acontece a cada muitos turnos, e não
        a cada turno. Dentro da persona, invalidaria sempre.

        A dica das regras vai no **fim**, depois da fala do aluno, pelo mesmo
        motivo: o cache é de prefixo. Posta logo após a persona, ela empurraria
        todo o histórico para fora do prefixo comum com o turno anterior e faria
        o modelo reavaliar a conversa inteira a cada turno.
        """
        # Persona, nível e modo: a parte estável, que o aquecimento já pôs no
        # cache. Nível antes de modo porque o cache é de prefixo e o nível muda
        # menos — trocar de modo não deveria invalidar o ajuste de nível que já
        # estava lá.
        messages = self.stable_prefix(level, mode)

        # O bloco do drill é dado apurado pela aplicação (qual regra, com que
        # palavras do aluno), não comportamento: o "como treinar" está no
        # partial acima. Estável dentro da sessão, então não custa cache.
        if mode == "drill":
            messages.append(Message(role="system", content=drill_block(drill)))

        # O histórico já inclui a fala do aluno recém-gravada, então a
        # mensagem nova não é acrescentada duas vezes.
        for turn in self._turns.history(session_id, limit=self._history_turns):
            role = "user" if turn.role == "student" else "assistant"
            messages.append(Message(role=role, content=turn.content))

        # A dica curada e o pedido do drill vão no fim, depois da fala do
        # aluno: medido na E4 e na E6, instrução longe do ponto de geração se
        # dilui. Mas vão **dentro da última mensagem do aluno**, e não como
        # mensagem de sistema — ver ``_with_notes``.
        notes = [self.rule_hint(student_text)]
        if mode == "drill":
            notes.append(drill_reminder(drill))

        return self._with_notes(messages, [note for note in notes if note])

    @staticmethod
    def _with_notes(messages: list[Message], notes: list[str]) -> list[Message]:
        """Anexa as notas do turno à última mensagem do aluno.

        Elas já estiveram como mensagem de sistema no fim da lista, e a E7
        mediu o preço disso: **o histórico inteiro parava de ser cacheado**. O
        total cacheado ficava preso no prefixo estável (674 tokens, turno após
        turno) enquanto o prompt crescia para 950, e a primeira palavra ia de
        2,8s para 9,1s em seis turnos.

        A causa não é o cache do Ollama, é o template de chat: ele junta as
        mensagens de sistema no topo. Uma mensagem de sistema no fim da lista é
        **renderizada logo depois da persona**, então ela muda a cada turno
        dentro do prefixo — e invalida tudo o que vem depois. Era isso que a
        §5.2 tinha diagnosticado como evicção causada pela análise; a análise
        não tinha nada a ver (plano §5.5).

        Dentro da última mensagem do aluno, a nota fica onde se pretendia que
        ficasse — no fim do prompt renderizado —, e o que muda a cada turno
        passa a ficar depois de todo o histórico, que volta a ser cacheado.
        """
        if not notes or not messages:
            return messages

        last = messages[-1]
        if last.role != "user":
            # Sem fala do aluno no fim não há onde anexar sem bagunçar a
            # ordem; cai no comportamento antigo, que é raro e inofensivo.
            return [*messages, Message(role="system", content="\n\n".join(notes))]

        joined = "\n\n".join(notes)
        return [
            *messages[:-1],
            Message(role="user", content=f"{last.content}\n\n[Teacher's notes: {joined}]"),
        ]

    # --- análise ---------------------------------------------------------- #

    def _start_analysis(
        self,
        student_text: str,
        level: str = DEFAULT_LEVEL,
        mode: str = DEFAULT_MODE,
    ) -> _Pending | None:
        if self._analyzer is None:
            return None
        # O teto de correções é do modo: em `correction` o aluno pediu os erros.
        return _Pending(
            task=asyncio.create_task(
                self._analyzer.analyze(student_text, level, max_findings=max_findings(mode))
            )
        )

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

    # --- modo -------------------------------------------------------------- #

    def session_mode(self, session_id: int) -> tuple[str, str | None]:
        """O modo da sessão e a regra em treino, com default seguro.

        Sessão que não existe mais (teste, banco limpo no meio do caminho) cai
        no modo de conversa em vez de derrubar o turno: perder a aula por causa
        de uma linha de sessão seria desproporcional.
        """
        try:
            session = self._sessions.get(session_id)
        except LookupError:
            return DEFAULT_MODE, None
        return session.mode, session.drill_rule_id

    def drill_target(self, student: Student | None, rule_id: str | None) -> DrillTarget | None:
        """A regra em treino, resolvida em alvo com as frases do próprio aluno.

        A escolha da regra acontece quando a sessão de drill começa, não aqui:
        uma regra por sessão (ver ``teaching/modes.py``).
        """
        if self._planner is None or student is None or not rule_id:
            return None
        return self._planner.target_for(student.id, rule_id)

    # --- nível ------------------------------------------------------------ #

    def _observe_level(
        self,
        student: Student,
        student_turn_id: int,
        text: str,
        analysis: Analysis | None,
    ) -> Leveled | None:
        """Acompanha o nível. Nunca levanta: a resposta do professor já saiu.

        O nível novo vale a partir do **próximo** turno — este já foi gerado com
        o registro antigo. Reabrir o prompt para refazer a resposta custaria o
        turno inteiro para ganhar um ajuste de registro que o aluno não pediu.
        """
        if self._tracker is None:
            return None
        try:
            decision = self._tracker.observe(student.id, student_turn_id, text, analysis)
        except Exception:  # noqa: BLE001 - nível nunca derruba o turno
            log.exception("acompanhamento de nível falhou (turno segue)")
            return None
        return Leveled(decision=decision)

    # --- o turno ---------------------------------------------------------- #

    async def run(
        self,
        session_id: int,
        student_text: str,
        student: Student | None = None,
    ) -> AsyncIterator[TurnEvent]:
        text = student_text.strip()
        if not text:
            raise ValueError("a fala do aluno está vazia")

        level = student.cefr_level if student is not None else DEFAULT_LEVEL
        mode, rule_id = self.session_mode(session_id)
        drill = self.drill_target(student, rule_id) if mode == "drill" else None

        # Grava a fala do aluno ANTES de chamar o modelo. Se a geração falhar,
        # o que o aluno disse não se perde.
        student_turn_id = self._turns.append(session_id, "student", text)

        messages = self.build_messages(session_id, text, level, mode, drill)

        # Quando as duas etapas disputam o mesmo modelo, começar a análise
        # aqui atrasa a primeira palavra do professor em segundos (medido:
        # 11,2s contra 0,7s). Default: ela espera o fim da resposta.
        pending = None if self._analysis_after_reply else self._start_analysis(text, level, mode)

        pieces: list[str] = []
        started = time.monotonic()
        first_chunk_ms: float | None = None
        # A análise, quando sai, é também insumo do nível: erro com regra
        # curada é o único sinal negativo da rubrica (plano §5.2).
        analysis: Analysis | None = None

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
                        analysis = event.analysis
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
            pending = self._start_analysis(text, level, mode)

        # O Analyzer tem timeout próprio, então esperar aqui não pendura o
        # turno.
        if pending is not None and not pending.emitted:
            await _settle(pending.task)
            event = self._collect(pending, student_turn_id)
            if event is not None:
                analysis = event.analysis
                yield event

        # O nível vem depois da análise porque depende dela, e antes do Done
        # porque o Done fecha o turno.
        if student is not None:
            leveled = self._observe_level(student, student_turn_id, text, analysis)
            if leveled is not None:
                yield leveled

        yield Done(
            reply=reply,
            student_turn_id=student_turn_id,
            teacher_turn_id=teacher_turn_id,
            first_chunk_ms=first_chunk_ms or 0.0,
            total_ms=(time.monotonic() - started) * 1000,
            analysis_ms=pending.elapsed_ms if pending is not None else 0.0,
        )
