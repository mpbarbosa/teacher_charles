"""Os três modos, e o drill que vem do erro real do aluno.

Modo não é um adjetivo no prompt. Cada um muda **três** coisas, e só a primeira
é texto:

1. O prompt de comportamento, em ``prompts/modes/``.
2. Quantas correções estruturadas saem por turno. Em conversa, duas é o teto —
   a persona pede uma. Em correção, o aluno pediu os erros, então o teto sobe:
   capar em dois ali seria esconder dele o que ele veio buscar.
3. No drill, **qual regra** vai ser treinada — e isso é escolhido por consulta
   ao error log e à agenda de revisão, não pelo modelo.

## Por que a regra do drill é escolhida em código

O critério de pronto da E6 é "o drill usa erros de fato cometidos, não
genéricos". Pedir ao modelo "crie um exercício a partir dos erros do aluno"
devolveria exercício plausível sobre erro inventado — é o problema nº 1 da
análise inicial outra vez, agora em forma de exercício.

Então o caminho é o mesmo da E4: a aplicação consulta
``correction`` agregada por ``rule_id``, escolhe a regra, busca na base
contrastiva o item curado (explicação, formas certas) e junta as frases que o
**aluno** escreveu. O modelo recebe isso pronto e faz o que modelo faz bem:
conversar. Nada de gramática improvisada, nada de erro que o aluno não cometeu.

## O que a E13 mudou aqui

A E6 ordenava as candidatas por **frequência**: a regra mais corrigida primeiro.
Quem ordena agora é a agenda de ``teaching/schedule.py`` — vencida primeiro —, e
a frequência ficou como último desempate entre regras igualmente urgentes. A
diferença que isso faz é a que a frequência não conseguia fazer: regra acertada
três vezes seguidas sai da fila e volta depois do intervalo, enquanto pela
contagem ela continuaria sendo a mais errada do histórico para sempre.

O filtro continua sendo o desta camada, e não mudou: só entra regra com
``rule_id`` que ainda existe na base curada.

Só entra correção com ``rule_id``: correção sem regra curada veio do modelo
sozinho, e treinar em cima dela seria ensinar o erro com mais repetição (plano
§5.2).

## Uma regra por sessão, não por turno

Pedagogia: drill que troca de regra a cada mensagem é questionário, não drill.
Técnica: o bloco do drill é mensagem de sistema estável dentro da sessão, então
o prefixo do prompt continua cacheado — trocar de regra a cada turno custaria a
reavaliação inteira a cada turno (plano §1.4).

A agenda da E13 não muda isso: ela é consultada **quando a sessão começa**, e as
revisões que acontecem durante a sessão só valem para a escolha seguinte. O
plano da Fase 2 (§4, "Fora") mantém drill com várias regras por sessão fora do
escopo pela mesma razão de cache.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime

from ..store.models import MODES, Mode, RuleCount, RuleSchedule
from ..store.repo import CorrectionRepo
from .contrastive import ContrastiveBase, ContrastiveItem
from .schedule import DRILL_ROTATION, Queued, Scheduler, is_graduated

DEFAULT_MODE: Mode = "conversation"

# Teto de correções estruturadas por turno, por modo.
#
# `conversation` acompanha a persona ("uma correção por turno é o bastante,
# duas é muito"). `correction` sobe porque o aluno pediu: ele abriu o modo para
# receber os erros, e devolver dois de cinco seria esconder o resto. `drill`
# fica baixo de propósito — ali o erro treinado já é o assunto, e anotar outros
# por cima dispersa.
MAX_FINDINGS_BY_MODE: dict[str, int] = {
    "conversation": 2,
    "correction": 5,
    "drill": 2,
}

# Quantas frases do próprio aluno vão no bloco do drill. São as palavras dele:
# reconhecer o próprio erro é o que faz o exercício colar.
MAX_STUDENT_EXAMPLES = 3

# Quantas formas corretas da base entram. A base tem de 2 a 4 por item; três
# bastam para o modelo variar o exercício sem encher o prompt.
MAX_RIGHT_EXAMPLES = 3


def is_mode(value: str) -> bool:
    return value in MODES


def partial_name(mode: str) -> str:
    """O prompt de comportamento deste modo, como a ``PromptLibrary`` espera."""
    return f"modes/{mode.lower()}"


def max_findings(mode: str) -> int:
    return MAX_FINDINGS_BY_MODE.get(mode, MAX_FINDINGS_BY_MODE[DEFAULT_MODE])


@dataclass(frozen=True, slots=True)
class DrillTarget:
    """A regra que está sendo treinada, com o que o aluno errou nela.

    ``item`` é o item curado da base; ``student_examples`` são as frases que o
    aluno escreveu de verdade. A distinção importa: a explicação e as formas
    certas vêm da curadoria revisada, os erros vêm do banco, e **nada** vem da
    imaginação do modelo.

    ``schedule`` é o estado desta regra na agenda (E13), ou ``None`` se ela
    nunca foi revisada. Vai junto porque o aluno tem direito de saber que está
    a um acerto de tirar a regra da fila — "you've got this right twice in a
    row" é afirmação verificável, e é o retorno que faz a repetição espaçada
    deixar de ser invisível para quem estuda.
    """

    item: ContrastiveItem
    times: int = 0
    student_examples: tuple[str, ...] = ()
    schedule: RuleSchedule | None = None

    @property
    def rule_id(self) -> str:
        return self.item.id

    @property
    def streak(self) -> int:
        return self.schedule.streak if self.schedule is not None else 0

    @property
    def graduated(self) -> bool:
        return is_graduated(self.schedule)

    def as_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "times": self.times,
            "explanation": self.item.explanation,
            "examples": list(self.student_examples),
            # A agenda, para a tela poder dizer onde o aluno está nela.
            "streak": self.streak,
            "graduated": self.graduated,
            "due_at": self.schedule.due_at if self.schedule is not None else "",
            "reviews": self.schedule.reviews if self.schedule is not None else 0,
            "lapses": self.schedule.lapses if self.schedule is not None else 0,
        }


def drill_block(target: DrillTarget | None) -> str:
    """O bloco de dados que vai ao modelo no modo drill.

    Dados, não comportamento: o "como treinar" está em ``prompts/modes/drill.md``,
    versionado e editável sem mexer em Python. Aqui vai só o que a aplicação
    apurou — qual regra, quantas vezes, com que palavras, e quais são as formas
    certas segundo a base curada.
    """
    if target is None:
        # Estado legítimo, e dito em voz alta: aluno sem erro registrado não
        # tem o que treinar, e inventar exercício genérico seria exatamente o
        # que o critério da E6 proíbe.
        return (
            "There is no rule to drill yet: this student has no recorded mistakes. "
            "Tell them so, and offer to just talk for a while instead."
        )

    lines = [
        "Drill this one rule, and nothing else.",
        "",
        f"Rule: {target.rule_id}",
        f"What to teach: {target.item.explanation}",
    ]
    if target.times:
        times = "once" if target.times == 1 else f"{target.times} times"
        lines.append(f"The student has made this mistake {times}.")
    if target.streak:
        # O que a agenda sabe e a contagem não sabia: o aluno está melhorando
        # nesta regra. Dizer isso ao modelo muda o tom do exercício, e dizer ao
        # aluno é metade do valor de repetição espaçada.
        right = "once" if target.streak == 1 else f"{target.streak} times"
        lines.append(f"They have got it right {right} in a row since then — say so.")
    if target.student_examples:
        joined = "; ".join(f'"{example}"' for example in target.student_examples)
        lines.append(f"Their own words: {joined}")
    right = target.item.right[:MAX_RIGHT_EXAMPLES]
    if right:
        lines.append("Correct forms: " + "; ".join(f'"{form}"' for form in right))
    return "\n".join(lines)


def drill_reminder(target: DrillTarget | None) -> str:
    """Uma linha no fim do prompt, lembrando qual é a tarefa deste turno.

    Mesma lição da E4 com a dica curada: instrução longe do ponto de geração
    se dilui, e num 4B ela se perde de vez. O bloco de dados fica no topo
    (estável, cacheado); o pedido fica aqui embaixo, logo depois da fala do
    aluno, onde o modelo ainda o está lendo quando começa a responder.

    Medido: sem esta linha, o 3B respondia à fala do aluno e esquecia de
    treinar — ver docs/plano-fase-1.md §5.4.
    """
    if target is None:
        return ""
    return (
        f"Now: react in one line to what the student just wrote, then give them "
        f"ONE short exercise on {target.rule_id} and stop. Wait for their answer."
    )


class DrillPlanner:
    """Escolhe a regra do próximo drill: error log do aluno, ordem da agenda.

    A divisão de trabalho com ``teaching/schedule.py`` é de propósito. O
    ``Scheduler`` sabe **quando** uma regra volta e não sabe nada de base
    contrastiva; o planner sabe **quais** regras podem virar drill (as que têm
    item curado) e monta o alvo com as frases do aluno. Nenhum dos dois decide
    sozinho.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        base: ContrastiveBase,
        rotation: int = DRILL_ROTATION,
    ) -> None:
        self._corrections = CorrectionRepo(conn)
        self._scheduler = Scheduler(conn, rotation=rotation)
        self._base = base
        self._rotation = rotation

    @property
    def scheduler(self) -> Scheduler:
        """A agenda, para quem precisa registrar revisão (o turno) ou contar
        o que está vencido (a tela)."""
        return self._scheduler

    def candidates(self, student_id: int, limit: int = 5) -> list[RuleCount]:
        """As regras do error log que ainda existem na base.

        Regra que saiu da base (renomeada, removida na curadoria) continua no
        histórico do aluno e não pode virar drill: sem item curado não há
        explicação revisada nem forma certa para treinar.
        """
        return [
            count
            for count in self._corrections.top_rules(student_id, limit=limit)
            if count.rule_id in self._base
        ]

    def queue(
        self,
        student_id: int,
        limit: int = 5,
        at: datetime | None = None,
    ) -> list[Queued]:
        """A fila, em ordem de urgência. É a agenda que ordena (E13).

        ``at`` existe para o relógio ser injetável daqui até o ``Scheduler``: a
        fila é feita de datas, e vencimento só se testa com o relógio na mão.
        """
        return self._scheduler.queue(student_id, self.candidates(student_id, limit=limit), at=at)

    def due_count(self, student_id: int, limit: int = 5, at: datetime | None = None) -> int:
        """Quantas regras estão vencidas agora.

        Diferente de ``len(candidates())``: candidata é regra que o aluno errou
        e que tem item curado; vencida é candidata que a agenda liberou. Zero
        vencidas com candidatas de sobra é o aluno em dia, e a tela diz isso.
        """
        return self._scheduler.due_count(
            student_id, self.candidates(student_id, limit=limit), at=at
        )

    def pick(self, student_id: int, at: datetime | None = None) -> DrillTarget | None:
        """A regra a treinar agora, ou ``None`` se não houver erro registrado.

        A ordem é a da agenda — ``Scheduler.queue`` explica cada termo da
        chave. O que esta camada garante é o resto: só regra com item curado
        entra, e ``None`` significa "não há o que treinar", nunca "está tudo
        em dia".

        Nada vencido **não** é ``None``: a fila devolve a de vencimento mais
        próximo e o aluno treina ela. É melhor repetir a regra que não treinar
        nada, que é a mesma escolha que a E6 fez com o rodízio.
        """
        queued = self.queue(student_id, at=at)
        if not queued:
            return None
        chosen = queued[0]
        return self.target_for(student_id, chosen.rule_id, times=chosen.times)

    def target_for(
        self,
        student_id: int,
        rule_id: str,
        times: int = 0,
    ) -> DrillTarget | None:
        """Monta o alvo de uma regra já escolhida (a da sessão em andamento)."""
        if rule_id not in self._base:
            return None
        examples = self._corrections.spans_for_rule(student_id, rule_id, limit=MAX_STUDENT_EXAMPLES)
        return DrillTarget(
            item=self._base.get(rule_id),
            times=times or len(examples),
            student_examples=tuple(examples),
            schedule=self._scheduler.schedule_for(student_id, rule_id),
        )


__all__ = [
    "DEFAULT_MODE",
    "DRILL_ROTATION",
    "MAX_FINDINGS_BY_MODE",
    "DrillPlanner",
    "DrillTarget",
    "drill_block",
    "drill_reminder",
    "is_mode",
    "max_findings",
    "partial_name",
]
