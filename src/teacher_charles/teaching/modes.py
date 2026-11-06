"""Os três modos, e o drill que vem do erro real do aluno.

Modo não é um adjetivo no prompt. Cada um muda **três** coisas, e só a primeira
é texto:

1. O prompt de comportamento, em ``prompts/modes/``.
2. Quantas correções estruturadas saem por turno. Em conversa, duas é o teto —
   a persona pede uma. Em correção, o aluno pediu os erros, então o teto sobe:
   capar em dois ali seria esconder dele o que ele veio buscar.
3. No drill, **qual regra** vai ser treinada — e isso é escolhido por consulta
   ao error log, não pelo modelo.

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

Só entra correção com ``rule_id``: correção sem regra curada veio do modelo
sozinho, e treinar em cima dela seria ensinar o erro com mais repetição (plano
§5.2).

## Uma regra por sessão, não por turno

Pedagogia: drill que troca de regra a cada mensagem é questionário, não drill.
Técnica: o bloco do drill é mensagem de sistema estável dentro da sessão, então
o prefixo do prompt continua cacheado — trocar de regra a cada turno custaria a
reavaliação inteira a cada turno (plano §1.4).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from ..store.models import MODES, Mode, RuleCount
from ..store.repo import CorrectionRepo, SessionRepo
from .contrastive import ContrastiveBase, ContrastiveItem

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

# Quantas regras recém-treinadas o rodízio evita. Com 3, o aluno não recebe a
# mesma regra em sessões seguidas enquanto houver outra no error log.
DRILL_ROTATION = 3


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
    """

    item: ContrastiveItem
    times: int = 0
    student_examples: tuple[str, ...] = ()

    @property
    def rule_id(self) -> str:
        return self.item.id

    def as_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "times": self.times,
            "explanation": self.item.explanation,
            "examples": list(self.student_examples),
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
    """Escolhe a regra do próximo drill a partir do error log do aluno."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        base: ContrastiveBase,
        rotation: int = DRILL_ROTATION,
    ) -> None:
        self._corrections = CorrectionRepo(conn)
        self._sessions = SessionRepo(conn)
        self._base = base
        self._rotation = rotation

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

    def pick(self, student_id: int) -> DrillTarget | None:
        """A regra a treinar agora, ou ``None`` se não houver erro registrado.

        Mais frequente primeiro, evitando as últimas já treinadas. Se todas as
        candidatas foram treinadas há pouco, volta para a mais frequente: o
        rodízio é preferência, não impedimento — é melhor repetir a regra do
        que não treinar nada.
        """
        counts = self.candidates(student_id)
        if not counts:
            return None

        recent = set(self._sessions.drilled_rules(student_id, limit=self._rotation))
        fresh = [count for count in counts if count.rule_id not in recent]
        chosen = (fresh or counts)[0]
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
