"""Nível CEFR: sinais por turno, estimativa e ajuste.

O perfil do aluno guarda um nível, e o nível muda o prompt da resposta
(``prompts/partials/level/``). Quem decide esse nível é **código**, não o
modelo, e isso não é preferência de estilo:

**Os sinais são determinísticos de propósito.** Pedir ``level_signals`` ao
modelo, como o plano §2 esboçava, não funcionaria neste desenho por dois
motivos medidos na E4. Primeiro, a análise **não chama o modelo** quando a base
contrastiva não levanta candidato (37% dos turnos da suíte) — e turno de inglês
correto é exatamente onde mora a evidência de nível alto. Segundo, uma nota que
o modelo atribui não é depurável: o mesmo turno receberia B1 ou B2 conforme o
humor da amostragem. Aqui, `"I have 30 years"` produz sempre os mesmos números,
e a conta que levou de números a nível está escrita abaixo.

**Um turno nunca muda o nível.** Uma frase curta não é evidência de nada: "ok,
thanks" tem a mesma métrica para um A2 e para um C1. Então um turno só conta
como evidência se tiver corpo (``MIN_WORDS``), e o nível só anda quando
``AGREEING_TURNS`` turnos contados seguidos apontam para o mesmo lado — um
degrau por vez. Sem essa inércia o aluno veria o registro do professor oscilar
a cada mensagem, o que é pior que um nível errado e estável.

**O nível autodeclarado é o ponto de partida, e fica registrado.**
``student.level_source`` diz se o nível em vigor veio do aluno (``self``) ou dos
sinais (``auto``), e ``level_change`` guarda o histórico. Sem isso não há como
responder "por que o Charles começou a falar difícil comigo?".

O ajuste do prompt entra **depois** da persona, como segunda mensagem de
sistema, nunca interpolado nela: o Ollama cacheia o prefixo do prompt e a
persona estável é o que mantém a primeira palavra em 0,7s (plano §1.4).
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..store.models import CEFR_LEVELS, CefrLevel
from ..store.repo import LevelRepo, StudentRepo

if TYPE_CHECKING:  # pragma: no cover - só para tipo
    from .analyzer import Analysis

log = logging.getLogger(__name__)

# --- a rubrica ------------------------------------------------------------- #
#
# Quatro sinais positivos e um negativo, cada um normalizado para 0..1 e
# somado com peso. A escolha de quatro sinais explicáveis em vez de um
# classificador é a mesma da E3 para a base contrastiva: o resultado tem que
# ser depurável à mão, e cada constante aqui foi calibrada contra
# `tests/fixtures/level_samples.yaml` (ver docs/plano-fase-1.md §5.3).

# Comprimento médio de frase que já satura o sinal. Acima de ~24 palavras o
# comprimento não distingue mais nada: tanto um B2 quanto um C1 escrevem
# período longo, e quem decide entre os dois são os outros três sinais.
SENTENCE_SATURATION = 24.0

# Proporção de palavras longas (>= 7 letras) que satura o sinal. Palavra longa
# é proxy de vocabulário latinado — que o brasileiro acerta cedo, daí saturar
# já em 22%.
LONG_WORD_SATURATION = 0.22

# Conectivos avançados e estruturas avançadas por 100 palavras que saturam.
# Baixo de propósito: um conectivo a cada 33 palavras já é texto articulado, e
# exigir mais que isso faria a rubrica premiar quem enfileira "however".
CONNECTIVE_SATURATION = 3.0
STRUCTURE_SATURATION = 4.0

# Erros por 100 palavras que saturam a penalidade. Só entram erros com regra
# curada: é a única contagem de erro em que este projeto confia (plano §5.2).
ERROR_SATURATION = 8.0

# Calibrados por busca em grade sobre `tests/fixtures/level_samples.yaml`, com
# piso de 0,15 por sinal (ver docs/plano-fase-1.md §5.3). A busca sem piso
# chegava a 88% de acerto dando 0,8 do peso a comprimento de frase e palavra
# longa, o que é a regra degenerada "texto mais longo é nível mais alto" — vale
# na amostra e quebra no primeiro C1 que escreva uma linha curta e certeira.
WEIGHT_SENTENCE = 0.35
WEIGHT_LONG_WORDS = 0.35
WEIGHT_CONNECTIVES = 0.15
WEIGHT_STRUCTURES = 0.15
WEIGHT_ERRORS = 0.35

# Faixas de score por nível. O teto de cada faixa é o piso da seguinte.
LEVEL_THRESHOLDS: tuple[tuple[float, CefrLevel], ...] = (
    (0.40, "A2"),
    (0.58, "B1"),
    (0.76, "B2"),
)

# A rubrica não estima C2, e isso é limite declarado, não esquecimento: não há
# sinal em mensagem de chat que separe C1 de C2 — os dois saturam comprimento
# de frase, palavra longa e estrutura. Medido: com um corte de C1→C2 em 0,92,
# três das oito amostras B2 iam para C2, errando por dois degraus. O corte
# saiu.
#
# Consequência, implementada em ``decide``: C2 é nível autodeclarado, e os
# sinais não movem quem está nele — sobre um C2 esta rubrica não tem o que
# dizer, nem para cima nem para baixo.
TOP_ESTIMATED_LEVEL: CefrLevel = "C1"

# --- a política de ajuste -------------------------------------------------- #

# Abaixo disto o turno não é evidência: "ok, thanks" não distingue A2 de C1.
MIN_WORDS = 12

# Quantos turnos contados seguidos têm que apontar para o mesmo lado.
AGREEING_TURNS = 4


# --- léxico ---------------------------------------------------------------- #

_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_SENTENCE_SPLIT = re.compile(r"[.!?…]+|\n+")

LONG_WORD_LETTERS = 7

# Conectivos que marcam coordenação de ideias, não só de frases. "and", "but"
# e "because" ficam fora de propósito: um A2 usa os três.
_CONNECTIVES: tuple[str, ...] = (
    r"although",
    r"even though",
    r"though",
    r"however",
    r"whereas",
    r"nevertheless",
    r"nonetheless",
    r"despite",
    r"in spite of",
    r"therefore",
    r"thus",
    r"moreover",
    r"furthermore",
    r"besides that",
    r"meanwhile",
    r"unless",
    r"whether",
    r"as long as",
    r"as far as",
    r"on the other hand",
    r"in order to",
    r"rather than",
    r"otherwise",
    r"consequently",
    r"as a result",
    r"for instance",
    r"in fact",
    r"to be honest",
    r"that said",
)

# Estruturas que um aluno de nível mais alto produz sem pensar, e que o
# brasileiro de B1 evita: aspecto perfeito, condicional, passiva, relativa.
_STRUCTURES: tuple[str, ...] = (
    # Aspecto perfeito, inclusive contraído: o aluno escreve "I've been
    # working", não "I have been working", e ignorar a contração foi medido
    # como o maior buraco da primeira versão da rubrica.
    r"\b(?:have|has|had|ve|s|d)\s+(?:been|already\s+|just\s+|never\s+)?\w+(?:ed|en)\b",
    r"\b(?:have|has|had|ve|s|d)\s+(?:been|gone|done|made|seen|taken|given|written|got|become)\b",
    r"\b(?:haven|hasn|hadn|wouldn|couldn|shouldn)\s*t\s+\w+",
    # Condicional e modal perfeito.
    r"\b(?:would|could|should|might|must)\s+(?:have|be|rather)\b",
    r"\bd\s+(?:have|rather|say|argue|expect)\b",
    r"\bif\s+(?:i|he|she|it|we|they|you)\s+(?:were|had|would|could)\b",
    r"\bas\s+if\b",
    # Passiva.
    r"\b(?:is|are|was|were|been|being)\s+(?:\w+ly\s+)?\w+(?:ed|en)\s+(?:by|to|as|in|for)\b",
    # Relativa não-essencial e oração reduzida inicial, as duas marcas de quem
    # já organiza período longo sem perder o fio.
    r"\b(?:which|whose|whom)\b",
    r"^\s*(?:having|looking|considering|given|despite|while|although)\b",
    # Fórmulas que o aluno de nível alto usa sem pensar.
    r"\bused\s+to\b",
    r"\bsupposed\s+to\b",
    r"\bend(?:ed|s|ing)?\s+up\b",
    r"\bno\s+matter\b",
    r"\bthe\s+\w+er\s+.*\bthe\s+\w+er\b",
)


def _compile_phrases(phrases: tuple[str, ...]) -> re.Pattern[str]:
    joined = "|".join(f"(?:{p})" for p in phrases)
    return re.compile(rf"\b(?:{joined})\b", re.IGNORECASE)


_CONNECTIVE_RE = _compile_phrases(_CONNECTIVES)
_STRUCTURE_RE = re.compile("|".join(f"(?:{p})" for p in _STRUCTURES), re.IGNORECASE)


def _clip(value: float) -> float:
    return 0.0 if value < 0 else 1.0 if value > 1 else value


# --- sinais ---------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Signals:
    """O que um turno do aluno diz sobre o nível dele, em números.

    Nada aqui depende do modelo: o mesmo texto produz sempre os mesmos valores,
    e cada um é conferível à mão sobre a frase.
    """

    words: int = 0
    sentences: int = 0
    long_words: int = 0
    connectives: int = 0
    structures: int = 0
    errors: int = 0

    @classmethod
    def of(cls, student_text: str, errors: int = 0) -> Signals:
        words = _WORD.findall(student_text)
        pieces = [p for p in _SENTENCE_SPLIT.split(student_text) if _WORD.search(p)]
        return cls(
            words=len(words),
            sentences=max(len(pieces), 1 if words else 0),
            long_words=sum(1 for w in words if len(w.strip("'’-")) >= LONG_WORD_LETTERS),
            connectives=len(_CONNECTIVE_RE.findall(student_text)),
            structures=len(_STRUCTURE_RE.findall(student_text)),
            errors=errors,
        )

    @property
    def mean_sentence_words(self) -> float:
        if not self.sentences:
            return 0.0
        return self.words / self.sentences

    @property
    def long_word_ratio(self) -> float:
        if not self.words:
            return 0.0
        return self.long_words / self.words

    def per_100_words(self, count: int) -> float:
        if not self.words:
            return 0.0
        return 100.0 * count / self.words

    @property
    def counts_as_evidence(self) -> bool:
        """Turno curto não é evidência de nível nenhum."""
        return self.words >= MIN_WORDS


@dataclass(frozen=True, slots=True)
class LevelEstimate:
    """O nível que **este** turno sugere, com a conta que levou até ele."""

    level: CefrLevel
    score: float
    signals: Signals
    counted: bool

    @property
    def summary(self) -> str:
        """Uma linha factual, para log e para o histórico de mudanças."""
        return (
            f"{self.signals.words}w, "
            f"{self.signals.mean_sentence_words:.0f}w/sentence, "
            f"{self.signals.connectives} connectives, "
            f"{self.signals.structures} structures, "
            f"{self.signals.errors} errors -> {self.level} ({self.score:.2f})"
        )


def score_of(signals: Signals) -> float:
    """A rubrica, em uma expressão.

    Positivos somam até 1,0; a penalidade de erro desconta até 0,35. O
    resultado é cortado em 0..1 para a faixa de nível ser comparável entre
    turnos.
    """
    positive = (
        WEIGHT_SENTENCE * _clip(signals.mean_sentence_words / SENTENCE_SATURATION)
        + WEIGHT_LONG_WORDS * _clip(signals.long_word_ratio / LONG_WORD_SATURATION)
        + WEIGHT_CONNECTIVES
        * _clip(signals.per_100_words(signals.connectives) / CONNECTIVE_SATURATION)
        + WEIGHT_STRUCTURES
        * _clip(signals.per_100_words(signals.structures) / STRUCTURE_SATURATION)
    )
    penalty = WEIGHT_ERRORS * _clip(signals.per_100_words(signals.errors) / ERROR_SATURATION)
    return _clip(positive - penalty)


def level_of_score(score: float) -> CefrLevel:
    for ceiling, level in LEVEL_THRESHOLDS:
        if score < ceiling:
            return level
    return TOP_ESTIMATED_LEVEL


def estimate(student_text: str, errors: int = 0) -> LevelEstimate:
    """O nível que este turno sugere. Puro: nada de banco, nada de modelo."""
    signals = Signals.of(student_text, errors=errors)
    score = score_of(signals)
    return LevelEstimate(
        level=level_of_score(score),
        score=score,
        signals=signals,
        counted=signals.counts_as_evidence,
    )


# --- ajuste ---------------------------------------------------------------- #


def index_of(level: str) -> int:
    try:
        return CEFR_LEVELS.index(level)  # type: ignore[arg-type]
    except ValueError as exc:
        raise ValueError(f"nível CEFR inválido: {level!r} (use um de {CEFR_LEVELS})") from exc


def step_toward(current: str, target: str) -> CefrLevel:
    """Um degrau na direção do alvo. Nunca pula nível."""
    here, there = index_of(current), index_of(target)
    if there == here:
        return CEFR_LEVELS[here]
    return CEFR_LEVELS[here + (1 if there > here else -1)]


@dataclass(frozen=True, slots=True)
class LevelDecision:
    """O que se concluiu sobre o nível depois deste turno."""

    level: CefrLevel
    previous: CefrLevel
    estimate: LevelEstimate
    changed: bool = False
    reason: str = ""

    @property
    def direction(self) -> int:
        return index_of(self.level) - index_of(self.previous)

    def as_dict(self) -> dict[str, object]:
        return {
            "level": self.level,
            "previous": self.previous,
            "changed": self.changed,
            "estimated": self.estimate.level,
            "score": round(self.estimate.score, 3),
            "counted": self.estimate.counted,
            "reason": self.reason,
        }


def decide(
    current: str,
    recent: list[CefrLevel],
    agreeing_turns: int = AGREEING_TURNS,
) -> tuple[CefrLevel, str]:
    """O nível depois de ver os ``recent`` turnos contados, do mais novo ao mais velho.

    Move um degrau quando os ``agreeing_turns`` turnos mais recentes **todos**
    apontam para o mesmo lado do nível atual. Um único turno acima não move
    nada, e um turno ruim no meio de uma sequência boa zera a contagem — é
    assim que se evita o nível oscilar a cada mensagem.
    """
    here = index_of(current)
    if here > index_of(TOP_ESTIMATED_LEVEL):
        # Aluno autodeclarado acima do teto da rubrica. Ela não estima C2, então
        # também não rebaixa quem está nele: a sequência de C1 que viria daqui
        # mediria o limite do estimador, não o inglês do aluno.
        return CEFR_LEVELS[here], ""
    if len(recent) < agreeing_turns:
        return CEFR_LEVELS[here], ""

    window = recent[:agreeing_turns]
    if all(index_of(level) > here for level in window):
        target = min(index_of(level) for level in window)
    elif all(index_of(level) < here for level in window):
        target = max(index_of(level) for level in window)
    else:
        return CEFR_LEVELS[here], ""

    moved = step_toward(current, CEFR_LEVELS[target])
    reason = f"{agreeing_turns} turns in a row estimated {'/'.join(dict.fromkeys(window))}"
    return moved, reason


class LevelTracker:
    """Liga os sinais ao perfil: observa o turno, grava e, se for o caso, ajusta.

    Vive em ``teaching/`` porque é regra pedagógica, não persistência: o
    ``store/`` só guarda as linhas.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        agreeing_turns: int = AGREEING_TURNS,
    ) -> None:
        self._signals = LevelRepo(conn)
        self._students = StudentRepo(conn)
        self._agreeing_turns = agreeing_turns

    def observe(
        self,
        student_id: int,
        turn_id: int,
        student_text: str,
        analysis: Analysis | None = None,
    ) -> LevelDecision:
        """Grava os sinais deste turno e devolve o nível em vigor depois dele.

        Conta como erro só a correção com regra curada: é a única contagem em
        que este projeto confia (plano §5.2), e inflar a penalidade com o que o
        modelo achou sozinho rebaixaria o aluno por engano.
        """
        errors = 0
        if analysis is not None:
            errors = sum(1 for f in analysis.findings if f.from_curated_rule)

        guess = estimate(student_text, errors=errors)
        signals = guess.signals
        # O repositório recebe números, não a rubrica: `store/` não conhece
        # `teaching/`, e a dependência corre num sentido só.
        signal_id = self._signals.record(
            turn_id,
            words=signals.words,
            sentences=signals.sentences,
            long_words=signals.long_words,
            connectives=signals.connectives,
            structures=signals.structures,
            errors=signals.errors,
            score=guess.score,
            estimated_level=guess.level,
            counted=guess.counted,
        )

        student = self._students.get(student_id)
        current: CefrLevel = student.cefr_level  # type: ignore[assignment]

        if not guess.counted:
            return LevelDecision(level=current, previous=current, estimate=guess)

        recent = self._signals.recent_levels(student_id, limit=self._agreeing_turns)
        target, reason = decide(current, recent, agreeing_turns=self._agreeing_turns)
        if target == current:
            return LevelDecision(level=current, previous=current, estimate=guess)

        self._students.set_level(student_id, target, source="auto")
        # `at_signal_id` fecha a janela aqui: os turnos que moveram o nível não
        # podem movê-lo outra vez no turno seguinte.
        self._signals.log_change(
            student_id,
            current,
            target,
            source="auto",
            reason=reason,
            at_signal_id=signal_id,
        )
        log.info("nível do aluno %d: %s -> %s (%s)", student_id, current, target, reason)
        return LevelDecision(
            level=target,
            previous=current,
            estimate=guess,
            changed=True,
            reason=reason,
        )


# --- prompt ---------------------------------------------------------------- #


def partial_name(level: str) -> str:
    """O prompt de ajuste deste nível, no formato que a ``PromptLibrary`` espera."""
    return f"partials/level/{level.lower()}"


# Apelido de pacote: ``estimate`` sozinho não diz do que, e `teaching/` exporta
# para quem está longe daqui.
estimate_level = estimate

__all__ = [
    "AGREEING_TURNS",
    "TOP_ESTIMATED_LEVEL",
    "MIN_WORDS",
    "LevelDecision",
    "LevelEstimate",
    "LevelTracker",
    "Signals",
    "decide",
    "estimate",
    "estimate_level",
    "index_of",
    "level_of_score",
    "partial_name",
    "score_of",
    "step_toward",
]
