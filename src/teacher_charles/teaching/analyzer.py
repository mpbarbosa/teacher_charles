"""A fala do aluno vira erro estruturado.

É aqui que o princípio do projeto se paga: **correção é dado, não prosa**. A
resposta do Charles continua sendo conversa, mas em paralelo cada turno produz
linhas contáveis — ``span``, ``correction``, ``rule_id``, ``explanation`` — que
sustentam o error log (E7), o drill a partir do erro real (E6) e a repetição
espaçada da Fase 2.

Três coisas não são confiadas ao prompt, porque prompt não é garantia:

**A regra citada é verificada, não pedida.** Quando a base contrastiva casa com
o trecho que o modelo apontou, o ``rule_id`` e a explicação passam a ser os da
regra curada — o modelo não tem voto. E ``rule_id`` que não existe na base é
descartado: inventar id seria pior que não citar nenhum, porque o error log
agregaria por uma regra fantasma. Esse é o problema nº 1 da análise inicial
tratado em código.

**Span que o aluno não disse é alucinação.** Toda correção tem que apontar para
um trecho literal da fala do aluno. O que não casa é descartado em silêncio: uma
correção sobre palavra que o aluno não escreveu desorienta mais que a falta de
correção.

**JSON inválido degrada, não quebra.** Um 4B erra o formato. Há uma retentativa
com instrução mais estrita e, se ela também falhar, o turno segue sem correção —
nunca com exceção na cara do aluno. Quantas vezes isso acontece é medido, não
presumido (plano §5.2).

O analisador tem provider próprio de propósito: a análise é trabalho de JSON
curto e determinístico, e nada impede que amanhã ela rode num modelo menor que o
da conversa.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..llm.base import ChatProvider, GenerationOptions, Message, ProviderError
from ..prompts import PromptLibrary
from .contrastive import ContrastiveBase, Match, normalize

log = logging.getLogger(__name__)

# Tipos de erro que o error log agrega. Os quatro primeiros espelham as
# categorias da base contrastiva; os outros existem para o que a base ainda não
# cobre. O que vier fora desta lista vira "other" em vez de derrubar o achado:
# perder a correção inteira por causa de um rótulo é caro demais.
ERROR_TYPES: tuple[str, ...] = (
    "false-friend",
    "calque",
    "grammar",
    "pronunciation",
    "vocabulary",
    "spelling",
    "other",
)

# A persona pede uma correção por turno, duas no máximo. O corte vale também
# aqui: o que não cabe na conversa não deveria virar linha no banco.
#
# É o default. A partir da E6 o modo manda: no modo de correção o aluno pediu
# os erros, e capar em dois ali seria esconder dele o que ele veio buscar (ver
# ``teaching/modes.py::MAX_FINDINGS_BY_MODE``).
MAX_FINDINGS = 2

# Correção sem regra curada por trás não chega ao aluno.
#
# Isto não é cautela de projeto, é resultado de medição (plano §5.2). Num 3B,
# sobre 18 frases corretas, a análise inventou 12 correções — e **nenhuma**
# citava regra. Várias ensinavam erro: "There have been" -> "has been",
# "depends on" -> "depend on", "because I was tired" -> "because I was sleepy".
# Nas 24 frases com erro que ela acertou, **todas** citavam a regra esperada.
#
# Ou seja: exigir regra curada eliminou 100% dos falsos positivos e não custou
# nenhum acerto. O preço é não corrigir o que a base ainda não cobre — preço
# que se paga curando a base, não soltando o modelo. Um modelo maior pode
# merecer confiança sem regra; daí isto ser configurável
# (``TC_REQUIRE_CURATED_RULE=false``) e não uma constante.
REQUIRE_CURATED_RULE = True

# Teto de tokens da resposta da análise. JSON de duas correções cabe em ~120;
# o teto existe para um modelo que comece a divagar não custar o turno inteiro.
MAX_ANALYSIS_TOKENS = 320

# A análise roda em paralelo com a resposta, mas não pode ficar pendurada para
# sempre: passado o prazo, o turno fecha sem correção.
DEFAULT_TIMEOUT_S = 45.0

# Quantos candidatos da base vão no prompt. Cada um é prompt a mais, e nesta
# máquina avaliar prompt custa tempo real (plano §1.3).
MAX_CANDIDATES = 4


# --- o que sai daqui ------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Finding:
    """Uma correção, pronta para virar linha em ``correction``."""

    span: str
    error_type: str
    correction: str
    explanation: str
    rule_id: str | None = None

    @property
    def from_curated_rule(self) -> bool:
        """A explicação veio da base curada, não da improvisação do modelo."""
        return self.rule_id is not None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "span": self.span,
            "type": self.error_type,
            "correction": self.correction,
            "rule_id": self.rule_id,
            "explanation": self.explanation,
        }


@dataclass(frozen=True, slots=True)
class Analysis:
    """O resultado de uma análise, com o que ela custou."""

    findings: tuple[Finding, ...] = ()
    # Candidatos que a base levantou, independentemente do que o modelo achou.
    # Guardados porque a diferença entre os dois é a medição da §5.2.
    candidates: tuple[str, ...] = ()
    degraded: bool = False
    detail: str = ""
    attempts: int = 0
    elapsed_ms: float = 0.0
    # Quantas correções o modelo propôs sem regra curada e foram descartadas.
    # Contado, e não só jogado fora, porque é a medida de quanto a base ainda
    # não cobre — a lista de curadoria da próxima rodada.
    unverified: int = 0
    # E a lista em si, que por um bom tempo era só o contador acima com o
    # texto indo para `log.debug`. Cada item aqui é um erro no texto real de um
    # aluno que `data/contrastive/` não cobre: o span vira `wrong:`, a proposta
    # do modelo vira candidato a `right:`. O `TurnRunner` persiste em
    # `unverified_finding` e `scripts/curation_queue.py` lê agregado.
    #
    # Fica FORA de `as_dict`, e isso é a garantia e não um esquecimento:
    # `as_dict` alimenta o SSE, e estes achados são exatamente a gramática
    # improvisada que o verificador acabou de barrar. Material de curadoria
    # nunca é material de aula.
    rejected: tuple[Finding, ...] = ()
    # A análise não aconteceu porque não podia dar em nada: a base não levantou
    # candidato e a política exige regra curada.
    skipped: bool = False

    def __bool__(self) -> bool:
        return bool(self.findings)

    @property
    def rule_ids(self) -> tuple[str, ...]:
        return tuple(f.rule_id for f in self.findings if f.rule_id)

    def as_dict(self) -> dict[str, object]:
        return {
            "errors": [f.as_dict() for f in self.findings],
            "candidates": list(self.candidates),
            "degraded": self.degraded,
            "unverified": self.unverified,
            "skipped": self.skipped,
            "analysis_ms": round(self.elapsed_ms),
        }


# --- o JSON que o modelo devolve ------------------------------------------- #


class _RawFinding(BaseModel):
    """Um achado como o modelo mandou, antes de qualquer verificação.

    ``extra="ignore"`` deixa passar campo a mais sem perder o achado — e é o
    que permite o prompt da E5 pedir ``level_signals`` sem mexer aqui.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    span: str = Field(min_length=1)
    error_type: str = Field(default="other", alias="type")
    correction: str = ""
    rule_id: str | None = None
    explanation: str = ""


class _RawAnalysis(BaseModel):
    model_config = ConfigDict(extra="ignore")

    errors: list[object] = Field(default_factory=list)


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?", re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, object] | None:
    """O primeiro objeto JSON completo de uma resposta, ou ``None``.

    Um 4B entrega JSON cercado de cerca de código, precedido de "Here is the
    JSON:" ou embrulhado em ``<think>``. Nada disso é erro de formato que
    justifique perder a correção, então é tudo removido antes de parsear. O que
    não dá para recuperar devolve ``None`` e vira retentativa.
    """
    if not text:
        return None

    cleaned = _FENCE.sub("", _THINK_BLOCK.sub("", text)).strip()

    start = cleaned.find("{")
    if start == -1:
        return None

    # Varredura com profundidade, respeitando string e escape: cortar no último
    # "}" do texto pegaria lixo depois do objeto, e no primeiro cortaria o
    # objeto no meio.
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(cleaned[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


# --- verificação ----------------------------------------------------------- #


def _locate_span(span: str, student_text: str) -> tuple[str, int, int] | None:
    """O trecho como o aluno escreveu e onde ele está, ou ``None``.

    Tolera diferença de caixa e de espaçamento — o modelo recopia o trecho e
    erra nisso o tempo todo — mas não tolera palavra inventada.

    Devolve a posição junto com o texto, e não só o texto, porque é ela que
    decide qual regra da base casou *neste* trecho. Procurar a posição depois,
    com um ``find``, acharia a primeira ocorrência, que numa frase como "I have
    30 years and she has 25 years" pode não ser a que o modelo apontou.
    """
    wanted = span.strip().strip('"').strip()
    if not wanted:
        return None

    haystack = normalize(student_text)
    needle = normalize(wanted)

    def at(start: int, end: int) -> tuple[str, int, int]:
        return student_text[start:end], start, end

    index = haystack.find(needle)
    if index != -1:
        return at(index, index + len(needle))

    index = haystack.lower().find(needle.lower())
    if index != -1:
        return at(index, index + len(needle))

    # Última tentativa: espaçamento diferente ("30  years" vs "30 years").
    flexible = re.compile(r"\s+".join(re.escape(part) for part in needle.split()), re.IGNORECASE)
    found = flexible.search(haystack)
    if found is not None:
        return at(found.start(), found.end())

    return None


_TRAILING = " \t\n.,;:!?\"'"


def _same_text(left: str, right: str) -> bool:
    """Mesmo texto a menos de caixa, espaçamento e pontuação de borda."""

    def flatten(value: str) -> str:
        return " ".join(normalize(value).lower().strip(_TRAILING).split())

    return flatten(left) == flatten(right)


def _overlaps(match: Match, start: int, end: int) -> bool:
    return match.start < end and start < match.end


@dataclass(slots=True)
class _Verifier:
    """Transforma achados crus em ``Finding``, aplicando as garantias."""

    base: ContrastiveBase
    student_text: str
    matches: Sequence[Match] = field(default_factory=tuple)
    require_curated_rule: bool = REQUIRE_CURATED_RULE
    max_findings: int = MAX_FINDINGS
    # Os achados barrados por não citar regra curada, na ordem em que o modelo
    # os propôs. Era um contador; virou lista quando a fila de curadoria
    # (`unverified_finding`) passou a precisar do span, e não só do quanto.
    rejected: list[Finding] = field(default_factory=list)

    def verify(self, raw_entries: Sequence[object]) -> list[Finding]:
        findings: list[Finding] = []
        seen: set[str] = set()

        for entry in raw_entries:
            if not isinstance(entry, dict):
                continue
            try:
                raw = _RawFinding.model_validate(entry)
            except ValidationError:
                # Um achado torto não invalida os outros.
                continue

            finding = self._verify_one(raw)
            if finding is None:
                continue
            key = finding.span.lower()
            if key in seen:
                continue
            seen.add(key)

            if self.require_curated_rule and not finding.from_curated_rule:
                # Medido: aqui é onde o modelo pequeno inventa gramática.
                #
                # Guardado, e não só contado: é um erro de aluno real que a
                # base não cobre, e a fila de curadoria é feita disto. O
                # descarte em si não mudou — nada daqui chega ao aluno.
                self.rejected.append(finding)
                log.debug(
                    "correção sem regra curada descartada: %r => %r",
                    finding.span,
                    finding.correction,
                )
                continue

            findings.append(finding)

        # Regra curada primeiro: se só uma correção couber, que seja a que tem
        # regra por trás.
        findings.sort(key=lambda f: not f.from_curated_rule)
        return findings[: self.max_findings]

    def _verify_one(self, raw: _RawFinding) -> Finding | None:
        located = _locate_span(raw.span, self.student_text)
        if located is None:
            log.debug("achado descartado: span %r não está na fala do aluno", raw.span)
            return None

        span, start, end = located
        rule = self._rule_for(raw.rule_id, start, end)

        if rule is not None:
            # A regra curada vence o modelo: explicação e tipo vêm dela. É o
            # anteparo contra gramática inventada.
            item = self.base.get(rule)
            explanation = item.explanation
            error_type = item.category
        else:
            explanation = raw.explanation.strip()
            error_type = raw.error_type.strip().lower()
            if error_type not in ERROR_TYPES:
                error_type = "other"

        correction = raw.correction.strip().strip('"').strip()
        if not explanation or not correction:
            # Correção sem o que corrigir, ou sem por quê, não ajuda ninguém.
            return None

        if _same_text(correction, span):
            # Medido: o 3B devolve "correção" idêntica ao trecho do aluno —
            # 'a question about the exercise?' => 'a question about the
            # exercise?'. Mostrar isso ao aluno é pior que não corrigir: ele
            # procura a diferença que não existe.
            log.debug("achado descartado: a correção é igual ao original (%r)", span)
            return None

        return Finding(
            span=span,
            error_type=error_type,
            correction=correction,
            explanation=explanation,
            rule_id=rule,
        )

    def _rule_for(self, claimed: str | None, start: int, end: int) -> str | None:
        """Qual regra citar, decidido aqui e não pelo modelo.

        Vale a regra que a base casou **no mesmo trecho**: é determinística e
        verificável. Um id que o modelo citou sem a base ter casado ali é
        aceito só se existir na base; o que não existe é descartado.
        """
        for match in self.matches:
            if _overlaps(match, start, end):
                return match.rule_id

        if claimed and claimed in self.base:
            return claimed
        if claimed:
            log.debug("rule_id inventado pelo modelo, descartado: %r", claimed)
        return None


# --- o analisador ---------------------------------------------------------- #


class Analyzer:
    def __init__(
        self,
        provider: ChatProvider,
        prompts: PromptLibrary,
        base: ContrastiveBase,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_candidates: int = MAX_CANDIDATES,
        require_curated_rule: bool = REQUIRE_CURATED_RULE,
    ) -> None:
        self._provider = provider
        self._prompts = prompts
        self._base = base
        self._timeout_s = timeout_s
        self._max_candidates = max_candidates
        self._require_curated_rule = require_curated_rule

    @property
    def base(self) -> ContrastiveBase:
        """A base contrastiva em uso.

        Exposta para o ``TurnRunner`` montar a dica da resposta com a mesma
        base da análise: duas referências poderiam divergir, e a dica citaria
        regra que a análise não conhece.
        """
        return self._base

    # --- prompt ---

    def candidates_block(self, student_text: str) -> str:
        """As regras que a base levantou, formatadas para o prompt.

        Curto de propósito: id e uma frase. Cada candidato é prompt a mais, e
        prompt não cacheado custa segundos nesta máquina.
        """
        items = self._base.candidates(student_text, limit=self._max_candidates)
        if not items:
            return ""
        lines = [f"- {item.id}: {item.explanation}" for item in items]
        return "Rules that may apply:\n" + "\n".join(lines)

    def build_messages(
        self,
        student_text: str,
        cefr_level: str = "B1",
        stricter: bool = False,
    ) -> list[Message]:
        parts = [f'Student (CEFR {cefr_level}) said:\n"""\n{student_text}\n"""']
        block = self.candidates_block(student_text)
        if block:
            parts.append(block)
        if stricter:
            # A retentativa não reescreve a instrução: acrescenta a correção do
            # que deu errado. Prompt diferente jogaria fora o cache de prefixo.
            parts.append(
                "Your last answer was not valid JSON. Answer with the JSON object only, "
                'starting with {"errors": and nothing else.'
            )

        return [
            Message(role="system", content=self._prompts.get("analysis")),
            Message(role="user", content="\n\n".join(parts)),
        ]

    # --- execução ---

    async def analyze(
        self,
        student_text: str,
        cefr_level: str = "B1",
        max_findings: int | None = None,
    ) -> Analysis:
        """Analisa um turno. Nunca levanta exceção por causa do modelo."""
        text = student_text.strip()
        matches = self._base.match(text)
        candidates = tuple(m.rule_id for m in matches)

        if not text:
            return Analysis(candidates=candidates)

        if self._require_curated_rule and not matches:
            # Sem candidato da base e com a política exigindo regra, nada que o
            # modelo devolvesse poderia chegar ao aluno. Então não se pergunta.
            #
            # Não é só economia de CPU: medido (plano §5.2), cada chamada de
            # análise joga o prompt da conversa fora do cache de prefixo do
            # Ollama, e a primeira palavra do turno seguinte custa segundos a
            # mais. Turno de inglês correto passa a não pagar nada disso.
            return Analysis(candidates=candidates, skipped=True)

        started = time.monotonic()
        options = GenerationOptions(
            # Análise não é criação: temperatura 0 para o mesmo erro receber
            # sempre a mesma correção, e para o JSON sair estável.
            temperature=0.0,
            max_tokens=MAX_ANALYSIS_TOKENS,
            think=False,
        )
        verifier = _Verifier(
            base=self._base,
            student_text=text,
            matches=matches,
            require_curated_rule=self._require_curated_rule,
            max_findings=max_findings or MAX_FINDINGS,
        )

        attempts = 0
        detail = ""
        for stricter in (False, True):
            attempts += 1
            messages = self.build_messages(text, cefr_level, stricter=stricter)
            try:
                raw_reply = await asyncio.wait_for(
                    self._provider.complete(messages, options),
                    timeout=self._timeout_s,
                )
            except TimeoutError:
                detail = f"a análise passou de {self._timeout_s:.0f}s"
                break
            except ProviderError as exc:
                detail = f"o modelo falhou na análise: {exc}"
                break

            document = extract_json_object(raw_reply)
            if document is None:
                detail = "o modelo não devolveu JSON válido"
                continue

            try:
                parsed = _RawAnalysis.model_validate(document)
            except ValidationError as exc:
                detail = f"JSON fora do contrato da análise: {exc}"
                continue

            findings = verifier.verify(parsed.errors)
            return Analysis(
                findings=tuple(findings),
                candidates=candidates,
                attempts=attempts,
                elapsed_ms=(time.monotonic() - started) * 1000,
                unverified=len(verifier.rejected),
                rejected=tuple(verifier.rejected),
            )

        # Degradar para "sem correção" é a escolha deliberada: o turno do aluno
        # vale mais que a correção dele. O motivo fica no log e no evento, para
        # a frequência disso ser medida e não adivinhada.
        log.warning("análise degradada (%d tentativa(s)): %s", attempts, detail)
        return Analysis(
            candidates=candidates,
            degraded=True,
            detail=detail,
            attempts=attempts,
            elapsed_ms=(time.monotonic() - started) * 1000,
        )


__all__ = [
    "ERROR_TYPES",
    "REQUIRE_CURATED_RULE",
    "MAX_FINDINGS",
    "Analysis",
    "Analyzer",
    "Finding",
    "extract_json_object",
]
