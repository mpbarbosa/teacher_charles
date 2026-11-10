"""A base contrastiva: os erros que o português produz no inglês do aluno.

É o anteparo contra o problema nº 1 da análise inicial — o modelo pequeno
inventar regra gramatical. Onde existe item curado aqui, a explicação que chega
ao aluno vem *daqui*, com ``rule_id``, e não da improvisação do modelo. Por isso
``data/contrastive/`` é versionado e tratado como código: schema, revisão e um
teste por item.

Três decisões moram neste módulo:

**Casamento determinístico, não vetorial.** ~40 itens (e depois ~100) cabem
inteiros na memória, e ``"I have 30 years"`` tem que devolver ``calque.age-have``
*sempre*, não "quase sempre". Embeddings acrescentam dependência, modelo extra e
resultado difuso para um problema que ainda não existe (plano §3, "Consulta à
base contrastiva").

**Os padrões são escritos à mão, os exemplos errados são o teste deles.** Quando
um item traz ``patterns``, só eles casam — os exemplos de ``wrong`` não entram
como literais. É o que mantém o teste por item com dente: se o padrão de
``calque.age-have`` não generalizar de *30* para *25*, o teste quebra. Sem
``patterns``, os literais de ``wrong`` viram o padrão, que é o bastante para
falso cognato de forma fixa.

**Um casamento é suspeita, não veredito.** Vários itens casam com inglês
legítimo (*I pretend to be a doctor* é correto; *I pretend to travel* é decalque
de *pretendo*). Esses vêm marcados com ``ambiguous: true``: o matcher os oferece
como candidatos e quem decide é a análise da E4, com o modelo. O que o matcher
garante é que o candidato certo *está* na lista.

As anotações de offset (``start``/``end``) apontam para o texto original do
aluno, não para uma versão normalizada: a correção mostra ao aluno o trecho como
ele escreveu. É por isso que a normalização aqui é caractere-a-caractere
(apóstrofo curvo, espaço duro) e o resto fica a cargo de ``re.IGNORECASE`` — ela
preserva o comprimento, então os índices continuam valendo no original.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..config import get_settings
from ..store.models import CEFR_LEVELS, CefrLevel

Category = Literal["false-friend", "calque", "grammar", "pronunciation"]

CATEGORIES: tuple[Category, ...] = ("false-friend", "calque", "grammar", "pronunciation")

# ``calque.age-have``: a categoria é parte do id de propósito. O ``rule_id`` vai
# para o banco (``correction.rule_id``) e aparece no error log da E7; lá, saber
# que o erro é decalque e não falso cognato sem precisar consultar a base é o
# que faz a agregação por tipo sair de graça.
RULE_ID_PATTERN = r"^(?:false-friend|calque|grammar|pronunciation)\.[a-z0-9]+(?:-[a-z0-9]+)*$"

# Mapeamento 1 caractere -> 1 caractere. Preservar o comprimento não é detalhe:
# é o que mantém os offsets do casamento válidos no texto original do aluno.
_NORMALIZE_MAP = str.maketrans(
    {
        "’": "'",  # ’ (o que o iOS e o Word produzem)
        "‘": "'",  # ‘
        "ʼ": "'",  # ʼ
        "“": '"',  # “
        "”": '"',  # ”
        " ": " ",  # espaço duro
        "–": "-",  # – en dash
        "—": "-",  # — em dash
    }
)


def normalize(text: str) -> str:
    """Unifica variantes tipográficas sem mudar o comprimento do texto."""
    return text.translate(_NORMALIZE_MAP)


def span_key(text: str) -> str:
    """A chave de agregação de um trecho do aluno.

    Usada pela fila de curadoria (``store.UnverifiedFindingRepo``) para somar
    o mesmo buraco vindo de turnos diferentes. Mora aqui, e não no
    repositório, porque tem de ser a MESMA normalização que o matcher aplica:
    a chave é o texto contra o qual o padrão novo vai ser escrito, e duas
    normalizações divergentes agrupariam errado justamente o caso que
    ``normalize`` existe para unificar — apóstrofo curvo, espaço duro.

    Ao contrário de ``normalize``, aqui o comprimento pode mudar: ninguém
    aponta offset para esta string, então colapsar espaço é de graça.
    """
    return " ".join(normalize(text).lower().split())


class ContrastiveError(ValueError):
    """A base não carregou. Traz todos os problemas, não só o primeiro."""


class ContrastiveItem(BaseModel):
    """Um item curado da base.

    ``extra="forbid"`` é deliberado: um campo com nome errado num YAML curado à
    mão é erro de curadoria, e falhar na carga é melhor que ignorar em silêncio
    um padrão que o curador achava que estava valendo.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=RULE_ID_PATTERN)
    category: Category

    # O português que causa o erro. Obrigatório: uma base contrastiva sem o lado
    # de origem é só uma lista de erros de inglês.
    pt: str = Field(min_length=1)

    # Exemplos do que o aluno diz, e as formas corretas. Os dois lados são
    # exercitados pelos testes: todo ``wrong`` tem que casar com o item, e
    # nenhum ``right`` pode casar (exceto em item ``ambiguous``).
    wrong: list[str] = Field(min_length=1)
    right: list[str] = Field(min_length=1)

    # Regex sobre o texto normalizado, com IGNORECASE. Vazio = derivar dos
    # literais de ``wrong``.
    patterns: list[str] = Field(default_factory=list)

    # Vai para o aluno; em inglês, decisão 1.1 do plano.
    explanation: str = Field(min_length=1)

    # Em que nível o item costuma aparecer. A E5 usa para calibrar o quanto
    # explicar.
    cefr: CefrLevel = "B1"

    tags: list[str] = Field(default_factory=list)

    # ``true`` quando o padrão também casa com inglês correto. O matcher oferece
    # o item como candidato; o veredito é da análise (E4).
    ambiguous: bool = False

    # Nota de curadoria, para quem edita o YAML. Nunca chega ao aluno, então
    # pode estar em português.
    notes: str | None = None

    @model_validator(mode="after")
    def _id_prefix_matches_category(self) -> ContrastiveItem:
        prefix = self.id.split(".", 1)[0]
        if prefix != self.category:
            raise ValueError(
                f"o id {self.id!r} começa com {prefix!r} mas a categoria é {self.category!r}"
            )
        return self


def _literal_pattern(example: str) -> str:
    """Um exemplo errado vira regex: espaços flexíveis, bordas de palavra."""
    words = example.split()
    if not words:
        raise ValueError("exemplo vazio")
    body = r"\s+".join(re.escape(word) for word in words)
    prefix = r"\b" if example[:1].isalnum() else ""
    suffix = r"\b" if example[-1:].isalnum() else ""
    return f"{prefix}{body}{suffix}"


def compile_patterns(item: ContrastiveItem) -> tuple[re.Pattern[str], ...]:
    sources = item.patterns or [_literal_pattern(example) for example in item.wrong]
    return tuple(re.compile(source, re.IGNORECASE) for source in sources)


@dataclass(frozen=True, slots=True)
class Match:
    """Uma suspeita levantada pela base, com o trecho original do aluno."""

    rule_id: str
    item: ContrastiveItem
    span: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _Compiled:
    item: ContrastiveItem
    regexes: tuple[re.Pattern[str], ...]


class ContrastiveBase:
    """A base carregada e compilada, consultável por texto."""

    def __init__(self, items: Sequence[ContrastiveItem]) -> None:
        self._items = tuple(items)
        self._by_id = {item.id: item for item in self._items}
        if len(self._by_id) != len(self._items):
            raise ContrastiveError("há ids repetidos na base")
        self._compiled = tuple(_Compiled(item, compile_patterns(item)) for item in self._items)

    # --- leitura ---------------------------------------------------------

    @property
    def items(self) -> tuple[ContrastiveItem, ...]:
        return self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[ContrastiveItem]:
        return iter(self._items)

    def __contains__(self, rule_id: object) -> bool:
        return rule_id in self._by_id

    def get(self, rule_id: str) -> ContrastiveItem:
        try:
            return self._by_id[rule_id]
        except KeyError as exc:
            raise LookupError(f"nenhum item com id {rule_id!r} na base contrastiva") from exc

    def by_category(self, category: str) -> tuple[ContrastiveItem, ...]:
        return tuple(item for item in self._items if item.category == category)

    # --- consulta --------------------------------------------------------

    def match(self, text: str) -> list[Match]:
        """Os itens que a fala do aluno levanta, em ordem de aparição.

        No máximo um casamento por item: para um turno, saber *que* o aluno
        cometeu o erro basta — repeti-lo três vezes na mesma frase não são três
        regras diferentes.
        """
        haystack = normalize(text)
        found: list[Match] = []

        for compiled in self._compiled:
            hit = None
            for regex in compiled.regexes:
                candidate = regex.search(haystack)
                if candidate is not None and (hit is None or candidate.start() < hit.start()):
                    hit = candidate
            if hit is None:
                continue
            found.append(
                Match(
                    rule_id=compiled.item.id,
                    item=compiled.item,
                    # O trecho sai do texto ORIGINAL: é o que o aluno escreveu,
                    # e é isso que a correção mostra de volta para ele.
                    span=text[hit.start() : hit.end()],
                    start=hit.start(),
                    end=hit.end(),
                )
            )

        found.sort(key=lambda m: (m.start, m.rule_id))
        return found

    def rule_ids(self, text: str) -> tuple[str, ...]:
        """Só os ids, para quando o chamador não precisa dos offsets."""
        return tuple(m.rule_id for m in self.match(text))

    def candidates(self, text: str, limit: int = 5) -> tuple[ContrastiveItem, ...]:
        """Os itens a oferecer à análise da E4, no prompt.

        Tem limite porque cada candidato é prompt a mais, e nesta máquina
        avaliar prompt custa tempo real (plano §1.3). Itens não ambíguos vêm
        primeiro: são os que a análise tem mais chance de confirmar.
        """
        ordered = sorted(self.match(text), key=lambda m: (m.item.ambiguous, m.start))
        return tuple(m.item for m in ordered[:limit])


# --- carga ----------------------------------------------------------------- #


def _items_from_document(document: Any, source: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Extrai a lista ``items`` de um YAML, ou descreve por que não deu."""
    if document is None:
        return [], [f"{source}: arquivo vazio"]
    if not isinstance(document, dict):
        return [], [f"{source}: o topo do arquivo tem que ser um mapeamento com 'items'"]

    raw = document.get("items")
    if raw is None:
        return [], [f"{source}: falta a chave 'items'"]
    if not isinstance(raw, list):
        return [], [f"{source}: 'items' tem que ser uma lista"]

    problems: list[str] = []
    entries: list[dict[str, Any]] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            problems.append(f"{source}[{index}]: item tem que ser um mapeamento")
            continue
        entries.append(entry)
    return entries, problems


def load_items(directory: Path) -> list[ContrastiveItem]:
    """Valida e devolve todos os itens de ``directory``, em ordem de arquivo.

    Acumula os problemas de todos os arquivos antes de falhar: quem está curando
    quer a lista inteira do que corrigir, não uma mensagem por rodada.
    """
    if not directory.is_dir():
        raise ContrastiveError(f"a base contrastiva não existe em {directory}")

    files = sorted(directory.glob("*.yaml"))
    if not files:
        raise ContrastiveError(f"nenhum arquivo .yaml em {directory}")

    items: list[ContrastiveItem] = []
    problems: list[str] = []
    seen: dict[str, str] = {}

    for path in files:
        source = path.name
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            problems.append(f"{source}: YAML inválido — {exc}")
            continue

        entries, file_problems = _items_from_document(document, source)
        problems.extend(file_problems)

        for index, entry in enumerate(entries):
            label = f"{source}[{index}]"
            try:
                item = ContrastiveItem.model_validate(entry)
            except Exception as exc:  # pydantic.ValidationError e ValueError
                identifier = entry.get("id", "<sem id>")
                problems.append(f"{label} ({identifier}): {exc}")
                continue

            if item.id in seen:
                problems.append(f"{label} ({item.id}): id repetido, já definido em {seen[item.id]}")
                continue
            seen[item.id] = label

            try:
                compile_patterns(item)
            except re.error as exc:
                problems.append(f"{label} ({item.id}): regex inválida — {exc}")
                continue

            items.append(item)

    if problems:
        raise ContrastiveError(
            f"{len(problems)} problema(s) na base contrastiva em {directory}:\n  - "
            + "\n  - ".join(problems)
        )

    return items


def load_base(directory: Path | None = None) -> ContrastiveBase:
    """Carrega a base de ``directory`` (default: ``data/contrastive/``)."""
    if directory is None:
        directory = get_settings().data_dir / "contrastive"
    return ContrastiveBase(load_items(directory))


@lru_cache(maxsize=4)
def get_base(directory: Path | None = None) -> ContrastiveBase:
    """A base do processo, carregada uma vez.

    Diferente do ``PromptLibrary``, aqui não há invalidação por mtime: editar a
    base é mexer em conteúdo validado por teste, não ajustar o tom de um prompt,
    e vale reiniciar o servidor.
    """
    return load_base(directory)


# --- schema ---------------------------------------------------------------- #

SCHEMA_RELATIVE_PATH = Path("schema") / "contrastive-item.schema.json"


def json_schema() -> dict[str, Any]:
    """O JSON Schema de um item, derivado do modelo pydantic.

    O modelo é a fonte única: o arquivo em ``data/schema/`` é gerado a partir
    dele (``scripts/dump_contrastive_schema.py``) e existe para editor e
    validador externos. Um teste garante que os dois não divergem.
    """
    schema = ContrastiveItem.model_json_schema()
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://teacher-charles.local/schema/contrastive-item.schema.json",
        **schema,
    }


def schema_json_text() -> str:
    return json.dumps(json_schema(), indent=2, ensure_ascii=False) + "\n"


__all__ = [
    "CATEGORIES",
    "CEFR_LEVELS",
    "Category",
    "ContrastiveBase",
    "ContrastiveError",
    "ContrastiveItem",
    "Match",
    "SCHEMA_RELATIVE_PATH",
    "compile_patterns",
    "get_base",
    "json_schema",
    "load_base",
    "load_items",
    "normalize",
    "schema_json_text",
    "span_key",
]
