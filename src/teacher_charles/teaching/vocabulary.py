"""A base de vocabulário: as palavras cujo sentido o modelo não pode improvisar.

Irmã da ``contrastive.py``, e pelo mesmo motivo — onde existe item curado, o
que chega ao aluno vem *daqui*. A diferença é o sinal que dispara cada uma: a
base contrastiva reage a um **erro** no texto do aluno; esta reage a uma
**pergunta**, ou à agenda, porque não saber uma palavra não produz erro nenhum.

A medição está em ``docs/medicao-vocabulario.md``, e cada campo aqui existe
por causa de uma falha que ela mostrou. Vale repetir as duas que decidem o
desenho: perguntado quatro vezes o que significa *fodder*, o modelo acertou
uma; perguntado cinco vezes por *cannon fodder*, acertou duas. Mesmo prompt,
mesmo modelo. E não há rede embaixo disso — o verificador da E4 só aceita
achado que cite regra curada, e não existe regra curada para o sentido de uma
palavra.

## Por que não é uma quinta categoria do ``ContrastiveItem``

Porque não cabe. O schema de lá exige ``wrong`` e ``right`` com ``minItems:
1``, e uma palavra que o aluno não tem **não tem forma errada**: ele não errou,
ele não usou. Enfiar vocabulário ali pediria afrouxar a obrigatoriedade dos
dois campos que dão dente ao teste por item da E3.

O que liga as duas bases é ``rule_ids``. Foi o buraco mais claro da medição:
``false-friend.actually-atualmente`` está curado há três entregas, e em 25
tentativas sobre as cinco palavras que têm item próprio a armadilha do
português **não foi citada uma vez** — a base casa padrão contra o texto do
aluno, e uma pergunta não traz erro para casar. Então o item de vocabulário
aponta para a regra, e quem monta o prompt junta os dois.

## Por que não existe campo de pronúncia

``extra="forbid"`` recusa um ``ipa:`` no YAML, e isso é deliberado. O símbolo
que o aluno lê sai do espeak em ``teaching/phonetics.py``, que é o mesmo G2P
que a voz pronuncia — uma transcrição curada à mão aqui seria a segunda fonte
que a regra do G2P único existe para não ter, e a tela acabaria ensinando
contra o alto-falante. Na medição o modelo não digitou IPA em nenhuma das 49
respostas; não é um problema que precise de dado novo para resolver.

## Por que os exemplos são nossos, e o corpus entra só como número

``corpus/`` é material de terceiro e o repositório é público (ver
``corpus/README.md``): *derivar para ``data/``, nunca copiar para dentro*.
Uma frase de legenda colada aqui seria redistribuição, por curta que seja.
Então ``examples`` é escrito pelo curador, e ``CorpusNote`` guarda **só a
contagem** — quantas vezes a palavra aparece, em quantos enunciados, de qual
fonte.

A contagem sozinha conclui pouco, e o limite está pinado em teste: ausência em
492 enunciados é evidência sobre palavra rara, não sobre palavra comum —
`pretend` também não aparece no episódio. É somada ao nível que ela decide
algo: ``fodder`` é C1 **e** ausente, e é por isso que o item é de
``fodder for`` e não de celeiro.

## O que esta base não resolve

O gatilho. Ter o item curado não faz o item chegar ao aluno: falta a pergunta
do aluno levantar a palavra e a agenda da E13 oferecê-la com a chave
generalizada. Enquanto isso não existe, esta base é conteúdo sem caminho — o
que também é o estado em que a base contrastiva ficou entre a E3 e a E4.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..config import get_settings
from ..store.models import CEFR_LEVELS, CefrLevel
from .contrastive import RULE_ID_PATTERN, normalize

# ``vocab.cannon-fodder``: o prefixo entra no id pelo mesmo motivo que em
# ``contrastive.py`` — um id que se explica sozinho no log não precisa de
# consulta à base para ser entendido.
ITEM_ID_PATTERN = r"^vocab\.[a-z0-9]+(?:-[a-z0-9]+)*$"

PartOfSpeech = Literal["noun", "verb", "adjective", "adverb", "phrase"]

# Literal contra figurado é **tipo de sentido**, não registro, e os dois
# saíram da medição separados. O tipo, porque o dicionário bilíngue lidera com
# o sentido agrícola de `fodder` e o modelo repetiu a liderança em uma das
# quatro tentativas: a ordem dos sentidos é conteúdo curado. O registro, porque
# 1 resposta em 49 disse qualquer coisa sobre registro, frequência ou nível.
SenseKind = Literal["literal", "figurative"]
# Chama-se `style` e não `register` porque `register` sombreia um atributo do
# `BaseModel` do pydantic, e a carga passa a avisar em toda importação. O
# conceito é registro; o nome cedeu ao framework.
Style = Literal["neutral", "formal", "informal", "technical", "pejorative"]


class VocabularyError(ValueError):
    """A base não carregou. Traz todos os problemas, não só o primeiro."""


class CorpusNote(BaseModel):
    """O que o ``corpus/`` desta máquina diz da palavra — em número, só.

    Não há campo de texto aqui, e a ausência é a regra do ``corpus/README.md``
    escrita em código: a observação derivada é nossa para versionar, a frase
    de onde ela saiu não é.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Rótulo da fonte, como o `corpus/README.md` a nomeia: `friends-1x01`.
    source: str = Field(min_length=1)
    hits: int = Field(ge=0)
    # Denominador: quantos enunciados de fala foram varridos. Sem ele `hits: 3`
    # não quer dizer nada.
    of: int = Field(gt=0)

    @model_validator(mode="after")
    def _hits_fit_in_the_sweep(self) -> CorpusNote:
        """Mais ocorrências que enunciados é erro de curadoria, não dado.

        A contagem é copiada à mão do `measure_vocab.py` para o YAML, e um
        número que não cabe no denominador passaria calado — virando a
        justificativa numérica de uma aula que ninguém mediu.
        """
        if self.hits > self.of:
            raise ValueError(f"{self.hits} ocorrências em apenas {self.of} enunciados")
        return self


class Sense(BaseModel):
    """Um sentido da palavra, com o que o aluno precisa para usá-lo."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Vai para o aluno; em inglês, decisão 1.1 do plano.
    gloss: str = Field(min_length=1)
    kind: SenseKind
    style: Style = "neutral"

    # Escritos pelo curador, nunca copiados do `corpus/`. O teste por item
    # exige que cada um realmente use a palavra.
    examples: list[str] = Field(min_length=1)

    # Onde a palavra de fato vive. Para `fodder` isto é o item inteiro: fora de
    # `fodder for` e `cannon fodder` o aluno não vai encontrá-la.
    collocations: list[str] = Field(default_factory=list)


class VocabularyItem(BaseModel):
    """Uma palavra curada.

    ``extra="forbid"`` pelo mesmo motivo da base contrastiva — nome de campo
    errado num YAML curado à mão é erro de curadoria, e falhar na carga é
    melhor que ignorar em silêncio — e por um motivo a mais: é o que recusa um
    ``ipa:`` aqui, que o cabeçalho do módulo explica.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=ITEM_ID_PATTERN)
    word: str = Field(min_length=1)
    pos: PartOfSpeech

    # Em ordem de ensino: o primeiro é o sentido que o aluno vai encontrar.
    # Não é a ordem do dicionário — a entrada PT do Cambridge para `fodder`
    # traz só o sentido agrícola, e o figurado aparece lá apenas dentro dos
    # exemplos do corpus.
    senses: list[Sense] = Field(min_length=1)

    # Para a agenda não oferecer C1 a quem está em A2.
    cefr: CefrLevel = "B1"

    # A ponte para a base contrastiva, que é o buraco que a medição achou.
    rule_ids: list[str] = Field(default_factory=list)

    corpus: CorpusNote | None = None
    tags: list[str] = Field(default_factory=list)

    # Nota de curadoria. Nunca chega ao aluno, então pode estar em português.
    notes: str | None = None

    @property
    def primary(self) -> Sense:
        """O sentido a ensinar primeiro."""
        return self.senses[0]

    @model_validator(mode="after")
    def _id_is_the_word(self) -> VocabularyItem:
        """O id tem que ser a palavra em slug.

        Sem isto os dois divergem na primeira renomeação, e o id é o que vai
        para o log e para a agenda — um ``vocab.fodder`` apontando para
        *forage* seria invisível até alguém ler o YAML.
        """
        expected = "vocab." + "-".join(self.word.lower().split())
        if self.id != expected:
            raise ValueError(f"o id de {self.word!r} tem que ser {expected!r}, não {self.id!r}")
        return self

    @model_validator(mode="after")
    def _rule_ids_look_like_rule_ids(self) -> VocabularyItem:
        for rule_id in self.rule_ids:
            if not re.match(RULE_ID_PATTERN, rule_id):
                raise ValueError(f"{self.id}: {rule_id!r} não tem a forma de um rule_id")
        return self


class VocabularyBase:
    """A base carregada, consultável por palavra."""

    def __init__(self, items: Sequence[VocabularyItem]) -> None:
        self._items = tuple(items)
        self._by_id = {item.id: item for item in self._items}
        if len(self._by_id) != len(self._items):
            raise VocabularyError("há ids repetidos na base de vocabulário")
        self._by_word = {self._key(item.word): item for item in self._items}

    @staticmethod
    def _key(word: str) -> str:
        return " ".join(normalize(word).lower().split())

    @property
    def items(self) -> tuple[VocabularyItem, ...]:
        return self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[VocabularyItem]:
        return iter(self._items)

    def __contains__(self, word: object) -> bool:
        return isinstance(word, str) and self._key(word) in self._by_word

    def get(self, word: str) -> VocabularyItem | None:
        """O item da palavra, ou ``None``.

        ``None`` é resposta, não falha. Foi o controle da medição: perguntado
        por *house* — palavra sem nada a ensinar — o modelo inventou conversa
        nas cinco tentativas, e nenhuma disse que não havia aula ali. Quem
        chamar isto tem que saber tratar a ausência.
        """
        return self._by_word.get(self._key(word))

    def by_id(self, item_id: str) -> VocabularyItem:
        try:
            return self._by_id[item_id]
        except KeyError:
            raise VocabularyError(f"não há item {item_id!r} na base de vocabulário") from None

    def at_or_below(self, level: CefrLevel) -> tuple[VocabularyItem, ...]:
        """Os itens que cabem em quem está em ``level``.

        A agenda precisa disto para não oferecer `cannon fodder` (C1) a um A2.
        A base contrastiva usa o ``cefr`` para calibrar o quanto explicar; aqui
        ele é filtro, porque uma palavra acima do nível não é aula difícil, é
        aula perdida.
        """
        ceiling = CEFR_LEVELS.index(level)
        return tuple(item for item in self._items if CEFR_LEVELS.index(item.cefr) <= ceiling)


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


def load_items(directory: Path) -> list[VocabularyItem]:
    """Valida e devolve todos os itens de ``directory``, em ordem de arquivo.

    Acumula os problemas de todos os arquivos antes de falhar, como a base
    contrastiva: quem está curando quer a lista inteira do que corrigir.
    """
    if not directory.is_dir():
        raise VocabularyError(f"a base de vocabulário não existe em {directory}")

    files = sorted(directory.glob("*.yaml"))
    if not files:
        raise VocabularyError(f"nenhum arquivo .yaml em {directory}")

    items: list[VocabularyItem] = []
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
                item = VocabularyItem.model_validate(entry)
            except Exception as exc:  # pydantic.ValidationError e ValueError
                identifier = entry.get("id", "<sem id>")
                problems.append(f"{label} ({identifier}): {exc}")
                continue

            if item.id in seen:
                problems.append(f"{label} ({item.id}): id repetido, já definido em {seen[item.id]}")
                continue
            seen[item.id] = label

            items.append(item)

    if problems:
        raise VocabularyError(
            f"{len(problems)} problema(s) na base de vocabulário em {directory}:\n  - "
            + "\n  - ".join(problems)
        )

    return items


def load_base(directory: Path | None = None) -> VocabularyBase:
    """Carrega a base de ``directory`` (default: ``data/vocab/``)."""
    if directory is None:
        directory = get_settings().data_dir / "vocab"
    return VocabularyBase(load_items(directory))


@lru_cache(maxsize=4)
def get_base(directory: Path | None = None) -> VocabularyBase:
    """A base do processo, carregada uma vez. Mesma política da contrastiva:
    sem invalidação por mtime, porque editar conteúdo validado por teste vale
    um reinício."""
    return load_base(directory)


# --- schema ---------------------------------------------------------------- #

SCHEMA_RELATIVE_PATH = Path("schema") / "vocab-item.schema.json"


def json_schema() -> dict[str, Any]:
    """O JSON Schema de um item, derivado do modelo pydantic.

    O modelo é a fonte única; o arquivo em ``data/schema/`` é gerado por
    ``scripts/dump_vocab_schema.py`` e existe para editor e validador externos.
    Um teste garante que os dois não divergem.
    """
    schema = VocabularyItem.model_json_schema()
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://teacher-charles.local/schema/vocab-item.schema.json",
        **schema,
    }


def schema_json_text() -> str:
    return json.dumps(json_schema(), indent=2, ensure_ascii=False) + "\n"


__all__ = [
    "ITEM_ID_PATTERN",
    "SCHEMA_RELATIVE_PATH",
    "CorpusNote",
    "PartOfSpeech",
    "Style",
    "Sense",
    "SenseKind",
    "VocabularyBase",
    "VocabularyError",
    "VocabularyItem",
    "get_base",
    "json_schema",
    "load_base",
    "load_items",
    "schema_json_text",
]
