"""Testes da base de vocabulário.

Duas metades, como em ``test_contrastive.py``:

**Loader e modelo** — com bases sintéticas em ``tmp_path``, para não depender
do conteúdo curado. Um loader que aceita item fora do schema é bug de código.

**A base de verdade, item por item** — cada exemplo tem que usar a palavra,
cada ``gloss`` tem que estar em inglês, e cada ``rule_ids`` tem que apontar
para um item que existe em ``data/contrastive/``. Esse último é o teste que
fecha o buraco da medição: a ponte entre as duas bases só vale se não apontar
para o vazio, e um ``rule_id`` com erro de digitação é exatamente o tipo de
coisa que ninguém nota até o aluno não receber a armadilha.
"""

from __future__ import annotations

import json
import re

import pytest
import yaml
from pydantic import ValidationError

from teacher_charles.config import REPO_ROOT
from teacher_charles.store.models import CEFR_LEVELS
from teacher_charles.teaching.contrastive import load_base as load_contrastive
from teacher_charles.teaching.vocabulary import (
    SCHEMA_RELATIVE_PATH,
    VocabularyBase,
    VocabularyError,
    VocabularyItem,
    load_base,
    load_items,
    schema_json_text,
)

VOCAB_DIR = REPO_ROOT / "data" / "vocab"
REAL_BASE = load_base(VOCAB_DIR)
REAL_ITEMS = list(REAL_BASE)
REAL_IDS = [item.id for item in REAL_ITEMS]
CONTRASTIVE = load_contrastive(REPO_ROOT / "data" / "contrastive")


def write_base(directory, items, filename="items.yaml") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text(
        yaml.safe_dump({"items": items}, allow_unicode=True), encoding="utf-8"
    )


def valid_item(**overrides) -> dict:
    item = {
        "id": "vocab.fodder",
        "word": "fodder",
        "pos": "noun",
        "senses": [
            {
                "gloss": "Material that feeds talk or a story.",
                "kind": "figurative",
                "examples": ["The scandal was fodder for the tabloids."],
            }
        ],
    }
    item.update(overrides)
    return item


# --- modelo ---------------------------------------------------------------- #


def test_item_minimo_carrega():
    item = VocabularyItem.model_validate(valid_item())
    assert item.primary.kind == "figurative"
    # Default: o estilo mais comum, para o YAML não repetir `neutral` em tudo.
    assert item.primary.style == "neutral"
    assert item.cefr == "B1"


def test_campo_desconhecido_e_recusado():
    """``extra="forbid"``: nome errado em YAML curado à mão é erro de curadoria."""
    with pytest.raises(ValidationError):
        VocabularyItem.model_validate(valid_item(registro="neutral"))


def test_nao_existe_campo_de_pronuncia_no_item():
    """O IPA sai do espeak, e esta base não pode abrir uma segunda fonte.

    É a regra do G2P único (CLAUDE.md, "o IPA na tela vem do espeak") escrita
    onde ela pode ser violada: um `ipa:` curado à mão aqui faria a tela
    ensinar contra o alto-falante. Sem `extra="forbid"` o campo entraria
    calado e ninguém saberia até alguém decidir usá-lo.
    """
    with pytest.raises(ValidationError):
        VocabularyItem.model_validate(valid_item(ipa="fˈɑːdɚ"))


def test_id_tem_que_ser_a_palavra_em_slug():
    with pytest.raises(ValidationError, match="vocab.fodder"):
        VocabularyItem.model_validate(valid_item(id="vocab.forage"))


def test_id_de_palavra_composta_usa_hifen():
    item = VocabularyItem.model_validate(
        valid_item(id="vocab.cannon-fodder", word="cannon fodder", pos="phrase")
    )
    assert item.id == "vocab.cannon-fodder"


def test_item_sem_sentido_nenhum_e_recusado():
    with pytest.raises(ValidationError):
        VocabularyItem.model_validate(valid_item(senses=[]))


def test_sentido_sem_exemplo_e_recusado():
    """Glosa sem exemplo é definição de dicionário, que é o que não funcionou."""
    with pytest.raises(ValidationError):
        VocabularyItem.model_validate(
            valid_item(senses=[{"gloss": "Something.", "kind": "literal", "examples": []}])
        )


def test_rule_id_de_forma_errada_e_recusado():
    with pytest.raises(ValidationError, match="rule_id"):
        VocabularyItem.model_validate(valid_item(rule_ids=["actually-atualmente"]))


def test_contagem_de_corpus_maior_que_a_varredura_e_recusada():
    """3 ocorrências em 2 enunciados não é dado, é erro de transcrição."""
    with pytest.raises(ValidationError, match="enunciados"):
        VocabularyItem.model_validate(
            valid_item(corpus={"source": "friends-1x01", "hits": 3, "of": 2})
        )


def test_contagem_zero_e_dado_valido():
    """Zero é o caso que justifica o campo: `fodder` não aparece no corpus."""
    item = VocabularyItem.model_validate(
        valid_item(corpus={"source": "friends-1x01", "hits": 0, "of": 492})
    )
    assert item.corpus.hits == 0


# --- loader ---------------------------------------------------------------- #


def test_loader_acumula_todos_os_problemas(tmp_path):
    """Quem cura quer a lista inteira do que corrigir, não uma por rodada."""
    write_base(tmp_path / "vocab", [valid_item(id="vocab.forage"), valid_item(pos="substantivo")])
    with pytest.raises(VocabularyError) as excinfo:
        load_items(tmp_path / "vocab")
    assert "2 problema(s)" in str(excinfo.value)


def test_id_repetido_e_problema(tmp_path):
    write_base(tmp_path / "vocab", [valid_item(), valid_item()])
    with pytest.raises(VocabularyError, match="id repetido"):
        load_items(tmp_path / "vocab")


def test_diretorio_ausente_falha_com_mensagem(tmp_path):
    with pytest.raises(VocabularyError, match="não existe"):
        load_items(tmp_path / "nao-existe")


def test_diretorio_sem_yaml_falha(tmp_path):
    (tmp_path / "vocab").mkdir()
    with pytest.raises(VocabularyError, match="nenhum arquivo"):
        load_items(tmp_path / "vocab")


def test_arquivo_sem_items_falha(tmp_path):
    directory = tmp_path / "vocab"
    directory.mkdir()
    (directory / "items.yaml").write_text("palavras: []\n", encoding="utf-8")
    with pytest.raises(VocabularyError, match="items"):
        load_items(directory)


# --- consulta -------------------------------------------------------------- #


def test_get_normaliza_caixa_e_espaco():
    base = VocabularyBase(
        [VocabularyItem.model_validate(valid_item(id="vocab.cannon-fodder", word="cannon fodder"))]
    )
    assert base.get("Cannon Fodder") is not None
    assert base.get("cannon  fodder") is not None
    assert "CANNON FODDER" in base


def test_palavra_sem_item_devolve_none():
    """``None`` é resposta, e é o controle da medição.

    Perguntado por *house* — palavra sem nada a ensinar — o modelo inventou
    conversa nas cinco tentativas. A base tem que ser capaz de dizer que não
    há aula, e é por isso que não existe item de `house` em `data/vocab/`.
    """
    assert REAL_BASE.get("house") is None
    assert "house" not in REAL_BASE


def test_at_or_below_nao_oferece_acima_do_nivel():
    """Palavra acima do nível não é aula difícil, é aula perdida."""
    a2 = {item.id for item in REAL_BASE.at_or_below("A2")}
    c1 = {item.id for item in REAL_BASE.at_or_below("C1")}
    assert "vocab.cannon-fodder" not in a2
    assert "vocab.cannon-fodder" in c1
    assert a2 <= c1


def test_by_id_desconhecido_falha():
    with pytest.raises(VocabularyError, match="não há item"):
        REAL_BASE.by_id("vocab.inexistente")


# --- a base de verdade, item por item -------------------------------------- #


def test_a_base_tem_itens():
    assert len(REAL_BASE) >= 7


@pytest.mark.parametrize("item", REAL_ITEMS, ids=REAL_IDS)
def test_todo_exemplo_usa_a_palavra(item):
    """Exemplo que não traz a palavra é exemplo de outra coisa.

    A comparação é por radical, e grosseira de propósito: `realize` aparece
    como *realized*, `pretend` como *pretended*. Cortar o `e` final cobre a
    flexão sem precisar de um lematizador para sete itens.
    """
    stem = item.word.lower().rstrip("e")
    for sense in item.senses:
        for example in sense.examples:
            assert stem in example.lower(), f"{item.id}: {example!r} não usa {item.word!r}"


@pytest.mark.parametrize("item", REAL_ITEMS, ids=REAL_IDS)
def test_glosa_esta_em_ingles(item):
    """Decisão 1.1: nada de português chega ao aluno.

    Mesma heurística do ``test_contrastive.py`` — prosa em português carrega
    acento ou cedilha —, e os trechos entre aspas ficam fora, porque citar a
    palavra do português é trabalho legítimo de uma glosa.
    """
    for sense in item.senses:
        prose = re.sub(r'"[^"]*"', "", sense.gloss)
        assert prose.isascii(), f"{item.id}: a glosa parece não estar em inglês"


@pytest.mark.parametrize("item", REAL_ITEMS, ids=REAL_IDS)
def test_todo_exemplo_esta_em_ingles(item):
    for sense in item.senses:
        for example in sense.examples:
            assert example.isascii(), f"{item.id}: o exemplo {example!r} não parece inglês"


@pytest.mark.parametrize("item", REAL_ITEMS, ids=REAL_IDS)
def test_rule_ids_apontam_para_item_que_existe(item):
    """A ponte entre as duas bases, que é o buraco que a medição achou.

    Em 25 tentativas sobre as cinco palavras com item contrastivo curado, a
    armadilha do português não foi citada uma vez. O ``rule_ids`` existe para
    isso — e um id com erro de digitação devolveria o silêncio que ele veio
    consertar, sem ninguém notar.
    """
    for rule_id in item.rule_ids:
        assert rule_id in CONTRASTIVE, f"{item.id}: {rule_id} não existe na base contrastiva"


@pytest.mark.parametrize("item", REAL_ITEMS, ids=REAL_IDS)
def test_item_com_mais_de_um_sentido_explica_a_ordem(item):
    """A ordem dos sentidos é conteúdo curado, então ela tem que estar dita.

    O dicionário bilíngue lidera com o sentido agrícola de *fodder*; aqui a
    ordem é a do ensino. Um item que inverte a ordem do dicionário sem dizer
    por quê é um item que o próximo curador vai "consertar".
    """
    if len(item.senses) > 1:
        assert item.notes, f"{item.id}: mais de um sentido e nenhuma nota de curadoria"


@pytest.mark.parametrize("item", REAL_ITEMS, ids=REAL_IDS)
def test_palavra_avancada_e_ausente_do_corpus_vive_de_colocacao(item):
    """Palavra avançada que o corpus não traz tem que dizer onde ela aparece.

    É o que separa `fodder` de uma entrada de dicionário: C1, 0 em 492
    enunciados, e o que serve ao aluno é `fodder for`.

    A guarda vale só de C1 para cima, e a primeira versão errou nisso — pedia
    colocação de todo item ausente do corpus, e `eventually`, `pretend` e
    `comprehensive` quebraram o teste. Estão certos: são palavras B1/B2
    comuns, e faltar num episódio de 22 minutos é tamanho de amostra, não
    raridade. 492 enunciados são evidência de que uma palavra rara é rara;
    não são evidência sobre uma palavra comum.
    """
    advanced = CEFR_LEVELS.index(item.cefr) >= CEFR_LEVELS.index("C1")
    if advanced and item.corpus is not None and item.corpus.hits == 0:
        assert any(sense.collocations for sense in item.senses), (
            f"{item.id}: avançada, ausente do corpus e sem colocação que justifique a aula"
        )


# --- schema versionado ----------------------------------------------------- #


def test_schema_versionado_bate_com_o_modelo():
    path = REPO_ROOT / "data" / SCHEMA_RELATIVE_PATH
    assert path.is_file(), f"falta {path}"
    assert path.read_text(encoding="utf-8") == schema_json_text(), (
        "schema fora de sincronia: rode `uv run python scripts/dump_vocab_schema.py`"
    )


def test_schema_versionado_e_json_schema_valido():
    path = REPO_ROOT / "data" / SCHEMA_RELATIVE_PATH
    schema = json.loads(path.read_text(encoding="utf-8"))
    assert schema["$schema"].startswith("https://json-schema.org/")
    for required in ("id", "word", "pos", "senses"):
        assert required in schema["required"]
