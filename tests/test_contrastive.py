"""Testes da base contrastiva (E3).

Duas metades, com propósitos diferentes:

**Loader e matcher** — exercitados com bases sintéticas em ``tmp_path``, para
não depender do conteúdo curado. Um loader que aceita item fora do schema é bug
de código.

**A base de verdade, item por item** — todo exemplo de ``wrong`` tem que casar
com o próprio item, e nenhum exemplo de ``right`` pode casar (nos itens não
ambíguos). É o "um teste por item" da E3: um padrão que não generaliza, ou que
morde inglês correto, quebra a suíte na hora em que é escrito. Como os itens com
``patterns`` ignoram os literais de ``wrong``, o teste não é vacuoso — é o padrão
que está sob prova.
"""

from __future__ import annotations

import json
import re

import pytest

from teacher_charles.config import REPO_ROOT
from teacher_charles.teaching.contrastive import (
    CATEGORIES,
    SCHEMA_RELATIVE_PATH,
    ContrastiveBase,
    ContrastiveError,
    ContrastiveItem,
    load_base,
    load_items,
    normalize,
    schema_json_text,
)

CONTRASTIVE_DIR = REPO_ROOT / "data" / "contrastive"


def write_base(directory, items, filename="items.yaml") -> None:
    """Escreve um YAML de base a partir de dicts, sem passar pelo pydantic."""
    import yaml

    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text(
        yaml.safe_dump({"items": items}, allow_unicode=True), encoding="utf-8"
    )


def valid_item(**overrides) -> dict:
    item = {
        "id": "calque.age-have",
        "category": "calque",
        "pt": "Eu tenho 30 anos",
        "wrong": ["I have 30 years"],
        "right": ["I am 30 years old"],
        "patterns": [r"\b(i|you|we|they) have \d+ years?\b"],
        "explanation": "Age is something you are, not something you have.",
    }
    item.update(overrides)
    return item


# --- loader ---------------------------------------------------------------- #


def test_loads_a_valid_base(tmp_path):
    write_base(tmp_path, [valid_item()])
    items = load_items(tmp_path)
    assert [item.id for item in items] == ["calque.age-have"]


def test_missing_directory_is_named_in_the_error(tmp_path):
    with pytest.raises(ContrastiveError) as caught:
        load_items(tmp_path / "ausente")
    assert "ausente" in str(caught.value)


def test_directory_without_yaml_is_refused(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "leia-me.md").write_text("nada aqui", encoding="utf-8")
    with pytest.raises(ContrastiveError):
        load_items(tmp_path)


def test_item_outside_the_schema_is_refused(tmp_path):
    """Critério de aceite da E3: o loader rejeita item fora do schema."""
    write_base(tmp_path, [valid_item(explanation=None)])
    with pytest.raises(ContrastiveError) as caught:
        load_items(tmp_path)
    assert "explanation" in str(caught.value)


def test_unknown_field_is_refused(tmp_path):
    """Campo com nome errado num YAML curado é erro de curadoria, não extra."""
    write_base(tmp_path, [valid_item(pattern=[r"\bfoo\b"])])
    with pytest.raises(ContrastiveError) as caught:
        load_items(tmp_path)
    assert "pattern" in str(caught.value)


def test_malformed_rule_id_is_refused(tmp_path):
    write_base(tmp_path, [valid_item(id="Calque.Age_Have")])
    with pytest.raises(ContrastiveError):
        load_items(tmp_path)


def test_rule_id_prefix_must_match_the_category(tmp_path):
    write_base(tmp_path, [valid_item(id="grammar.age-have")])
    with pytest.raises(ContrastiveError) as caught:
        load_items(tmp_path)
    assert "categoria" in str(caught.value)


def test_unknown_category_is_refused(tmp_path):
    write_base(tmp_path, [valid_item(id="slang.whatever", category="slang")])
    with pytest.raises(ContrastiveError):
        load_items(tmp_path)


def test_empty_wrong_or_right_is_refused(tmp_path):
    write_base(tmp_path, [valid_item(wrong=[])])
    with pytest.raises(ContrastiveError):
        load_items(tmp_path)


def test_duplicate_id_across_files_is_refused(tmp_path):
    write_base(tmp_path, [valid_item()], filename="a.yaml")
    write_base(tmp_path, [valid_item()], filename="b.yaml")
    with pytest.raises(ContrastiveError) as caught:
        load_items(tmp_path)
    assert "repetido" in str(caught.value)


def test_invalid_regex_is_refused_at_load_time(tmp_path):
    """Regex quebrada tem que falhar na carga, não no primeiro turno do aluno."""
    write_base(tmp_path, [valid_item(patterns=[r"\b(i have"])])
    with pytest.raises(ContrastiveError) as caught:
        load_items(tmp_path)
    assert "regex" in str(caught.value)


def test_all_problems_are_reported_at_once(tmp_path):
    """Quem cura quer a lista inteira, não uma mensagem por rodada."""
    write_base(
        tmp_path,
        [
            valid_item(id="calque.um", explanation=None),
            valid_item(id="grammar.dois", category="calque"),
            valid_item(id="calque.tres", patterns=[r"\b(unclosed"]),
        ],
    )
    with pytest.raises(ContrastiveError) as caught:
        load_items(tmp_path)
    message = str(caught.value)
    assert "3 problema" in message
    assert "calque.um" in message and "grammar.dois" in message and "calque.tres" in message


def test_broken_yaml_is_reported_with_the_file_name(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "torto.yaml").write_text("items:\n  - id: [unclosed\n", encoding="utf-8")
    with pytest.raises(ContrastiveError) as caught:
        load_items(tmp_path)
    assert "torto.yaml" in str(caught.value)


def test_file_without_items_key_is_reported(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "vazio.yaml").write_text("version: 1\n", encoding="utf-8")
    with pytest.raises(ContrastiveError) as caught:
        load_items(tmp_path)
    assert "items" in str(caught.value)


# --- matcher --------------------------------------------------------------- #


def base_with(*items) -> ContrastiveBase:
    return ContrastiveBase([ContrastiveItem.model_validate(item) for item in items])


def test_match_returns_the_rule_id():
    """Critério de aceite da E3."""
    base = load_base(CONTRASTIVE_DIR)
    assert base.rule_ids("I have 30 years") == ("calque.age-have",)


def test_match_generalizes_beyond_the_examples():
    base = load_base(CONTRASTIVE_DIR)
    # Nenhuma dessas frases está nos exemplos do item.
    assert "calque.age-have" in base.rule_ids("my daughter has 11 years and loves it")
    assert "calque.age-have" in base.rule_ids("they have 44 years of marriage")


def test_span_points_into_the_original_text():
    """A correção mostra ao aluno o trecho como ele escreveu."""
    base = base_with(valid_item())
    text = "Hello! I HAVE 30 YEARS, and you?"
    (match,) = base.match(text)
    assert text[match.start : match.end] == match.span == "I HAVE 30 YEARS"


def test_matching_survives_curly_apostrophes():
    """O que o celular e o Word produzem tem que casar igual."""
    base = base_with(
        valid_item(
            id="false-friend.support-suportar",
            category="false-friend",
            wrong=["I can't support this noise"],
            right=["I can't stand this noise"],
            patterns=[r"\b(can't|cannot) support\b"],
        )
    )
    text = "I can’t support this noise"
    (match,) = base.match(text)
    assert match.rule_id == "false-friend.support-suportar"
    # O span sai do original, com o apóstrofo curvo preservado.
    assert "’" in match.span


def test_normalize_preserves_length():
    """É o que mantém os offsets válidos no texto original."""
    text = "I can’t — really — do it now"
    assert len(normalize(text)) == len(text)


def test_without_patterns_the_wrong_examples_become_the_pattern():
    base = base_with(
        valid_item(
            id="false-friend.pasta-pasta",
            category="false-friend",
            wrong=["the pasta with the documents"],
            right=["the folder with the documents"],
            patterns=[],
        )
    )
    assert base.rule_ids("send me the pasta with the documents") == ("false-friend.pasta-pasta",)
    # Literal: não generaliza, e é por isso que itens de forma variável
    # precisam de `patterns` escritos à mão.
    assert base.rule_ids("send me the pasta with the files") == ()


def test_literal_patterns_tolerate_extra_whitespace():
    base = base_with(valid_item(patterns=[], wrong=["I have 30 years"]))
    assert base.rule_ids("I  have\n30 years") == ("calque.age-have",)


def test_one_match_per_item_even_when_repeated():
    base = base_with(valid_item())
    matches = base.match("I have 30 years and they have 12 years")
    assert [m.rule_id for m in matches] == ["calque.age-have"]
    assert matches[0].span == "I have 30 years"


def test_matches_come_in_order_of_appearance():
    base = load_base(CONTRASTIVE_DIR)
    ids = base.rule_ids("It depends of you, but you have reason")
    assert ids == ("calque.depend-of", "calque.have-reason")


def test_no_match_on_clean_english():
    base = load_base(CONTRASTIVE_DIR)
    text = "I am 34 years old and I have lived in Recife since 2015. It is raining today."
    assert base.rule_ids(text) == ()


def test_candidates_put_unambiguous_items_first():
    """A análise da E4 recebe primeiro o que tem mais chance de confirmar."""
    base = base_with(
        valid_item(id="calque.hedge", ambiguous=True, patterns=[r"\bmaybe\b"]),
        valid_item(id="calque.age-have"),
    )
    text = "maybe I have 30 years"
    assert [item.id for item in base.candidates(text)] == ["calque.age-have", "calque.hedge"]


def test_candidates_respects_the_limit():
    base = load_base(CONTRASTIVE_DIR)
    text = "I have 30 years, it depends of you, you have reason, explain me this"
    assert len(base.match(text)) > 2
    assert len(base.candidates(text, limit=2)) == 2


def test_get_and_contains():
    base = load_base(CONTRASTIVE_DIR)
    assert base.get("calque.age-have").category == "calque"
    assert "calque.age-have" in base
    with pytest.raises(LookupError):
        base.get("calque.nao-existe")


def test_duplicate_ids_are_refused_by_the_base():
    with pytest.raises(ContrastiveError):
        base_with(valid_item(), valid_item())


# --- a base de verdade, do repo -------------------------------------------- #


REAL_BASE = load_base(CONTRASTIVE_DIR)
REAL_ITEMS = [pytest.param(item, id=item.id) for item in REAL_BASE]


def test_the_real_base_has_the_e3_volume():
    """A E3 entrega ~40 itens; crescer é esperado, encolher não."""
    assert len(REAL_BASE) >= 40


def test_the_real_base_covers_the_curated_categories():
    present = {item.category for item in REAL_BASE}
    assert {"false-friend", "calque", "grammar"} <= present
    assert present <= set(CATEGORIES)


@pytest.mark.parametrize("item", REAL_ITEMS)
def test_every_wrong_example_matches_its_own_item(item):
    """O padrão tem que cobrir o que o item diz cobrir."""
    for example in item.wrong:
        assert item.id in REAL_BASE.rule_ids(example), (
            f"{item.id}: o padrão não pega o próprio exemplo {example!r}"
        )


@pytest.mark.parametrize("item", REAL_ITEMS)
def test_right_examples_do_not_match_the_item(item):
    """Não corrigir inglês correto é metade do valor da base curada."""
    if item.ambiguous:
        pytest.skip("item marcado como ambíguo: quem decide é a análise da E4")
    for example in item.right:
        assert item.id not in REAL_BASE.rule_ids(example), (
            f"{item.id}: o padrão morde a forma correta {example!r}"
        )


@pytest.mark.parametrize("item", REAL_ITEMS)
def test_explanation_is_in_english(item):
    """Decisão 1.1: nada de português chega ao aluno.

    Heurística barata e suficiente: prosa em português carrega acento ou
    cedilha. Fora da checagem ficam os trechos entre aspas — citar a palavra
    que causa o erro ("fábrica" é "factory") é justamente o trabalho de uma
    base contrastiva — e ``notes``, que é para o curador, não para o aluno.
    """
    prose = re.sub(r'"[^"]*"', "", item.explanation)
    assert prose.isascii(), f"{item.id}: a explicação parece não estar em inglês"
    # E o português citado tem que estar entre aspas, não solto na frase.
    assert item.explanation.count('"') % 2 == 0, f"{item.id}: aspas desbalanceadas"


@pytest.mark.parametrize("item", REAL_ITEMS)
def test_ambiguous_items_explain_themselves(item):
    """Item ambíguo sem nota é item que o próximo curador não sabe por que é."""
    if item.ambiguous:
        assert item.notes or len(item.patterns) >= 1


def test_ambiguous_items_are_the_minority():
    """Se a maioria da base precisa do modelo para decidir, ela não é anteparo."""
    ambiguous = [item.id for item in REAL_BASE if item.ambiguous]
    assert len(ambiguous) <= len(REAL_BASE) // 3, ambiguous


# --- schema versionado ----------------------------------------------------- #


def test_versioned_schema_matches_the_model():
    """O JSON Schema é gerado do modelo; divergir é não ter schema nenhum."""
    path = REPO_ROOT / "data" / SCHEMA_RELATIVE_PATH
    assert path.is_file(), f"falta {path}"
    assert path.read_text(encoding="utf-8") == schema_json_text(), (
        "schema fora de sincronia: rode `uv run python scripts/dump_contrastive_schema.py`"
    )


def test_versioned_schema_is_valid_json_schema():
    path = REPO_ROOT / "data" / SCHEMA_RELATIVE_PATH
    schema = json.loads(path.read_text(encoding="utf-8"))
    assert schema["$schema"].startswith("https://json-schema.org/")
    assert "id" in schema["required"] and "explanation" in schema["required"]
