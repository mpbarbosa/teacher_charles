"""Testes do carregador de prompts (E2)."""

from __future__ import annotations

import os

import pytest

from teacher_charles.config import REPO_ROOT
from teacher_charles.prompts import PromptLibrary, PromptNotFound


def test_loads_and_strips(tmp_path):
    (tmp_path / "persona.md").write_text("\n  You are Charles.  \n\n", encoding="utf-8")
    assert PromptLibrary(tmp_path).get("persona") == "You are Charles."


def test_accepts_name_with_or_without_extension(tmp_path):
    (tmp_path / "persona.md").write_text("x", encoding="utf-8")
    library = PromptLibrary(tmp_path)
    assert library.get("persona") == library.get("persona.md") == "x"


def test_nested_names(tmp_path):
    (tmp_path / "modes").mkdir()
    (tmp_path / "modes" / "drill.md").write_text("drill", encoding="utf-8")
    assert PromptLibrary(tmp_path).get("modes/drill") == "drill"


def test_missing_prompt_names_the_expected_path(tmp_path):
    with pytest.raises(PromptNotFound) as caught:
        PromptLibrary(tmp_path).get("persona")
    assert "persona" in str(caught.value)


def test_escaping_the_prompts_dir_is_refused(tmp_path):
    (tmp_path.parent / "segredo.md").write_text("nope", encoding="utf-8")
    with pytest.raises(PromptNotFound):
        PromptLibrary(tmp_path).get("../segredo")


def test_cache_invalidates_on_edit(tmp_path):
    """Editar um .md deve valer no turno seguinte, sem reiniciar o servidor."""
    path = tmp_path / "persona.md"
    path.write_text("primeira", encoding="utf-8")
    library = PromptLibrary(tmp_path)
    assert library.get("persona") == "primeira"

    path.write_text("segunda", encoding="utf-8")
    # mtime tem granularidade; força a diferença para o teste não depender dela.
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 10))

    assert library.get("persona") == "segunda"


def test_cache_is_used_when_mtime_is_unchanged(tmp_path):
    """Arquivo intocado não deve ser relido do disco.

    O teste reescreve o conteúdo mas restaura o mtime, então um ``get`` que
    relesse devolveria o texto novo. O ``stat()`` continua acontecendo a cada
    chamada — é ele que detecta a edição — e isso é intencional: barato, e é a
    única forma de invalidar sem reiniciar o servidor.
    """
    path = tmp_path / "persona.md"
    path.write_text("original", encoding="utf-8")
    library = PromptLibrary(tmp_path)
    assert library.get("persona") == "original"

    stat_before = path.stat()
    path.write_text("reescrito por fora", encoding="utf-8")
    os.utime(path, (stat_before.st_atime, stat_before.st_mtime))

    assert library.get("persona") == "original", "releu o disco com mtime igual"


def test_missing_file_raises_even_if_cached(tmp_path):
    """Prompt apagado tem que falhar alto, não servir texto fantasma."""
    path = tmp_path / "persona.md"
    path.write_text("original", encoding="utf-8")
    library = PromptLibrary(tmp_path)
    library.get("persona")

    path.unlink()
    with pytest.raises(PromptNotFound):
        library.get("persona")


def test_exists(tmp_path):
    (tmp_path / "persona.md").write_text("x", encoding="utf-8")
    library = PromptLibrary(tmp_path)
    assert library.exists("persona")
    assert not library.exists("ausente")
    assert not library.exists("../fora")


# --- A persona de verdade, do repo ----------------------------------------- #


def test_real_persona_exists_and_is_english_only():
    """A persona versionada tem que carregar e respeitar a decisão 1.1."""
    text = PromptLibrary(REPO_ROOT / "prompts").get("persona")

    assert text, "a persona está vazia"
    assert "Charles" in text
    # Decisão 1.1: conversa 100% em inglês. A instrução tem que estar explícita.
    assert "only English" in text
    # E tem que saber que o aluno é brasileiro — é o diferencial do produto.
    assert "Brazilian" in text
