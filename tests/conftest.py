"""Isolamento dos testes em relação ao ambiente da máquina.

`Settings` lê variáveis `TC_*` e o `.env` da raiz, o que é certo em produção e
errado num teste: a suíte passaria ou falharia conforme o que o dev tem
configurado. Um `.env` com `TC_LLM_MODEL` apontando para outro modelo já
quebrou um teste aqui.
"""

from __future__ import annotations

import pytest

from teacher_charles.config import Settings, get_settings


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """Remove toda variável ``TC_*`` e zera o cache de settings."""
    for key in list(__import__("os").environ):
        if key.startswith("TC_"):
            monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def make_settings(**overrides) -> Settings:
    """Settings de teste, sem ler o ``.env`` do repositório."""
    return Settings(_env_file=None, **overrides)
