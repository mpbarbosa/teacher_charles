"""Isolamento dos testes em relação ao ambiente da máquina.

`Settings` lê variáveis `TC_*` e o `.env` da raiz, o que é certo em produção e
errado num teste: a suíte passaria ou falharia conforme o que o dev tem
configurado. Um `.env` com `TC_LLM_MODEL` apontando para outro modelo já
quebrou um teste aqui.
"""

from __future__ import annotations

import os

import pytest

from teacher_charles.config import Settings, get_settings
from teacher_charles.llm import OllamaProvider, list_models
from teacher_charles.llm.base import ProviderError


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


# --- Integração com o Ollama de verdade ----------------------------------- #
#
# Compartilhado por mais de um arquivo de integração (E1 e E4), por isso mora
# aqui. Pula sozinho quando não há Ollama ou modelo: quem clona o repo sem nada
# instalado ainda vê a suíte verde.

OLLAMA_URL = os.environ.get("TC_OLLAMA_URL", "http://localhost:11434")


async def pick_ollama_model() -> str:
    try:
        available = await list_models(OLLAMA_URL, timeout_s=3.0)
    except ProviderError as exc:
        pytest.skip(f"Ollama não está acessível em {OLLAMA_URL}: {exc}")

    if not available:
        pytest.skip("Ollama está de pé, mas não há nenhum modelo baixado")

    forced = os.environ.get("TC_TEST_MODEL", "").strip()
    if forced:
        if forced not in available:
            pytest.skip(f"TC_TEST_MODEL={forced} não está baixado")
        return forced

    configured = os.environ.get("TC_LLM_MODEL", "").strip()
    if configured and configured in available:
        return configured

    # O menor que estiver presente: o teste deve ser rápido, não representativo.
    def size_hint(name: str) -> tuple[int, str]:
        for marker, rank in (("1b", 0), ("3b", 1), ("4b", 2), ("7b", 3), ("8b", 4)):
            if marker in name.lower():
                return (rank, name)
        return (9, name)

    return sorted(available, key=size_hint)[0]


@pytest.fixture
def ollama_url() -> str:
    return OLLAMA_URL


@pytest.fixture
async def ollama_provider():
    model = await pick_ollama_model()
    instance = OllamaProvider(model=model, base_url=OLLAMA_URL, request_timeout_s=180.0)
    yield instance
    await instance.aclose()
