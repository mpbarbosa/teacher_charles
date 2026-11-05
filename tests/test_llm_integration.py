"""Teste de integração: um turno de verdade contra o Ollama local.

É o critério de pronto da E1. Tudo o mais na suíte roda sem modelo; este
exercita o caminho real — HTTP, NDJSON, streaming, um modelo carregando na CPU.

Pula sozinho quando não há Ollama ou modelo, então quem clona o repo sem nada
instalado ainda vê a suíte verde. Rode só este com::

    uv run pytest -m integration

O modelo usado é, em ordem: ``TC_TEST_MODEL``, o ``TC_LLM_MODEL`` configurado
se estiver baixado, ou o menor modelo presente na máquina — para não obrigar
ninguém a baixar 2,5 GB só para rodar a suíte.
"""

from __future__ import annotations

import os
import time

import pytest

from teacher_charles.llm import (
    GenerationOptions,
    Message,
    ModelNotFound,
    OllamaProvider,
    list_models,
)
from teacher_charles.llm.base import ProviderError

pytestmark = pytest.mark.integration

OLLAMA_URL = os.environ.get("TC_OLLAMA_URL", "http://localhost:11434")


async def _pick_model() -> str:
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
async def provider():
    model = await _pick_model()
    instance = OllamaProvider(model=model, base_url=OLLAMA_URL, request_timeout_s=180.0)
    yield instance
    await instance.aclose()


async def test_real_turn_streams_text(provider):
    """Um turno completo: pergunta vai, texto volta em pedaços.

    Pede várias frases de propósito. Uma resposta de uma palavra chega num
    único pedaço, e aí o teste passaria mesmo se o ``stream: true`` estivesse
    sendo ignorado — não provaria nada sobre streaming.
    """
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="In three sentences, describe the city of Paris."),
    ]
    options = GenerationOptions(temperature=0.0, max_tokens=120, think=False)

    chunks: list[str] = []
    started = time.monotonic()
    first_chunk_at: float | None = None

    async for chunk in provider.stream(messages, options):
        if first_chunk_at is None:
            first_chunk_at = time.monotonic() - started
        chunks.append(chunk)

    answer = "".join(chunks)
    total = time.monotonic() - started

    print(
        f"\n  modelo={provider.model}  pedaços={len(chunks)}  "
        f"1º pedaço={first_chunk_at:.2f}s  total={total:.2f}s"
    )

    assert answer.strip(), "o modelo não devolveu texto nenhum"
    assert "paris" in answer.lower(), f"resposta inesperada: {answer!r}"
    # Esta é a asserção que importa: streaming de verdade entrega em muitos
    # pedaços. Um blob único significaria que `stream: true` foi ignorado, ou
    # que estamos acumulando a resposta antes de repassá-la.
    assert len(chunks) > 5, f"esperava streaming, vieram {len(chunks)} pedaço(s)"
    # E o primeiro pedaço tem que chegar bem antes do fim, senão não há ganho
    # de latência percebida nenhum.
    assert first_chunk_at is not None
    assert first_chunk_at < total, "o primeiro pedaço chegou junto com o último"


async def test_real_correction_turn_shows_why_structure_belongs_in_code(provider):
    """Prova que o canal funciona — e, de passagem, que o prompt não basta.

    Este teste afirma SÓ que um turno de correção vai e volta. Deliberadamente
    não afirma que a correção está certa, porque medido num 3B ela não está:

        aluno:     "I have 30 years."
        esperado:  "I am 30 years old."   (decalque do português)
        obtido:    "I have thirty years." (trocou o dígito, manteve o erro)

    O modelo escreveu o número em palavras e deixou o decalque de pé. É o
    problema nº 1 da análise inicial em uma linha, e a razão de a base
    contrastiva (E3) e a saída estruturada com regra citada (E4) existirem:
    com uma regra curada para `calque.age-have`, a correção não depende de o
    modelo ter essa intuição.

    Cobrar a correção certa aqui seria testar o modelo, não o código.
    """
    messages = [
        Message(
            role="system",
            content=(
                "You are an English teacher. The student is Brazilian. "
                "Reply with ONLY the corrected sentence, nothing else."
            ),
        ),
        Message(role="user", content="I have 30 years."),
    ]
    answer = await provider.complete(
        messages, GenerationOptions(temperature=0.0, max_tokens=32, think=False)
    )
    print(f"\n  modelo={provider.model}  correção={answer.strip()!r}")

    assert answer.strip(), "o modelo não devolveu texto nenhum"
    # O canal entregou uma frase em inglês; o acerto pedagógico é da E3/E4.
    assert len(answer.split()) >= 2, f"resposta degenerada: {answer!r}"


async def test_real_missing_model_raises_model_not_found():
    """O 404 do Ollama tem que chegar como ModelNotFound, não como erro genérico."""
    provider = OllamaProvider(
        model="modelo-que-nao-existe:0b",
        base_url=OLLAMA_URL,
        request_timeout_s=15.0,
    )
    try:
        with pytest.raises(ModelNotFound):
            await provider.complete([Message(role="user", content="hi")])
    finally:
        await provider.aclose()


async def test_real_list_models_returns_sorted_names():
    try:
        models = await list_models(OLLAMA_URL, timeout_s=3.0)
    except ProviderError as exc:
        pytest.skip(f"Ollama não está acessível: {exc}")

    assert models == sorted(models)
    assert all(isinstance(name, str) and name for name in models)
