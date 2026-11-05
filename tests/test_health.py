"""Testes da sondagem do modelo e do /health (E0)."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from teacher_charles.app import create_app
from teacher_charles.config import Settings
from teacher_charles.health import _matches, probe_model
from teacher_charles.llm import FakeProvider
from teacher_charles.teaching.contrastive import ContrastiveError


def make_settings(tmp_path, **overrides) -> Settings:
    base = {
        "provider": "ollama",
        "llm_model": "qwen3:4b",
        "ollama_url": "http://localhost:11434",
        "db_path": tmp_path / "health.db",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


@pytest.mark.parametrize(
    ("wanted", "available", "expected"),
    [
        ("qwen3:4b", "qwen3:4b", True),
        ("qwen3", "qwen3:latest", True),  # tag implícita do Ollama
        ("qwen3:4b", "qwen3:8b", False),
        ("qwen3:4b", "qwen3:latest", False),
        ("qwen3", "qwen3:4b", False),
    ],
)
def test_model_name_matching(wanted, available, expected):
    assert _matches(wanted, available) is expected


def _mock_transport(payload, status_code=200):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(status_code, json=payload)

    return httpx.MockTransport(handler)


@pytest.fixture
def patched_client(monkeypatch):
    """Troca o AsyncClient do módulo health por um com transporte falso."""

    def install(payload, status_code=200):
        transport = _mock_transport(payload, status_code)
        real = httpx.AsyncClient

        def factory(*args, **kwargs):
            kwargs["transport"] = transport
            return real(*args, **kwargs)

        monkeypatch.setattr("teacher_charles.llm.ollama.httpx.AsyncClient", factory)

    return install


async def test_probe_finds_present_model(tmp_path, patched_client):
    patched_client({"models": [{"name": "qwen3:4b"}, {"name": "llama3.1:8b"}]})
    status = await probe_model(make_settings(tmp_path))

    assert status.reachable
    assert status.model_present
    assert status.ok
    assert status.available_models == ["llama3.1:8b", "qwen3:4b"]


async def test_probe_reports_missing_model_with_alternatives(tmp_path, patched_client):
    patched_client({"models": [{"name": "qwen2.5:3b-instruct-q4_K_M"}]})
    status = await probe_model(make_settings(tmp_path))

    assert status.reachable
    assert not status.model_present
    assert not status.ok
    assert "ollama pull qwen3:4b" in status.detail


async def test_probe_handles_unreachable_ollama(tmp_path, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    real = httpx.AsyncClient
    monkeypatch.setattr(
        "teacher_charles.llm.ollama.httpx.AsyncClient",
        lambda *a, **kw: real(*a, **{**kw, "transport": httpx.MockTransport(handler)}),
    )

    status = await probe_model(make_settings(tmp_path))
    assert not status.reachable
    assert not status.ok
    assert "não respondeu" in status.detail


async def test_probe_handles_empty_model_list(tmp_path, patched_client):
    patched_client({"models": []})
    status = await probe_model(make_settings(tmp_path))

    assert status.reachable
    assert not status.model_present
    assert "nenhum modelo baixado" in status.detail


async def test_openai_provider_requires_key(tmp_path):
    status = await probe_model(make_settings(tmp_path, provider="openai_compat"))
    assert not status.ok
    assert "TC_OPENAI_API_KEY" in status.detail


async def test_unknown_provider_is_reported(tmp_path):
    status = await probe_model(make_settings(tmp_path, provider="carrier_pigeon"))
    assert not status.ok
    assert "provider desconhecido" in status.detail


def test_health_endpoint_degrades_when_model_absent(tmp_path, patched_client):
    """Serviço de pé mas modelo ausente é 503, não 200.

    Um health verde com modelo faltando mandaria o dev procurar o problema no
    lugar errado.
    """
    patched_client({"models": []})
    with TestClient(create_app(make_settings(tmp_path), provider=FakeProvider())) as client:
        response = client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["database"]["ok"] is True
    assert body["database"]["migrations_applied"] == ["001_initial.sql"]
    assert body["model"]["ok"] is False


def test_health_endpoint_ok_when_everything_ready(tmp_path, patched_client):
    patched_client({"models": [{"name": "qwen3:4b"}]})
    with TestClient(create_app(make_settings(tmp_path), provider=FakeProvider())) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    # A base contrastiva da E3 é conteúdo do produto: o /health conta os itens.
    assert body["contrastive"]["ok"] is True
    assert body["contrastive"]["items"] >= 40


def test_health_endpoint_degrades_when_contrastive_base_is_broken(
    tmp_path, patched_client, monkeypatch
):
    """Base contrastiva que não carrega é 503.

    Sem ela o professor volta a improvisar gramática — o problema nº 1 da
    análise. Descobrir isso no /health é melhor que no primeiro turno do aluno.
    """
    patched_client({"models": [{"name": "qwen3:4b"}]})

    def quebrada(_directory):
        raise ContrastiveError("id repetido: calque.age-have")

    monkeypatch.setattr("teacher_charles.app.load_base", quebrada)

    with TestClient(create_app(make_settings(tmp_path), provider=FakeProvider())) as client:
        response = client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model"]["ok"] is True
    assert body["contrastive"]["ok"] is False
    assert "calque.age-have" in body["contrastive"]["detail"]


def test_index_page_renders(tmp_path, patched_client):
    patched_client({"models": [{"name": "qwen3:4b"}]})
    with TestClient(create_app(make_settings(tmp_path), provider=FakeProvider())) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Teacher Charles" in response.text
