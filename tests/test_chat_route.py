"""Testes das rotas de conversa (E2)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from teacher_charles.app import create_app
from teacher_charles.config import Settings
from teacher_charles.llm import FakeProvider, ProviderUnavailable


def settings_for(tmp_path, **overrides) -> Settings:
    # llm_warmup desligado: o aquecimento consumiria uma resposta roteirizada
    # do FakeProvider e apareceria em provider.calls, bagunçando as asserções
    # sobre o que o turno mandou ao modelo. Ele é testado à parte, abaixo.
    base = {"db_path": tmp_path / "route.db", "provider": "fake", "llm_warmup": False}
    base.update(overrides)
    return Settings(_env_file=None, **base)


@pytest.fixture
def client_and_provider(tmp_path):
    provider = FakeProvider(replies="I am 30 years old. What did you do?", chunk_size=6)
    with TestClient(create_app(settings_for(tmp_path), provider=provider)) as client:
        yield client, provider


def frames_from(response) -> list[dict]:
    """Decodifica o corpo SSE em eventos."""
    out = []
    for block in response.text.split("\n\n"):
        block = block.strip()
        if block.startswith("data:"):
            out.append(json.loads(block[5:].strip()))
    return out


# --- /api/chat ------------------------------------------------------------- #


def test_chat_streams_chunks_then_done(client_and_provider):
    client, _ = client_and_provider
    response = client.post("/api/chat", json={"message": "I have 30 years"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    frames = frames_from(response)
    chunks = [f for f in frames if f["type"] == "chunk"]
    assert len(chunks) > 1, "esperava vários eventos de chunk"
    assert "".join(f["text"] for f in chunks) == "I am 30 years old. What did you do?"

    assert frames[-1]["type"] == "done"
    assert frames[-1]["first_chunk_ms"] >= 0
    assert frames[-1]["total_ms"] >= 0


def test_chat_does_not_buffer(client_and_provider):
    """Um proxy no caminho não deve poder segurar o corpo."""
    client, _ = client_and_provider
    response = client.post("/api/chat", json={"message": "hi"})
    assert response.headers["x-accel-buffering"] == "no"
    assert response.headers["cache-control"] == "no-store"


def test_chat_persists_the_turn(client_and_provider):
    client, _ = client_and_provider
    client.post("/api/chat", json={"message": "I go to gym yesterday"})

    turns = client.get("/api/history").json()["turns"]
    assert [(t["role"], t["content"]) for t in turns] == [
        ("student", "I go to gym yesterday"),
        ("teacher", "I am 30 years old. What did you do?"),
    ]


def test_chat_keeps_the_same_session_across_turns(client_and_provider):
    client, _ = client_and_provider
    client.post("/api/chat", json={"message": "one"})
    first = client.get("/api/history").json()["session_id"]
    client.post("/api/chat", json={"message": "two"})
    second = client.get("/api/history").json()["session_id"]

    assert first == second
    assert len(client.get("/api/history").json()["turns"]) == 4


def test_chat_sends_history_to_the_model(client_and_provider):
    client, provider = client_and_provider
    client.post("/api/chat", json={"message": "first"})
    client.post("/api/chat", json={"message": "second"})

    roles = [(m.role, m.content) for m in provider.last_call.messages]
    assert roles[0][0] == "system"
    assert ("user", "first") in roles
    assert ("user", "second") in roles


def test_empty_message_is_rejected_by_validation(client_and_provider):
    client, _ = client_and_provider
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
    assert client.post("/api/chat", json={}).status_code == 422


def test_whitespace_only_message_reports_an_error_frame(client_and_provider):
    """Passa a validação de tamanho, mas o runner recusa — e o cliente tem que saber."""
    client, _ = client_and_provider
    response = client.post("/api/chat", json={"message": "   "})

    frames = frames_from(response)
    assert frames[-1]["type"] == "error"
    assert client.get("/api/history").json()["turns"] == []


def test_overlong_message_is_rejected(client_and_provider):
    client, _ = client_and_provider
    assert client.post("/api/chat", json={"message": "x" * 4001}).status_code == 422


def test_provider_failure_becomes_an_error_frame(tmp_path):
    """Falha do modelo não deve virar 500: o aluno precisa ver o motivo."""
    provider = FakeProvider(error=ProviderUnavailable("Ollama não está de pé"))
    with TestClient(create_app(settings_for(tmp_path), provider=provider)) as client:
        response = client.post("/api/chat", json={"message": "hello"})

        assert response.status_code == 200
        frames = frames_from(response)
        assert frames[-1]["type"] == "error"
        assert "Ollama" in frames[-1]["message"]

        # A fala do aluno fica gravada mesmo com a geração falhando.
        turns = client.get("/api/history").json()["turns"]
        assert [t["role"] for t in turns] == ["student"]


def test_non_ascii_survives_the_round_trip(client_and_provider):
    client, _ = client_and_provider
    response = client.post("/api/chat", json={"message": "Eu não entendi — explique"})
    assert response.status_code == 200
    assert client.get("/api/history").json()["turns"][0]["content"] == "Eu não entendi — explique"


# --- /api/reset ------------------------------------------------------------ #


def test_reset_starts_a_new_session_and_keeps_the_old_turns(client_and_provider):
    client, _ = client_and_provider
    client.post("/api/chat", json={"message": "before reset"})
    old_session = client.get("/api/history").json()["session_id"]

    new_session = client.post("/api/reset").json()["session_id"]
    assert new_session != old_session

    # A sessão nova está vazia...
    assert client.get("/api/history").json()["turns"] == []
    # ...mas o histórico antigo continua no banco, que é o material da E7.
    rows = client.app.state.db.execute("SELECT count(*) AS n FROM turn").fetchone()
    assert rows["n"] == 2


# --- A página -------------------------------------------------------------- #


def test_index_renders_the_chat(client_and_provider):
    client, _ = client_and_provider
    response = client.get("/")

    assert response.status_code == 200
    assert "Teacher Charles" in response.text
    assert 'id="composer"' in response.text
    assert "/static/app.js" in response.text


def test_index_shows_the_empty_state_first(client_and_provider):
    client, _ = client_and_provider
    assert "Tell me about your day" in client.get("/").text


def test_index_replays_history_after_a_reload(client_and_provider):
    client, _ = client_and_provider
    client.post("/api/chat", json={"message": "I have 30 years"})

    page = client.get("/").text
    assert "I have 30 years" in page
    assert "I am 30 years old." in page
    assert "Tell me about your day" not in page


def test_index_escapes_student_text(client_and_provider):
    """O texto do aluno vai para o HTML; um script não deve executar."""
    client, _ = client_and_provider
    client.post("/api/chat", json={"message": "<script>alert(1)</script>"})

    page = client.get("/").text
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_static_files_are_served(client_and_provider):
    client, _ = client_and_provider
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


# --- Aquecimento do modelo ------------------------------------------------- #


def test_warmup_loads_the_model_on_startup(tmp_path):
    """Sem isto, os ~15s de carga do modelo caem no primeiro turno do aluno."""
    provider = FakeProvider(replies="ok")
    settings = settings_for(tmp_path, llm_warmup=True)

    with TestClient(create_app(settings, provider=provider)) as client:
        client.get("/health")  # dá uma volta no event loop para a task rodar
        assert provider.call_count >= 1, "o aquecimento não chamou o modelo"
        warm = provider.calls[0]
        # Tem que gerar o mínimo: a intenção é carregar, não conversar.
        assert warm.options.max_tokens == 1
        # E tem que usar a persona: aquecer com outro prompt carrega o modelo
        # mas deixa o prefixo fora do cache, e aí o primeiro turno do aluno
        # paga a avaliação inteira da persona (medido: 19s).
        assert warm.messages[0].role == "system"
        assert "Charles" in warm.messages[0].content


def test_warmup_can_be_disabled(tmp_path):
    provider = FakeProvider(replies="ok")
    with TestClient(create_app(settings_for(tmp_path), provider=provider)):
        pass
    assert provider.call_count == 0


def test_warmup_failure_does_not_break_startup(tmp_path):
    """Modelo ausente é assunto do /health, não motivo para a subida falhar."""
    provider = FakeProvider(error=ProviderUnavailable("sem Ollama"))
    settings = settings_for(tmp_path, llm_warmup=True)

    with TestClient(create_app(settings, provider=provider)) as client:
        assert client.get("/api/history").status_code == 200
