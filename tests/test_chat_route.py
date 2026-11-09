"""Testes das rotas de conversa (E2)."""

from __future__ import annotations

import json
import signal

import pytest
from fastapi.testclient import TestClient

from teacher_charles.app import create_app, request_stop
from teacher_charles.config import Settings
from teacher_charles.llm import FakeProvider, ProviderUnavailable
from teacher_charles.prompts import PromptLibrary


def settings_for(tmp_path, **overrides) -> Settings:
    # llm_warmup desligado: o aquecimento consumiria uma resposta roteirizada
    # do FakeProvider e apareceria em provider.calls, bagunçando as asserções
    # sobre o que o turno mandou ao modelo. Ele é testado à parte, abaixo.
    base = {"db_path": tmp_path / "route.db", "provider": "fake", "llm_warmup": False}
    base.update(overrides)
    return Settings(_env_file=None, **base)


# A etapa de análise do turno duplo passa pelo mesmo provider que a conversa.
# Roteirizada pelo system prompt, não por ordem de chamada: as duas rodam em
# paralelo e a ordem não é contrato.
ANALYSIS_JSON = json.dumps(
    {
        "errors": [
            {
                "span": "I have 30 years",
                "type": "calque",
                "correction": "I am 30 years old",
                "rule_id": "calque.age-have",
                "explanation": "ignorada: a regra curada vence",
            }
        ]
    }
)
NO_ERRORS_JSON = '{"errors": []}'


@pytest.fixture
def client_and_provider(tmp_path):
    provider = FakeProvider(
        replies="I am 30 years old. What did you do?",
        chunk_size=6,
        replies_by_system={"You find mistakes": ANALYSIS_JSON},
    )
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

    # A etapa da conversa, não a da análise: desde a E4 as duas passam pelo
    # mesmo provider, e é a persona que distingue uma da outra.
    reply_call = provider.calls_matching("Charles")[-1]
    roles = [(m.role, m.content) for m in reply_call.messages]
    assert roles[0][0] == "system"
    assert ("user", "first") in roles
    assert ("user", "second") in roles


def test_chat_emits_the_analysis_before_done(client_and_provider):
    """A correção é frame próprio, não texto do professor."""
    client, _ = client_and_provider
    response = client.post("/api/chat", json={"message": "I have 30 years"})

    frames = frames_from(response)
    kinds = [f["type"] for f in frames]
    assert "analysis" in kinds
    assert kinds.index("analysis") < kinds.index("done")

    analysis = next(f for f in frames if f["type"] == "analysis")
    (error,) = analysis["errors"]
    assert error["span"] == "I have 30 years"
    assert error["correction"] == "I am 30 years old"
    assert error["rule_id"] == "calque.age-have"
    # A explicação é a da base curada, não a que o modelo mandou.
    assert "age is something you are" in error["explanation"]
    assert analysis["degraded"] is False


def test_chat_persists_the_correction(client_and_provider):
    """Sem linha no banco não há error log (E7) nem drill (E6)."""
    client, _ = client_and_provider
    client.post("/api/chat", json={"message": "I have 30 years"})

    rows = client.app.state.db.execute("SELECT * FROM correction").fetchall()
    assert len(rows) == 1
    assert rows[0]["rule_id"] == "calque.age-have"
    assert rows[0]["span"] == "I have 30 years"
    assert rows[0]["error_type"] == "calque"

    # E aponta para o turno do ALUNO, que é o que contém o erro.
    turn = client.app.state.db.execute(
        "SELECT role FROM turn WHERE id = ?", (rows[0]["turn_id"],)
    ).fetchone()
    assert turn["role"] == "student"


def test_history_carries_the_corrections(client_and_provider):
    client, _ = client_and_provider
    client.post("/api/chat", json={"message": "I have 30 years"})

    turns = client.get("/api/history").json()["turns"]
    student = next(t for t in turns if t["role"] == "student")
    assert student["corrections"][0]["rule_id"] == "calque.age-have"
    assert next(t for t in turns if t["role"] == "teacher")["corrections"] == []


def test_the_reply_prompt_carries_the_curated_rule(client_and_provider):
    """A dica determinística é o que impede o 4B de inventar a regra.

    Ela chega no fim do prompt, dentro da fala do aluno — como mensagem de
    sistema o template a levava para o topo, e com ela ia o cache do histórico
    (plano §5.5).
    """
    client, provider = client_and_provider
    client.post("/api/chat", json={"message": "I have 30 years"})

    reply_call = provider.calls_matching("Charles")[-1]
    last = reply_call.messages[-1]
    assert last.role == "user"
    assert last.content.startswith("I have 30 years")
    assert "age is something you are" in last.content
    assert reply_call.messages[0].role == "system"
    assert "Charles" in reply_call.messages[0].content


def test_clean_english_produces_no_correction(tmp_path):
    provider = FakeProvider(
        replies="Nice! Tell me more.",
        replies_by_system={"You find mistakes": NO_ERRORS_JSON},
    )
    with TestClient(create_app(settings_for(tmp_path), provider=provider)) as client:
        response = client.post("/api/chat", json={"message": "I am 30 years old"})

        analysis = next(f for f in frames_from(response) if f["type"] == "analysis")
        assert analysis["errors"] == []
        assert (
            client.app.state.db.execute("SELECT count(*) AS n FROM correction").fetchone()["n"] == 0
        )

        # Sem casamento na base, nada de dica no prompt da resposta.
        reply_call = provider.calls_matching("Charles")[-1]
        assert reply_call.messages[-1].role == "user"


def test_broken_analysis_json_does_not_break_the_turn(tmp_path):
    """Degradar é a escolha: o turno do aluno vale mais que a correção dele."""
    provider = FakeProvider(
        replies="Tell me about your birthday.",
        replies_by_system={"You find mistakes": "Sure! Here are the errors I found:"},
    )
    with TestClient(create_app(settings_for(tmp_path), provider=provider)) as client:
        response = client.post("/api/chat", json={"message": "I have 30 years"})

        frames = frames_from(response)
        assert frames[-1]["type"] == "done"
        analysis = next(f for f in frames if f["type"] == "analysis")
        assert analysis["degraded"] is True
        assert analysis["errors"] == []
        # A conversa saiu inteira, apesar da análise ter falhado.
        assert "".join(f["text"] for f in frames if f["type"] == "chunk") == (
            "Tell me about your birthday."
        )
        # E a dica curada ainda foi para o prompt da resposta: a regra
        # determinística não depende da análise ter dado certo.
        reply_call = provider.calls_matching("Charles")[-1]
        assert "age is something you are" in reply_call.messages[-1].content


def test_index_shows_past_corrections(client_and_provider):
    client, _ = client_and_provider
    client.post("/api/chat", json={"message": "I have 30 years"})

    page = client.get("/").text
    assert "calque.age-have" in page
    assert "age is something you are" in page


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


# --- /api/shutdown --------------------------------------------------------- #


def test_shutdown_answers_and_asks_the_process_to_stop(client_and_provider):
    client, _ = client_and_provider
    asked = []
    # A rota lê ``app.state.stop`` a cada chamada justamente para isto: o teste
    # troca o sinal por um registro, em vez de matar o pytest.
    client.app.state.stop = lambda: asked.append("stop")

    response = client.post("/api/shutdown")

    assert response.status_code == 200
    assert response.json() == {"stopping": True}
    # A parada é um BackgroundTask: ela roda depois do corpo ter ido para o
    # socket. Parar antes faria o navegador ver conexão cortada no lugar da
    # despedida — mesmo clique, cara de queda.
    assert asked == ["stop"]


def test_the_default_stop_is_the_signal_of_ctrl_c():
    """Sinal, e não ``os._exit``: o ``finally`` do lifespan fecha o banco."""
    seen = []
    previous = signal.signal(signal.SIGINT, lambda *_: seen.append("sigint"))
    try:
        request_stop()
    finally:
        signal.signal(signal.SIGINT, previous)

    assert seen == ["sigint"]


def test_shutdown_refuses_a_request_from_another_page(client_and_provider):
    client, _ = client_and_provider
    asked = []
    client.app.state.stop = lambda: asked.append("stop")

    response = client.post("/api/shutdown", headers={"Origin": "http://evil.example"})

    assert response.status_code == 403
    # O servidor não tem autenticação nenhuma, e qualquer aba aberta consegue
    # fazer POST entre sites: sem este teste, uma página qualquer desligaria o
    # Charles do aluno no meio da aula.
    assert asked == []


def test_shutdown_accepts_the_request_from_its_own_page(client_and_provider):
    client, _ = client_and_provider
    asked = []
    client.app.state.stop = lambda: asked.append("stop")

    response = client.post("/api/shutdown", headers={"Origin": "http://testserver"})

    assert response.status_code == 200
    assert asked == ["stop"]


def test_index_renders_the_shutdown_button(client_and_provider):
    client, _ = client_and_provider
    page = client.get("/").text

    assert 'id="shutdown"' in page
    assert "Shut down" in page


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
    page = client.get("/").text
    assert "empty-state" in page
    assert "Tell me about your day" in page


def test_index_replays_history_after_a_reload(client_and_provider):
    """Com conversa na mesa, a bolha de abertura sai da tela.

    A asserção é sobre a bolha (``empty-state``), não sobre a frase: desde que
    a saudação ganhou IPA ela viaja também no ``data-greeting`` do ``<main>``,
    porque a conversa nova precisa dela com a transcrição e o ``app.js`` não
    tem espeak para gerá-la. A frase estar no HTML não é a mesma coisa que ela
    estar desenhada.
    """
    client, _ = client_and_provider
    client.post("/api/chat", json={"message": "I have 30 years"})

    page = client.get("/").text
    assert "I have 30 years" in page
    assert "I am 30 years old." in page
    assert "empty-state" not in page


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


def test_warmup_sends_exactly_the_prefix_the_turn_sends(tmp_path):
    """O aquecimento não pode envelhecer em relação ao turno.

    Já envelheceu: a E5 e a E6 acrescentaram nível e modo ao prompt, o
    aquecimento continuou mandando só a persona, e o primeiro turno chegava com
    53% do prompt em cache e 11,6s até a primeira palavra (plano §5.5). Agora
    os dois saem de ``TurnRunner.stable_prefix``, e este teste é o que impede a
    lista de voltar a ser duas.
    """
    provider = FakeProvider(replies="ok")
    app = create_app(settings_for(tmp_path, llm_warmup=True), provider=provider)

    with TestClient(app) as client:
        client.get("/health")
        warm = provider.calls[0]
        expected = app.state.runner.stable_prefix()

    assert [m.content for m in warm.messages[:-1]] == [m.content for m in expected]
    assert len(expected) == 4, (
        "persona, capacidades, nível e modo — o prefixo estável de um turno real"
    )
    assert warm.messages[-1].role == "user"


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


# --- Perfil e nível (E5) -------------------------------------------------- #


def test_profile_starts_at_the_entry_level(client_and_provider):
    client, _ = client_and_provider
    body = client.get("/api/profile").json()

    assert body["level"] == "B1"
    assert body["self_declared_level"] == "B1"
    assert body["level_source"] == "self"
    assert body["levels"] == ["A2", "B1", "B2", "C1", "C2"]
    assert body["history"] == []


def test_student_can_declare_the_level(client_and_provider):
    client, _ = client_and_provider
    body = client.post("/api/profile", json={"level": "b2"}).json()

    assert body["level"] == "B2"
    assert body["self_declared_level"] == "B2", "o aluno redefine o ponto de partida"
    assert body["level_source"] == "self"
    assert body["history"][0] == {
        **body["history"][0],
        "from": "B1",
        "to": "B2",
        "source": "self",
    }


def test_invalid_level_is_refused(client_and_provider):
    client, _ = client_and_provider
    response = client.post("/api/profile", json={"level": "Z9"})
    assert response.status_code == 422
    assert client.get("/api/profile").json()["level"] == "B1"


def test_naming_a_new_student_switches_profile(client_and_provider):
    """Vários alunos numa máquina, sem login (plano §4)."""
    client, _ = client_and_provider
    client.post("/api/profile", json={"level": "C1"})

    other = client.post("/api/profile", json={"name": "ana"}).json()
    assert other["name"] == "ana"
    assert other["level"] == "B1", "perfil novo começa no nível de entrada"

    # E o perfil ativo continua sendo o último escolhido.
    assert client.get("/api/profile").json()["name"] == "ana"

    back = client.post("/api/profile", json={"name": "student"}).json()
    assert back["level"] == "C1", "o perfil antigo manteve o nível dele"


def test_the_level_in_force_reaches_the_prompt(client_and_provider):
    client, provider = client_and_provider
    client.post("/api/profile", json={"level": "A2"})
    client.post("/api/chat", json={"message": "I work with marketing"})

    conversation = [c for c in provider.calls if "You find mistakes" not in c.messages[0].content]
    system = "\n".join(m.content for m in conversation[-1].messages if m.role == "system")
    assert "A2" in system


def test_chat_reports_the_level_in_every_turn(client_and_provider):
    client, _ = client_and_provider
    response = client.post("/api/chat", json={"message": "I have 30 years"})
    frames = frames_from(response)

    level = [f for f in frames if f["type"] == "level"]
    assert len(level) == 1
    assert level[0]["level"] == "B1"
    assert level[0]["changed"] is False
    assert level[0]["counted"] is False, "quatro palavras não são evidência de nível"
    # O frame de nível vem antes do done, que fecha o turno.
    assert frames[-1]["type"] == "done"


def test_the_level_moves_after_enough_turns(client_and_provider):
    client, _ = client_and_provider
    text = (
        "Although I have worked in logistics for almost ten years, I would rather "
        "understand how the current process works, which nobody has documented, "
        "before suggesting any change to the team."
    )

    moved = []
    for _ in range(4):
        frames = frames_from(client.post("/api/chat", json={"message": text}))
        moved += [f for f in frames if f["type"] == "level" and f["changed"]]

    assert [(f["previous"], f["level"]) for f in moved] == [("B1", "B2")]

    profile = client.get("/api/profile").json()
    assert profile["level"] == "B2"
    assert profile["level_source"] == "auto"
    assert profile["self_declared_level"] == "B1"
    assert profile["history"][0]["to"] == "B2"


def test_index_renders_the_profile(client_and_provider):
    client, _ = client_and_provider
    client.post("/api/profile", json={"level": "C1"})
    page = client.get("/").text

    assert 'id="level"' in page
    assert '<option value="C1" selected>' in page
    assert 'id="student-name"' in page


def test_health_reports_missing_level_partials(tmp_path):
    """Partial que falta tira metade da E5 em silêncio; o /health conta."""
    empty = tmp_path / "prompts"
    empty.mkdir()
    (empty / "persona.md").write_text("You are Charles.", encoding="utf-8")
    (empty / "analysis.md").write_text("You find mistakes.", encoding="utf-8")

    app = create_app(settings_for(tmp_path), provider=FakeProvider())
    with TestClient(app) as client:
        # Troca a biblioteca de prompts por uma sem partial de nível, que é o
        # estado que o teste descreve: o repo de verdade tem os cinco.
        app.state.prompts = PromptLibrary(empty)
        body = client.get("/health").json()

    assert body["level_partials"] == {
        "ok": False,
        "missing": ["A2", "B1", "B2", "C1", "C2"],
    }
    # Mas o serviço não fica degradado por isso: o turno ainda funciona sem o
    # ajuste de registro, como na E2.
    assert body["database"]["ok"] is True


def test_declaring_a_level_survives_the_next_turn(client_and_provider):
    """O aluno tem a última palavra, e ela não pode ser desfeita no turno seguinte."""
    client, _ = client_and_provider
    text = (
        "Although I have worked in logistics for almost ten years, I would rather "
        "understand how the current process works, which nobody has documented, "
        "before suggesting any change to the team."
    )
    for _ in range(3):
        client.post("/api/chat", json={"message": text})

    assert client.post("/api/profile", json={"level": "A2"}).json()["level"] == "A2"

    frames = frames_from(client.post("/api/chat", json={"message": text}))
    level = [f for f in frames if f["type"] == "level"][0]
    assert level["changed"] is False
    assert client.get("/api/profile").json()["level"] == "A2"


# --- Os três modos (E6) --------------------------------------------------- #


def test_mode_starts_as_conversation(client_and_provider):
    client, _ = client_and_provider
    body = client.get("/api/mode").json()

    assert body["mode"] == "conversation"
    assert body["modes"] == ["conversation", "correction", "drill"]
    assert body["drill"] is None
    assert body["drillable"] == 0


def test_switching_mode_opens_a_new_session(client_and_provider):
    """Trocar de modo é trocar de atividade: a transcrição não continua."""
    client, _ = client_and_provider
    first = client.post("/api/chat", json={"message": "I have 30 years"})
    assert first.status_code == 200
    before = client.get("/api/history").json()
    assert before["turns"]

    switched = client.post("/api/mode", json={"mode": "correction"}).json()
    assert switched["mode"] == "correction"
    assert switched["session_id"] != before["session_id"]

    after = client.get("/api/history").json()
    assert after["turns"] == [], "a conversa anterior não é a transcrição do novo modo"


def test_invalid_mode_is_refused(client_and_provider):
    client, _ = client_and_provider
    response = client.post("/api/mode", json={"mode": "telepathy"})
    assert response.status_code == 422
    assert client.get("/api/mode").json()["mode"] == "conversation"


def test_the_mode_prompt_reaches_the_model(client_and_provider):
    client, provider = client_and_provider
    client.post("/api/mode", json={"mode": "correction"})
    client.post("/api/chat", json={"message": "I have 30 years"})

    conversation = [c for c in provider.calls if "You find mistakes" not in c.messages[0].content]
    system = "\n".join(m.content for m in conversation[-1].messages if m.role == "system")
    assert "Mode: correction" in system
    assert "Mode: free conversation" not in system


def test_drill_uses_a_mistake_the_student_really_made(client_and_provider):
    """O critério de pronto da E6, de ponta a ponta pela rota.

    O turno com *I have 30 years* produz correção com `calque.age-have`; o modo
    drill, aberto depois, tem que treinar **essa** regra, com as palavras do
    aluno no prompt.
    """
    client, provider = client_and_provider
    client.post("/api/chat", json={"message": "I have 30 years"})

    body = client.post("/api/mode", json={"mode": "drill"}).json()
    assert body["drillable"] == 1
    assert body["drill"]["rule_id"] == "calque.age-have"
    assert body["drill"]["examples"] == ["I have 30 years"]

    client.post("/api/chat", json={"message": "I am 30 years old"})
    conversation = [c for c in provider.calls if "You find mistakes" not in c.messages[0].content]
    system = "\n".join(m.content for m in conversation[-1].messages if m.role == "system")
    assert "Mode: drill" in system
    assert "calque.age-have" in system
    assert "I have 30 years" in system


def test_drill_without_an_error_log_is_honest(client_and_provider):
    client, provider = client_and_provider
    body = client.post("/api/mode", json={"mode": "drill"}).json()

    assert body["drillable"] == 0
    assert body["drill"] is None

    client.post("/api/chat", json={"message": "hello there"})
    conversation = [c for c in provider.calls if "You find mistakes" not in c.messages[0].content]
    system = "\n".join(m.content for m in conversation[-1].messages if m.role == "system")
    assert "no rule to drill" in system.lower()


def test_new_conversation_keeps_the_mode(client_and_provider):
    """Quem clica em "new conversation" quer outro assunto, não sair do drill."""
    client, _ = client_and_provider
    client.post("/api/chat", json={"message": "I have 30 years"})
    client.post("/api/mode", json={"mode": "drill"})

    body = client.post("/api/reset").json()
    assert body["mode"] == "drill"
    assert body["drill"]["rule_id"] == "calque.age-have"


def test_index_renders_the_modes(client_and_provider):
    client, _ = client_and_provider
    client.post("/api/chat", json={"message": "I have 30 years"})
    client.post("/api/mode", json={"mode": "drill"})
    page = client.get("/").text

    assert 'data-mode="drill"' in page
    assert 'class="mode active"' in page
    assert "calque.age-have" in page, "o aluno tem que ver qual erro dele virou exercício"


# --- A tela do error log (E7) --------------------------------------------- #


def test_progress_is_empty_for_a_new_student(client_and_provider):
    client, _ = client_and_provider
    body = client.get("/api/progress").json()

    assert body["rules"] == []
    assert body["totals"] == {"corrections": 0, "rules": 0, "without_rule": 0}
    assert "Nothing here yet" in client.get("/progress").text


def test_progress_aggregates_by_rule(client_and_provider):
    """O que a E4 prometeu: correção é linha contável, e aqui ela vira resposta."""
    client, _ = client_and_provider
    for _ in range(2):
        client.post("/api/chat", json={"message": "I have 30 years"})

    body = client.get("/api/progress").json()
    assert body["totals"]["corrections"] == 2
    assert body["totals"]["rules"] == 1

    rule = body["rules"][0]
    assert rule["rule_id"] == "calque.age-have"
    assert rule["times"] == 2
    assert rule["examples"] == ["I have 30 years"], "as frases são as do próprio aluno"
    # A explicação vem da base curada, não do que o modelo disse no turno.
    assert "age is something you are" in rule["explanation"]
    assert rule["right"], "a tela mostra a forma certa, e ela vem da base"
    assert rule["retired"] is False


def test_progress_page_shows_the_students_own_words(client_and_provider):
    client, _ = client_and_provider
    client.post("/api/chat", json={"message": "I have 30 years"})

    page = client.get("/progress").text
    assert "I have 30 years" in page
    assert "calque.age-have" in page
    assert "1×" in page


def test_progress_is_per_student(client_and_provider):
    client, _ = client_and_provider
    client.post("/api/chat", json={"message": "I have 30 years"})
    client.post("/api/profile", json={"name": "ana"})

    assert client.get("/api/progress").json()["rules"] == []
    client.post("/api/profile", json={"name": "student"})
    assert client.get("/api/progress").json()["rules"][0]["times"] == 1


def test_progress_shows_the_level_history(client_and_provider):
    client, _ = client_and_provider
    client.post("/api/profile", json={"level": "B2"})

    body = client.get("/api/progress").json()
    assert body["level"] == "B2"
    assert body["level_history"][0]["to"] == "B2"
    assert "B1 → B2" in client.get("/progress").text


def test_the_conversation_links_to_the_error_log(client_and_provider):
    client, _ = client_and_provider
    assert 'href="/progress"' in client.get("/").text


# --- As capacidades do aplicativo chegam ao modelo ------------------------- #


@pytest.mark.parametrize(
    ("voice_enabled", "present", "absent"),
    [(True, "Hold to talk", "no microphone"), (False, "no microphone", "Hold to talk")],
)
def test_the_turn_tells_the_model_what_the_app_can_do(tmp_path, voice_enabled, present, absent):
    """A rota monta o runner; é ela que tem o flag da voz.

    Numa sessão real o Charles respondeu que não tinha microfone com o
    push-to-talk na tela do aluno, porque nada no prompt falava do produto. O
    flag só chegava ao ``config.py`` e ao ``voice/``.
    """
    provider = FakeProvider(replies="Yes — hold the button and talk to me.")
    settings = settings_for(tmp_path, voice_enabled=voice_enabled)

    with TestClient(create_app(settings, provider=provider)) as client:
        response = client.post("/api/chat", json={"message": "do you have a microphone?"})
        assert response.status_code == 200

    system = " ".join(provider.calls[0].messages[0].content.split())
    prompt = " ".join(provider.system_prompt_of_last_call().split())
    assert present in prompt
    assert absent not in prompt
    # E nada disso entrou na persona, que é o prefixo cacheado.
    assert present not in system


def test_the_warmup_carries_the_capabilities_too(tmp_path):
    """Mesma lição da §5.5: o aquecimento não pode divergir do turno.

    Um prefixo com as capacidades no turno e sem elas no aquecimento faria o
    primeiro turno do aluno reavaliar tudo o que vem depois da persona.
    """
    provider = FakeProvider(replies="ok")
    settings = settings_for(tmp_path, llm_warmup=True, voice_enabled=True)
    app = create_app(settings, provider=provider)

    with TestClient(app) as client:
        client.get("/health")
        warm = provider.calls[0]
        expected = app.state.runner.stable_prefix()

    assert [m.content for m in warm.messages[:-1]] == [m.content for m in expected]
    assert "Hold to talk" in " ".join(expected[1].content.split())


# --- A pronúncia na página ------------------------------------------------- #


def test_chat_envia_fonetica_dos_dois_lados(client_and_provider):
    client, _ = client_and_provider
    frames = frames_from(client.post("/api/chat", json={"message": "I have 30 years"}))

    spelled = [f for f in frames if f["type"] == "phonetics"]
    assert [f["role"] for f in spelled] == ["student", "teacher"]
    assert all(f["ipa"] for f in spelled)


def test_a_fonetica_do_aluno_chega_antes_do_texto_do_professor(client_and_provider):
    """A bolha do aluno não espera o professor para ganhar a linha dela."""
    client, _ = client_and_provider
    frames = frames_from(client.post("/api/chat", json={"message": "I have 30 years"}))
    types = [f["type"] for f in frames]

    assert types.index("phonetics") < types.index("chunk")


def test_sem_fonetica_nao_ha_quadro_nenhum(tmp_path):
    provider = FakeProvider(
        replies="Nice.",
        replies_by_system={"You find mistakes": NO_ERRORS_JSON},
    )
    settings = settings_for(tmp_path, phonetics_enabled=False)
    with TestClient(create_app(settings, provider=provider)) as client:
        frames = frames_from(client.post("/api/chat", json={"message": "Hello"}))

    assert not [f for f in frames if f["type"] == "phonetics"]


def test_a_pagina_mostra_a_saudacao_com_a_pronuncia(client_and_provider):
    client, _ = client_and_provider
    page = client.get("/").text

    assert 'class="ipa"' in page
    assert 'id="show-ipa"' in page


def test_o_historico_volta_com_a_pronuncia(client_and_provider):
    """O IPA não é guardado no banco: é função determinística do texto.

    Guardá-lo criaria uma coluna que pode discordar do que o espeak diz hoje —
    e a releitura da página é justamente onde isso apareceria.
    """
    client, _ = client_and_provider
    client.post("/api/chat", json={"message": "I have 30 years"})

    page = client.get("/").text
    assert page.count('class="ipa"') >= 2


def test_sem_fonetica_a_pagina_nao_desenha_linha_nem_interruptor(tmp_path):
    provider = FakeProvider(replies="Nice.")
    settings = settings_for(tmp_path, phonetics_enabled=False)
    with TestClient(create_app(settings, provider=provider)) as client:
        page = client.get("/").text

    assert 'class="ipa"' not in page
    assert 'id="show-ipa"' not in page


def test_health_reporta_a_fonetica(client_and_provider):
    client, _ = client_and_provider
    body = client.get("/health").json()

    assert body["phonetics"]["enabled"] is True
    assert body["phonetics"]["language"] == "en-us"
    assert body["phonetics"]["detail"] is None


def test_fonetica_quebrada_nao_deixa_o_health_vermelho(tmp_path, monkeypatch):
    """A aula funciona inteira sem a linha; banco, modelo e base é que não."""
    from teacher_charles.teaching.phonetics import Phonetics, PhoneticsUnavailable

    monkeypatch.setattr(
        Phonetics,
        "_load",
        lambda self: (_ for _ in ()).throw(PhoneticsUnavailable("espeak sumiu")),
    )

    provider = FakeProvider(replies="Nice.")
    with TestClient(create_app(settings_for(tmp_path), provider=provider)) as client:
        client.post("/api/chat", json={"message": "Hello"})
        body = client.get("/health").json()

    assert body["phonetics"]["ok"] is False
    assert "espeak sumiu" in body["phonetics"]["detail"]
    # O motivo aparece, mas não entra na conta do status: só banco, modelo,
    # base contrastiva e voz pedida decidem "pronto para ensinar".
    assert body["status"] == "ok"
