"""As rotas de voz e os quadros ``sentence`` no stream do turno (Fase 1.5)."""

from __future__ import annotations

import wave
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from teacher_charles.app import create_app
from teacher_charles.config import Settings
from teacher_charles.llm import FakeProvider
from teacher_charles.voice import FakeSTT, FakeTTS, Voice, VoiceBadAudio


def settings_for(tmp_path, **overrides) -> Settings:
    # llm_warmup desligado pelo mesmo motivo do test_chat_route: o aquecimento
    # consumiria uma resposta roteirizada e apareceria em provider.calls.
    base = {
        "db_path": tmp_path / "voice.db",
        "provider": "fake",
        "llm_warmup": False,
        "voice_enabled": True,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def frames_from(response) -> list[dict]:
    out = []
    for block in response.text.split("\n\n"):
        block = block.strip()
        if block.startswith("data:"):
            import json

            out.append(json.loads(block[5:].strip()))
    return out


@pytest.fixture
def client(tmp_path):
    with TestClient(
        create_app(
            settings_for(tmp_path),
            provider=FakeProvider(replies="Good morning! How was your weekend?", chunk_size=5),
        )
    ) as handle:
        yield handle


@pytest.fixture
def mute_client(tmp_path):
    """Servidor com a voz desligada — o default do produto."""
    with TestClient(
        create_app(
            settings_for(tmp_path, voice_enabled=False),
            provider=FakeProvider(replies="Good morning! How was your weekend?"),
        )
    ) as handle:
        yield handle


class TestVozDesligada:
    """O chat por texto da Fase 1 segue inteiro, e as rotas de voz dizem por quê."""

    def test_a_pagina_nao_mostra_microfone(self, mute_client):
        body = mute_client.get("/").text
        assert 'id="talk"' not in body
        assert "voice.js" not in body

    def test_stt_responde_503_com_instrucao(self, mute_client):
        response = mute_client.post("/api/stt", content=b"x" * 2000)
        assert response.status_code == 503
        # 503 e não 404: a rota existe, o serviço não está pronto para ela.
        assert "TC_VOICE_ENABLED" in response.json()["how"]

    def test_tts_responde_503(self, mute_client):
        assert mute_client.post("/api/tts", json={"text": "hi there"}).status_code == 503

    def test_api_voice_diz_que_esta_desligada(self, mute_client):
        assert mute_client.get("/api/voice").json() == {"enabled": False}

    def test_o_turno_nao_ganha_quadro_de_sentenca(self, mute_client):
        frames = frames_from(mute_client.post("/api/chat", json={"message": "Hello."}))
        assert not [f for f in frames if f["type"] == "sentence"]
        # E o texto continua chegando: a voz é opcional, o chat não.
        assert "".join(f["text"] for f in frames if f["type"] == "chunk")

    def test_health_fica_ok_sem_voz(self, mute_client):
        body = mute_client.get("/health").json()
        assert body["voice"] == {"ok": True, "enabled": False}


class TestSTT:
    def test_devolve_o_texto_e_o_custo(self, client):
        response = client.post("/api/stt", content=b"x" * 4096)
        assert response.status_code == 200
        body = response.json()
        assert body["text"] == "I have 30 years."
        assert body["bytes"] == 4096
        assert "elapsed_ms" in body

    def test_corpo_vazio_e_422(self, client):
        response = client.post("/api/stt", content=b"")
        assert response.status_code == 422
        assert "áudio" in response.json()["detail"]

    def test_audio_ilegivel_vira_422_e_nao_500(self, tmp_path):
        # O aluno não deveria ver stack trace porque o microfone gravou lixo.
        voice = Voice(stt=FakeSTT(error=VoiceBadAudio("não decodifica")), tts=FakeTTS())
        app = create_app(settings_for(tmp_path), provider=FakeProvider())
        with TestClient(app) as handle:
            app.state.voice = voice
            response = handle.post("/api/stt", content=b"x" * 2000)
        assert response.status_code == 422
        assert "decodifica" in response.json()["detail"]

    def test_nao_grava_turno_nenhum(self, client):
        """A transcrição vai para o campo de texto, não para a conversa.

        É a salvaguarda contra o pior defeito possível deste produto: um erro
        de STT que vira turno faz o Charles corrigir um erro que o aluno não
        cometeu.
        """
        client.post("/api/stt", content=b"x" * 4096)
        assert client.get("/api/history").json()["turns"] == []


class TestTTS:
    def test_devolve_wav_tocavel(self, client):
        response = client.post("/api/tts", json={"text": "Good morning."})
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        with wave.open(BytesIO(response.content), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getnframes() > 0

    def test_cabecalhos_carregam_a_medicao(self, client):
        response = client.post("/api/tts", json={"text": "Good morning."})
        # Medir o turno de voz sem instrumentar o navegador.
        assert int(response.headers["X-Synthesis-Ms"]) >= 0
        assert float(response.headers["X-Audio-Seconds"]) > 0

    def test_texto_vazio_e_recusado_pelo_schema(self, client):
        assert client.post("/api/tts", json={"text": ""}).status_code == 422

    def test_paragrafo_e_recusado_pelo_schema(self, client):
        # O teto do schema é o mesmo do motor: uma chamada por sentença é o
        # desenho, não um detalhe.
        assert client.post("/api/tts", json={"text": "x" * 601}).status_code == 422

    def test_voz_pode_ser_escolhida_por_pedido(self, tmp_path):
        tts = FakeTTS(voice="bm_george")
        app = create_app(settings_for(tmp_path), provider=FakeProvider())
        with TestClient(app) as handle:
            app.state.voice = Voice(stt=FakeSTT(), tts=tts)
            handle.post("/api/tts", json={"text": "Good morning.", "voice": "af_heart"})
        assert tts.last_call.voice == "af_heart"


class TestQuadrosDeSentenca:
    def test_cada_sentenca_fechada_rende_um_quadro_indexado(self, client):
        frames = frames_from(client.post("/api/chat", json={"message": "Hello Charles."}))
        sentences = [f for f in frames if f["type"] == "sentence"]
        assert [f["text"] for f in sentences] == ["Good morning!", "How was your weekend?"]
        # O índice é o que permite a página tocar em ordem apesar de os
        # pedidos ao /api/tts resolverem fora de ordem.
        assert [f["index"] for f in sentences] == [0, 1]

    def test_o_quadro_da_sentenca_vem_depois_do_texto_dela(self, client):
        frames = frames_from(client.post("/api/chat", json={"message": "Hello Charles."}))
        kinds = [f["type"] for f in frames]
        first_sentence = kinds.index("sentence")
        # Pelo menos um pedaço de texto já passou: a letra na tela nunca espera
        # pelo pedido de áudio.
        assert "chunk" in kinds[:first_sentence]

    def test_o_texto_falado_e_a_soma_do_texto_escrito(self, client):
        frames = frames_from(client.post("/api/chat", json={"message": "Hello Charles."}))
        written = "".join(f["text"] for f in frames if f["type"] == "chunk")
        spoken = " ".join(f["text"] for f in frames if f["type"] == "sentence")
        assert spoken == written.strip()

    def test_o_rabo_sem_pontuacao_sai_antes_do_done(self, tmp_path):
        # O cliente trata `done` como "não vem mais áudio" e fecha a fila; um
        # quadro depois dele ficaria sem quem o tocasse.
        app = create_app(
            settings_for(tmp_path),
            provider=FakeProvider(replies="Tell me about your week", chunk_size=4),
        )
        with TestClient(app) as handle:
            frames = frames_from(handle.post("/api/chat", json={"message": "Hi."}))
        kinds = [f["type"] for f in frames]
        assert kinds.index("sentence") < kinds.index("done")
        assert [f["text"] for f in frames if f["type"] == "sentence"] == ["Tell me about your week"]

    def test_markdown_da_persona_nao_chega_ao_sintetizador(self, tmp_path):
        app = create_app(
            settings_for(tmp_path),
            provider=FakeProvider(replies="Age uses *to be* in English. Try again."),
        )
        with TestClient(app) as handle:
            frames = frames_from(handle.post("/api/chat", json={"message": "Hi."}))
        spoken = [f["text"] for f in frames if f["type"] == "sentence"]
        assert spoken == ["Age uses to be in English.", "Try again."]
        # E a tela continua recebendo o markdown: a limpeza é da borda da fala.
        written = "".join(f["text"] for f in frames if f["type"] == "chunk")
        assert "*to be*" in written

    def test_turno_que_falha_nao_rende_sentenca(self, tmp_path):
        from teacher_charles.llm import ProviderUnavailable

        app = create_app(
            settings_for(tmp_path),
            provider=FakeProvider(error=ProviderUnavailable("ollama fora")),
        )
        with TestClient(app) as handle:
            frames = frames_from(handle.post("/api/chat", json={"message": "Hi."}))
        assert not [f for f in frames if f["type"] == "sentence"]
        assert frames[-1]["type"] == "error"


class TestPaginaEHealth:
    def test_a_pagina_mostra_microfone_e_alto_falante(self, client):
        body = client.get("/").text
        assert 'id="talk"' in body
        assert 'id="speak"' in body
        assert "voice.js" in body

    def test_api_voice_e_a_pagina_concordam(self, client):
        body = client.get("/api/voice").json()
        assert body["enabled"] is True
        assert body["voice"] in client.get("/").text

    def test_health_reporta_os_dois_motores(self, client):
        body = client.get("/health").json()
        assert body["voice"]["enabled"] is True
        assert body["voice"]["stt"]["model"] == "fake-stt"
        assert body["voice"]["tts"]["voices"] == 2

    async def test_motor_que_nao_carregou_deixa_o_health_degradado(self, tmp_path):
        """Pedida e não carregada é degradado, e com o motivo.

        O aquecimento engole o erro de propósito; sem este caminho ele sumiria,
        e o dev veria a rota responder 422 sem saber que falta um peso.

        Os motores são injetados depois da subida, e não pedidos à
        configuração: pedir ``stt_model`` de verdade aqui faria um teste de
        rota baixar 150 MB do Hugging Face na primeira execução.
        """
        from pathlib import Path

        from teacher_charles.voice.kokoro import KokoroTTS
        from teacher_charles.voice.whisper import WhisperSTT

        broken = Voice(
            stt=WhisperSTT(model="tiny.en"),
            tts=KokoroTTS(
                model_path=Path("/nao/existe.onnx"),
                voices_path=Path("/nao/existe.bin"),
            ),
        )
        # Só o TTS: ele falha no disco, sem rede. O STT fica não-aquecido, que
        # é o outro jeito de não estar pronto.
        await broken.tts.warm_up()

        app = create_app(settings_for(tmp_path), provider=FakeProvider())
        with TestClient(app) as handle:
            app.state.voice = broken
            response = handle.get("/health")

        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "degraded"
        assert body["voice"]["ok"] is False
        assert body["voice"]["tts"]["ready"] is False
        assert "fetch_voice_models" in body["voice"]["tts"]["detail"]
        assert body["voice"]["stt"]["ready"] is False
