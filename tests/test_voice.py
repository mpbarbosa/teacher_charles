"""Contratos, dublês e fábrica da voz (Fase 1.5).

Sem peso de modelo. Os anteparos dos motores reais **são** testados aqui, e
isso não é contradição: eles disparam antes da carga preguiçosa, de propósito —
recusar um upload de 50 MB não deveria custar o carregamento de um modelo.
"""

from __future__ import annotations

import wave
from io import BytesIO
from pathlib import Path

import pytest

from teacher_charles.config import Settings
from teacher_charles.voice import (
    FakeSTT,
    FakeTTS,
    Speech,
    Transcript,
    Voice,
    VoiceError,
    build_voice,
)
from teacher_charles.voice.kokoro import MAX_SPEECH_CHARS, KokoroTTS
from teacher_charles.voice.whisper import MAX_AUDIO_BYTES, MIN_AUDIO_BYTES, WhisperSTT


def make_settings(**overrides) -> Settings:
    """Settings de teste, sem ler o ``.env`` do repositório."""
    return Settings(_env_file=None, **overrides)


class TestSpeech:
    def test_duracao_vem_do_pcm_16_bits(self):
        # Um segundo a 24 kHz são 24000 amostras de 2 bytes.
        speech = Speech(pcm=b"\x00\x00" * 24_000, sample_rate=24_000, elapsed_ms=500)
        assert speech.duration_s == pytest.approx(1.0)

    def test_fator_de_tempo_real_e_maquina_por_segundo_de_fala(self):
        # Meio segundo de máquina para um segundo de fala: RTF 0,5. É o número
        # que decide se a voz engasga entre sentenças.
        speech = Speech(pcm=b"\x00\x00" * 24_000, sample_rate=24_000, elapsed_ms=500)
        assert speech.real_time_factor == pytest.approx(0.5)

    def test_sem_audio_nao_ha_fator_de_tempo_real(self):
        assert Speech(pcm=b"", sample_rate=24_000, elapsed_ms=10).real_time_factor is None

    def test_as_wav_monta_cabecalho_que_o_navegador_le(self):
        speech = Speech(pcm=b"\x01\x02" * 100, sample_rate=24_000, elapsed_ms=1)
        data = speech.as_wav()
        assert data[:4] == b"RIFF"
        with wave.open(BytesIO(data), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == 24_000
            assert handle.getnframes() == 100

    def test_as_wav_nao_altera_as_amostras(self):
        pcm = bytes(range(256))
        speech = Speech(pcm=pcm, sample_rate=16_000, elapsed_ms=1)
        with wave.open(BytesIO(speech.as_wav()), "rb") as handle:
            assert handle.readframes(handle.getnframes()) == pcm


class TestTranscript:
    def test_fator_de_tempo_real(self):
        result = Transcript(text="hi", duration_s=4.0, elapsed_ms=1000)
        assert result.real_time_factor == pytest.approx(0.25)

    def test_audio_de_duracao_zero_nao_tem_fator(self):
        assert Transcript(text="", duration_s=0.0, elapsed_ms=0).real_time_factor is None

    def test_as_dict_arredonda_e_nao_inventa_confianca(self):
        result = Transcript(text="hi", duration_s=1.2345, elapsed_ms=987.6, confidence=None)
        assert result.as_dict() == {
            "text": "hi",
            "duration_s": 1.234,
            "elapsed_ms": 988,
            "language": "en",
            "confidence": None,
        }


class TestFakes:
    async def test_stt_consome_as_transcricoes_em_ordem_e_repete_a_ultima(self):
        stt = FakeSTT(transcripts=["first", "second"])
        assert (await stt.transcribe(b"x" * 100)).text == "first"
        assert (await stt.transcribe(b"x" * 100)).text == "second"
        assert (await stt.transcribe(b"x" * 100)).text == "second"
        assert stt.call_count == 3

    async def test_stt_guarda_o_audio_que_recebeu(self):
        stt = FakeSTT()
        await stt.transcribe(b"audio-bytes")
        assert stt.last_call == b"audio-bytes"

    async def test_stt_propaga_o_erro_configurado(self):
        stt = FakeSTT(error=VoiceError("sem orelha"))
        with pytest.raises(VoiceError):
            await stt.transcribe(b"x" * 100)
        # Mesmo falhando, registrou: o teste de borda quer saber que chegou lá.
        assert stt.call_count == 1

    async def test_tts_devolve_duracao_plausivel_para_o_texto(self):
        tts = FakeTTS()
        speech = await tts.speak("Good morning, how are you today?")
        # ~14 caracteres por segundo de fala.
        assert 1.5 < speech.duration_s < 3.0

    async def test_tts_guarda_texto_e_voz_de_cada_chamada(self):
        tts = FakeTTS(voice="bm_george")
        await tts.speak("One.")
        await tts.speak("Two.", voice="af_heart")
        assert tts.spoken == ["One.", "Two."]
        assert [call.voice for call in tts.calls] == ["bm_george", "af_heart"]

    async def test_tts_recusa_texto_vazio(self):
        with pytest.raises(VoiceError):
            await FakeTTS().speak("   ")

    async def test_voice_aquece_e_fecha_os_dois(self):
        voice = Voice(stt=FakeSTT(), tts=FakeTTS())
        await voice.warm_up()
        await voice.aclose()
        assert voice.stt.warmed and voice.tts.warmed
        assert voice.stt.closed and voice.tts.closed

    def test_dubles_se_declaram_prontos(self):
        # `ready` é o que o /health lê; quem não carrega nada está sempre pronto.
        assert FakeSTT().ready and FakeTTS().ready
        assert FakeSTT().last_error is None


class TestBuildVoice:
    def test_desligada_por_default(self):
        assert build_voice(make_settings()) is None

    def test_provider_fake_rende_dubles(self):
        voice = build_voice(make_settings(voice_enabled=True, provider="fake"))
        assert isinstance(voice.stt, FakeSTT)
        assert isinstance(voice.tts, FakeTTS)

    def test_ligada_rende_os_motores_de_verdade_sem_carregar_nada(self):
        settings = make_settings(voice_enabled=True, voice_dir=Path("/nao/existe"))
        voice = build_voice(settings)
        assert isinstance(voice.stt, WhisperSTT)
        assert isinstance(voice.tts, KokoroTTS)
        # Construir não toca o disco: a carga é preguiçosa, e é o que permite a
        # subida não depender de 450 MB de pesos.
        assert not voice.stt.ready
        assert not voice.tts.ready

    def test_a_configuracao_chega_aos_motores(self):
        settings = make_settings(
            voice_enabled=True,
            stt_model="tiny.en",
            tts_voice="af_heart",
            voice_dir=Path("/tmp/vozes"),
        )
        voice = build_voice(settings)
        assert voice.stt.model == "tiny.en"
        assert voice.tts.voice == "af_heart"
        assert voice.tts.model == settings.tts_model


class TestAnteparos:
    """Os limites dos motores reais disparam antes de qualquer carga.

    Cada um destes testes passaria mesmo numa máquina sem nenhum peso baixado —
    é essa a propriedade sendo afirmada. Se um deles começar a precisar de
    modelo, o anteparo deixou de ser barato.
    """

    @pytest.fixture
    def stt(self) -> WhisperSTT:
        return WhisperSTT(model="nao-existe")

    @pytest.fixture
    def tts(self) -> KokoroTTS:
        return KokoroTTS(model_path=Path("/nao/existe.onnx"), voices_path=Path("/nao/existe.bin"))

    async def test_audio_curto_devolve_texto_vazio_em_vez_de_alucinar(self, stt):
        # Clique no botão sem fala. O Whisper responderia "Thank you."
        result = await stt.transcribe(b"x" * (MIN_AUDIO_BYTES - 1))
        assert result.text == ""
        assert not stt.ready

    async def test_audio_grande_demais_e_recusado(self, stt):
        with pytest.raises(VoiceError):
            await stt.transcribe(b"x" * (MAX_AUDIO_BYTES + 1))
        assert not stt.ready

    async def test_texto_vazio_nao_vira_sintese(self, tts):
        with pytest.raises(VoiceError):
            await tts.speak("   \n ")
        assert not tts.ready

    async def test_paragrafo_e_recusado_para_nao_anular_o_tts_incremental(self, tts):
        with pytest.raises(VoiceError, match="voice/segment"):
            await tts.speak("word " * (MAX_SPEECH_CHARS // 2))
        assert not tts.ready

    async def test_aquecimento_nao_propaga_erro_e_guarda_o_motivo(self, tts):
        # Derrubar a subida por causa de um peso ausente seria pior que um
        # /health degradado que diz o que falta.
        await tts.warm_up()
        assert not tts.ready
        assert "fetch_voice_models" in (tts.last_error or "")
