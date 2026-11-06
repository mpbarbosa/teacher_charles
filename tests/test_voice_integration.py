"""Teste de integração: a voz de verdade, com os pesos no disco.

É o critério de pronto da Fase 1.5. Tudo o mais na suíte de voz roda com
dublês; este exercita o caminho real — ctranslate2, onnxruntime, PCM, e a
volta inteira texto → fala → texto.

Pula sozinho quando os pesos não estão baixados, então quem clona o repo sem
rodar o ``fetch_voice_models.py`` ainda vê a suíte verde. Rode só estes com::

    uv run pytest -m integration -k voice

O STT usa ``TC_TEST_STT_MODEL`` se estiver definido, senão ``tiny.en``: o
``tiny.en`` são 75 MB contra os 150 MB do ``base.en`` default, e para o que
este arquivo afirma — que o pipeline fecha — o modelo menor basta.
"""

from __future__ import annotations

import os
import wave
from io import BytesIO

import pytest

from teacher_charles.config import Settings
from teacher_charles.voice.base import Speech
from teacher_charles.voice.segment import split_sentences

pytestmark = pytest.mark.integration


def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture
def tts():
    from teacher_charles.voice.kokoro import KokoroTTS

    config = settings()
    if not config.tts_model_path.is_file() or not config.tts_voices_path.is_file():
        pytest.skip(
            f"pesos de TTS ausentes em {config.resolved_voice_dir} — "
            "rode `uv run python scripts/fetch_voice_models.py`"
        )
    engine = KokoroTTS(
        model_path=config.tts_model_path,
        voices_path=config.tts_voices_path,
        voice=config.tts_voice,
        cpu_threads=config.tts_cpu_threads,
    )
    yield engine


@pytest.fixture
def stt():
    from teacher_charles.voice.whisper import WhisperSTT

    config = settings()
    model = os.environ.get("TC_TEST_STT_MODEL", "").strip() or "tiny.en"
    engine = WhisperSTT(
        model=model,
        compute_type=config.stt_compute_type,
        beam_size=config.stt_beam_size,
        cpu_threads=config.stt_cpu_threads,
    )
    yield engine


async def warmed(engine):
    await engine.warm_up()
    if not engine.ready:
        pytest.skip(f"motor não carregou: {engine.last_error}")
    return engine


# --- TTS -------------------------------------------------------------------- #


async def test_tts_fala_uma_sentenca_e_devolve_pcm_tocavel(tts):
    await warmed(tts)
    speech = await tts.speak("Good morning. How was your weekend?")

    assert isinstance(speech, Speech)
    assert speech.sample_rate == 24_000
    # Dois segundos é o piso plausível para esta frase; menos que isso é
    # silêncio ou truncamento, não fala.
    assert speech.duration_s > 1.5
    # PCM 16-bit: o tamanho tem que ser par, senão uma amostra ficou partida.
    assert len(speech.pcm) % 2 == 0


async def test_tts_acompanha_a_reproducao(tts):
    """O número que decide a Fase 1.5: RTF abaixo de 1,0.

    Acima disso a síntese não acompanha a reprodução e a voz engasga entre
    sentenças, por mais rápido que o modelo de linguagem seja. Medido nesta
    máquina em 0,54 mediano; o portão é generoso de propósito, porque uma
    máquina mais modesta ainda deve passar enquanto a voz for usável.
    """
    await warmed(tts)
    speech = await tts.speak("That is one of the most common mistakes Brazilian learners make.")
    assert speech.real_time_factor < 1.0


async def test_tts_monta_wav_que_o_modulo_wave_le(tts):
    await warmed(tts)
    speech = await tts.speak("Tell me about your work.")
    with wave.open(BytesIO(speech.as_wav()), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 24_000
        assert handle.getnframes() * 2 == len(speech.pcm)


async def test_tts_conhece_as_vozes_so_depois_de_carregar(tts):
    # `voices()` não força a carga de propósito: o /health não pode ter efeito
    # colateral de centenas de megabytes.
    assert tts.voices() == ()
    await warmed(tts)
    available = tts.voices()
    assert len(available) > 10
    assert tts.voice in available


# --- STT -------------------------------------------------------------------- #


async def test_stt_recusa_audio_curto_sem_carregar_o_modelo(stt):
    # O anteparo é barato de propósito: um clique no botão não deveria custar
    # o carregamento de um modelo, nem render alucinação ("Thank you.").
    result = await stt.transcribe(b"\x00" * 512)
    assert result.text == ""
    assert not stt.ready


async def test_stt_transcreve_audio_de_verdade(stt, tts):
    """A volta inteira: texto → Kokoro → WAV → Whisper → texto.

    Áudio sintético é o caso fácil (sem sotaque, sem ruído), então isto afirma
    que o pipeline fecha — não que a exatidão com um aluno brasileiro falando
    num microfone de notebook seja esta. Essa medição precisa de voz humana, e
    está em ``scripts/measure_voice.py --record``.
    """
    await warmed(tts)
    await warmed(stt)

    spoken = "I work in marketing here in Brazil."
    audio = (await tts.speak(spoken)).as_wav()
    result = await stt.transcribe(audio)

    assert "marketing" in result.text.lower()
    assert "brazil" in result.text.lower()
    assert result.duration_s > 1.0
    # Confiança reportada, não inventada: o motor expõe `avg_logprob`.
    assert result.confidence is not None


async def test_stt_decodifica_o_que_o_navegador_grava(stt, tts):
    """WAV aqui, mas o container é do PyAV, não nosso.

    O navegador grava webm/opus ou ogg/opus. Quem decodifica é o ``av`` que
    vem com o faster-whisper, e é por isso que não há ffmpeg em subprocesso em
    nenhum ponto do caminho de áudio. O que este teste prende é a fronteira:
    bytes de arquivo entram, texto sai, sem passo de conversão nosso no meio.
    """
    await warmed(tts)
    await warmed(stt)
    audio = (await tts.speak("Good morning, Charles.")).as_wav()
    # Bytes crus, exatamente como o corpo do POST /api/stt chega.
    result = await stt.transcribe(audio)
    assert result.text.strip()


# --- o caminho do turno ----------------------------------------------------- #


async def test_a_resposta_do_professor_vira_fala_sentenca_por_sentenca(tts):
    """O que o turno de voz faz: segmentar, e sintetizar cada parte.

    A primeira sentença é a única que o aluno espera em silêncio; o resto é
    sintetizado enquanto ele ouve a anterior. Este teste afirma as duas
    metades: que cada sentença rende áudio, e que a soma do tempo de síntese
    fica abaixo da soma da duração.
    """
    await warmed(tts)
    reply = (
        'You said "I have 30 years." In English, age uses *to be*: '
        '"I am 30 years old." So, what did you do on your birthday?'
    )
    sentences = split_sentences(reply)
    assert len(sentences) >= 3
    # O markdown da persona não chega ao sintetizador.
    assert not any("*" in sentence for sentence in sentences)

    pieces = [await tts.speak(sentence) for sentence in sentences]
    assert all(piece.duration_s > 0.3 for piece in pieces)

    synthesis = sum(piece.elapsed_ms for piece in pieces) / 1000
    audio = sum(piece.duration_s for piece in pieces)
    # Folga de verdade, não empate: é o que faz a fala sair contínua.
    assert synthesis < audio
