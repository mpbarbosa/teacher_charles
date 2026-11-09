"""Configuração do Teacher Charles.

Todos os ajustes vêm de variáveis de ambiente com prefixo ``TC_`` (ou de um
``.env`` na raiz do repo). Os defaults aqui são os da decisão 1.3 do plano da
Fase 1: alvo mínimo é CPU/iGPU sem CUDA, então o modelo default é 4B-class.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent

APP_DIR_NAME = "teacher-charles"

# As vozes do Kokoro vêm num arquivo só, e o nome dele é parte do release
# — não é escolha de configuração como o modelo é.
TTS_VOICES_FILE = "voices-v1.0.bin"


def xdg_data_home() -> Path:
    """``$XDG_DATA_HOME``, com o fallback que a spec manda."""
    raw = os.environ.get("XDG_DATA_HOME", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".local" / "share"


def xdg_cache_home() -> Path:
    """``$XDG_CACHE_HOME``, com o fallback que a spec manda."""
    raw = os.environ.get("XDG_CACHE_HOME", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".cache"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TC_",
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- Modelo ----------------------------------------------------------
    provider: str = "ollama"

    # Default da Fase 1. Nesta classe de hardware o gargalo é banda de
    # memória, e o Ollama oficial roda em CPU pura (não usa a iGPU Intel).
    # Medido: 9,3 tok/s num 3B Q4 e 4,0 num 7B, o que põe um 4B em ~7-8 tok/s
    # e um 8B em ~3,5. Ver docs/plano-fase-1.md §1.3.
    llm_model: str = "qwen3:4b"

    ollama_url: str = "http://localhost:11434"
    request_timeout_s: float = 120.0

    # Carregar um modelo do disco custa ~15s nesta máquina, e isso cai todo
    # no primeiro turno do aluno. Aquecer na subida move essa espera para
    # enquanto a página ainda está sendo lida.
    llm_warmup: bool = True

    # Modelos de raciocínio (Qwen3, entre outros) pensam antes de responder, e
    # isso custa segundos que uma conversa não tem. None deixa o default do
    # servidor valer; False desliga, que é o que se quer num professor que
    # precisa responder em 1-2s.
    llm_think: bool | None = None

    # --- Ensino ----------------------------------------------------------
    # Só manda ao aluno correção com regra curada por trás.
    #
    # Medido num 3B (plano §5.2): sobre 18 frases corretas, a análise inventou
    # 12 correções, nenhuma citando regra, várias ensinando erro. Nas 24 frases
    # com erro que acertou, todas citavam a regra esperada. Exigir a regra
    # eliminou todos os falsos positivos sem custar nenhum acerto.
    #
    # Desligue se o modelo for grande o bastante para merecer confiança sem
    # regra — e meça antes de confiar.
    require_curated_rule: bool = True

    # A análise espera a resposta terminar, em vez de correr junto com ela.
    #
    # Medido (plano §5.2): o Ollama atende um pedido por vez neste modelo, então
    # disparar as duas juntas só adia a resposta — 11,2s até a primeira palavra
    # contra 0,7s. Num provider de nuvem, ou num servidor que atenda dois
    # pedidos ao mesmo tempo, `false` volta a ser melhor.
    analysis_after_reply: bool = True

    # --- Fonética --------------------------------------------------------
    # A linha de IPA embaixo de toda fala, a do aluno e a do professor.
    #
    # **Ligada** por default, ao contrário da voz, e a diferença é de custo:
    # aqui não há modelo nem peso baixado — é o espeak-ng do wheel, ~0,3 ms
    # por sentença e ~0,25s de carga na subida. Desligar é para quem acha a
    # linha ruidosa, não para quem não tem máquina.
    phonetics_enabled: bool = True

    # Americano por default. É o sotaque que o aluno brasileiro encontra em
    # todo lugar, e é o inventário que a medição da Fase 2 usou para o
    # gabarito. Qualquer língua que o espeak-ng conheça serve (`en-gb`...).
    phonetics_language: str = "en-us"

    # Marca de tônica (ˈ, ˌ) na transcrição. Default ligada: `ˈrecord` e
    # `reˈcord` são palavras diferentes, e esconder isso tiraria metade da
    # lição. O gabarito do `scripts/measure_phonemes.py` faz o oposto, e por
    # motivo oposto — lá a tônica inventaria erro contra um reconhecedor que
    # não a emite.
    phonetics_stress: bool = True

    # --- Voz (Fase 1.5) --------------------------------------------------
    # Desligada por default. Ligar baixa dois modelos e carrega dois runtimes
    # (ctranslate2 e onnxruntime) na memória do processo; quem só quer o chat
    # por texto da Fase 1 não deveria pagar por isso sem pedir.
    voice_enabled: bool = False

    # `base.en`, e **não** o `small.en` que o plano §4 nomeou. Medido nesta
    # máquina em int8, numa fala de 2,6s (plano §5.6):
    #
    #     tiny.en  0,56s    base.en  0,91s    small.en  2,40s
    #
    # O STT roda com o aluno parado esperando — não há streaming para esconder
    # nada. 2,4s do `small.en` furam sozinhos a meta de 1-2s do turno inteiro;
    # 0,9s do `base.en` cabem. O sufixo `.en` segue sendo consequência da
    # decisão 1.1: produto monolíngue permite o modelo menor e mais exato na
    # única língua que interessa.
    stt_model: str = "base.en"
    stt_compute_type: str = "int8"

    # Feixe 1 é busca gulosa. Numa frase curta de aluno o ganho de qualidade
    # do feixe 5 não paga o tempo nesta CPU — medido em §5.6.
    stt_beam_size: int = 1

    # O Whisper alucina em silêncio ("Thank you.", "Thanks for watching!"). O
    # limiar descarta o segmento cuja probabilidade média é baixa demais para
    # ser fala de verdade.
    stt_no_speech_threshold: float = 0.6

    # Kokoro-82M exportado para ONNX, e o int8 ganhou nas duas pontas: RTF
    # mediano 0,54 contra 0,62 do fp32, em 92 MB contra 325 MB. Não houve
    # troca a fazer — ele é mais rápido E menor (plano §5.6).
    tts_model: str = "kokoro-v1.0.int8.onnx"

    # Charles é um professor, e um professor tem uma voz. `bm_george` é a voz
    # masculina britânica do Kokoro — escolha de coerência com a persona, não
    # de qualidade medida, e reversível numa variável de ambiente. As 54 vozes
    # do arquivo aparecem em `/health`.
    tts_voice: str = "bm_george"
    tts_speed: float = 1.0

    # Quantas threads cada motor usa, e os dois defaults saíram de medição
    # oposta um ao outro (plano §5.6).
    #
    # STT: o default do ctranslate2 deixa dinheiro na mesa. Numa fala de 2,6s
    # no `small.en`, 3,37s com o default contra 2,40s com 8 threads; 14 piora
    # (2,52s), porque metade são hyperthreads que disputam a mesma unidade.
    #
    # TTS: o onnxruntime já acerta sozinho, e limitar **piora** — RTF 0,54 com
    # o default contra 0,72 com 4 threads. Daí 0 aqui, que é "não mexa".
    stt_cpu_threads: int = 8
    tts_cpu_threads: int = 0

    # Onde os pesos de voz ficam. É cache, não dado do aluno: são
    # redownloadáveis e não têm nada de pessoal dentro.
    voice_dir: Path | None = Field(default=None)

    # --- Nuvem opcional (recomendação §6 da análise inicial) -------------
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""

    # --- Servidor --------------------------------------------------------
    # Escuta só em loopback: o produto é local por definição.
    host: str = "127.0.0.1"
    port: int = 8000

    # --- Caminhos --------------------------------------------------------
    db_path: Path | None = Field(default=None)

    @property
    def resolved_db_path(self) -> Path:
        """Onde fica o progresso do aluno.

        Fora do repo, sempre: é dado pessoal. ``data/`` no repo guarda
        conhecimento curado (a base contrastiva), nunca dados de aluno.
        """
        if self.db_path is not None:
            return self.db_path.expanduser()
        return xdg_data_home() / APP_DIR_NAME / "charles.db"

    @property
    def resolved_voice_dir(self) -> Path:
        """Onde moram os pesos de voz.

        Terceiro destino, e não um quarto: peso de modelo nunca entra no repo
        nem no diretório de dados do aluno. O Whisper usa o cache do
        ``huggingface_hub``; só o Kokoro precisa de um lugar nosso, porque ele
        vem de um release do GitHub e não de um hub com cache próprio.
        """
        if self.voice_dir is not None:
            return self.voice_dir.expanduser()
        return xdg_cache_home() / APP_DIR_NAME / "voice"

    @property
    def tts_model_path(self) -> Path:
        return self.resolved_voice_dir / self.tts_model

    @property
    def tts_voices_path(self) -> Path:
        return self.resolved_voice_dir / TTS_VOICES_FILE

    @property
    def prompts_dir(self) -> Path:
        return REPO_ROOT / "prompts"

    @property
    def data_dir(self) -> Path:
        return REPO_ROOT / "data"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
