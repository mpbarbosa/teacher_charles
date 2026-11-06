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


def xdg_data_home() -> Path:
    """``$XDG_DATA_HOME``, com o fallback que a spec manda."""
    raw = os.environ.get("XDG_DATA_HOME", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".local" / "share"


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
    def prompts_dir(self) -> Path:
        return REPO_ROOT / "prompts"

    @property
    def data_dir(self) -> Path:
        return REPO_ROOT / "data"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
