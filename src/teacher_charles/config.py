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
    # memória: 4B Q4 roda a ~20-25 tok/s, 8B a ~10-13, 14B a ~6 (inviável).
    llm_model: str = "qwen3:4b"

    ollama_url: str = "http://localhost:11434"
    request_timeout_s: float = 120.0

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
