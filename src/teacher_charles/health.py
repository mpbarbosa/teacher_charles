"""Sondagem do modelo, para o ``/health``.

Pergunta uma coisa só: o modelo configurado está ao alcance deste processo?

O conhecimento sobre o formato da API vive em ``llm/``; aqui fica só a decisão
de como reportar. Sondamos a **API HTTP**, nunca o binário ``ollama`` no PATH:
é comum o serviço rodar com o binário fora do PATH (em ``~/.local/ollama/bin``,
por exemplo), e um ``which ollama`` diria "não instalado" sobre um Ollama
perfeitamente funcional.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .config import Settings
from .llm.base import ProviderError
from .llm.ollama import list_models


@dataclass(slots=True)
class ModelStatus:
    provider: str
    wanted_model: str
    reachable: bool = False
    model_present: bool = False
    available_models: list[str] = field(default_factory=list)
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.reachable and self.model_present

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ok": self.ok}


def _matches(wanted: str, available: str) -> bool:
    """Compara nomes de modelo tolerando a tag implícita ``:latest``.

    O Ollama guarda ``qwen3:4b`` como ``qwen3:4b``, mas ``qwen3`` como
    ``qwen3:latest``, então a comparação crua erraria no segundo caso.
    """
    if wanted == available:
        return True
    if ":" not in wanted:
        return available == f"{wanted}:latest"
    return False


async def probe_model(settings: Settings) -> ModelStatus:
    status = ModelStatus(provider=settings.provider, wanted_model=settings.llm_model)

    if settings.provider == "fake":
        status.reachable = True
        status.model_present = True
        status.detail = "provider 'fake': nenhum modelo de verdade em uso"
        return status

    if settings.provider == "openai_compat":
        if not settings.openai_api_key:
            status.detail = "TC_PROVIDER=openai_compat exige TC_OPENAI_API_KEY"
            return status
        # Não sondamos a nuvem: custaria uma chamada paga a cada /health.
        status.reachable = True
        status.model_present = True
        status.detail = f"provider de nuvem configurado ({settings.openai_base_url})"
        return status

    if settings.provider != "ollama":
        status.detail = f"provider desconhecido: {settings.provider!r}"
        return status

    try:
        status.available_models = await list_models(settings.ollama_url)
    except ProviderError as exc:
        status.detail = str(exc)
        return status

    status.reachable = True
    status.model_present = any(_matches(settings.llm_model, m) for m in status.available_models)

    if status.model_present:
        status.detail = f"modelo {settings.llm_model!r} pronto"
    elif status.available_models:
        status.detail = (
            f"modelo {settings.llm_model!r} não foi baixado. "
            f"Rode: ollama pull {settings.llm_model} "
            f"(ou aponte TC_LLM_MODEL para um dos já presentes)"
        )
    else:
        status.detail = "Ollama respondeu, mas não há nenhum modelo baixado"

    return status
