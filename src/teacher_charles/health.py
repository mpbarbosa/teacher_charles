"""Sondagem do modelo, para o ``/health``.

A E1 traz o ``ChatProvider`` de verdade. Aqui só se pergunta uma coisa: o
modelo configurado está ao alcance deste processo?

Sondamos a **API HTTP**, nunca o binário ``ollama`` no PATH. Nesta máquina de
desenvolvimento, por exemplo, o serviço roda em ``~/.local/ollama/bin/ollama`` e
não está no PATH — um ``which ollama`` diria "não instalado" sobre um Ollama
perfeitamente funcional.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from .config import Settings


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

    if settings.provider == "openai_compat":
        if not settings.openai_api_key:
            status.detail = "TC_PROVIDER=openai_compat exige TC_OPENAI_API_KEY"
            return status
        # A E1 valida o provider de nuvem de verdade; aqui basta ter chave.
        status.reachable = True
        status.model_present = True
        status.detail = "provider de nuvem configurado (não sondado na E0)"
        return status

    if settings.provider != "ollama":
        status.detail = f"provider desconhecido: {settings.provider!r}"
        return status

    url = f"{settings.ollama_url.rstrip('/')}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        status.detail = f"Ollama não respondeu em {settings.ollama_url}: {exc}"
        return status
    except ValueError as exc:
        status.detail = f"Ollama respondeu algo que não é JSON: {exc}"
        return status

    status.reachable = True
    status.available_models = sorted(
        m["name"] for m in payload.get("models", []) if isinstance(m, dict) and "name" in m
    )
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
