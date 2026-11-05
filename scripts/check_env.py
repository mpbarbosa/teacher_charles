#!/usr/bin/env python3
"""Diz se esta máquina está pronta para rodar o Teacher Charles.

Só stdlib, de propósito: este script tem que rodar ANTES do ``uv sync``, senão
não serve para a primeira pergunta que um novo dev faz. Rode com qualquer
Python 3.9+::

    python3 scripts/check_env.py

Saída 0 = pronto. 1 = há bloqueio. Avisos sozinhos não derrubam a saída.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Decisão 1.3 do plano da Fase 1: o alvo mínimo é CPU/iGPU sem CUDA, e o
# gargalo é banda de memória. Daí o default 4B-class.
MODEL_DEFAULT = "qwen3:4b"
OLLAMA_URL_DEFAULT = "http://localhost:11434"

# Um 4B Q4 ocupa ~2,5 GB e o processo em volta pede folga.
RAM_RECOMMENDED_GB = 8.0
RAM_MINIMUM_GB = 6.0
DISK_RECOMMENDED_GB = 10.0

OK, WARN, FAIL = "ok", "warn", "fail"
GLYPH = {OK: "\033[32m✓\033[0m", WARN: "\033[33m!\033[0m", FAIL: "\033[31m✗\033[0m"}

_results: list[tuple[str, str, str]] = []


def report(status: str, label: str, detail: str = "") -> None:
    _results.append((status, label, detail))
    line = f"  {GLYPH[status]} {label}"
    if detail:
        line += f"\n      {detail}"
    print(line)


def env(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


def read_dotenv() -> dict[str, str]:
    """Lê o ``.env`` sem depender de pydantic-settings."""
    path = REPO_ROOT / ".env"
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


# --------------------------------------------------------------------------- #


def check_python() -> None:
    major, minor = sys.version_info[:2]
    running = f"{major}.{minor}.{sys.version_info[2]}"
    if (major, minor) == (3, 12):
        report(OK, f"Python {running}")
        return
    # Este script roda em qualquer Python; o projeto é que pede 3.12.
    report(
        WARN,
        f"Python {running} executando este script",
        "O projeto pede 3.12 (ver .python-version). Os wheels de ctranslate2 e "
        "onnxruntime, que a voz vai precisar na Fase 1.5, ficam atrás das "
        "versões novas. `uv sync` resolve isso sozinho.",
    )


def check_ram() -> None:
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        total_kb = next(
            int(line.split()[1]) for line in meminfo.splitlines() if line.startswith("MemTotal:")
        )
        total_gb = total_kb / 1024 / 1024
    except (OSError, StopIteration, ValueError, IndexError):
        report(WARN, "RAM", "não deu para ler /proc/meminfo nesta plataforma")
        return

    detail = f"{total_gb:.1f} GB no total"
    if total_gb >= RAM_RECOMMENDED_GB:
        report(OK, "RAM", detail)
    elif total_gb >= RAM_MINIMUM_GB:
        report(WARN, "RAM", f"{detail} — aperta com um modelo 4B; feche o resto")
    else:
        report(FAIL, "RAM", f"{detail} — abaixo do mínimo de {RAM_MINIMUM_GB:.0f} GB")


def check_disk() -> None:
    target = Path.home()
    try:
        free_gb = shutil.disk_usage(target).free / 1e9
    except OSError as exc:
        report(WARN, "Espaço em disco", str(exc))
        return

    detail = f"{free_gb:.1f} GB livres em {target}"
    if free_gb >= DISK_RECOMMENDED_GB:
        report(OK, "Espaço em disco", detail)
    else:
        report(WARN, "Espaço em disco", f"{detail} — um modelo 4B Q4 pede ~2,5 GB")


def check_uv() -> None:
    path = shutil.which("uv")
    if path:
        report(OK, "uv", path)
    else:
        report(
            FAIL,
            "uv não encontrado",
            "Instale: curl -LsSf https://astral.sh/uv/install.sh | sh",
        )


def _fetch_tags(base_url: str) -> list[dict]:
    url = f"{base_url.rstrip('/')}/api/tags"
    with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310
        payload = json.load(response)
    models = payload.get("models", [])
    return [m for m in models if isinstance(m, dict)]


def check_ollama_and_model(base_url: str, wanted: str) -> None:
    # Sondamos a API HTTP, nunca `which ollama`. O Ollama costuma ficar fora do
    # PATH (em ~/.local/ollama/bin, por exemplo) enquanto o serviço roda muito
    # bem — um check pelo binário reportaria "não instalado" sobre um Ollama
    # funcional, que é exatamente o falso negativo que atrapalha o novo dev.
    try:
        models = _fetch_tags(base_url)
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
        report(
            FAIL,
            f"Ollama não respondeu em {base_url}",
            f"{exc}\n      Suba o serviço com `ollama serve` "
            f"(ou ajuste TC_OLLAMA_URL se ele escuta em outro lugar).",
        )
        return

    report(OK, f"Ollama respondendo em {base_url}")

    names = sorted(m["name"] for m in models if "name" in m)
    present = wanted in names or (":" not in wanted and f"{wanted}:latest" in names)

    if present:
        report(OK, f"Modelo {wanted} baixado")
        return

    if not names:
        report(
            FAIL,
            f"Modelo {wanted} não está baixado",
            f"Nenhum modelo presente. Rode: ollama pull {wanted}",
        )
        return

    listing = "\n".join(
        f"        - {m['name']}  ({m.get('size', 0) / 1e9:.1f} GB,"
        f" {m.get('details', {}).get('parameter_size', '?')})"
        for m in sorted(models, key=lambda m: m.get("size", 0))
    )
    report(
        WARN,
        f"Modelo {wanted} não está baixado",
        f"Rode: ollama pull {wanted}\n"
        f"      Ou aponte TC_LLM_MODEL para um destes, já presentes:\n{listing}",
    )


def check_db_writable() -> None:
    dotenv = read_dotenv()
    configured = os.environ.get("TC_DB_PATH", "").strip() or dotenv.get("TC_DB_PATH", "").strip()
    if configured:
        db_path = Path(configured).expanduser()
    else:
        xdg = os.environ.get("XDG_DATA_HOME", "").strip()
        base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
        db_path = base / "teacher-charles" / "charles.db"

    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        probe = db_path.parent / ".write-probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        report(FAIL, "Banco do aluno não é gravável", f"{db_path}\n      {exc}")
        return

    state = "já existe" if db_path.exists() else "será criado na primeira subida"
    report(OK, "Banco do aluno gravável", f"{db_path} ({state})")


# --------------------------------------------------------------------------- #


def main() -> int:
    dotenv = read_dotenv()
    base_url = env("TC_OLLAMA_URL", dotenv.get("TC_OLLAMA_URL", OLLAMA_URL_DEFAULT))
    wanted = env("TC_LLM_MODEL", dotenv.get("TC_LLM_MODEL", MODEL_DEFAULT))

    print("\nTeacher Charles — checagem de ambiente\n")

    print(" Runtime")
    check_python()
    check_uv()

    print("\n Hardware")
    check_ram()
    check_disk()

    print("\n Modelo")
    check_ollama_and_model(base_url, wanted)

    print("\n Dados")
    check_db_writable()

    fails = sum(1 for status, _, _ in _results if status == FAIL)
    warns = sum(1 for status, _, _ in _results if status == WARN)

    print()
    if fails:
        print(f" {fails} bloqueio(s), {warns} aviso(s). Resolva os ✗ antes de seguir.\n")
        return 1
    if warns:
        print(f" Pronto para rodar, com {warns} aviso(s).\n")
        return 0
    print(" Tudo pronto.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
