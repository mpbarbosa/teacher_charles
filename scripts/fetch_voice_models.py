#!/usr/bin/env python3
"""Baixa os pesos de voz da Fase 1.5 para fora do repo.

    uv run python scripts/fetch_voice_models.py          # o que a config pede
    uv run python scripts/fetch_voice_models.py --all    # int8 e fp32, para medir

O Whisper **não** está aqui de propósito: o ``faster-whisper`` baixa o modelo
sozinho no cache do ``huggingface_hub`` na primeira vez que é usado, e
reimplementar isso só criaria um segundo cache. O Kokoro precisa deste script
porque vem de um release do GitHub, sem hub e sem cache por trás.

Os arquivos vão para ``$XDG_CACHE_HOME/teacher-charles/voice`` (ver
``Settings.resolved_voice_dir``): peso de modelo não é dado do aluno e não é
conteúdo curado, então não entra nem em ``XDG_DATA_HOME`` nem no repo.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from teacher_charles.config import TTS_VOICES_FILE, Settings  # noqa: E402

RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

# Tamanho conhecido de cada arquivo. Não é verificação de integridade — é como
# o script sabe que um download interrompido ficou pela metade, em vez de
# deixar um ONNX truncado explodir dentro do onnxruntime depois.
FILES: dict[str, int] = {
    "kokoro-v1.0.int8.onnx": 92_361_271,
    "kokoro-v1.0.onnx": 325_532_387,
    TTS_VOICES_FILE: 28_214_398,
}

# O arquivo de vozes serve aos dois modelos, então vem sempre.
ALWAYS = (TTS_VOICES_FILE,)


def human(size: float) -> str:
    return f"{size / 1024 / 1024:.0f} MB"


def download(url: str, target: Path, expected: int) -> None:
    """Baixa para um ``.part`` e só então renomeia.

    Sem isto, um Ctrl-C no meio deixa no lugar final um arquivo com o nome
    certo e o conteúdo pela metade — e a próxima execução o consideraria
    pronto.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")

    print(f"  baixando {target.name} ({human(expected)})...", end="", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as out:
            digest = hashlib.sha256()
            while chunk := response.read(1024 * 256):
                out.write(chunk)
                digest.update(chunk)
    except (urllib.error.URLError, OSError) as exc:
        part.unlink(missing_ok=True)
        print(f" falhou: {exc}")
        raise SystemExit(1) from exc

    got = part.stat().st_size
    if got != expected:
        part.unlink(missing_ok=True)
        print(f" tamanho inesperado: {got} bytes, esperava {expected}")
        raise SystemExit(1)

    part.replace(target)
    print(f" ok  sha256={digest.hexdigest()[:16]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="baixa int8 e fp32, para a medição comparar os dois",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="destino (default: o que TC_VOICE_DIR / XDG_CACHE_HOME dizem)",
    )
    args = parser.parse_args()

    settings = Settings()
    destination = args.dir.expanduser() if args.dir else settings.resolved_voice_dir

    wanted = list(ALWAYS)
    if args.all:
        wanted += [name for name in FILES if name not in ALWAYS]
    else:
        wanted.append(settings.tts_model)

    unknown = [name for name in wanted if name not in FILES]
    if unknown:
        print(f" modelo desconhecido: {', '.join(unknown)}")
        print(f" conhecidos: {', '.join(FILES)}")
        return 1

    free_gb = shutil.disk_usage(destination.parent.parent).free / 1024**3
    needed = sum(FILES[name] for name in wanted) / 1024**3
    if free_gb < needed + 0.5:
        print(
            f" disco insuficiente em {destination}: {free_gb:.1f} GB livres, "
            f"precisa de {needed:.1f} GB"
        )
        return 1

    print(f"\n Pesos de voz em {destination}\n")
    for name in wanted:
        target = destination / name
        if target.is_file() and target.stat().st_size == FILES[name]:
            print(f"  {name} já está lá ({human(FILES[name])})")
            continue
        download(f"{RELEASE}/{name}", target, FILES[name])

    print(
        "\n Pronto. O Whisper se baixa sozinho no primeiro uso.\n"
        " Ligue a voz com TC_VOICE_ENABLED=true.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
