"""Entrada de linha de comando: ``uv run teacher-charles``."""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from . import __version__
from .config import get_settings


def build_parser(defaults_host: str, defaults_port: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="teacher-charles",
        description="Sobe o Teacher Charles em localhost.",
    )
    parser.add_argument("--host", default=defaults_host, help=f"default: {defaults_host}")
    parser.add_argument("--port", type=int, default=defaults_port, help=f"default: {defaults_port}")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="recarrega a cada mudança de arquivo (desenvolvimento)",
    )
    parser.add_argument("--version", action="version", version=f"teacher-charles {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    args = build_parser(settings.host, settings.port).parse_args(argv)

    # Sem isto o root logger fica em WARNING e as mensagens do app somem —
    # inclusive o aviso de aquecimento falho, que é justamente o que se quer
    # ver quando o primeiro turno está lento.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    print(f"Teacher Charles {__version__}")
    print(f"  modelo : {settings.llm_model} via {settings.provider}")
    print(f"  banco  : {settings.resolved_db_path}")
    print(f"  http   : http://{args.host}:{args.port}")
    print()

    uvicorn.run(
        "teacher_charles.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
