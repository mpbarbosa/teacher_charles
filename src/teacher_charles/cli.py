"""Entrada de linha de comando: ``uv run teacher-charles``."""

from __future__ import annotations

import argparse
import logging
import socket
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


def port_is_taken(host: str, port: int) -> bool:
    """Diz se alguém já escuta em ``host:port``.

    Por que antes de subir, e não deixar o uvicorn reclamar: o bind é a última
    coisa que acontece. Sem este teste o erro só aparece depois do banner, do
    lifespan e da base contrastiva carregada — um ``ERROR`` no meio de linhas
    de sucesso, seguido de ``Application shutdown complete``, que se lê como
    queda do app e não como porta ocupada. E quem ocupa a porta costuma ser
    outro serviço da máquina, não um Charles esquecido.

    O ``SO_REUSEADDR`` é o mesmo que o uvicorn usa: sem ele um socket em
    TIME_WAIT de um Charles recém-fechado seria lido como porta ocupada, e o
    guarda recusaria uma porta que o servidor conseguiria usar.
    """
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        # Nome que não resolve não é porta ocupada: deixa o uvicorn dar o erro
        # certo em vez de trocá-lo por um palpite errado.
        return False

    for family, socktype, proto, _canonname, sockaddr in infos:
        with socket.socket(family, socktype, proto) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(sockaddr)
            except OSError:
                return True
    return False


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

    if port_is_taken(args.host, args.port):
        print(
            f"teacher-charles: a porta {args.port} em {args.host} já está ocupada.\n"
            f"  quem está lá : ss -ltnp | grep :{args.port}\n"
            f"  outra porta  : teacher-charles --port {args.port + 1}"
            f" (ou scripts/run.sh --port {args.port + 1})",
            file=sys.stderr,
        )
        return 1

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
