"""O guarda de porta ocupada da linha de comando.

Sem ele o bind é a última coisa do boot: o erro chega depois do banner e da
base contrastiva carregada, no meio de linhas de sucesso, e se lê como queda
do app. Com ele, nada sobe — e a mensagem diz quem olhar e qual a saída.
"""

from __future__ import annotations

import socket

import pytest

from teacher_charles import cli


@pytest.fixture
def busy_port():
    """Uma porta de verdade ocupada por um listener de verdade."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        yield holder.getsockname()[1]


@pytest.fixture
def free_port():
    """Uma porta que estava livre no instante do teste."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return port


def test_porta_ocupada_e_reconhecida(busy_port):
    assert cli.port_is_taken("127.0.0.1", busy_port) is True


def test_porta_livre_nao_acusa_ocupada(free_port):
    assert cli.port_is_taken("127.0.0.1", free_port) is False


def test_host_que_nao_resolve_nao_vira_porta_ocupada():
    # Não é papel deste guarda julgar o host: o erro certo é o do uvicorn.
    assert cli.port_is_taken("nao-existe.invalid", 8000) is False


def test_nao_sobe_o_servidor_com_a_porta_ocupada(busy_port, monkeypatch, capsys):
    chamadas = []
    monkeypatch.setattr(cli.uvicorn, "run", lambda *a, **k: chamadas.append((a, k)))

    codigo = cli.main(["--host", "127.0.0.1", "--port", str(busy_port)])

    assert codigo == 1
    assert chamadas == []
    erro = capsys.readouterr().err
    assert str(busy_port) in erro
    assert str(busy_port + 1) in erro  # a saída sugerida


def test_sobe_normalmente_com_a_porta_livre(free_port, monkeypatch):
    chamadas = []
    monkeypatch.setattr(cli.uvicorn, "run", lambda *a, **k: chamadas.append((a, k)))

    codigo = cli.main(["--host", "127.0.0.1", "--port", str(free_port)])

    assert codigo == 0
    assert len(chamadas) == 1
    assert chamadas[0][1]["port"] == free_port
