"""O laço do `--corpus`, que grava as oito frases curadas da E9.

Um teste só, e não uma suíte: o resto do `measure_phonemes.py` precisa dos 318
MB do reconhecedor ou do microfone desta máquina, e nenhum dos dois cabe na
suíte rápida. O que cabe — e o que falhou — é a ordem: **gravar antes de
medir**. A primeira versão do laço montava o caminho do WAV e chamava o
`run_audio` direto; o microfone nunca abria, e a execução morria num
`FileNotFoundError` sobre o arquivo que ninguém escreveu, na terceira frase,
depois de a pessoa já ter esperado a carga do modelo.

Importar um script em vez de um módulo do pacote é exceção nesta suíte, e vale
aqui porque o `measure_phonemes.py` só usa a stdlib no topo: o `numpy`, o
`onnxruntime` e o `phonemizer` entram dentro das funções. O teste não carrega
nenhum deles.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest import mock

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_phonemes.py"


@pytest.fixture(scope="module")
def measure_phonemes():
    spec = importlib.util.spec_from_file_location("measure_phonemes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # O módulo precisa estar no `sys.modules` antes de executar: há
    # `@dataclass(slots=True)` no topo, e o decorador procura o módulo da
    # classe pelo nome para resolver as anotações.
    sys.modules["measure_phonemes"] = module
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        del sys.modules["measure_phonemes"]


def test_grava_cada_frase_antes_de_medir(measure_phonemes):
    mp = measure_phonemes
    gravadas: list[Path] = []

    def fake_record(seconds, out):
        assert seconds == 6
        gravadas.append(out)
        return out

    def fake_run_audio(recognizer, g2p, path, said, rule_id=None):
        # O WAV medido é o que acabou de ser gravado, e não um caminho montado
        # por conta própria: é esse descasamento que o teste existe para pegar.
        assert path == gravadas[-1]
        return {"regra": rule_id, "per": 0.1, "esperado": ["a"], "ouvido": ["a"]}

    with (
        mock.patch.object(mp, "record", fake_record),
        mock.patch.object(mp, "run_audio", fake_run_audio),
        # Sem isto o laço pularia o que já está medido no disco desta máquina.
        mock.patch.object(mp, "_clean_report_for", lambda phrase: None),
        mock.patch("builtins.input", side_effect=EOFError),
    ):
        assert mp.run_corpus(object(), object(), 6, redo=True) == 0

    assert [p.name for p in gravadas] == [
        f"e9-{item.rule_id.split('.', 1)[-1]}.wav" for item in mp.SUBSTITUTIONS
    ]
