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
    medidas: list[Path] = []
    tabela: list[dict] = []

    def fake_record(seconds, out):
        assert seconds == 6
        gravadas.append(out)
        return out

    def fake_run_audio(recognizer, g2p, path, said, rule_id=None):
        # O WAV medido é o que acabou de ser gravado, e não um caminho montado
        # por conta própria: é esse descasamento que o teste existe para pegar.
        assert path == gravadas[-1]
        medidas.append(path)
        return {"regra": rule_id, "per": 0.1, "esperado": ["a"], "ouvido": ["a"]}

    with (
        mock.patch.object(mp, "record", fake_record),
        mock.patch.object(mp, "run_audio", fake_run_audio),
        # Sem isto o laço pularia o que já está medido no disco desta máquina.
        mock.patch.object(mp, "_clean_report_for", lambda phrase: None),
        mock.patch("builtins.input", side_effect=EOFError),
        mock.patch.object(mp, "_report_corpus", lambda results: tabela.extend(results) or 0),
    ):
        assert mp.run_corpus(object(), object(), 6, redo=True) == 0

    assert [p.name for p in gravadas] == [
        f"e9-{item.rule_id.split('.', 1)[-1]}.wav" for item in mp.SUBSTITUTIONS
    ]
    # Gravar sem medir não é execução do corpus. Se o laço perder a chamada ao
    # `run_audio`, as oito gravações continuam acontecendo e o
    # `_report_corpus([])` continua devolvendo 0 — o teste passava sem medir uma
    # frase. A asserção é sobre o par: uma medida para cada gravação.
    assert medidas == gravadas
    # E a tabela recebe o resultado medido, não o payload de antes da medição:
    # sem `per`, a elocução que acabou de ser medida sai como "recusada".
    assert [linha.get("per") for linha in tabela] == [0.1] * len(mp.SUBSTITUTIONS)


def test_recusa_aparece_na_tabela(measure_phonemes):
    """Sinal recusado é linha da tabela, não silêncio.

    Antes, o `run_audio` devolvia `None` na recusa e o laço descartava o
    resultado: a frase desaparecia da tabela, e quem gravou ficava sem saber se
    ela foi recusada ou nunca gravada — a diferença que decide se regrava.
    """
    mp = measure_phonemes
    tabela: list[dict] = []

    def fake_run_audio(recognizer, g2p, path, said, rule_id=None):
        return {"regra": rule_id, "recusado": "silêncio: nada acima do piso"}

    with (
        mock.patch.object(mp, "record", lambda seconds, out: out),
        mock.patch.object(mp, "run_audio", fake_run_audio),
        mock.patch.object(mp, "_clean_report_for", lambda phrase: None),
        mock.patch("builtins.input", side_effect=EOFError),
        mock.patch.object(mp, "_report_corpus", lambda results: tabela.extend(results) or 0),
    ):
        assert mp.run_corpus(object(), object(), 6, redo=True) == 0

    assert len(tabela) == len(mp.SUBSTITUTIONS)
    assert all(linha["recusado"] for linha in tabela)


# O `run_audio` é testável sem os 318 MB porque os pesados entram por função:
# `read_wav`, o diagnóstico, o G2P e o reconhecedor são todos dublados aqui.
# O que se prende aqui é o **contrato de retorno**, que é o que o `--corpus` lê.
#
# O `numpy` é dublado junto, e não por elegância: ele **não** está no extra
# `dev`: chega com o `voice`. Quem tem a voz instalada vê estes testes passarem
# com o numpy de verdade; o CI roda `uv sync --extra dev --frozen` e quebrava no
# `import numpy as np` de dentro da função. Dublar é honesto aqui porque nada
# abaixo do numpy é exercido: todo helper que o recebe está dublado também.


class _Recognizer:
    model = "fake"
    vocab = {"a"}

    def recognize(self, audio):
        return ["a"], 100.0


class _Alinhamento:
    per = 0.125
    errors = 1
    substitutions = [("ð", "d")]
    deletions = []


@pytest.fixture
def numpy_dublado():
    """Põe um `numpy` de mentira no `sys.modules`.

    O `import` de dentro do `run_audio` encontra o que já está no
    `sys.modules` e nunca vai ao disco — então isto vale tanto na máquina que
    tem numpy quanto no CI que não tem.
    """
    with mock.patch.dict(sys.modules, {"numpy": mock.MagicMock(name="numpy")}):
        yield


def test_run_audio_devolve_os_campos_medidos(measure_phonemes, numpy_dublado, tmp_path):
    """O retorno é o relatório, não o payload de antes da medição.

    Devolver o payload inicial é silencioso: o arquivo no disco fica certo, e
    só a tabela de quem chamou mente — mostrando como "recusada" a elocução
    que acabou de ser medida.
    """
    mp = measure_phonemes
    escrito: dict = {}

    with (
        mock.patch.object(mp, "read_wav", lambda path, np: [0] * mp.SAMPLE_RATE),
        mock.patch.object(mp, "_audio_fingerprint", lambda wav: {"md5": "0" * 32}),
        mock.patch.object(mp, "diagnose_signal", lambda samples, np: None),
        mock.patch.object(mp, "speech_span", lambda samples, np: (0.1, 0.9)),
        mock.patch.object(mp, "warn_if_quiet", lambda samples, np: None),
        mock.patch.object(mp, "warn_if_head_cut", lambda span: None),
        mock.patch.object(mp, "warn_if_truncated", lambda samples, span: None),
        mock.patch.object(mp, "phonemize", lambda g2p, text: "a"),
        mock.patch.object(mp, "tokenize", lambda ipa, vocab: ["a"]),
        mock.patch.object(mp, "align", lambda expected, heard: _Alinhamento()),
        mock.patch.object(
            mp,
            "write_report",
            lambda wav, payload: escrito.update(payload) or tmp_path / "r.json",
        ),
    ):
        medida = mp.run_audio(
            _Recognizer(), object(), tmp_path / "e9.wav", "Other", rule_id="phoneme.th"
        )

    assert medida["per"] == 0.125
    assert medida["ouvido"] == ["a"]
    assert medida["regra"] == "phoneme.th"
    # O retorno é exatamente o que foi para o disco: é essa igualdade que faz a
    # tabela do `--corpus` e o relatório contarem a mesma história.
    assert medida == escrito


def test_run_audio_devolve_a_recusa_como_dado(measure_phonemes, numpy_dublado, tmp_path):
    """Recusa volta como dicionário com `recusado`, não como `None`.

    O `None` fazia a frase recusada desaparecer da tabela do `--corpus`, e
    desaparecer é pior que aparecer recusada: quem gravou não sabe se precisa
    regravar.
    """
    mp = measure_phonemes

    class _Problema:
        message = "silêncio: nada acima do piso"
        gain_related = True

    with (
        mock.patch.object(mp, "read_wav", lambda path, np: [0] * mp.SAMPLE_RATE),
        mock.patch.object(mp, "_audio_fingerprint", lambda wav: {"md5": "0" * 32}),
        mock.patch.object(mp, "diagnose_signal", lambda samples, np: _Problema()),
        mock.patch.object(mp, "write_report", lambda wav, payload: tmp_path / "r.json"),
    ):
        medida = mp.run_audio(
            _Recognizer(), object(), tmp_path / "e9.wav", "Other", rule_id="phoneme.th"
        )

    assert medida["recusado"] == _Problema.message
    assert "per" not in medida
