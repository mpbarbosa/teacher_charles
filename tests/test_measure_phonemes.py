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

    def fake_run_audio(recognizer, g2p, path, said, rule_id=None, wrong=False):
        # O WAV medido é o que acabou de ser gravado, e não um caminho montado
        # por conta própria: é esse descasamento que o teste existe para pegar.
        assert path == gravadas[-1]
        medidas.append(path)
        return {"regra": rule_id, "per": 0.1, "esperado": ["a"], "ouvido": ["a"]}

    with (
        mock.patch.object(mp, "record", fake_record),
        mock.patch.object(mp, "run_audio", fake_run_audio),
        # Sem isto o laço pularia o que já está medido no disco desta máquina.
        mock.patch.object(mp, "_clean_report_for", lambda phrase, wrong=False: None),
        mock.patch("builtins.input", side_effect=EOFError),
        mock.patch.object(
            mp, "_report_corpus", lambda results, wrong=False: tabela.extend(results) or 0
        ),
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

    def fake_run_audio(recognizer, g2p, path, said, rule_id=None, wrong=False):
        return {"regra": rule_id, "recusado": "silêncio: nada acima do piso"}

    with (
        mock.patch.object(mp, "record", lambda seconds, out: out),
        mock.patch.object(mp, "run_audio", fake_run_audio),
        mock.patch.object(mp, "_clean_report_for", lambda phrase, wrong=False: None),
        mock.patch("builtins.input", side_effect=EOFError),
        mock.patch.object(
            mp, "_report_corpus", lambda results, wrong=False: tabela.extend(results) or 0
        ),
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


def _degrau_com_travessia(
    path,
    sr: int = 16000,
    travessia_blocos: int = 1,
    vies_na_fala: bool = False,
) -> float:
    """Escreve um WAV com o perfil **medido** de uma abertura de conversor.

    Os números vêm de uma gravação de verdade da E9, bloco de 10 ms por bloco
    de 10 ms: um bloco já enviesado, depois o bloco em que o degrau atravessa
    o zero (média +0,1004 contra RMS 0,7222 — razão 0,14, abaixo do
    `DC_DOMINANCE`), e só então o degrau cravado em -1,0 decaindo. Devolve
    quantos segundos de transiente foram escritos.
    """
    import wave

    import numpy as np

    bloco = sr // 100
    partes = [np.full(bloco, -0.3, dtype=np.float32)]
    # A travessia: parte acima de zero, parte no fundo. A média quase se
    # anula contra um RMS enorme, e é isso que a peneira de DC não vê.
    for _ in range(travessia_blocos):
        travessia = np.full(bloco, -1.0, dtype=np.float32)
        travessia[: int(bloco * 0.6)] = 0.78
        partes.append(travessia)
    # O degrau decaindo, constante dentro de cada bloco: razão 1,0 em todos.
    for i in range(48):
        partes.append(np.full(bloco, -1.0 * (0.94**i), dtype=np.float32))
    transiente = sum(len(p) for p in partes) / sr
    # Fala: média zero, nível parecido com o da gravação (RMS ~0,017).
    rng = np.random.default_rng(7)
    if vies_na_fala:
        # Um bloco de fala, depois um bloco de fala **enviesado** — razão 1,0,
        # dominado por DC como o degrau. Medido: no primeiro segundo de uma
        # gravação de verdade há seis falhas curtas entre blocos assim, todas
        # dentro da elocução.
        partes.append(rng.normal(0.0, 0.017, size=bloco).astype(np.float32))
        partes.append(np.full(bloco, 0.02, dtype=np.float32))
    partes.append(rng.normal(0.0, 0.017, size=int(2.0 * sr)).astype(np.float32))
    partes.append(np.zeros(int(0.5 * sr), dtype=np.float32))
    amostras = np.clip(np.concatenate(partes), -1.0, 1.0)
    with wave.open(str(path), "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(sr)
        dst.writeframes((amostras * 32767).astype(np.int16).tobytes())
    return transiente


def test_corta_o_degrau_inteiro_apesar_da_travessia(measure_phonemes, tmp_path):
    """O corte atravessa o bloco limpo que existe **dentro** do degrau.

    A versão contígua parava nele: cortava um bloco só, anunciava "10 ms de
    transiente cortados" e devolvia um arquivo que o `diagnose_signal`
    recusava em seguida — duas elocuções do corpus da E9 se perderam assim, e
    a mensagem da recusa mandava usar o `--record`, que é justamente quem
    acabou de cortar errado.
    """
    np = pytest.importorskip("numpy")
    mp = measure_phonemes
    wav = tmp_path / "abertura.wav"
    transiente = _degrau_com_travessia(wav)

    cortou = mp._trim_lead_in(wav)

    assert cortou == pytest.approx(transiente, abs=0.02), (
        f"cortou {cortou * 1000:.0f} ms de {transiente * 1000:.0f} ms de transiente"
    )
    # O que importa não é o número de milissegundos, é o veredito de quem
    # recusa: o mesmo diagnóstico que rejeitou as duas elocuções.
    import wave

    with wave.open(str(wav), "rb") as src:
        sobrou = (
            np.frombuffer(src.readframes(src.getnframes()), dtype=np.int16).astype(np.float32)
            / 32768.0
        )
    assert mp.diagnose_signal(sobrou, np) is None, "o diagnóstico ainda recusa o arquivo cortado"
    # E a fala continua lá: um corte que passasse a peneira comendo a
    # elocução também faria o diagnóstico calar.
    assert len(sobrou) / 16000 == pytest.approx(2.5, abs=0.05)


def test_atravessa_travessia_de_dois_blocos(measure_phonemes, tmp_path):
    """A travessia pode ocupar dois blocos, e o corte ainda chega ao fim.

    `GAP_MAX_BLOCKS` vale 2 justamente para isso: a travessia cabe em um bloco
    na resolução de 10 ms e o segundo é a margem. Sem este caso, o número 2
    seria uma afirmação sem medição por trás.
    """
    pytest.importorskip("numpy")
    mp = measure_phonemes
    wav = tmp_path / "travessia-dupla.wav"
    transiente = _degrau_com_travessia(wav, travessia_blocos=2)

    cortou = mp._trim_lead_in(wav)

    assert cortou == pytest.approx(transiente, abs=0.02), (
        f"cortou {cortou * 1000:.0f} ms de {transiente * 1000:.0f} ms de transiente"
    )


def test_nao_estende_o_corte_por_bloco_enviesado_na_fala(measure_phonemes, tmp_path):
    """Um bloco de fala dominado por DC **não** puxa o corte para dentro dela.

    É o perigo que a contiguidade protegia, e atravessar toda falha curta o
    reintroduziu: numa gravação de verdade o corte foi de 490 ms para 940 ms
    comendo o começo da elocução, e no alinhamento isso volta como deleção de
    fonema — a aplicação acusaria um erro que o aluno não cometeu. A travessia
    é atravessada uma vez e dentro da janela do onset; o que aparece depois é
    fala.
    """
    pytest.importorskip("numpy")
    mp = measure_phonemes
    wav = tmp_path / "vies-na-fala.wav"
    transiente = _degrau_com_travessia(wav, vies_na_fala=True)

    cortou = mp._trim_lead_in(wav)

    assert cortou == pytest.approx(transiente, abs=0.001), (
        f"cortou {cortou * 1000:.0f} ms, {(cortou - transiente) * 1000:.0f} ms "
        "além do transiente — isso é fala"
    )


def test_guarda_o_bruto_antes_de_cortar(measure_phonemes, tmp_path):
    """O WAV de antes do corte fica num irmão `.bruto.wav`.

    O corte é uma heurística sobre um dado que custou alguém falar, e
    reescrever no lugar torna um corte errado irreversível: reverter o código
    não devolve a amostra. Já cobrou uma vez — quando a primeira versão desta
    função cortou 10 ms de um degrau de 980 ms, o original tinha sido
    reescrito e a falha não pôde ser reproduzida com o áudio de verdade.
    """
    pytest.importorskip("numpy")
    mp = measure_phonemes
    import wave

    wav = tmp_path / "abertura.wav"
    _degrau_com_travessia(wav)
    with wave.open(str(wav), "rb") as src:
        antes = src.readframes(src.getnframes())

    cortou = mp._trim_lead_in(wav)
    assert cortou > 0, "sem corte este teste não prova nada"

    bruto = tmp_path / "abertura.bruto.wav"
    assert bruto.is_file(), "o bruto não foi guardado"
    with wave.open(str(bruto), "rb") as src:
        assert src.readframes(src.getnframes()) == antes, "o bruto não é o áudio de antes"
    with wave.open(str(wav), "rb") as src:
        assert len(src.readframes(src.getnframes())) < len(antes), "o arquivo não foi cortado"


def _dois_degraus(path, sr: int = 16000) -> None:
    """Um WAV em que a **segunda** passada do corte também corta.

    Não é hipótese: aconteceu nesta base. O `record` cortou 10 ms de um degrau
    de 980 ms, e rodar o mesmo corte sobre o resultado tirou os 970 ms que
    tinham sobrado — porque cortar o começo traz o resto do degrau para dentro
    da janela do onset. O perfil aqui reproduz isso: um degrau curto, uma
    falha longa demais para ser atravessada, e outro degrau.
    """
    import wave

    import numpy as np

    bloco = sr // 100
    partes = [np.full(bloco, -0.3, dtype=np.float32)]
    travessia = np.full(bloco, -1.0, dtype=np.float32)
    travessia[: int(bloco * 0.6)] = 0.78
    partes.append(travessia)
    partes += [np.full(bloco, -0.9, dtype=np.float32) for _ in range(3)]
    # Falha maior que `GAP_MAX_BLOCKS`: a primeira passada para aqui.
    rng = np.random.default_rng(11)
    partes += [rng.normal(0.0, 0.017, size=bloco).astype(np.float32) for _ in range(5)]
    partes += [np.full(bloco, -0.8 * (0.9**i), dtype=np.float32) for i in range(10)]
    partes.append(rng.normal(0.0, 0.017, size=int(1.0 * sr)).astype(np.float32))
    amostras = np.clip(np.concatenate(partes), -1.0, 1.0)
    with wave.open(str(path), "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(sr)
        dst.writeframes((amostras * 32767).astype(np.int16).tobytes())


def test_segunda_passada_nao_sobrescreve_o_bruto(measure_phonemes, tmp_path):
    """Remedir não troca o bruto pelo arquivo já cortado.

    Numa segunda passada o original de verdade é o da primeira. Sobrescrever
    guardaria uma cópia do já cortado e perderia exatamente o que o irmão
    existe para preservar — com a agravante de parecer que está tudo guardado.

    O perfil tem dois degraus de propósito: com um só, a segunda passada não
    acha onset, devolve 0 e nem chega a escrever o bruto, então o teste
    passaria sem a guarda e não provaria nada. Foi o que ele fez na primeira
    versão.
    """
    pytest.importorskip("numpy")
    mp = measure_phonemes
    import wave

    wav = tmp_path / "abertura.wav"
    _dois_degraus(wav)
    with wave.open(str(wav), "rb") as src:
        original = src.readframes(src.getnframes())

    primeiro = mp._trim_lead_in(wav)
    segundo = mp._trim_lead_in(wav)
    assert primeiro > 0 and segundo > 0, (
        f"as duas passadas precisam cortar para o teste valer: {primeiro}, {segundo}"
    )

    with wave.open(str(tmp_path / "abertura.bruto.wav"), "rb") as src:
        assert src.readframes(src.getnframes()) == original, (
            "a segunda passada sobrescreveu o bruto com o arquivo já cortado"
        )


def _relatorio(path, dito: str, *, errado: bool, per: float = 0.1) -> None:
    import json

    dados = {"dito": dito, "per": per, "esperado": ["a"], "ouvido": ["a"]}
    if errado:
        dados["errado"] = True
    path.write_text(json.dumps(dados))


def test_as_duas_passadas_nao_dispensam_uma_a_outra(measure_phonemes, tmp_path, monkeypatch):
    """A tomada correta e a com erro de propósito são a **mesma frase**.

    Sem separar as passadas, a correta já limpa no disco fazia o `--corpus
    --errado` pular a regra — e vice-versa —, de modo que a passada que decide
    o portão nunca seria gravada. O sintoma seria "já medida e limpa", que é
    exatamente o que ninguém investiga.
    """
    mp = measure_phonemes
    monkeypatch.setattr(mp, "RECORDING_DIR", tmp_path)
    frase = mp.SUBSTITUTIONS[0].phrase
    _relatorio(tmp_path / "certo.json", frase, errado=False)

    assert mp._clean_report_for(frase, wrong=False) is not None, "a correta devia dispensar a si"
    assert mp._clean_report_for(frase, wrong=True) is None, (
        "a tomada correta dispensou a passada errada, que então nunca é gravada"
    )

    _relatorio(tmp_path / "errado.json", frase, errado=True)
    assert mp._clean_report_for(frase, wrong=True) is not None


def test_passada_errada_grava_em_arquivo_proprio(measure_phonemes):
    """O WAV da passada errada não sobrescreve o da correta.

    Mesmo nome de arquivo significaria que gravar a passada errada apaga a
    elocução correta já medida — e ela é dado que custou alguém falar.
    """
    mp = measure_phonemes
    gravadas: list[str] = []

    def fake_record(seconds, out):
        gravadas.append(out.name)
        return out

    def fake_run_audio(recognizer, g2p, path, said, rule_id=None, wrong=False):
        assert wrong is True, "a passada errada tem de marcar o relatório"
        return {"regra": rule_id, "per": 0.4, "esperado": ["a"], "ouvido": ["b"]}

    with (
        mock.patch.object(mp, "record", fake_record),
        mock.patch.object(mp, "run_audio", fake_run_audio),
        mock.patch.object(mp, "_clean_report_for", lambda phrase, wrong=False: None),
        mock.patch("builtins.input", side_effect=EOFError),
    ):
        assert mp.run_corpus(object(), object(), 6, redo=True, wrong=True) == 0

    esperados = [f"e9-{item.rule_id.split('.', 1)[-1]}-errado.wav" for item in mp.SUBSTITUTIONS]
    assert gravadas == esperados


def test_passada_errada_sem_medicao_nao_conclui_nada(measure_phonemes, capsys):
    """Zero medidas não são zero acertos.

    Sem a guarda, uma passada em que tudo foi pulado caía no ramo do "não
    nomeou em nenhuma" e anunciava que o avaliador não faz o trabalho — uma
    conclusão sobre o portão da §3.3 tirada de nenhum dado. É o inverso do
    erro que a própria E8 evita ao recusar sinal ruim em vez de devolver um
    número que parece medição.
    """
    mp = measure_phonemes

    assert mp._report_corpus([], wrong=True) == 0
    saida = capsys.readouterr().out
    assert "não há o que concluir" in saida
    assert "não fazendo o" not in saida, "concluiu sobre o avaliador sem medir nada"
