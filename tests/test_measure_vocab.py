"""A pontuação da medição de vocabulário, que é onde as conclusões moram.

O `measure_vocab.py` não decide nada sozinho — ele conta. Então o que precisa
de teste não é o laço que fala com o modelo, é o que transforma a prosa do
professor em número: se a contagem erra, a conclusão erra com ela, e erra
calada.

Não é hipótese. A primeira versão do padrão de rótulo pedia `\\bfigurative\\b`
e perdeu "used figuratively", então a tabela anunciou **0 de 10** rótulos num
conjunto que trazia um. Zero era a conclusão que a medição queria ouvir, e foi
por isso que passou. Cada teste aqui fixa um caso que o padrão já errou ou
erraria.

Importar um script em vez de um módulo do pacote é exceção nesta suíte, e vale
pelo mesmo motivo do `test_measure_phonemes.py`: o topo do arquivo só traz o
pacote e a stdlib, e nada aqui carrega modelo, espeak ou Ollama.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_vocab.py"


@pytest.fixture(scope="module")
def vocab():
    spec = importlib.util.spec_from_file_location("measure_vocab", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["measure_vocab"] = module
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        del sys.modules["measure_vocab"]


# --- o IPA, que é o único sinal com gabarito ------------------------------- #


def test_acha_a_transcricao_que_o_modelo_digitou(vocab):
    reply = 'The word "thorough" is pronounced /ˈθʌroʊ/ in American English.'
    assert vocab.typed_ipa(reply) == "ˈθʌroʊ"


def test_colchete_tambem_conta_como_transcricao(vocab):
    assert vocab.typed_ipa("Say it like [fˈɑːdɚ] when you read it.") == "fˈɑːdɚ"


def test_barra_sem_simbolo_de_ipa_nao_e_transcricao(vocab):
    """`he/she or they/them` não é pronúncia, e sem a guarda viraria.

    É o motivo de `_IPA_CHARS` ser exigido. Duas barras na prosa bastam para
    o delimitador fechar sozinho, e uma falsa transcrição aqui inventaria
    divergência com a linha da tela — que é justamente o que esta medição
    afirma ter medido. Uma barra só não serve de teste: o padrão precisa de
    par para casar, então o caso de uma barra passa mesmo sem a guarda.
    """
    reply = 'You can say "he/she" or "they/them" when the person is unknown.'
    assert vocab.typed_ipa(reply) is None


def test_prosa_sem_transcricao_nenhuma_devolve_none(vocab):
    assert vocab.typed_ipa('"House" means a building where people live.') is None


def test_notacao_diferente_nao_conta_como_discordancia(vocab):
    """Tônica, comprimento e delimitador são notação, não conteúdo.

    A comparação é frouxa de propósito: acusar divergência onde só há estilo
    de transcrição transformaria a medição num gerador de falsos positivos
    contra o espeak.
    """
    signals = vocab.describe("It is /fɑdɚ/, roughly.", "fˈɑːdɚ", "fodder", "fˈɑːdɚ")
    assert signals["ipa_typed"] == "fɑdɚ"
    assert signals["ipa_agrees"] is True


def test_simbolo_diferente_e_discordancia(vocab):
    """O caso que importa: a tela contra o alto-falante, na mesma bolha.

    `schedule` é o exemplo real — a linha da tela sai do espeak em US, e um
    modelo que digita a pronúncia UK põe duas transcrições incompatíveis à
    vista do aluno.
    """
    signals = vocab.describe("In Britain, /ˈʃɛdjuːl/.", "skˈɛdʒuːl", "schedule", "skˈɛdʒuːl")
    assert signals["ipa_agrees"] is False


def test_sem_transcricao_o_veredito_e_ausente_e_nao_falso(vocab):
    """`None` e `False` dizem coisas diferentes, e somar os dois mentiria.

    Não digitar IPA é o comportamento certo; digitar errado é o problema. Se
    a ausência virasse `False`, a linha de resumo contaria toda resposta
    silenciosa como divergência.
    """
    signals = vocab.describe('"House" means a building.', "hˈaʊs", "house", "hˈaʊs")
    assert signals["ipa_agrees"] is None


# --- os sinais de prosa, todos grosseiros e por isso fixados --------------- #


def test_rotulo_pega_figuratively_e_nao_so_figurative(vocab):
    """O bug que esta suíte existe para não repetir.

    `\\bfigurative\\b` não casa com "figuratively", e foi assim que a primeira
    rodada contou 0 rótulos numa resposta que rotulava.
    """
    signals = vocab.describe("It's used figuratively to describe people.", "", "fodder", "fˈɑːdɚ")
    assert signals["labels"] == ["figuratively"]


@pytest.mark.parametrize(
    "reply",
    [
        "This is a formal word.",
        "It sounds literary today.",
        "The term is old-fashioned.",
        "This is a C1 word.",
    ],
)
def test_rotulo_de_registro_e_de_nivel_conta(vocab, reply):
    assert vocab.describe(reply, "", "x", "x")["labels"]


def test_portugues_pelo_acento(vocab):
    """A decisão 1.1 é 100% em inglês, e glosar palavra é a maior tentação."""
    assert vocab.describe('"Fodder" é forragem.', "", "fodder", "fˈɑːdɚ")["pt"] is True


def test_portugues_pelo_marcador_sem_acento(vocab):
    assert vocab.describe('"Fodder" significa forragem', "", "f", "f")["pt"] is True


def test_ingles_limpo_nao_acusa_portugues(vocab):
    reply = '"Fodder" is food for farm animals, like hay or straw.'
    assert vocab.describe(reply, "", "fodder", "fˈɑːdɚ")["pt"] is False


@pytest.mark.parametrize(
    "reply",
    [
        "Now repeat after me: thorough.",
        "Say it out loud and I will listen.",
        "Let me hear your pronunciation.",
    ],
)
def test_alegacao_de_ouvir_e_pega(vocab, reply):
    """O que o CLAUDE.md proíbe nos dois estados de voz: a E9 é portão aberto.

    Nada do que chega ao professor é som — é transcrição —, então oferecer
    escuta é prometer um produto que não existe.
    """
    assert vocab.describe(reply, "", "x", "x")["hears"] is True


def test_convite_a_praticar_por_escrito_nao_e_alegacao_de_ouvir(vocab):
    reply = 'Try writing a sentence with "thorough" and I will check it.'
    assert vocab.describe(reply, "", "thorough", "θˈʌɹoʊ")["hears"] is False


def test_conta_os_exemplos_entre_aspas(vocab):
    reply = '"Fodder" is animal food. For example, "the fodder was wet".'
    assert vocab.describe(reply, "", "fodder", "fˈɑːdɚ")["examples"] == 2


# --- o corpus, contado por frase e não por cue ----------------------------- #

# Uma frase partida em duas cues, que é o caso comum num arquivo de legenda:
# a tela acaba antes da frase. Contar cue daria dois exemplos onde há um.
SPLIT = """1
00:00:01,000 --> 00:00:03,000
Her marriage became fodder

2
00:00:03,200 --> 00:00:05,000
for sentimental narrative.

3
00:00:10,000 --> 00:00:12,000
[horse neighs]

4
00:00:13,000 --> 00:00:15,000
We bought hay instead.
"""


def test_frase_partida_em_duas_cues_conta_uma_vez(vocab, tmp_path):
    show = tmp_path / "show"
    show.mkdir()
    (show / "show-1x01.en.srt").write_text(SPLIT, encoding="utf-8")

    hits, total, example = vocab.corpus_hits("fodder", root=tmp_path)

    assert hits == 1
    assert example == "Her marriage became fodder for sentimental narrative."
    # Duas frases de fala: a cue do `[horse neighs]` não é fala de ninguém.
    assert total == 2


def test_palavra_ausente_devolve_zero_e_nenhum_exemplo(vocab, tmp_path):
    show = tmp_path / "show"
    show.mkdir()
    (show / "show-1x01.en.srt").write_text(SPLIT, encoding="utf-8")

    assert vocab.corpus_hits("schedule", root=tmp_path) == (0, 2, None)


def test_corpus_ausente_nao_derruba_a_medicao(vocab, tmp_path):
    """A metade determinística tem que rodar em quem clonou sem corpus.

    `corpus/*` não é versionado, então na máquina de quem clonou a pasta não
    existe — e a coluna da base curada, que é o achado principal, não depende
    dela.
    """
    assert vocab.corpus_hits("fodder", root=tmp_path / "nao-existe") == (0, 0, None)


def test_a_lista_de_palavras_cobre_os_dois_lados_do_corte(vocab):
    """Metade com item curado e metade sem — é o corte que mostra o achado.

    Se a lista virasse só palavras sem item curado, a coluna vazia da base
    seria consequência da escolha das palavras, e não da base ser cega a uma
    pergunta.
    """
    probes = dict(vocab.WORDS)
    curated = [w for w, why in probes.items() if "curado" in why]
    assert len(curated) == 5
    assert len(probes) == 10
