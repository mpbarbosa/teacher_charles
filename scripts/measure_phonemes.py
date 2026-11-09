#!/usr/bin/env python3
"""Mede o reconhecedor de fonemas da Fase 2 nesta máquina, antes de confiar nele.

A Fase 1.5 abriu por uma medição e não por integração (plano 1.5 §2), e deu
certo: a medição mudou o desenho em quatro lugares antes de haver código para
jogar fora. A Fase 2 abre do mesmo jeito, e por uma razão mais forte — a
análise inicial §3 chamou a avaliação de pronúncia de "a parte com mais
engenharia", e ela depende de uma coisa que nenhuma documentação responde:

    o piso de erro do reconhecedor é menor que o erro que ele precisa detectar?

Um avaliador de pronúncia compara o que o aluno falou com o que ele deveria
ter falado, fonema por fonema. Se o próprio reconhecedor erra 20% dos fonemas
numa fala **correta**, ele não pode acusar uma troca de θ por t: o sinal fica
dentro do ruído, e o aluno recebe correção de erro que não cometeu — a mesma
classe de falha que a E4 existe para evitar do lado do modelo de linguagem, e
que a §3.2 da Fase 1.5 evita no STT mandando a transcrição para o campo de
texto. Aqui não há campo de texto onde se escondar: ou o piso é baixo, ou a
feature não existe.

Modos:

    uv run python scripts/measure_phonemes.py --rtf
        Carga e fator de tempo real do reconhecedor em falas de 3, 6 e 10
        segundos. O orçamento é o da §3.1 da Fase 1.5, que já está estourado:
        qualquer coisa aqui entra depois do turno, nunca dentro dele.

    uv run python scripts/measure_phonemes.py --floor
        O piso: taxa de erro de fonema (PER) do reconhecedor contra fala
        sintética **correta**. É o número que decide a fase.

    uv run python scripts/measure_phonemes.py --discriminate
        O sinal: o mesmo PER contra fala com a substituição típica de
        brasileiro injetada no nível do fonema. Compara com o piso e diz se as
        duas distribuições se separam.

    uv run python scripts/measure_phonemes.py --record 5 --say "the red car is ready"
        Grava 5 segundos do microfone desta máquina, deixa o WAV no disco e
        mede em cima dele. É a E9 do plano da Fase 2 — o portão que decide se
        o avaliador de pronúncia é viável.

    uv run python scripts/measure_phonemes.py --audio fala.wav --say "what I said"
        O mesmo, sobre um WAV que já existe. É assim que se mede o que importa.

    uv run python scripts/measure_phonemes.py --rescore
        Responde outra vez "o erro previsto foi nomeado?" sobre os relatórios
        que já estão no disco — sem microfone, sem áudio e sem carregar os 318
        MB do reconhecedor. É a §3.4 do plano aplicada: a assinatura de um item
        é um **conjunto** de operações aceitáveis, e abrir o lado ouvido do
        gabarito à classe do fonema não pede gravação nova, porque a resposta
        está em `esperado` e `ouvido`.

## O que esta medição NÃO diz

O áudio das duas tabelas é **sintético**: o Kokoro lê uma transcrição de
fonemas, então a substituição é exata e o resto da fala é idêntico entre os
dois lados do par. Isso é bom para a pergunta desta medição — isola a troca de
fonema de timbre, prosódia e ruído, que é a única forma de atribuir a
diferença de PER à troca e não ao falante.

Não é a mesma coisa que um aluno brasileiro falando num microfone de notebook.
Fala humana tem coarticulação, hesitação e um sotaque que não é um conjunto de
substituições discretas: um brasileiro que troca θ por t também encurta a
vogal seguinte e muda o ritmo da frase. **Esta medição é o teto do
desempenho**, não a expectativa — exatamente como os 0% de WER da §2.3 da Fase
1.5 são piso e não expectativa. ``--audio`` existe para quem for fazer a
medição que importa.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# O reconhecedor é o `facebook/wav2vec2-lv-60-espeak-cv-ft` exportado para
# ONNX pela onnx-community, na variante int8. A escolha não é de qualidade, é
# de orçamento, e tem três razões:
#
# 1. **Zero wheel nova.** `onnxruntime` já está no extra `voice` por causa do
#    Kokoro, e `phonemizer` + `espeakng_loader` vêm com o `kokoro-onnx`. O
#    caminho oficial (transformers + torch) acrescentaria ~2 GB de wheels a um
#    projeto que fixou o interpretador em 3.12 para caber nos wheels que já
#    tem (plano §1.3).
# 2. **int8 na CPU é o padrão da casa.** O Whisper roda int8 e o Kokoro roda
#    int8. 318 MB contra 1.264 MB do fp32.
# 3. **O alfabeto casa.** Este modelo emite IPA do espeak, que é exatamente o
#    que o `phonemizer` desta máquina produz para o texto esperado. Sem isso
#    seria preciso um mapa entre dois inventários de fonema, e todo mapa desses
#    é uma fonte de erro que não se distingue de erro do modelo.
HF_REPO = "onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX"
HF_MODEL_FILE = "onnx/model_int8.onnx"

# O wav2vec2 foi treinado em 16 kHz, igual ao Whisper.
SAMPLE_RATE = 16_000

# Voz inglesa, para os dois lados do par: o que muda entre eles é a
# transcrição de fonemas, e só ela.
KOKORO_VOICE = "bm_george"

# O proxy de sotaque: uma voz brasileira do Kokoro lendo o **texto** inglês
# pelo G2P de português. Não é um aluno brasileiro — é a fonologia do
# português aplicada à ortografia do inglês, que é um pedaço do que o aluno
# faz e não o todo (falta coarticulação, hesitação e ritmo). Serve para uma
# pergunta só, e é a pergunta certa: diante de fala que não é um par
# controlado, o reconhecedor continua nomeando os erros da lista curada, ou
# desaba num PER alto que não ensina nada?
ACCENT_VOICE = "pm_alex"
ACCENT_LANG = "pt-br"


# ---------------------------------------------------------------------------
# As substituições
# ---------------------------------------------------------------------------


# Cada item é um erro de pronúncia documentado de brasileiro falando inglês, e
# vira um par: a mesma frase sintetizada com a transcrição correta e com a
# transcrição trocada. A lista não é exaustiva de propósito — ela cobre as
# classes que a análise inicial §3 nomeia (θ e vogal epentética) mais as que
# aparecem em toda bibliografia de interferência PT→EN.
#
# `swap` é aplicado sobre o IPA do espeak, não sobre a ortografia: trocar
# "think" por "tink" no texto faria o espeak mudar também a vogal, e a
# diferença de PER deixaria de ser atribuível à troca.
@dataclass(frozen=True, slots=True)
class Substitution:
    """Um erro típico, no nível do fonema."""

    rule_id: str
    phrase: str
    wrong: str
    right: str
    # O que o brasileiro faz: `(de, para)` sobre a string de IPA.
    swap: tuple[str, str]
    why: str


SUBSTITUTIONS: tuple[Substitution, ...] = (
    Substitution(
        "pron.th-unvoiced",
        "I think this is the best one",
        "tink",
        "think",
        ("θ", "t"),
        "O português não tem /θ/. A substituição mais comum é a oclusiva /t/.",
    ),
    Substitution(
        "pron.th-voiced",
        "They brought the other brother",
        "de udder brudder",
        "the other brother",
        ("ð", "d"),
        "Mesma ausência, lado sonoro: /ð/ vira /d/.",
    ),
    Substitution(
        "pron.epenthesis-initial",
        "Sport is good for stress",
        "isport, istress",
        "sport, stress",
        ("sp", "ɪsp"),
        "Português não começa sílaba com /sp/ nem /st/; entra um /i/ antes.",
    ),
    Substitution(
        "pron.epenthesis-final",
        "It was a big club night",
        "bigi clubi",
        "big, club",
        ("ɡ", "ɡi"),
        "Oclusiva final não existe em português; entra um /i/ depois.",
    ),
    Substitution(
        "pron.vowel-length",
        "The beach was full of people",
        "bitch",
        "beach",
        ("iː", "ɪ"),
        "Português não distingue /iː/ de /ɪ/, e o par mínimo aqui é caro.",
    ),
    Substitution(
        "pron.h-dropping",
        "He has a house here",
        "e as a ouse",
        "he has a house",
        ("h", ""),
        "O <h> português é mudo; o /h/ inglês desaparece ou vira /x/.",
    ),
    Substitution(
        "pron.ing-alveolar",
        "I am speaking and reading",
        "speakin, readin",
        "speaking, reading",
        ("ŋ", "n"),
        "/ŋ/ final não é fonema do português; vira /n/.",
    ),
    Substitution(
        "pron.r-initial",
        "The red car is ready",
        "hed, heady",
        "red, ready",
        ("ɹ", "h"),
        "O <r> inicial do português é fricativa glotal, não aproximante.",
    ),
)

# Falas de aluno para o fator de tempo real. Os mesmos três tamanhos da
# §2.2 da Fase 1.5, para a comparação com o Whisper ser direta.
RTF_UTTERANCES = (
    "I think the weather is nice today.",
    "Yesterday I went to the beach with my family and we stayed there all afternoon.",
    "I have been studying English for three years now, but I still feel nervous when "
    "I need to speak with a native speaker on the phone, especially at work.",
)


# ---------------------------------------------------------------------------
# G2P: o texto esperado, em fonemas
# ---------------------------------------------------------------------------


def _build_g2p():
    """O `phonemizer` desta máquina, apontado para o espeak-ng do wheel.

    Importa aqui dentro, e não no topo, pela mesma razão que o
    ``voice/__init__.py`` importa os motores dentro da função: o extra
    ``voice`` não é obrigatório.
    """
    import espeakng_loader
    from phonemizer.backend import EspeakBackend
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    # `with_stress=False`: o reconhecedor não emite marca de tônica, e
    # comparar com ela do lado esperado inventaria erro em toda palavra.
    return EspeakBackend("en-us", with_stress=False, preserve_punctuation=False)


def phonemize(g2p, text: str) -> str:
    """IPA do espeak para uma frase, sem marca de tônica."""
    return g2p.phonemize([text])[0].strip()


def tokenize(ipa: str, vocab: set[str]) -> list[str]:
    """Quebra a string de IPA nos símbolos que o modelo conhece.

    Casamento mais longo primeiro, porque IPA tem símbolos de dois caracteres
    (``tʃ``, ``iː``, ``aɪ``) cujo prefixo também é símbolo válido. Caractere
    que não está no inventário é descartado **e contado**: um descarte
    silencioso viraria erro de fonema indistinguível de erro do modelo, que é
    exatamente o que esta medição não pode se permitir.
    """
    out: list[str] = []
    i = 0
    longest = max((len(t) for t in vocab), default=1)
    while i < len(ipa):
        if ipa[i] == " ":
            i += 1
            continue
        for size in range(min(longest, len(ipa) - i), 0, -1):
            chunk = ipa[i : i + size]
            if chunk in vocab:
                out.append(chunk)
                i += size
                break
        else:
            out.append("?" + ipa[i])
            i += 1
    return out


# ---------------------------------------------------------------------------
# O reconhecedor
# ---------------------------------------------------------------------------


def _vocab_raw() -> dict[str, int]:
    """O `vocab.json` do reconhecedor, que são alguns kilobytes."""
    from huggingface_hub import hf_hub_download

    vocab_path = hf_hub_download(HF_REPO, "vocab.json")
    return json.loads(Path(vocab_path).read_text(encoding="utf-8"))


def load_vocab() -> set[str]:
    """O vocabulário de fonemas **sem** carregar o reconhecedor.

    O `_error_signature` precisa de duas coisas, e nenhuma delas é áudio: o
    G2P e o vocabulário em que a transcrição é tokenizada. Quem só vai
    responder a coluna do gabarito — o `--rescore` — não abre WAV nenhum, e
    pagar 318 MB e ~1,8 s de carga para ler um arquivo de poucos kilobytes
    seria o defeito que o `/health` da Fase 1.5 evita: efeito colateral de
    centenas de megabytes numa operação que não os pede.
    """
    return {t for t in _vocab_raw() if not (t.startswith("<") and t.endswith(">"))}


class Vocabulary:
    """O que o gabarito precisa do reconhecedor, que é só o vocabulário.

    Entra onde um `PhonemeRecognizer` entraria, e por isso tem o mesmo nome de
    atributo: o `_error_signature` nunca tocou em mais do que isto.
    """

    def __init__(self) -> None:
        self.vocab = load_vocab()


class PhonemeRecognizer:
    """O wav2vec2 de fonemas, em ONNX, na CPU.

    Decodificação CTC gulosa: sem beam, sem modelo de linguagem de fonema.
    Beam ajudaria a exatidão e **atrapalharia** o propósito — um decodificador
    que sabe que sequências de fonema são plausíveis em inglês conserta a
    pronúncia do aluno antes de ela chegar ao avaliador, que é o mesmo defeito
    que a análise inicial §3 aponta no Whisper ("tink" vira "think").
    """

    def __init__(self, threads: int = 0) -> None:
        import numpy as np
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        self._np = np
        t0 = time.perf_counter()
        model_path = hf_hub_download(HF_REPO, HF_MODEL_FILE)
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self._session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.load_ms = (time.perf_counter() - t0) * 1000

        raw: dict[str, int] = _vocab_raw()
        self._id_to_token = {i: t for t, i in raw.items()}
        self.vocab = {t for t in raw if not (t.startswith("<") and t.endswith(">"))}
        self.pad_id = raw.get("<pad>", 0)
        self._input_name = self._session.get_inputs()[0].name
        self.model = f"{HF_REPO}#{HF_MODEL_FILE}"

    def recognize(self, pcm: Any) -> tuple[list[str], float]:
        """Fonemas reconhecidos e milissegundos gastos.

        ``pcm`` é float32 mono em 16 kHz, no intervalo [-1, 1].
        """
        np = self._np
        audio = np.asarray(pcm, dtype=np.float32)
        # Normalização de média zero e variância um: é o que o
        # `preprocessor_config.json` do modelo pede (`do_normalize`), e sem
        # ela o modelo emite lixo — não um pouco pior, lixo.
        audio = (audio - audio.mean()) / (audio.std() + 1e-7)
        t0 = time.perf_counter()
        logits = self._session.run(None, {self._input_name: audio[None, :]})[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        ids = logits[0].argmax(axis=-1)
        # CTC: colapsa repetição e tira o branco.
        out: list[str] = []
        previous = -1
        for token_id in ids.tolist():
            if token_id != previous and token_id != self.pad_id:
                token = self._id_to_token.get(token_id, "")
                if token and not (token.startswith("<") and token.endswith(">")):
                    out.append(token)
            previous = token_id
        return out, elapsed_ms


# ---------------------------------------------------------------------------
# Alinhamento: PER e onde o erro cai
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Alignment:
    """O que separa o que se esperava do que se ouviu."""

    expected: tuple[str, ...]
    heard: tuple[str, ...]
    substitutions: tuple[tuple[str, str], ...]
    deletions: tuple[str, ...]
    insertions: tuple[str, ...]

    @property
    def errors(self) -> int:
        return len(self.substitutions) + len(self.deletions) + len(self.insertions)

    @property
    def per(self) -> float:
        """Taxa de erro de fonema. Zero é fala idêntica ao esperado."""
        return self.errors / max(len(self.expected), 1)


def align(expected: list[str], heard: list[str]) -> Alignment:
    """Levenshtein com backtrace, porque o PER sozinho não ensina nada.

    O avaliador de pronúncia precisa dizer **qual** fonema saiu errado, não
    quantos: "your /θ/ came out as /t/" é ensino, "82% accuracy" é placar. O
    backtrace é o que transforma um no outro, e é a razão de esta medição
    reportar a substituição mais frequente por par e não só a distância.
    """
    n, m = len(expected), len(heard)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if expected[i - 1] == heard[j - 1]:
                cost[i][j] = cost[i - 1][j - 1]
            else:
                cost[i][j] = 1 + min(cost[i - 1][j - 1], cost[i - 1][j], cost[i][j - 1])

    subs: list[tuple[str, str]] = []
    dels: list[str] = []
    ins: list[str] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and expected[i - 1] == heard[j - 1] and cost[i][j] == cost[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + 1:
            subs.append((expected[i - 1], heard[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            dels.append(expected[i - 1])
            i -= 1
        else:
            ins.append(heard[j - 1])
            j -= 1
    return Alignment(
        expected=tuple(expected),
        heard=tuple(heard),
        substitutions=tuple(reversed(subs)),
        deletions=tuple(reversed(dels)),
        insertions=tuple(reversed(ins)),
    )


def ops(alignment: Alignment) -> set[tuple[str, ...]]:
    """As operações do alinhamento, etiquetadas, para dois poderem se comparar.

    Substituição, inserção e deleção juntas num conjunto só. Sem isto a
    verificação de "o reconhecedor pegou o erro?" só sabia olhar substituição
    — e vogal epentética (*isport*, *bigi*), que é metade da lista de erros de
    brasileiro, **é inserção**. A primeira rodada desta medição reportou os
    dois itens de epêntese como não detectados por causa disso, com o PER
    subindo na coluna ao lado. Era defeito do critério, não resultado.
    """
    tagged: set[tuple[str, ...]] = {
        ("sub", expected, heard) for expected, heard in alignment.substitutions
    }
    tagged |= {("del", phoneme) for phoneme in alignment.deletions}
    tagged |= {("ins", phoneme) for phoneme in alignment.insertions}
    return tagged


# As classes de fonema que esta pilha não resolve, do lado **ouvido** de uma
# operação do gabarito. Não é a tabela de equivalência de inglês correto — essa
# é outra coisa, é da E10, e valeria para o alinhamento inteiro (`/ɔː/ ≡ /ɑː/`,
# a fusão *cot/caught* que a §3.5 do plano encontrou em fala correta). Esta
# aqui serve a uma pergunta só: a operação que o reconhecedor produziu **é o
# erro curado?** Logo não toca em PER, em alinhamento, nem no que conta como
# fala correta.
#
# A §3.4 do plano prescreve este conjunto e diz por quê: a identidade exata de
# uma vogal não é confiável aqui, a presença de uma vogal onde não devia haver
# é. O mesmo item foi medido errando o símbolo duas vezes — o par sintético da
# §2.3 ouviu `+/ɐ/` onde o gabarito dizia `+/ɪ/`, e a elocução humana
# deliberada ouviu `+/i/`.
PHONEME_CLASSES: tuple[frozenset[str], ...] = (frozenset({"ɪ", "i", "ɐ", "ə"}),)


def _class_of(phoneme: str) -> frozenset[str]:
    for klass in PHONEME_CLASSES:
        if phoneme in klass:
            return klass
    return frozenset({phoneme})


def widen(signature: set[tuple[str, ...]]) -> set[tuple[str, ...]]:
    """O gabarito com o lado ouvido aberto à classe do fonema.

    Só o lado ouvido. A origem vem do G2P sobre a frase curada e não se move;
    as posições e os tipos de operação continuam saindo das duas transcrições,
    que é a lição do `ops`. O que se abre é o que saiu da boca do aluno, porque
    o reconhecedor escreve `/i/` onde o gabarito diz `/ɪ/` e isso não é outro
    erro — é o mesmo, transcrito com o símbolo vizinho.

    Deleção não tem lado ouvido e passa intacta. E isto **não** entra no
    `run_pairs`: os números da §2 do plano foram medidos com a interseção
    estrita, e trocá-la lá redefiniria aquelas tabelas em silêncio.
    """
    wider: set[tuple[str, ...]] = set(signature)
    for op in signature:
        if op[0] == "sub":
            wider |= {("sub", op[1], alt) for alt in _class_of(op[2])}
        elif op[0] == "ins":
            wider |= {("ins", alt) for alt in _class_of(op[1])}
    return wider


# ---------------------------------------------------------------------------
# Síntese
# ---------------------------------------------------------------------------


class Synth:
    """O Kokoro lendo uma transcrição de fonemas, não um texto.

    ``is_phonemes=True`` é o que permite o par controlado: os dois lados saem
    da mesma voz, com a mesma prosódia, e diferem só no fonema trocado.
    """

    def __init__(self, settings) -> None:
        import numpy as np
        from kokoro_onnx import Kokoro

        self._np = np
        self._kokoro = Kokoro(str(settings.tts_model_path), str(settings.tts_voices_path))

    def speak_phonemes(self, ipa: str) -> Any:
        samples, rate = self._kokoro.create(
            ipa, voice=KOKORO_VOICE, speed=1.0, lang="en-us", is_phonemes=True
        )
        assert rate == 24_000, rate
        return self._resample(samples, 24_000, SAMPLE_RATE)

    def speak_text(self, text: str) -> Any:
        samples, rate = self._kokoro.create(text, voice=KOKORO_VOICE, speed=1.0, lang="en-us")
        return self._resample(samples, rate, SAMPLE_RATE)

    def speak_accented(self, text: str) -> Any:
        """O mesmo texto inglês, lido por voz brasileira pelo G2P de português."""
        samples, rate = self._kokoro.create(text, voice=ACCENT_VOICE, speed=1.0, lang=ACCENT_LANG)
        return self._resample(samples, rate, SAMPLE_RATE)

    def _resample(self, samples, source_rate: int, target_rate: int):
        """Reamostragem linear.

        Suficiente aqui, e a razão é medida, não preguiça: o wav2vec2
        reamostra internamente de qualquer jeito, e a alternativa seria
        acrescentar `scipy` ou `librosa` — dezenas de MB de wheel — a um
        projeto cujo orçamento de wheels é a razão de o modelo ser int8.
        """
        np = self._np
        if source_rate == target_rate:
            return samples.astype(np.float32)
        duration = len(samples) / source_rate
        target_n = int(duration * target_rate)
        source_x = np.arange(len(samples), dtype=np.float64)
        target_x = np.linspace(0, len(samples) - 1, target_n)
        return np.interp(target_x, source_x, samples).astype(np.float32)


# Pisos de sanidade do sinal. Não são números de laboratório: saíram da
# primeira gravação de verdade feita com este script, nesta máquina, que veio
# com +42 dB de ganho analógico (`Capture` em +30, `Internal Mic Boost` em
# +12) e produziu um arquivo cravado no fundo da escala, com 36% da energia
# abaixo de 100 Hz e picos em 8-31 Hz. O reconhecedor respondeu a isso
# devolvendo fonema nenhum — e o script reportou **PER 1,000**, que parece
# resultado e não é.
#
# É a mesma classe de erro que o piso de bytes do `voice/whisper.py` evita do
# outro lado (Whisper alucinando "Thank you." em silêncio): um número que o
# modelo produz honestamente e que não significa o que parece significar. A
# §3.3 do plano da Fase 2 chama isso pelo nome e dá a regra — **calar é
# resposta**. Aqui calar é recusar o arquivo com o diagnóstico, em vez de
# publicar um PER que iria para uma tabela.
CLIPPED_FRACTION_MAX = 0.005
LOW_FREQ_FRACTION_MAX = 0.50
RMS_MIN = 0.001
# Abaixo disto o arquivo ainda é áudio, e ainda é ruim: quantização grosseira,
# relação sinal-ruído pobre, consoante surda indistinguível do piso de ruído.
# Recusar seria errado (o arquivo tem fala, e o modelo talvez acerte), calar
# também — um corpus de dez elocuções gravado baixo é dez elocuções
# desperdiçadas, e isso só se descobre depois de medir as dez. Daí ser aviso e
# não recusa: a decisão é de quem gravou.
RMS_LOW = 0.010
PEAK_LOW_DBFS = -20.0
BLOCK_SECONDS = 0.1
# Quanto vale um ponto percentual de `Capture` nesta classe de máquina, para o
# aviso de nível poder dizer de quanto é o buraco em vez de sugerir um número
# redondo. Ajustado em dois pontos do mixer deste notebook — passo 63 → +30,00
# dB e passo 25 → +1,50 dB, logo 0,75 dB por passo — e os 63 passos cobrem os
# 100%: 0,63 passo por ponto. A fórmula previu passo 38 → +11,25 dB antes de a
# máquina confirmar, que é o teste que ela tinha de passar.
ALSA_DB_PER_PERCENT = 0.47
# Uma elocução tem pausa; a sala não. Daí a última pergunta do diagnóstico ser
# sobre **contraste**, a razão entre o percentil 90 e o 10 dos blocos: medido,
# 12 na gravação que transcreveu limpo e 2,8 na que o microfone só ouviu a
# sala. É a pergunta mais fundamental das cinco, porque não depende de nível
# nenhum — e foi a última a aparecer porque só se revela quando o ganho já
# está certo e o PER sai errado de todo jeito.
SPEECH_CONTRAST_MIN = 4.0
# O dispositivo rabeia ao abrir, e isso entra em **toda** gravação: um degrau
# de DC positivo que decai por metade a cada ~0,1s — média de bloco +0,585 em
# 0,0s, +0,073 em 0,4s, +0,002 em 1,0s, com o silêncio do arquivo em 0,004.
# Não é fala, é o conversor ligando.
#
# O corte é **por decaimento, não por duração**, e isso foi aprendido caro: com
# 1,0s fixo, a primeira elocução de verdade da E9 saiu com o `/ð ə/` de "the"
# comido, porque o aluno fez o que a mensagem pedia e falou no primeiro
# instante — dentro do segundo que o corte joga fora. Dois fonemas de "erro"
# que eram do script, na frase de `pron.r-initial`; em `pron.epenthesis-initial`,
# cuja regra é justamente sobre o começo da sílaba, teria invalidado a medição.
#
# O discriminante é o mesmo da recusa: num degrau de DC a média do bloco vale
# quase o RMS dele, e em fala a média é quase zero. Então corta-se enquanto o
# bloco for dominado por DC, com um teto para não comer a elocução inteira se
# algo der errado.
LEAD_IN_MAX_SECONDS = 1.0
DC_DOMINANCE = 0.5
# Em quanto tempo o degrau tem de começar para contar como transiente de
# abertura. Existe porque o degrau não começa necessariamente na amostra zero:
# medido, um bloco de 10 ms limpo antes dele.
ONSET_WINDOW_SECONDS = 0.1
# Quantos blocos não dominados a travessia do zero pode ocupar. Não é folga
# de segurança: o bloco em que o degrau **atravessa o zero** tem média quase
# nula contra um RMS enorme e não passa pelo `DC_DOMINANCE`, então um corte
# puramente contíguo para dentro do próprio degrau. Dois blocos porque a
# travessia cabe em um na resolução de 10 ms, e o segundo é a margem. O que
# impede o corte de andar dentro da fala não é este número, é a travessia ser
# atravessada **uma vez só** — ver `_trim_lead_in`.
GAP_MAX_BLOCKS = 2


@dataclass(frozen=True, slots=True)
class Problem:
    """Por que este áudio não serve, e se a nota de ganho tem a ver.

    A nota existe porque o ganho é a causa comum nesta máquina, mas ela manda
    **baixar** o ganho — imprimi-la junto de uma recusa que não é de ganho
    manda o contrário do remédio. O transiente de abertura é o caso: baixar o
    ganho por causa dele piora a fala, que já estava baixa.
    """

    message: str
    gain_related: bool


def block_levels(samples: Any, np: Any, rate: int = SAMPLE_RATE) -> tuple[Any, Any, Any]:
    """RMS e pico por bloco de 100 ms, e quais blocos são fala.

    Um bloco é fala quando passa de quatro vezes o **piso de ruído**, estimado
    pelo décimo percentil dos blocos. As duas âncoras óbvias erram, e as duas
    foram medidas neste arquivo:

    * **pico** — um estouro no começo (o transiente de ``LEAD_IN_SECONDS``,
      0,36 de RMS contra 0,07 da fala) põe o limiar acima da própria fala, e a
      fala toda vira silêncio;
    * **mediana** — só serve se o silêncio for a maioria. Aqui a gravação é
      meio a meio, a mediana cai em 0,031 no vão entre os dois modos, e quatro
      quintos da fala ficam de fora.

    O piso não tem esse problema: ele mede a parte calada, que é a única que
    não depende de quanto o aluno falou nem de quão alto.

    Por bloco, e não sobre o arquivo, porque toda pergunta sobre nível quer a
    elocução e não a gravação: o silêncio em volta dilui o RMS global e o
    transiente o infla. Daí os dois lados — ``speech_span`` e
    ``warn_if_quiet`` — saírem daqui.
    """
    block = int(rate * BLOCK_SECONDS)
    if len(samples) < block * 3:
        return None, None, None
    count = len(samples) // block
    shaped = samples[: count * block].reshape(count, block)
    rms = np.sqrt((shaped**2).mean(axis=1))
    peak = np.abs(shaped).max(axis=1)
    floor = float(np.percentile(rms, 10))
    loud = np.flatnonzero(rms >= max(4 * floor, 0.002))
    return rms, peak, loud


def speech_span(samples: Any, np: Any, rate: int = SAMPLE_RATE) -> tuple[float, float]:
    """Primeiro e último instante com energia de fala, em segundos.

    É a extensão dos blocos de fala, buracos inclusive — uma frase tem pausa
    entre palavras, e recortar a pausa daria uma lista de ilhas onde o que se
    quer é "a fala está entre aqui e aqui".
    """
    _, _, loud = block_levels(samples, np, rate)
    if loud is None or len(loud) == 0:
        return 0.0, len(samples) / rate
    return float(loud[0]) * BLOCK_SECONDS, float(loud[-1] + 1) * BLOCK_SECONDS


def diagnose_signal(samples: Any, np: Any, rate: int = SAMPLE_RATE) -> Problem | None:
    """O que está errado com este áudio, ou ``None`` se ele serve.

    Quatro perguntas, nesta ordem, e todas respondidas antes de o modelo ver
    um sample — porque um reconhecedor não tem como distinguir "o aluno falou
    isso" de "o conversor saturou", e quem recebe o PER depois não tem como
    distinguir uma coisa da outra tampouco.
    """
    if len(samples) == 0:
        return Problem("o arquivo não tem amostra nenhuma", gain_related=False)

    rms = float(np.sqrt((samples**2).mean()))
    if rms < RMS_MIN:
        return Problem(
            f"silêncio: RMS {rms:.6f}. O microfone gravou, mas não havia som.",
            gain_related=True,
        )

    # Um degrau de DC no começo, que é o dispositivo abrindo. Houve aqui uma
    # recusa por offset DC do arquivo inteiro, e ela media a coisa errada: o
    # offset global de +0,031 deste microfone **era** este transiente diluído
    # em cinco segundos, não um viés do conversor — cortado o primeiro
    # segundo, a média cai para +0,0007. Pior, subtrair aquela média global
    # empurrava o silêncio de 0,004 para 0,031 e afogava toda estatística de
    # nível depois.
    #
    # O discriminante é que **fala tem média zero** e um degrau não: por isso
    # a pergunta é sobre a média do começo, não sobre a do arquivo, e um aluno
    # que fale imediatamente não dispara nada. Refazer é mais honesto que
    # remendar — `record` já corta `LEAD_IN_SECONDS` por isso.
    head = samples[: int(0.3 * rate)]
    tail = samples[int(0.3 * rate) :]
    if len(tail) and abs(float(head.mean())) > max(0.02, 2 * float(np.sqrt((tail**2).mean()))):
        return Problem(
            f"transiente de abertura: os primeiros 0,3s têm média {float(head.mean()):+.3f}, "
            "um degrau de DC e não som. Fala tem média zero; o conversor ligando não. "
            "Corte o começo — o `--record` corta o transiente sozinho.",
            gain_related=False,
        )

    clipped = float((np.abs(samples) >= 0.999).mean())
    if clipped > CLIPPED_FRACTION_MAX:
        return Problem(
            f"saturado: {100 * clipped:.2f}% das amostras no fundo da escala "
            f"(teto {100 * CLIPPED_FRACTION_MAX:.1f}%), RMS {rms:.3f}. "
            "Ganho de captura alto demais — ver a nota de ganho abaixo.",
            gain_related=True,
        )

    # Fala tem a energia nos formantes, não embaixo deles. Ruído de ganho
    # excessivo, zumbido de alimentação e deriva de DC moram abaixo de 100 Hz,
    # onde a voz humana quase não tem nada — a fundamental mais grave de fala
    # adulta fica em ~85 Hz e carrega pouca energia.
    window = 2048
    if len(samples) >= window:
        freqs = np.fft.rfftfreq(window, 1 / rate)
        spectrum = np.zeros(window // 2 + 1)
        taper = np.hanning(window)
        blocks = 0
        for start in range(0, len(samples) - window, window):
            spectrum += np.abs(np.fft.rfft(samples[start : start + window] * taper))
            blocks += 1
        if blocks:
            total = spectrum.sum()
            low = spectrum[freqs < 100].sum()
            fraction = float(low / total) if total else 0.0
            if fraction > LOW_FREQ_FRACTION_MAX:
                return Problem(
                    f"{100 * fraction:.0f}% da energia abaixo de 100 Hz "
                    f"(teto {100 * LOW_FREQ_FRACTION_MAX:.0f}%). Fala não se parece com "
                    "isso; zumbido e deriva de DC se parecem.",
                    gain_related=True,
                )

    # "Isto é uma elocução?", depois de "isto é áudio?". Sem blocos de fala o
    # limiar não separou nada, e aí o PER mede o que o modelo faz com ruído —
    # que não é nada: este arquivo devolveu 65 fonemas de salada em sete
    # segundos e um PER de 3,462, número que iria para uma tabela.
    rms_blocks, _, loud = block_levels(samples, np, rate)
    if rms_blocks is not None:
        low = max(float(np.percentile(rms_blocks, 10)), 1e-12)
        contrast = float(np.percentile(rms_blocks, 90)) / low
        if len(loud) == 0 or contrast < SPEECH_CONTRAST_MIN:
            return Problem(
                f"sem contraste de fala: o bloco alto é só {contrast:.1f}× o baixo "
                f"(mínimo {SPEECH_CONTRAST_MIN:.0f}×), nível uniforme do começo ao fim. "
                "Fala tem pausa, sala não — isto é o microfone ouvindo o ambiente. "
                "Ganho não resolve, porque o piso sobe junto: fale perto do microfone, "
                "em sala silenciosa, com margem de silêncio em volta da frase.",
                gain_related=False,
            )
    return None


def warn_if_quiet(samples: Any, np: Any) -> str | None:
    """O nível da **fala** está baixo? Aviso, não recusa — ver ``RMS_LOW``.

    A estatística é a **mediana dos blocos de fala**, não o RMS do intervalo
    entre o primeiro e o último. A diferença não é cosmética: o intervalo
    carrega as pausas e um transiente de abertura, e na gravação que motivou
    isto o RMS do intervalo dava 0,123 — doze vezes o teto de aviso — para uma
    fala que estava em 0,07. Mediana de bloco é imune às duas coisas, porque
    nenhuma delas é a maioria dos blocos **de fala**.
    """
    if len(samples) == 0:
        return None
    rms_blocks, peak_blocks, loud = block_levels(samples, np)
    if rms_blocks is None or len(loud) == 0:
        # Sem blocos de fala não há nível de fala para avaliar, e medir o
        # arquivo inteiro no lugar foi como este aviso ficou calado sobre uma
        # gravação que era só sala: o arquivo uniforme passava o teto que a
        # fala não teria passado. `diagnose_signal` recusa antes de chegar
        # aqui; o aviso explícito é para o caso de alguém chamar isto sozinho.
        return (
            "não achei blocos de fala para medir o nível — ver a recusa por "
            "contraste em `diagnose_signal`."
        )
    rms = float(np.median(rms_blocks[loud]))
    peak = float(np.median(peak_blocks[loud]))
    floor = max(float(np.percentile(rms_blocks, 10)), 1e-12)
    snr_db = 20 * float(np.log10(max(rms, 1e-12) / floor))
    peak_dbfs = 20 * np.log10(max(peak, 1e-12))
    if rms >= RMS_LOW and peak_dbfs >= PEAK_LOW_DBFS:
        return None
    # Quanto falta, em dB, e não um palpite de porcentagem: a primeira versão
    # disto sugeria `Capture 70%` fixo, que nesta máquina são +4,5 dB sobre os
    # +11,25 dB de então — menos da metade do que o arquivo que a disparou
    # precisava. Conselho que não cobre o buraco é pior que nenhum, porque
    # custa uma gravação para descobrir.
    deficit = max(
        PEAK_LOW_DBFS - peak_dbfs,
        20 * float(np.log10(RMS_LOW / max(rms, 1e-12))),
    )
    return (
        f"nível baixo: bloco de fala típico com pico {peak_dbfs:.1f} dBFS e RMS {rms:.4f} "
        f"(desejável: pico acima de {PEAK_LOW_DBFS:.0f} dBFS, RMS acima de {RMS_LOW:.3f}). "
        f"Faltam ~{deficit:.0f} dB. O PER abaixo vale, mas vale menos.\n"
        f"      Falar mais perto do microfone é a alavanca melhor — ganho amplifica o\n"
        f"      ruído junto, e a relação sinal-ruído desta fala é "
        f"{snr_db:.0f} dB. Se preferir o mixer, nesta\n"
        f"      máquina cada ponto percentual de `Capture` vale ~{ALSA_DB_PER_PERCENT:.2f} dB, "
        f"logo ~+{deficit / ALSA_DB_PER_PERCENT:.0f} pontos."
    )


def warn_if_head_cut(span: tuple[float, float]) -> str | None:
    """A fala começa no primeiro instante do arquivo? Então pode faltar o início.

    Simétrico de ``warn_if_truncated``, e tão necessário quanto: um fonema
    inicial perdido vira uma **deleção** no alinhamento, indistinguível de o
    aluno não o ter pronunciado. Foi o que aconteceu na primeira elocução da
    E9 — `/ð ə/` ausente, fala começando em 0,0s — e dali não há como dizer se
    o aluno comeu "the" ou se o corte comeu.
    """
    if span[0] > 0.1:
        return None
    return (
        "a fala começa em 0,0s, encostada no início do arquivo: o primeiro fonema "
        "pode ter ficado fora, e no alinhamento isso aparece como deleção. "
        "Deixe um instante de silêncio antes de falar e grave outra vez."
    )


def warn_if_truncated(samples: Any, span: tuple[float, float]) -> str | None:
    """A fala vai até o fim do arquivo? Então provavelmente falta o fim dela.

    Uma elocução cortada no último fonema devolve um PER que parece medição e
    não é — a mesma classe de erro que ``diagnose_signal`` recusa, num grau
    que não dá para recusar: daqui de dentro não há como distinguir "a frase
    acabou junto com a janela" de "a janela acabou antes da frase". Quem
    gravou sabe, e é a quem o aviso se dirige.
    """
    duration = len(samples) / SAMPLE_RATE
    start, end = span
    if end < duration - 0.15:
        return None
    # A janela sugerida sai da duração deste arquivo, não de um número fixo: a
    # primeira versão sugeria `--record 7` sempre, e chegou a sugerir 7 para
    # quem já havia gravado 7.
    return (
        f"a fala vai até {end:.1f}s de {duration:.1f}s, encostada no fim do arquivo: "
        "o último fonema pode ter ficado fora. Grave com a janela mais longa "
        f"(`--record {math.ceil(duration) + 3}`) e comece a falar logo."
    )


GAIN_NOTE = """
  Nota de ganho, para esta classe de máquina. O `arecord` e o `pw-record` saem
  os dois do mesmo mixer ALSA, então trocar de ferramenta não resolve. Veja o
  que o dispositivo está somando:

      amixer -c 0 sget Capture
      amixer -c 0 sget 'Internal Mic Boost'

  Um microfone interno com +42 dB satura com a respiração. Para baixar:

      amixer -c 0 sset 'Internal Mic Boost' 0
      amixer -c 0 sset Capture 40%

  Depois grave outra vez e confira que o diagnóstico sai limpo.
"""


# As gravações e seus relatórios moram juntos, fora do repo. O corpus da E9 são
# dez elocuções mais dez relatórios, e espalhá-los pela raiz tornaria difícil
# conferir o conjunto — além de o relatório ser dado pessoal pela mesma razão
# que o WAV é: ele carrega a transcrição do que a pessoa disse. Daí a pasta
# inteira estar no `.gitignore`, como o `*.db` do progresso do aluno.
RECORDING_DIR = Path("gravacoes")
DEFAULT_RECORDING = RECORDING_DIR / "fala.wav"


def record(seconds: int, out: Path) -> Path:
    """Grava do microfone desta máquina com o ``arecord`` do alsa-utils.

    Mesmo caminho do ``measure_voice.py --record``, com uma diferença que é a
    razão de esta versão existir: **o arquivo fica no disco.** Lá a gravação é
    uma amostra descartável para medir latência; aqui ela é o corpus da E9, e
    um corpus precisa poder ser remedido quando o portão da §3.3 mudar de
    valor ou o modelo mudar de variante.

    Escreve WAV num caminho em vez de PCM cru no stdout porque o ``arecord``
    canalizado deixa os campos de tamanho do cabeçalho em aberto, e o
    ``wave`` da stdlib tropeça neles.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    # "Fale agora, sem esperar" was right when the trim took a fixed 1,0 s and
    # wrong since it became a decay search: it now stops at the first block not
    # dominated by DC — 40 ms on a clean start — so speaking instantly buys no
    # audio and trips the head-cut warning, which excludes the take from the
    # criterion. §3.5 of the plan records this message eating a `/ð ə/` once;
    # the second time it cost a take of `pron.epenthesis-final` measured at PER
    # 0,062 and thrown out for an edge warning. The prompt now asks for what
    # the diagnostic wants.
    print(f"\n  Gravando {seconds}s em {out} — deixe um instante de silêncio e fale.", flush=True)
    try:
        subprocess.run(
            [
                "arecord",
                "-q",
                "-t",
                "wav",
                "-f",
                "S16_LE",
                "-r",
                str(SAMPLE_RATE),
                "-c",
                "1",
                "-d",
                str(seconds + 1),
                str(out),
            ],
            capture_output=True,
            check=True,
            timeout=seconds + 15,
        )
    except FileNotFoundError:
        raise SystemExit(
            "  `arecord` não existe. Instale alsa-utils, ou grave por fora e use --audio."
        ) from None
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"  a gravação falhou: {exc.stderr.decode(errors='replace')[:200]}"
        ) from exc
    cut = _trim_lead_in(out)
    size_kb = out.stat().st_size / 1024
    nota = f", {cut * 1000:.0f} ms de transiente cortados" if cut else ""
    # Dizer onde o bruto ficou é parte da garantia: um corte errado só é
    # reversível por quem sabe que o original existe.
    bruto = f" (bruto em {out.with_suffix('.bruto.wav').name})" if cut else ""
    print(f"  pronto, {size_kb:.0f} KB{nota}{bruto}")
    return out


def _trim_lead_in(path: Path, max_seconds: float = LEAD_IN_MAX_SECONDS) -> float:
    """Joga fora só o transiente de abertura, e devolve quanto cortou.

    Varre blocos de 10 ms do começo, acha onde o degrau começa e segue
    enquanto o bloco for dominado por DC — ver ``DC_DOMINANCE``. A corrida é
    contígua com **uma** exceção: a travessia do zero do próprio degrau, uma
    falha de até ``GAP_MAX_BLOCKS`` blocos que é atravessada no máximo uma
    vez. Depois dela a contiguidade volta a valer, e é isso que impede um
    bloco enviesado no meio da fala de puxar o corte para dentro da elocução.

    Reescreve o arquivo no lugar: o WAV é o corpus da E9, e um corpus com o
    estouro do conversor dentro precisa ser limpo outra vez toda vez que for
    remedido.
    """
    import numpy as np

    with wave.open(str(path), "rb") as src:
        params = src.getparams()
        raw = src.readframes(src.getnframes())
    if params.sampwidth != 2 or params.nchannels != 1:
        return 0.0
    frame = params.sampwidth * params.nchannels
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    block = max(params.framerate // 100, 1)
    limit = min(int(max_seconds * params.framerate), len(samples))

    def dc_dominated(start: int) -> bool:
        piece = samples[start : start + block]
        rms = float(np.sqrt((piece**2).mean()))
        return rms > 0 and abs(float(piece.mean())) > DC_DOMINANCE * rms

    # Procura o **início** do degrau na primeira janela, em vez de exigir que
    # ele comece no primeiro bloco. Exigir isso foi um erro medido: numa
    # gravação real o bloco de 0 ms saiu limpo (média +0,0004) porque o
    # dispositivo ainda não havia rabeado, o degrau começou em 10 ms e cravou
    # em -1,0 até 110 ms — e a versão que parava no primeiro bloco não
    # dominado por DC parava no bloco 0 e não cortava nada.
    onset = None
    for start in range(0, min(limit, int(ONSET_WINDOW_SECONDS * params.framerate)), block):
        if dc_dominated(start):
            onset = start
            break
    if onset is None:
        return 0.0

    def run_end(start: int) -> int:
        """Onde termina a corrida contígua de blocos dominados a partir daqui."""
        end = start
        while end + block <= limit and dc_dominated(end):
            end += block
        return end

    # A corrida é contígua, com **uma** exceção atravessada **uma** vez: no
    # bloco em que o degrau atravessa o zero a média quase se anula contra um
    # RMS enorme (medido numa gravação de verdade: média +0,1004, RMS 0,7222,
    # razão 0,14), e esse bloco não passa pelo `DC_DOMINANCE`. A versão
    # puramente contígua parava nele — cortava um bloco só, anunciava "10 ms
    # de transiente cortados" e deixava 970 ms de degrau no arquivo, que o
    # `diagnose_signal` recusava em seguida. É o espelho do erro do bloco de
    # onset acima: lá o degrau não começava no primeiro bloco, aqui ele tem um
    # bloco limpo **dentro**.
    #
    # Uma vez, e **dentro da janela do onset**. Não "sempre que a falha for
    # curta", e isso é medido também: numa gravação de verdade o primeiro
    # segundo tem seis falhas de 1 ou 2 blocos entre blocos dominados **dentro
    # da fala** — um bloco de fala com viés passa pela peneira de DC de vez em
    # quando. Atravessar todas levava o corte de 490 ms a 940 ms, comendo o
    # começo da elocução, e no alinhamento isso reaparece como deleção de
    # fonema. A travessia do zero é parte do transiente de **abertura**, logo
    # cai na mesma janela em que o onset tem de cair; fora dela, o que parece
    # falha curta é fala, e vale a contiguidade.
    janela = int(ONSET_WINDOW_SECONDS * params.framerate)
    cut = run_end(onset)
    for gap in range(1, GAP_MAX_BLOCKS + 1):
        resume = cut + gap * block
        if resume >= janela:
            break
        if resume + block <= limit and dc_dominated(resume):
            cut = run_end(resume)
            break
    skip = cut * frame
    if skip >= len(raw):
        return 0.0

    # O bruto fica antes de a reescrita acontecer, e isso não é zelo: o corte
    # é uma heurística sobre um dado que custou alguém falar, e reescrever no
    # lugar torna um corte errado irreversível — reverter o código não devolve
    # a amostra. Já cobrou: quando a primeira versão desta função cortou 10 ms
    # de um degrau de 980 ms, o original tinha sido reescrito e não houve como
    # reproduzir a falha com o áudio de verdade; o caso de teste teve de ser
    # sintetizado a partir do perfil medido.
    #
    # É a mesma razão por que o WAV fica no disco em vez de ser amostra
    # descartável (ver `record`): o corpus precisa poder ser remedido quando o
    # portão da §3.3 mudar de valor ou o modelo mudar de variante. Aqui foi o
    # **cortador** que mudou, duas vezes num dia, e o argumento é igual.
    #
    # Não sobrescreve um bruto que já exista: numa segunda passada o original
    # de verdade é o da primeira, não o arquivo já cortado. E `*.wav` inteiro
    # está no `.gitignore`, então o irmão não vira ruído no `git status` nem
    # entra no repo — é dado pessoal como o resto da pasta. Nada no projeto
    # varre `*.wav` (o `_clean_report_for` olha `*.json`), logo ele é inerte
    # para a lógica do corpus.
    bruto = path.with_suffix(".bruto.wav")
    if not bruto.exists():
        with wave.open(str(bruto), "wb") as copia:
            copia.setparams(params)
            copia.writeframes(raw)

    with wave.open(str(path), "wb") as dst:
        dst.setparams(params)
        dst.writeframes(raw[skip:])
    return cut / params.framerate


def read_wav(path: Path, np) -> Any:
    """Um WAV mono 16-bit do disco, em float32.

    Sem `av` de propósito: quem grava para medir grava WAV, e um decodificador
    a mais aqui seria dependência para nada.
    """
    with wave.open(str(path), "rb") as src:
        if src.getnchannels() != 1 or src.getsampwidth() != 2:
            raise SystemExit(f"{path}: esperava WAV mono 16-bit")
        raw = src.readframes(src.getnframes())
        rate = src.getframerate()
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if rate != SAMPLE_RATE:
        duration = len(samples) / rate
        target_n = int(duration * SAMPLE_RATE)
        samples = np.interp(
            np.linspace(0, len(samples) - 1, target_n),
            np.arange(len(samples), dtype=np.float64),
            samples,
        ).astype(np.float32)
    return samples


# ---------------------------------------------------------------------------
# Os modos
# ---------------------------------------------------------------------------


def _fmt_phonemes(tokens: tuple[str, ...] | list[str], limit: int = 28) -> str:
    joined = " ".join(tokens)
    return joined if len(joined) <= limit * 3 else joined[: limit * 3] + "…"


def run_rtf(recognizer: PhonemeRecognizer, synth: Synth, reps: int) -> None:
    print(f"\n## Fator de tempo real — {recognizer.model}")
    print(f"carga do modelo: {recognizer.load_ms / 1000:.2f} s\n")
    print(f"{'áudio':>8}  {'mediana':>9}  {'RTF':>6}")
    for text in RTF_UTTERANCES:
        audio = synth.speak_text(text)
        seconds = len(audio) / SAMPLE_RATE
        times = []
        for _ in range(reps):
            _, elapsed_ms = recognizer.recognize(audio)
            times.append(elapsed_ms)
        median_ms = statistics.median(times)
        print(f"{seconds:7.1f}s  {median_ms / 1000:8.2f}s  {median_ms / 1000 / seconds:6.2f}")


def run_pairs(
    recognizer: PhonemeRecognizer,
    synth: Synth,
    g2p,
    discriminate: bool,
) -> None:
    """O piso e o sinal, lado a lado.

    Os dois juntos num modo só, porque separados não querem dizer nada: um PER
    de 0,18 é bom ou ruim dependendo exclusivamente do PER do lado errado.
    """
    rows = []
    print(f"\n## Piso e sinal — {recognizer.model}\n")
    for item in SUBSTITUTIONS:
        expected_ipa = phonemize(g2p, item.phrase)
        source, target = item.swap
        if source not in expected_ipa:
            print(f"  ! {item.rule_id}: '{source}' não aparece em /{expected_ipa}/ — pulado")
            continue
        wrong_ipa = expected_ipa.replace(source, target)
        expected = tokenize(expected_ipa, recognizer.vocab)
        # A assinatura do erro **no papel**: o que a troca faz à sequência de
        # fonemas, pelas duas transcrições, sem áudio no meio. É o gabarito
        # contra o qual se pergunta se o reconhecedor pegou o erro — e vem
        # das transcrições de propósito, não de eu nomear à mão o que esperar:
        # nomear à mão foi o que fez a primeira rodada errar a epêntese.
        signature = ops(align(expected, tokenize(wrong_ipa, recognizer.vocab)))

        heard_right, _ = recognizer.recognize(synth.speak_phonemes(expected_ipa))
        right = align(expected, heard_right)
        row = {"item": item, "right": right, "wrong": None, "signature": signature, "caught": None}
        if discriminate:
            heard_wrong, _ = recognizer.recognize(synth.speak_phonemes(wrong_ipa))
            wrong = align(expected, heard_wrong)
            row["wrong"] = wrong
            row["caught"] = bool(signature & ops(wrong))
        rows.append(row)

    header = f"{'regra':<26} {'PER correto':>12}"
    if discriminate:
        header += f" {'PER errado':>11} {'Δ':>7} {'pegou?':>7}"
    print(header)
    for row in rows:
        item, right, wrong = row["item"], row["right"], row["wrong"]
        line = f"{item.rule_id:<26} {right.per:11.2f} "
        if wrong is not None:
            line += f"{wrong.per:11.2f} {wrong.per - right.per:+7.2f} "
            line += f"{'sim' if row['caught'] else 'NÃO':>7}"
        print(line)

    floors = [row["right"].per for row in rows]
    if floors:
        print(f"\npiso (mediana do PER em fala correta): {statistics.median(floors):.3f}")
    if discriminate:
        signals = [row["wrong"].per for row in rows if row["wrong"] is not None]
        caught = [row for row in rows if row["caught"]]
        if signals:
            print(f"sinal (mediana do PER em fala trocada): {statistics.median(signals):.3f}")
            print(f"erros localizados pelo fonema certo: {len(caught)} de {len(rows)}")
    print("\nDetalhe por item — o que o reconhecedor ouviu em lugar do esperado:")
    for row in rows:
        item, right = row["item"], row["right"]
        if right.substitutions or right.insertions or right.deletions:
            print(f"  {item.rule_id:<26} correto: {_describe(right)}")
        if row["wrong"] is not None:
            print(f"  {item.rule_id:<26} errado:  {_describe(row['wrong'])}")
            gabarito = ", ".join(sorted(_render(op) for op in row["signature"]))
            print(f"  {'':<26} gabarito: {gabarito}")


def _render(op: tuple[str, ...]) -> str:
    """Uma operação de alinhamento em notação de fonema."""
    if op[0] == "sub":
        return f"/{op[1]}/→/{op[2]}/"
    if op[0] == "ins":
        return f"+/{op[1]}/"
    return f"-/{op[1]}/"


def _describe(alignment: Alignment, limit: int = 6) -> str:
    rendered = sorted(_render(op) for op in ops(alignment))
    shown = ", ".join(rendered[:limit])
    return shown + ("…" if len(rendered) > limit else "")


def run_accent(recognizer: PhonemeRecognizer, synth: Synth, g2p) -> None:
    """O proxy de sotaque contra as mesmas frases.

    Reporta duas coisas, e a segunda é a que importa: o PER (que vai ser
    alto, e alto não é notícia) e **quantas das substituições curadas
    aparecem nomeadas no alinhamento**. Um PER alto com os erros certos
    nomeados é um avaliador; um PER alto sem eles é um placar.
    """
    print(f"\n## Proxy de sotaque ({ACCENT_VOICE}, G2P {ACCENT_LANG}) — {recognizer.model}\n")
    print(f"{'regra':<26} {'PER sotaque':>12} {'gabarito nomeado?':>19}")
    pers, hits = [], 0
    detail = []
    for item in SUBSTITUTIONS:
        expected_ipa = phonemize(g2p, item.phrase)
        source, target = item.swap
        if source not in expected_ipa:
            continue
        expected = tokenize(expected_ipa, recognizer.vocab)
        signature = ops(
            align(expected, tokenize(expected_ipa.replace(source, target), recognizer.vocab))
        )
        heard, _ = recognizer.recognize(synth.speak_accented(item.phrase))
        result = align(expected, heard)
        found = bool(signature & ops(result))
        hits += found
        pers.append(result.per)
        print(f"{item.rule_id:<26} {result.per:11.2f} {('sim' if found else 'NÃO'):>19}")
        detail.append((item.rule_id, result))
    if pers:
        print(f"\nPER mediano do proxy: {statistics.median(pers):.3f}")
        print(f"gabaritos nomeados: {hits} de {len(pers)}")
    print("\nDetalhe — o que o reconhecedor ouviu:")
    for rule_id, result in detail:
        print(f"  {rule_id:<26} {_describe(result, limit=8)}")


def write_report(wav: Path, payload: dict) -> Path:
    """Grava ao lado do WAV o que foi medido nele, com o md5 do próprio áudio.

    Existe porque nesta máquina a saída do terminal e o disco divergiram
    repetidamente: saída antiga reaparecendo para comando idêntico, arquivo
    sem o corte que o script acabara de fazer, duração impressa diferente da
    do arquivo. Nenhuma dessas divergências é do script — ele é determinístico,
    e foi verificado rodando duas vezes sobre o mesmo arquivo.

    O relatório fecha a brecha do jeito que ela se fecha de verdade: o
    artefato no disco diz de qual áudio ele fala. Com o md5 dentro, não há como
    confundir a medição de uma gravação com a de outra, nem tomar saída
    repetida por medição nova. É também o que a E9 precisa — o critério pede
    PER e fração de gabaritos *registrados*, e um corpus que não se deixa
    remedir não é corpus.
    """
    # Sempre em `RECORDING_DIR`, mesmo quando o áudio está noutro lugar: o
    # relatório é dado pessoal — carrega a transcrição do que a pessoa disse —
    # e escrevê-lo ao lado de um `--audio` qualquer já o depositou na raiz do
    # repo, fora do `.gitignore`. Uma pasta ignorada só protege o que está
    # dentro dela.
    # O nome leva o md5 do áudio, não só o stem. Com o stem sozinho, dois
    # arquivos de mesmo nome em pastas diferentes — `e9-r-initial.wav` na raiz
    # e em `gravacoes/`, que foi o que aconteceu — colidem num relatório só, e
    # o segundo apaga o primeiro sem dizer nada. Num corpus de dez elocuções
    # isso destrói evidência caladamente, que é precisamente o que o relatório
    # existe para impedir. Remedir o mesmo áudio sobrescreve o relatório dele
    # mesmo, que é o certo.
    RECORDING_DIR.mkdir(parents=True, exist_ok=True)
    report = RECORDING_DIR / f"{wav.stem}-{payload['md5'][:8]}.json"
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return report


def _audio_fingerprint(wav: Path) -> dict:
    raw = wav.read_bytes()
    return {
        "arquivo": str(wav),
        "bytes": len(raw),
        "md5": hashlib.md5(raw).hexdigest(),
        "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(wav.stat().st_mtime)),
    }


def run_audio(
    recognizer: PhonemeRecognizer,
    g2p,
    path: Path,
    said: str,
    rule_id: str | None = None,
    wrong: bool = False,
) -> dict:
    """Mede um WAV e devolve **o que foi escrito no relatório**, não o payload
    inicial.

    A distinção não é estética: quem chama monta tabela com `per`, `ouvido`,
    `avisos` e `recusado`, e devolver o payload de antes da medição fazia a
    elocução medida com sucesso aparecer como recusada. A recusa também volta
    como dicionário (com `recusado`), para a tabela do `--corpus` mostrar a
    linha em vez de omiti-la — quem decide código de saída olha `recusado`, e
    não a verdade do valor.
    """
    import numpy as np

    audio = read_wav(path, np)
    seconds = len(audio) / SAMPLE_RATE
    payload = _audio_fingerprint(path) | {
        "medido_em": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duracao_s": round(seconds, 3),
        "dito": said,
        "modelo": recognizer.model,
        **({"regra": rule_id} if rule_id else {}),
        # Qual passada produziu esta elocução. Sem isto, as duas tomadas da
        # mesma frase — a correta e a com o erro de propósito — são
        # indistinguíveis no disco, e o `_clean_report_for` deixaria de pedir
        # uma porque a outra já está limpa.
        **({"errado": True} if wrong else {}),
    }
    problem = diagnose_signal(audio, np)
    if problem:
        refused = payload | {"recusado": problem.message}
        report = write_report(path, refused)
        print(f"\n## {path.name} ({seconds:.1f}s) — recusado")
        print(f"  {problem.message}")
        if problem.gain_related:
            print(GAIN_NOTE)
        print(
            "  Nenhum PER foi calculado de propósito: o reconhecedor devolveria um\n"
            "  número que parece medição e não é. Ver a §3.3 do plano da Fase 2."
        )
        print(f"  relatório: {report}  (md5 {payload['md5'][:12]})")
        return refused
    span = speech_span(audio, np)
    remarks = [
        remark
        for remark in (
            warn_if_quiet(audio, np),
            warn_if_head_cut(span),
            warn_if_truncated(audio, span),
        )
        if remark
    ]
    heard, elapsed_ms = recognizer.recognize(audio)
    expected_ipa = phonemize(g2p, said)
    expected = tokenize(expected_ipa, recognizer.vocab)
    result = align(expected, heard)
    print(f"\n## Áudio de verdade — {path.name} ({seconds:.1f}s)")
    for remark in remarks:
        print(f"  ! {remark}")
    print(f"fala em  : {span[0]:.1f}s a {span[1]:.1f}s")
    print(f"esperado : /{' '.join(expected)}/")
    print(f"ouvido   : /{' '.join(heard)}/")
    print(f"PER      : {result.per:.3f}  ({result.errors} erros em {len(expected)} fonemas)")
    print(f"tempo    : {elapsed_ms / 1000:.2f}s  (RTF {elapsed_ms / 1000 / seconds:.2f})")
    if result.substitutions:
        print("trocas   : " + ", ".join(f"/{a}/→/{b}/" for a, b in result.substitutions[:12]))
    if result.deletions:
        print("comidos  : " + ", ".join(f"/{p}/" for p in result.deletions[:12]))
    # A assinatura do erro curado **no papel**: o que a troca faz à sequência
    # de fonemas, pelas duas transcrições, sem áudio no meio. Vem para cá por
    # dois motivos. Primeiro, é aqui que existem o G2P e o vocabulário do
    # reconhecedor — o `_report_corpus` só vê relatórios. Segundo, ela fica
    # registrada no relatório, e por isso a resposta de "o avaliador pegou o
    # erro?" passa a ser auditável depois, em vez de ser recalculada de
    # memória.
    #
    # Derivada das transcrições e não nomeada à mão, que é a lição que o
    # `ops` já carrega: epêntese (*isport*, *bigi*) é **inserção**, e /h/
    # mudo é **deleção**. Procurar só uma substituição `/sp/→/ɪsp/` nunca
    # casa, e três das oito regras curadas ficavam estruturalmente impossíveis
    # de nomear — "não" garantido pelo critério, não medido.
    signature = _error_signature(recognizer, g2p, said, rule_id)
    measured = payload | {
        # The key is present whenever a rule was given, **even when the
        # signature is empty**, and that is the difference the backfill reads.
        # No key means nobody computed the gabarito; `"assinatura": []` means it
        # was computed and this phrase has none, because the rule's source
        # phoneme does not occur in it. Keyed off `signature` instead, the two
        # were the same report, and a backfill could not tell which reports it
        # still owed an answer.
        **({"assinatura": sorted(list(op) for op in signature)} if rule_id else {}),
        "avisos": remarks,
        "fala_de_s": round(span[0], 2),
        "fala_ate_s": round(span[1], 2),
        "esperado": expected,
        "ouvido": heard,
        "per": round(result.per, 4),
        "erros": result.errors,
        "trocas": [list(t) for t in result.substitutions],
        "comidos": list(result.deletions),
    }
    report = write_report(path, measured)
    print(f"relatório: {report}  (md5 {payload['md5'][:12]})")
    return measured


def _error_signature(
    recognizer: PhonemeRecognizer, g2p, said: str, rule_id: str | None
) -> set[tuple[str, ...]]:
    """As operações que a troca curada produz na sequência de fonemas.

    O mesmo cálculo do `run_pairs`: fonemiza a frase, aplica a troca sobre a
    transcrição e alinha as duas. Devolve conjunto vazio quando não há regra,
    ou quando o fonema de origem não aparece na transcrição desta frase —
    nesse caso não há gabarito, e o `_report_corpus` diz `—` em vez de "não".
    """
    if not rule_id:
        return set()
    item = next((i for i in SUBSTITUTIONS if i.rule_id == rule_id), None)
    if item is None:
        return set()
    expected_ipa = phonemize(g2p, said)
    source, target = item.swap
    if source not in expected_ipa:
        return set()
    expected = tokenize(expected_ipa, recognizer.vocab)
    wrong = tokenize(expected_ipa.replace(source, target), recognizer.vocab)
    return ops(align(expected, wrong))


def _slug(rule_id: str) -> str:
    return rule_id.split(".", 1)[-1]


def _same_phrase(a: str | None, b: str) -> bool:
    """A mesma frase, ignorando caixa, pontuação e espaço repetido."""
    if not a:
        return False
    keep = str.maketrans("", "", ".,;:!?\"'")
    return " ".join(a.translate(keep).lower().split()) == " ".join(
        b.translate(keep).lower().split()
    )


def _clean_reports_for(phrase: str, wrong: bool = False) -> list[Path]:
    """Every clean report on disk for this phrase, oldest name first.

    All of them, and not the first one, because the E9 criterion counts
    **utterances** and not rules: ten clean utterances over eight phrases means
    replicas, and a replica is the only thing that separates a measurement from
    a coincidence (§3.5). Reading one report per phrase reported five clean
    utterances with eight on disk, and the one it dropped for `pron.th-voiced`
    was the take that carried the curated `/ð/→/d/` — the single piece of real
    evidence the gate turns on.

    "Clean" is exactly the criterion: it passed the signal diagnostic and
    carries no edge warning. A report with a warning does **not** count,
    because that is how a `/ð ə/` eaten by the lead-in trim nearly entered the
    table as the student's error.
    """
    if not RECORDING_DIR.is_dir():
        return []
    found: list[Path] = []
    for report in sorted(RECORDING_DIR.glob("*.json")):
        try:
            data = json.loads(report.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        # Comparação frouxa de propósito: `dito` é texto livre que alguém
        # digitou no `--say`, e as gravações desta sessão saíram em minúsculas
        # contra a frase curada capitalizada. Casar exato fazia o `--corpus`
        # pedir para regravar o que já estava medido e limpo.
        if _same_phrase(data.get("dito"), phrase) is False:
            continue
        # A mesma frase tem duas tomadas com propósitos opostos; cada uma só
        # dispensa a si mesma.
        if bool(data.get("errado")) is not wrong:
            continue
        if data.get("recusado") or data.get("avisos"):
            continue
        found.append(report)
    return found


def _clean_report_for(phrase: str, wrong: bool = False) -> Path | None:
    """The first clean report, which is all "does this still need recording?"
    needs: one clean take dispenses the phrase from being recorded again."""
    reports = _clean_reports_for(phrase, wrong=wrong)
    return reports[0] if reports else None


def _with_signature(
    recognizer: PhonemeRecognizer, g2p, report: Path, item: Substitution, data: dict
) -> dict:
    """Writes the gabarito into a clean report that was measured without one.

    The gabarito is not a measurement: ``_error_signature`` phonemizes the
    curated phrase, applies the swap to the transcription and aligns the two,
    so it needs the G2P and the recognizer's vocabulary and **no audio at
    all**. Reports written before that calculation existed therefore answer
    ``—`` in the "erro previsto nomeado?" column — the one column the E9 gate
    reads — over audio that already holds the answer.

    Until now the only way out was ``--redo``, and ``--redo`` records over the
    take: it spends someone's voice to recover a number nobody had to speak
    for, and it destroys a clean utterance of a corpus that needs ten. Worse,
    it cannot reach the two takes whose WAV the next recording of the same rule
    already overwrote, and those two are evidence that still exists — the
    report keeps ``esperado`` and ``ouvido``, which is everything the
    alignment needs.

    So the answer is filled in where it belongs: in the report on disk, at its
    own path. Not through ``write_report``, which names the file after the
    audio's md5 and would leave a second copy of a report whose name predates
    that convention.
    """
    if "assinatura" in data:
        return data
    said = data.get("dito") or item.phrase
    signature = _error_signature(recognizer, g2p, said, item.rule_id)
    filled = data | {"assinatura": sorted(list(op) for op in signature)}
    report.write_text(json.dumps(filled, ensure_ascii=False, indent=2) + "\n")
    print(f"   gabarito preenchido no relatório: {len(signature)} operação(ões)")
    return filled


def run_corpus(
    recognizer: PhonemeRecognizer, g2p, seconds: int, redo: bool, wrong: bool = False
) -> int:
    """Grava e mede as oito frases curadas numa execução só.

    Uma execução e não oito por uma razão que esta máquina impôs: o terminal
    daqui reexibiu saída antiga para comando repetido várias vezes nesta
    sessão, e seis invocações são seis chances de a gravação não acontecer sem
    que a saída denuncie. Numa, o `--say` sai da lista curada em vez de ser
    digitado — descasamento entre o que foi lido e o que foi declarado já
    custou três gravações —, cada arquivo é nomeado pela regra, e a tabela
    final é a que o critério da E9 pede.

    Entre as frases espera o Enter: quem grava precisa escolher quando falar.
    Sem terminal interativo, segue sem pausa.
    """
    titulo = "com o erro de propósito" if wrong else "oito frases curadas"
    print(f"\n## Corpus da E9 — {titulo}")
    print(f"   {seconds}s por elocução, em {RECORDING_DIR}/")
    print("   Diga só a frase. O primeiro instante é descartado, então fale sem pressa.")
    if wrong:
        # Por que esta passada existe. Na passada correta, "erro previsto não
        # nomeado" soma duas coisas opostas: o avaliador não achou o erro, e o
        # erro não estava lá. Medido, foi 0 de 8 — e com PER entre 0,12 e 0,50
        # em seis das oito, nenhuma das duas leituras se sustenta sozinha. É a
        # cegueira da §6 caída justamente no que o portão da §3.3 decide.
        #
        # Cometer o erro de propósito desfaz o empate: se o avaliador nomeia a
        # substituição curada quando ela está de fato lá, a base é boa e o
        # 0 de 8 só dizia que o falante não comete esses erros. Se não nomeia
        # nem assim, o avaliador não faz o trabalho.
        print("   Esta é a passada ERRADA: cometa o erro curado de propósito.")
        print("   O que se mede aqui não é o PER, é se o erro certo foi nomeado.")

    results: list[dict] = []
    for item in SUBSTITUTIONS:
        # Every clean take of the phrase, and not just the one that dispenses
        # the recording: the criterion counts utterances, and the replicas are
        # what make the number a measurement instead of a coincidence.
        done = [] if redo else _clean_reports_for(item.phrase, wrong=wrong)
        if done:
            for report in done:
                print(f"\n[{item.rule_id}] já medida e limpa em {report.name} — pulando")
                data = json.loads(report.read_text()) | {"regra": item.rule_id}
                results.append(_with_signature(recognizer, g2p, report, item, data))
            continue
        print(f'\n[{item.rule_id}]  "{item.phrase}"')
        if wrong:
            # O como-dizer-errado já está na base curada desde o início: é o
            # par `wrong`/`right` de cada item, que até aqui só servia de
            # documentação.
            print(f'   DIGA ERRADO: "{item.wrong}" em vez de "{item.right}"')
        print(f"   hipótese curada: /{item.swap[0]}/ → /{item.swap[1]}/ — {item.why}")
        try:
            answer = input("   Enter para gravar, 'p' para pular: ").strip().lower()
        except EOFError:
            answer = ""
        if answer == "p":
            print("   pulada")
            continue
        # A gravação acontece **aqui**, e não no `main`: o `--corpus` é o único
        # modo que grava mais de uma vez, e a primeira versão deste laço só
        # montava o caminho e chamava o `run_audio`. O microfone nunca abria, e
        # o que o terminal mostrava era um `FileNotFoundError` sobre o WAV que
        # ninguém escreveu — exatamente a falha calada que o docstring acima diz
        # que a execução única existe para evitar.
        sufixo = "-errado" if wrong else ""
        path = record(seconds, RECORDING_DIR / f"e9-{_slug(item.rule_id)}{sufixo}.wav")
        # A tabela recebe o resultado completo — medições e recusa —, e não o
        # payload de antes da medição: `_report_corpus` lê `per`, `ouvido`,
        # `avisos` e `recusado` de cada linha. Com o payload inicial, as oito
        # frases recém-medidas saíam todas como "recusada", nenhum erro curado
        # era nomeado, e a elocução de fato recusada não aparecia na tabela.
        medida = run_audio(
            recognizer,
            g2p,
            path,
            item.phrase,
            rule_id=item.rule_id,
            wrong=wrong,
        )
        if medida:
            results.append(medida)

    return _report_corpus(results, wrong=wrong)


def _rule_of(data: dict) -> Substitution | None:
    """A regra curada deste relatório, pelo campo ou pela frase dita.

    Pela frase também, e não só pelo campo: os relatórios anteriores ao campo
    `regra` guardam o `dito`, e a frase é o que identifica a regra. Descartá-los
    jogaria fora elocução limpa que o critério da E9 conta — cinco delas, aqui.
    """
    by_id = next((i for i in SUBSTITUTIONS if i.rule_id == data.get("regra")), None)
    if by_id is not None:
        return by_id
    return next((i for i in SUBSTITUTIONS if _same_phrase(data.get("dito"), i.phrase)), None)


def run_rescore(vocabulary: Any, g2p) -> int:
    """Responde outra vez a coluna do gabarito sobre o que já está no disco.

    Nada aqui é medição nova, e é essa a razão de o modo existir. A §3.4 do
    plano diz que a assinatura de um item é um **conjunto** de operações
    aceitáveis; abrir o lado ouvido do gabarito à classe do fonema muda quais
    relatórios respondem "sim" e não muda um alinhamento, um PER nem um byte de
    áudio. A resposta já está em `esperado` e `ouvido`, que todo relatório
    medido guarda — foi o que o `_with_signature` estabeleceu quando a
    alternativa era o `--redo` gravar por cima de elocução limpa.

    O que ele escreve no disco é o gabarito faltante, pela mesma razão: ele não
    é medição. Inclusive nos relatórios **com aviso**, que o `--corpus` pula de
    propósito — o aviso exclui a elocução da contagem da E9, não a transcrição
    que ela já carrega.
    """
    print("\n## Re-score dos relatórios no disco — sem microfone, sem modelo")
    certas: list[dict] = []
    erradas: list[dict] = []
    sem_regra, preenchidos = 0, 0
    for report in sorted(RECORDING_DIR.glob("*.json")) if RECORDING_DIR.is_dir() else []:
        try:
            data = json.loads(report.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        item = _rule_of(data)
        if item is None:
            sem_regra += 1
            continue
        data = data | {"regra": item.rule_id}
        if "assinatura" not in data and not data.get("recusado"):
            signature = _error_signature(
                vocabulary, g2p, data.get("dito") or item.phrase, item.rule_id
            )
            data = data | {"assinatura": sorted(list(op) for op in signature)}
            report.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
            preenchidos += 1
        (erradas if data.get("errado") else certas).append(data)
    print(f"   {len(certas)} relatório(s) da passada correta, {len(erradas)} da errada")
    if preenchidos:
        print(f"   gabarito preenchido em {preenchidos} relatório(s) que não tinham")
    if sem_regra:
        print(f"   {sem_regra} relatório(s) sem regra curada reconhecível — fora das tabelas")
    if certas:
        _report_corpus(certas)
    if erradas:
        _report_corpus(erradas, wrong=True)
    if not certas and not erradas:
        print("   Nenhum relatório com regra curada no disco: não há o que re-scorar.")
    return 0


def _per_do_certo(phrase: str) -> float | None:
    """O PER da tomada **correta** e limpa desta frase, se houver uma."""
    report = _clean_report_for(phrase, wrong=False)
    if report is None:
        return None
    try:
        per = json.loads(report.read_text()).get("per")
    except (OSError, json.JSONDecodeError):
        return None
    return per if isinstance(per, (int, float)) else None


def _report_corpus(results: list[dict], wrong: bool = False) -> int:
    """A tabela final: PER por regra, e se o erro previsto foi nomeado.

    Na passada errada a coluna que decide não é o PER, é `nomeou?` — e ela vem
    ao lado do PER da tomada correta de propósito, porque é o par que resolve
    a ambiguidade do `previsto? não`.
    """
    by_rule = {item.rule_id: item for item in SUBSTITUTIONS}
    print("\n## Tabela do corpus" + (" — passada errada" if wrong else ""))
    # The md5 is a column and not decoration: with replicas in the table two
    # rows carry the same rule id, and the plan's tables identify a take by the
    # md5 of its audio.
    if wrong:
        print(f"   {'regra':<26} {'áudio':<9} {'certo':>6} {'errado':>7}  {'nomeou?':<9} avisos")
    else:
        print(f"   {'regra':<26} {'áudio':<9} {'PER':>6}  {'previsto?':<10} avisos")
    limpas, nomeados, nomeaveis = 0, 0, 0
    # O "sim" exato e o "sim" por classe são contados à parte, e saem à parte.
    # A §2 do plano foi medida com a interseção estrita; somar as duas numa
    # coluna só redefiniria aquelas tabelas em silêncio, que é exatamente o que
    # o comentário sobre a construção do gabarito pede para não fazer.
    exatos, por_classe_total = 0, 0
    # E a mesma fração sobre as limpas, que são as que o critério conta. Sem
    # ela o re-score anunciou "6 de 8" com uma das tomadas sendo as 44 fonemas
    # de salada do relatório `d2d3b42e`, onde um `ins /i/` aparece por acidente:
    # num alinhamento de PER 2,500 o gabarito casa por sorte, e o "sim" parece
    # medição. É a mesma distinção que a mediana já faz entre o que foi medido
    # e o que conta.
    nomeados_limpos, nomeaveis_limpos = 0, 0
    detalhe: list[tuple[str, list[str]]] = []
    for data in results:
        rule_id = data.get("regra", "?")
        item = by_rule.get(rule_id)
        heard, expected = data.get("ouvido"), data.get("esperado")
        assinatura = data.get("assinatura")
        named = "—"
        medida_nomeavel = False
        avisos = len(data.get("avisos", []))
        limpa = not data.get("recusado") and not avisos
        if item and heard and expected and assinatura:
            # Interseção, como no `run_pairs`: basta o reconhecedor produzir
            # **uma** das operações do gabarito. Relatório sem assinatura fica
            # em `—` de propósito: é anterior a este cálculo, e chamá-lo de
            # "não" seria inventar uma medida que ninguém fez.
            gabarito = {tuple(op) for op in assinatura}
            produzidas = ops(align(expected, heard))
            casadas = gabarito & produzidas
            # E depois das exatas, nunca em vez delas: a classe só é consultada
            # onde o símbolo não casou, e a coluna diz qual dos dois aconteceu.
            por_classe = (widen(gabarito) & produzidas) - casadas
            found = bool(casadas or por_classe)
            named = "sim" if casadas else ("classe" if por_classe else "não")
            nomeados += bool(found)
            exatos += bool(casadas)
            por_classe_total += bool(por_classe and not casadas)
            medida_nomeavel = True
            nomeaveis_limpos += limpa
            nomeados_limpos += bool(found) and limpa
            if found:
                # **Qual** operação casou, e não só que casou. O gabarito vem
                # de trocar o fonema em toda a transcrição, então numa frase
                # com várias ocorrências ele cobre mais do que a regra
                # descreve: em "The red car is ready" há três /ɹ/, e o
                # gabarito da `r-initial` inclui `sub /ɑːɹ/→/h/`, que é o /ɹ/
                # de *car* — final de sílaba, o oposto de inicial. Casar com
                # essa daria um "sim" pela razão errada. A construção não muda
                # aqui de propósito: é a mesma do `run_pairs`, e mexer nela
                # redefiniria em silêncio os números da §2 do plano. O que
                # muda é que a operação casada fica à vista, para quem lê a
                # tabela poder recusá-la.
                detalhe.append(
                    (
                        rule_id,
                        sorted("/".join(op) for op in casadas)
                        + sorted(f"{'/'.join(op)} (classe)" for op in por_classe),
                    )
                )
        if limpa:
            limpas += 1
        nomeaveis += medida_nomeavel
        per = data.get("per")
        per_txt = f"{per:.3f}" if isinstance(per, (int, float)) else "recusada"
        md5 = str(data.get("md5", ""))[:8] or "—"
        if wrong:
            certo = _per_do_certo(item.phrase) if item else None
            certo_txt = f"{certo:.3f}" if certo is not None else "—"
            print(
                f"   {rule_id:<26} {md5:<9} {certo_txt:>6} {per_txt:>7}  {named:<9} {avisos or ''}"
            )
        else:
            print(f"   {rule_id:<26} {md5:<9} {per_txt:>6}  {named:<10} {avisos or ''}")

    pers = [d["per"] for d in results if isinstance(d.get("per"), (int, float))]
    # The median over the **clean** takes, because that is the set the criterion
    # counts and the set the plan's tables are built from. Over every measured
    # row it mixed in the ones a warning excludes: one run printed 0,094 across
    # ten rows while its nine clean ones sat at 0,125, so the headline number
    # was one no table in the plan could be built from — the same confusion
    # between "what was measured" and "what counts" that `limpas` exists to
    # keep apart.
    pers_limpos = [
        d["per"]
        for d in results
        if isinstance(d.get("per"), (int, float)) and not d.get("recusado") and not d.get("avisos")
    ]
    print(f"\n   elocuções limpas: {limpas} (o critério da E9 pede 10 no total)")
    if pers_limpos:
        print(f"   PER mediano das limpas: {statistics.median(pers_limpos):.3f}")
    elif pers:
        # Measured but none of it clean. Printing the median anyway would put a
        # number the criterion rejects where the headline goes.
        print("   Nenhuma elocução limpa: não há mediana que o critério aceite.")
    resumo = f"   erros curados nomeados: {nomeados} de {nomeaveis} com gabarito"
    if por_classe_total:
        # O split fica na linha de manchete porque é ele que diz o que o
        # número é: um "sim" por classe é a §3.4 sendo aplicada, não a §2.3
        # sendo melhorada.
        resumo += f" — {exatos} exato(s), {por_classe_total} por classe de fonema (§3.4)"
    print(resumo)
    if nomeaveis_limpos != nomeaveis:
        print(
            f"   nas limpas, que são as que o critério conta: "
            f"{nomeados_limpos} de {nomeaveis_limpos}"
        )
    if detalhe:
        print("\n   Qual operação do gabarito casou — confira se é a que a regra descreve:")
        for rule_id, casadas in detalhe:
            print(f"     {rule_id:<26} {', '.join(casadas)}")
    if nomeaveis < len(pers):
        # This used to send the reader to `--redo`, which records over the take.
        # The `--corpus` now fills the gabarito into a clean report by itself, so
        # what is left here is the report this run did not open: measured outside
        # the corpus, by `--audio` without `--rule`. Measuring the same WAV again
        # with the rule answers it, and costs nobody a word.
        print(
            f"   ({len(pers) - nomeaveis} elocução(ões) sem gabarito no relatório — "
            'remeça o WAV com `--audio <wav> --rule <rule_id> --say "<frase>"`)'
        )
    if wrong:
        # A leitura é o produto desta passada, e não o número: dito de outro
        # modo, é aqui que o portão da §3.3 abre ou fecha.
        if not nomeaveis:
            # Zero medidas com gabarito não são zero acertos. Sem esta guarda,
            # uma passada em que tudo foi pulado — ou cujos relatórios são
            # anteriores ao gabarito — anunciava que o avaliador não nomeou
            # erro nenhum, uma conclusão sobre o portão tirada de nada.
            print("   Nenhuma elocução com gabarito nesta passada: não há o que concluir.")
        elif nomeados == nomeaveis:
            print(
                "   O avaliador nomeou o erro curado em todas as elocuções em que ele\n"
                "   estava de fato lá. A base serve, e o 0 de 8 da passada correta dizia\n"
                "   apenas que o falante não comete esses erros."
            )
        elif nomeados:
            print(
                f"   O avaliador nomeou o erro em {nomeados} de {nomeaveis}: ele enxerga\n"
                "   parte das regras curadas e não todas. Olhe regra por regra — o portão\n"
                "   pode abrir para um subconjunto em vez de abrir ou fechar inteiro."
            )
        else:
            print(
                "   O avaliador não nomeou o erro curado em nenhuma elocução em que ele\n"
                "   estava de fato lá. Isso não é o falante: é o avaliador não fazendo o\n"
                "   trabalho, e é o que a §3.3 pede para decidir antes de haver produto."
            )
    if limpas < len(SUBSTITUTIONS):
        print("   Regrave as que faltam ou saíram com aviso — `--corpus` pula as limpas.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtf", action="store_true", help="carga e fator de tempo real")
    parser.add_argument("--floor", action="store_true", help="PER em fala correta")
    parser.add_argument("--discriminate", action="store_true", help="piso contra sinal")
    parser.add_argument("--accent", action="store_true", help="proxy de sotaque brasileiro")
    parser.add_argument("--audio", type=Path, help="WAV mono 16-bit de verdade")
    parser.add_argument(
        "--record",
        type=int,
        default=0,
        help=f"grava N segundos em --out (default {DEFAULT_RECORDING})",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_RECORDING, help="onde a gravação fica")
    parser.add_argument(
        "--corpus",
        type=int,
        nargs="?",
        const=6,
        # `None` e não `0`: com `0` como default, `--corpus 0` é indistinguível
        # de "não pediu modo nenhum", e o atalho que liga `--rtf`/`--discriminate`
        # engolia o pedido em vez de recusá-lo.
        default=None,
        metavar="SEGUNDOS",
        help="grava e mede as oito frases curadas numa execução (default 6s cada)",
    )
    parser.add_argument("--redo", action="store_true", help="com --corpus, regrava as já limpas")
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="responde a coluna do gabarito outra vez sobre os relatórios no disco",
    )
    parser.add_argument(
        "--errado",
        action="store_true",
        help="com --corpus, grava a passada em que o erro curado é cometido de propósito",
    )
    parser.add_argument("--say", help="o que foi dito no --audio ou no --record")
    parser.add_argument(
        "--rule",
        # Without this, a WAV measured outside the `--corpus` could never carry a
        # gabarito: `run_audio` takes the rule, and `main` was the one caller that
        # never passed it. The column that decides the E9 gate was reachable only
        # by recording again.
        help="rule_id curado do --audio/--record, para o relatório levar o gabarito",
    )
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    if not any(
        [
            args.rtf,
            args.floor,
            args.discriminate,
            args.accent,
            args.audio,
            args.record,
            args.corpus is not None,
            args.rescore,
        ]
    ):
        args.rtf = args.discriminate = True

    # Tudo o que pode ser recusado é recusado **antes** de carregar o modelo.
    # São 318 MB e ~1,8 s de carga, e a primeira vez que alguém rodou o modo
    # `--audio` com o nome de arquivo de exemplo do README pagou os dois para
    # receber um traceback de arquivo inexistente.
    if args.corpus is not None and (args.audio or args.record):
        raise SystemExit(
            "--corpus grava as frases curadas sozinho; não combine com --audio/--record"
        )
    if args.rescore and (args.corpus is not None or args.audio or args.record):
        raise SystemExit("--rescore lê os relatórios que já existem; não grava nem mede áudio")
    if args.corpus is not None and args.corpus < 1:
        raise SystemExit("--corpus precisa de pelo menos 1 segundo por elocução")
    if args.errado and args.corpus is None:
        raise SystemExit("--errado é uma passada do --corpus; use `--corpus --errado`")
    if args.rule and not (args.audio or args.record):
        raise SystemExit("--rule é do --audio/--record; o --corpus tira a regra da lista curada")
    if args.rule and args.rule not in {item.rule_id for item in SUBSTITUTIONS}:
        raise SystemExit(
            f"{args.rule}: não é regra curada. São: "
            + ", ".join(item.rule_id for item in SUBSTITUTIONS)
        )
    if (args.audio or args.record) and not args.say:
        raise SystemExit("--audio/--record precisam de --say para ter o que comparar")
    if args.audio and not args.audio.is_file():
        raise SystemExit(
            f'{args.audio}: não existe. Grave com `--record 5 --say "..."`, '
            "ou aponte --audio para um WAV mono 16-bit."
        )
    if args.record and args.record < 1:
        raise SystemExit("--record precisa de pelo menos 1 segundo")

    if args.rescore:
        # Antes de tudo o que custa: este modo não abre áudio nenhum, então não
        # carrega o reconhecedor (318 MB), não precisa do Kokoro e não lê
        # `Settings`. É a mesma regra das recusas acima, um passo adiante.
        return run_rescore(Vocabulary(), _build_g2p())

    from teacher_charles.config import Settings

    settings = Settings(_env_file=None)

    # A gravação também vem antes da carga: ninguém deve esperar o modelo para
    # o microfone abrir.
    audio_path = args.audio
    if args.record:
        audio_path = record(args.record, args.out)

    recognizer = PhonemeRecognizer(threads=args.threads)
    g2p = _build_g2p()

    if args.corpus is not None:
        return run_corpus(recognizer, g2p, args.corpus, args.redo, wrong=args.errado)

    if audio_path:
        # Pela recusa e não pela verdade do dicionário: o `run_audio` passou a
        # devolver a recusa como dado para a tabela do `--corpus`, e um `dict`
        # não vazio é verdadeiro. Sem isto, sinal recusado sairia com 0.
        medida = run_audio(recognizer, g2p, audio_path, args.say, rule_id=args.rule)
        return 2 if not medida or medida.get("recusado") else 0

    synth = Synth(settings)
    if args.rtf:
        run_rtf(recognizer, synth, args.reps)
    if args.floor or args.discriminate:
        run_pairs(recognizer, synth, g2p, discriminate=args.discriminate)
    if args.accent:
        run_accent(recognizer, synth, g2p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
