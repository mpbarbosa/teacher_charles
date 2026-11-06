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
import json
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
        vocab_path = hf_hub_download(HF_REPO, "vocab.json")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self._session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.load_ms = (time.perf_counter() - t0) * 1000

        raw: dict[str, int] = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
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


DEFAULT_RECORDING = Path("fala.wav")


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
    print(f"\n  Gravando {seconds}s em {out} — fale agora.", flush=True)
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
                str(seconds),
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
    print(f"  pronto, {out.stat().st_size / 1024:.0f} KB")
    return out


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


def run_audio(recognizer: PhonemeRecognizer, g2p, path: Path, said: str) -> None:
    import numpy as np

    audio = read_wav(path, np)
    seconds = len(audio) / SAMPLE_RATE
    heard, elapsed_ms = recognizer.recognize(audio)
    expected_ipa = phonemize(g2p, said)
    expected = tokenize(expected_ipa, recognizer.vocab)
    result = align(expected, heard)
    print(f"\n## Áudio de verdade — {path.name} ({seconds:.1f}s)")
    print(f"esperado : /{' '.join(expected)}/")
    print(f"ouvido   : /{' '.join(heard)}/")
    print(f"PER      : {result.per:.3f}  ({result.errors} erros em {len(expected)} fonemas)")
    print(f"tempo    : {elapsed_ms / 1000:.2f}s  (RTF {elapsed_ms / 1000 / seconds:.2f})")
    if result.substitutions:
        print("trocas   : " + ", ".join(f"/{a}/→/{b}/" for a, b in result.substitutions[:12]))
    if result.deletions:
        print("comidos  : " + ", ".join(f"/{p}/" for p in result.deletions[:12]))


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
    parser.add_argument("--say", help="o que foi dito no --audio ou no --record")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    if not any([args.rtf, args.floor, args.discriminate, args.accent, args.audio, args.record]):
        args.rtf = args.discriminate = True

    # Tudo o que pode ser recusado é recusado **antes** de carregar o modelo.
    # São 318 MB e ~1,8 s de carga, e a primeira vez que alguém rodou o modo
    # `--audio` com o nome de arquivo de exemplo do README pagou os dois para
    # receber um traceback de arquivo inexistente.
    if (args.audio or args.record) and not args.say:
        raise SystemExit("--audio/--record precisam de --say para ter o que comparar")
    if args.audio and not args.audio.is_file():
        raise SystemExit(
            f'{args.audio}: não existe. Grave com `--record 5 --say "..."`, '
            "ou aponte --audio para um WAV mono 16-bit."
        )
    if args.record and args.record < 1:
        raise SystemExit("--record precisa de pelo menos 1 segundo")

    from teacher_charles.config import Settings

    settings = Settings(_env_file=None)

    # A gravação também vem antes da carga: ninguém deve esperar o modelo para
    # o microfone abrir.
    audio_path = args.audio
    if args.record:
        audio_path = record(args.record, args.out)

    recognizer = PhonemeRecognizer(threads=args.threads)
    g2p = _build_g2p()

    if audio_path:
        run_audio(recognizer, g2p, audio_path, args.say)
        return 0

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
