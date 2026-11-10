#!/usr/bin/env python3
"""Mede o que o produto faz hoje quando o aluno pergunta por uma palavra.

Uma "aula de palavra" não tem número de entrega nem seção de plano — a única
menção é a §4.3 da `analise-inicial.md`, *repetição espaçada para
vocabulário*, o único item daquela lista que nunca foi entregue. Antes de
escrever a entrega, mede-se o que já existe, como a 1.5 e a 2 abriram.

A pergunta não é "o professor sabe a palavra". É **quem responde**: o app tem
base curada, agenda e um G2P determinístico, e nada disso é consultado por uma
pergunta de vocabulário. O que sobra é o modelo sozinho, que é exatamente o
arranjo que a E4 existe para recusar na gramática.

    uv run python scripts/measure_vocab.py --no-model
        A metade determinística, em milissegundos e sem Ollama: quantos itens
        curados uma pergunta sobre palavra levanta, e quantas vezes cada
        palavra aparece no `corpus/` desta máquina.

    uv run python scripts/measure_vocab.py
        O caminho real: uma pergunta por palavra, em conversa livre, sessão
        nova a cada palavra para o histórico de uma não contaminar a seguinte.
        Imprime a resposta e o que dá para contar nela. Precisa de Ollama de pé.

## O que se conta, e por que cada contagem é grosseira

**IPA.** O único sinal com gabarito de verdade. A linha de IPA da tela sai do
espeak e vale para as duas bolhas (`Phonemized.role`), então quando o modelo
digita uma transcrição na prosa o aluno vê **duas** na mesma bolha: a inventada
e a real, logo abaixo. Comparar as duas é medir a tela contra o alto-falante,
que é a falha que a regra do G2P único existe para evitar um andar abaixo. A
comparação normaliza acento tônico, comprimento e delimitador — é frouxa de
propósito, para não acusar divergência onde só há notação.

**Português.** A decisão 1.1 do plano é 100% em inglês, e glosar palavra é a
tentação mais forte que existe contra ela. Detectado por marcador e acento:
grosseiro, e um falso negativo é certo num texto sem acento.

**Alegação de ouvir.** O que o CLAUDE.md proíbe nos dois estados de voz, e a
E9 é portão não vencido. Uma pergunta sobre palavra convida: *repeat after me*.

**Rótulo.** *formal*, *literary*, *C1*, *rare*: fato sobre a palavra, sem
nenhuma fonte curada no app para conferir. A saída imprime os rótulos achados
em vez de só contá-los, porque alguns estarão certos — e estar certo sem fonte
é o que o verificador da E4 chama de gramática inventada.

**Exemplos.** Trechos entre aspas. Um item de vocabulário teria que tirá-los do
`corpus/`, derivados e não copiados, e é por isso que a contagem do corpus
anda junto: ela é que diz se a palavra merece aula nesta máquina.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from teacher_charles.config import get_settings  # noqa: E402
from teacher_charles.llm import build_provider, default_options  # noqa: E402
from teacher_charles.llm.base import Message  # noqa: E402
from teacher_charles.prompts import PromptLibrary  # noqa: E402
from teacher_charles.store.db import init_db  # noqa: E402
from teacher_charles.store.models import Student  # noqa: E402
from teacher_charles.store.repo import SessionRepo, StudentRepo  # noqa: E402
from teacher_charles.subtitles import read_srt, utterances  # noqa: E402
from teacher_charles.teaching import (  # noqa: E402
    Analyzer,
    Chunk,
    Failed,
    Phonemized,
    TurnRunner,
)
from teacher_charles.teaching.contrastive import load_base  # noqa: E402
from teacher_charles.teaching.modes import DrillPlanner  # noqa: E402
from teacher_charles.teaching.phonetics import Phonetics, build_phonetics  # noqa: E402

# As dez palavras, e a armadilha que cada uma sonda. Metade tem item curado na
# base e metade não — é esse corte que mostra que a base não é consultada por
# uma pergunta, e não a ausência de curadoria.
WORDS: tuple[tuple[str, str], ...] = (
    ("fodder", "sentido útil é figurado; o do dicionário é agrícola"),
    ("actually", "falso cognato curado (false-friend.actually-atualmente)"),
    ("pretend", "falso cognato curado (false-friend.pretend-pretender)"),
    ("eventually", "falso cognato curado (false-friend.eventually-eventualmente)"),
    ("comprehensive", "falso cognato curado (false-friend.comprehensive-compreensivo)"),
    ("realize", "falso cognato curado (false-friend.realize-realizar)"),
    ("thorough", "armadilha de pronúncia: th mais schwa"),
    ("schedule", "pronúncia UK/US divide; a linha da tela é US"),
    ("cannon fodder", "colocação de duas palavras, não palavra"),
    ("house", "controle: palavra sem nada a ensinar"),
)

QUESTION = 'What does "{word}" mean?'

# Caracteres que só aparecem em transcrição, exigidos para um trecho entre
# barras ou colchetes contar como IPA. Sem isso "and/or" viraria transcrição.
_IPA_CHARS = "ˈˌːəɑɒɔɪʊʌæθðʃʒŋɚɹɜɝɡˑ"
_IPA_SPAN = re.compile(rf"[/\[]([^/\]\n]{{2,40}}[{_IPA_CHARS}][^/\]\n]{{0,40}})[/\]]")
# Notação que não é conteúdo: tônica, comprimento, delimitador, espaço.
_IPA_NOISE = re.compile(r"[ˈˌːˑ\s/\[\]().]")

_PT = re.compile(
    r"[ãõçáéíóúâêôà]|\b(significa|quer dizer|em portugu[eê]s|forragem|ra[çc][ãa]o)\b",
    re.IGNORECASE,
)
_HEARS = re.compile(
    r"\b(repeat after me|say it (out )?(loud|aloud)|read it (out )?loud"
    r"|let me hear|i('ll| will| can| am going to)? (listen|hear)"
    r"|your pronunciation|how you (say|said|pronounce))\b",
    re.IGNORECASE,
)
_LABEL = re.compile(
    # Radical e não palavra inteira: a primeira rodada desta medição perdeu
    # "used figuratively" porque o padrão pedia `\bfigurative\b`, e contou
    # 0/10 rótulos num texto que trazia um. O rótulo é o sinal mais fraco da
    # tabela; errar a contagem dele para baixo é errar a favor da conclusão.
    r"\b(formal|informal|slang|colloquial|literar\w*|old-fashioned|archaic|poetic"
    r"|rare|uncommon|advanced|idiomatic|figurativ\w*|literal\w*|technical"
    r"|british|american|[ABC][12])\b",
    re.IGNORECASE,
)

_QUOTED = re.compile(r'"[^"\n]{3,}"')


def normalize_ipa(text: str) -> str:
    return _IPA_NOISE.sub("", text).lower()


def typed_ipa(reply: str) -> str | None:
    """A primeira transcrição que o modelo digitou na prosa, se digitou."""
    found = _IPA_SPAN.search(reply)
    return found.group(1).strip() if found else None


def corpus_hits(word: str, root: Path | None = None) -> tuple[int, int, str | None]:
    """Ocorrências na fala do `corpus/`, com uma frase de verdade se houver.

    Conta sobre `utterances`, não sobre as cues: é a frase que seria o exemplo
    de uma aula, e o arquivo quebra frase onde a tela acaba, não onde a frase
    acaba. Uma frase partida em duas cues conta **uma** vez.

    `root` é parâmetro para o teste poder apontar para um `.srt` de mentira;
    o default é o corpus desta máquina, que é o que a medição usa.
    """
    root = root if root is not None else REPO_ROOT / "corpus" / "subtitles"
    if not root.is_dir():
        return 0, 0, None
    pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    hits = 0
    total = 0
    example: str | None = None
    for path in sorted(root.rglob("*.srt")):
        said = utterances(read_srt(path))
        total += len(said)
        for one in said:
            if pattern.search(one.text):
                hits += 1
                if example is None:
                    example = one.text
    return hits, total, example


# --- metade 1: o que não precisa de modelo --------------------------------- #


def deterministic() -> None:
    settings = get_settings()
    base = load_base(settings.data_dir / "contrastive")

    print(f"base curada: {len(base)} itens")
    print("corpus:", REPO_ROOT / "corpus" / "subtitles")
    print()
    print(f"{'palavra':16} {'itens levantados':17} {'no corpus':>10}  exemplo autêntico")
    print("-" * 92)
    for word, _ in WORDS:
        rules = base.rule_ids(QUESTION.format(word=word))
        hits, total, example = corpus_hits(word)
        shown = ", ".join(rules) if rules else "—"
        snippet = (example[:44] + "…") if example and len(example) > 45 else (example or "—")
        print(f"{word:16} {shown:17} {hits:>4}/{total:<5}  {snippet}")
    print()
    print(
        "A coluna do meio é o achado: a base casa `patterns` contra o texto do\n"
        "aluno, e uma pergunta não traz erro nenhum para casar. Com\n"
        "`require_curated_rule` ligado a análise nem roda, então a resposta sai\n"
        "sem nada curado dentro — inclusive nas palavras que têm item próprio."
    )


# --- metade 2: o caminho real ---------------------------------------------- #


def describe(reply: str, ipa_line: str, word: str, real_ipa: str) -> dict[str, object]:
    guessed = typed_ipa(reply)
    agrees: bool | None = None
    if guessed is not None:
        agrees = normalize_ipa(guessed) == normalize_ipa(real_ipa)
    return {
        "words": len(reply.split()),
        "ipa_typed": guessed,
        "ipa_real": real_ipa,
        "ipa_agrees": agrees,
        "pt": bool(_PT.search(reply)),
        "hears": bool(_HEARS.search(reply)),
        "labels": sorted({m.group(0).lower() for m in _LABEL.finditer(reply)}),
        "examples": len(_QUOTED.findall(reply)),
        "line": ipa_line,
    }


async def against_model(runs: int) -> int:
    settings = get_settings()
    prompts = PromptLibrary(settings.prompts_dir)
    base = load_base(settings.data_dir / "contrastive")
    provider = build_provider(settings)
    phonetics = build_phonetics(settings)
    if phonetics is None:
        print("sem G2P: a comparação de IPA é o gabarito desta medição", file=sys.stderr)
        return 1

    rows: list[tuple[str, dict[str, object]]] = []
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "vocab.db")
        student = StudentRepo(conn).get_or_create("medicao")
        sessions = SessionRepo(conn)
        runner = TurnRunner(
            provider=provider,
            prompts=prompts,
            conn=conn,
            options=default_options(settings),
            planner=DrillPlanner(conn, base),
            analyzer=Analyzer(provider=provider, prompts=prompts, base=base),
            phonetics=phonetics,
        )

        print(f"modelo: {provider.model}")
        print(f"pergunta: {QUESTION.format(word='<palavra>')!r}")
        print("modo: conversation, sessão nova por palavra\n")

        # Aquece com o prefixo do turno, pela mesma razão e pelo mesmo caminho
        # que o `app.py`: `stable_prefix` é a lista única. Sem isso a primeira
        # palavra da lista paga a carga do modelo dentro do turno e estoura o
        # `request_timeout_s` — foi o que aconteceu na primeira rodada desta
        # medição, e o que ela mediu foi o timeout, não a palavra.
        warm = [*runner.stable_prefix("B1", "conversation"), Message(role="user", content="Hello.")]
        warm_started = time.monotonic()
        try:
            await provider.complete(warm, options=default_options(settings))
            print(f"aquecimento: {time.monotonic() - warm_started:.1f}s\n")
        except Exception as error:  # noqa: BLE001 — mesma política do app
            print(f"aquecimento falhou ({error}); segue mesmo assim\n")

        try:
            for sweep in range(runs):
                if runs > 1:
                    print(f"########## varredura {sweep + 1} de {runs} ##########\n")
                for word, probes in WORDS:
                    row = await one_word(word, probes, runner, sessions, student, phonetics)
                    if row is not None:
                        rows.append(row)
        finally:
            await provider.aclose()
            conn.close()

    summary(rows)
    return 0


async def one_word(
    word: str,
    probes: str,
    runner: TurnRunner,
    sessions: SessionRepo,
    student: Student,
    phonetics: Phonetics,
) -> tuple[str, dict[str, object]] | None:
    """Uma pergunta, uma resposta, os sinais dela — ou ``None`` se o turno caiu.

    Sessão nova a cada chamada: o histórico da palavra anterior mudaria a
    resposta, e o que se mede aqui é a palavra, não a conversa.
    """
    session_id = sessions.switch_mode(student.id, "conversation", None)
    started = time.monotonic()
    pieces: list[str] = []
    teacher_line = ""
    failure: str | None = None

    async for event in runner.run(session_id, QUESTION.format(word=word), student):
        if isinstance(event, Chunk):
            pieces.append(event.text)
        elif isinstance(event, Phonemized) and event.role == "teacher":
            teacher_line = event.ipa
        elif isinstance(event, Failed):
            # Engolir isto foi o erro da primeira rodada: a palavra saiu com
            # 0 palavras e a tabela não disse por quê.
            failure = event.message

    elapsed = time.monotonic() - started
    print(f"--- {word} · {probes} ---")
    if failure is not None:
        print(f"  FALHOU em {elapsed:.1f}s: {failure}\n")
        return None

    reply = "".join(pieces).strip()
    signals = describe(reply, teacher_line, word, phonetics.transcribe(word))
    print(f"  {signals['words']} palavras · {elapsed:.1f}s")
    print("  " + reply.replace("\n", "\n  "))
    print()
    return word, signals


def summary(rows: list[tuple[str, dict[str, object]]]) -> None:
    print("=" * 92)
    print(
        f"{'palavra':16} {'pal':>4} {'ex':>3} {'pt':>3} {'ouve':>5} "
        f"{'IPA digitado':14} {'IPA real':14} ok  rótulos"
    )
    print("-" * 92)
    for word, s in rows:
        typed = str(s["ipa_typed"] or "—")
        agrees = {True: "sim", False: "NÃO", None: "—"}[s["ipa_agrees"]]
        print(
            f"{word:16} {s['words']:>4} {s['examples']:>3} "
            f"{'sim' if s['pt'] else '—':>3} {'SIM' if s['hears'] else '—':>5} "
            f"{typed:14.14} {str(s['ipa_real']):14.14} {agrees:3} "
            f"{', '.join(s['labels']) or '—'}"
        )
    typed_count = sum(1 for _, s in rows if s["ipa_typed"] is not None)
    wrong = sum(1 for _, s in rows if s["ipa_agrees"] is False)
    print()
    print(
        f"digitou IPA em {typed_count}/{len(rows)}; discorda da linha da tela em "
        f"{wrong}. Toda discordância é a tela contra o alto-falante na mesma bolha."
    )
    print(
        f"alegou ouvir em {sum(1 for _, s in rows if s['hears'])}/{len(rows)}; "
        f"usou português em {sum(1 for _, s in rows if s['pt'])}/{len(rows)}; "
        f"rotulou registro em {sum(1 for _, s in rows if s['labels'])}/{len(rows)} "
        "(sem fonte curada no app para conferir nenhum)."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help=(
            "quantas varreduras da lista (default: 1). A mesma palavra com o "
            "mesmo prompt mudou de resposta entre rodadas, e uma aula montada "
            "sobre geração não verificada é cara ou coroa: use 3 para ver."
        ),
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="só a metade determinística: base curada e corpus, sem Ollama",
    )
    args = parser.parse_args()
    deterministic()
    if args.no_model:
        return 0
    print()
    return asyncio.run(against_model(max(args.runs, 1)))


if __name__ == "__main__":
    raise SystemExit(main())
