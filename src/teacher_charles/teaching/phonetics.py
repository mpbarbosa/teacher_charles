"""A pronúncia esperada de um texto, em IPA.

Toda fala escrita na tela — a do aluno e a do professor — sai acompanhada da
transcrição fonética. É uma aula de inglês: a ortografia inglesa esconde o som,
e *though*, *through* e *tough* só deixam de ser armadilha quando o aluno vê
``ðoʊ``, ``θɹuː`` e ``tˈʌf`` embaixo delas.

## Por que o espeak, e não o modelo

Porque o modelo inventa. Está registrado no CLAUDE.md: perguntado pelo IPA, o
Charles ofereceu *pat* contra *bat* como contraste de **vogal** — a vogal é a
mesma nas duas, e o contraste é a consoante que ele próprio tinha proposto um
turno antes. Pronúncia é conhecimento de dicionário, e dicionário é tabela, não
geração. O mesmo princípio da base contrastiva: quando existe dado curado para
a coisa, o aplicativo cita o dado em vez de deixar o modelo improvisar.

O G2P é o ``phonemizer`` apontado para o ``espeak-ng`` do wheel — exatamente o
que a medição da Fase 2 usou para o gabarito (plano §2) e exatamente o que o
Kokoro usa para falar. Disso sai uma propriedade que nenhum outro G2P daria de
graça: **o IPA na tela é o IPA que a voz pronuncia**. Com voz ligada, o aluno lê
o símbolo e ouve o som correspondente na mesma sentença; se os dois
divergissem, a tela estaria ensinando contra o alto-falante.

## Por que isto mora em ``teaching/`` e não em ``voice/``

Porque não é voz. Entra texto, sai texto; não há áudio em lado nenhum deste
módulo. ``voice/`` são as duas pontas — bytes para texto, texto para PCM — e
pendurar aqui faria o IPA desaparecer junto com a voz, que é **desligada por
default**. A transcrição é do produto da Fase 1, não do extra da 1.5.

A consequência está no ``pyproject``: ``phonemizer`` e ``espeakng_loader``
saíram do extra ``voice`` e viraram dependência base. São ~21 MB contra os
~400 MB do extra, e é o preço de a funcionalidade existir na instalação que
todo mundo faz.

## O que esta transcrição é, e o que ela não é

É a pronúncia **do texto escrito**, palavra por palavra, no sotaque
``en-us``. Não é medida de nada: ninguém ouviu o aluno. Quando o texto veio do
microfone, ele veio do Whisper, e o IPA transcreve o que o Whisper entendeu —
não o que a boca do aluno fez. Avaliar pronúncia de verdade depende do
reconhecedor de fonema da E11, que é gate ainda aberto (plano §2.5). A tela diz
isso com todas as letras, e o prompt de capacidades também.

Marca de tônica **entra** (``with_stress=True``), ao contrário do gabarito do
``scripts/measure_phonemes.py``, e por motivo oposto: lá ela inventaria erro
contra um reconhecedor que não emite tônica; aqui ela é metade da lição, porque
``ˈrecord`` e ``reˈcord`` são palavras diferentes.
"""

from __future__ import annotations

import logging
import re
import threading
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any

from ..config import Settings
from ..stress_shift import read_nouns_as_nouns

log = logging.getLogger(__name__)

# O sotaque do produto. `bm_george`, a voz default do Kokoro, é britânica, mas
# a referência de pronúncia aqui segue americana porque é a que o aluno
# brasileiro encontra em todo lugar — e porque é o inventário que a medição da
# Fase 2 usou. Trocar aqui troca os dois lados ao mesmo tempo.
DEFAULT_LANGUAGE = "en-us"

# Quantas transcrições ficam em memória. O IPA é função pura do texto, e o
# texto se repete muito: a saudação, o enunciado do drill que o modo repete, a
# frase que o aluno reescreve depois da correção. Uma transcrição custa ~0,3 ms
# e algumas centenas de bytes, então o cache é mais sobre não repetir trabalho
# bobo no caminho do turno do que sobre tempo.
CACHE_SIZE = 512


class PhoneticsUnavailable(Exception):
    """O G2P não está utilizável: wheel ausente, espeak que não carrega.

    Reaja pelo tipo, nunca por substring — mesma regra dos erros de provider e
    dos erros de voz.
    """


class Phonetics:
    """O G2P do espeak, carregado quando alguém precisa dele.

    Carrega preguiçosamente e **nunca derruba o turno**: ``transcribe``
    devolve string vazia quando o backend não sobe, e a página simplesmente não
    mostra a linha de IPA. Mesma política do aquecimento da voz — o que fica
    degradado é a funcionalidade, não a aula.

    ``ready`` e ``last_error`` existem pela razão que o ``voice/base.Loadable``
    documenta: sem eles o erro sumiria, e o dev veria a linha não aparecer sem
    saber por quê. ``ready`` **não** dispara carga; quem carrega é o
    ``warm_up`` ou a primeira transcrição.
    """

    def __init__(
        self,
        language: str = DEFAULT_LANGUAGE,
        with_stress: bool = True,
    ) -> None:
        self.language = language
        self.with_stress = with_stress
        self._backend: Any | None = None
        self._last_error: str | None = None
        # O espeak-ng é uma biblioteca C com estado global, e o `phonemizer`
        # não promete reentrância. O aquecimento roda em thread de executor e
        # o turno roda no loop: o lock é o que impede os dois de se cruzarem
        # dentro do espeak.
        self._lock = threading.Lock()
        self._cache = lru_cache(maxsize=CACHE_SIZE)(self._phonemize)

    # --- estado ----------------------------------------------------------- #

    @property
    def ready(self) -> bool:
        """O backend já está carregado? Não força carga nenhuma."""
        return self._backend is not None

    @property
    def last_error(self) -> str | None:
        """Do que a última tentativa de carga reclamou, se reclamou."""
        return self._last_error

    # --- carga ------------------------------------------------------------ #

    def _load(self) -> Any:
        """O ``EspeakBackend`` desta máquina, apontado para o espeak do wheel.

        Importa aqui dentro, e não no topo, pela mesma razão que
        ``voice/__init__.py`` importa os motores dentro da função: subir o
        espeak custa ~0,25s, e ``import teacher_charles.app`` não deveria
        pagar isso para rodar um teste que não usa IPA.
        """
        try:
            import espeakng_loader
            from phonemizer.backend import EspeakBackend
            from phonemizer.backend.espeak.wrapper import EspeakWrapper
        except ImportError as exc:  # pragma: no cover - só sem o wheel
            raise PhoneticsUnavailable(f"phonemizer/espeakng_loader ausente: {exc}") from exc

        try:
            EspeakWrapper.set_library(espeakng_loader.get_library_path())
            EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
            return EspeakBackend(
                self.language,
                with_stress=self.with_stress,
                # A pontuação fica: a linha de IPA corre embaixo da linha de
                # texto, e sem vírgula e ponto de interrogação o aluno perde a
                # correspondência entre as duas na primeira sentença longa.
                preserve_punctuation=True,
                # Palavra que o espeak acha que é de outra língua não entra
                # marcada com `(pt)...(en)`: a marca é ruído para quem está
                # aprendendo o símbolo, e o que importa é como aquilo soa na
                # boca de quem fala inglês.
                language_switch="remove-flags",
            )
        except Exception as exc:  # noqa: BLE001 - o espeak estoura de várias formas
            raise PhoneticsUnavailable(f"espeak não carregou: {exc}") from exc

    def _ensure(self) -> Any:
        if self._backend is None:
            self._backend = self._load()
            self._last_error = None
        return self._backend

    async def warm_up(self) -> None:
        """Carrega o espeak antes de o aluno mandar a primeira mensagem.

        Nunca propaga erro, pela mesma razão que o aquecimento do LLM e o da
        voz não propagam: o ``/health`` já reporta, e derrubar a subida por
        causa do IPA seria pior que a aula sem IPA.
        """
        try:
            with self._lock:
                self._ensure()
        except PhoneticsUnavailable as exc:
            self._last_error = str(exc)
            log.warning("IPA indisponível, a aula segue sem ele: %s", exc)

    # --- transcrição ------------------------------------------------------ #

    def _phonemize(self, text: str) -> str:
        with self._lock:
            backend = self._ensure()
            ipa = str(backend.phonemize([text])[0]).strip()
            # Still under the lock: the readings that settle "record by
            # record" come from the same espeak, which is what makes them
            # comparable — and Kokoro settles its own the same way.
            return read_nouns_as_nouns(text, ipa, lambda t: str(backend.phonemize([t])[0]))

    def transcribe(self, text: str) -> str:
        """O IPA deste texto, ou string vazia quando não dá para produzi-lo.

        String vazia é resposta de primeira classe, não falha: a página não
        desenha a linha, e o turno segue. Texto em branco também volta vazio —
        não há som no nada.
        """
        clean = text.strip()
        if not clean:
            return ""

        try:
            return self._cache(clean)
        except PhoneticsUnavailable as exc:
            self._last_error = str(exc)
            return ""
        except Exception as exc:  # noqa: BLE001 - IPA nunca derruba o turno
            self._last_error = str(exc)
            log.exception("transcrição fonética falhou (turno segue)")
            return ""


def build_phonetics(settings: Settings) -> Phonetics | None:
    """O G2P que a configuração pede, ou ``None`` com o IPA desligado.

    ``None`` é estado de primeira classe, igual ao ``build_voice``: quem não
    quer a linha de IPA embaixo de cada fala desliga, e nada mais no caminho do
    turno precisa saber disso.
    """
    if not settings.phonetics_enabled:
        return None
    return Phonetics(
        language=settings.phonetics_language,
        with_stress=settings.phonetics_stress,
    )


def partial_name(phonetics_enabled: bool) -> str | None:
    """O partial que conta ao professor que o IPA aparece sozinho na tela.

    ``None`` quando não há o que contar. Existe pela mesma razão que
    ``capabilities.partial_name``: sem dizer ao modelo o que o produto faz, ele
    responde pelo palpite genérico de chatbot — e no caso da pronúncia o
    palpite dele já foi medido errado.
    """
    return "partials/app/phonetics" if phonetics_enabled else None


__all__ = [
    "CACHE_SIZE",
    "DEFAULT_LANGUAGE",
    "Phonetics",
    "PhoneticsUnavailable",
    "build_phonetics",
    "ipa_owners",
    "ipa_pairs",
    "partial_name",
]


# --- which IPA word is which written word ----------------------------------- #

# Stress and length marks, and the punctuation espeak keeps, carry no letters
# to compare: "ˈɔːl," and "ɔl" are the same word to the matcher.
_IPA_NOISE = re.compile(r"[ˈˌːˑ.,;:!?…—–\-'\"“”‘’()\[\]/]+")

# How many written words one IPA word can cover, and the reverse. espeak joins
# function words in connected speech ("on the" -> "ɔnðə", "of an" -> "əvən");
# three is the longest join seen in the lesson texts, with room.
_MAX_GROUP = 3

# What one pairing step is worth, in sounds. See the comment where it is used.
_STEP_BONUS = 1.5


def ipa_owners(words: list[str], tokens: list[str], per_word: list[str]) -> list[list[int]]:
    """For each written word, the IPA words (indices into ``tokens``) it is part of.

    ``tokens`` is espeak's reading of the whole sentence, split on spaces;
    ``per_word`` is its reading of each written word alone. The two do not
    line up one to one, because espeak joins words the way a speaker runs
    them together: *on the truth* is ``ɔnðə tɹˈuːθ``, two IPA words for three
    written ones. The sentence's reading is the one shown — it is what the
    voice says — so it is the per-word readings that are fitted to it.

    The fit is a small dynamic programme: in order, each step pairs one
    written word with up to three IPA words, or up to three written words with
    one IPA word, and the pairing whose sounds agree best wins. Every written
    word gets at least one IPA word, so the page always has something to light.
    """
    n, m = len(words), len(tokens)
    if n == 0 or m == 0:
        return [[] for _ in words]

    def sounds(text: str) -> str:
        return _IPA_NOISE.sub("", text)

    def agree(a: str, b: str) -> float:
        # Sounds in common count for, sounds left over on either side count
        # against: a score that only rewarded overlap would let a word grab
        # its neighbour's IPA for free ("days" taking "tə").
        a, b = sounds(a), sounds(b)
        blocks = SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()
        same = sum(x.size for x in blocks)
        return same - 0.5 * ((len(a) - same) + (len(b) - same))

    neg = float("-inf")
    best = [[neg] * (m + 1) for _ in range(n + 1)]
    step: list[list[tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if best[i][j] == neg:
                continue
            moves = [(1, k) for k in range(1, _MAX_GROUP + 1)]
            moves += [(k, 1) for k in range(2, _MAX_GROUP + 1)]
            for dw, dt in moves:
                ii, jj = i + dw, j + dt
                if ii > n or jj > m:
                    continue
                score = best[i][j] + agree("".join(per_word[i:ii]), "".join(tokens[j:jj]))
                # A bonus per step: every join costs a step, so words are
                # joined only where the sounds call for it. Without it a word
                # whose sounds change in context ("a" is ˈeɪ alone and ɐ in a
                # sentence) or that has none ("—") was cheaper absorbed into
                # its neighbour, and a sentence that lines up one to one came
                # back scrambled.
                score += _STEP_BONUS
                if score > best[ii][jj]:
                    best[ii][jj] = score
                    step[ii][jj] = (dw, dt)

    if best[n][m] == neg:
        # No pairing within the group limits: each word lights the IPA word
        # at its own proportion of the sentence, which is never far off.
        return [[min(m - 1, i * m // n)] for i in range(n)]

    owners: list[list[int]] = [[] for _ in words]
    i, j = n, m
    while i or j:
        dw, dt = step[i][j]  # type: ignore[misc]
        for w in range(i - dw, i):
            owners[w] = list(range(j - dt, j))
        i, j = i - dw, j - dt
    return owners


def ipa_pairs(words: list[str], tokens: list[str], owners: list[list[int]]) -> list[dict[str, str]]:
    """The written words and their IPA side by side, joined where espeak joins.

    ``owners`` is what :func:`ipa_owners` returned for the same ``words`` and
    ``tokens``. Its pairings are one written word to several IPA words, or
    several written words to one IPA word (*on the* -> ``ɔnðə``), so a run of
    written words owning the same IPA words is one pair: the page draws
    ``on the`` over ``ɔnðə`` rather than ``ɔnðə`` twice.

    A written word that owns nothing gets an empty IPA, never a guess.
    """
    pairs: list[dict[str, str]] = []
    last: list[int] | None = None
    for word, owned in zip(words, owners, strict=True):
        if owned and owned == last:
            pairs[-1]["text"] += f" {word}"
            continue
        pairs.append({"text": word, "ipa": " ".join(tokens[j] for j in owned)})
        last = owned
    return pairs
