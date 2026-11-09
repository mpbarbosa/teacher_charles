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
import threading
from functools import lru_cache
from typing import Any

from ..config import Settings

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
            return str(backend.phonemize([text])[0]).strip()

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
    "partial_name",
]
