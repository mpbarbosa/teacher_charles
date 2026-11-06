"""Transformar um fluxo de pedaços em sentenças que podem ser faladas.

Este módulo é a razão de a voz caber no orçamento desta máquina. O professor
gera a 7-8 tok/s; esperar a resposta inteira para começar a sintetizar somaria
os dois tempos e poria o turno em dez segundos ou mais. Falando por sentença, a
síntese da primeira corre **enquanto** o modelo escreve a segunda, e o que o
aluno espera é só a primeira sentença — que são 10 a 15 tokens.

É também o módulo que não precisa de peso nenhum para ser testado, e por isso
onde a maior parte dos testes de voz mora.

## O que é uma fronteira de sentença aqui

Um ``.``, ``!`` ou ``?``, seguido das aspas ou do parêntese que fecham, e
depois espaço ou fim do texto. Três exceções, e as três aparecem de verdade na
prosa que a persona pede:

- **Abreviação** — ``Mr. Silva`` é uma sentença, não duas.
- **Número decimal** — ``3.5 hours``.
- **Inicial** — ``J. R.``: uma letra sozinha antes do ponto.

E uma regra que não é sobre gramática: sentença curtíssima **não** é emitida
sozinha. Cada chamada de síntese tem custo fixo, e ``Oh!`` sozinho gastaria uma
chamada inteira para 300 ms de áudio — além de soar picado quando a reprodução
pula de um pedaço minúsculo para o seguinte. Ela espera e sai junto com a
próxima.

## O fim do pedaço também fecha sentença, e isso é escolha de latência

A fronteira vale com espaço depois **ou** no fim do que já chegou. Esperar o
espaço seria mais seguro: enquanto o texto chega em pedaços, ``It takes 3.``
pode ser fim de sentença ou o começo de ``3.5 hours``. Mas esperar custaria um
token de atraso em **toda** sentença, e o token é o recurso caro aqui — a
7-8 tok/s ele vale ~130 ms, pagos no único ponto do turno que o aluno espera
em silêncio.

O risco se paga sem o atraso, porque ele tem forma conhecida: o caso ruim é um
dígito antes do ponto. Esse, e só esse, espera o pedaço seguinte
(``_is_false_boundary``). Abreviação e inicial já são detectadas pelo que está
antes do ponto, que sempre chegou.
"""

from __future__ import annotations

import re

# Sentença com menos que isto espera pela próxima. 12 caracteres deixam
# "Good morning!" sair sozinho — é uma abertura de turno inteira — e seguram
# "Oh!" e "Right." para a sentença seguinte.
MIN_SENTENCE_CHARS = 12

# Abreviações que a prosa da persona de fato produz. Lista curta de propósito:
# cada item aqui é um ponto que o professor **não** vai usar como pausa, e
# errar para o lado de falar demais é pior que errar para o lado de uma pausa
# sobrando.
ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "st",
        "jr",
        "sr",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "approx",
        "no",
    }
)

# Pontuação de fim, mais o que pode vir depois dela ainda dentro da sentença:
# aspas e parênteses que fecham. `You said "I have 30 years."` é uma sentença,
# e o ponto dela está dentro das aspas.
_BOUNDARY = re.compile(r'([.!?])([")\]”’]*)(\s|$)')

# A última "palavra" antes do ponto, para testar abreviação e inicial.
_LAST_WORD = re.compile(r"([\w.]+)$", re.UNICODE)

_MARKDOWN_EMPHASIS = re.compile(r"(\*{1,3}|_{1,3}|`+)(.+?)\1", re.DOTALL)
_MARKDOWN_LEADING = re.compile(r"^[>#\-\*\+]+\s*", re.MULTILINE)
_LEFTOVER_MARKS = re.compile(r"[*_`#]+")
_WHITESPACE = re.compile(r"\s+")


def for_speech(text: str) -> str:
    """O mesmo texto sem a marcação que a persona usa para a tela.

    A persona escreve em markdown — ``*to be*`` para destacar a estrutura,
    ``>`` para o exemplo de correção. Na tela isso ajuda; na voz, o Kokoro lê
    os asteriscos. Esta função é a borda entre as duas saídas, e não altera o
    que vai para o ``correction`` nem para o histórico: o texto gravado
    continua sendo o que o modelo escreveu.
    """
    cleaned = _MARKDOWN_EMPHASIS.sub(r"\2", text)
    cleaned = _MARKDOWN_LEADING.sub("", cleaned)
    cleaned = _LEFTOVER_MARKS.sub("", cleaned)
    return _WHITESPACE.sub(" ", cleaned).strip()


def _is_false_boundary(buffer: str, dot_index: int) -> bool:
    """O ponto em ``dot_index`` fecha a sentença, ou é abreviação/decimal/inicial?"""
    before = buffer[:dot_index]
    after = buffer[dot_index + 1 :]

    # Decimal: dígito antes do ponto. Com dígito depois, é certeza (`3.5`); com
    # o buffer acabando no ponto, é indecidível, e indecidível aqui espera —
    # é o único caso em que o fim do pedaço não fecha sentença (ver o
    # cabeçalho do módulo).
    if before[-1:].isdigit() and (after[:1].isdigit() or not after):
        return True

    match = _LAST_WORD.search(before)
    if match is None:
        return False
    word = match.group(1).rstrip(".").lower()

    # Inicial: uma letra sozinha. `J. R. Tolkien`
    if len(word) == 1 and word.isalpha():
        return True

    return word in ABBREVIATIONS


def split_sentences(text: str, min_chars: int = MIN_SENTENCE_CHARS) -> list[str]:
    """Divide um texto completo em sentenças faláveis.

    Usado quando o texto inteiro já existe — na medição e nos testes. O
    caminho do turno usa o ``SentenceAssembler``, que faz o mesmo sem esperar
    o fim.
    """
    assembler = SentenceAssembler(min_chars=min_chars)
    sentences = assembler.push(text)
    tail = assembler.flush()
    if tail:
        sentences.append(tail)
    return sentences


class SentenceAssembler:
    """Junta pedaços de texto e devolve sentenças quando elas fecham.

    Um objeto por turno. ``push`` devolve o que já pode ser falado (em geral
    nada, às vezes uma, raramente duas) e ``flush`` devolve o rabo que sobrou
    quando o professor para de falar sem pontuar.
    """

    __slots__ = ("_buffer", "_min_chars")

    def __init__(self, min_chars: int = MIN_SENTENCE_CHARS) -> None:
        self._buffer = ""
        self._min_chars = min_chars

    @property
    def pending(self) -> str:
        """O que está acumulado e ainda não virou sentença."""
        return self._buffer.strip()

    def push(self, chunk: str) -> list[str]:
        """Acrescenta um pedaço e devolve as sentenças que fecharam."""
        if not chunk:
            return []
        self._buffer += chunk

        out: list[str] = []
        while True:
            sentence = self._take_one()
            if sentence is None:
                break
            out.append(sentence)
        return out

    def _take_one(self) -> str | None:
        """A primeira sentença fechada do buffer, se houver uma longa o bastante."""
        search_from = 0
        while True:
            match = _BOUNDARY.search(self._buffer, search_from)
            if match is None:
                return None

            dot_index = match.start(1)
            if _is_false_boundary(self._buffer, dot_index):
                search_from = match.end(1)
                continue

            end = match.end(2)
            candidate = for_speech(self._buffer[:end])

            # Curta demais: não corta aqui, procura a próxima fronteira. O
            # pedaço curto sai junto com a sentença seguinte.
            if len(candidate) < self._min_chars:
                search_from = match.end(2)
                continue

            self._buffer = self._buffer[match.end(2) :]
            return candidate

    def flush(self) -> str | None:
        """O resto, limpo para fala. Esvazia o buffer; ``None`` se não sobrou nada."""
        rest = for_speech(self._buffer)
        self._buffer = ""
        return rest or None
