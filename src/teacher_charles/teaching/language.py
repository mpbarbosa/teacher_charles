"""Did the student write in Portuguese? — and what the teacher is told if so.

Decision 1.1 makes the product English only, and the persona says so in its
first section: *if the student writes in Portuguese, answer in simple English
and invite them to try again in English.* Measured, that sentence is not
enough. ``qwen2.5:3b-instruct`` — the default since the prose measurement —
answered all three Portuguese turns of ``scripts/measure_prose.py`` in
Portuguese, fluently and warmly, which is the one reply the persona forbids
(``docs/plano-fase-1.md`` §5.5). ``qwen3:4b-instruct`` never did.

The cure is the one E4, E6 and the Structure button already measured: the
instruction counts when it sits next to the point of generation. So the turn
asks this module whether the student's message is Portuguese and, when it is,
a note rides inside that message (``TurnRunner._with_notes``), never at the
top of the prompt.

## Why two word lists and not a language model

The same reason the curated hint is a pattern: deterministic, free and
debuggable, and the question is coarse. Any sentence is held together by
function words — *eu*, *não*, *que*, *de* on one side, *I*, *the*, *is*, *to*
on the other — whatever it is about. So the module counts the function words
of each language in the message, and Portuguese has to outnumber English.

A share of Portuguese words was tried first and failed in the obvious place:
*Eu trabalho num banco desde 2019* is Portuguese with a single listed word in
six, because the rest is content (*trabalho*, *banco*, *desde*) that no short
list covers. Comparing against English instead of against the length is what
makes the content words drop out of the count on both sides.

## Which way it errs

Towards *not* matching a message that is mostly English. *I went to the feira
with my mother* is an English turn with a gap in the vocabulary, and the
teacher should treat it as one: offer *market*, keep talking. A note there
would tell him the student wrote Portuguese, which is false. So a tie goes to
English, and a word in quotes — the student pointing at a word, not writing in
it — is not counted at all. A Portuguese message that slips through costs what
it cost before this module existed: the persona alone, and nothing worse.
"""

from __future__ import annotations

import re

# Portuguese words that are not English words, so a hit is evidence and not an
# accident of spelling: function words and the commonest verbs. Words both
# languages write (*a*, *me*, *no*, *do*, *um*, *era*, *as*, *ali*, *num*,
# *dos*) are left out on purpose — counting them would hand Portuguese the
# English sentence they sit in.
_PORTUGUESE_WORDS = frozenset(
    """
    não nao você voce vocês eu ele ela eles elas nós nos meu minha meus minhas
    seu sua teu tua nosso nossa dele dela isso isto aquilo esse essa este esta
    que quê porque porquê por para pra com sem mas também tambem muito muita
    muitos muitas pouco mais menos já ainda sempre nunca hoje ontem amanhã
    agora aqui lá onde quando como qual quais quem se novo nova
    é e são sou está estou estava estão foi fui ser estar ter tenho
    tem tinha temos vou vai vamos fazer faz fiz pode posso poder quero quer
    queria sei sabe entendi entendo explicar dizer diz disse falar fala
    obrigado obrigada desculpa desculpe olá oi tchau sim tudo bem bom boa
    uma uns umas da das na nas ao aos à às pelo pela numa
    de em
    """.split()
)

# English function words, for the other side of the scale. Same rule in
# reverse: nothing a Portuguese sentence also writes (*a*, *no*, *me*, *do*).
_ENGLISH_WORDS = frozenset(
    """
    i i'm im you he she it we they my your his her its our their him us them
    the an this that these those what which who where when why how
    is are was were be been am do does did have has had will would can could
    should must not and or but so because if then than of to in on at for
    with from by about into there here very just like want go went get got
    yes ok okay please thanks thank hello hi sorry english
    """.split()
)

# Letters English writes only in loanwords. Outside quotes, a tilde or a
# cedilla makes the word count as Portuguese even when it is not on the list
# (*coração*, *pão*) — which still loses to the English around a name like
# *São Paulo* in an English sentence.
_PORTUGUESE_LETTERS = re.compile(r"[ãõç]", re.IGNORECASE)

# A quoted span is the student pointing at a word: *How do you say 'saudade'?*
# The guards on both sides keep an apostrophe inside a word (*I'm*, *Let's*)
# from opening a quote that swallows English.
_QUOTED = re.compile(r"(?<!\w)[\"'“‘][^\"'“”‘’\n]{1,40}[\"'”’](?!\w)")

_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)

NOTE = (
    "The student wrote this message in Portuguese. Answer in English only — "
    "not one word of Portuguese, not even to be kind or to translate. Use very "
    "short, simple English. Say that you will help, answer what they asked if "
    "you can, and invite them to try writing it in English, mistakes and all."
)


def evidence(text: str) -> tuple[int, int]:
    """Portuguese and English function words in the message, outside quotes."""
    words = [w.lower() for w in _WORD.findall(_QUOTED.sub(" ", text))]
    portuguese = sum(1 for w in words if w in _PORTUGUESE_WORDS or _PORTUGUESE_LETTERS.search(w))
    english = sum(1 for w in words if w in _ENGLISH_WORDS)
    return portuguese, english


def is_portuguese(text: str) -> bool:
    """Is this message written in Portuguese, rather than quoting it?"""
    portuguese, english = evidence(text)
    return portuguese > english


def portuguese_note(student_text: str) -> str:
    """The note for this turn, or an empty string."""
    return NOTE if is_portuguese(student_text) else ""


__all__ = [
    "NOTE",
    "evidence",
    "is_portuguese",
    "portuguese_note",
]
