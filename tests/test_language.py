"""The Portuguese guard: a note that keeps the teacher in English.

Measured before it existed (``docs/plano-fase-1.md`` §5.5): the default 3B
answered all three Portuguese turns of ``scripts/measure_prose.py`` in
Portuguese, against a persona that forbids exactly that. The detector is what
decides when the note rides along, so it is held here case by case, and the
turn tests at the bottom hold where the note goes.
"""

from __future__ import annotations

import pytest

from teacher_charles.llm import FakeProvider
from teacher_charles.prompts import PromptLibrary
from teacher_charles.store import db
from teacher_charles.store.repo import SessionRepo, StudentRepo
from teacher_charles.teaching import TurnRunner
from teacher_charles.teaching.language import NOTE, evidence, is_portuguese, portuguese_note

PORTUGUESE = [
    # The three probes of the measurement.
    "Desculpa, não entendi. Pode explicar de novo?",
    "Hoje eu fui ao médico e estou muito cansado.",
    "Como se diz 'saudade' em inglês?",
    # Short turns a student sends without thinking.
    "Oi, tudo bem?",
    "Obrigado!",
    "Eu não sei",
    "Sim",
    # Mostly content words, which no list covers: the case that sank the
    # first version, a share of listed words against the message's length.
    "Eu trabalho num banco desde 2019",
    "Era uma vez",
    "Coração",
    "Você pode me ajudar com o meu trabalho?",
    # Mostly Portuguese with an English word in it is still a Portuguese turn.
    "Teacher, eu não entendi the homework",
]

ENGLISH = [
    # A Portuguese word inside an English sentence: a gap in the vocabulary,
    # and the teacher should offer the word, not switch the rules.
    "Yesterday I went to the feira with my mother and we bought many vegetables.",
    "I am going to the praia with my amigos, it is very bom.",
    "I like to eat pão de queijo in the morning.",
    # Names carry tildes and *de*; the English around them wins.
    "My name is João and I am from São Paulo.",
    "Do you know Ali? He is from Rio de Janeiro.",
    # Quoted words are the student pointing at a word, not writing in it.
    "How do you say 'saudade' in English?",
    "What does 'você' mean?",
    "Is the 'th' in 'think' the same sound as the 'th' in 'this'?",
    # Turns of the prose battery, errors included: a calque is English.
    "I have 30 years and I work in a bank.",
    "I lost the bus this morning, so I arrived late at work.",
    "OK, I'm ready. Let's practise.",
    "e.g. I like it",
    # Nothing to count on either side: not Portuguese.
    "Saudade",
    "Football tomorrow?",
    "",
]


@pytest.mark.parametrize("text", PORTUGUESE)
def test_portuguese_is_seen(text):
    assert is_portuguese(text), evidence(text)


@pytest.mark.parametrize("text", ENGLISH)
def test_english_with_portuguese_in_it_is_not(text):
    assert not is_portuguese(text), evidence(text)


def test_a_tie_goes_to_english():
    """Erring towards English is the cheap side: a miss costs the persona alone."""
    text = "Teacher, eu the"
    assert evidence(text) == (1, 1)
    assert not is_portuguese(text)


def test_an_apostrophe_inside_a_word_does_not_open_a_quote():
    """*I'm … Let's* once read as one quoted span and swallowed the English."""
    portuguese, english = evidence("OK, I'm ready. Let's practise.")
    assert portuguese == 0
    assert english >= 2


def test_the_note_is_empty_for_english():
    assert portuguese_note("I have 30 years and I work in a bank.") == ""
    assert portuguese_note("Desculpa, não entendi.") == NOTE


def test_the_note_says_english_only():
    """The note is the rule, not a description — and it names the temptation."""
    assert "English only" in NOTE
    assert "not one word of Portuguese" in NOTE


# --- where the note goes --------------------------------------------------- #


@pytest.fixture
def conn(tmp_path):
    connection = db.init_db(tmp_path / "language.db")
    yield connection
    connection.close()


@pytest.fixture
def prompts(tmp_path):
    root = tmp_path / "prompts"
    root.mkdir()
    (root / "persona.md").write_text("You are Charles.", encoding="utf-8")
    return PromptLibrary(root)


async def _last_messages(prompts, conn, text):
    student = StudentRepo(conn).get_or_create("ana")
    session_id = SessionRepo(conn).current_or_start(student.id)
    provider = FakeProvider(replies="ok")
    runner = TurnRunner(provider=provider, prompts=prompts, conn=conn)
    async for _ in runner.run(session_id, text, student):
        pass
    return provider.calls_matching("You are Charles")[-1].messages


async def test_the_note_rides_with_the_student_message(prompts, conn):
    """Inside the last user message, as the curated hint does — never a system
    message, which the chat template would hoist above the history."""
    messages = await _last_messages(prompts, conn, "Desculpa, não entendi. Pode explicar?")

    assert [m.role for m in messages] == ["system", "user"]
    last = messages[-1]
    assert last.content.startswith("Desculpa, não entendi.")
    assert NOTE in last.content


async def test_an_english_turn_carries_no_note(prompts, conn):
    messages = await _last_messages(prompts, conn, "I went to the feira with my mother.")

    assert NOTE not in messages[-1].content
    assert "Teacher's notes" not in messages[-1].content


async def test_the_persona_is_untouched(prompts, conn):
    """The prefix cache: a Portuguese turn must not change the system prompt."""
    messages = await _last_messages(prompts, conn, "Oi, tudo bem?")

    assert messages[0].content == "You are Charles."
