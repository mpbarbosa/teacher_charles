"""The syntax view: how a sentence is built, from a parser and never the model.

Three halves, split by what they need installed:

**The tree-to-breakdown logic** — driven by hand-built trees. ``analyse`` reads
tokens by attribute and imports nothing from spaCy, so these run with the
``syntax`` extra absent, which is how CI runs.

**The loader and the route** — with ``_load`` swapped for a stub. What they
hold in place is policy: nothing loads by accident, a parser that is not there
costs the button and never the lesson, and the route records nothing.

**The real pipeline** — marked ``integration`` and skipped without the extra,
the same arrangement as the voice tests that load Whisper.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from teacher_charles.app import create_app
from teacher_charles.config import Settings
from teacher_charles.llm import FakeProvider
from teacher_charles.prompts import PromptLibrary
from teacher_charles.store.db import init_db
from teacher_charles.teaching import TurnRunner
from teacher_charles.teaching import syntax as syntax_module
from teacher_charles.teaching.syntax import (
    NOTE,
    REQUEST_NOTE,
    Breakdown,
    Syntax,
    SyntaxUnavailable,
    analyse,
    asks_for_breakdown,
    build_syntax,
    on_screen,
    partial_name,
    request_note,
)


def make_settings(**overrides) -> Settings:
    """Test settings, without reading the repository's ``.env`` (see conftest)."""
    return Settings(_env_file=None, **overrides)


# --- Hand-built trees ------------------------------------------------------ #


@dataclass(eq=False)
class Tok:
    """The handful of attributes ``analyse`` reads off a spaCy token."""

    i: int
    text: str
    pos_: str
    tag_: str
    dep_: str
    head_i: int
    whitespace_: str = " "
    doc: list[Tok] = field(default_factory=list, repr=False)

    @property
    def head(self) -> Tok:
        return self.doc[self.head_i]

    @property
    def children(self) -> list[Tok]:
        return [t for t in self.doc if t.head_i == self.i and t is not self]

    @property
    def subtree(self) -> list[Tok]:
        out = [self]
        for child in self.children:
            out.extend(child.subtree)
        return sorted(out, key=lambda t: t.i)


class Sent(list):
    @property
    def root(self) -> Tok:
        return next(t for t in self if t.dep_ == "ROOT")

    @property
    def text(self) -> str:
        return "".join(t.text + t.whitespace_ for t in self)


@dataclass
class Doc:
    sents: list[Sent]


def tree(*rows: tuple) -> Doc:
    """One sentence from ``(text, pos, tag, dep, head index[, whitespace])`` rows."""
    tokens = [Tok(i, *row) for i, row in enumerate(rows)]
    for token in tokens:
        token.doc = tokens
    return Doc(sents=[Sent(tokens)])


def parts_of(doc: Doc) -> list[tuple[str, str]]:
    (sentence,) = analyse(doc).sentences
    return [(part.text, part.role) for part in sentence.parts]


# What spaCy returned for the sentence that started this, token for token.
IN_A_SEWER = (
    ("I", "PRON", "PRP", "nsubj", 1, ""),
    ("'m", "AUX", "VBP", "ROOT", 1),
    ("in", "ADP", "IN", "prep", 1),
    ("a", "DET", "DT", "det", 4),
    ("sewer", "NOUN", "NN", "pobj", 2, ""),
    (".", "PUNCT", ".", "punct", 1, ""),
)


# --- The breakdown --------------------------------------------------------- #


def test_the_sentence_that_started_it():
    assert parts_of(tree(*IN_A_SEWER)) == [
        ("I", "subject"),
        ("'m", "verb"),
        ("in a sewer", "prepositional phrase"),
    ]


def test_every_word_names_its_part_of_speech_and_what_it_hangs_from():
    (sentence,) = analyse(tree(*IN_A_SEWER)).sentences
    words = sentence.parts[2].words
    assert [(w.text, w.pos, w.role, w.head) for w in words] == [
        ("in", "preposition", "head of the prepositional phrase", "'m"),
        ("a", "determiner", "determiner", "sewer"),
        ("sewer", "noun", "object of the preposition", "in"),
    ]


def test_be_as_the_only_verb_is_not_called_an_auxiliary():
    """The parser tags *'m* AUX; with no other verb to help, the panel says verb."""
    (sentence,) = analyse(tree(*IN_A_SEWER)).sentences
    (be,) = sentence.parts[1].words
    assert be.pos == "verb"
    assert be.role == "main verb"
    assert be.head == ""


def test_auxiliaries_next_to_the_verb_fuse_with_it():
    doc = tree(
        ("He", "PRON", "PRP", "nsubj", 3),
        ("has", "AUX", "VBZ", "aux", 3),
        ("been", "AUX", "VBN", "aux", 3),
        ("painting", "VERB", "VBG", "ROOT", 3, ""),
        (".", "PUNCT", ".", "punct", 3, ""),
    )
    assert parts_of(doc) == [("He", "subject"), ("has been painting", "verb")]
    # Fused in the chunk, still told apart word by word.
    (sentence,) = analyse(doc).sentences
    assert [w.pos for w in sentence.parts[1].words] == ["auxiliary verb", "auxiliary verb", "verb"]


def test_an_auxiliary_apart_from_the_verb_stays_apart():
    """*did … do*: the subject between them is the whole lesson of a question."""
    doc = tree(
        ("What", "PRON", "WP", "dobj", 3),
        ("did", "AUX", "VBD", "aux", 3),
        ("you", "PRON", "PRP", "nsubj", 3),
        ("do", "VERB", "VB", "ROOT", 3, ""),
        ("?", "PUNCT", ".", "punct", 3, ""),
    )
    assert parts_of(doc) == [
        ("What", "direct object"),
        ("did", "auxiliary"),
        ("you", "subject"),
        ("do", "verb"),
    ]


def test_a_sentence_without_a_verb_is_one_chunk_and_says_so():
    doc = tree(
        ("Good", "ADJ", "JJ", "amod", 1),
        ("morning", "NOUN", "NN", "ROOT", 1, ""),
        (".", "PUNCT", ".", "punct", 1, ""),
    )
    assert parts_of(doc) == [("Good morning", "phrase without a verb")]
    (sentence,) = analyse(doc).sentences
    assert sentence.parts[0].words[1].role == "head of the phrase"


def test_punctuation_is_text_and_never_a_part_or_a_word():
    (sentence,) = analyse(tree(*IN_A_SEWER)).sentences
    assert sentence.text == "I'm in a sewer."
    assert all("." not in part.text for part in sentence.parts)
    assert all(word.text != "." for part in sentence.parts for word in part.words)


def test_a_sentence_of_punctuation_alone_yields_nothing():
    doc = tree(("...", "PUNCT", ":", "ROOT", 0, ""))
    assert analyse(doc).sentences == ()


def test_a_tag_outside_the_tables_is_shown_and_not_guessed():
    """The tables are closed on purpose; a new tag must not get an invented name."""
    doc = tree(
        ("It", "PRON", "PRP", "nsubj", 1),
        ("works", "VERB", "VBZ", "ROOT", 1),
        ("zzz", "QQQ", "QQ", "newdep", 1, ""),
    )
    (sentence,) = analyse(doc).sentences
    (odd,) = sentence.parts[2].words
    assert odd.pos == "other (QQQ)"
    assert odd.role == "other (newdep)"


def test_the_tree_decides_between_preposition_and_conjunction():
    """*since Monday* came back SCONJ heading a prepositional phrase."""
    doc = tree(
        ("Wait", "VERB", "VB", "ROOT", 0),
        ("since", "SCONJ", "IN", "prep", 0),
        ("Monday", "PROPN", "NNP", "pobj", 1, ""),
    )
    (sentence,) = analyse(doc).sentences
    assert sentence.parts[1].words[0].pos == "preposition"


def test_every_label_a_student_can_read_is_plain_text():
    """No parser tag leaks to the page by way of the tables themselves."""
    tables = (
        syntax_module.PARTS_OF_SPEECH,
        syntax_module.WORD_ROLES,
        syntax_module.PART_ROLES,
    )
    for table in tables:
        for tag, label in table.items():
            assert label and label == label.strip()
            assert label != tag or tag == "X"
            assert "_" not in label


# --- The note -------------------------------------------------------------- #


def test_the_note_travels_with_every_result():
    """A tidy breakdown under a wrong sentence must not read as approval.

    The page prints whatever ``note`` carries, so the guard is here: delete it
    from ``as_dict`` and the panel loses the one line that says a breakdown is
    not a verdict.
    """
    payload = analyse(tree(*IN_A_SEWER)).as_dict()
    assert payload["note"] == NOTE
    assert "not whether it is correct" in NOTE
    assert "not from Charles" in NOTE


def test_the_payload_is_what_the_page_draws():
    payload = analyse(tree(*IN_A_SEWER)).as_dict()
    assert set(payload) == {"sentences", "note"}
    (sentence,) = payload["sentences"]
    assert set(sentence) == {"text", "parts"}
    assert set(sentence["parts"][0]) == {"text", "role", "words"}
    assert set(sentence["parts"][0]["words"][0]) == {"text", "pos", "role", "head"}


# --- The loader ------------------------------------------------------------ #


@pytest.fixture
def stub(monkeypatch) -> Syntax:
    """A ``Syntax`` whose pipeline is a function returning the tree above."""
    parser = Syntax()
    parser.calls = []

    def nlp(text):
        parser.calls.append(text)
        return tree(*IN_A_SEWER)

    monkeypatch.setattr(parser, "_load", lambda: nlp)
    return parser


def test_parse_returns_the_breakdown(stub):
    result = stub.parse("I'm in a sewer.")
    assert isinstance(result, Breakdown)
    assert [part.role for part in result.sentences[0].parts] == [
        "subject",
        "verb",
        "prepositional phrase",
    ]


def test_ready_and_installed_load_nothing(monkeypatch):
    parser = Syntax()
    monkeypatch.setattr(parser, "_load", lambda: pytest.fail("loaded by a status check"))
    assert parser.ready is False
    assert parser.last_error is None
    _ = parser.installed
    assert parser.ready is False


def test_a_pipeline_that_is_not_there_is_not_installed():
    assert Syntax(model="no_such_pipeline_xyz").installed is False
    # `find_spec` raises on a dotted name with no parent instead of saying no.
    assert Syntax(model="no_such_package.pipeline").installed is False


def test_a_parser_that_will_not_load_raises_its_own_type(monkeypatch):
    parser = Syntax()

    def broken():
        raise SyntaxUnavailable("spaCy went missing")

    monkeypatch.setattr(parser, "_load", broken)
    with pytest.raises(SyntaxUnavailable):
        parser.parse("Hello there.")
    # Kept for `/health`: the route answers the student, this answers the dev.
    assert parser.last_error == "spaCy went missing"
    assert parser.ready is False


async def test_warm_up_loads(stub):
    await stub.warm_up()
    assert stub.ready is True


async def test_warm_up_never_propagates(monkeypatch):
    parser = Syntax()

    def broken():
        raise SyntaxUnavailable("no pipeline")

    monkeypatch.setattr(parser, "_load", broken)
    await parser.warm_up()
    assert parser.ready is False
    assert parser.last_error == "no pipeline"


def test_the_cache_does_not_repeat_the_work(stub):
    stub.parse("I'm in a sewer.")
    stub.parse("  I'm in a sewer.  ")
    assert stub.calls == ["I'm in a sewer."]


def test_blank_text_is_an_empty_breakdown_and_loads_nothing(monkeypatch):
    parser = Syntax()
    monkeypatch.setattr(parser, "_load", lambda: pytest.fail("loaded for blank text"))
    assert parser.parse("   ").sentences == ()


# --- Configuration --------------------------------------------------------- #


def test_off_by_default():
    """Like the voice: an extra the base install does not carry."""
    assert make_settings().syntax_enabled is False
    assert build_syntax(make_settings()) is None


def test_build_follows_the_configuration():
    parser = build_syntax(make_settings(syntax_enabled=True, syntax_model="en_core_web_md"))
    assert parser is not None
    assert parser.model == "en_core_web_md"
    assert parser.ready is False


def test_asked_for_but_not_installed_is_off_screen():
    assert on_screen(None) is False
    assert on_screen(Syntax(model="no_such_pipeline_xyz")) is False


def test_the_partial_exists_only_with_the_button_on_screen():
    assert partial_name(False) is None
    assert partial_name(True) == "partials/app/syntax"


def test_the_partial_the_name_promises_exists():
    prompts = PromptLibrary(make_settings().prompts_dir)
    text = prompts.get(partial_name(True))
    assert "**Structure**" in text


def test_the_partial_does_not_describe_what_the_panel_shows():
    """Measured: a partial that listed the panel's contents primed the model.

    With "subject, verb, objects … the part of speech of each word" in the
    prefix, a 3B model typed its own breakdown three times out of three; with
    no partial at all it typed none. The words that name the analysis stay
    out of the top of the prompt — the instruction lives in ``REQUEST_NOTE``.
    """
    prompts = PromptLibrary(make_settings().prompts_dir)
    text = prompts.get(partial_name(True)).lower()
    for primer in ("subject", "verb", "object", "part of speech", "parts of speech", "phrase"):
        assert primer not in text, primer


# --- When the student asks ------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        'Can you make a syntax analysis of the sentence "I\'m in a sewer."?',
        "Make syntax analysis of the sentence.",
        "please do a syntactic analysis",
        "Could you parse this for me: I am tired",
        "What are the parts of speech here?",
        "Can you analyse this sentence?",
        "Analyze the sentence I wrote, please",
        "what is the structure of this sentence?",
        "Can you break it down?",
        "Break this sentence down for me",
        "Which is the subject and the verb?",
    ],
)
def test_asking_for_a_breakdown_is_recognised(text):
    assert asks_for_breakdown(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "I'm in a sewer.",
        "I had a nervous breakdown last year.",
        "My car broke down on the highway.",
        "The structure of the building is old.",
        "I will analyse the report tomorrow.",
        "My parents are sparse with praise.",
        "We talked about the subject of taxes.",
    ],
)
def test_ordinary_talk_is_not_a_request(text):
    assert asks_for_breakdown(text) is False


def test_no_note_without_the_button():
    """Sending the student to a control that is not there is the microphone
    mistake again."""
    asking = "Can you make a syntax analysis of this?"
    assert request_note(False, asking) == ""
    assert request_note(True, asking) == REQUEST_NOTE
    assert request_note(True, "I'm in a sewer.") == ""


def test_the_note_forbids_the_analysis_and_names_the_button():
    assert "Do not analyse it yourself" in REQUEST_NOTE
    assert "Structure button" in REQUEST_NOTE


# --- The teacher's prompt -------------------------------------------------- #


def runner(tmp_path, **kwargs) -> TurnRunner:
    settings = make_settings()
    return TurnRunner(
        provider=FakeProvider(),
        prompts=PromptLibrary(settings.prompts_dir),
        conn=init_db(tmp_path / "syntax.db"),
        **kwargs,
    )


def test_the_teacher_is_told_about_the_button_only_when_it_is_there(tmp_path):
    without = runner(tmp_path).app_prompt()
    with_button = runner(tmp_path, syntax_enabled=True).app_prompt()
    assert "Structure" not in without
    assert "**Structure**" in with_button


def test_the_button_does_not_add_a_message_to_the_prefix(tmp_path):
    """It rides inside the app-capabilities message, like the IPA partial.

    A message of its own would shift every later one and cost the prefix cache
    its shape for a paragraph that is fixed at startup.
    """
    plain = runner(tmp_path).stable_prefix("B1", "conversation")
    with_button = runner(tmp_path, syntax_enabled=True).stable_prefix("B1", "conversation")
    assert len(with_button) == len(plain)
    # And the persona, first and unaltered, is still what leads.
    assert with_button[0] == plain[0]


# --- The route ------------------------------------------------------------- #


def settings_for(tmp_path, **overrides) -> Settings:
    base = {
        "db_path": tmp_path / "syntax.db",
        "provider": "fake",
        "llm_warmup": False,
        "syntax_enabled": True,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


@pytest.fixture
def installed(monkeypatch):
    """Makes the extra look installed and the pipeline a stub.

    Patched on the class, before the app is built: the lifespan decides what
    to warm up and what to tell the teacher from ``installed``.
    """
    monkeypatch.setattr(Syntax, "installed", property(lambda self: True))
    monkeypatch.setattr(Syntax, "_load", lambda self: lambda text: tree(*IN_A_SEWER))


@pytest.fixture
def client(tmp_path, installed):
    with TestClient(create_app(settings_for(tmp_path), provider=FakeProvider())) as handle:
        yield handle


def test_the_route_returns_the_breakdown_and_the_note(client):
    response = client.post("/api/syntax", json={"text": "I'm in a sewer."})
    assert response.status_code == 200
    body = response.json()
    assert [part["role"] for part in body["sentences"][0]["parts"]] == [
        "subject",
        "verb",
        "prepositional phrase",
    ]
    assert body["note"] == NOTE


def test_the_route_records_nothing(client):
    """Looking at a sentence is not a turn, a correction or a review.

    Same reason ``/api/stt`` records no turn: a route that reads must not
    leave evidence the level tracker or the schedule would later count.
    """
    conn = client.app.state.db
    tables = ("turn", "correction", "rule_review", "level_signal")

    def counts() -> list[int]:
        return [conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables]

    before = counts()
    client.post("/api/syntax", json={"text": "I'm in a sewer."})
    assert counts() == before
    assert client.app.state.provider.calls == []


def test_the_route_refuses_empty_and_oversized_text(client):
    assert client.post("/api/syntax", json={"text": ""}).status_code == 422
    assert client.post("/api/syntax", json={"text": "a" * 4001}).status_code == 422


def test_the_page_draws_the_button_only_when_it_can_work(client):
    assert 'data-syntax="on"' in client.get("/").text


def test_view_off_means_no_button_and_a_503_that_says_how(tmp_path):
    settings = settings_for(tmp_path, syntax_enabled=False)
    with TestClient(create_app(settings, provider=FakeProvider())) as off:
        assert "data-syntax" not in off.get("/").text
        response = off.post("/api/syntax", json={"text": "Hello."})
        # 503 and not 404: the route exists, the service is not ready for it.
        assert response.status_code == 503
        assert "TC_SYNTAX_ENABLED" in response.json()["how"]
        assert off.get("/health").json()["syntax"] == {"ok": True, "enabled": False}


def test_asked_for_without_the_extra_costs_the_button_not_the_lesson(tmp_path, monkeypatch):
    """The default install with the flag on: no button, no partial, a reason."""
    monkeypatch.setattr(Syntax, "installed", property(lambda self: False))

    def missing(self):
        raise SyntaxUnavailable("spaCy is not installed")

    monkeypatch.setattr(Syntax, "_load", missing)

    with TestClient(create_app(settings_for(tmp_path), provider=FakeProvider())) as bare:
        assert "data-syntax" not in bare.get("/").text
        # The teacher is not told to point at a button the page did not draw.
        assert "Structure" not in bare.app.state.runner.app_prompt()
        # Nothing was warmed up, so nothing logged a failure at startup.
        assert bare.app.state.syntax_warmup is None

        response = bare.post("/api/syntax", json={"text": "Hello."})
        assert response.status_code == 503
        assert "spaCy is not installed" in response.json()["detail"]

        health = bare.get("/health").json()["syntax"]
        assert health["ok"] is False
        assert health["installed"] is False
        assert "uv sync --extra syntax" in health["detail"]


def test_the_view_does_not_decide_whether_the_service_is_ready(tmp_path, monkeypatch):
    """Same as the IPA line: the lesson is whole without the panel."""
    monkeypatch.setattr(Syntax, "installed", property(lambda self: False))
    with (
        TestClient(create_app(settings_for(tmp_path), provider=FakeProvider())) as bare,
        TestClient(
            create_app(
                settings_for(tmp_path / "off", syntax_enabled=False), provider=FakeProvider()
            )
        ) as off,
    ):
        assert bare.get("/health").json()["status"] == off.get("/health").json()["status"]


def test_the_teacher_learns_about_the_button_through_the_app(client):
    assert "**Structure**" in client.app.state.runner.app_prompt()


ASKING = 'Can you make a syntax analysis of the sentence "I\'m in a sewer."?'


def test_the_request_note_rides_with_the_student_message(client):
    """At the end of the prompt, inside the student's turn — never a system
    message after the history, which the chat template would hoist to the top
    and which would cost the history its cache (CLAUDE.md, E7).
    """
    client.post("/api/chat", json={"message": ASKING})
    messages = client.app.state.provider.last_call.messages

    last = messages[-1]
    assert last.role == "user"
    assert last.content.startswith(ASKING)
    assert REQUEST_NOTE in last.content
    # Every system message sits before the first turn of the conversation.
    roles = [m.role for m in messages]
    assert "system" not in roles[roles.index("user") :]


def test_an_ordinary_turn_carries_no_request_note(client):
    client.post("/api/chat", json={"message": "I'm in a sewer."})
    last = client.app.state.provider.last_call.messages[-1]
    assert "Structure" not in last.content


def test_without_the_button_the_question_carries_no_note(tmp_path):
    settings = settings_for(tmp_path, syntax_enabled=False)
    with TestClient(create_app(settings, provider=FakeProvider())) as off:
        off.post("/api/chat", json={"message": ASKING})
        last = off.app.state.provider.last_call.messages[-1]
        assert "Structure" not in last.content


# --- The real pipeline ----------------------------------------------------- #

needs_spacy = pytest.mark.skipif(
    importlib.util.find_spec("spacy") is None
    or importlib.util.find_spec(syntax_module.DEFAULT_MODEL) is None,
    reason="the `syntax` extra is not installed (uv sync --extra syntax)",
)


@pytest.fixture(scope="module")
def real() -> Syntax:
    return Syntax()


@needs_spacy
@pytest.mark.integration
def test_real_pipeline_on_the_sentence_that_started_it(real):
    (sentence,) = real.parse("I'm in a sewer.").sentences
    assert [(part.text, part.role) for part in sentence.parts] == [
        ("I", "subject"),
        ("'m", "verb"),
        ("in a sewer", "prepositional phrase"),
    ]
    assert real.ready is True


@needs_spacy
@pytest.mark.integration
def test_real_pipeline_splits_a_message_into_sentences(real):
    result = real.parse("Hello! I'm Charles. Tell me about your day — in English.")
    assert [s.text for s in result.sentences] == [
        "Hello!",
        "I'm Charles.",
        "Tell me about your day — in English.",
    ]
    assert result.sentences[0].parts[0].role == "phrase without a verb"


@needs_spacy
@pytest.mark.integration
def test_real_pipeline_breaks_down_a_wrong_sentence_without_blinking(real):
    """Why the note exists, measured: the mistake is in the verb form, and the
    structure is fine — so the breakdown is as tidy as for a correct sentence.
    """
    (sentence,) = real.parse("I didn't went to the party.").sentences
    assert [(part.text, part.role) for part in sentence.parts] == [
        ("I", "subject"),
        ("didn't went", "verb"),
        ("to the party", "prepositional phrase"),
    ]


@needs_spacy
@pytest.mark.integration
def test_real_pipeline_shows_no_label_from_outside_the_tables(real):
    """Across a spread of constructions, nothing falls through to "other (…)"."""
    texts = [
        "The man who lives next door has been painting his house since Monday.",
        "If it rains tomorrow, we will stay home and watch a film.",
        "She gave me a book yesterday.",
        "The window was broken by the kids.",
        "Please turn the light off.",
        "I want to learn English, but it is hard.",
        "There are many people in the street.",
    ]
    for text in texts:
        for sentence in real.parse(text).sentences:
            for part in sentence.parts:
                assert not part.role.startswith("other ("), (text, part)
                for word in part.words:
                    assert not word.pos.startswith("other ("), (text, word)
                    assert not word.role.startswith("other ("), (text, word)
