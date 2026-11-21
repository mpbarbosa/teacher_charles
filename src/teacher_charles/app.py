"""Aplicação FastAPI.

E2: a conversa por texto funciona de ponta a ponta no navegador. A resposta sai
em streaming porque, medido nesta máquina (7-8 tok/s num 4B), é isso que faz a
espera ser tolerável — o aluno começa a ler no primeiro segundo em vez de
encarar uma tela parada por dez.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import signal
import sqlite3
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from . import __version__
from .build_stamp import read_build_stamp
from .config import PACKAGE_ROOT, Settings, get_settings
from .health import probe_model
from .llm import ChatProvider, build_provider, default_options
from .llm.base import GenerationOptions, Message
from .prompts import PromptLibrary
from .store.db import init_db
from .store.models import CEFR_LEVELS, MODES, Student
from .store.repo import (
    CorrectionRepo,
    LevelRepo,
    ReadAloudRepo,
    ScheduleRepo,
    SessionRepo,
    StudentRepo,
    TurnRepo,
)
from .teaching import (
    Analyzed,
    Analyzer,
    Chunk,
    Done,
    Failed,
    Leveled,
    Phonemized,
    TurnRunner,
    read_aloud,
)
from .teaching.contrastive import ContrastiveError, load_base
from .teaching.dictionary import MAX_WORD_CHARS, Dictionary, DictionaryUnavailable, build_dictionary
from .teaching.dictionary import on_screen as dictionary_on_screen
from .teaching.ipa_symbols import SymbolError, SymbolTable, load_symbols
from .teaching.lessons import (
    ASPECTS,
    Lesson,
    LessonError,
    LessonShelf,
    LocalText,
    load_shelf,
    place_cites,
    read_local_cues,
    read_local_text,
    rehearsal_plan,
)
from .teaching.level import LevelTracker
from .teaching.level import partial_name as level_partial_name
from .teaching.modes import DEFAULT_MODE, DrillPlanner
from .teaching.modes import partial_name as mode_partial_name
from .teaching.phonetics import Phonetics, build_phonetics, ipa_owners, ipa_pairs
from .teaching.pronunciation import (
    Evaluator,
    Judgement,
    PronunciationError,
    PronunciationItem,
    PronunciationUnavailable,
)
from .teaching.pronunciation import load_base as load_pronunciation
from .teaching.schedule import Scheduler, is_graduated
from .teaching.syntax import Syntax, SyntaxUnavailable, build_syntax
from .teaching.syntax import on_screen as syntax_on_screen
from .teaching.turn import DEFAULT_LEVEL
from .voice import (
    PhonemeRecognizer,
    SentenceAssembler,
    Voice,
    VoiceError,
    build_phonemes,
    build_voice,
    split_sentences,
)
from .voice.align import word_starts
from .voice.base import STT_SAMPLE_RATE, Speech
from .voice.follow import last_heard
from .voice.kokoro import MAX_SPEED as KOKORO_MAX_SPEED
from .voice.kokoro import MIN_SPEED as KOKORO_MIN_SPEED
from .voice.signal import SignalVerdict, samples_from_pcm
from .voice.signal import diagnose as diagnose_signal

log = logging.getLogger(__name__)

WEB_DIR = PACKAGE_ROOT / "web"

# O perfil que existe antes de o aluno dizer o nome dele. A partir da E5 há
# vários perfis numa máquina — o ativo é o visto por último —, mas não há login:
# multiusuário fica fora da Fase 1 (plano §4).
DEFAULT_STUDENT = "student"

# A primeira coisa que o aluno lê na tela, antes de existir conversa nenhuma.
#
# Mora aqui, e não solta no template, desde que a linha de IPA passou a correr
# embaixo de toda fala: o template não tem como chamar o G2P, e uma saudação
# sem transcrição seria o único texto inglês da página sem ela — exatamente a
# exceção que a funcionalidade existe para não ter.
GREETING = "Hello! I'm Charles. Tell me about your day — in English."


def request_stop() -> None:
    """Pede ao processo que pare, com o mesmo sinal do Ctrl-C.

    Por que um sinal e não ``os._exit``: o desligamento precisa passar pelo
    ``finally`` do lifespan, que é onde o provider e a voz fecham e onde a
    conexão do SQLite é fechada. Um ``_exit`` pularia tudo isso e deixaria o
    banco do aluno para o sistema operacional fechar.

    É o mesmo caminho do Ctrl-C no terminal: o uvicorn trata o SIGINT, marca
    ``should_exit`` e desce sozinho depois de terminar o que está em curso —
    inclusive a resposta desta rota, que a essa altura já saiu.

    Com ``--reload`` o sinal chega só ao processo que serve, e o supervisor
    do reload fica vivo sem filho. Fica assim de propósito: ``--reload`` é
    modo de desenvolvimento, e lá o terminal está à mão.
    """
    signal.raise_signal(signal.SIGINT)


def from_this_page(request: Request) -> bool:
    """Diz se o pedido de desligar veio da própria página do Charles.

    O servidor escuta em localhost sem autenticação nenhuma, e POST entre
    sites é coisa que qualquer aba aberta no navegador consegue fazer: sem
    este teste, uma página qualquer poderia desligar o Charles do aluno no
    meio da aula. O navegador manda ``Origin`` em todo POST, então comparar
    com o ``Host`` basta — e sai bem mais barato que uma camada de sessão
    para proteger um botão.

    Pedido sem ``Origin`` não veio de navegador (curl, script, o
    ``TestClient``), e esse continua podendo desligar: um
    ``curl -X POST .../api/shutdown`` na própria máquina é exatamente o que
    se espera que funcione.
    """
    origin = request.headers.get("origin")
    if origin is None:
        return True
    return urlsplit(origin).netloc == request.headers.get("host")


async def _warm_up(provider: ChatProvider, prefix: list[Message]) -> None:
    """Carrega o modelo **e** põe o prefixo estável do turno no cache.

    O prompt tem que ser o mesmo de um turno real. Medido na E2: aquecer com um
    "hi" solto carrega o modelo mas deixa os 427 tokens da persona fora do
    cache, e o primeiro turno do aluno paga 19s só de avaliação de prompt.

    Medido de novo na E7, e dessa vez o errado era este arquivo: a E5 e a E6
    acrescentaram ajuste de nível e prompt de modo ao turno, e o aquecimento
    seguiu mandando a persona sozinha. Resultado — primeiro turno com 53% do
    prompt em cache e 11,6s até a primeira palavra. Por isso o prefixo agora
    vem de ``TurnRunner.stable_prefix``: há uma lista só, e ela não tem como
    divergir do que o turno manda (plano §5.5).

    Nunca propaga erro: se o modelo não existe ou o Ollama está fora do ar, o
    /health já diz isso, e derrubar a subida por causa do aquecimento seria
    pior do que um primeiro turno lento.
    """
    messages = [*prefix, Message(role="user", content="Hello.")]

    try:
        await provider.complete(
            messages,
            GenerationOptions(temperature=0.0, max_tokens=1, think=False),
        )
        log.info("modelo %s aquecido, persona em cache", provider.model)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - aquecimento nunca derruba a subida
        log.warning("aquecimento do modelo falhou (segue assim mesmo): %s", exc)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class ProfileRequest(BaseModel):
    """O que o aluno pode mudar no perfil dele.

    Os dois campos são opcionais: trocar de perfil e corrigir o nível são
    pedidos diferentes, e a UI faz um por vez.
    """

    name: str | None = Field(default=None, min_length=1, max_length=60)
    level: str | None = None


class ModeRequest(BaseModel):
    mode: str


class SpeakRequest(BaseModel):
    """Um trecho para falar.

    O teto de 600 caracteres não é arbitrário: é o mesmo do ``KokoroTTS``, e
    existe porque uma chamada por sentença é o desenho (ver
    ``voice/segment.py``). Pedido maior que isso é sinal de que alguém está
    mandando o parágrafo inteiro — o que anularia o TTS incremental.
    """

    text: str = Field(min_length=1, max_length=600)
    voice: str | None = None
    # Keep the audio so its word timings can be asked for next, at
    # ``/api/tts/{id}/words``: the lesson reader lights the word being said.
    # Off by default -- the chat's voice lights nothing, and holds nothing.
    timings: bool = False
    # Kokoro's own speaking rate for this sentence, 1.0 its natural pace;
    # absent is ``TC_TTS_SPEED``. The lesson page's "Voice speed" slider. The
    # bounds are Kokoro's (``voice/kokoro.py``), held here so a bad value is a
    # 422 before it costs a synthesis.
    speed: float | None = Field(default=None, ge=KOKORO_MIN_SPEED, le=KOKORO_MAX_SPEED)


# How many synthesised sentences wait for their word timings to be asked for.
# The reader asks for each right after its audio arrives and reads a few
# sentences ahead, so a handful is the working set; the cap is what keeps a
# page that never asks from growing the server. ~8 s of 24 kHz PCM is ~380 KB.
TIMED_SPEECH_KEPT = 16


# The longest stretch of the microphone ``/api/follow`` takes, in seconds. The
# page sends the last ~3.5 s; the ceiling is what keeps one request from
# holding the CPU, and it is in base64 because the audio rides in JSON beside
# the words.
FOLLOW_MAX_SECONDS = 8
FOLLOW_MAX_AUDIO_CHARS = (FOLLOW_MAX_SECONDS * STT_SAMPLE_RATE * 2 * 4) // 3 + 4


# A read-aloud take: one curated phrase, a breath long. Ten seconds is the
# window with room; the recognizer's own ceiling is thirty.
PRONUNCIATION_MAX_SECONDS = 10
PRONUNCIATION_MAX_AUDIO_CHARS = (PRONUNCIATION_MAX_SECONDS * STT_SAMPLE_RATE * 2 * 4) // 3 + 4


class PronunciationRequest(BaseModel):
    """One take of one curated phrase, for a verdict (E11).

    ``audio`` is 16 kHz mono 16-bit PCM in base64, as ``/api/follow`` takes it.
    The phrase is named by rule and index, never sent as text: the expected
    text is the curated one, and a page that could send its own would let any
    sentence be judged against a rule written for another.
    """

    rule_id: str = Field(min_length=1, max_length=80)
    phrase: int = Field(default=0, ge=0)
    audio: str = Field(min_length=1, max_length=PRONUNCIATION_MAX_AUDIO_CHARS)


class ReadAloudStart(BaseModel):
    """Open a read-aloud session (E12). No ``rule_id`` lets the schedule pick."""

    rule_id: str | None = Field(default=None, min_length=1, max_length=80)


class ReadAloudTake(BaseModel):
    """One take of the session's phrase. The phrase is the session's, not the page's."""

    session_id: int = Field(ge=1)
    # Metodo §2.4's take said wrong on purpose: once per rule, before the drill.
    deliberate: bool = False
    audio: str = Field(min_length=1, max_length=PRONUNCIATION_MAX_AUDIO_CHARS)


class ReadAloudFinish(BaseModel):
    session_id: int = Field(ge=1)


class FollowRequest(BaseModel):
    """The last seconds of a student reading aloud, and the words around them.

    ``audio`` is 16 kHz mono PCM, 16-bit little-endian, in base64: the page
    captures and downsamples it itself, because a ``MediaRecorder`` slice past
    the first has no container header and nothing can decode it alone.
    ``words`` are the text's words the page wants placed, as it shows them:
    a window around the light, or the whole text while the light is lost.
    The text is the page's, not looked up here, so the route needs no lesson.
    """

    audio: str = Field(min_length=1, max_length=FOLLOW_MAX_AUDIO_CHARS)
    words: list[str] = Field(min_length=1, max_length=2000)


class SyntaxRequest(BaseModel):
    """A message to break down.

    Same ceiling as ``ChatRequest``: what gets analysed is a bubble, and a
    bubble is a chat message.
    """

    text: str = Field(min_length=1, max_length=4000)


# How long a text the phonetics page takes. It is a box for a sentence or a
# short paragraph, not a document: the IPA comes back sentence by sentence,
# and each sentence's words are fitted to its IPA by a small dynamic
# programme (``ipa_owners``) whose cost grows with the square of its length.
PHONETICS_MAX_CHARS = 1000

# Past this many words in one sentence the page shows its IPA line alone, with
# no word-by-word pairing: a run-on with no full stop would otherwise cost
# seconds in ``ipa_owners`` for a table nobody reads at that length.
PHONETICS_PAIRED_WORDS = 80


class PhoneticsRequest(BaseModel):
    """A sentence (or a few) to spell out in IPA."""

    text: str = Field(min_length=1, max_length=PHONETICS_MAX_CHARS)


class DefineRequest(BaseModel):
    """One word of the page, as the page cut it (punctuation and all)."""

    word: str = Field(min_length=1, max_length=MAX_WORD_CHARS)


class _RevalidatedStatic(StaticFiles):
    """Static files the browser must check before reusing.

    Starlette sends an ETag but no Cache-Control, so the browser falls back to
    heuristic freshness and keeps an old style.css after a deploy: the lesson
    page then rendered with the browser's default link blue on the dark theme.
    `no-cache` still lets it reuse its copy, via a 304 that costs nothing
    on localhost.
    """

    async def get_response(self, path: str, scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


def create_app(settings: Settings | None = None, provider: ChatProvider | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Migrar na subida: um dev que acabou de clonar não deveria precisar
        # rodar nada à mão antes do primeiro `uv run teacher-charles`.
        conn = init_db(settings.resolved_db_path)
        # Construir o provider aqui faz uma configuração inválida estourar na
        # subida, e não no meio do primeiro turno do aluno.
        chat = provider if provider is not None else build_provider(settings)

        prompts = PromptLibrary(settings.prompts_dir)
        # Falha cedo se a persona não existe: sem ela o Charles não é ninguém.
        prompts.get("persona")
        # Idem para a instrução da análise: sem ela não há correção estruturada.
        prompts.get("analysis")

        # A base contrastiva é carregada uma vez na subida. Se ela estiver
        # quebrada, o /health reporta e a conversa segue sem regra curada — o
        # que é pior que o normal, mas melhor que não subir.
        try:
            base = load_base(settings.data_dir / "contrastive")
            log.info("base contrastiva: %d itens", len(base))
            analyzer = Analyzer(
                provider=chat,
                prompts=prompts,
                base=base,
                require_curated_rule=settings.require_curated_rule,
            )
            # O drill treina regra da base: sem base não há o que treinar, e o
            # modo diz isso ao aluno em vez de inventar exercício.
            planner = DrillPlanner(conn, base)
        except ContrastiveError as exc:
            log.error("base contrastiva não carregou, turno segue sem análise: %s", exc)
            analyzer = None
            planner = None

        # ``None`` quando a voz está desligada, e isso é estado de primeira
        # classe: o chat por texto da Fase 1 funciona inteiro sem voz.
        #
        # Construir não carrega peso nenhum — a carga é preguiçosa e acontece
        # no aquecimento, que engole o erro. Então pedir voz com o peso fora
        # do disco **não** derruba a subida; quem denuncia é o ``/health``,
        # que fica degradado e diz qual motor reclamou do quê.
        voice = build_voice(settings)
        if voice is not None:
            log.info("voz ligada: STT %s, TTS %s", voice.stt.model, voice.tts.model)

        # ``None`` quando o IPA está desligado — mesmo estado de primeira
        # classe da voz. Construir também não carrega nada: o espeak sobe no
        # aquecimento, e um espeak que não sobe degrada o ``/health`` em vez de
        # derrubar a aula.
        phonetics = build_phonetics(settings)

        # The phoneme ear of E11, ``None`` with pronunciation off (the
        # default). Building loads nothing; the 318 MB arrive in the warm-up.
        phonemes = build_phonemes(settings)
        pronunciation: Evaluator | None = None
        pronunciation_error: str | None = None
        if phonemes is not None:
            # Curated content, so the shelf's policy: a broken base is reported
            # by /health and the route answers 503, the lesson goes on. The
            # expected text comes from its own espeak with no stress marks,
            # which is how the recognizer writes and how E9's reports wrote it.
            try:
                pronunciation = Evaluator(
                    load_pronunciation(settings.data_dir / "contrastive" / "pronunciation.yaml"),
                    Phonetics(with_stress=False).transcribe,
                )
            except PronunciationError as exc:
                pronunciation_error = str(exc)
                log.error("the pronunciation base did not load: %s", exc)

        # ``None`` with the syntax view off, which is the default. Building
        # loads nothing; whether the extra is even installed is asked of the
        # import system, not found out by importing 3 s of spaCy here.
        syntax = build_syntax(settings)

        # ``None`` only when switched off. Building opens nothing, and the
        # lookup file being absent is the ordinary state of a fresh clone: the
        # words stay plain text and /health says how to fetch it.
        dictionary = build_dictionary(settings)

        # Lessons are curated content like the two bases, and the same policy
        # holds: a broken shelf is reported by /health and leaves the page
        # empty, it does not bring the conversation down.
        try:
            lessons: LessonShelf | None = load_shelf(settings.data_dir / "lessons")
            log.info("lessons: %d", len(lessons))
        except LessonError as exc:
            log.error("lessons did not load, the shelf stays empty: %s", exc)
            lessons = None

        # The IPA symbols, each with a word to hear it in: the chart under a
        # lesson's full text. Curated like the shelf, and the same policy: a
        # broken table leaves the chart out, not the lesson.
        try:
            symbols_path = settings.data_dir / "phonetics" / "symbols.yaml"
            symbols: SymbolTable | None = load_symbols(symbols_path)
        except SymbolError as exc:
            log.error("IPA symbols did not load, lessons go without the chart: %s", exc)
            symbols = None

        app.state.settings = settings
        app.state.db = conn
        # Em app.state, e não fechado na rota: é aqui que o teste troca o
        # sinal por um registro de que o desligamento foi pedido.
        app.state.stop = request_stop
        app.state.provider = chat
        app.state.voice = voice
        # Synthesised sentences awaiting their word timings, oldest first;
        # see ``/api/tts/{speech_id}/words``.
        app.state.timed_speech = OrderedDict()
        app.state.phonetics = phonetics
        app.state.phonemes = phonemes
        app.state.pronunciation = pronunciation
        app.state.pronunciation_error = pronunciation_error
        app.state.syntax = syntax
        app.state.dictionary = dictionary
        app.state.lessons = lessons
        app.state.ipa_symbols = symbols
        app.state.prompts = prompts
        app.state.analyzer = analyzer
        app.state.base = base if analyzer is not None else None
        app.state.tracker = LevelTracker(conn)
        app.state.planner = planner
        app.state.runner = TurnRunner(
            provider=chat,
            prompts=prompts,
            conn=conn,
            options=default_options(settings),
            analyzer=analyzer,
            analysis_after_reply=settings.analysis_after_reply,
            tracker=app.state.tracker,
            planner=planner,
            # O professor tem que saber se existe botão de falar na tela do
            # aluno. Vai o flag, não o objeto ``voice``: `teaching/` não
            # importa `voice/` (ver ``teaching/capabilities.py``), e o que o
            # prompt descreve é o produto, não o motor carregado.
            voice_enabled=settings.voice_enabled,
            # O G2P vai inteiro, e não um flag como a voz: aqui `teaching/` é
            # o dono do módulo — IPA é texto virando texto, não áudio (ver
            # ``teaching/phonetics.py``).
            phonetics=phonetics,
            # Whether the Structure button is on screen — asked for *and*
            # installed. The same function the page uses, so the teacher
            # never points at a button the page did not draw.
            syntax_enabled=syntax_on_screen(syntax),
        )

        # Carregar o modelo do disco custa ~15s nesta máquina, e medido na E2
        # essa espera caía inteira no primeiro turno do aluno (16,3s até a
        # primeira palavra, contra 1,1s nos turnos seguintes). Aquecer em
        # segundo plano move isso para enquanto a página ainda está sendo lida.
        # Aquece com o nível e o modo do perfil ativo: é o prefixo que o
        # primeiro turno vai mandar de verdade.
        active = StudentRepo(conn).active()
        warm_level = active.cefr_level if active is not None else DEFAULT_LEVEL
        warm_session = SessionRepo(conn).current(active.id) if active is not None else None
        warm_mode = warm_session.mode if warm_session is not None else DEFAULT_MODE
        warmup = (
            asyncio.create_task(
                _warm_up(chat, app.state.runner.stable_prefix(warm_level, warm_mode))
            )
            if settings.llm_warmup
            else None
        )
        app.state.warmup = warmup

        # A voz aquece em paralelo com o LLM, e por motivo diferente: o
        # ``small.en`` chega a baixar na primeira vez, e o onnxruntime paga o
        # primeiro traçado do grafo. Nenhum dos dois propaga erro — mesma
        # política do aquecimento do modelo (ver ``base.SpeechToText.warm_up``).
        app.state.voice_warmup = asyncio.create_task(voice.warm_up()) if voice is not None else None

        # O espeak custa ~0,25s para subir, e esse quarto de segundo cairia
        # inteiro no primeiro turno do aluno. Mesma política dos outros dois
        # aquecimentos: em segundo plano, e sem propagar erro.
        app.state.phonetics_warmup = (
            asyncio.create_task(phonetics.warm_up()) if phonetics is not None else None
        )

        # Importing spaCy and loading the pipeline cost ~3.5 s, which would
        # otherwise land on the first press of the button. Only when the extra
        # is there: warming up a parser that is not installed would just log
        # the failure `/health` already reports.
        app.state.syntax_warmup = (
            asyncio.create_task(syntax.warm_up()) if syntax_on_screen(syntax) else None
        )

        # 318 MB on the first start, then a ~2 s load. In the background and
        # without propagating, like every other warm-up here.
        app.state.phonemes_warmup = (
            asyncio.create_task(phonemes.warm_up()) if phonemes is not None else None
        )

        try:
            yield
        finally:
            for task in (
                warmup,
                app.state.voice_warmup,
                app.state.phonetics_warmup,
                app.state.syntax_warmup,
                app.state.phonemes_warmup,
            ):
                if task is not None and not task.done():
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
            await chat.aclose()
            if voice is not None:
                await voice.aclose()
            if phonemes is not None:
                await phonemes.aclose()
            conn.close()

    app = FastAPI(
        title="Teacher Charles",
        version=__version__,
        summary="Um professor de inglês que roda localmente",
        lifespan=lifespan,
    )

    app.mount("/static", _RevalidatedStatic(directory=WEB_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=WEB_DIR / "templates")
    # Every page's last line names the commit (build_stamp.py). Read here, once:
    # the footer names the code this process loaded, not whatever HEAD is now.
    build = read_build_stamp(PACKAGE_ROOT)
    templates.env.globals["build"] = build

    # --- helpers ---------------------------------------------------------- #

    def active_student(conn: sqlite3.Connection) -> Student:
        """O perfil em uso: o visto por último, ou o default na primeira vez."""
        students = StudentRepo(conn)
        return students.active() or students.get_or_create(DEFAULT_STUDENT)

    def active_session(conn: sqlite3.Connection) -> int:
        return SessionRepo(conn).current_or_start(active_student(conn).id)

    def session_payload(conn: sqlite3.Connection) -> dict[str, Any]:
        """Em que modo o aluno está, e o que o drill está treinando.

        O alvo do drill vai junto porque o aluno tem direito de saber **qual**
        erro dele virou exercício: "drilling: age with to be, you did it 3
        times" é uma afirmação verificável; "vamos praticar gramática" não é.
        """
        student = active_student(conn)
        session = SessionRepo(conn).current(student.id)
        mode = session.mode if session is not None else DEFAULT_MODE
        planner: DrillPlanner | None = app.state.planner

        drill: dict[str, Any] | None = None
        if session is not None and session.drill_rule_id and planner is not None:
            target = planner.target_for(student.id, session.drill_rule_id)
            drill = target.as_dict() if target is not None else None

        return {
            "session_id": session.id if session is not None else None,
            "mode": mode,
            "modes": list(MODES),
            "drill": drill,
            # Quantas regras do error log dariam drill agora. Zero é resposta
            # honesta: o aluno ainda não errou nada com regra curada.
            "drillable": len(planner.candidates(student.id)) if planner is not None else 0,
            # E quantas delas a agenda liberou (E13). As duas contagens são
            # diferentes de propósito: `drillable` é o que existe, `due` é o
            # que está vencido. `drillable` alto com `due` zero é o aluno em
            # dia — e é a única maneira de a tela distinguir isso de "não há o
            # que treinar", que é `drillable` zero.
            "due": planner.due_count(student.id) if planner is not None else 0,
        }

    def progress_payload(conn: sqlite3.Connection, limit: int = 20) -> dict[str, Any]:
        """O error log do aluno, agregado por regra.

        É a E7 pagando o que a E4 prometeu: a correção virou linha contável, e
        aqui as linhas viram a resposta de "no que eu mais erro?". A explicação
        de cada regra vem da base curada quando ela ainda existe — assim a tela
        mostra a redação revisada, e não o que o modelo disse naquele turno.

        Correções sem regra entram só na contagem do rodapé. Elas não são
        agregáveis (o modelo escreve a explicação que quiser, cada vez
        diferente) e, pela política da §5.2, nem chegam ao aluno — o número
        existe para medir quanto a base ainda não cobre.
        """
        student = active_student(conn)
        corrections = CorrectionRepo(conn)
        base = app.state.base
        # A agenda da E13, numa consulta só para todas as regras da tela.
        schedules = ScheduleRepo(conn).for_student(student.id)

        rules = []
        for count in corrections.top_rules(student.id, limit=limit):
            item = base.get(count.rule_id) if base is not None and count.rule_id in base else None
            schedule = schedules.get(count.rule_id)
            rules.append(
                {
                    "rule_id": count.rule_id,
                    "times": count.times,
                    "last_at": count.last_at,
                    "category": item.category if item is not None else "other",
                    "explanation": item.explanation if item is not None else "",
                    "right": list(item.right[:2]) if item is not None else [],
                    "examples": corrections.spans_for_rule(student.id, count.rule_id, limit=3),
                    # Regra que saiu da base continua no histórico do aluno; a
                    # tela diz isso em vez de fingir que a regra ainda existe.
                    "retired": item is None,
                    # A agenda (E13). O error log sozinho só sabe somar erros, e
                    # com isso a tela não conseguia dizer a única coisa que o
                    # aluno quer saber: se ele está melhorando nesta regra.
                    "streak": schedule.streak if schedule is not None else 0,
                    "graduated": is_graduated(schedule),
                    "due_at": schedule.due_at if schedule is not None else "",
                    "reviews": schedule.reviews if schedule is not None else 0,
                    "lapses": schedule.lapses if schedule is not None else 0,
                }
            )

        totals = corrections.progress(student.id)
        levels = LevelRepo(conn).changes(student.id, limit=10)

        # Read aloud (E12), by rule: in takes and sessions, never as a rate.
        # Its own list and not merged into `rules`: those are the student's
        # written sentences, and a pronunciation rule is never in the text
        # base (the E10 invariant), so the two never share a row.
        evaluator: Evaluator | None = app.state.pronunciation
        read_aloud_rules = []
        for count in ReadAloudRepo(conn).by_rule(student.id):
            item = (
                evaluator.base.get(count.rule_id)
                if evaluator is not None and count.rule_id in evaluator.base
                else None
            )
            schedule = schedules.get(count.rule_id)
            read_aloud_rules.append(
                {
                    "rule_id": count.rule_id,
                    "looks_for": read_aloud.looked_for(item) if item is not None else "",
                    "right": item.right if item is not None else "",
                    "sessions": count.sessions,
                    "takes": count.takes,
                    "named": count.named,
                    "last_outcome": count.last_outcome,
                    "last_at": count.last_at,
                    "retired": item is None,
                    "streak": schedule.streak if schedule is not None else 0,
                    "graduated": is_graduated(schedule),
                    "due_at": schedule.due_at if schedule is not None else "",
                }
            )
        return {
            "student": student.name,
            "level": student.cefr_level,
            "level_source": student.level_source,
            "self_declared_level": student.self_declared_level,
            "totals": totals,
            "rules": rules,
            "read_aloud": read_aloud_rules,
            "level_history": [
                {
                    "from": change.from_level,
                    "to": change.to_level,
                    "source": change.source,
                    "reason": change.reason,
                    "at": change.created_at,
                }
                for change in levels
            ],
        }

    def voice_payload() -> dict[str, Any]:
        """O que a página precisa saber antes de mostrar microfone e alto-falante.

        Existe como função, e não inline nos dois lugares, porque a página e o
        ``/api/voice`` têm que concordar — duplicar a montagem é o jeito de
        eles divergirem (plano §5.5).
        """
        voice: Voice | None = app.state.voice
        if voice is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "stt_model": voice.stt.model,
            "tts_model": voice.tts.model,
            "voice": voice.tts.voice,
            # Vazio até o TTS carregar: `voices()` não força a carga, de
            # propósito (ver ``voice/kokoro.py``).
            "voices": list(voice.tts.voices()),
            "ready": voice.stt.ready and voice.tts.ready,
            # Where the lesson page's "Voice speed" slider starts before the
            # student moves it: the rate a request without `speed` gets.
            "speed": settings.tts_speed,
        }

    def spell(text: str) -> str:
        """O IPA deste texto para a página, ou string vazia.

        Vazia cobre os dois casos em que não há linha para desenhar — IPA
        desligado e espeak que não carregou — e o template trata os dois do
        mesmo jeito, que é não desenhando nada.
        """
        phonetics: Phonetics | None = app.state.phonetics
        return phonetics.transcribe(text) if phonetics is not None else ""

    def phonetics_payload() -> dict[str, Any]:
        """O que a página precisa saber antes de mostrar a linha de IPA.

        Existe como função pela mesma razão do ``voice_payload``: a página e o
        ``/health`` têm que concordar, e montar o dicionário duas vezes é o
        jeito de eles divergirem.
        """
        phonetics: Phonetics | None = app.state.phonetics
        if phonetics is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "language": phonetics.language,
            "stress": phonetics.with_stress,
            # Falso até a primeira transcrição ou o fim do aquecimento: a
            # carga é preguiçosa de propósito, e `ready` nunca a dispara.
            "ready": phonetics.ready,
        }

    def syntax_payload() -> dict[str, Any]:
        """What the page needs to know before drawing the Structure button.

        ``enabled`` here means *on screen*: asked for and installed. Asked for
        without the extra is ``False`` — `/health` carries the reason, and the
        student is not handed a button that answers 503 every time.
        """
        syntax: Syntax | None = app.state.syntax
        if syntax is None or not syntax_on_screen(syntax):
            return {"enabled": False}
        return {"enabled": True, "model": syntax.model, "ready": syntax.ready}

    def lesson_card(lesson: Lesson) -> dict[str, Any]:
        """What the shelf shows of a lesson before it is opened."""
        return {
            "slug": lesson.slug,
            "title": lesson.title,
            "summary": lesson.summary,
            "cefr": lesson.cefr,
            # Whether the lesson says why it is at that level (LevelReading):
            # the shelf links the badge to that section only when it exists.
            "level_explained": lesson.level is not None,
            "source": lesson.source.model_dump(mode="json"),
        }

    def lesson_payload(lesson: Lesson) -> dict[str, Any]:
        """A whole lesson for the page and the API, with the IPA filled in.

        The IPA is the one thing not in the YAML: it is computed here by the
        same espeak that draws the line under every chat bubble, so the lesson
        and the conversation can never disagree about a word.
        """
        body = lesson.model_dump(mode="json", exclude={"notes", "sections"})
        body["slug"] = lesson.slug
        body["sections"] = [s.model_dump(mode="json") for s in lesson.ordered_sections()]
        for note in body["words"] + body["sounds"]:
            note["ipa"] = spell(note["word"])
        return body

    def full_text_sentences(full_text: LocalText | None) -> list[list[dict[str, Any]]]:
        """Each paragraph of the copy as sentences, words, and their IPA.

        ## Why the sentences are cut here, once, and not in the page

        The reader speaks one sentence per ``/api/tts`` call and lights the
        sentence it is on, in the text and in the IPA line under it. That only
        lines up if the three agree on where a sentence ends, so there is one
        cut and it is ``voice/segment.py``'s — the one the chat's voice already
        uses, which keeps "Mr. Smith" whole and lets "Yes!" ride with the next
        sentence. It joins back to the paragraph exactly, so the page shows
        the text as it is.

        Within a sentence the reader lights the word being said, by timings
        ``/api/tts`` measures (``voice/align.py``). The words are the
        sentence split on whitespace, the split the route uses too, so word
        *k* on the page is word *k* in the timings.

        The IPA is espeak's reading of the whole sentence, which is what the
        voice says, cut into its own words; ``ipa_of`` says which IPA words
        each written word is part of, because espeak runs function words
        together ("on the" is one IPA word, ``ɔnðə``). Empty with the IPA off.
        """
        if full_text is None:
            return []
        return [[sentence_view(s) for s in split_sentences(p) or [p]] for p in full_text.paragraphs]

    def point_passages(lesson: Lesson, full_text: LocalText | None) -> list[list[dict[str, Any]]]:
        """For each point, in the page's order, the sentences of the copy it cites.

        Aligned with ``lesson_payload``'s sections and their points, so the
        template walks both with the same indices. The sentences are cut by
        the same ``split_sentences`` the full text above uses, so a passage
        beside a point is one sentence of that text, never a stretch of two.
        With no copy every citation is unplaced and the point shows its bare
        quotes, as a clone without the copy always did.
        """
        sentences = (
            [s for p in full_text.paragraphs for s in split_sentences(p) or [p]]
            if full_text is not None
            else []
        )
        out = []
        for section in lesson.ordered_sections():
            row = []
            for point in section.points:
                placed = place_cites(sentences, point.cites)
                row.append(
                    {
                        "passages": [
                            [{"text": t, "cite": c} for t, c in runs] for runs in placed.passages
                        ],
                        "unplaced": list(placed.unplaced),
                    }
                )
            out.append(row)
        return out

    def rehearsal_view(
        lesson: Lesson, sentences: list[list[dict[str, Any]]]
    ) -> dict[str, Any] | None:
        """The copy as blocks to rehearse aloud, or ``None`` without a copy.

        Cut from the sentences the full text already shows, so a block is a
        paragraph of that text and its word count is the one the page reads.
        The cue words are the copy's (``read_local_cues``), so this is here
        with the copy and never in ``lesson_payload``.
        """
        if not sentences:
            return None
        plan = rehearsal_plan(
            lesson,
            [[s["text"] for s in paragraph] for paragraph in sentences],
            read_local_cues(settings.resolved_corpus_dir, lesson),
        )
        return {
            "blocks": [
                {
                    "paragraph": b.paragraph,
                    "title": b.title,
                    "target": b.target,
                    "words": b.words,
                    "cues": list(b.cues),
                    "own_cues": b.own_cues,
                    # The sentence the block's "Hear it" starts the reader on.
                    "first": f"{b.paragraph}.0",
                }
                for b in plan.blocks
            ],
            "planned": plan.planned,
            "plan_set_aside": plan.plan_set_aside,
            "planned_blocks": len(lesson.rehearsal or []),
            "cues_set_aside": plan.cues_set_aside,
            "timed": len(plan.timed),
            "target": plan.target,
        }

    def symbol_chart(sentences: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """The IPA symbols the full text uses, grouped, each with its example.

        Read off the IPA the page already shows, so a tile is never a sound
        the text does not have. The example's IPA is espeak's, made here, for
        ``teaching/ipa_symbols.py``'s reason. Empty with the IPA off or the
        table missing, and then the template draws no chart.
        """
        table: SymbolTable | None = app.state.ipa_symbols
        if table is None:
            return []
        groups = table.chart([s["ipa"] for p in sentences for s in p if s["ipa"]])
        for group in groups:
            for row in group["symbols"]:
                row["example_ipa"] = spell(row["example"])
        return groups

    def sentence_view(text: str) -> dict[str, Any]:
        words = text.split()
        ipa = spell(text)
        ipa_words = ipa.split()
        owners = ipa_owners(words, ipa_words, [spell(w) for w in words]) if ipa_words else []
        return {
            "text": text,
            "words": words,
            "ipa": ipa,
            "ipa_words": ipa_words,
            "ipa_of": [" ".join(str(j) for j in o) for o in owners],
        }

    def spelled_out(text: str) -> list[list[dict[str, Any]]]:
        """Each line of the text as sentences, each with its IPA and its pairs.

        The cut is the lesson reader's (``split_sentences``), for its reason:
        espeak reads a sentence the way the voice says it, and the pairing of
        written words to IPA words runs per sentence, where it is cheap.
        """
        view: list[list[dict[str, Any]]] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            sentences = []
            for sentence in split_sentences(line) or [line]:
                words = sentence.split()
                ipa = spell(sentence)
                tokens = ipa.split()
                pairs: list[dict[str, str]] = []
                if tokens and len(words) <= PHONETICS_PAIRED_WORDS:
                    owners = ipa_owners(words, tokens, [spell(w) for w in words])
                    pairs = ipa_pairs(words, tokens, owners)
                sentences.append({"text": sentence, "ipa": ipa, "pairs": pairs})
            view.append(sentences)
        return view

    def shelf() -> LessonShelf | None:
        return app.state.lessons

    def profile_payload(conn: sqlite3.Connection) -> dict[str, Any]:
        student = active_student(conn)
        changes = LevelRepo(conn).changes(student.id, limit=10)
        return {
            "name": student.name,
            "level": student.cefr_level,
            "self_declared_level": student.self_declared_level,
            "level_source": student.level_source,
            "levels": list(CEFR_LEVELS),
            "history": [
                {
                    "from": c.from_level,
                    "to": c.to_level,
                    "source": c.source,
                    "reason": c.reason,
                    "at": c.created_at,
                }
                for c in changes
            ],
        }

    # --- páginas ---------------------------------------------------------- #

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        conn: sqlite3.Connection = app.state.db
        session_id = active_session(conn)
        history = TurnRepo(conn).history(session_id, limit=50)

        # As correções de turnos passados voltam com o histórico: o que o aluno
        # errou ontem continua visível quando ele reabre a página.
        by_turn: dict[int, list[Any]] = {}
        for c in CorrectionRepo(conn).for_session(session_id):
            by_turn.setdefault(c.turn_id, []).append(c)

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "version": __version__,
                "model": settings.llm_model,
                "profile": profile_payload(conn),
                "session": session_payload(conn),
                "voice": voice_payload(),
                "phonetics": phonetics_payload(),
                "syntax": syntax_payload(),
                # The word card on the messages, as on a lesson's text: only
                # with the lookup file fetched (teaching/dictionary.py).
                "word_cards": dictionary_on_screen(app.state.dictionary),
                # A saudação com a transcrição junto: a tela vazia também é
                # inglês escrito, e a linha vale ali como em qualquer outra.
                "greeting": {"content": GREETING, "ipa": spell(GREETING)},
                # O IPA volta com o histórico pelo mesmo motivo que as
                # correções voltam: o que o aluno leu ontem continua na tela
                # quando ele reabre a página. Não é consultado do banco — é
                # função determinística do texto, e guardá-lo criaria uma
                # coluna que pode discordar do que o espeak diz hoje.
                "history": [
                    {
                        "role": t.role,
                        "content": t.content,
                        "ipa": spell(t.content),
                        "corrections": by_turn.get(t.id, []),
                    }
                    for t in history
                ],
            },
        )

    @app.get("/progress", response_class=HTMLResponse)
    async def progress(request: Request) -> HTMLResponse:
        """A tela do error log: no que este aluno mais erra."""
        return templates.TemplateResponse(
            request=request,
            name="progress.html",
            context={
                "version": __version__,
                "model": settings.llm_model,
                "progress": progress_payload(app.state.db),
            },
        )

    @app.get("/read-aloud", response_class=HTMLResponse)
    async def read_aloud_page(request: Request) -> HTMLResponse:
        """The read-aloud drill (E12): hear the pair, read the phrase three times."""
        conn: sqlite3.Connection = app.state.db
        return templates.TemplateResponse(
            request=request,
            name="read-aloud.html",
            context={
                "version": __version__,
                "level": active_student(conn).cefr_level,
                "read_aloud": read_aloud_payload(conn),
            },
        )

    @app.get("/phonetics", response_class=HTMLResponse)
    async def phonetics_page(request: Request) -> HTMLResponse:
        """A box for a sentence, and its IPA under it.

        The page is empty on load: the sentence goes to ``/api/phonetics`` in
        a POST body, never in this page's query string, for the reason
        ``/api/chat`` gives.
        """
        return templates.TemplateResponse(
            request=request,
            name="phonetics.html",
            context={
                "version": __version__,
                "level": active_student(app.state.db).cefr_level,
                "phonetics": phonetics_payload(),
                # Listen buttons only with a voice: /api/tts answers 503
                # without one, and a button that always fails is worse than none.
                "voice": voice_payload(),
                "max_chars": PHONETICS_MAX_CHARS,
                # The word card on the typed words, as on a lesson's text:
                # only with the lookup file fetched (teaching/dictionary.py).
                "word_cards": dictionary_on_screen(app.state.dictionary),
            },
        )

    @app.get("/lessons", response_class=HTMLResponse)
    async def lessons_page(request: Request) -> HTMLResponse:
        """The shelf: every curated lesson, for every student."""
        lessons = shelf()
        return templates.TemplateResponse(
            request=request,
            name="lessons.html",
            context={
                "version": __version__,
                "level": active_student(app.state.db).cefr_level,
                "lessons": [lesson_card(x) for x in lessons] if lessons is not None else [],
                "lesson": None,
            },
        )

    @app.get("/lessons/{slug}", response_class=HTMLResponse)
    async def lesson_page(request: Request, slug: str) -> HTMLResponse:
        lessons = shelf()
        lesson = lessons.get(slug) if lessons is not None else None
        if lesson is None:
            return HTMLResponse("No such lesson.", status_code=404)
        full_text = read_local_text(settings.resolved_corpus_dir, lesson)
        sentences = full_text_sentences(full_text)
        return templates.TemplateResponse(
            request=request,
            name="lessons.html",
            context={
                "version": __version__,
                "level": active_student(app.state.db).cefr_level,
                "lessons": [],
                "lesson": lesson_payload(lesson),
                "aspects": ASPECTS,
                # This machine's copy of the source, if it holds one. Kept out
                # of lesson_payload on purpose: that is also /api/lessons, and
                # the copy is not ours to serve (teaching/lessons.py).
                "full_text": full_text,
                # The copy cut into sentences, each with its IPA, so the reader
                # can light the sentence it is speaking in both lines at once.
                "full_text_sentences": sentences,
                # The symbols that IPA uses, each playing an example word.
                "symbol_chart": symbol_chart(sentences),
                # Beside each point, the sentence of the copy it quotes. From
                # the copy, so it is here and not in lesson_payload either.
                "point_passages": point_passages(lesson, full_text),
                # The copy as blocks to rehearse: from the copy, so here too.
                "rehearsal": rehearsal_view(lesson, sentences),
                # The speaker beside each word asks /api/tts, which answers
                # 503 with voice off — so with voice off there is no button.
                "voice": voice_payload(),
                # Who reads the full text aloud: a voice of its own, not
                # Charles's (config.py).
                "reader_voice": settings.tts_reader_voice,
                # The card under the pointer, only with the lookup fetched: a
                # hover that always answers 503 is worse than none.
                "word_cards": dictionary_on_screen(app.state.dictionary),
            },
        )

    # --- API -------------------------------------------------------------- #

    @app.get("/api/lessons")
    async def get_lessons() -> JSONResponse:
        lessons = shelf()
        return JSONResponse(
            {"lessons": [lesson_card(x) for x in lessons] if lessons is not None else []}
        )

    @app.get("/api/lessons/{slug}")
    async def get_lesson(slug: str) -> JSONResponse:
        lessons = shelf()
        lesson = lessons.get(slug) if lessons is not None else None
        if lesson is None:
            return JSONResponse({"error": f"no lesson {slug!r}"}, status_code=404)
        return JSONResponse(lesson_payload(lesson))

    @app.get("/api/progress")
    async def get_progress() -> JSONResponse:
        return JSONResponse(progress_payload(app.state.db))

    @app.post("/api/chat")
    async def chat(payload: ChatRequest) -> StreamingResponse:
        """Um turno, em streaming.

        É POST com corpo JSON, não ``EventSource``: aquele só faz GET, e a fala
        do aluno é dado pessoal que não deve ir em query string. O cliente lê a
        resposta com ``fetch`` e um reader de stream.

        ## Os quadros ``sentence``, e por que o áudio não vem por aqui

        Com voz ligada, o stream ganha um quadro ``sentence`` a cada sentença
        que fecha — índice e texto limpo para fala. O áudio **não** vem nele: a
        página pede cada sentença ao ``/api/tts`` num pedido próprio.

        É de propósito. Áudio em SSE teria que ir em base64, o que são 33% a
        mais de bytes num quadro que já atravessa o caminho crítico do texto; e
        a síntese passaria a correr dentro do mesmo gerador que repassa os
        pedaços do professor, atrasando a letra para adiantar o som. Num pedido
        separado a síntese da sentença 1 corre **enquanto** o modelo escreve a
        sentença 2, que é exatamente o que o TTS incremental existe para fazer.
        """
        conn: sqlite3.Connection = app.state.db
        runner: TurnRunner = app.state.runner
        voice: Voice | None = app.state.voice
        student = active_student(conn)
        session_id = SessionRepo(conn).current_or_start(student.id)

        async def events() -> AsyncIterator[bytes]:
            # Um montador por turno. Sem voz não há montador: não há para quem
            # mandar sentença, e segmentar à toa seria trabalho perdido no
            # caminho crítico.
            assembler = SentenceAssembler() if voice is not None else None
            spoken = 0

            def sentence_frames(sentences: list[str]) -> list[dict[str, Any]]:
                nonlocal spoken
                frames = []
                for text in sentences:
                    frames.append({"type": "sentence", "index": spoken, "text": text})
                    spoken += 1
                return frames

            try:
                async for event in runner.run(session_id, payload.message, student):
                    # Lista, e não quadro único, porque um pedaço de texto pode
                    # fechar uma sentença (ou duas) junto com ele. A ordem
                    # dentro da lista importa: texto antes do pedido de áudio
                    # dele, e o rabo antes do `done`.
                    frames: list[dict[str, Any]] = []
                    if isinstance(event, Chunk):
                        frames.append({"type": "chunk", "text": event.text})
                        if assembler is not None:
                            frames += sentence_frames(assembler.push(event.text))
                    elif isinstance(event, Phonemized):
                        # A pronúncia é dado derivado do texto, não prosa do
                        # professor: vai em quadro próprio, com o papel, para
                        # a página saber embaixo de qual bolha pendurá-la. A
                        # do aluno chega antes do primeiro `chunk`; a do
                        # professor, depois do último.
                        frames.append(
                            {
                                "type": "phonetics",
                                "role": event.role,
                                "ipa": event.ipa,
                            }
                        )
                    elif isinstance(event, Analyzed):
                        # A correção é dado, não prosa: vai num frame próprio,
                        # para a UI anotar o que o aluno disse em vez de
                        # misturar com o texto do professor.
                        frames.append({"type": "analysis", **event.analysis.as_dict()})
                    elif isinstance(event, Leveled):
                        # Vai em todo turno, não só na mudança: a UI mostra o
                        # nível em vigor, e um frame só na mudança deixaria o
                        # badge defasado depois de um reinício.
                        frames.append({"type": "level", **event.decision.as_dict()})
                    elif isinstance(event, Done):
                        # O rabo que o professor deixou sem pontuar sai ANTES
                        # do `done`: a página trata `done` como "não vem mais
                        # áudio" e fecha a fila.
                        if assembler is not None:
                            tail = assembler.flush()
                            frames += sentence_frames([tail] if tail else [])
                        frames.append(
                            {
                                "type": "done",
                                "first_chunk_ms": round(event.first_chunk_ms),
                                "total_ms": round(event.total_ms),
                                "analysis_ms": round(event.analysis_ms),
                            }
                        )
                    elif isinstance(event, Failed):
                        frames.append({"type": "error", "message": event.message})
                    else:  # pragma: no cover - união fechada
                        continue
                    # Formato SSE, mesmo sobre POST: uma linha `data:` por
                    # evento. Simples de ler no cliente e de depurar no curl.
                    for frame in frames:
                        yield f"data: {json.dumps(frame, ensure_ascii=False)}\n\n".encode()
            except ValueError as exc:
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n".encode()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-store",
                # Sem isto um proxy no caminho pode segurar o corpo e anular o
                # streaming inteiro.
                "X-Accel-Buffering": "no",
            },
        )

    # --- voz (Fase 1.5) ---------------------------------------------------- #

    def voice_or_503() -> Voice | JSONResponse:
        """Os motores, ou a resposta que explica a ausência deles.

        503 e não 404: a rota existe, o serviço é que não está pronto para
        atendê-la — mesma distinção que o ``/health`` faz entre "subiu" e
        "pronto para ensinar".
        """
        voice: Voice | None = app.state.voice
        if voice is None:
            return JSONResponse(
                {
                    "detail": "a voz está desligada neste servidor",
                    "how": "suba com TC_VOICE_ENABLED=true "
                    "(e `uv sync --extra voice` antes, se for a primeira vez)",
                },
                status_code=503,
            )
        return voice

    @app.post("/api/stt")
    async def speech_to_text(request: Request) -> JSONResponse:
        """O que o aluno falou, em texto.

        Recebe os bytes do arquivo no **corpo cru**, não em multipart: é um
        blob só, o ``fetch`` do navegador manda um ``Blob`` direto, e multipart
        exigiria ``python-multipart`` só para embrulhar um campo. Quem
        decodifica o container (webm/opus, ogg, wav) é o motor.

        Devolver só o texto é decisão de produto, não economia: a página põe a
        transcrição **no campo de texto** em vez de mandar o turno. O STT erra,
        e um erro de STT que vira turno faz o professor corrigir um erro que o
        aluno não cometeu — o pior defeito possível neste produto, e justamente
        o que a E3/E4 existem para evitar do lado do modelo. Com a transcrição
        no campo, o erro do STT vira um ajuste de uma palavra.
        """
        voice = voice_or_503()
        if isinstance(voice, JSONResponse):
            return voice

        audio = await request.body()
        if not audio:
            return JSONResponse({"detail": "não veio áudio nenhum"}, status_code=422)

        try:
            transcript = await voice.stt.transcribe(audio)
        except VoiceError as exc:
            # Tipado, nunca por substring: ver ``voice/base.py``.
            return JSONResponse({"detail": str(exc)}, status_code=422)

        return JSONResponse({"bytes": len(audio), **transcript.as_dict()})

    @app.post("/api/tts")
    async def text_to_speech(payload: SpeakRequest) -> Response:
        """Uma sentença falada, em WAV.

        Um pedido por sentença, e a página faz vários em sequência enquanto o
        professor ainda escreve — ver o docstring de ``/api/chat``. O
        ``KokoroTTS`` serializa com lock próprio, então pedidos concorrentes
        viram fila em vez de disputa por CPU.
        """
        voice = voice_or_503()
        if isinstance(voice, JSONResponse):
            return voice

        try:
            speech = await voice.tts.speak(payload.text, payload.voice, payload.speed)
        except VoiceError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=422)

        headers = {
            "Cache-Control": "no-store",
            # Medição do turno de voz sem instrumentar o navegador: o
            # tempo de síntese e a duração do áudio viajam com ele.
            "X-Synthesis-Ms": str(round(speech.elapsed_ms)),
            "X-Audio-Seconds": f"{speech.duration_s:.2f}",
        }
        if payload.timings:
            # The timings are not computed here: a Whisper pass is ~1.2 s,
            # and in this response it would sit between the synthesis and the
            # sound. The audio goes now; the page asks for the timings while
            # it plays, and the pass runs beside the next sentence's synthesis.
            kept: OrderedDict[str, tuple[str, Any]] = app.state.timed_speech
            speech_id = uuid.uuid4().hex
            kept[speech_id] = (payload.text, speech)
            while len(kept) > TIMED_SPEECH_KEPT:
                kept.popitem(last=False)
            headers["X-Speech-Id"] = speech_id
        return Response(content=speech.as_wav(), media_type="audio/wav", headers=headers)

    @app.get("/api/tts/{speech_id}/words")
    async def speech_words(speech_id: str) -> JSONResponse:
        """When each word of a sentence ``/api/tts`` just spoke starts, in seconds.

        One start per whitespace-split word of the text that was sent. Asked
        for once: the entry is dropped as it is read, so the server holds a
        sentence only between its audio and its timings. ``starts`` is empty
        when the STT could not tell, and then the page lights the sentence.
        """
        voice = voice_or_503()
        if isinstance(voice, JSONResponse):
            return voice
        kept: OrderedDict[str, tuple[str, Any]] = app.state.timed_speech
        entry = kept.pop(speech_id, None)
        if entry is None:
            return JSONResponse(
                {"detail": "no such speech: asked for already, or dropped"}, status_code=404
            )
        text, speech = entry
        return JSONResponse({"starts": await timed_words(voice, text, speech)})

    async def timed_words(voice: Voice, text: str, speech: Any) -> list[float]:
        """When each word of ``text`` starts in ``speech``, or ``[]``.

        Kokoro gives no timings, so they are measured: the STT listens to the
        audio just made, and ``voice/align.py`` maps what it heard back onto
        the words. Empty is a first-class answer — no STT that can tell, or one
        that failed — and then the page lights the sentence, not the word.
        """
        try:
            heard = await voice.word_timer.word_times(speech)
        except VoiceError as exc:
            log.warning("no word timings for the reader: %s", exc)
            return []
        return word_starts(text.split(), heard, speech.duration_s)

    @app.post("/api/follow")
    async def follow_reading(payload: FollowRequest) -> JSONResponse:
        """Which of ``words`` the student said last in ``audio``, for "Follow me".

        ``at`` is an index into ``words``, or ``None`` when nothing heard is
        surely in them (``voice/follow.py``). The timing Whisper does the
        hearing, the small one: the light is a second behind the voice at
        best, and the bigger model would double that.

        **It records nothing and returns no transcript.** The audio is heard
        and dropped in this request, and what Whisper heard never leaves it:
        the rehearsal is not a check of what the student said (ADR 0001), and
        a transcript on the page would read as one. Only where the student is.
        """
        voice = voice_or_503()
        if isinstance(voice, JSONResponse):
            return voice
        try:
            pcm = base64.b64decode(payload.audio, validate=True)
        except ValueError:
            return JSONResponse({"detail": "the audio is not base64"}, status_code=422)
        if len(pcm) % 2:
            return JSONResponse({"detail": "the audio is not 16-bit PCM"}, status_code=422)
        speech = Speech(pcm=pcm, sample_rate=STT_SAMPLE_RATE, elapsed_ms=0.0)
        try:
            heard = await voice.word_timer.word_times(speech)
        except VoiceError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=422)
        return JSONResponse(
            {"at": last_heard(payload.words, [w.word for w in heard]), "heard": len(heard)},
            headers={"Cache-Control": "no-store"},
        )

    # --- pronunciation (E11, E12) ----------------------------------------- #

    def pronunciation_or_503() -> tuple[PhonemeRecognizer, Evaluator] | JSONResponse:
        """The recognizer and the evaluator, or the response saying why not."""
        phonemes: PhonemeRecognizer | None = app.state.phonemes
        evaluator: Evaluator | None = app.state.pronunciation
        if phonemes is None:
            return JSONResponse(
                {
                    "detail": "pronunciation is off on this server",
                    "how": "start with TC_PRONUNCIATION_ENABLED=true "
                    "(and `uv sync --extra voice` first, if it is the first time)",
                },
                status_code=503,
            )
        if evaluator is None:
            return JSONResponse(
                {
                    "detail": "the pronunciation base did not load",
                    "why": app.state.pronunciation_error,
                },
                status_code=503,
            )
        return phonemes, evaluator

    def graded_item(
        evaluator: Evaluator, rule_id: str, phrase: int = 0
    ) -> PronunciationItem | JSONResponse:
        """The rule, if it exists, gives verdicts, and has that phrase."""
        try:
            item = evaluator.item(rule_id)
        except LookupError:
            return JSONResponse({"detail": f"no rule {rule_id!r}"}, status_code=404)
        if item.discrimination.status != "graded":
            return JSONResponse(
                {"detail": f"{item.id} is {item.discrimination.status} and gives no verdict"},
                status_code=409,
            )
        if phrase >= len(item.phrases):
            return JSONResponse({"detail": f"{item.id} has no phrase {phrase}"}, status_code=404)
        return item

    def decode_pcm(audio: str) -> bytes | JSONResponse:
        try:
            pcm = base64.b64decode(audio, validate=True)
        except ValueError:
            return JSONResponse({"detail": "the audio is not base64"}, status_code=422)
        if len(pcm) % 2:
            return JSONResponse({"detail": "the audio is not 16-bit PCM"}, status_code=422)
        return pcm

    async def judge_take(
        phonemes: PhonemeRecognizer, evaluator: Evaluator, item_id: str, phrase: int, pcm: bytes
    ) -> tuple[Judgement, SignalVerdict, float]:
        """Metodo §3's three gates over one take, in order.

        Gate 1 runs on the audio **before** the recognizer is asked, so a room
        is never handed to it and cannot become a verdict. Raises
        ``VoiceError`` and ``PronunciationUnavailable`` for the caller to map.
        """
        import numpy as np  # the voice extra: the recognizer cannot be on without it

        samples = samples_from_pcm(pcm, np)
        capture = diagnose_signal(samples, np)
        duration_s = len(samples) / STT_SAMPLE_RATE
        if not capture.usable:
            judgement = evaluator.evaluate(
                item_id, (), frozenset(), signal_usable=False, signal_advice=capture.advice
            )
            return judgement, capture, duration_s
        heard = await phonemes.recognize(
            Speech(pcm=pcm, sample_rate=STT_SAMPLE_RATE, elapsed_ms=0.0)
        )
        judgement = evaluator.evaluate(
            item_id, heard.phonemes, await phonemes.vocabulary(), phrase=phrase
        )
        return judgement, capture, duration_s

    @app.post("/api/pronunciation")
    async def pronunciation_verdict(payload: PronunciationRequest) -> JSONResponse:
        """The verdict on one read-aloud take of a curated phrase (E11).

        Three gates in order, ``docs/metodo-fonetico.md`` §3: the signal
        (``voice/signal.py``), whether the take is the phrase, and whether the
        alignment holds an operation of the rule's signature. Only that last
        one ever names an error, and only with the rule's ``rule_id``; the
        payload carries no PER and no transcription (``Judgement.as_dict``).

        **It records nothing.** No turn, no signal row, no review. The drill
        that persists takes is ``/api/read-aloud/*`` (E12), because what a take
        is worth to the schedule depends on the student's history and on the
        takes around it, which this request does not read.
        """
        ready = pronunciation_or_503()
        if isinstance(ready, JSONResponse):
            return ready
        phonemes, evaluator = ready
        item = graded_item(evaluator, payload.rule_id, payload.phrase)
        if isinstance(item, JSONResponse):
            return item
        pcm = decode_pcm(payload.audio)
        if isinstance(pcm, JSONResponse):
            return pcm
        try:
            judgement, _, _ = await judge_take(phonemes, evaluator, item.id, payload.phrase, pcm)
        except VoiceError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=422)
        except PronunciationUnavailable as exc:
            return JSONResponse({"detail": str(exc)}, status_code=503)
        return JSONResponse(judgement.as_dict(), headers={"Cache-Control": "no-store"})

    def read_aloud_rule(item: PronunciationItem, student_id: int, conn: sqlite3.Connection):
        """What the page shows about one graded rule before a session."""
        schedule = ScheduleRepo(conn).get(student_id, item.id)
        return {
            "rule_id": item.id,
            "target": item.target,
            "right": item.right,
            "wrong": item.wrong,
            "looks_for": read_aloud.looked_for(item),
            "streak": schedule.streak if schedule is not None else 0,
            "graduated": is_graduated(schedule),
            "due_at": schedule.due_at if schedule is not None else "",
        }

    def read_aloud_payload(conn: sqlite3.Connection) -> dict[str, Any]:
        """The page's state: on or off, and the graded rules in schedule order."""
        evaluator: Evaluator | None = app.state.pronunciation
        if app.state.phonemes is None or evaluator is None:
            return {
                "enabled": False,
                "why": app.state.pronunciation_error,
                "rules": [],
            }
        student = active_student(conn)
        repo = ReadAloudRepo(conn)
        order = read_aloud.queue(
            evaluator.base,
            Scheduler(conn),
            student.id,
            errors=repo.errors_by_rule(student.id),
            recent=repo.recent_rules(student.id),
        )
        return {
            "enabled": True,
            "pair": app.state.voice is not None,
            "takes": read_aloud.TAKES,
            "rules": [read_aloud_rule(evaluator.item(r), student.id, conn) for r in order],
        }

    def open_session(
        payload_session: int, conn: sqlite3.Connection
    ) -> tuple[Any, Student] | JSONResponse:
        """The session, if it is this student's and still open."""
        student = active_student(conn)
        session = ReadAloudRepo(conn).get(payload_session)
        if session is None or session.student_id != student.id:
            return JSONResponse({"detail": "no such read-aloud session"}, status_code=404)
        if session.ended_at is not None:
            return JSONResponse({"detail": "this session has ended"}, status_code=409)
        return session, student

    @app.post("/api/read-aloud/start")
    async def read_aloud_start(payload: ReadAloudStart) -> JSONResponse:
        """Open a session over one graded rule (E12).

        The student names the rule, or the schedule picks the most urgent:
        E13's ``Scheduler.queue`` over the graded rules, with the takes that
        named each one as the frequency tiebreak. The phrase rotates over the
        rule's phrases, one per session, so that "2 of 3" is always about the
        same sentence. Whether the deliberate take is still owed is in the
        answer: it is owed once per rule, and never again (metodo §2.4).
        """
        ready = pronunciation_or_503()
        if isinstance(ready, JSONResponse):
            return ready
        _, evaluator = ready
        conn: sqlite3.Connection = app.state.db
        student = active_student(conn)
        repo = ReadAloudRepo(conn)
        if payload.rule_id is None:
            order = read_aloud.queue(
                evaluator.base,
                Scheduler(conn),
                student.id,
                errors=repo.errors_by_rule(student.id),
                recent=repo.recent_rules(student.id),
            )
            if not order:
                return JSONResponse({"detail": "no graded rule to practise"}, status_code=404)
            rule_id, picked_by = order[0], "schedule"
        else:
            rule_id, picked_by = payload.rule_id, "student"
        item = graded_item(evaluator, rule_id)
        if isinstance(item, JSONResponse):
            return item

        phrase = repo.sessions_for(student.id, item.id) % len(item.phrases)
        session_id = repo.start(student.id, item.id, phrase, picked_by)
        text = item.phrases[phrase].text
        return JSONResponse(
            {
                "session_id": session_id,
                "picked_by": picked_by,
                "phrase": phrase,
                "text": text,
                "ipa": spell(text),
                "deliberate": not repo.has_deliberate(student.id, item.id),
                "takes": read_aloud.TAKES,
                "pair": app.state.voice is not None,
                **read_aloud_rule(item, student.id, conn),
            },
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/read-aloud/pair/{rule_id}/{side}")
    async def read_aloud_pair(rule_id: str, side: str, phrase: int = 0) -> Response:
        """One side of the contrast pair, in WAV (metodo §2.2).

        Both sides come from the same phoneme string, the wrong one with the
        curated swap applied, so they differ in the contrast and nothing else.
        It plays audio and judges none, which is why it needs the voice and
        not the recognizer: a rule can be *heard* whatever E9's gate says.
        """
        if side not in ("right", "wrong"):
            return JSONResponse({"detail": "side is right or wrong"}, status_code=404)
        evaluator: Evaluator | None = app.state.pronunciation
        if evaluator is None:
            return JSONResponse({"detail": "pronunciation is off on this server"}, status_code=503)
        item = graded_item(evaluator, rule_id, phrase)
        if isinstance(item, JSONResponse):
            return item
        voice = voice_or_503()
        if isinstance(voice, JSONResponse):
            return voice
        try:
            phonemes = await voice.tts.phonemize(item.phrases[phrase].text)
            if side == "wrong":
                swapped = read_aloud.wrong_side(phonemes, item)
                if swapped is None:
                    return JSONResponse(
                        {"detail": f"{item.id} has no wrong side to synthesize"}, status_code=409
                    )
                phonemes = swapped
            speech = await voice.tts.speak_phonemes(phonemes)
        except VoiceError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=503)
        return Response(
            content=speech.as_wav(),
            media_type="audio/wav",
            # The same two files for everybody: what the pair says is a
            # function of the curated phrase, not of the student.
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @app.post("/api/read-aloud/take")
    async def read_aloud_take(payload: ReadAloudTake) -> JSONResponse:
        """One take of an open session, through E11's three gates.

        A take that reaches gate 3 is recorded as a ``pronunciation_signal``,
        with its two numbers and its operating point; a silenced take records
        **nothing** (metodo §3) and does not count towards the three — the
        student reads it again. The answer carries the verdict, but the page
        tells the student about the session, not the take (plan §5).
        """
        ready = pronunciation_or_503()
        if isinstance(ready, JSONResponse):
            return ready
        phonemes, evaluator = ready
        conn: sqlite3.Connection = app.state.db
        found = open_session(payload.session_id, conn)
        if isinstance(found, JSONResponse):
            return found
        session, student = found
        repo = ReadAloudRepo(conn)
        takes = repo.takes(session.id)
        drill = [t for t in takes if t.pass_ == "unmonitored"]
        if payload.deliberate:
            # Once per rule, and before the drill: the session ends on the
            # correct production (metodo §2.4).
            if repo.has_deliberate(student.id, session.rule_id):
                return JSONResponse(
                    {"detail": "the deliberate take is recorded once per rule"}, status_code=409
                )
            if drill:
                return JSONResponse(
                    {"detail": "the deliberate take comes before the drill takes"},
                    status_code=409,
                )
        elif len(drill) >= read_aloud.TAKES:
            return JSONResponse(
                {"detail": f"{read_aloud.TAKES} takes are judged; finish the session"},
                status_code=409,
            )

        pcm = decode_pcm(payload.audio)
        if isinstance(pcm, JSONResponse):
            return pcm
        try:
            judgement, capture, duration_s = await judge_take(
                phonemes, evaluator, session.rule_id, session.phrase, pcm
            )
        except VoiceError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=422)
        except PronunciationUnavailable as exc:
            return JSONResponse({"detail": str(exc)}, status_code=503)

        recorded = judgement.verdict != "silence"
        if recorded:
            repo.add(
                session,
                pass_="deliberate" if payload.deliberate else "unmonitored",
                verdict=judgement.verdict,
                opportunities=judgement.opportunities,
                errors=len(judgement.findings),
                findings=json.dumps([f.as_dict() for f in judgement.findings]),
                model=phonemes.model,
                duration_s=duration_s,
                speech_from_s=capture.speech_from_s,
                speech_to_s=capture.speech_to_s,
            )
        judged = len(drill) + (1 if recorded and not payload.deliberate else 0)
        body = read_aloud.as_take_payload(judgement, recorded=recorded, judged=judged)
        if payload.deliberate:
            body["note"] = read_aloud.deliberate_note(evaluator.item(session.rule_id), judgement)
        return JSONResponse(body, headers={"Cache-Control": "no-store"})

    @app.post("/api/read-aloud/finish")
    async def read_aloud_finish(payload: ReadAloudFinish) -> JSONResponse:
        """Close the session: the tally, the card, and at most **one** review.

        The close is one statement that only the first finish wins, so a
        second click cannot review twice; the review then goes in like any
        other, event before state (E13). A session that ends 'once' or 'none'
        reviews nothing, and that is the tally's answer, not a failure.
        """
        ready = pronunciation_or_503()
        if isinstance(ready, JSONResponse):
            return ready
        _, evaluator = ready
        conn: sqlite3.Connection = app.state.db
        found = open_session(payload.session_id, conn)
        if isinstance(found, JSONResponse):
            return found
        session, student = found
        repo = ReadAloudRepo(conn)
        item = evaluator.item(session.rule_id)
        result = read_aloud.tally(
            item.id, repo.takes(session.id), ever_named=repo.ever_named(student.id, item.id)
        )
        if not repo.end(session.id, result.outcome):
            return JSONResponse({"detail": "this session has ended"}, status_code=409)

        review = None
        if result.review_hit is not None:
            review = Scheduler(conn).review(
                student.id,
                item.id,
                None,
                hit=result.review_hit,
                source="pronunciation",
                pronunciation_session_id=session.id,
            )
        return JSONResponse(
            {
                "card": read_aloud.card(
                    item,
                    result,
                    due_in_days=review.interval_days if review is not None else None,
                ),
                "review": review.as_dict() if review is not None else None,
            },
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/syntax")
    async def syntax_breakdown(payload: SyntaxRequest) -> JSONResponse:
        """How one message is built, for the Structure button under it.

        POST with a JSON body for the reason ``/api/chat`` is: what the student
        wrote is personal data and does not go in a query string.

        **It records nothing.** No turn, no correction, no review: the student
        looked at a sentence, and looking is not evidence of anything. The
        text is not even checked against the session — a breakdown is a pure
        function of the words.

        The parse runs in a thread. It takes ~3 ms once the pipeline is up,
        but the first one after a failed or skipped warm-up loads spaCy, and
        that is seconds the event loop cannot spend frozen.
        """
        syntax: Syntax | None = app.state.syntax
        if syntax is None:
            return JSONResponse(
                {
                    "detail": "the syntax view is off on this server",
                    "how": "start with TC_SYNTAX_ENABLED=true "
                    "(and `uv sync --extra syntax` first, if it is the first time)",
                },
                status_code=503,
            )

        try:
            breakdown = await asyncio.to_thread(syntax.parse, payload.text)
        except SyntaxUnavailable as exc:
            # Typed, never by substring — same rule as the voice routes.
            return JSONResponse({"detail": str(exc)}, status_code=503)

        return JSONResponse(breakdown.as_dict())

    @app.post("/api/phonetics")
    async def phonetics_of(payload: PhoneticsRequest) -> JSONResponse:
        """The IPA of a text the student typed, for the phonetics page.

        The same espeak as the line under every chat bubble and in every
        lesson, so the three never disagree about a word. **It records
        nothing**, for ``/api/syntax``'s reason: looking up how a sentence
        sounds is not evidence of anything about the student.

        In a thread because the first call after a skipped warm-up loads
        espeak (~0.25 s), and the loop should not freeze for it.
        """
        phonetics: Phonetics | None = app.state.phonetics
        if phonetics is None:
            return JSONResponse(
                {
                    "detail": "the IPA is off on this server",
                    "how": "start without TC_PHONETICS_ENABLED=false",
                },
                status_code=503,
            )

        lines = await asyncio.to_thread(spelled_out, payload.text)
        if lines and not phonetics.ready:
            # Any non-blank text makes espeak load, so here it tried and failed.
            # Empty IPA from a loaded espeak is an answer (a line of only
            # punctuation has no sound); empty IPA because espeak never came
            # up is not, and the page must not draw it as one.
            return JSONResponse(
                {"detail": phonetics.last_error or "espeak did not load"}, status_code=503
            )
        return JSONResponse(
            {
                "language": phonetics.language,
                "stress": phonetics.with_stress,
                "lines": lines,
            }
        )

    @app.post("/api/define")
    async def define(payload: DefineRequest) -> JSONResponse:
        """The card for one word of a lesson's full text.

        Its senses in WordNet's order, each with an English definition, an
        example and the Portuguese words for it, plus the IPA of each headword
        from the same espeak as every other IPA on the page — never from the
        model (``teaching/dictionary.py``). A word with no entry is a 200 with
        ``found: false``, because "no entry" is an answer the card draws.

        POST for ``/api/syntax``'s reason, although the word comes from the
        lesson and not from the student: the route reads any text it is sent,
        and that is a body, never a query string. **It records nothing** —
        looking a word up is not evidence of anything about the student.
        """
        dictionary: Dictionary | None = app.state.dictionary
        if dictionary is None:
            return JSONResponse(
                {
                    "detail": "word cards are off on this server",
                    "how": "start without TC_DICTIONARY_ENABLED=false",
                },
                status_code=503,
            )
        try:
            entry = await asyncio.to_thread(dictionary.lookup, payload.word)
        except DictionaryUnavailable as exc:
            return JSONResponse({"detail": str(exc)}, status_code=503)

        body = entry.as_dict()
        for headword in body["headwords"]:
            headword["ipa"] = spell(headword["lemma"])
        return JSONResponse(body)

    @app.get("/api/voice")
    async def voice_status() -> JSONResponse:
        """O mesmo que a página recebe no carregamento, para depurar sem F12."""
        return JSONResponse(voice_payload())

    @app.get("/api/history")
    async def history() -> JSONResponse:
        conn: sqlite3.Connection = app.state.db
        session_id = active_session(conn)
        turns = TurnRepo(conn).history(session_id, limit=200)
        corrections = CorrectionRepo(conn).for_session(session_id)
        by_turn: dict[int, list[dict[str, Any]]] = {}
        for c in corrections:
            by_turn.setdefault(c.turn_id, []).append(
                {
                    "span": c.span,
                    "correction": c.corrected,
                    "type": c.error_type,
                    "rule_id": c.rule_id,
                    "explanation": c.explanation,
                }
            )
        return JSONResponse(
            {
                "session_id": session_id,
                "turns": [
                    {
                        "role": t.role,
                        "content": t.content,
                        "created_at": t.created_at,
                        "corrections": by_turn.get(t.id, []),
                    }
                    for t in turns
                ],
            }
        )

    @app.get("/api/profile")
    async def get_profile() -> JSONResponse:
        return JSONResponse(profile_payload(app.state.db))

    @app.post("/api/profile")
    async def set_profile(payload: ProfileRequest) -> JSONResponse:
        """Troca de perfil e/ou declara o nível.

        Mandar um ``name`` que não existe cria o perfil — é como se escolhe o
        aluno, sem login e sem tela de cadastro. Mandar ``level`` redefine o
        nível autodeclarado: o aluno tem a última palavra sobre o próprio nível,
        e o ajuste automático parte do que ele disse.
        """
        conn: sqlite3.Connection = app.state.db
        students = StudentRepo(conn)

        level = payload.level.strip().upper() if payload.level else None
        if level is not None and level not in CEFR_LEVELS:
            return JSONResponse(
                {"detail": f"nível inválido: {payload.level!r}", "levels": list(CEFR_LEVELS)},
                status_code=422,
            )

        name = payload.name.strip() if payload.name else None
        if name:
            student = students.get_or_create(name, cefr_level=level or DEFAULT_LEVEL)
            students.touch(student.id)
        else:
            student = active_student(conn)

        if level is not None and level != student.cefr_level:
            levels = LevelRepo(conn)
            levels.log_change(
                student.id,
                student.cefr_level,
                level,
                source="self",
                reason="the student set the level",
                # Zera a janela de histerese: a evidência acumulada antes da
                # declaração desfaria a escolha do aluno no turno seguinte.
                at_signal_id=levels.last_signal_id(student.id),
            )
            students.set_level(student.id, level, source="self")

        return JSONResponse(profile_payload(conn))

    @app.get("/api/mode")
    async def get_mode() -> JSONResponse:
        return JSONResponse(session_payload(app.state.db))

    @app.post("/api/mode")
    async def set_mode(payload: ModeRequest) -> JSONResponse:
        """Troca de modo, o que é trocar de atividade.

        Modo é propriedade da sessão desde a E0, então trocar abre uma sessão
        nova: a transcrição de um drill não é continuação da conversa livre. O
        histórico antigo fica no banco — é o material do error log.

        É aqui que a regra do drill é escolhida, uma vez por sessão: por
        consulta ao error log do aluno, não pelo modelo, e não a cada turno.
        """
        conn: sqlite3.Connection = app.state.db
        mode = payload.mode.strip().lower()
        if mode not in MODES:
            return JSONResponse(
                {"detail": f"modo inválido: {payload.mode!r}", "modes": list(MODES)},
                status_code=422,
            )

        student = active_student(conn)
        planner: DrillPlanner | None = app.state.planner

        rule_id = None
        if mode == "drill" and planner is not None:
            target = planner.pick(student.id)
            # None é estado legítimo: aluno sem erro registrado ainda não tem o
            # que treinar, e o modo diz isso em vez de inventar exercício.
            rule_id = target.rule_id if target is not None else None

        SessionRepo(conn).switch_mode(student.id, mode, rule_id)
        return JSONResponse(session_payload(conn))

    @app.post("/api/reset")
    async def reset() -> JSONResponse:
        """Encerra a sessão atual e começa outra.

        O histórico antigo fica no banco — é o material do error log da E7.
        """
        conn: sqlite3.Connection = app.state.db
        student = active_student(conn)
        sessions = SessionRepo(conn)
        current = sessions.current(student.id)
        # Conversa nova continua no mesmo modo: quem clica em "new conversation"
        # quer recomeçar o assunto, não sair do drill.
        mode = current.mode if current is not None else DEFAULT_MODE
        rule_id = current.drill_rule_id if current is not None else None
        if current is not None:
            sessions.end(current.id)
        sessions.start(student.id, mode, rule_id)
        return JSONResponse(session_payload(conn))

    @app.post("/api/shutdown")
    async def shutdown(request: Request) -> JSONResponse:
        """Desliga o Charles a pedido do botão da página.

        A parada vai num ``BackgroundTask`` porque ele só roda depois que o
        corpo da resposta foi para o socket. Sinalizar antes faria o
        navegador ver a conexão cair no lugar do "parando" — o mesmo clique,
        mas com cara de queda, que é justamente o que a E0 tirou da subida.
        """
        if not from_this_page(request):
            return JSONResponse(
                {"detail": "shutdown is only accepted from the Charles page"},
                status_code=403,
            )
        return JSONResponse({"stopping": True}, background=BackgroundTask(app.state.stop))

    @app.get("/health")
    async def health() -> JSONResponse:
        model = await probe_model(settings)

        conn: sqlite3.Connection = app.state.db
        try:
            applied = [
                row["name"]
                for row in conn.execute("SELECT name FROM schema_migrations ORDER BY name")
            ]
            db_info: dict[str, Any] = {
                "ok": True,
                "path": str(settings.resolved_db_path),
                "migrations_applied": applied,
            }
        except sqlite3.Error as exc:
            db_info = {
                "ok": False,
                "path": str(settings.resolved_db_path),
                "detail": str(exc),
            }

        # A base contrastiva é conteúdo do produto, versionado e validado por
        # schema: se ela não carrega, o professor perde a regra curada e volta
        # a improvisar gramática — exatamente o que a E3 existe para evitar.
        # Melhor descobrir no /health que no primeiro turno do aluno.
        try:
            base = load_base(settings.data_dir / "contrastive")
            contrastive_info: dict[str, Any] = {"ok": True, "items": len(base)}
        except ContrastiveError as exc:
            contrastive_info = {"ok": False, "items": 0, "detail": str(exc)}

        # Prompt de nível ou de modo que falta não derruba o turno — a resposta
        # sai com a persona sozinha —, mas tira em silêncio metade da E5 e da
        # E6. Silencioso é exatamente o que o /health existe para evitar, então
        # o que falta aparece aqui sem mudar o status.
        prompts: PromptLibrary = app.state.prompts
        missing_levels = [
            level for level in CEFR_LEVELS if not prompts.exists(level_partial_name(level))
        ]
        missing_modes = [mode for mode in MODES if not prompts.exists(mode_partial_name(mode))]

        # A voz é opcional, então ela só conta para o status quando foi
        # pedida. Pedida e não carregada é degradado, e com o motivo: o
        # aquecimento engole o erro de propósito, e sem isto ele sumiria.
        voice: Voice | None = app.state.voice
        if voice is None:
            voice_info: dict[str, Any] = {"ok": True, "enabled": False}
        else:
            voice_info = {
                "ok": voice.stt.ready and voice.tts.ready,
                "enabled": True,
                "stt": {
                    "model": voice.stt.model,
                    "ready": voice.stt.ready,
                    "detail": voice.stt.last_error,
                },
                "tts": {
                    "model": voice.tts.model,
                    "voice": voice.tts.voice,
                    "ready": voice.tts.ready,
                    "detail": voice.tts.last_error,
                    "voices": len(voice.tts.voices()),
                },
            }

        # Pronunciation counts for the status only when asked for, like the
        # voice: asked for and not loaded is degraded, with the reason.
        phonemes_engine: PhonemeRecognizer | None = app.state.phonemes
        if phonemes_engine is None:
            pronunciation_info: dict[str, Any] = {"ok": True, "enabled": False}
        else:
            evaluator_state: Evaluator | None = app.state.pronunciation
            pronunciation_info = {
                "ok": phonemes_engine.ready and evaluator_state is not None,
                "enabled": True,
                "model": phonemes_engine.model,
                "ready": phonemes_engine.ready,
                "detail": phonemes_engine.last_error or app.state.pronunciation_error,
            }

        # O IPA não conta para o status, e isso é escolha: a aula funciona
        # inteira sem a linha de pronúncia, enquanto sem banco, modelo ou base
        # ela não funciona. Mas aparece com o motivo — o aquecimento engole o
        # erro, e sem isto um espeak que não carregou sumiria em silêncio.
        phonetics: Phonetics | None = app.state.phonetics
        if phonetics is None:
            phonetics_info: dict[str, Any] = {"ok": True, "enabled": False}
        else:
            phonetics_info = {
                "ok": phonetics.last_error is None,
                "enabled": True,
                "language": phonetics.language,
                "stress": phonetics.with_stress,
                "ready": phonetics.ready,
                "detail": phonetics.last_error,
            }

        # The syntax view does not count towards the status either, for the
        # IPA line's reason: the lesson is whole without the panel. Asked for
        # and not installed is the case worth the block — the button is absent
        # from the page, and this is the only place that says why.
        syntax: Syntax | None = app.state.syntax
        if syntax is None:
            syntax_info: dict[str, Any] = {"ok": True, "enabled": False}
        else:
            installed = syntax.installed
            syntax_info = {
                "ok": installed and syntax.last_error is None,
                "enabled": True,
                "model": syntax.model,
                "installed": installed,
                "ready": syntax.ready,
                "detail": syntax.last_error
                if installed
                else "spaCy or the pipeline is not installed; run `uv sync --extra syntax`",
            }

        # Word cards do not count towards the status either: the lesson is
        # whole without them. Asked for and not fetched is a fresh clone's
        # ordinary state, and this block is the one place that says so.
        dictionary: Dictionary | None = app.state.dictionary
        if dictionary is None:
            dictionary_info: dict[str, Any] = {"ok": True, "enabled": False}
        else:
            installed = dictionary.installed
            dictionary_info = {
                "ok": installed and dictionary.last_error is None,
                "enabled": True,
                "installed": installed,
                "path": str(dictionary.path),
                "ready": dictionary.ready,
                "detail": dictionary.last_error
                if installed
                else "no lookup file; run "
                "`uv run --with wn==1.1.1 python scripts/fetch_dictionary.py`",
            }

        # Lessons do not count towards the status, for the IPA line's reason:
        # the conversation is whole without them. But a shelf that failed to
        # load would otherwise only be an empty page, so the reason shows here.
        lessons_shelf: LessonShelf | None = app.state.lessons
        if lessons_shelf is not None:
            lessons_info: dict[str, Any] = {"ok": True, "items": len(lessons_shelf)}
        else:
            try:
                lessons_info = {
                    "ok": True,
                    "items": len(load_shelf(settings.data_dir / "lessons")),
                }
            except LessonError as exc:
                lessons_info = {"ok": False, "items": 0, "detail": str(exc)}

        body = {
            # O serviço está de pé, mas "pronto para ensinar" exige banco,
            # modelo e base contrastiva. Separar os três evita um health verde
            # enganoso.
            "status": "ok"
            if (
                db_info["ok"]
                and model.ok
                and contrastive_info["ok"]
                and voice_info["ok"]
                and pronunciation_info["ok"]
            )
            else "degraded",
            "version": __version__,
            # The running code's version (semver.py): equal to the release on
            # the release commit, a `-dev.N` pre-release of the next one past it.
            "build_version": build.version or __version__,
            "database": db_info,
            "model": model.as_dict(),
            "contrastive": contrastive_info,
            "level_partials": {
                "ok": not missing_levels,
                "missing": missing_levels,
            },
            "mode_prompts": {
                "ok": not missing_modes,
                "missing": missing_modes,
            },
            "provider": type(app.state.provider).__name__,
            "voice": voice_info,
            "pronunciation": pronunciation_info,
            "phonetics": phonetics_info,
            "syntax": syntax_info,
            "dictionary": dictionary_info,
            "lessons": lessons_info,
        }
        return JSONResponse(body, status_code=200 if body["status"] == "ok" else 503)

    return app


app = create_app()
