"""Aplicação FastAPI.

E2: a conversa por texto funciona de ponta a ponta no navegador. A resposta sai
em streaming porque, medido nesta máquina (7-8 tok/s num 4B), é isso que faz a
espera ser tolerável — o aluno começa a ler no primeiro segundo em vez de
encarar uma tela parada por dez.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sqlite3
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
)
from .teaching.contrastive import ContrastiveError, load_base
from .teaching.level import LevelTracker
from .teaching.level import partial_name as level_partial_name
from .teaching.modes import DEFAULT_MODE, DrillPlanner
from .teaching.modes import partial_name as mode_partial_name
from .teaching.phonetics import Phonetics, build_phonetics
from .teaching.schedule import is_graduated
from .teaching.turn import DEFAULT_LEVEL
from .voice import SentenceAssembler, Voice, VoiceError, build_voice

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

        app.state.settings = settings
        app.state.db = conn
        # Em app.state, e não fechado na rota: é aqui que o teste troca o
        # sinal por um registro de que o desligamento foi pedido.
        app.state.stop = request_stop
        app.state.provider = chat
        app.state.voice = voice
        app.state.phonetics = phonetics
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

        try:
            yield
        finally:
            for task in (warmup, app.state.voice_warmup, app.state.phonetics_warmup):
                if task is not None and not task.done():
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
            await chat.aclose()
            if voice is not None:
                await voice.aclose()
            conn.close()

    app = FastAPI(
        title="Teacher Charles",
        version=__version__,
        summary="Um professor de inglês que roda localmente",
        lifespan=lifespan,
    )

    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=WEB_DIR / "templates")

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
        return {
            "student": student.name,
            "level": student.cefr_level,
            "level_source": student.level_source,
            "self_declared_level": student.self_declared_level,
            "totals": totals,
            "rules": rules,
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

    # --- API -------------------------------------------------------------- #

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
            speech = await voice.tts.speak(payload.text, payload.voice)
        except VoiceError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=422)

        return Response(
            content=speech.as_wav(),
            media_type="audio/wav",
            headers={
                "Cache-Control": "no-store",
                # Medição do turno de voz sem instrumentar o navegador: o
                # tempo de síntese e a duração do áudio viajam com ele.
                "X-Synthesis-Ms": str(round(speech.elapsed_ms)),
                "X-Audio-Seconds": f"{speech.duration_s:.2f}",
            },
        )

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

        body = {
            # O serviço está de pé, mas "pronto para ensinar" exige banco,
            # modelo e base contrastiva. Separar os três evita um health verde
            # enganoso.
            "status": "ok"
            if (db_info["ok"] and model.ok and contrastive_info["ok"] and voice_info["ok"])
            else "degraded",
            "version": __version__,
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
            "phonetics": phonetics_info,
        }
        return JSONResponse(body, status_code=200 if body["status"] == "ok" else 503)

    return app


app = create_app()
