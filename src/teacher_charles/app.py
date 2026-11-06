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
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from . import __version__
from .config import PACKAGE_ROOT, Settings, get_settings
from .health import probe_model
from .llm import ChatProvider, build_provider, default_options
from .llm.base import GenerationOptions, Message
from .prompts import PromptLibrary
from .store.db import init_db
from .store.models import CEFR_LEVELS, MODES, Student
from .store.repo import CorrectionRepo, LevelRepo, SessionRepo, StudentRepo, TurnRepo
from .teaching import Analyzed, Analyzer, Chunk, Done, Failed, Leveled, TurnRunner
from .teaching.contrastive import ContrastiveError, load_base
from .teaching.level import LevelTracker, partial_name
from .teaching.modes import DEFAULT_MODE, DrillPlanner
from .teaching.turn import DEFAULT_LEVEL

log = logging.getLogger(__name__)

WEB_DIR = PACKAGE_ROOT / "web"

# O perfil que existe antes de o aluno dizer o nome dele. A partir da E5 há
# vários perfis numa máquina — o ativo é o visto por último —, mas não há login:
# multiusuário fica fora da Fase 1 (plano §4).
DEFAULT_STUDENT = "student"


async def _warm_up(provider: ChatProvider, prompts: PromptLibrary, level: str) -> None:
    """Carrega o modelo **e** põe a persona no cache de prefixo.

    O prompt tem que ser o mesmo de um turno real, com a persona como mensagem
    de sistema. Medido: aquecer com um "hi" solto carrega o modelo mas deixa os
    427 tokens da persona fora do cache, e o primeiro turno do aluno paga 19s
    só de avaliação de prompt. Com o prefixo cacheado, os mesmos 427 tokens
    custam 0,4-0,7s.

    Nunca propaga erro: se o modelo não existe ou o Ollama está fora do ar, o
    /health já diz isso, e derrubar a subida por causa do aquecimento seria
    pior do que um primeiro turno lento.
    """
    messages = [Message(role="system", content=prompts.get("persona"))]
    # O prefixo cacheado tem que ser o do turno real, e desde a E5 o turno real
    # manda persona **e** ajuste de nível como mensagens de sistema. Aquecer só
    # com a persona deixaria o partial fora do cache, e o primeiro turno pagaria
    # a avaliação dele.
    name = partial_name(level)
    if prompts.exists(name):
        messages.append(Message(role="system", content=prompts.get(name)))
    messages.append(Message(role="user", content="Hello."))

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

        app.state.settings = settings
        app.state.db = conn
        app.state.provider = chat
        app.state.prompts = prompts
        app.state.analyzer = analyzer
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
        )

        # Carregar o modelo do disco custa ~15s nesta máquina, e medido na E2
        # essa espera caía inteira no primeiro turno do aluno (16,3s até a
        # primeira palavra, contra 1,1s nos turnos seguintes). Aquecer em
        # segundo plano move isso para enquanto a página ainda está sendo lida.
        # Aquece com o nível do perfil ativo: é o prompt que o primeiro turno
        # vai mandar de verdade.
        active = StudentRepo(conn).active()
        warm_level = active.cefr_level if active is not None else DEFAULT_LEVEL
        warmup = (
            asyncio.create_task(_warm_up(chat, prompts, warm_level))
            if settings.llm_warmup
            else None
        )
        app.state.warmup = warmup

        try:
            yield
        finally:
            if warmup is not None and not warmup.done():
                warmup.cancel()
                with suppress(asyncio.CancelledError):
                    await warmup
            await chat.aclose()
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
                "history": [
                    {
                        "role": t.role,
                        "content": t.content,
                        "corrections": by_turn.get(t.id, []),
                    }
                    for t in history
                ],
            },
        )

    # --- API -------------------------------------------------------------- #

    @app.post("/api/chat")
    async def chat(payload: ChatRequest) -> StreamingResponse:
        """Um turno, em streaming.

        É POST com corpo JSON, não ``EventSource``: aquele só faz GET, e a fala
        do aluno é dado pessoal que não deve ir em query string. O cliente lê a
        resposta com ``fetch`` e um reader de stream.
        """
        conn: sqlite3.Connection = app.state.db
        runner: TurnRunner = app.state.runner
        student = active_student(conn)
        session_id = SessionRepo(conn).current_or_start(student.id)

        async def events() -> AsyncIterator[bytes]:
            try:
                async for event in runner.run(session_id, payload.message, student):
                    if isinstance(event, Chunk):
                        frame = {"type": "chunk", "text": event.text}
                    elif isinstance(event, Analyzed):
                        # A correção é dado, não prosa: vai num frame próprio,
                        # para a UI anotar o que o aluno disse em vez de
                        # misturar com o texto do professor.
                        frame = {"type": "analysis", **event.analysis.as_dict()}
                    elif isinstance(event, Leveled):
                        # Vai em todo turno, não só na mudança: a UI mostra o
                        # nível em vigor, e um frame só na mudança deixaria o
                        # badge defasado depois de um reinício.
                        frame = {"type": "level", **event.decision.as_dict()}
                    elif isinstance(event, Done):
                        frame = {
                            "type": "done",
                            "first_chunk_ms": round(event.first_chunk_ms),
                            "total_ms": round(event.total_ms),
                            "analysis_ms": round(event.analysis_ms),
                        }
                    elif isinstance(event, Failed):
                        frame = {"type": "error", "message": event.message}
                    else:  # pragma: no cover - união fechada
                        continue
                    # Formato SSE, mesmo sobre POST: uma linha `data:` por
                    # evento. Simples de ler no cliente e de depurar no curl.
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

        # Partial de nível que falta não derruba o turno — a resposta sai com a
        # persona sozinha —, mas tira silenciosamente o ajuste por nível, que é
        # metade da E5. Silencioso é exatamente o que o /health existe para
        # evitar, então o que falta aparece aqui sem mudar o status.
        prompts: PromptLibrary = app.state.prompts
        missing_levels = [level for level in CEFR_LEVELS if not prompts.exists(partial_name(level))]

        body = {
            # O serviço está de pé, mas "pronto para ensinar" exige banco,
            # modelo e base contrastiva. Separar os três evita um health verde
            # enganoso.
            "status": "ok"
            if (db_info["ok"] and model.ok and contrastive_info["ok"])
            else "degraded",
            "version": __version__,
            "database": db_info,
            "model": model.as_dict(),
            "contrastive": contrastive_info,
            "level_partials": {
                "ok": not missing_levels,
                "missing": missing_levels,
            },
            "provider": type(app.state.provider).__name__,
        }
        return JSONResponse(body, status_code=200 if body["status"] == "ok" else 503)

    return app


app = create_app()
