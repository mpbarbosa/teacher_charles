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
from .store.repo import SessionRepo, StudentRepo, TurnRepo
from .teaching import Chunk, Done, Failed, TurnRunner

log = logging.getLogger(__name__)

WEB_DIR = PACKAGE_ROOT / "web"

# Na E2 há um aluno só. Perfil e múltiplos alunos são da E5.
DEFAULT_STUDENT = "student"


async def _warm_up(provider: ChatProvider, prompts: PromptLibrary) -> None:
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
    try:
        await provider.complete(
            [
                Message(role="system", content=prompts.get("persona")),
                Message(role="user", content="Hello."),
            ],
            GenerationOptions(temperature=0.0, max_tokens=1, think=False),
        )
        log.info("modelo %s aquecido, persona em cache", provider.model)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - aquecimento nunca derruba a subida
        log.warning("aquecimento do modelo falhou (segue assim mesmo): %s", exc)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


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

        app.state.settings = settings
        app.state.db = conn
        app.state.provider = chat
        app.state.prompts = prompts
        app.state.runner = TurnRunner(
            provider=chat,
            prompts=prompts,
            conn=conn,
            options=default_options(settings),
        )

        # Carregar o modelo do disco custa ~15s nesta máquina, e medido na E2
        # essa espera caía inteira no primeiro turno do aluno (16,3s até a
        # primeira palavra, contra 1,1s nos turnos seguintes). Aquecer em
        # segundo plano move isso para enquanto a página ainda está sendo lida.
        warmup = asyncio.create_task(_warm_up(chat, prompts)) if settings.llm_warmup else None
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

    def active_session(conn: sqlite3.Connection) -> int:
        student = StudentRepo(conn).get_or_create(DEFAULT_STUDENT)
        return SessionRepo(conn).current_or_start(student.id)

    # --- páginas ---------------------------------------------------------- #

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        conn: sqlite3.Connection = app.state.db
        session_id = active_session(conn)
        history = TurnRepo(conn).history(session_id, limit=50)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "version": __version__,
                "model": settings.llm_model,
                "history": [{"role": t.role, "content": t.content} for t in history],
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
        session_id = active_session(conn)

        async def events() -> AsyncIterator[bytes]:
            try:
                async for event in runner.run(session_id, payload.message):
                    if isinstance(event, Chunk):
                        frame = {"type": "chunk", "text": event.text}
                    elif isinstance(event, Done):
                        frame = {
                            "type": "done",
                            "first_chunk_ms": round(event.first_chunk_ms),
                            "total_ms": round(event.total_ms),
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
        return JSONResponse(
            {
                "session_id": session_id,
                "turns": [
                    {"role": t.role, "content": t.content, "created_at": t.created_at}
                    for t in turns
                ],
            }
        )

    @app.post("/api/reset")
    async def reset() -> JSONResponse:
        """Encerra a sessão atual e começa outra.

        O histórico antigo fica no banco — é o material do error log da E7.
        """
        conn: sqlite3.Connection = app.state.db
        student = StudentRepo(conn).get_or_create(DEFAULT_STUDENT)
        sessions = SessionRepo(conn)
        sessions.end(sessions.current_or_start(student.id))
        return JSONResponse({"session_id": sessions.current_or_start(student.id)})

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

        body = {
            # O serviço está de pé, mas "pronto para ensinar" exige banco e
            # modelo. Separar os dois evita um health verde enganoso.
            "status": "ok" if (db_info["ok"] and model.ok) else "degraded",
            "version": __version__,
            "database": db_info,
            "model": model.as_dict(),
            "provider": type(app.state.provider).__name__,
        }
        return JSONResponse(body, status_code=200 if body["status"] == "ok" else 503)

    return app


app = create_app()
