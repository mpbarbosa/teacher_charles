"""Aplicação FastAPI.

E0 entrega só a fundação: subir, migrar o banco e dizer num ``/health`` honesto
se o modelo está ao alcance. A conversa chega na E2.
"""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from . import __version__
from .config import Settings, get_settings
from .health import probe_model
from .store.db import init_db


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Migrar na subida: um dev que acabou de clonar não deveria precisar
        # rodar nada à mão antes do primeiro `uv run teacher-charles`.
        conn = init_db(settings.resolved_db_path)
        app.state.settings = settings
        app.state.db = conn
        try:
            yield
        finally:
            conn.close()

    app = FastAPI(
        title="Teacher Charles",
        version=__version__,
        summary="Um professor de inglês que roda localmente",
        lifespan=lifespan,
    )

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
        }
        return JSONResponse(body, status_code=200 if body["status"] == "ok" else 503)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        model = await probe_model(settings)
        badge = "pronto" if model.ok else "pendente"
        colour = "#1a7f37" if model.ok else "#9a6700"
        return HTMLResponse(
            f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Teacher Charles</title>
<style>
  body {{ font: 16px/1.6 system-ui, sans-serif; max-width: 34rem;
         margin: 4rem auto; padding: 0 1rem; color: #1f2328; }}
  code {{ background: #f6f8fa; padding: .1rem .3rem; border-radius: .2rem; }}
  .badge {{ color: {colour}; font-weight: 600; }}
</style>
<h1>Teacher Charles <small style="font-weight:400">v{__version__}</small></h1>
<p>Backend de pé. A interface de conversa chega na entrega <strong>E2</strong>.</p>
<p>Modelo <code>{model.wanted_model}</code>: <span class="badge">{badge}</span><br>
<small>{model.detail}</small></p>
<p><a href="/health">/health</a> &middot; <a href="/docs">/docs</a></p>
</html>"""
        )

    return app


app = create_app()
