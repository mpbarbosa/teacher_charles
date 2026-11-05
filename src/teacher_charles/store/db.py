"""Conexão com o SQLite e migrações.

As migrações são arquivos ``.sql`` numerados em ``migrations/``, aplicados em
ordem e registrados em ``schema_migrations``. Sem framework: o esquema é
pequeno, e um ``.sql`` legível no diff vale mais aqui que um ORM.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def connect(db_path: Path) -> sqlite3.Connection:
    """Abre uma conexão com os pragmas que queremos em todo lugar.

    ``check_same_thread=False`` porque o app mantém **uma** conexão em
    ``app.state.db`` e a usa de mais de uma thread: as rotas ``async`` rodam no
    event loop, mas qualquer rota síncrona que venha a existir roda no
    threadpool do FastAPI, e o ``TestClient`` também executa o app em outra
    thread. É seguro aqui porque ``sqlite3.threadsafety == 3`` (modo
    serializado): o guard do Python é mais estrito que o do SQLite.

    A contrapartida, e ela é real: **nada no caminho de uma requisição pode
    abrir transação de várias instruções** nesta conexão, porque duas
    requisições podem se intercalar. Por isso todo método dos repositórios é
    uma instrução autônoma. As migrações são a exceção, e rodam na subida,
    antes de o servidor aceitar conexão.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # ON DELETE CASCADE do esquema só vale com isto ligado, e o SQLite
    # desliga por default em cada conexão nova.
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def _ensure_migrations_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            name       TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )


def pending_migrations(conn: sqlite3.Connection) -> list[Path]:
    _ensure_migrations_table(conn)
    applied = {row["name"] for row in conn.execute("SELECT name FROM schema_migrations")}
    return [p for p in sorted(MIGRATIONS_DIR.glob("*.sql")) if p.name not in applied]


def migrate(conn: sqlite3.Connection) -> list[str]:
    """Aplica as migrações pendentes. Devolve os nomes das que rodaram."""
    applied: list[str] = []
    for path in pending_migrations(conn):
        # Uma migração é atômica: ou o script inteiro entra, ou nada entra e
        # ela continua pendente.
        #
        # O BEGIN vai DENTRO do script de propósito. `executescript` encerra
        # qualquer transação já aberta na conexão, então um `conn.execute("BEGIN")`
        # aqui fora seria desfeito antes de o script rodar e o ROLLBACK não
        # protegeria nada. Com o BEGIN no próprio script a transação continua
        # aberta depois do erro, e aí o ROLLBACK desfaz o DDL parcial.
        sql = path.read_text(encoding="utf-8")
        try:
            conn.executescript("BEGIN;\n" + sql)
            conn.execute("INSERT INTO schema_migrations (name) VALUES (?)", (path.name,))
            conn.execute("COMMIT")
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        applied.append(path.name)
    return applied


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = connect(db_path)
    migrate(conn)
    return conn


@contextmanager
def session(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = init_db(db_path)
    try:
        yield conn
    finally:
        conn.close()
