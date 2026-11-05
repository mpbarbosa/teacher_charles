"""Persistência: SQLite, migrações e acesso aos dados."""

from .db import connect, init_db, migrate, pending_migrations, session
from .models import Correction, Student, Turn
from .repo import SessionRepo, StudentRepo, TurnRepo

__all__ = [
    "Correction",
    "SessionRepo",
    "Student",
    "StudentRepo",
    "Turn",
    "TurnRepo",
    "connect",
    "init_db",
    "migrate",
    "pending_migrations",
    "session",
]
