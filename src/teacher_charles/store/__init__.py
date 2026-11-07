"""Persistência: SQLite, migrações e acesso aos dados."""

from .db import connect, init_db, migrate, pending_migrations, session
from .models import (
    Correction,
    LevelChange,
    LevelSignal,
    RuleCount,
    RuleReview,
    RuleSchedule,
    Session,
    Student,
    Turn,
)
from .repo import (
    CorrectionRepo,
    LevelRepo,
    ScheduleRepo,
    SessionRepo,
    StudentRepo,
    TurnRepo,
)

__all__ = [
    "Correction",
    "CorrectionRepo",
    "LevelChange",
    "LevelRepo",
    "LevelSignal",
    "RuleCount",
    "RuleReview",
    "RuleSchedule",
    "ScheduleRepo",
    "Session",
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
