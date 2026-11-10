"""Persistência: SQLite, migrações e acesso aos dados."""

from .db import connect, init_db, migrate, pending_migrations, session
from .models import (
    Correction,
    GapCount,
    LevelChange,
    LevelSignal,
    RuleCount,
    RuleReview,
    RuleSchedule,
    Session,
    Student,
    Turn,
    UnverifiedFinding,
)
from .repo import (
    CorrectionRepo,
    LevelRepo,
    ScheduleRepo,
    SessionRepo,
    StudentRepo,
    TurnRepo,
    UnverifiedFindingRepo,
)

__all__ = [
    "Correction",
    "CorrectionRepo",
    "GapCount",
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
    "UnverifiedFinding",
    "UnverifiedFindingRepo",
    "connect",
    "init_db",
    "migrate",
    "pending_migrations",
    "session",
]
