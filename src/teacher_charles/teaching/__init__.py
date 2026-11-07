"""A estrutura pedagógica.

Este pacote é o produto. O modelo é um componente substituível; o que faz o
Charles ser um professor e não um chat genérico mora aqui.
"""

from .analyzer import Analysis, Analyzer, Finding
from .contrastive import (
    ContrastiveBase,
    ContrastiveError,
    ContrastiveItem,
    Match,
    get_base,
    load_base,
)
from .level import (
    LevelDecision,
    LevelEstimate,
    LevelTracker,
    Signals,
    estimate_level,
    partial_name,
)
from .modes import DrillPlanner, DrillTarget, drill_block
from .schedule import (
    GRADUATED_STREAK,
    INTERVALS_DAYS,
    Queued,
    Review,
    Scheduler,
    grade,
    interval_for,
    is_graduated,
)
from .turn import Analyzed, Chunk, Done, Failed, Leveled, TurnEvent, TurnRunner

__all__ = [
    "Analysis",
    "Analyzed",
    "Analyzer",
    "Chunk",
    "ContrastiveBase",
    "ContrastiveError",
    "ContrastiveItem",
    "Done",
    "DrillPlanner",
    "DrillTarget",
    "GRADUATED_STREAK",
    "INTERVALS_DAYS",
    "Failed",
    "Finding",
    "LevelDecision",
    "LevelEstimate",
    "LevelTracker",
    "Leveled",
    "Match",
    "Queued",
    "Review",
    "Scheduler",
    "Signals",
    "TurnEvent",
    "TurnRunner",
    "drill_block",
    "estimate_level",
    "get_base",
    "grade",
    "interval_for",
    "is_graduated",
    "load_base",
    "partial_name",
]
