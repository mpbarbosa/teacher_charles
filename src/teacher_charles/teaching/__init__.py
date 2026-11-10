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
    span_key,
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
from .phonetics import Phonetics, PhoneticsUnavailable, build_phonetics
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
from .turn import Analyzed, Chunk, Done, Failed, Leveled, Phonemized, TurnEvent, TurnRunner

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
    "Phonemized",
    "Phonetics",
    "PhoneticsUnavailable",
    "Queued",
    "Review",
    "Scheduler",
    "Signals",
    "TurnEvent",
    "TurnRunner",
    "build_phonetics",
    "drill_block",
    "estimate_level",
    "get_base",
    "grade",
    "interval_for",
    "is_graduated",
    "load_base",
    "partial_name",
    "span_key",
]
