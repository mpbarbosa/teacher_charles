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
from .turn import Analyzed, Chunk, Done, Failed, TurnEvent, TurnRunner

__all__ = [
    "Analysis",
    "Analyzed",
    "Analyzer",
    "Chunk",
    "ContrastiveBase",
    "ContrastiveError",
    "ContrastiveItem",
    "Done",
    "Failed",
    "Finding",
    "Match",
    "TurnEvent",
    "TurnRunner",
    "get_base",
    "load_base",
]
