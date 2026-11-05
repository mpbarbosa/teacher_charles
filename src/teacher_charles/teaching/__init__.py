"""A estrutura pedagógica.

Este pacote é o produto. O modelo é um componente substituível; o que faz o
Charles ser um professor e não um chat genérico mora aqui.
"""

from .contrastive import (
    ContrastiveBase,
    ContrastiveError,
    ContrastiveItem,
    Match,
    get_base,
    load_base,
)
from .turn import Chunk, Done, Failed, TurnEvent, TurnRunner

__all__ = [
    "Chunk",
    "ContrastiveBase",
    "ContrastiveError",
    "ContrastiveItem",
    "Done",
    "Failed",
    "Match",
    "TurnEvent",
    "TurnRunner",
    "get_base",
    "load_base",
]
