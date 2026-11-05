"""A estrutura pedagógica.

Este pacote é o produto. O modelo é um componente substituível; o que faz o
Charles ser um professor e não um chat genérico mora aqui.
"""

from .turn import Chunk, Done, Failed, TurnEvent, TurnRunner

__all__ = ["Chunk", "Done", "Failed", "TurnEvent", "TurnRunner"]
