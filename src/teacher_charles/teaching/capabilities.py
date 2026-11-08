"""Que produto o Charles está rodando — e o que ele não sabe fazer.

Até aqui a persona não dizia nada sobre o aplicativo, e o modelo respondia pelo
palpite genérico de um chatbot. Numa sessão real, perguntado se tinha microfone
para capturar a voz do aluno, o Charles respondeu que não tinha e sugeriu que o
aluno gravasse por fora — com o botão de push-to-talk na tela, ao lado da
própria frase.

O texto **varia** com ``settings.voice_enabled``, e por isso não pode ser
interpolado na persona: a persona é o prefixo que o Ollama cacheia, e valor que
muda ali jogaria o cache fora a cada turno. Entra então como mensagem de
sistema separada, no arranjo que ``partials/level/`` já estabeleceu (ver
``teaching/level.py``).

Vem logo depois da persona porque é o **mais estável** de tudo o que sobra: a
voz é decidida na subida do servidor e não muda enquanto ele está de pé — ao
contrário do nível, que muda a cada muitos turnos, e do modo, que o aluno troca
na tela. O cache é de prefixo, então o mais estável vem primeiro.

A escolha é um ``bool``, não o objeto ``Voice``: `teaching/` não importa
`voice/` (CLAUDE.md), e isto aqui não é exceção — o que o professor precisa
saber é se o botão existe, não qual motor está carregado.
"""

from __future__ import annotations


def partial_name(voice_enabled: bool) -> str:
    """O prompt de capacidades deste estado, como a ``PromptLibrary`` espera."""
    return f"partials/app/{'voice' if voice_enabled else 'text'}"


__all__ = ["partial_name"]
