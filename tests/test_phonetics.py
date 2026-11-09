"""A transcrição fonética que corre embaixo de toda fala.

O espeak é dependência base (ver o ``pyproject``), então estes testes rodam sem
peso nenhum baixado e **não** são de integração: o que eles exercitam é uma
tabela, não um modelo.
"""

from __future__ import annotations

import pytest

from teacher_charles.config import Settings
from teacher_charles.teaching.phonetics import (
    Phonetics,
    PhoneticsUnavailable,
    build_phonetics,
    partial_name,
)


def make_settings(**overrides) -> Settings:
    """Settings de teste, sem ler o ``.env`` do repositório (ver conftest)."""
    return Settings(_env_file=None, **overrides)


@pytest.fixture
def g2p() -> Phonetics:
    return Phonetics()


# --- O que sai ------------------------------------------------------------- #


def test_transcreve_em_ipa(g2p):
    ipa = g2p.transcribe("Hello")
    assert ipa
    # O símbolo é IPA, não a ortografia de volta.
    assert "həl" in ipa
    assert "Hello" not in ipa


def test_marca_tonica_por_default(g2p):
    """``ˈrecord`` e ``reˈcord`` são palavras diferentes — a tônica é a lição.

    É o oposto do gabarito do ``scripts/measure_phonemes.py``, e por motivo
    oposto: lá a tônica inventaria erro contra um reconhecedor que não a emite.
    """
    assert "ˈ" in g2p.transcribe("Tell me about your day")


def test_sem_tonica_quando_pedido():
    sem = Phonetics(with_stress=False).transcribe("Tell me about your day")
    assert "ˈ" not in sem and "ˌ" not in sem


def test_pontuacao_sobrevive(g2p):
    """A linha de IPA corre embaixo da linha de texto, e tem que acompanhá-la.

    Sem vírgula nem ponto de interrogação o aluno perde a correspondência entre
    as duas na primeira sentença longa.
    """
    ipa = g2p.transcribe("How old are you? I am fine, thanks.")
    assert "?" in ipa
    assert "," in ipa
    assert ipa.endswith(".")


def test_a_ortografia_inglesa_esconde_o_som(g2p):
    """*though*, *through* e *tough* rimam na escrita e não na boca."""
    though, through, tough = (g2p.transcribe(w) for w in ("though", "through", "tough"))
    assert len({though, through, tough}) == 3


def test_palavra_estrangeira_nao_vem_com_marca_de_lingua(g2p):
    """``language_switch='remove-flags'``: nada de ``(pt)...(en)`` na tela."""
    ipa = g2p.transcribe("I like pão de queijo")
    assert "(" not in ipa and ")" not in ipa


# --- O que não sai --------------------------------------------------------- #


def test_texto_vazio_volta_vazio(g2p):
    assert g2p.transcribe("") == ""
    assert g2p.transcribe("   \n ") == ""


def test_backend_que_nao_sobe_nao_derruba_o_turno(monkeypatch, g2p):
    """String vazia é resposta de primeira classe; exceção seria a aula parada."""

    def explode() -> None:
        raise PhoneticsUnavailable("espeak não carregou")

    monkeypatch.setattr(g2p, "_load", explode)

    assert g2p.transcribe("Hello") == ""
    assert g2p.last_error is not None
    assert "espeak" in g2p.last_error


def test_erro_inesperado_tambem_e_engolido(monkeypatch, g2p):
    """Qualquer estouro do espeak vira linha ausente, nunca turno perdido."""

    class Boom:
        def phonemize(self, _texts):
            raise RuntimeError("segfault do espeak")

    monkeypatch.setattr(g2p, "_load", lambda: Boom())

    assert g2p.transcribe("Hello") == ""
    assert g2p.last_error is not None


# --- Carga ----------------------------------------------------------------- #


def test_ready_nao_dispara_carga(g2p):
    """Mesma regra do ``voice/base.Loadable``: health check sem efeito colateral."""
    assert g2p.ready is False
    g2p.transcribe("Hello")
    assert g2p.ready is True


async def test_warm_up_carrega(g2p):
    await g2p.warm_up()
    assert g2p.ready is True
    assert g2p.last_error is None


async def test_warm_up_nunca_propaga(monkeypatch, g2p):
    def explode() -> None:
        raise PhoneticsUnavailable("peso ausente")

    monkeypatch.setattr(g2p, "_load", explode)

    await g2p.warm_up()  # não levanta
    assert g2p.ready is False
    assert g2p.last_error is not None


def test_cache_nao_repete_o_trabalho(monkeypatch, g2p):
    calls = []
    real = g2p._phonemize

    def counted(text: str) -> str:
        calls.append(text)
        return real(text)

    # Recria o cache em cima do contador: o original já embrulhou o método.
    from functools import lru_cache

    monkeypatch.setattr(g2p, "_cache", lru_cache(maxsize=8)(counted))

    assert g2p.transcribe("Hello") == g2p.transcribe("Hello")
    assert calls == ["Hello"]


# --- A fábrica e o partial ------------------------------------------------- #


def test_build_devolve_none_quando_desligado():
    assert build_phonetics(make_settings(phonetics_enabled=False)) is None


def test_build_respeita_a_configuracao():
    g2p = build_phonetics(make_settings(phonetics_language="en-gb", phonetics_stress=False))
    assert g2p is not None
    assert g2p.language == "en-gb"
    assert g2p.with_stress is False


def test_partial_so_existe_com_a_funcionalidade_ligada():
    assert partial_name(True) == "partials/app/phonetics"
    assert partial_name(False) is None


def test_o_partial_que_o_nome_promete_existe():
    """O professor só não inventa IPA se alguém contar a ele que a linha existe."""
    from teacher_charles.prompts import PromptLibrary

    prompts = PromptLibrary(make_settings().prompts_dir)
    assert prompts.exists(partial_name(True))
