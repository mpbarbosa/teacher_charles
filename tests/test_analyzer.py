"""Testes da correção estruturada (E4). Nenhum usa modelo de verdade.

Duas metades:

**A suíte de frases com erro conhecido** (`tests/fixtures/known_errors.yaml`) é
o critério de aceite da E4. Aqui ela exercita a parte determinística: a base tem
que levantar a regra esperada em toda frase errada e nenhuma nas corretas. É
portão, não medição — roda em milissegundos e quebra quando um padrão regride. A
medição contra o modelo de verdade é do `scripts/measure_analysis.py`, e o
resultado fica no plano §5.2.

**As garantias do analisador**, que não são confiadas ao prompt: regra citada
vem da base, `rule_id` inventado é descartado, span que o aluno não disse é
descartado, JSON inválido degrada em vez de quebrar o turno.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import yaml

from teacher_charles.config import REPO_ROOT
from teacher_charles.llm import FakeProvider
from teacher_charles.llm.base import ProviderUnavailable
from teacher_charles.prompts import PromptLibrary
from teacher_charles.teaching.analyzer import (
    MAX_FINDINGS,
    Analyzer,
    extract_json_object,
)
from teacher_charles.teaching.contrastive import load_base

BASE = load_base(REPO_ROOT / "data" / "contrastive")
PROMPTS = PromptLibrary(REPO_ROOT / "prompts")

SUITE = yaml.safe_load(
    (Path(__file__).parent / "fixtures" / "known_errors.yaml").read_text(encoding="utf-8")
)


def analyzer_for(reply: str | list[str], **kwargs) -> tuple[Analyzer, FakeProvider]:
    provider = FakeProvider(replies=reply)
    return Analyzer(provider=provider, prompts=PROMPTS, base=BASE, **kwargs), provider


def errors_json(*errors: dict) -> str:
    return json.dumps({"errors": list(errors)})


# --- A suíte de frases com erro conhecido ---------------------------------- #


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c["rules"][0] + ":" + c["text"][:24]) for c in SUITE["errors"]],
)
def test_the_base_catches_every_known_error(case):
    """Critério de aceite da E4, parte determinística."""
    found = set(BASE.rule_ids(case["text"]))
    missing = set(case["rules"]) - found
    assert not missing, f"a base não levantou {missing} em {case['text']!r} (achou {found})"


@pytest.mark.parametrize("text", [pytest.param(t, id=t[:34]) for t in SUITE["clean"]])
def test_correct_english_raises_no_rule(text):
    """Corrigir inglês certo é pior que deixar erro passar."""
    assert BASE.rule_ids(text) == ()


def test_the_suite_has_the_promised_size_and_spread():
    cases = SUITE["errors"]
    assert len(cases) >= 25, "a E4 prometeu ~25 frases com erro conhecido"
    categories = {rule.split(".")[0] for case in cases for rule in case["rules"]}
    assert categories == {"calque", "false-friend", "grammar"}
    # Toda regra citada na suíte tem que existir na base.
    for case in cases:
        for rule in case["rules"]:
            assert rule in BASE, f"{rule} não existe na base"


# --- Extração do JSON ------------------------------------------------------ #


def test_extracts_plain_json():
    assert extract_json_object('{"errors": []}') == {"errors": []}


def test_extracts_json_from_a_code_fence():
    raw = 'Here is the result:\n```json\n{"errors": [{"span": "x"}]}\n```\nHope it helps!'
    assert extract_json_object(raw) == {"errors": [{"span": "x"}]}


def test_extracts_json_after_a_think_block():
    """Modelo de raciocínio pensa antes, mesmo com think desligado."""
    raw = '<think>The student said... let me check.</think>\n{"errors": []}'
    assert extract_json_object(raw) == {"errors": []}


def test_ignores_trailing_garbage_after_the_object():
    raw = '{"errors": []}\n\nLet me know if you want more detail.'
    assert extract_json_object(raw) == {"errors": []}


def test_handles_braces_inside_strings():
    raw = '{"errors": [{"span": "the {weird} thing", "type": "other"}]}'
    assert extract_json_object(raw)["errors"][0]["span"] == "the {weird} thing"


@pytest.mark.parametrize(
    "raw",
    ["", "no json here", "{not even close", '["errors"]', "{}{"],
)
def test_unrecoverable_text_yields_nothing(raw):
    result = extract_json_object(raw)
    assert result is None or result == {}


# --- As garantias ---------------------------------------------------------- #


async def test_curated_rule_overrides_what_the_model_said():
    """A explicação vem da base, não da improvisação do modelo.

    É o problema nº 1 da análise inicial tratado em código: um 3B "corrigiu"
    *I have 30 years* para *I have thirty years*. Com a regra curada casando no
    mesmo trecho, a explicação deixa de ser voto do modelo.
    """
    analyzer, _ = analyzer_for(
        errors_json(
            {
                "span": "I have 30 years",
                "type": "grammar",
                "correction": "I am 30 years old",
                "rule_id": None,
                "explanation": "In English we spell numbers out.",
            }
        )
    )
    analysis = await analyzer.analyze("I have 30 years and I like it")

    (finding,) = analysis.findings
    assert finding.rule_id == "calque.age-have"
    assert finding.explanation == BASE.get("calque.age-have").explanation
    assert finding.error_type == "calque"
    assert finding.from_curated_rule


async def test_invented_rule_id_is_dropped():
    """Id fantasma agregaria o error log por uma regra que não existe.

    Com ``require_curated_rule=False``, que é como se vê o que sobra do achado
    depois de o id inventado cair.
    """
    analyzer, _ = analyzer_for(
        errors_json(
            {
                "span": "very good",
                "type": "vocabulary",
                "correction": "excellent",
                "rule_id": "calque.totally-made-up",
                "explanation": "Stronger word.",
            }
        ),
        require_curated_rule=False,
    )
    analysis = await analyzer.analyze("It was very good")

    (finding,) = analysis.findings
    assert finding.rule_id is None
    assert not finding.from_curated_rule
    assert finding.explanation == "Stronger word."


async def test_existing_rule_id_claimed_by_the_model_is_kept():
    """Se o id existe na base, vale — mesmo sem o matcher ter casado ali.

    A frase tem um decalque em outro trecho de propósito: é ele que faz a
    análise rodar. Sem candidato nenhum, a política dispensa a chamada.
    """
    analyzer, _ = analyzer_for(
        errors_json(
            {
                "span": "I go",
                "type": "grammar",
                "correction": "I went",
                "rule_id": "grammar.did-base-form",
                "explanation": "Use the past.",
            }
        )
    )
    analysis = await analyzer.analyze("It depends of you. Yesterday I go to the beach")

    (finding,) = analysis.findings
    assert finding.rule_id == "grammar.did-base-form"
    assert finding.explanation == BASE.get("grammar.did-base-form").explanation


async def test_span_the_student_never_said_is_dropped():
    """Correção sobre palavra inventada desorienta mais que correção nenhuma."""
    analyzer, _ = analyzer_for(
        errors_json(
            {
                "span": "I have went to Paris",
                "type": "grammar",
                "correction": "I went to Paris",
                "explanation": "Simple past.",
            }
        )
    )
    analysis = await analyzer.analyze("I visited Paris last year")
    assert analysis.findings == ()
    assert not analysis.degraded, "span inválido não é falha de formato"


async def test_span_is_recovered_with_the_student_spelling():
    """O modelo recopia o trecho e erra a caixa; isso não custa a correção."""
    analyzer, _ = analyzer_for(
        errors_json(
            {
                "span": "i have 30 YEARS",
                "type": "calque",
                "correction": "I am 30 years old",
                "explanation": "ignorada",
            }
        )
    )
    (finding,) = (await analyzer.analyze("Hello! I have 30 Years, you?")).findings
    # O span gravado é o do aluno, não o do modelo: é o que a UI risca na tela.
    assert finding.span == "I have 30 Years"


async def test_finding_without_correction_is_dropped():
    analyzer, _ = analyzer_for(
        errors_json({"span": "I have 30 years", "correction": "", "explanation": "x"})
    )
    assert (await analyzer.analyze("I have 30 years")).findings == ()


async def test_correction_identical_to_the_span_is_dropped():
    """Medido no 3B: ele devolve "correção" igual ao original.

    Mostrar isso ao aluno é pior que não corrigir — ele procura a diferença que
    não existe. A comparação ignora caixa, espaço e pontuação de borda.
    """
    analyzer, _ = analyzer_for(
        errors_json(
            {
                "span": "a question about the exercise?",
                "type": "other",
                "correction": "A question about the exercise",
                "explanation": "Something is off here.",
            }
        )
    )
    assert (await analyzer.analyze("Can I ask a question about the exercise?")).findings == ()


async def test_real_correction_with_the_same_words_is_kept():
    analyzer, _ = analyzer_for(
        errors_json(
            {
                "span": "I have 30 years",
                "type": "calque",
                "correction": "I am 30 years old",
                "explanation": "x",
            }
        )
    )
    assert len((await analyzer.analyze("I have 30 years")).findings) == 1


def test_the_prompt_forbids_style_rewrites():
    """Medido: sem isso o 3B reescreve inglês correto para soar melhor."""
    text = PROMPTS.get("analysis")
    assert "Style is not a mistake" in text
    assert "shortest phrase" in text


async def test_findings_are_capped_and_curated_first():
    """A persona pede uma correção por turno, duas no máximo."""
    analyzer, _ = analyzer_for(
        errors_json(
            {
                "span": "very good",
                "type": "vocabulary",
                "correction": "excellent",
                "explanation": "Stronger word.",
            },
            {
                "span": "people is",
                "type": "grammar",
                "correction": "people are",
                "explanation": "ignorada",
            },
            {
                "span": "It depends of",
                "type": "grammar",
                "correction": "It depends on",
                "explanation": "ignorada",
            },
        )
    )
    analysis = await analyzer.analyze("It depends of the day. People is very good here")

    assert len(analysis.findings) == MAX_FINDINGS
    # As duas que sobraram são as que têm regra curada por trás.
    assert all(f.from_curated_rule for f in analysis.findings)
    assert set(analysis.rule_ids) == {"calque.depend-of", "grammar.people-plural"}


async def test_duplicate_spans_are_collapsed():
    error = {
        "span": "I have 30 years",
        "type": "calque",
        "correction": "I am 30 years old",
        "explanation": "x",
    }
    analyzer, _ = analyzer_for(errors_json(error, dict(error, span="i have 30 years")))
    assert len((await analyzer.analyze("I have 30 years")).findings) == 1


async def test_unknown_error_type_becomes_other():
    analyzer, _ = analyzer_for(
        errors_json(
            {
                "span": "gonna",
                "type": "register-and-style",
                "correction": "going to",
                "explanation": "Write it in full.",
            }
        ),
        require_curated_rule=False,
    )
    (finding,) = (await analyzer.analyze("I gonna travel")).findings
    assert finding.error_type == "other"


async def test_extra_fields_in_the_json_are_ignored():
    """Deixa o prompt da E5 pedir level_signals sem mexer no analisador."""
    raw = json.dumps(
        {
            "errors": [
                {
                    "span": "I have 30 years",
                    "type": "calque",
                    "correction": "I am 30 years old",
                    "explanation": "x",
                    "confidence": 0.9,
                }
            ],
            "level_signals": {"estimated_cefr": "B1"},
        }
    )
    analyzer, _ = analyzer_for(raw)
    assert len((await analyzer.analyze("I have 30 years")).findings) == 1


async def test_malformed_entry_does_not_lose_the_good_ones():
    raw = json.dumps(
        {
            "errors": [
                "isto não é um objeto",
                {"type": "calque"},
                {
                    "span": "It depends of",
                    "type": "calque",
                    "correction": "It depends on",
                    "explanation": "x",
                },
            ]
        }
    )
    analyzer, _ = analyzer_for(raw)
    analysis = await analyzer.analyze("It depends of the weather")
    assert analysis.rule_ids == ("calque.depend-of",)
    assert not analysis.degraded


# --- A política: sem regra curada, não vai para o aluno -------------------- #
#
# Medido num 3B (plano §5.2): 12 falsos positivos em 18 frases corretas, nenhum
# citando regra, vários ensinando erro — "There have been" -> "has been",
# "depends on" -> "depend on". E 24 de 24 acertos citavam a regra esperada.
# Exigir a regra eliminou os 12 sem custar nenhum dos 24.


async def test_correction_without_a_curated_rule_is_not_shown():
    analyzer, _ = analyzer_for(
        errors_json(
            {
                "span": "There have been",
                "type": "grammar",
                "correction": "There has been",
                "explanation": "Agreement.",
            }
        )
    )
    # O decalque no começo é o que faz a análise rodar; o achado do modelo é
    # sobre outro trecho, e é esse que a política barra.
    analysis = await analyzer.analyze("I have 30 years. There have been many changes")

    assert analysis.findings == ()
    # Contado, não só descartado: é a medida do que a base ainda não cobre.
    assert analysis.unverified == 1
    assert not analysis.degraded


async def test_the_policy_can_be_turned_off():
    """Um modelo maior pode merecer confiança sem regra — medindo antes."""
    analyzer, _ = analyzer_for(
        errors_json(
            {
                "span": "There have been",
                "type": "grammar",
                "correction": "There has been",
                "explanation": "Agreement.",
            }
        ),
        require_curated_rule=False,
    )
    analysis = await analyzer.analyze("There have been many changes since March")

    (finding,) = analysis.findings
    assert finding.rule_id is None
    assert analysis.unverified == 0
    assert not analysis.skipped


async def test_the_policy_keeps_the_rule_backed_finding_of_a_mixed_answer():
    analyzer, _ = analyzer_for(
        errors_json(
            {
                "span": "this morning",
                "type": "other",
                "correction": "that morning",
                "explanation": "Sem regra nenhuma.",
            },
            {
                "span": "I lost the bus",
                "type": "calque",
                "correction": "I missed the bus",
                "explanation": "ignorada",
            },
        )
    )
    analysis = await analyzer.analyze("I lost the bus this morning")

    assert analysis.rule_ids == ("calque.lose-the-bus",)
    assert analysis.unverified == 1


# --- Degradação ------------------------------------------------------------ #


async def test_invalid_json_is_retried_with_a_stricter_instruction():
    analyzer, provider = analyzer_for(["Sure! Let me help:", errors_json()])
    analysis = await analyzer.analyze("I have 30 years")

    assert analysis.attempts == 2
    assert not analysis.degraded
    # A retentativa acrescenta a correção, não reescreve a instrução: prompt
    # diferente jogaria o cache de prefixo fora.
    assert provider.call_count == 2
    assert "not valid JSON" in provider.calls[1].messages[-1].content
    assert provider.calls[0].messages[0].content == provider.calls[1].messages[0].content


async def test_two_failures_degrade_instead_of_raising():
    """O turno do aluno vale mais que a correção dele."""
    analyzer, provider = analyzer_for("I could not find any JSON today.")
    analysis = await analyzer.analyze("I have 30 years")

    assert analysis.degraded
    assert analysis.findings == ()
    assert analysis.attempts == 2
    assert "JSON" in analysis.detail
    # Mesmo degradada, a análise reporta o que a base levantou — é o que a
    # medição da §5.2 compara.
    assert analysis.candidates == ("calque.age-have",)


async def test_provider_failure_degrades_without_retrying():
    """Ollama fora do ar não vai responder melhor na segunda tentativa."""
    provider = FakeProvider(error=ProviderUnavailable("sem Ollama"))
    analyzer = Analyzer(provider=provider, prompts=PROMPTS, base=BASE)

    analysis = await analyzer.analyze("I have 30 years")
    assert analysis.degraded
    assert analysis.attempts == 1
    assert "sem Ollama" in analysis.detail


async def test_timeout_degrades():
    class Slow(FakeProvider):
        async def complete(self, messages, options=None):  # type: ignore[override]
            await asyncio.sleep(5)
            return errors_json()

    analyzer = Analyzer(provider=Slow(), prompts=PROMPTS, base=BASE, timeout_s=0.05)
    analysis = await analyzer.analyze("I have 30 years")

    assert analysis.degraded
    assert "passou de" in analysis.detail


async def test_empty_text_does_not_call_the_model():
    analyzer, provider = analyzer_for(errors_json())
    analysis = await analyzer.analyze("   ")
    assert provider.call_count == 0
    assert not analysis.degraded and analysis.findings == ()


# --- O prompt da análise --------------------------------------------------- #


async def test_candidates_go_into_the_prompt():
    analyzer, provider = analyzer_for(errors_json())
    await analyzer.analyze("I have 30 years")

    user_message = provider.last_call.messages[-1].content
    assert "I have 30 years" in user_message
    assert "calque.age-have" in user_message
    assert BASE.get("calque.age-have").explanation in user_message


async def test_no_candidates_means_no_rules_block():
    """Sem candidato, nenhum bloco de regras no prompt.

    Com a política ligada esta chamada nem acontece; desligada, ela acontece e
    o prompt tem que sair sem bloco de regras.
    """
    analyzer, provider = analyzer_for(errors_json(), require_curated_rule=False)
    await analyzer.analyze("I am 30 years old and I like my job")
    assert "Rules that may apply" not in provider.last_call.messages[-1].content


async def test_candidates_are_capped():
    analyzer, provider = analyzer_for(errors_json(), max_candidates=2)
    await analyzer.analyze("I have 30 years, it depends of you, explain me this, people is nice")

    block = provider.last_call.messages[-1].content
    assert block.count("\n- ") == 2


async def test_analysis_is_deterministic_and_short():
    """Análise não é criação: temperatura 0 e teto de tokens."""
    analyzer, provider = analyzer_for(errors_json())
    await analyzer.analyze("I have 30 years")

    options = provider.last_call.options
    assert options.temperature == 0.0
    assert options.max_tokens is not None and options.max_tokens <= 512
    # Raciocinar antes de responder custa segundos que o turno não tem.
    assert options.think is False


def test_the_real_analysis_prompt_asks_for_json_only():
    text = PROMPTS.get("analysis")
    assert '"errors"' in text
    assert "JSON only" in text
    # Decisão 1.1: o que chega ao aluno é inglês.
    assert "in English" in text


async def test_level_reaches_the_prompt():
    """A E5 vai ajustar por nível; o campo já atravessa."""
    analyzer, provider = analyzer_for(errors_json())
    await analyzer.analyze("I have 30 years", cefr_level="B2")
    assert "CEFR B2" in provider.last_call.messages[-1].content


async def test_the_rule_is_chosen_by_where_the_span_actually_is():
    """Com o mesmo erro duas vezes, vale a regra que casa no trecho apontado.

    Procurar a posição com um `find` depois pegaria a primeira ocorrência, que
    nem sempre é a que o modelo apontou — e aí a correção citaria a regra de
    outro trecho da frase.
    """
    text = "It depends of you, and people is nice here"
    analyzer, _ = analyzer_for(
        errors_json(
            {
                "span": "people is",
                "type": "grammar",
                "correction": "people are",
                "explanation": "ignorada",
            }
        )
    )
    (finding,) = (await analyzer.analyze(text)).findings
    assert finding.rule_id == "grammar.people-plural"
    assert finding.span == "people is"


async def test_no_candidate_means_no_model_call_at_all():
    """Se nada poderia chegar ao aluno, não se pergunta ao modelo.

    Com a política exigindo regra curada e a base sem candidato, a análise não
    tem como produzir correção. Chamar o modelo ali custaria ~9s e, pior,
    jogaria o prompt da conversa fora do cache de prefixo do Ollama — o turno
    seguinte pagaria a persona inteira de novo (plano §5.2).
    """
    analyzer, provider = analyzer_for(errors_json())
    analysis = await analyzer.analyze("I am 30 years old and I work in marketing")

    assert provider.call_count == 0
    assert analysis.skipped
    assert analysis.findings == ()
    assert not analysis.degraded


async def test_the_model_is_still_asked_when_the_base_has_a_candidate():
    analyzer, provider = analyzer_for(errors_json())
    analysis = await analyzer.analyze("I have 30 years")

    assert provider.call_count == 1
    assert not analysis.skipped


async def test_without_the_policy_the_model_is_always_asked():
    """Política desligada: aí a opinião do modelo sozinho conta, e é pedida."""
    analyzer, provider = analyzer_for(errors_json(), require_curated_rule=False)
    analysis = await analyzer.analyze("I am 30 years old and I work in marketing")

    assert provider.call_count == 1
    assert not analysis.skipped
