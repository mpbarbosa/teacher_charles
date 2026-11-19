"""Testes dos providers (E1). Nenhum precisa de modelo baixado."""

from __future__ import annotations

import json

import httpx
import pytest

from teacher_charles.config import Settings
from teacher_charles.llm import (
    FakeProvider,
    GenerationOptions,
    Message,
    ModelNotFound,
    OllamaProvider,
    OpenAICompatProvider,
    ProviderBadResponse,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
    UnknownProvider,
    build_provider,
    default_options,
)

HELLO = [Message(role="user", content="I have 30 years")]


# --- O contrato comum ------------------------------------------------------ #


async def test_complete_is_built_from_stream():
    """``complete`` não deve ter implementação própria que possa divergir."""
    provider = FakeProvider(replies="I am 30 years old.", chunk_size=3)
    assert await provider.complete(HELLO) == "I am 30 years old."


async def test_stream_yields_chunks_not_one_blob():
    provider = FakeProvider(replies="abcdefgh", chunk_size=3)
    chunks = [c async for c in provider.stream(HELLO)]
    assert chunks == ["abc", "def", "gh"]


async def test_empty_message_list_is_rejected():
    with pytest.raises(ValueError):
        [c async for c in FakeProvider().stream([])]


async def test_fake_records_calls_and_options():
    provider = FakeProvider(replies=["first", "second"])
    options = GenerationOptions(temperature=0.1, max_tokens=7)

    assert await provider.complete(HELLO, options) == "first"
    assert await provider.complete(HELLO) == "second"
    # Esgotadas as respostas, a última se repete.
    assert await provider.complete(HELLO) == "second"

    assert provider.call_count == 3
    assert provider.calls[0].options.temperature == 0.1
    assert provider.calls[0].options.max_tokens == 7


async def test_fake_can_raise_for_error_paths():
    provider = FakeProvider(error=ProviderUnavailable("boom"))
    with pytest.raises(ProviderUnavailable):
        await provider.complete(HELLO)


# --- Ollama: streaming ----------------------------------------------------- #


def ndjson(*chunks: str, done_separately: bool = True) -> bytes:
    """Monta um corpo igual ao que o Ollama 0.35 devolve."""
    lines = [
        json.dumps({"model": "m", "message": {"role": "assistant", "content": c}, "done": False})
        for c in chunks
    ]
    if done_separately:
        lines.append(
            json.dumps(
                {
                    "model": "m",
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": "stop",
                    "eval_count": len(chunks),
                }
            )
        )
    return ("\n".join(lines) + "\n").encode()


# The lines that decide, out of two real templates. What separates them is one
# variable: the first reads ``$.Think`` and so can be told to stop reasoning;
# the second never does, and opens ``<think>`` whatever the request says.
#
# The second one is kept with its ``.Thinking`` and ``$.IsThinkSet`` on
# purpose. Both are in every thinking template, neither is the flag, and a
# check loose enough to take them for it would send ``think: false`` to
# exactly the model that leaks.
HYBRID_TEMPLATE = """\
{{- if and $.IsThinkSet (eq $i $lastUserIdx) }}
   {{- if $.Think -}}
      {{- " "}}/think
   {{- else -}}
      {{- " "}}/no_think
   {{- end -}}
{{- end }}<|im_end|>
{{ if and $.IsThinkSet (not $.Think) -}}
<think>

</think>

{{ end -}}
"""  # qwen3:8b

THINKING_ONLY_TEMPLATE = """\
{{ if (and $.IsThinkSet (and .Thinking (or $last (gt $i $lastUserIdx)))) -}}
<think>{{ .Thinking }}</think>
{{ end -}}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
<think>
{{ end }}
"""  # qwen3:4b, which is Qwen3-4B-Thinking-2507

PLAIN_TEMPLATE = """\
{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""  # qwen2.5: no reasoning at all

HYBRID = {"template": HYBRID_TEMPLATE, "capabilities": ["completion", "tools", "thinking"]}
THINKING_ONLY = {
    "template": THINKING_ONLY_TEMPLATE,
    "capabilities": ["completion", "tools", "thinking"],
}
PLAIN = {"template": PLAIN_TEMPLATE, "capabilities": ["completion", "tools"]}

# What ``/api/show`` really says about ``qwen3:4b-instruct``: a Jinja template,
# so there is no ``.Think`` to find, and ``thinking`` among the capabilities
# of a model that does not reason.
INSTRUCT_TEMPLATE = """\
{%- if message.role == "assistant" %}
    {%- set content = content.split('</think>')[-1].lstrip('\\n') %}
{%- endif %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
{%- endif %}
"""
INSTRUCT = {"template": INSTRUCT_TEMPLATE, "capabilities": ["tools", "thinking", "completion"]}


def ollama_with(handler, show=HYBRID) -> OllamaProvider:
    """A provider whose server is ``handler``.

    ``/api/show`` is answered here and never reaches ``handler``, so a test
    about the chat request keeps seeing chat requests only. ``show`` is the
    JSON the server would return, or a callable for the tests that are about
    the lookup itself.
    """

    def route(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return show(request) if callable(show) else httpx.Response(200, json=show)
        return handler(request)

    return OllamaProvider(
        model="qwen3:4b",
        client=httpx.AsyncClient(transport=httpx.MockTransport(route)),
    )


async def test_ollama_streams_ndjson():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["model"] == "qwen3:4b"
        assert body["messages"] == [{"role": "user", "content": "I have 30 years"}]
        return httpx.Response(200, content=ndjson("I am ", "30 ", "years old."))

    provider = ollama_with(handler)
    chunks = [c async for c in provider.stream(HELLO)]
    assert chunks == ["I am ", "30 ", "years old."]
    assert await provider.complete(HELLO) == "I am 30 years old."


async def test_ollama_stops_at_done_and_ignores_trailing_lines():
    """O ``done: true`` encerra o stream; nada depois dele deve ser lido."""
    body = (
        ndjson("hello")
        + json.dumps({"message": {"content": " LIXO DEPOIS DO DONE"}, "done": False}).encode()
    )

    provider = ollama_with(lambda r: httpx.Response(200, content=body))
    assert await provider.complete(HELLO) == "hello"


async def test_ollama_tolerates_blank_lines():
    body = (
        b'\n\n{"message":{"content":"ok"},"done":false}\n\n{"message":{"content":""},"done":true}\n'
    )
    provider = ollama_with(lambda r: httpx.Response(200, content=body))
    assert await provider.complete(HELLO) == "ok"


async def test_ollama_404_is_model_not_found_with_instruction():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'qwen3:4b' not found"})

    with pytest.raises(ModelNotFound) as caught:
        await ollama_with(handler).complete(HELLO)

    # A mensagem tem que dizer o que fazer, não só o que houve.
    assert "ollama pull qwen3:4b" in str(caught.value)


async def test_ollama_500_is_generic_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "out of memory"})

    with pytest.raises(ProviderError) as caught:
        await ollama_with(handler).complete(HELLO)
    assert not isinstance(caught.value, ModelNotFound)
    assert "out of memory" in str(caught.value)


async def test_ollama_connect_error_is_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(ProviderUnavailable) as caught:
        await ollama_with(handler).complete(HELLO)
    assert "está de pé" in str(caught.value)


async def test_ollama_timeout_is_timeout_not_unavailable():
    """Distinção que importa: "não alcanço" e "demorou" têm remédios opostos."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(ProviderTimeout) as caught:
        await ollama_with(handler).complete(HELLO)
    assert "TC_REQUEST_TIMEOUT_S" in str(caught.value)


async def test_ollama_error_midstream_is_raised():
    """HTTP 200 já foi enviado e o erro chega no meio do NDJSON."""
    body = (
        json.dumps({"message": {"content": "I am "}, "done": False})
        + "\n"
        + json.dumps({"error": "context canceled"})
        + "\n"
    )
    provider = ollama_with(lambda r: httpx.Response(200, content=body.encode()))
    with pytest.raises(ProviderError, match="context canceled"):
        await provider.complete(HELLO)


async def test_ollama_garbage_line_is_bad_response():
    provider = ollama_with(lambda r: httpx.Response(200, content=b"nao e json\n"))
    with pytest.raises(ProviderBadResponse):
        await provider.complete(HELLO)


# --- Ollama: tradução das opções ------------------------------------------- #


async def test_ollama_translates_options():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=ndjson("ok"))

    options = GenerationOptions(temperature=0.2, max_tokens=64, stop=("</end>",), think=False)
    await ollama_with(handler).complete(HELLO, options)

    assert seen["options"]["temperature"] == 0.2
    assert seen["options"]["num_predict"] == 64
    assert seen["options"]["stop"] == ["</end>"]
    assert seen["think"] is False


async def test_ollama_omits_think_when_unset():
    """Sem decisão explícita, o campo não vai — o default do servidor vale."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=ndjson("ok"))

    await ollama_with(handler).complete(HELLO, GenerationOptions())
    assert "think" not in seen
    assert "num_predict" not in seen["options"]


# --- Ollama: `think: false` only where the model obeys it ------------------ #
#
# Measured on Ollama 0.35 with the default model: `think: false` did not stop
# qwen3:4b from reasoning, it stopped the server from separating the reasoning
# out, and the student read all of it. These are the tests that go red if the
# flag is ever sent on trust again.


def recording(seen: list[dict], reply: str = "ok"):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=ndjson(reply))

    return handler


THINK_OFF = GenerationOptions(think=False)


async def test_ollama_keeps_think_false_from_a_model_that_ignores_it(caplog):
    """The leak itself: this template opens ``<think>`` regardless.

    Leaving the field out is what makes the server hide the reasoning. The
    warning is part of the contract too -- the operator asked for reasoning
    off and is not getting it, and silence would let the latency pass for a
    slow machine.
    """
    seen: list[dict] = []
    provider = ollama_with(recording(seen), show=THINKING_ONLY)

    with caplog.at_level("WARNING", logger="teacher_charles.llm.ollama"):
        await provider.complete(HELLO, THINK_OFF)

    assert "think" not in seen[0]
    assert "cannot be turned off" in caplog.text


async def test_ollama_sends_think_false_to_a_template_that_reads_it(caplog):
    """The other half: the caution must not cost the models where it works."""
    seen: list[dict] = []
    provider = ollama_with(recording(seen), show=HYBRID)

    with caplog.at_level("WARNING", logger="teacher_charles.llm.ollama"):
        await provider.complete(HELLO, THINK_OFF)

    assert seen[0]["think"] is False
    assert caplog.text == ""


async def test_ollama_says_nothing_about_a_model_that_does_not_reason(caplog):
    """No flag to read and nothing to hide: leaving the field out is free,
    and warning about reasoning that does not exist would be noise."""
    seen: list[dict] = []
    provider = ollama_with(recording(seen), show=PLAIN)

    with caplog.at_level("WARNING", logger="teacher_charles.llm.ollama"):
        await provider.complete(HELLO, THINK_OFF)

    assert "think" not in seen[0]
    assert caplog.text == ""


async def test_ollama_does_not_take_the_capability_list_for_reasoning(caplog):
    """``thinking`` in the capabilities is not a model that reasons.

    Measured on ``qwen3:4b-instruct``: it lists the capability and answers in
    two tokens with the field sent or left out. The warning tells the operator
    that reasoning cannot be turned off, so on this model -- the default -- it
    would be false at every startup. Only a template that opens the block
    itself proves it.
    """
    seen: list[dict] = []
    provider = ollama_with(recording(seen), show=INSTRUCT)

    with caplog.at_level("WARNING", logger="teacher_charles.llm.ollama"):
        await provider.complete(HELLO, THINK_OFF)

    assert "think" not in seen[0]
    assert caplog.text == ""


async def test_ollama_asks_how_the_model_takes_think_only_once():
    """Analysis and warm-up both pass ``think=False`` on every call; a lookup
    per call would be a round trip on the path to the first word."""
    lookups: list[dict] = []

    def show(request: httpx.Request) -> httpx.Response:
        lookups.append(json.loads(request.content))
        return httpx.Response(200, json=THINKING_ONLY)

    seen: list[dict] = []
    provider = ollama_with(recording(seen), show=show)
    for _ in range(3):
        await provider.complete(HELLO, THINK_OFF)

    assert lookups == [{"model": "qwen3:4b"}]
    assert all("think" not in payload for payload in seen)


async def test_ollama_does_not_ask_when_think_is_not_being_turned_off():
    """Only ``False`` is the dangerous value, so only ``False`` pays for the
    lookup. ``True`` and unset go out exactly as they came in."""

    def show(request: httpx.Request) -> httpx.Response:
        raise AssertionError("nothing here needed /api/show")

    seen: list[dict] = []
    provider = ollama_with(recording(seen), show=show)
    await provider.complete(HELLO, GenerationOptions(think=True))
    await provider.complete(HELLO, GenerationOptions())

    assert seen[0]["think"] is True
    assert "think" not in seen[1]


@pytest.mark.parametrize(
    "broken",
    [
        httpx.Response(500, json={"error": "boom"}),
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, json=["not", "an", "object"]),
        httpx.Response(200, json={"capabilities": ["thinking"]}),
    ],
    ids=["http-500", "not-json", "not-an-object", "no-template"],
)
async def test_ollama_treats_an_unreadable_answer_as_a_model_that_ignores_think(broken):
    """Unknown lands on the side that cannot leak."""
    seen: list[dict] = []
    provider = ollama_with(recording(seen), show=lambda request: broken)

    assert await provider.complete(HELLO, THINK_OFF) == "ok"
    assert "think" not in seen[0]


async def test_ollama_asks_again_after_a_lookup_that_failed():
    """A server that was down for the first call is not written off as a
    model that ignores the flag for the rest of the process."""
    answers = [httpx.Response(503, json={"error": "loading"}), httpx.Response(200, json=HYBRID)]

    seen: list[dict] = []
    provider = ollama_with(recording(seen), show=lambda request: answers.pop(0))
    await provider.complete(HELLO, THINK_OFF)
    await provider.complete(HELLO, THINK_OFF)

    assert "think" not in seen[0]
    assert seen[1]["think"] is False


async def test_ollama_lookup_failure_leaves_the_chat_error_typed():
    """With the server down the lookup fails first. What the caller gets must
    still be the chat request's own ``ProviderUnavailable``, not an httpx
    error that escaped from a question the caller never asked."""

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(ProviderUnavailable):
        await ollama_with(down, show=down).complete(HELLO, THINK_OFF)


async def test_ollama_never_relays_what_the_server_set_aside_as_thinking():
    """What the lookup above relies on: with the field left out the server
    puts the reasoning in ``message.thinking``, and that never leaves here."""
    lines = [
        {"message": {"role": "assistant", "content": "", "thinking": "The user asks "}},
        {"message": {"role": "assistant", "content": "", "thinking": "for a colour."}},
        {"message": {"role": "assistant", "content": "blue"}},
        {"message": {"role": "assistant", "content": ""}, "done": True},
    ]
    body = "".join(json.dumps(line) + "\n" for line in lines).encode()
    provider = ollama_with(lambda request: httpx.Response(200, content=body))

    assert [c async for c in provider.stream(HELLO)] == ["blue"]


# --- OpenAI-compatível: SSE ------------------------------------------------ #


def sse(*contents: str, with_done: bool = True) -> bytes:
    events = [
        "data: " + json.dumps({"choices": [{"delta": {"content": c}, "index": 0}]})
        for c in contents
    ]
    if with_done:
        events.append("data: [DONE]")
    return ("\n\n".join(events) + "\n\n").encode()


def openai_with(handler, **kwargs) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        model="gpt-x",
        api_key="sk-test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


async def test_openai_streams_sse():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, content=sse("I am ", "30."))

    assert await openai_with(handler).complete(HELLO) == "I am 30."


async def test_openai_stops_at_done_sentinel():
    body = sse("kept", with_done=False) + b"\ndata: [DONE]\n\n" + sse("descartado")
    assert await openai_with(lambda r: httpx.Response(200, content=body)).complete(HELLO) == "kept"


async def test_openai_ignores_keepalive_and_empty_deltas():
    body = (
        b": keep-alive\n\n"
        b'data: {"choices":[{"delta":{}}]}\n\n'
        b'data: {"choices":[]}\n\n'
        b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    assert await openai_with(lambda r: httpx.Response(200, content=body)).complete(HELLO) == "ok"


async def test_openai_sends_bearer_token():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, content=sse("ok"))

    # O header é posto no cliente que o provider cria, então aqui construímos
    # um sem injetar cliente para exercitar esse caminho.
    provider = OpenAICompatProvider(model="gpt-x", api_key="sk-secret")
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer sk-secret"},
    )
    await provider.complete(HELLO)
    assert seen["auth"] == "Bearer sk-secret"


@pytest.mark.parametrize(
    ("status", "needle"),
    [
        (401, "TC_OPENAI_API_KEY"),
        (403, "TC_OPENAI_API_KEY"),
        (429, "rate limit"),
        (500, "HTTP 500"),
    ],
)
async def test_openai_error_statuses(status, needle):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "nope"}})

    with pytest.raises(ProviderError) as caught:
        await openai_with(handler).complete(HELLO)
    assert needle in str(caught.value)


async def test_openai_404_is_model_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "no such model"}})

    with pytest.raises(ModelNotFound):
        await openai_with(handler).complete(HELLO)


async def test_openai_ignores_think_option():
    """O contrato diz que opção não suportada é ignorada, não é erro."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=sse("ok"))

    await openai_with(handler).complete(HELLO, GenerationOptions(think=True))
    assert "think" not in seen


# --- A fábrica ------------------------------------------------------------- #


def settings_for(tmp_path, **overrides) -> Settings:
    base = {"db_path": tmp_path / "f.db"}
    base.update(overrides)
    return Settings(_env_file=None, **base)


def test_build_provider_defaults_to_ollama(tmp_path):
    provider = build_provider(settings_for(tmp_path))
    assert isinstance(provider, OllamaProvider)
    assert provider.model == "qwen3:4b-instruct"
    assert provider.name == "ollama"


def test_build_provider_openai_requires_key(tmp_path):
    with pytest.raises(UnknownProvider, match="TC_OPENAI_API_KEY"):
        build_provider(settings_for(tmp_path, provider="openai_compat"))


def test_build_provider_openai_with_key(tmp_path):
    provider = build_provider(
        settings_for(tmp_path, provider="openai_compat", openai_api_key="sk-x")
    )
    assert isinstance(provider, OpenAICompatProvider)


def test_build_provider_fake(tmp_path):
    assert isinstance(build_provider(settings_for(tmp_path, provider="fake")), FakeProvider)


def test_build_provider_unknown_names_the_options(tmp_path):
    with pytest.raises(UnknownProvider) as caught:
        build_provider(settings_for(tmp_path, provider="carrier_pigeon"))
    message = str(caught.value)
    assert "ollama" in message and "openai_compat" in message and "fake" in message


def test_default_options_carries_think_from_settings(tmp_path):
    assert default_options(settings_for(tmp_path)).think is None
    assert default_options(settings_for(tmp_path, llm_think=False)).think is False
    assert default_options(settings_for(tmp_path), temperature=0.0).temperature == 0.0


# --- Ciclo de vida do cliente --------------------------------------------- #


async def test_provider_closes_client_it_owns():
    provider = OllamaProvider(model="m")
    assert not provider._client.is_closed
    await provider.aclose()
    assert provider._client.is_closed


async def test_provider_leaves_injected_client_alone():
    """Cliente injetado é de quem injetou; fechá-lo surpreenderia o dono."""
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    provider = OllamaProvider(model="m", client=client)
    await provider.aclose()
    assert not client.is_closed
    await client.aclose()


# --- Métricas do servidor (E7) --------------------------------------------- #


def test_ollama_records_the_metrics_of_the_done_line():
    """Sem estas métricas a E7 não teria como explicar a latência.

    A pergunta "o primeiro turno custou carga de modelo, avaliação de prompt ou
    evicção de cache?" só o servidor responde, e ele responde na última linha
    do NDJSON.
    """
    provider = OllamaProvider(model="m")
    assert provider.last_metrics is None

    provider._parse_line(
        json.dumps(
            {
                "done": True,
                "message": {"content": ""},
                "load_duration": 4_100_000_000,
                "prompt_eval_count": 800,
                "prompt_cache_hit_count": 600,
                "prompt_eval_duration": 2_000_000_000,
                "eval_count": 20,
                "eval_duration": 2_000_000_000,
                "total_duration": 8_100_000_000,
            }
        )
    )

    metrics = provider.last_metrics
    assert metrics is not None
    assert metrics.load_ms == 4100
    assert metrics.prompt_tokens == 800
    assert metrics.cache_hit_ratio == 0.75
    assert metrics.output_tokens_per_s == 10.0


def test_ollama_accepts_the_older_name_for_the_cache_counter():
    """O campo mudou de nome entre versões do Ollama; perder a métrica por
    causa disso seria perder justamente o número que a E7 foi medir."""
    provider = OllamaProvider(model="m")
    provider._parse_line(
        json.dumps(
            {
                "done": True,
                "message": {"content": ""},
                "prompt_eval_count": 100,
                "prompt_eval_cached_count": 90,
            }
        )
    )
    assert provider.last_metrics.prompt_cached == 90


def test_metrics_without_numbers_do_not_invent_any():
    provider = OllamaProvider(model="m")
    provider._parse_line(json.dumps({"done": True, "message": {"content": ""}}))

    metrics = provider.last_metrics
    assert metrics.prompt_tokens is None
    assert metrics.cache_hit_ratio is None
    assert metrics.output_tokens_per_s is None
