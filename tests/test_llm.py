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


def ollama_with(handler) -> OllamaProvider:
    transport = httpx.MockTransport(handler)
    return OllamaProvider(
        model="qwen3:4b",
        client=httpx.AsyncClient(transport=transport),
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
    return Settings(**base)


def test_build_provider_defaults_to_ollama(tmp_path):
    provider = build_provider(settings_for(tmp_path))
    assert isinstance(provider, OllamaProvider)
    assert provider.model == "qwen3:4b"
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
