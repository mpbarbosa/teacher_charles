# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Idioma

Código, comentários, docstrings, commits e docs em **português**. Tudo que o
aluno vê — persona, strings da UI, resposta do modelo — em **inglês**, sem
exceção: o produto é "100% em inglês" por decisão de público (`docs/plano-fase-1.md`
§1.1).

## Comandos

```bash
uv sync --extra dev                      # ambiente (Python 3.12, pinado)
python3 scripts/check_env.py             # RAM, disco, uv, Ollama, modelo — roda com o Python do sistema
uv run teacher-charles --reload          # http://localhost:8000
```

```bash
uv run pytest                            # tudo (integração inclusa, bate no Ollama de verdade)
uv run pytest -m "not integration"       # 522 testes, ~4s, sem modelo nenhum
uv run pytest -m integration             # só os 9 que exercitam o Ollama local
uv run pytest tests/test_turn.py::test_nome -x
uv run ruff check . && uv run ruff format --check .
```

`uv run pytest` sem filtro **não** é o caminho rápido: os testes de integração
se pulam sozinhos sem Ollama, mas com o Ollama de pé eles carregam modelo e o
tempo vai de ~3s para minutos. Durante o desenvolvimento use
`-m "not integration"`.

Para trabalhar na interface sem modelo: `TC_PROVIDER=fake uv run teacher-charles`.
Para usar um modelo que já está baixado: `TC_LLM_MODEL=qwen2.5:3b-instruct-q4_K_M uv run teacher-charles`.

## O princípio que organiza o código

> **A estrutura pedagógica mora na aplicação, não no prompt. Correção é dado
> estruturado, não prosa.**

Consequência prática: quando houver uma regra curada para o erro, a correção
**cita o `rule_id`** de `data/contrastive/` em vez de deixar o modelo inventar
gramática. Por isso a tabela `correction` existe desde a primeira migração, com
`rule_id` anulável — `NULL` marca "veio só do modelo", distinção que a E4 mede e
a E7 mostra. Antes de resolver algo acrescentando frase ao prompt, verifique se
o lugar certo não é código em `teaching/`.

## Camadas

| | |
|---|---|
| `llm/` | Providers atrás de `base.ChatProvider`. Nada fora deste pacote sabe se o modelo está no Ollama, num llama.cpp ou na nuvem. Só `stream()` é abstrato; `complete()` deriva dele. |
| `teaching/` | O produto. **Agnóstico quanto à origem do texto** — recebe string, devolve eventos `Chunk`/`Analyzed`/`Leveled`/`Done`/`Failed`. É o que permite acoplar STT/TTS na Fase 1.5 sem reescrever o meio. |
| `store/` | SQLite pelo `sqlite3` da stdlib. Esquema em `.sql` numerado, sem ORM; as dataclasses de `models.py` só dão nome às linhas. |
| `prompts.py` | Carrega `prompts/*.md` com cache por mtime: editar a persona vale no turno seguinte, sem reiniciar. |
| `app.py` | FastAPI. Monta tudo no `lifespan` e guarda em `app.state`. |
| `web/` | Página única: template Jinja + JS/CSS sem build step. |

`config.py` nunca importa `llm` — a dependência corre num sentido só. É por isso
que `default_options(settings)` vive em `llm/__init__.py` e não em `Settings`.

## Invariantes que não são óbvias no arquivo que você está editando

**Uma conexão SQLite, compartilhada.** `app.state.db` guarda **uma** conexão com
`check_same_thread=False` e `isolation_level=None`. Daí a regra: **nada no caminho
de uma requisição abre transação de várias instruções** — duas requisições podem
se intercalar. Todo método de repositório é uma instrução autônoma. As migrações
são a exceção e rodam na subida, antes de o servidor aceitar conexão.

**O `BEGIN` vai dentro do script da migração.** `executescript` encerra qualquer
transação já aberta, então `conn.execute("BEGIN")` do lado de fora seria desfeito
e o `ROLLBACK` não protegeria o DDL parcial. Ver `store/db.py::migrate`.

**A persona vem primeiro e inalterada.** O Ollama cacheia o prefixo do prompt;
interpolar valor que muda no system prompt joga esse cache fora e o aluno paga a
reavaliação a cada turno. Prompt que varia por nível vai em
`prompts/partials/level/` e entra como **segunda mensagem de sistema**, não
dentro da persona: assim ele só invalida o cache quando o nível de fato muda.
Por isso o aquecimento também manda o partial do nível do perfil ativo.

**O modo muda política, não só prosa.** `teaching/modes.py` decide o teto de
correções por turno (5 em `correction`, 2 nos outros) e, no drill, **qual regra**
é treinada — por consulta ao `correction` agregado, nunca pelo modelo. Uma regra
por sessão (`session.drill_rule_id`), não por turno: drill que troca de regra a
cada mensagem é questionário, e bloco estável no prompt é prefixo cacheado.

**A ordem das mensagens de sistema é persona → nível → modo → dados do drill**,
e depois, no fim do prompt, a dica curada e o pedido do drill. O começo é por
estabilidade (o que muda menos vem antes, por causa do cache de prefixo); o fim
é por eficácia — medido duas vezes, na E4 com a dica e na E6 com o drill:
instrução longe do ponto de geração se dilui e o modelo pequeno a ignora.

**O nível do aluno é decidido por código, não pelo modelo.** `teaching/level.py`
tem uma rubrica determinística (comprimento de frase, palavra longa, conectivos,
estruturas, menos erro com regra curada) calibrada em
`tests/fixtures/level_samples.yaml`. Mexer nas constantes exige rodar
`scripts/measure_level.py` — os números estão no plano §5.3, e um teste quebra se
o acerto cair. Duas consequências que não são óbvias no arquivo: a rubrica
**não estima C2** (tentar errava por dois degraus) e **nenhum turno sozinho move
o nível** (4 turnos contados seguidos, um degrau por vez). A janela fecha em
toda mudança de nível, via `level_change.at_signal_id` — sem isso os mesmos
quatro turnos moveriam o nível de novo no turno seguinte.

**O aquecimento manda o prefixo do turno, não um "hi" nem "só a persona".**
Existe uma lista só, `TurnRunner.stable_prefix()`, e o aquecimento e o turno
usam a mesma — porque já foram duas e divergiram: a E5 e a E6 acrescentaram
nível e modo ao turno, o aquecimento ficou na persona, e o primeiro turno
chegava com 53% do prompt em cache e 11,6s até a primeira palavra. Com a mesma
lista: 0,9s (plano §5.5). `_warm_up` nunca propaga erro — o `/health` já
reporta modelo inacessível, e derrubar a subida por causa do aquecimento seria
pior.

**Nada de mensagem de sistema depois do histórico.** O template de chat junta
as mensagens de sistema no topo, então uma "dica no fim da lista" é colada
logo após a persona e invalida o cache de todo o histórico — medido: o total
cacheado ficava preso nos 674 tokens do prefixo e a primeira palavra ia de
2,8s a 9,1s em seis turnos. A dica curada e o pedido do drill vão **dentro da
última mensagem do aluno** (`TurnRunner._with_notes`). É contraintuitivo o
bastante para alguém "consertar" sem querer; o teste
`test_rule_hint_rides_with_the_student_message` existe para isso.

**A fala do aluno é gravada antes de chamar o modelo**, e o histórico já a inclui
— por isso `build_messages` não acrescenta a mensagem nova outra vez.

**O perfil ativo é `student.seen_seq`, um contador — não uma marca de tempo.**
Duas versões com relógio (segundo e milissegundo) quebraram o mesmo teste de
troca-e-volta de perfil: as escritas caem no mesmo instante e o desempate
devolve sempre o mesmo aluno.

**Erros de provider são tipados de propósito** (`ProviderUnavailable`,
`ModelNotFound`, `ProviderTimeout`, `ProviderBadResponse`). Reaja pelo tipo;
nunca por substring da mensagem.

**`/api/chat` é POST com corpo JSON e resposta em quadros SSE** (`data: {...}`,
tipos `chunk` / `done` / `error`). Não é `EventSource`: aquele só faz GET, e a
fala do aluno não vai em query string. O header `X-Accel-Buffering: no` existe
para um proxy no caminho não segurar o corpo e anular o streaming.

**`/health` responde 503 quando degradado** (banco ou modelo fora). É
deliberado: separar "o serviço subiu" de "está pronto para ensinar" evita health
verde enganoso.

## Testes

`tests/conftest.py` apaga toda variável `TC_*` e zera o cache de `get_settings`
em cada teste — um `.env` local já quebrou um teste aqui. Construa settings com
`Settings(_env_file=None, **overrides)` (ou o `make_settings` do conftest); nunca
conte com o ambiente da máquina.

Em `tests/test_level.py` o portão sobre as amostras rotuladas é uma **fração**
(70% de acerto exato, zero erro de dois degraus), não um valor fixo: a suíte pode
crescer sem o teste virar frágil, mas uma regressão real quebra.

Em teste de rota, passe `llm_warmup=False`: o aquecimento consumiria uma das
respostas roteirizadas do `FakeProvider` e apareceria em `provider.calls`,
bagunçando as asserções sobre o que o turno mandou ao modelo.

`FakeProvider` registra cada chamada (`calls`, `last_call`,
`system_prompt_of_last_call()`) e fatia a resposta de propósito, para quem
consome ter que lidar com pedaços como no provider real.

Os testes de integração escolhem o modelo em ordem: `TC_TEST_MODEL`, o
`TC_LLM_MODEL` configurado se estiver baixado, ou o menor presente na máquina —
ninguém precisa baixar 2,5 GB para a suíte ficar verde.

## O orçamento de hardware decide o design

Alvo mínimo é CPU/iGPU sem CUDA, e o Ollama oficial não usa a iGPU Intel. Medido
nesta máquina: **9,3 tok/s num 3B Q4 e 4,0 num 7B**, o que põe um 4B em ~7–8 e um
8B em ~3,5 — por isso o default é 4B-class e não 14B. Isso é premissa de projeto,
não detalhe de configuração:

- Streaming não é enfeite: é o que torna 7–8 tok/s tolerável. Não acumule a
  resposta antes de repassar.
- `DEFAULT_HISTORY_TURNS = 20` existe porque avaliar prompt custa tempo real
  (~54 tok/s num 3B), não por limite de janela de contexto.
- Modelos de raciocínio (Qwen3 entre eles) pensam antes de responder;
  `TC_LLM_THINK=false` desliga. `None` deixa o default do servidor valer.
- Python está pinado em `>=3.12,<3.13` de propósito: os wheels de `ctranslate2` e
  `onnxruntime`, que a voz da Fase 1.5 vai precisar, ficam atrás das versões
  novas. Não troque o interpretador.

## Onde o dado mora

Três destinos, e confundi-los é erro de categoria:

- **Pesos do modelo** — em `~/.ollama`, nunca no repo.
- **Progresso do aluno** — `$XDG_DATA_HOME/teacher-charles/charles.db`, fora do
  repo. É dado pessoal. `*.db` está no `.gitignore`.
- **Conhecimento curado** — `data/contrastive/*.yaml`, **versionado**: é conteúdo
  do produto, revisado e testado item por item. Hoje `data/` está vazia; a base
  chega na E3.

## Estado e para onde o código cresce

Todas as entregas da Fase 1 estão prontas: E0 (fundação), E1 (provider), E2
(chat por texto), E3 (base contrastiva), E4 (correção estruturada), E5 (perfil
e nível), E6 (os três modos) e E7 (tela de progresso, README e latência). A ordem e o critério
de pronto de cada entrega estão em `docs/plano-fase-1.md` §5;
`docs/analise-inicial.md` tem a análise de viabilidade que originou o projeto.

A Fase 1 está completa. O que vem depois está em `docs/plano-fase-1.md` §7:
voz (Fase 1.5), avaliação de pronúncia e repetição espaçada (Fase 2).

Duas dívidas conhecidas, as duas medidas e registradas na §5.5: a análise custa
~10s por turno com erro (o caminho é um modelo melhor só para ela, e a medição
mostra que isso não atrapalha a conversa), e a primeira palavra está em 2,7s
contra a meta de 1–2s da Fase 1.5.

Há vários perfis de aluno numa máquina, e o ativo é o de maior `seen_seq`; login
e multiusuário de verdade ficam fora da Fase 1 (plano §4).

Consulta à base contrastiva na Fase 1 é **casamento por lema e padrão sobre YAML
curado**, não busca vetorial: ~100 itens cabem na memória, o casamento é
determinístico e depurável (`"I have 30 years"` → `calque.age-have`, sempre), e
cada item ganha um teste. Embeddings são anotados como item da Fase 2.
