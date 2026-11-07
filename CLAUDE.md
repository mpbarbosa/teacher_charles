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
python3 scripts/check_env.py             # RAM, disco, uv, Ollama, modelo, pesos de voz
uv run teacher-charles --reload          # http://localhost:8000
```

Voz (Fase 1.5) é **extra e desligada por default** — o chat por texto funciona
inteiro sem ela, e são ~400 MB de wheels mais ~120 MB de pesos:

```bash
uv sync --extra dev --extra voice
uv run python scripts/fetch_voice_models.py      # Kokoro; o Whisper se baixa sozinho
TC_VOICE_ENABLED=true uv run teacher-charles
```

```bash
uv run pytest                            # tudo (integração inclusa, bate no Ollama de verdade)
uv run pytest -m "not integration"       # 620 testes, ~16s, sem modelo nenhum
uv run pytest -m integration             # só os 9 que exercitam o Ollama local
uv run pytest tests/test_turn.py::test_nome -x
uv run ruff check . && uv run ruff format --check .
```

Os testes de voz seguem a mesma regra: os de dublê rodam sem peso nenhum
(`test_voice*.py`), e os 8 que carregam Whisper e Kokoro de verdade estão em
`-m integration` e se pulam sozinhos quando os pesos não estão no disco.

`uv run pytest` sem filtro **não** é o caminho rápido: os testes de integração
se pulam sozinhos sem Ollama, mas com o Ollama de pé eles carregam modelo e o
tempo vai de ~3s para minutos. Durante o desenvolvimento use
`-m "not integration"`.

Para trabalhar na interface sem modelo: `TC_PROVIDER=fake uv run teacher-charles`.
Para usar um modelo que já está baixado: `TC_LLM_MODEL=qwen2.5:3b-instruct-q4_K_M uv run teacher-charles`.

## O princípio que organiza o código

> **A estrutura pedagógica mora na aplicação, não no prompt. Correção é dado
> estruturado, não prosa.**

A E13 acrescentou a segunda metade disso: **o que a aplicação sabe sobre o
aluno também é dado estruturado, não um ranking.** A fila do drill deixou de ser
contagem de erros e passou a ser agenda de revisão, porque contagem só sabe
somar — ela nunca registra que o aluno aprendeu.

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
| `voice/` | As duas pontas da Fase 1.5 — STT e TTS — atrás de `base.SpeechToText` / `base.TextToSpeech`, no mesmo arranjo de `llm/`. **Não importa `teaching/`**: recebe bytes, devolve texto; recebe texto, devolve PCM. |
| `store/` | SQLite pelo `sqlite3` da stdlib. Esquema em `.sql` numerado, sem ORM; as dataclasses de `models.py` só dão nome às linhas. |
| `teaching/schedule.py` | A agenda de revisão (E13). Sabe **quando** uma regra volta e não sabe nada de base contrastiva; o `DrillPlanner` sabe **quais** regras podem virar drill. Nenhum dos dois decide sozinho. |
| `prompts.py` | Carrega `prompts/*.md` com cache por mtime: editar a persona vale no turno seguinte, sem reiniciar. |
| `app.py` | FastAPI. Monta tudo no `lifespan` e guarda em `app.state`. |
| `web/` | Página única: template Jinja + JS/CSS sem build step. |

`config.py` nunca importa `llm` **nem `voice`** — a dependência corre num sentido
só. É por isso que `default_options(settings)` vive em `llm/__init__.py` e não em
`Settings`.

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
é treinada — por consulta ao banco, nunca pelo modelo. Uma regra
por sessão (`session.drill_rule_id`), não por turno: drill que troca de regra a
cada mensagem é questionário, e bloco estável no prompt é prefixo cacheado.

**A fila do drill é agenda, não ranking.** Quem ordena é `Scheduler.queue`
(E13): vencida primeiro, e a frequência de `correction` ficou como *último*
desempate entre regras igualmente urgentes. A troca não foi cosmética —
frequência é um número que só sobe, então a regra mais errada do histórico
continuava sendo a mais errada depois de o aluno acertar dez vezes e o drill
nunca a soltava. Reverter a chave de ordenação para `-times` deixa quatro testes
vermelhos, inclusive um da E6.

**Acerto é a ausência de erro onde o erro teria aparecido, e isso é assimétrico.**
O analisador devolve erros, não acertos, então acerto só pode ser inferido —
e só vale no drill, sobre a regra em treino, porque ali o exercício construiu a
oportunidade. Correção pela regra é erro em qualquer modo; turno limpo em
conversa livre **não é nada**, e contá-lo graduaria regras pelo silêncio.
A guarda que mais importa: **análise que não rodou não gradua ninguém** —
sem ela, analisador fora do ar viraria acerto, porque produz a mesma ausência
de correção. Ponto cego conhecido e anotado na §6 do plano: dentro do drill,
"não errou" e "não tentou" são a mesma observação.

**O primeiro degrau da escada de intervalos vale meio dia, não zero**, e é isso
que substitui o rodízio da E6. Com zero, a regra recém-errada vence na hora e
volta a ser a mais vencida na sessão seguinte, para sempre — o aluno nunca veria
a segunda regra. O `DRILL_ROTATION` sobreviveu só para o caso que a agenda não
vê (regra escolhida numa sessão que terminou sem resposta), e como termo de
ordenação em vez de filtro.

**As duas escritas de uma revisão não são uma transação.** É a regra da conexão
compartilhada, e a ordem é escolhida por causa dela: `rule_review` (o evento)
antes de `rule_schedule` (o estado), porque o estado se reconstrói do evento e
não o contrário.

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

**A transcrição do microfone vai para o campo de texto, não para o turno.** O
STT erra; um erro de STT que vira turno faz o Charles corrigir um erro que o
aluno não cometeu — a mesma classe de falha que a E3/E4 existem para evitar do
lado do modelo. `POST /api/stt` devolve só o texto e **não grava turno nenhum**
(`test_nao_grava_turno_nenhum`). É a única salvaguarda que não depende de qual
modelo de STT está carregado, e ela existe porque a exatidão com sotaque
brasileiro é a dívida aberta da Fase 1.5 (plano 1.5 §2.3).

**O áudio não viaja no SSE.** Os quadros `sentence` do `/api/chat` carregam
índice e texto; a página pede cada sentença ao `/api/tts` num pedido próprio.
Base64 seria 33% a mais de bytes no caminho crítico do texto — mas a razão real
é outra: sintetizar dentro do gerador que repassa os pedaços atrasaria a letra
para adiantar o som. Separado, a síntese da sentença 1 corre enquanto o modelo
escreve a 2, que é a única razão de o TTS por sentença existir.

**O índice do quadro `sentence` não é decoração.** Os pedidos ao `/api/tts`
resolvem fora de ordem; a fila do `voice.js` toca por índice. Tocar por ordem
de chegada embaralharia a fala do professor na primeira vez que a rede
hesitasse.

**Fronteira de sentença vale também no fim do pedaço**, e isso é escolha de
latência, não descuido (`voice/segment.py`). Esperar o espaço custaria um token
de atraso em toda sentença — ~130 ms a 7-8 tok/s — pagos no único ponto do
turno em que o aluno espera calado. O único caso indecidível, dígito antes do
ponto (`3.` que pode virar `3.5`), espera; abreviação e inicial são decididas
pelo que está antes do ponto.

**Os motores de voz carregam preguiçosamente e o aquecimento engole o erro.**
Derrubar a subida por causa de um peso ausente seria pior que um `/health`
degradado. Daí `ready` e `last_error` em `voice/base.Loadable`: é o `/health`
que denuncia, e `ready` nunca dispara carga — um health check não pode ter
efeito colateral de centenas de megabytes.

**`av` está preso em `<19`.** O `decode_audio` do `faster-whisper` 1.2 chama
`av.open(metadata_errors=...)`, argumento que o PyAV 19 removeu. Com o 19
instalado, **toda** transcrição estoura em `TypeError` antes de o modelo ver um
sample. Não é cautela de versão; é incompatibilidade medida.

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

- **Pesos do modelo** — em `~/.ollama`, nunca no repo. Os de voz também não:
  Kokoro em `$XDG_CACHE_HOME/teacher-charles/voice` (cache, redownloadável) e
  Whisper no cache do `huggingface_hub`. O reconhecedor de fonema da Fase 2
  segue o Whisper, pela mesma razão — é repositório do Hub, e o Hub já tem
  cache. **Nunca em `XDG_DATA_HOME`**, que é dado pessoal do aluno.
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

A **E13 (repetição espaçada) está entregue**, fora de ordem e de propósito: o
plano da Fase 2 §5 diz que ela não depende do portão da E9, e foi o caminho
tomado enquanto a E9 segue aberta. O que ela trocou está na §7 de
`docs/plano-fase-2.md`, e as duas dívidas que ela abriu na §6 de lá.

A Fase 1 está completa, e a **Fase 1.5 (voz) também** —
`docs/plano-fase-1-5.md`. O aluno fala por push-to-talk, o Charles responde em
voz alta sentença por sentença, tudo local. `teaching/`, `llm/` e `store/` não
mudaram uma linha para isso: a voz entrou como duas pontas, que era a aposta
do desvio da §4.

A Fase 1.5 deixou duas dívidas, as duas medidas em `docs/plano-fase-1-5.md`:
**5,4-7,4s até a primeira palavra falada** contra a meta de 1-2s (e a maior
fatia é o LLM, não o áudio — a alavanca é a iGPU ociosa da §1.3), e a
**exatidão do STT com sotaque brasileiro, que não foi medida** — o áudio das
tabelas é sintético, e os 0% de WER são piso, não expectativa.

A **Fase 2 está aberta** — `docs/plano-fase-2.md`: pronúncia por fonemas e
repetição espaçada. Ela abriu pela medição, como a 1.5, e a medição já moveu o
desenho em cinco lugares (§3 de lá). Os números que importam:

- Metade da pilha de pronúncia **já está instalada** pelo extra `voice`: o
  `phonemizer` + `espeakng_loader` que o Kokoro traz dão o G2P, e o
  `onnxruntime` dá o runtime. O que falta é um peso de 318 MB.
- O reconhecedor de fonema custa **RTF 0,17** — mais rápido que o Whisper que
  já está no caminho — e por isso roda **depois** do turno, nunca dentro dele:
  o caminho até a primeira palavra já está estourado em 3,99 s.
- O piso é **0,061** de taxa de erro de fonema em fala correta e o sinal é
  **0,119**, com 6 dos 8 erros curados nomeados pelo fonema certo. Mas o proxy
  de sotaque brasileiro vai a **0,676**, e aí o alinhamento desaba: quando dois
  de cada três fonemas discordam, nada é *o* erro.
- Daí a E9 ser **portão e não etapa**: medir com voz humana brasileira é o que
  decide se o avaliador existe. Nada de produto antes dela.

O que vem depois está em `docs/plano-fase-2.md` §8: turn-taking sem botão,
prosódia, a dívida da análise, e mobile/multiusuário.

Duas dívidas conhecidas da Fase 1, as duas medidas e registradas na §5.5: a análise custa
~10s por turno com erro (o caminho é um modelo melhor só para ela, e a medição
mostra que isso não atrapalha a conversa), e a primeira palavra está em 2,7s
contra a meta de 1–2s da Fase 1.5. A §7 do plano 1.5 fechou a segunda ligando a
iGPU (0,9s) e **não** fechou a primeira: a análise *gera* JSON, e geração
empata entre CPU e iGPU.

Há vários perfis de aluno numa máquina, e o ativo é o de maior `seen_seq`; login
e multiusuário de verdade ficam fora da Fase 1 (plano §4).

Consulta à base contrastiva na Fase 1 é **casamento por lema e padrão sobre YAML
curado**, não busca vetorial: ~100 itens cabem na memória, o casamento é
determinístico e depurável (`"I have 30 years"` → `calque.age-have`, sempre), e
cada item ganha um teste. Embeddings são anotados como item da Fase 2.
