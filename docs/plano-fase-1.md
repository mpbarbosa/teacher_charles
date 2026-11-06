# Teacher Charles — Estrutura do repositório e plano da Fase 1

*Novembro de 2026 — proposta para aprovação*

Continuação de [analise-inicial.md](analise-inicial.md). Este documento fecha as três
questões em aberto da seção 8 daquele texto e define o que será construído na Fase 1.

---

## 1. Decisões fechadas

### 1.1 Público-alvo: **B1 em diante, 100% em inglês**

Sem português na interface, nas respostas ou na voz. Imersão total.

O conhecimento contrastivo PT→EN **continua sendo o diferencial** — o professor sabe que
o aluno é brasileiro e antecipa os erros típicos — mas explica em inglês:

> *"Careful — 'pretend' doesn't mean 'pretender'. You want 'intend'."*

O que isso compra (seção 5 da análise): voz monolíngue, Whisper `.en` menor e mais
preciso, modelo de linguagem menor viável, prompt e testes com uma língua a menos
para validar.

O que isso custa, e é aceito: iniciantes A1/A2 ficam fora do produto, e o aluno que
não entende uma explicação não tem rede de segurança em português.

### 1.2 Plataforma: **web local — backend Python + navegador**

Backend FastAPI em `localhost`, interface no navegador. "Local" aqui significa *roda na
máquina do usuário*, não na nuvem — nada sai do computador.

Por quê:

- Python dá acesso direto a `faster-whisper`, Kokoro e às bibliotecas de fonema que a
  Fase 2 vai precisar. Essa é a razão principal.
- O navegador resolve microfone, reprodução de áudio e UI sem nenhum SDK nativo.
- Nenhum passo de empacotamento entra no caminho antes de existir produto.

Descartados: **WebGPU puro** (limita a modelos de 1–4B no browser, dificulta
Whisper/Kokoro e praticamente inviabiliza a análise de fonemas da Fase 2);
**desktop nativo** (Tauri/Electron acrescenta build e ponte com o Python antes de haver
qualquer coisa para empacotar — fica como opção de distribuição futura, reaproveitando
este mesmo backend).

### 1.3 Hardware mínimo: **CPU / iGPU, sem CUDA**

O alvo mínimo é a máquina de desenvolvimento atual:

| | |
|---|---|
| CPU | Intel Core Ultra 5 135U (Meteor Lake-P, 14 threads) |
| GPU | Intel Arc integrada — **sem CUDA, sem VRAM dedicada** |
| RAM | 30 GB |

Isto é mais modesto que o cenário da análise inicial ("GPU de 8 a 12 GB ou Apple
Silicon") e tem uma consequência direta: **o gargalo é banda de memória, não VRAM.**

#### Vazão medida (não estimada) — corrigido na E1

A primeira versão deste documento estimava 20–25 tok/s para um 4B. **A estimativa
estava cerca de 2,5× otimista.** Medido nesta máquina com o Ollama 0.35, modelo já
carregado, 150 tokens gerados, usando `eval_count`/`eval_duration` do próprio servidor:

| Modelo (Q4_K_M) | Tamanho | Geração | Prompt |
|---|---|---|---|
| qwen2.5 **3B** | 2,2 GB | **9,3 tok/s** | 54 tok/s |
| qwen2.5 **7B** | 5,1 GB | **4,0 tok/s** | 21 tok/s |

Extrapolando pela razão de tamanho: **4B ≈ 7–8 tok/s, 8B ≈ 3,5 tok/s, 14B ≈ 2 tok/s.**

A causa está medida também: `/api/ps` reporta `size_vram=0.00GB` para os dois modelos.
**O Ollama padrão roda 100% em CPU e não toca a iGPU Intel** — não há build com
suporte a Arc/SYCL no pacote oficial. Os 14 threads do Core Ultra 5 135U (que é um chip
U-series, de baixo consumo) são tudo o que está trabalhando.

Consequências, e nenhuma delas derruba as decisões 1.1–1.3:

- **4B continua sendo o default certo**, e agora por uma margem maior: 7–8 tok/s dá uma
  resposta de 60 tokens em ~8s. Com streaming o aluno começa a ler no primeiro segundo,
  então para **chat por texto isto é tolerável**. Um 8B a 3,5 tok/s não é.
- **O turno duplo da §2 fica mais caro do que o risco da §6 supunha.** A análise tem que
  produzir JSON curto (poucas dezenas de tokens) e não pode bloquear a resposta. A E4
  precisa medir isso, não presumir.
- **A voz da Fase 1.5 fica em dúvida real.** A meta de 1–2s por turno pressupõe gerar a
  primeira sentença em menos de 1s; a 7 tok/s, 15 tokens levam ~2s, antes do TTS. Isso
  não condena a Fase 1.5, mas significa que ela começa por uma medição, não por
  integração.
- **Há um caminho de aumento não explorado**: habilitar a iGPU via IPEX-LLM ou um
  llama.cpp com Vulkan/SYCL. A Arc integrada do Meteor Lake tem compute relevante e hoje
  está ociosa. Fica registrado como alavanca para a Fase 1.5, não como trabalho da
  Fase 1 — e é exatamente o tipo de troca que a abstração `ChatProvider` da E1 torna
  barata (o `openai_compat` já serve um llama.cpp local).

**Modelo default: um 4B-class quantizado (Qwen3 4B como primeira escolha).** A seção 5
da análise apoia isso: sendo tudo em inglês, "um modelo de 4B a 8B explica gramática em
inglês com boa confiabilidade". O 8B deixa de ser "opção para quem tem folga" nesta
máquina e passa a ser opção só para quem tem GPU.

A decisão 1.1 é o que torna 1.3 sustentável. Se o produto fosse bilíngue, 4B não daria
conta das explicações em português e o hardware mínimo teria que subir.

### 1.4 Latência medida na E2

Com a conversa funcionando, dá para medir o que o aluno sente, e não só tokens
por segundo. Medido no navegador e por `curl`, com `qwen2.5:3b-instruct-q4_K_M`:

| Momento | Primeira palavra | Turno completo |
|---|---|---|
| Primeiro turno, Ollama frio, **sem** aquecimento | 16–19 s | ~23 s |
| Primeiro turno, com aquecimento na subida | **7,4 s** | 11,0 s |
| Turnos seguintes (regime permanente) | **0,7–1,1 s** | 4,3–4,6 s |

Duas descobertas estão por trás disso, e as duas viraram código:

**Carregar o modelo não era o problema principal.** A primeira versão do
aquecimento mandava um `"hi"` solto, o que carregava o modelo e mesmo assim
deixava o primeiro turno em 19 s. A causa era o **cache de prefixo**: os 427
tokens da persona não estavam nele, e o turno pagava a avaliação inteira. Os
mesmos 427 tokens custam 0,4–0,7 s quando o prefixo está cacheado
(`prompt_eval_cached_count` = 412). O aquecimento passou a mandar a persona
exatamente como um turno real manda.

**Daí a persona ter que ser estável.** Interpolar valores que mudam no prompt
de sistema — o nível do aluno, a data, um contador — jogaria esse cache fora a
cada turno e devolveria os segundos que o aquecimento economizou. Isto é uma
restrição de projeto para a E5, que vai querer ajustar o prompt por nível: o
ajuste vai **depois** da persona, não dentro dela.

O que isto diz para o resto do plano:

- Para **texto**, 0,7 s até a primeira palavra e ~4,3 s de turno é confortável.
  O streaming faz o trabalho: o aluno lê enquanto o resto chega.
- Para a **voz da Fase 1.5**, a meta de 1–2 s deixou de ser implausível. 0,7 s
  até o primeiro token dá margem para TTS por sentença — desde que a Fase 1.5
  comece medindo, como a §1.3 já dizia.
- Sobra um incômodo honesto: 7,4 s no primeiro turno mesmo com aquecimento,
  contra 0,7 s nos seguintes. Não está explicado. Fica para a E7, que é onde a
  latência é medida a sério.

---

## 2. Princípio de arquitetura que organiza tudo

> **A estrutura pedagógica mora no código, não no prompt.**

É a conclusão central da análise inicial, e na Fase 1 ela se traduz em uma escolha
concreta: **correção é dado estruturado, não prosa.**

Cada turno do aluno passa por duas etapas distintas:

1. **Análise** — recebe a fala do aluno, o nível dele e os candidatos da base
   contrastiva; devolve JSON validado: `{errors: [{span, type, correction, rule_id,
   explanation}], level_signals: {…}}`.
2. **Resposta** — o professor responde conversando, já informado do que a análise achou.

Três coisas decorrem disso, e nenhuma delas sai de graça de um prompt:

- Quando a base contrastiva tem uma regra para o erro, a correção **precisa citar o
  `rule_id`**. É assim que se evita o modelo inventar regra gramatical — o problema nº 1
  da análise.
- Correções viram linhas em SQLite, contáveis e agregáveis. Sem isso não existe
  repetição espaçada na Fase 2, nem exercício gerado a partir do erro real do aluno.
- O erro recorrente do aluno fica visível para ele.

---

## 3. Estrutura do repositório

```
teacher_charles/
├── README.md
├── pyproject.toml                  # uv; requires-python = ">=3.12,<3.13"
├── .python-version
├── .env.example
│
├── docs/
│   ├── analise-inicial.md
│   ├── plano-fase-1.md             # este arquivo
│   └── adr/                        # decisões de arquitetura, uma por arquivo
│
├── data/                           # VERSIONADO — é conteúdo do produto
│   ├── contrastive/                # a base de erros típicos de brasileiros
│   │   ├── false-friends.yaml      # pretend/pretender, actually/atualmente
│   │   ├── calques.yaml            # "I have 30 years", "make a question"
│   │   ├── grammar.yaml            # interferência estrutural PT→EN
│   │   └── pronunciation.yaml      # th, vogal epentética — texto na Fase 1
│   └── schema/
│       └── contrastive-item.schema.json
│
├── prompts/                        # VERSIONADO, fora do código Python
│   ├── persona.md                  # quem é o Charles
│   ├── analysis.md                 # instrução da etapa de análise
│   ├── modes/
│   │   ├── conversation.md
│   │   ├── correction.md
│   │   └── drill.md
│   └── partials/
│       └── level/{a2,b1,b2,c1}.md  # ajuste de registro por nível
│
├── src/teacher_charles/
│   ├── config.py                   # caminhos, modelo, perfil de hardware
│   ├── app.py                      # FastAPI: rotas + streaming SSE
│   ├── llm/
│   │   ├── base.py                 # Protocol ChatProvider
│   │   ├── ollama.py               # local, default
│   │   ├── openai_compat.py        # nuvem opcional (recomendação §6 da análise)
│   │   └── fake.py                 # usado nos testes
│   ├── teaching/                   # ← o coração do produto
│   │   ├── turn.py                 # orquestra um turno completo
│   │   ├── analyzer.py             # fala do aluno → erros estruturados
│   │   ├── contrastive.py          # carrega e consulta data/contrastive
│   │   ├── level.py                # CEFR: estimativa e ajuste
│   │   └── modes.py                # conversation | correction | drill
│   ├── store/
│   │   ├── db.py                   # SQLite + migrações
│   │   ├── models.py
│   │   └── repo.py                 # perfil, sessões, erros, agregações
│   └── web/
│       ├── static/                 # JS/CSS sem build step na Fase 1
│       └── templates/
│
├── tests/
│   ├── test_contrastive.py
│   ├── test_analyzer.py            # suíte de frases com erro conhecido
│   ├── test_level.py
│   └── fixtures/
│
└── scripts/
    ├── check_env.py                # valida RAM, modelo baixado, Ollama de pé
    └── seed_profile.py
```

### Onde fica cada coisa, e por quê

| | Onde | Motivo |
|---|---|---|
| **Pesos do modelo** | Fora do repo, em `~/.ollama` (caminho configurável) | Gigabytes não entram em git. `models/` vai no `.gitignore`. |
| **Prompts** | `prompts/*.md`, versionados, carregados em runtime | Ajustar a persona não deve exigir editar Python nem redeploy. O diff de um prompt fica legível no git. |
| **Base contrastiva** | `data/contrastive/*.yaml`, versionada, com JSON Schema | É "o principal diferencial do produto" (análise, §4.4). Tratada como código: revisada, validada, testada item por item. |
| **Progresso do aluno** | `~/.local/share/teacher-charles/charles.db` (XDG) | Dado pessoal do usuário. Nunca no repo. `data/` guarda conhecimento curado, não dados de aluno. |

### Consulta à base contrastiva: sem vetores na Fase 1

A análise fala em RAG. Na Fase 1 a consulta será **casamento por lema e padrão sobre um
YAML curado de ~100 itens**, não busca vetorial.

Motivo: o corpus cabe inteiro na memória, o casamento é determinístico e depurável
(`"I have 30 years"` → `calque.age-have`, sempre), e cada item ganha um teste. Embeddings
acrescentam dependência, modelo extra e resultado difuso para um problema que ainda não
existe. Quando a base passar de algumas centenas de itens, embeddings entram como
melhoria — fica anotado como item da Fase 2.

### Stack

Python 3.12 (não 3.14 — os wheels de `ctranslate2` e `onnxruntime`, que a voz vai
precisar na Fase 1.5, ficam atrás das versões novas; não vale trocar o interpretador no
meio do caminho) · uv · FastAPI + uvicorn · pydantic · SQLite via `sqlite3` da stdlib ·
Ollama como servidor do modelo, atrás da abstração `ChatProvider` · frontend em
HTML/CSS/JS sem build step · pytest.

---

## 4. Escopo da Fase 1

### Dentro

1. Chat por texto, 100% em inglês, com resposta em streaming.
2. Perfil do aluno: CEFR inicial autodeclarado, ajustado por sinais dos turnos.
3. Três modos com comportamento de fato distinto: *free conversation*, *correction*, *drill*.
4. Correção estruturada, validada, com citação obrigatória de regra quando houver match.
5. Base contrastiva inicial curada (~100 itens) com schema e testes.
6. *Error log* do aluno: erros recorrentes agregados por regra, visíveis numa tela.
7. `ChatProvider` abstraído: Ollama local como default, API de nuvem como opção.
8. `check_env.py`: diz ao usuário se a máquina e o modelo estão prontos.

### Fora, explicitamente

| Fica fora | Quando entra |
|---|---|
| Voz — STT e TTS (`faster-whisper` + Kokoro) | **Fase 1.5** — ver §5 |
| Avaliação de pronúncia por fonemas | Fase 2 |
| Repetição espaçada | Fase 2 |
| Qualquer português na UI ou nas respostas | Não entra (decisão 1.1) |
| RAG vetorial / embeddings | Fase 2, se a base crescer |
| Modelo de 14B, fine-tuning | Não entra (decisão 1.3) |
| Mobile, multiusuário, login, deploy em nuvem, instalador | Depois da Fase 2 |

### Desvio em relação à análise inicial — precisa de decisão sua

A análise colocava **voz dentro da Fase 1**. Esta proposta a move para uma **Fase 1.5**,
depois que o núcleo pedagógico estiver provado.

A razão: voz é latência e integração, não ensino. Se o professor ainda não corrige bem
por texto, a voz só esconde esse problema atrás de um pipeline de áudio. E ela nunca é
jogada fora — o `turn.py` já nasce agnóstico quanto à origem do texto, então acoplar
STT/TTS depois é acrescentar duas pontas, não reescrever o meio.

Fase 1.5, uma vez que a Fase 1 esteja de pé: `faster-whisper` `small.en` em int8 na CPU,
Kokoro-82M na CPU, VAD do Silero, TTS por sentença (começa a falar antes de a resposta
terminar) e medição honesta da latência nesta máquina contra a meta de 1–2 s.

Se você preferir voz já na Fase 1, é uma troca razoável — o custo é um MVP mais demorado
para chegar ao primeiro uso real.

---

## 5. Ordem das entregas

Cada entrega termina em algo verificável. A E2 é o primeiro ponto em que existe produto.

| | Entrega | Pronto quando |
|---|---|---|
| **E0** ✅ | **Fundação** — pyproject/uv, pacote, SQLite + migração, `check_env.py`, FastAPI com `/health` | `uv run teacher-charles` sobe em `localhost:8000` e `/health` reporta se o modelo está acessível |
| **E1** ✅ | **Provider do LLM** — Protocol `ChatProvider`, Ollama com streaming, `openai_compat` atrás de flag, erros e timeout tratados, `fake` para testes | **Feito.** 4 testes de integração fazem turnos reais contra o Ollama local (47 pedaços de streaming, 1º em 1,0s); 58 unitários rodam sem modelo nenhum |
| **E2** ✅ | **Chat mínimo** — rota SSE, UI de página única, histórico persistido, persona vinda de `prompts/persona.md` | **Feito.** Conversa em inglês de ponta a ponta no navegador, verificada com o modelo real. Latência medida na §1.4 |
| **E3** ✅ | **Base contrastiva** — schema, ~40 itens iniciais (falsos cognatos + decalques), loader validado, matcher, um teste por item | **Feito.** 43 itens (17 falsos cognatos, 17 decalques, 9 de interferência estrutural) e 205 testes, dos quais 172 são quatro por item; 0 falso positivo em 16 frases corretas, 9 de 10 frases com erro típico pegas. Ver §5.1 |
| **E4** ✅ | **Correção estruturada** — `analyzer.py` com saída JSON validada por pydantic, regra citada quando há match, fallback para JSON inválido, persistência das correções | **Feito.** 31 frases com erro conhecido: 87% recebem correção, todas citando a regra esperada. 18 frases corretas: 0 corrigidas por engano. 0 de 49 turnos com JSON inválido. Ver §5.2 |
| **E5** | **Perfil e nível** — CEFR inicial, sinais por turno, prompt ajustado por nível via `prompts/partials/level/` | Trocar o nível do aluno muda visivelmente a resposta, verificado em teste de snapshot |
| **E6** | **Modos** — os três com prompt próprio; *drill* gera exercício a partir do error log real do aluno | Os três modos funcionam na UI e o drill usa erros de fato cometidos, não genéricos |
| **E7** | **Progresso e acabamento** — tela de error log agregado por regra, README de dev, latência medida e registrada | Um aluno consegue ver no que erra mais; um dev clona e roda seguindo só o README |

### 5.1 O que a E3 estabeleceu

**O padrão é escrito à mão; os exemplos errados são o teste dele.** Quando um item
traz `patterns`, só eles casam — os literais de `wrong` não entram. É o que impede o
teste por item de ser vacuoso: se o padrão de `calque.age-have` não generalizar de
*30* para *25*, a suíte quebra. Os 43 itens têm padrão escrito à mão, exercitado por
102 exemplos errados e 108 corretos. Derivar o padrão dos literais de `wrong` continua
possível, e é o atalho para quem for curar item de forma fixa.

**Um casamento é suspeita, não veredito.** Oito itens casam também com inglês
correto (*I pretend to be a doctor* está certo; *I pretend to travel* é decalque de
*pretendo*). Eles vêm marcados `ambiguous: true`: a base os oferece como candidatos
e quem decide é a análise da E4. O que o matcher garante é que o candidato certo
*está* na lista — e um teste limita os ambíguos a no máximo um terço da base, porque
uma base em que a maioria depende do modelo não é anteparo nenhum.

**Medição, não estimativa.** Contra 16 frases corretas que um aluno B1 produz:
nenhum falso positivo. Contra 10 frases com erro típico: 9 pegas. A que escapou era
`has many beaches in Bahia` depois de vírgula — e escapar ali está certo, porque
*Brazil is big and has many beaches* é inglês correto com o mesmo recorte. Essa é a
fronteira do casamento determinístico, e é onde a E4 entra.

**Dois achados que só a medição daria.** O padrão de *comprehensive* exigia pronome
e perdia justamente a forma mais comum do erro (*my boss is very comprehensive*);
foi corrigido com lista de substantivos de pessoa, para não morder *the report is
very comprehensive*. E `I have a cold` (correto, "estou resfriado") passou a ser
exemplo de `right` em `calque.have-cold`, virando guarda contra o padrão crescer
demais.

**`pronunciation.yaml` não entrou.** A estrutura em §3 o prevê, mas não há o que
casar em texto: *th* e vogal epentética são erros de fala. O arquivo entra na Fase
1.5, junto com o áudio. A categoria já existe no schema.

**O schema é gerado, não mantido à mão.** O modelo pydantic é a fonte única;
`data/schema/contrastive-item.schema.json` sai de
`scripts/dump_contrastive_schema.py` e um teste falha se os dois divergirem.

**O `/health` conta os itens.** Base que não carrega é 503, não primeiro turno
quebrado na cara do aluno.

### 5.2 O que a E4 mediu, e o que a medição mudou no desenho

Modelo: `qwen2.5:3b-instruct-q4_K_M` em CPU pura. Suíte:
`tests/fixtures/known_errors.yaml` — 31 frases com erro típico de brasileiro e
18 frases corretas, as mesmas que o teste determinístico usa como portão.
Reproduzir: `uv run python scripts/measure_analysis.py`.

#### Resultado final

| | |
|---|---|
| Frase com erro recebe correção | **87%** (27/31) |
| Dessas, citando a regra esperada | **100%** (27/27) |
| Frase correta corrigida por engano | **0%** (0/18) |
| JSON inválido (precisou de retentativa) | **0%** (0/49) |
| Análise degradada para "sem correção" | **0%** (0/49) |
| Análise dispensada por não poder dar em nada | 37% (18/49) |
| Latência da análise, isolada | mediana 8,4s |

Verificado no navegador com o modelo real, e a diferença entre os dois tipos de
turno é visível no rodapé: *I have 35 years* → primeira palavra 2,9s, turno
14,3s, análise 9,4s, com a anotação curada aparecendo sobre a fala do aluno;
*There have been many changes since the new manager arrived* → primeira palavra
2,7s, turno **3,9s**, nenhuma análise e nenhuma correção. Inglês correto saiu
quatro vezes mais barato que inglês com erro, que é como tem que ser.

Três decisões saíram da medição, e nenhuma delas estava no plano original.

#### 1. Correção sem regra curada não chega ao aluno

A primeira medição, com o modelo livre para corrigir o que quisesse, deu
**12 falsos positivos em 18 frases corretas** — e o detalhe que decidiu tudo:
**nenhum** deles citava regra curada, enquanto **24 de 24** acertos citavam a
regra esperada. Pior, vários falsos positivos *ensinavam erro*:

| O aluno escreveu | O 3B "corrigiu" para |
|---|---|
| There have been many changes | There **has** been |
| It depends on my boss | It **depend** on my boss |
| because I was tired | because I was **sleepy** |
| a doctor from Recife | a doctor from **Rio de Janeiro** |
| an English course | a course in English |

Então a política passou a ser: **só vai ao aluno a correção que cita regra da
base**. Isso eliminou 100% dos falsos positivos sem custar nenhum acerto — não
é troca, é ganho nos dois lados. O que o modelo propõe sem regra é contado
(`Analysis.unverified`), porque essa contagem é a lista de curadoria da próxima
rodada da base.

É configurável (`TC_REQUIRE_CURATED_RULE=false`) porque a premissa é do
modelo, não do desenho: um modelo maior pode merecer confiança sem regra. Mas
quem desligar, meça antes — o script existe para isso
(`--allow-unverified` mostra exatamente o que o modelo diria sozinho).

#### 2. Conversa primeiro; "paralelo" aqui era ficção

O plano §2 pedia as duas etapas no mesmo turno, e a primeira implementação as
disparava juntas. Medido em 5 turnos de ponta a ponta:

| | primeira palavra | turno completo |
|---|---|---|
| Análise em paralelo com a resposta | mediana **15,3s** | 16,4s |
| Conversa primeiro, análise depois | mediana **6,1s** | 18,1s |

O Ollama atende um pedido por vez neste modelo, então disparar as duas juntas
não paraleliza nada: só põe a análise na frente da resposta. A ordem default
passou a ser conversa primeiro — o aluno lê a resposta enquanto a correção é
calculada, e a anotação aparece sobre o que ele escreveu alguns segundos
depois. O modo paralelo fica como opção (`TC_ANALYSIS_AFTER_REPLY=false`),
porque num provider de nuvem ou num servidor que atenda dois pedidos ao mesmo
tempo ele volta a ser o melhor arranjo.

Fica registrado o efeito colateral que a medição expôs: **a primeira palavra
degrada turno a turno** (0,1s → 3,8s → 6,1s → 7,9s → 10,8s em 5 turnos). Cada
chamada de análise joga o prompt da conversa fora do cache de prefixo do
Ollama, e o turno seguinte paga a persona e o histórico de novo. Isso é dívida
conhecida, não resolvida: a saída provável é a análise num segundo modelo, com
cache próprio, e a medição vira item da E7.

#### 3. Frase sem candidato não vira pergunta ao modelo

Consequência direta da política: se a base não levantou candidato e a correção
precisa de regra, nada que o modelo devolvesse poderia chegar ao aluno. Então
não se pergunta. São 37% dos turnos da suíte — turnos de inglês correto — que
passaram a custar **zero** chamada de modelo, zero espera e, principalmente,
zero evicção do cache de prefixo da conversa.

#### O que ficou ruim, dito por inteiro

**4 de 31 erros escapam** (`I lost the bus`, `I have many parents`, `we
realized a meeting`, `I work here since 2019`). Em todos, a base levantou a
regra certa e a pôs no prompt, e o 3B respondeu "sem erros" — são frases
gramaticais em inglês, erradas só na leitura contrastiva. Uma linha no prompt
("as regras abaixo foram revisadas à mão; se uma casa, reporte") subiu o recall
de 77% para 87%; os 13% restantes são limite do modelo, não do código.

Nesses quatro turnos o aluno não fica sem correção — mas não por garantia, e sim
porque o outro canal funcionou. Rodando os quatro de ponta a ponta, o professor
corrigiu **nos quatro**, na prosa, com a redação curada que a dica levou ao
prompt da resposta:

> aluno: *I lost the bus this morning.*
> Charles: *When you arrive too late, you "miss" it: "I missed the bus."*

O que falta nesses turnos é o registro estruturado, e com ele a linha no error
log. E o inverso também acontece: na verificação no navegador, com *I have 35
years*, a anotação estruturada saiu com a regra citada e a prosa do professor
**não** corrigiu nada — só puxou assunto. Os dois canais se cobrem, nenhum dos
dois sozinho é garantia, e é por isso que a correção que conta como dado é a
estruturada.

**O `corrected` de um 3B é medíocre.** Ele cita a regra certa e devolve
`'has 32 years, she is a nurse.'` => `'is 32 years, she is a nurse.'`, que
ainda está errado. A explicação que o aluno lê é a curada, então o dano é
limitado — mas o texto corrigido, não. É o argumento mais forte a favor de um
modelo melhor para a etapa de análise, que é barata em tokens e não precisa ser
o mesmo modelo da conversa.

**A janela de 2 correções por turno nunca encheu** na suíte: o 3B devolve uma
por vez. Não é limite exercitado ainda.

### Rodar em dev

```bash
uv sync
uv run python scripts/check_env.py      # confere RAM, Ollama, modelo
ollama pull qwen3:4b                    # modelo default
uv run teacher-charles --reload         # http://localhost:8000
```

Variáveis de ambiente em `.env` (modelo, provider, caminho do banco), com
`.env.example` versionado.

---

## 6. Riscos conhecidos da Fase 1

| Risco | Como lidamos |
|---|---|
| O 4B devolve JSON malformado na etapa de análise | Validação pydantic + retry com instrução mais estrita + fallback que degrada para "sem correção" em vez de quebrar o turno. **Medido na E4: 0 de 49 turnos precisaram de retentativa.** O risco não se materializou neste modelo; o anteparo fica, porque é barato e o modelo pode mudar. |
| O 4B explica gramática pior que o esperado | **Materializou-se, e o anteparo funcionou.** A E4 mediu um 3B inventando 12 correções em 18 frases corretas, várias ensinando erro (*There have been* → *has been*), e nenhuma citando regra — enquanto 100% dos acertos citavam a regra esperada. Daí a política: sem regra curada, a correção não chega ao aluno (§5.2). Medido antes, na E1: um 3B "corrigiu" *I have 30 years* para *I have thirty years* — trocou o dígito e manteve o decalque. É o problema nº 1 da análise em uma linha, e a razão de a E3/E4 existirem. Subir para 8B **não** é mais a saída nesta máquina (3,5 tok/s); a saída é a regra curada. |
| Latência do turno duplo (análise + resposta) | **Materializou-se, e pior do que o previsto.** A E4 mediu: em paralelo, a primeira palavra sai em 15,3s porque o Ollama atende um pedido por vez — o paralelismo era ficção. Default virou conversa primeiro, análise depois (6,1s), e turno sem candidato na base não chama o modelo. Resta a evicção do cache de prefixo, que faz a primeira palavra degradar turno a turno; ver §5.2. |
| Curar ~100 itens contrastivos dá trabalho | E3 entrega 40 e segue crescendo. O schema e os testes garantem que crescer não degrada. |

---

## 7. Fora do caminho crítico, registrado para depois

- **Fase 2**, como na análise inicial: avaliação de pronúncia por fonemas (wav2vec2 +
  alinhamento forçado), repetição espaçada, testes em hardware mais modesto.
- Embeddings na base contrastiva, se ela passar de algumas centenas de itens.
- Empacotamento desktop (Tauri) reaproveitando este backend.
- Reabrir o público A1/A2 com apoio em português — a decisão 1.1 é reversível, e o
  `prompts/partials/` é justamente o lugar onde esse apoio entraria.
