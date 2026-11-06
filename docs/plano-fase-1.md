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
  integração. **A dúvida se confirmou.** Medido na Fase 1.5: 5,4–7,4s até a primeira
  palavra falada, e o LLM sozinho responde por 3,0–4,2s deles — mais que o STT e o TTS
  somados. O áudio coube; o modelo de linguagem é que não.
- **Há um caminho de aumento não explorado**: habilitar a iGPU via IPEX-LLM ou um
  llama.cpp com Vulkan/SYCL. A Arc integrada do Meteor Lake tem compute relevante e hoje
  está ociosa. Fica registrado como alavanca para a Fase 1.5 — e depois dela medida,
  como **a** alavanca: é a única que move os 7s do turno de voz. Não como trabalho da
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
ajuste vai **depois** da persona, não dentro dela. (Foi o que a E5 fez: segunda
mensagem de sistema, que só invalida o cache quando o nível de fato muda — ver
§5.3.)

O que isto diz para o resto do plano:

- Para **texto**, 0,7 s até a primeira palavra e ~4,3 s de turno é confortável.
  O streaming faz o trabalho: o aluno lê enquanto o resto chega.
- Para a **voz da Fase 1.5**, a meta de 1–2 s deixou de ser implausível. 0,7 s
  até o primeiro token dá margem para TTS por sentença — desde que a Fase 1.5
  comece medindo, como a §1.3 já dizia.
- Sobrava um incômodo honesto: 7,4 s no primeiro turno mesmo com aquecimento,
  contra 0,7 s nos seguintes. **Resolvido na E7** (§5.5): o aquecimento mandava
  um prompt parecido com o do turno, não o mesmo, então parte do prefixo ficava
  fora do cache. Com a mesma lista nos dois lugares, o primeiro turno sai em
  0,9 s.

---

## 2. Princípio de arquitetura que organiza tudo

> **A estrutura pedagógica mora no código, não no prompt.**

É a conclusão central da análise inicial, e na Fase 1 ela se traduz em uma escolha
concreta: **correção é dado estruturado, não prosa.**

Cada turno do aluno passa por duas etapas distintas:

1. **Análise** — recebe a fala do aluno, o nível dele e os candidatos da base
   contrastiva; devolve JSON validado: `{errors: [{span, type, correction, rule_id,
   explanation}]}`.

   O `level_signals` que esta linha previa **não** veio do modelo: a E5 o move
   para código determinístico, porque a análise nem chama o modelo em turno de
   inglês correto — que é onde mora a evidência de nível alto — e porque nota
   atribuída por modelo não é depurável. Ver §5.3.
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
| Voz — STT e TTS (`faster-whisper` + Kokoro) | **Fase 1.5 — feita.** Ver [plano-fase-1-5.md](plano-fase-1-5.md) |
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

**Como terminou.** A Fase 1.5 foi feita e está em
[plano-fase-1-5.md](plano-fase-1-5.md). A aposta do desvio se pagou: `teaching/`,
`llm/` e `store/` não mudaram uma linha, e a voz entrou como duas pontas. A meta
de latência, essa não: são 5,4–7,4 s até a primeira palavra falada, contra os
1–2 s projetados — e a maior fatia é o LLM, não o áudio, exatamente onde a §1.3
dizia que estaria.

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
| **E5** ✅ | **Perfil e nível** — CEFR inicial, sinais por turno, prompt ajustado por nível via `prompts/partials/level/` | **Feito.** A rubrica determinística acerta o nível exato em 78% de 32 amostras rotuladas e nunca erra por dois degraus; o nível entra como segunda mensagem de sistema e muda a resposta do modelo real (frase de 6,5 para 10,0 palavras entre A2 e C1, e o erro escolhido muda). Ver §5.3 |
| **E6** ✅ | **Modos** — os três com prompt próprio; *drill* gera exercício a partir do error log real do aluno | **Feito.** Os três trocam na UI e mudam três coisas cada um (prompt, teto de correções, regra treinada). O drill escolhe a regra por consulta ao `correction` agregado e põe as frases do próprio aluno no prompt — sem error log, ele diz que não há o que treinar. Ver §5.4 |
| **E7** ✅ | **Progresso e acabamento** — tela de error log agregado por regra, README de dev, latência medida e registrada | **Feito.** `/progress` mostra no que o aluno mais erra, com as frases dele e a explicação curada. A latência foi medida e as duas dívidas pagas: o primeiro turno caiu para 0,9s e a degradação turno a turno acabou — a causa não era a análise, era a dica de sistema sendo içada ao topo pelo template. Ver §5.5 |

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
degrada turno a turno** (0,1s → 3,8s → 6,1s → 7,9s → 10,8s em 5 turnos).

> **A E7 mediu e o diagnóstico acima estava errado.** A degradação é real, mas
> não vem da chamada de análise: com a análise num segundo modelo, com cache
> próprio — a "saída provável" que este parágrafo sugeria —, ela acontece
> igual. O culpado era a dica curada como mensagem de sistema, que o template
> de chat iça para o topo do prompt. Ver §5.5.

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

### 5.3 O que a E5 mediu, e o que a medição mudou no desenho

A E5 dá ao perfil um nível que **faz diferença**: ele entra no prompt como
segunda mensagem de sistema (`prompts/partials/level/`), e muda o registro da
resposta. Quem decide esse nível é código — uma rubrica determinística sobre o
texto do aluno — e não o modelo. Reproduzir:

```bash
uv run python scripts/measure_level.py              # matriz de confusão, sem modelo
uv run python scripts/measure_level.py --hysteresis # quantos turnos até o nível andar
uv run python scripts/measure_level.py --reply      # a mesma fala em cada nível, no modelo real
```

#### Por que os sinais são código, e não uma pergunta ao modelo

O plano §2 pedia `level_signals` no JSON da análise. Isso não sobreviveu ao que
a E4 construiu, por dois motivos:

1. **A análise não chama o modelo quando a base não levanta candidato** — 37%
   dos turnos da suíte da E4. E turno de inglês correto é exatamente onde mora
   a evidência de nível alto. Pedir o sinal à análise seria medir nível só nos
   turnos com erro.
2. **Nota de modelo não é depurável.** O mesmo turno receberia B1 ou B2
   conforme a amostragem, e não haveria como explicar nenhum dos dois. A
   rubrica dá sempre o mesmo número para o mesmo texto, e cada número é
   conferível à mão sobre a frase.

A rubrica são quatro sinais positivos — comprimento médio de frase, proporção
de palavra longa, conectivos avançados e estruturas avançadas por 100 palavras —
menos uma penalidade por erro com regra curada. Erro **sem** regra curada não
penaliza: é a mesma política da §5.2, e inflar a penalidade com o que o modelo
achou sozinho rebaixaria o aluno por engano.

#### Resultado: 78% de acerto exato, 100% a um degrau

Suíte: `tests/fixtures/level_samples.yaml` — 32 amostras escritas à mão imitando
o que cada nível de fato manda num chat, 8 por nível.

| | A2 | B1 | B2 | C1 |
|---|---|---|---|---|
| **rótulo A2** | **8** | | | |
| **rótulo B1** | | **7** | 1 | |
| **rótulo B2** | | 1 | **3** | 4 |
| **rótulo C1** | | | 1 | **7** |

| | |
|---|---|
| Acerto exato | **78%** (25/32) |
| Acerto a um degrau | **100%** (32/32) |
| Pontas da escala (A2 e C1) | 15 de 16 |

O que a matriz diz, e é o achado mais importante da E5: **a rubrica separa bem
as pontas e confunde o meio.** A2 nunca passa de A2; C1 erra uma vez para B2. O
B2 é a classe fraca — metade das amostras lê como C1 —, e isso é limite do
método, não bug: a diferença entre um B2 e um C1 num chat está em precisão
lexical e nuance, não em contagem de palavra longa.

É por isso que o ajuste tem inércia, e não é por cautela genérica: um erro de um
degrau se corrige sozinho em quatro turnos, e nenhum turno sozinho move nada.

#### 1. A rubrica não estima C2 — porque tentar piorava

A primeira versão tinha corte de C1→C2 em 0,82, depois 0,92. Medido: **três das
oito amostras B2 iam para C2**, errando por dois degraus, porque o score satura
fácil (uma frase longa com um conectivo já encosta no teto). O corte saiu.

Consequência implementada: C2 é nível **autodeclarado**, e os sinais não movem
quem está nele — sobre um C2 esta rubrica não tem o que dizer, nem para cima nem
para baixo. Era isso ou fingir uma precisão que o método não tem.

#### 2. A contração era o maior buraco, e não o peso dos sinais

A primeira rodada deu 44% de acerto, e a inspeção mostrou por quê: as amostras
B1 e B2 vinham com **zero** estruturas avançadas. O padrão de aspecto perfeito
pedia `have|has|had`, e o aluno escreve `I've been working`. Tratar a contração
subiu o acerto mais que qualquer ajuste de peso depois.

#### 3. A calibração sem restrição chegava a uma regra degenerada

A busca em grade livre atinge **88%** de acerto — dando 0,8 do peso a
comprimento de frase e palavra longa, ou seja, "texto mais longo é nível mais
alto". Vale na amostra e quebra no primeiro C1 que escreva uma linha curta e
certeira. A calibração final impõe piso de 0,15 por sinal e aceita 78%: 10
pontos de acerto na amostra pagos por uma rubrica que mede quatro coisas em vez
de uma.

#### 4. O preço da inércia, medido

`--hysteresis` simula um aluno cujas mensagens são todas do mesmo nível e conta
turnos até o perfil chegar lá:

| Declarado | Escreve como | Caminho | Turnos |
|---|---|---|---|
| A2 | B1 | A2 → B1 | 4 |
| A2 | C1 | A2 → B1 → B2 → C1 | 12 |
| B1 | B2 | B1 → B2 | 4 |
| C1 | B1 | C1 → B2 → B1 | 8 |
| C1 | B2 | — | **nunca** |

A simulação acima zera a janela a cada degrau, e foi ela que expôs um bug: o
código **não** zerava. Os mesmos quatro turnos que moveram o nível continuavam
na janela e o moviam de novo no turno seguinte — quatro turnos bons levariam um
A2 a C1 em seis. `level_change.at_signal_id` marca até onde a janela já foi
consumida, e declarar o nível na mão também a fecha: senão a evidência
acumulada antes da escolha do aluno a desfaria no turno seguinte, que é o
oposto de ele ter a última palavra.

Quatro turnos por degrau é o custo de não oscilar. A última linha é o preço da
confusão B2/C1: como metade das amostras B2 lê como C1, a janela nunca fecha
para baixo, e um C1 declarado que escreve como B2 fica em C1. É dívida
conhecida; a saída é a rubrica melhorar no meio da escala, não a janela
encurtar.

#### 5. O nível muda a resposta, e dá para ver

Mesma fala do aluno, mesmo modelo (`qwen2.5:3b-instruct-q4_K_M`), só mudando o
partial de nível. Entrada: *"I have 35 years and I work with marketing.
Yesterday I go to a meeting with my boss."*

| Nível | Palavras por frase | Resposta |
|---|---|---|
| A2 | 6,5 | *You said "I have 35 years." In English, age uses to be: "I am 35 years old." How was your day? Did you have any meetings?* |
| B1 | 7,7 | *…"I am 35 years old." How was your day at work?* |
| B2 | 8,3 | *You said "I go to a meeting." In English, we use "I went to a meeting." So — what did you and your boss discuss?* |
| C1 | 10,0 | *…"I went to a meeting" for the past. So — what marketing projects are you working on right now?* |

O efeito é real e modesto, e vale dizer as duas coisas. A frase cresce
monotonicamente com o nível (6,5 → 7,7 → 8,3 → 10,0), e **o erro escolhido
muda**: em A2 e B1 o 3B corrige o decalque de idade, que é o erro que trava
comunicação; em B2 e C1 ele deixa isso de lado e corrige o tempo verbal, com
pergunta de seguimento mais específica. O que o 3B não faz é a parte mais
ambiciosa dos partials — idioma, nuance, discordar do aluno em C1. Isso é limite
do modelo: o ajuste chega ao prompt, o prompt chega ao modelo, e o modelo entrega
o que um 3B entrega.

E a mesma sonda num 7B (`qwen2.5:7b-instruct-q4_K_M`) desmente qualquer
conclusão fácil sobre *direção*: lá a resposta de A2 saiu com **70** palavras e a
de C1 com **58** — ou seja, mais longa no nível que pede frase curta, porque o
7B resolveu explicar as duas correções com calma para o A2. O que as duas
medições concordam é que **o ajuste chega e muda a saída**; o quanto e para que
lado é propriedade do modelo, não do partial. (De passagem, o 7B levou 28,9s num
turno já aquecido contra 7,8s do 3B — a decisão 1.3 continua de pé.)

Verificado no navegador, com o provider falso para a mecânica da tela: quatro
turnos escritos como um B2 moveram o badge de B1 para B2, com a nota *"Charles
moved you up to B2."* no rodapé e a borda do seletor mudando de cor para marcar
que o nível agora é estimado e não declarado. Trocar o nome no cabeçalho troca
de perfil — `ana` entra em B1 com a tela limpa, voltar para `student` traz B2 e
o histórico de volta — e escolher o nível na mão volta a marcá-lo como
autodeclarado.

#### 6. Perfil ativo é contador, não relógio

São vários alunos numa máquina (o nome no cabeçalho troca de perfil), sem login:
multiusuário continua fora da Fase 1. O perfil ativo é o visto por último, e as
duas primeiras implementações disso estavam erradas — `datetime('now')`
(resolução de segundo) e depois `strftime` com milissegundo. Um teste de
troca-e-volta de perfil quebrou nas duas: as escritas caem no mesmo instante, o
desempate por id devolve sempre o mesmo aluno. `student.seen_seq` é um contador,
não depende de relógio nenhum, e continua uma instrução SQL autônoma — que é o
que a conexão compartilhada exige.

#### O que ficou ruim, dito por inteiro

**32 amostras escritas à mão não são um corpus.** As constantes foram calibradas
contra elas, então 78% é o acerto na suíte que as escolheu — não uma medida
independente. O que protege contra overfit é a restrição do piso por sinal e o
teste que proíbe erro de dois degraus; o que resolveria de verdade é recalibrar
sobre turnos reais, e `level_signal` guarda os números de todo turno justamente
para isso ser possível depois.

**Nenhuma amostra C2, de propósito.** Ver item 1.

**A penalidade por erro quase não foi exercitada.** As amostras não têm
correção associada, então o peso do erro (0,35) está escolhido por
proporcionalidade — um erro a cada 12 palavras derruba um degrau —, e não
medido. É o primeiro número a revisar quando houver turno real no banco.

**O nível novo só vale no turno seguinte.** O turno que move o nível já foi
gerado com o registro antigo. Refazer a resposta custaria o turno inteiro para
ganhar um ajuste que o aluno não pediu.

### 5.4 O que a E6 mediu, e o que a medição mudou no desenho

Modo não é um adjetivo no prompt. Cada um dos três muda **três** coisas, e só a
primeira é texto:

| | conversation | correction | drill |
|---|---|---|---|
| Prompt de comportamento | `prompts/modes/conversation.md` | `…/correction.md` | `…/drill.md` |
| Teto de correções estruturadas por turno | 2 | **5** | 2 |
| Regra treinada | — | — | escolhida do error log **do aluno** |

O teto por modo não é detalhe de configuração: no modo de correção o aluno
abriu o modo justamente para receber os erros, e devolver dois de cinco seria
esconder dele o que ele veio buscar. É a mesma ideia da E4 — política no
código, não no prompt.

Reproduzir:

```bash
uv run python scripts/measure_modes.py            # a mesma fala nos três modos
uv run python scripts/measure_modes.py --turns 3  # um drill andando
```

#### A regra do drill é consulta ao banco, não pedido ao modelo

O critério de pronto da E6 é *"o drill usa erros de fato cometidos, não
genéricos"*. "Crie um exercício a partir dos erros do aluno" devolveria
exercício plausível sobre erro inventado — o problema nº 1 da análise inicial
outra vez, agora em forma de exercício.

Então o caminho é o da E4: `correction` agregada por `rule_id` (só correção com
regra curada, pelo motivo da §5.2), a regra mais frequente que ainda existe na
base, o item curado dali — explicação revisada e formas certas — e as frases que
o **aluno** escreveu, tiradas do banco. O modelo recebe tudo pronto e faz o que
faz bem: conversar.

Uma regra por **sessão**, não por turno. Pedagogia: drill que troca de regra a
cada mensagem é questionário. Técnica: bloco estável dentro da sessão é prefixo
cacheado — uma regra por turno pagaria a reavaliação inteira a cada turno
(§1.4). Trocar de modo abre sessão nova, porque modo é propriedade da sessão
desde a E0 e a transcrição de um drill não é continuação da conversa livre.

Quando o aluno não tem erro registrado, o drill **diz isso**. Não há exercício
genérico de reserva: seria exatamente o que o critério proíbe, e o aluno nem
saberia que o "erro dele" não era dele.

#### O que a medição mudou: o pedido tem que ficar no fim do prompt

Primeira versão, com o drill inteiro descrito no bloco de dados do topo. Mesmo
aluno, mesmo erro plantado, `qwen2.5:3b-instruct-q4_K_M`, três turnos:

> **drill, antes** — *"I have 35 years and I work with marketing. Can I ask a
> question about the report?"* → depois: *"Yes, this is correct. How was your
> weekend?"*

O modelo corrigia a fala do aluno e ia embora conversar. **Nenhum exercício em
três turnos.** O dado estava certo — regra certa, palavras do aluno no prompt —
e o comportamento não acontecia.

A correção é a mesma lição que a E4 já tinha dado com a dica curada:
**instrução longe do ponto de geração se dilui**. O bloco de dados continua no
topo, onde é estável e cacheado; o pedido virou uma linha no fim do prompt,
depois da fala do aluno:

> `Now: react in one line to what the student just wrote, then give them ONE
> short exercise on calque.age-have and stop. Wait for their answer.`

> **drill, depois** — *"Good start! Now let's practice the rule on age: 'I am 35
> years old' or 'I'm 35.' Now, try this: 'She is 25 and she works in marketing…'"*
> → depois: *"Yes, 'I am 35 years old' is correct! How about this: …"*

Exercício em todos os turnos, sempre sobre a regra treinada.

> **Duas frases acima estão erradas, e a E7 as corrigiu.** O pedido virou uma
> mensagem de sistema no fim da *lista*, que não é o fim do *prompt*: o
> template iça as mensagens de sistema para o topo. Então ele nunca esteve
> "depois da fala do aluno", e "não custa cache nenhum" era o oposto da
> verdade — custava o cache do histórico inteiro. O que a medição desta seção
> mostra, e continua valendo, é que **um pedido explícito por turno** faz o
> drill acontecer. Desde a E7 ele vai dentro da fala do aluno, onde de fato
> fica no fim do prompt. Ver §5.5.

#### Os três modos, medidos

Mesma fala — *"I have 35 years and I work with marketing. Can I make a question
about the report?"* —, mesmo modelo, mesmo error log plantado:

| Modo | Palavras | Trechos citados | O que fez |
|---|---|---|---|
| conversation | 20 | 1 | Corrigiu a idade e devolveu uma pergunta sobre o assunto |
| correction | 33 | 4 | Corrigiu **os dois** erros, cada um com a forma certa |
| drill | 36 | 3 | Ignorou o assunto, treinou a regra do error log, pediu exercício |

Os três são distintos no que importa: quantos erros a prosa traz, e se a
mensagem termina em conversa ou em tarefa. O teste de integração afirma o
contrato (três respostas diferentes para a mesma fala); esta tabela é o que o
contrato significa na prática.

Verificado no navegador com o modelo real, nos três modos. Em conversa, *"I
have 35 years and I work with marketing"* recebeu a anotação estruturada
`calque.age-have`. O botão **Drill**, clicado em seguida, abriu sessão nova com
a faixa *"Drilling: In English, age is something you are, not something you
have… calque.age-have · 1×"* — o aluno vê qual erro **dele** virou exercício — e
o turno seguinte veio com exercício sobre essa regra. Em correção, *"Yesterday I
go to a meeting and I make a question for my boss"* recebeu as duas correções na
prosa (tempo verbal e decalque) mais a anotação estruturada da regra curada.

#### O que a medição mudou outra vez: "translate" não podia estar no prompt

Rodando a mesma suíte num 7B, o drill obedeceu bem demais a uma palavra do
`drill.md`: *"ask them to translate a situation"*. Ele escreveu:

> *"Now, can you translate this situation: 'Ela tem 28 anos.' (She has 28
> years) to the correct English form?"*

Português na tela do aluno, que é a decisão 1.1 do plano. O prompt pedia, o
modelo entregou — e a culpa é de quem escreveu o prompt. A palavra saiu, e
entrou no lugar uma proibição explícita: *"Write the exercise in English,
always. Never give the student a Portuguese sentence to translate."*

Fica o padrão, que já apareceu na E4 e agora de novo: **todo verbo no prompt é
uma permissão.** Modelo maior não erra menos; erra com mais convicção.

Remedido no mesmo 7B depois da troca, três turnos de drill: nenhum português.
O item que antes vinha como *"traduza 'Ela tem 28 anos'"* virou *"let's try a
new sentence: 'She has 28 years,' and say it using 'is' instead"* — mesmo
exercício, inteiro em inglês, que é o que a decisão 1.1 pede.

#### O 7B é mais fluente e mais errado

A mesma suíte no 7B dá a melhor ilustração que este projeto tem do seu próprio
princípio. No modo de correção, sobre *"I have 35 years"*, ele escreveu:

> *"You could say 'I'm 35' or 'I have 35 years old' if you prefer to be more
> formal."*

Inventou uma regra de registro inexistente e ensinou o decalque como
alternativa formal. É o problema nº 1 da análise inicial num modelo duas vezes
maior — e é exatamente por isso que a correção que **conta como dado** exige
regra curada (§5.2) e que o drill tira a regra do banco em vez de pedir ao
modelo. A prosa é do modelo; a estrutura não pode ser.

#### O que ficou ruim, dito por inteiro

**O modo de correção herda a formatação da persona, e ela atrapalha.** O 3B
copia o `>` do exemplo de correção da persona e às vezes solta um bloco de
citação no meio da resposta. A persona foi escrita para conversa livre, onde há
uma correção por turno; com cinco, o formato dela não escala. É prompt a
ajustar, e não foi ajustado nesta entrega.

**Os exercícios do drill são simples demais.** O 3B varia pouco: repete o mesmo
item com palavras trocadas. O `drill.md` pede variedade (corrigir, traduzir,
completar, responder) e o modelo entrega sobretudo "corrija esta frase". Limite
do modelo, e o lugar natural de medir de novo é com um modelo melhor só para
esta etapa — o mesmo argumento da §5.2 sobre a análise.

**O rodízio é por sessão, então é fácil de furar.** Quem trocar de modo
repetidamente recebe regras diferentes a cada vez, o que é o desejado, mas
quem ficar na mesma sessão treina a mesma regra para sempre. Sessão longa de
drill com várias regras é item da Fase 2, junto com a repetição espaçada.

**No modo de correção, o 7B repete o erro do aluno dentro da sugestão.** Na
remedição ele escreveu *"Here's how you might say it: 'I have 35 years and I
work with marketing. Can I ask a question about the report?'"* — corrigiu o
decalque da pergunta e deixou o da idade **dentro da frase modelo**. O aluno lê
uma frase apresentada como certa e com o erro dele preservado. É o mesmo
argumento de sempre: a prosa do modelo não é garantia de nada, e quem protege é
a anotação estruturada, que nesse turno citou a regra certa.

**O 3B prefere o exemplo curado ao do aluno.** O bloco leva as frases que o
aluno escreveu (*"I have 35 years"*), e na verificação no navegador o modelo
montou o exercício com o exemplo da base (*"I have 30 years"*). A intenção do
`drill.md` — fazer o aluno reconhecer as próprias palavras — não se cumpre
sempre. O dado chega; o uso dele é do modelo.

**Nada mede se o drill ensina.** Medimos que ele treina a regra certa, com as
palavras certas, e que o modelo dá exercício. Se o aluno erra menos depois, não
sabemos — é pergunta que precisa de uso real e de mais de uma sessão por aluno.
Fica anotada para a E7, onde o error log agregado passa a existir na tela.

### 5.5 O que a E7 mediu, e as duas dívidas que ela pagou

A E7 tinha duas dívidas nominais, as duas sobre latência e as duas sobre
**por quê**:

- §1.4: *"7,4s no primeiro turno mesmo com aquecimento, contra 0,7s nos
  seguintes. Não está explicado."*
- §5.2: *"a primeira palavra degrada turno a turno (0,1s → 3,8s → 6,1s → 7,9s →
  10,8s em 5 turnos)"*, com a hipótese de que **a chamada de análise** joga o
  prompt da conversa fora do cache de prefixo do Ollama.

Cronômetro não responde nenhuma das duas. O que responde é o que o servidor
conta de cada chamada — quanto do prompt estava em cache, quanto foi avaliado
agora, quanto custou carregar o modelo. Daí `ChatProvider.last_metrics` e
`scripts/measure_latency.py`.

Modelo: `qwen2.5:3b-instruct-q4_K_M`, CPU pura, sessão de 6 turnos alternando
frase com erro e frase correta.

#### Resultado final

| | |
|---|---|
| Primeira palavra | mediana **2,7s** · pior 3,4s |
| Primeira palavra, 2 primeiros turnos vs 2 últimos | 2,8s → **2,7s** (não degrada mais) |
| Turno completo, frase correta | **4,1s** |
| Turno completo, frase com erro (inclui análise) | 12,5–16,2s |
| Análise, isolada | mediana 10,1s, **depois** da resposta |
| Prompt da conversa em cache | 88% → **95%**, subindo ao longo da sessão |
| Primeiro turno, modelo já carregado e aquecido | **0,9s** |

#### Dívida 1: o primeiro turno, explicado e resolvido

Com o modelo descarregado da memória, medindo um turno cru:

| | primeira palavra | decomposição |
|---|---|---|
| Sem aquecimento | **19,1s** | 4,1s de carga do modelo + 15,1s avaliando 436 tokens com 0% em cache |
| Com aquecimento | **0,9s** | o aquecimento pagou os 4,1s + 13,6s; o turno chegou com 94% em cache |

Os 7,4s que a §1.4 não explicava eram **aquecimento parcial**: ele mandava um
prompt parecido com o do turno, não o mesmo. E a E5 e a E6 pioraram isso sem
querer — acrescentaram ajuste de nível e prompt de modo ao turno, e o
aquecimento continuou mandando só a persona. Medido antes da correção: primeiro
turno com **53% do prompt em cache e 11,6s** até a primeira palavra.

A correção não foi acrescentar as duas mensagens ao aquecimento; foi tirar a
duplicata. `TurnRunner.stable_prefix()` é a única definição do prefixo estável,
o turno a usa e o aquecimento a usa. Não há mais como as duas listas
divergirem — e um teste falha se alguém tentar.

#### Dívida 2: a hipótese da §5.2 estava errada

A degradação era real. A causa, não. Três sessões iguais, mudando uma coisa de
cada vez:

| | cache da conversa | primeira palavra (1º → 6º turno) |
|---|---|---|
| Sem análise e sem dica curada | 97% estável | 1,0s → 1,0s |
| Análise ligada, mesmo modelo | 88% → 72% | 2,8s → 8,7s |
| Análise ligada, **outro modelo** (7B) | 88% → 72% | 2,8s → 9,3s |
| Só a dica curada, **sem chamada** de análise | 88% → 71% | 2,8s → 9,1s |

A terceira linha derruba a hipótese: com a análise num modelo separado — cache
separado, slot separado — a conversa degrada exatamente igual. A quarta mostra
o culpado: **a dica curada**, sem chamada de análise nenhuma, reproduz a
degradação inteira.

E o número que fecha o caso: em todas as sessões degradadas, o total cacheado
ficava **preso em ~674 tokens** enquanto o prompt crescia de 767 para 950. 674
é exatamente o prefixo estável. Ou seja: o histórico da conversa nunca entrava
no cache.

O que isso evitou, de passagem: a E7 **não** ganhou um `TC_ANALYSIS_MODEL`. Era
a feature que a §5.2 encomendava, estava pronta para ser escrita — o `Analyzer`
já aceita provider próprio — e teria custado configuração nova, um segundo
modelo na memória e uma análise 2,5× mais lenta para resolver um problema que
ela não causava. Duas horas de medição valeram mais que a feature.

A explicação não é o cache do Ollama, é o **template de chat**: ele junta as
mensagens de sistema no topo. A dica era a última mensagem da lista, mas não a
última do prompt renderizado — ela era colada logo depois da persona. Como
muda a cada turno, invalidava tudo o que vinha depois: o histórico inteiro.

> A E4 pôs a dica no fim da lista *por causa do cache*, e mediu que isso
> ajudava a prosa do professor a usar a redação curada. As duas conclusões
> estavam certas pelos motivos errados: a dica nunca esteve no fim do prompt.

**A correção**: a dica e o pedido do drill passam a ir **dentro da última
mensagem do aluno**, entre colchetes. Ali ficam de fato no fim do prompt
renderizado, e tudo o que vem antes volta a ser cacheado.

| | cache | primeira palavra (1º → 6º) |
|---|---|---|
| Dica como mensagem de sistema | 88% → 71% | 2,8s → 9,1s |
| Dica dentro da fala do aluno | 88% → **92%** | 3,0s → **2,9s** |

E a prosa melhorou junto, que é o efeito colateral que ninguém esperava. Mesmas
três frases, mesmo modelo:

| O aluno escreveu | Com a dica como sistema | Com a dica na fala do aluno |
|---|---|---|
| *I lost the bus this morning* | não corrigiu | *"We usually say 'I missed the bus'"* ✅ |
| *Can I make a question* | não corrigiu | não corrigiu |
| *My manager is very comprehensive* | *"My manager is very **thorough**"* ❌ | *"Your manager is very **understanding**"* ✅ |

Faz sentido: a instrução que o template empurrava para junto da persona ficava
a 900 tokens do ponto de geração. Dentro da fala do aluno, ela está na última
linha do prompt.

Com isso, o produto inteiro ligado — análise incluída — a primeira palavra sai
em 2,7s e **não degrada**: 2,8s no começo da sessão, 2,7s no fim, com o cache
subindo de 88% para 95%.

#### O outro critério: um dev clona e roda seguindo só o README

Verificado como se verifica isto: uma cópia limpa do projeto — sem `.venv`, sem
`.env`, sem caches — e os comandos do README, na ordem em que ele os dá.

| | |
|---|---|
| `python3 scripts/check_env.py` | roda com o Python do sistema; avisou que o modelo default não estava baixado e **listou os que havia na máquina, com tamanho** |
| `uv sync --extra dev` | resolveu o Python 3.12 e as dependências |
| `uv run pytest -m "not integration"` | 508 testes verdes em 4,5s, sem modelo nenhum |
| `uv run ruff check .` | limpo |

O README ganhou o que faltava para isso ser verdade sem ajuda: como trabalhar
sem modelo (`TC_PROVIDER=fake`), o loop rápido de teste em vez do `pytest`
puro, uma tabela de medições apontando cada número a um script, e uma seção
"quando dá errado" com os sintomas que o produto de fato produz — 503 no
`/health`, correção que não aparece porque não há regra curada, turno de 15s
em frase com erro.

#### O que ficou ruim, dito por inteiro

**Frase com erro ainda custa um turno de 12 a 16 segundos.** A análise leva
~10s e roda depois da resposta, então o aluno lê enquanto ela corre — mas a
anotação demora. O caminho é a análise num modelo melhor e mais rápido, e a
medição da §5.5 mostra que isso **não** piora a conversa (o cache dela é
independente). Fica para a Fase 2.

**A nota entre colchetes está na fala do aluno.** Funciona e não vaza para a
tela — verificado —, mas é um lugar estranho para instrução de professor, e um
modelo diferente pode reagir diferente. Se vazar algum dia, o sintoma será
visível na hora.

**2,7s ainda não é conversa.** A Fase 1.5 quer 1–2s com voz, e a medida sem
dica (1,0s) mostra que o teto desta máquina dá para isso — o que falta é a
dica custar menos, ou um modelo mais rápido.

**Um 3B, não o 4B default.** Os números acima são do modelo que está baixado
nesta máquina. Um 4B deve ficar ~30% mais lento, pela §1.3; não foi medido.

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
| Latência do turno duplo (análise + resposta) | **Materializou-se, e foi resolvido em duas etapas.** A E4 mediu: em paralelo, a primeira palavra sai em 15,3s porque o Ollama atende um pedido por vez — o paralelismo era ficção. Default virou conversa primeiro, análise depois, e turno sem candidato na base não chama o modelo. A degradação turno a turno que sobrou foi diagnosticada errado ali e corrigida na E7: era a dica de sistema sendo içada ao topo pelo template, não a análise. Hoje a primeira palavra fica em 2,7s e não cresce ao longo da sessão (§5.5). |
| Curar ~100 itens contrastivos dá trabalho | E3 entrega 40 e segue crescendo. O schema e os testes garantem que crescer não degrada. |

---

## 7. Fora do caminho crítico, registrado para depois

- **Fase 1.5 — feita.** Voz (STT e TTS) local, em
  [plano-fase-1-5.md](plano-fase-1-5.md), com a medição que a §1.3 exigia como
  pré-requisito e as duas dívidas que ela deixou: a latência de 5,4–7,4 s e a
  exatidão do STT com sotaque brasileiro, ainda não medida.
- **Fase 2**, como na análise inicial: avaliação de pronúncia por fonemas (wav2vec2 +
  alinhamento forçado), repetição espaçada, testes em hardware mais modesto.
- Embeddings na base contrastiva, se ela passar de algumas centenas de itens.
- Empacotamento desktop (Tauri) reaproveitando este backend.
- Reabrir o público A1/A2 com apoio em português — a decisão 1.1 é reversível, e o
  `prompts/partials/` é justamente o lugar onde esse apoio entraria.
