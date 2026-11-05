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
| **E3** | **Base contrastiva** — schema, ~40 itens iniciais (falsos cognatos + decalques), loader validado, matcher, um teste por item | `match("I have 30 years")` devolve `calque.age-have`; o loader rejeita item fora do schema |
| **E4** | **Correção estruturada** — `analyzer.py` com saída JSON validada por pydantic, regra citada quando há match, fallback para JSON inválido, persistência das correções | Suíte de ~25 frases com erro conhecido passa; a precisão medida fica registrada neste doc |
| **E5** | **Perfil e nível** — CEFR inicial, sinais por turno, prompt ajustado por nível via `prompts/partials/level/` | Trocar o nível do aluno muda visivelmente a resposta, verificado em teste de snapshot |
| **E6** | **Modos** — os três com prompt próprio; *drill* gera exercício a partir do error log real do aluno | Os três modos funcionam na UI e o drill usa erros de fato cometidos, não genéricos |
| **E7** | **Progresso e acabamento** — tela de error log agregado por regra, README de dev, latência medida e registrada | Um aluno consegue ver no que erra mais; um dev clona e roda seguindo só o README |

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
| O 4B devolve JSON malformado na etapa de análise | Validação pydantic + retry com instrução mais estrita + fallback que degrada para "sem correção" em vez de quebrar o turno. Medido na E4. |
| O 4B explica gramática pior que o esperado | A base contrastiva é o anteparo: onde há regra curada, a explicação vem dela, não do modelo. Medido na E1: um 3B "corrigiu" *I have 30 years* para *I have thirty years* — trocou o dígito e manteve o decalque. É o problema nº 1 da análise em uma linha, e a razão de a E3/E4 existirem. Subir para 8B **não** é mais a saída nesta máquina (3,5 tok/s); a saída é a regra curada. |
| Latência do turno duplo (análise + resposta) | **Agravado pela medição da §1.3 (7–8 tok/s num 4B).** A resposta sai em streaming enquanto a análise roda, e a análise tem que devolver JSON curto. A E4 mede o custo real do turno duplo em vez de presumir. |
| Curar ~100 itens contrastivos dá trabalho | E3 entrega 40 e segue crescendo. O schema e os testes garantem que crescer não degrada. |

---

## 7. Fora do caminho crítico, registrado para depois

- **Fase 2**, como na análise inicial: avaliação de pronúncia por fonemas (wav2vec2 +
  alinhamento forçado), repetição espaçada, testes em hardware mais modesto.
- Embeddings na base contrastiva, se ela passar de algumas centenas de itens.
- Empacotamento desktop (Tauri) reaproveitando este backend.
- Reabrir o público A1/A2 com apoio em português — a decisão 1.1 é reversível, e o
  `prompts/partials/` é justamente o lugar onde esse apoio entraria.
