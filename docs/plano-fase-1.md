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

Estimativa de vazão para este chip (LPDDR5 dual-channel, ~60–70 GB/s efetivos):

| Modelo (Q4) | Tamanho | Vazão estimada | Veredito |
|---|---|---|---|
| 4B | ~2,5 GB | ~20–25 tok/s | **Default.** Confortável, inclusive para voz. |
| 8B | ~4,7 GB | ~10–13 tok/s | Opção. Bom para texto, apertado para voz. |
| 14B | ~9 GB | ~6 tok/s | Fora. Inviável para conversa. |

**Modelo default: um 4B-class quantizado (Qwen3 4B como primeira escolha), com 8B
configurável.** A seção 5 da análise já apoia isso: sendo tudo em inglês, "um modelo de
4B a 8B explica gramática em inglês com boa confiabilidade".

A decisão 1.1 é o que torna 1.3 sustentável. Se o produto fosse bilíngue, 4B não daria
conta das explicações em português e o hardware mínimo teria que subir.

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
| **E0** | **Fundação** — pyproject/uv, pacote, SQLite + migração, `check_env.py`, FastAPI com `/health` | `uv run teacher-charles` sobe em `localhost:8000` e `/health` reporta se o modelo está acessível |
| **E1** | **Provider do LLM** — Protocol `ChatProvider`, Ollama com streaming, `openai_compat` atrás de flag, erros e timeout tratados, `fake` para testes | Um teste de integração faz um turno real contra o Ollama local; os unitários rodam sem modelo nenhum |
| **E2** | **Chat mínimo** — rota SSE, UI de página única, histórico persistido, persona vinda de `prompts/persona.md` | **Dá para conversar em inglês de ponta a ponta no navegador.** Primeiro momento em que isto é um produto |
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
| O 4B explica gramática pior que o esperado | A base contrastiva é o anteparo: onde há regra curada, a explicação vem dela, não do modelo. Se a E4 mostrar precisão baixa, subir para 8B é trocar uma linha de config. |
| Latência do turno duplo (análise + resposta) | A resposta começa a sair em streaming enquanto a análise roda; a correção aparece depois, junto da mensagem. Medido na E7. |
| Curar ~100 itens contrastivos dá trabalho | E3 entrega 40 e segue crescendo. O schema e os testes garantem que crescer não degrada. |

---

## 7. Fora do caminho crítico, registrado para depois

- **Fase 2**, como na análise inicial: avaliação de pronúncia por fonemas (wav2vec2 +
  alinhamento forçado), repetição espaçada, testes em hardware mais modesto.
- Embeddings na base contrastiva, se ela passar de algumas centenas de itens.
- Empacotamento desktop (Tauri) reaproveitando este backend.
- Reabrir o público A1/A2 com apoio em português — a decisão 1.1 é reversível, e o
  `prompts/partials/` é justamente o lugar onde esse apoio entraria.
