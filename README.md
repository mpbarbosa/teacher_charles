# Teacher Charles

Um professor de inglês que roda **localmente** na sua máquina. Sem API paga, sem
mandar suas conversas para a nuvem.

O aluno conversa em inglês; o professor corrige, explica e acompanha o nível. A
persona sabe que o aluno é brasileiro e antecipa os erros típicos de quem vem do
português — mas explica em inglês.

> **Estado: Fase 1, entregas E0–E4.** Dá para conversar em inglês de ponta a
> ponta no navegador, e cada turno produz correção estruturada — com a regra
> curada citada, persistida em SQLite e visível ao lado do que o aluno escreveu.
> O perfil por nível, os modos e a tela de error log ainda não existem. Veja
> [docs/plano-fase-1.md](docs/plano-fase-1.md) para o escopo e a ordem das
> entregas.

## Decisões que moldam o projeto

| | |
|---|---|
| **Público** | B1 em diante, 100% em inglês. Sem português na interface ou nas respostas. |
| **Plataforma** | Web local: backend Python em `localhost`, interface no navegador. |
| **Hardware mínimo** | CPU / iGPU, sem CUDA. O modelo default é 4B-class quantizado. |

O raciocínio por trás de cada uma está em [docs/plano-fase-1.md](docs/plano-fase-1.md),
e a análise de viabilidade que originou o projeto em
[docs/analise-inicial.md](docs/analise-inicial.md).

O princípio que organiza o código: **a estrutura pedagógica mora na aplicação, não
no prompt.** Correção é dado estruturado — com regra citada, persistida e contável —
não prosa que o modelo improvisa.

## Requisitos

- Python 3.12 (o `uv` resolve isso por você)
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) rodando localmente
- ~8 GB de RAM e ~3 GB de disco para o modelo

## Rodar em desenvolvimento

Antes de qualquer coisa, pergunte à sua máquina se ela está pronta. Este script
roda com o Python do sistema, sem precisar instalar nada:

```bash
python3 scripts/check_env.py
```

Ele confere RAM, disco, `uv`, se o Ollama responde e se o modelo está baixado. Se
o modelo configurado não estiver lá, ele lista os que você já tem.

Depois:

```bash
uv sync --extra dev
```

```bash
ollama pull qwen3:4b
```

```bash
uv run teacher-charles --reload
```

Abra <http://localhost:8000> e comece a conversar. O endpoint
<http://localhost:8000/health> diz, em JSON, se o banco migrou e se o modelo
está ao alcance.

O modelo é aquecido em segundo plano na subida, com a persona, para o primeiro
turno não pagar a carga do modelo nem a avaliação do prompt. Medido: sem isso o
primeiro turno leva ~19s até a primeira palavra; com isso, ~7s, e os turnos
seguintes ~0,7s. Desligue com `TC_LLM_WARMUP=false`.

### Já tem outro modelo baixado?

Não precisa baixar o `qwen3:4b` só para experimentar. Aponte para o que você tem:

```bash
TC_LLM_MODEL=qwen2.5:3b-instruct-q4_K_M uv run teacher-charles
```

Um 3B dá para desenvolver. Para uso de verdade, 4B é o default por um motivo: nesta
classe de hardware o gargalo é banda de memória, e a vazão cai rápido com o tamanho.
Medido nesta máquina (Intel Core Ultra 5 135U, **Ollama em CPU pura** — o pacote oficial
não usa a iGPU Intel): **9,3 tok/s num 3B e 4,0 num 7B**, o que põe um 4B em ~7–8 tok/s
e um 8B em ~3,5. Com streaming, 4B é tolerável para texto; 8B não é.
Ver [docs/plano-fase-1.md](docs/plano-fase-1.md) §1.3 para o método da medição.

## Configuração

Tudo por variáveis de ambiente com prefixo `TC_`, ou por um `.env` na raiz.
Copie o exemplo e ajuste:

```bash
cp .env.example .env
```

Os defaults estão em [`src/teacher_charles/config.py`](src/teacher_charles/config.py).
O mais útil: `TC_LLM_MODEL`, `TC_OLLAMA_URL`, `TC_PORT`, `TC_DB_PATH`.

## Testes

```bash
uv run pytest
```

```bash
uv run ruff check . && uv run ruff format --check .
```

A suíte não precisa de modelo nenhum: o `health` é exercitado com transporte HTTP
falso e o chat com um provider `fake`. Os testes marcados `integration` batem num
Ollama de verdade e se pulam sozinhos quando não há um:

```bash
uv run pytest -m integration
```

## A base contrastiva

`data/contrastive/*.yaml` é o diferencial do produto: as regras curadas que impedem
o modelo de inventar gramática. São 43 itens em três arquivos — falsos cognatos
(*pretend* / *pretender*), decalques (*I have 30 years*) e interferência estrutural
(*people is*).

Para acrescentar um item, copie um vizinho no YAML e preencha:

| Campo | O que é |
|---|---|
| `id` | `categoria.nome-curto`, e a categoria tem que bater com o campo `category` |
| `pt` | o português que causa o erro — sem isso não é uma base contrastiva |
| `wrong` / `right` | exemplos reais; os testes usam os dois lados |
| `patterns` | regex sobre a fala do aluno. Sem eles, os literais de `wrong` viram o padrão |
| `explanation` | vai para o aluno, **em inglês** |
| `ambiguous` | `true` quando o padrão também casa com inglês correto |

Então rode a suíte. Todo exemplo de `wrong` tem que casar com o próprio item, e
nenhum `right` pode casar — é assim que um padrão largo demais é pego na hora de
escrever, e não na frente do aluno:

```bash
uv run pytest tests/test_contrastive.py
```

O schema em `data/schema/` é **gerado** do modelo pydantic, nunca editado à mão:

```bash
uv run python scripts/dump_contrastive_schema.py
```

## Como um turno funciona

Cada turno do aluno tem duas etapas, nesta ordem:

```
fala do aluno
   │
   ├─ casa com data/contrastive/        (microssegundos, determinístico)
   │     └─ a regra entra no prompt da resposta como dica curada
   │
   ├─ 1. resposta do professor ─────────> streaming na tela, primeira palavra em ~1s
   │
   └─ 2. análise (JSON validado) ───────> correção anotada sobre o que o aluno escreveu
                                          e persistida em SQLite
```

Quem informa a resposta **não** é a análise, é o casamento determinístico — que
é de graça. A análise vem depois do último pedaço da resposta porque disparar as
duas juntas não paraleliza nada: o Ollama atende um pedido por vez, e medido
(plano §5.2) a primeira palavra do professor saía em 15,3s em vez de 6,1s. O
aluno lê a resposta enquanto a correção é calculada.

Três coisas não são confiadas ao prompt, porque prompt não é garantia:

| | |
|---|---|
| **Correção sem regra curada não chega ao aluno** | Medido: num 3B, 12 de 18 frases corretas recebiam correção inventada, e nenhuma citava regra — várias ensinavam erro. Exigir a regra zerou os falsos positivos sem custar nenhum acerto. |
| **A regra citada é verificada** | Se a base casou no mesmo trecho, o `rule_id` e a explicação são os da regra curada. O modelo não tem voto — e `rule_id` que não existe na base é descartado. |
| **Span que o aluno não disse é descartado** | Correção sobre palavra inventada desorienta mais que correção nenhuma. |

Daí decorre uma economia: frase que não casa com regra nenhuma **não vira
pergunta ao modelo**, porque a resposta não poderia chegar ao aluno de todo
jeito. São 37% dos turnos da suíte de medição.

JSON inválido tem uma retentativa com instrução mais estrita e, se falhar de
novo, o turno segue **sem correção** — nunca com exceção na cara do aluno. A
frequência disso é medida, não presumida (deu 0 em 49 turnos):

```bash
uv run python scripts/measure_analysis.py
```

```bash
uv run python scripts/measure_analysis.py --double-turn 5
```

O primeiro mede qualidade (precisão, falso positivo, JSON degradado) sobre as
frases de `tests/fixtures/known_errors.yaml`; o segundo mede o custo real do
turno duplo. Os números desta máquina estão no plano §5.2 — e `--allow-unverified`
mostra o que o modelo diria se a política de regra curada estivesse desligada.

Duas variáveis mudam esse comportamento, e as duas têm número por trás:
`TC_REQUIRE_CURATED_RULE=false` solta o modelo para corrigir sem regra, e
`TC_ANALYSIS_AFTER_REPLY=false` volta a disparar as duas etapas juntas (melhor
num provider que atenda dois pedidos ao mesmo tempo).

## Onde fica cada coisa

```
data/        base de erros típicos de brasileiros — VERSIONADA, é conteúdo do produto
  contrastive/ os itens curados em YAML; schema/ o JSON Schema gerado deles
prompts/     persona e instruções em .md, versionados, recarregados sem reiniciar
src/         o código
  llm/         providers: Ollama (local), OpenAI-compat (nuvem), fake (testes)
  teaching/    a estrutura pedagógica — é esta pasta que é o produto
    turn.py      orquestra o turno: resposta e análise em paralelo
    analyzer.py  fala do aluno -> correção estruturada e verificada
    contrastive.py  carrega e consulta as regras curadas
  store/       SQLite: esquema em .sql, migrações numeradas
  web/         página única: template Jinja, JS e CSS sem build step
docs/        análise, plano e decisões de arquitetura
scripts/     check_env.py (só stdlib, roda antes do uv sync) e as medições
```

Editar um arquivo de `prompts/` vale no turno seguinte, sem reiniciar o servidor.

Duas coisas **nunca** entram no repo:

- **Pesos de modelo.** Ficam onde o Ollama os guarda (`~/.ollama`).
- **Progresso do aluno.** Fica em `$XDG_DATA_HOME/teacher-charles/charles.db`.
  É dado pessoal. `data/` guarda conhecimento curado, nunca dados de aluno.

## Licença

Ver [LICENSE](LICENSE).
