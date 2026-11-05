# Teacher Charles

Um professor de inglês que roda **localmente** na sua máquina. Sem API paga, sem
mandar suas conversas para a nuvem.

O aluno conversa em inglês; o professor corrige, explica e acompanha o nível. A
persona sabe que o aluno é brasileiro e antecipa os erros típicos de quem vem do
português — mas explica em inglês.

> **Estado: Fase 1, entregas E0–E2.** Dá para conversar em inglês de ponta a
> ponta no navegador. A correção estruturada com regra citada (E3/E4), os modos
> e o error log ainda não existem. Veja [docs/plano-fase-1.md](docs/plano-fase-1.md)
> para o escopo e a ordem das entregas.

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

Os testes não precisam de modelo nenhum: o `health` é exercitado com transporte HTTP
falso. Os testes de integração contra o Ollama de verdade chegam na E1.

## Onde fica cada coisa

```
data/        base de erros típicos de brasileiros — VERSIONADA, é conteúdo do produto
prompts/     persona e instruções em .md, versionados, recarregados sem reiniciar
src/         o código
  llm/         providers: Ollama (local), OpenAI-compat (nuvem), fake (testes)
  teaching/    a estrutura pedagógica — é esta pasta que é o produto
  store/       SQLite: esquema em .sql, migrações numeradas
  web/         página única: template Jinja, JS e CSS sem build step
docs/        análise, plano e decisões de arquitetura
scripts/     check_env.py (só stdlib, roda antes do uv sync)
```

Editar um arquivo de `prompts/` vale no turno seguinte, sem reiniciar o servidor.

Duas coisas **nunca** entram no repo:

- **Pesos de modelo.** Ficam onde o Ollama os guarda (`~/.ollama`).
- **Progresso do aluno.** Fica em `$XDG_DATA_HOME/teacher-charles/charles.db`.
  É dado pessoal. `data/` guarda conhecimento curado, nunca dados de aluno.

## Licença

Ver [LICENSE](LICENSE).
