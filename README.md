# Teacher Charles

[![CI](https://github.com/mpbarbosa/teacher_charles/actions/workflows/ci.yml/badge.svg)](https://github.com/mpbarbosa/teacher_charles/actions/workflows/ci.yml)

Um professor de inglês que roda **localmente** na sua máquina. Sem API paga, sem
mandar suas conversas para a nuvem.

O aluno conversa em inglês; o professor corrige, explica e acompanha o nível. A
persona sabe que o aluno é brasileiro e antecipa os erros típicos de quem vem do
português — mas explica em inglês.

> **Estado: Fase 1 completa, E0–E7.** Dá para conversar em inglês de ponta a
> ponta no navegador, e cada turno produz correção estruturada — com a regra
> curada citada, persistida em SQLite e visível ao lado do que o aluno escreveu.
> O perfil tem nível CEFR: ele muda o registro das respostas e se ajusta sozinho
> a partir dos sinais de cada turno. Há três modos — conversa, correção e drill
> —, o drill treina um erro que o aluno de fato cometeu, e a tela
> `/progress` mostra no que ele mais erra, com as frases dele. Veja
> [docs/plano-fase-1.md](docs/plano-fase-1.md) para o escopo, a ordem das
> entregas e o que cada uma mediu.
>
> **Fase 1.5 (voz) também completa.** O aluno segura um botão e fala; o Charles
> ouve, responde e fala de volta, sentença por sentença — tudo local, sem
> nuvem. A meta de latência **não** foi atingida: são 5,4–7,4 s até a primeira
> palavra falada, contra 1–2 s projetados, e a maior fatia é o modelo de
> linguagem, não o áudio. Os números e o que eles mudaram no desenho estão em
> [docs/plano-fase-1-5.md](docs/plano-fase-1-5.md).
>
> **Pronúncia na tela.** Toda fala escrita — a do aluno e a do Charles — sai com
> a transcrição fonética em IPA embaixo dela. Vem do espeak-ng, não do modelo: é
> o mesmo G2P que a voz usa para falar, então o símbolo que o aluno lê é o som
> que ele ouve. Desligável no cabeçalho, e por configuração em
> `TC_PHONETICS_ENABLED`.

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
- O IPA embaixo de cada fala já vem na instalação base: são ~21 MB de wheels
  (`phonemizer` e o espeak-ng do `espeakng_loader`), sem peso para baixar
- Para a voz (opcional): mais ~400 MB de wheels e ~120 MB de pesos, e um
  microfone

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
<http://localhost:8000/health> diz, em JSON, se o banco migrou, se o modelo
está ao alcance, se a base contrastiva carregou e se algum prompt de nível ou
de modo está faltando.

Para parar, o **Shut down** do cabeçalho faz o mesmo que o Ctrl-C no terminal:
pede a parada com um SIGINT, então o desligamento passa pelo `finally` do
lifespan — provider e voz fechados, conexão do SQLite fechada — em vez de o
processo sumir. Ele existe porque quem sobe o Charles por `scripts/run.sh` e
estuda no navegador não tem o terminal à vista. O servidor responde antes de
parar, e a página fica legível e inerte; o Charles só volta pelo terminal.
A rota é `POST /api/shutdown`, e um `curl -X POST
http://localhost:8000/api/shutdown` funciona igual — o que ela recusa é POST
vindo de **outra** página do navegador, que sem isso desligaria a aula de fora.
Com `--reload` o sinal chega ao processo que serve, e o supervisor do reload
fica vivo sem filho: lá o Ctrl-C ainda é o jeito certo.

Se preferir um atalho para os modos acima, `scripts/run.sh` monta o ambiente de
cada um e sobe o servidor:

```bash
scripts/run.sh --help
```

```bash
scripts/run.sh --check --reload
```

Ele **não** roda `uv sync` por conta própria: `uv sync` poda o que os extras
pedidos não nomeiam, então sincronizar só com `--extra dev` desinstalaria a voz
— e com ela o `onnxruntime` de que as medições da Fase 2 dependem.
Sincronizar é opt-in, com `--sync`, e `--sync --voice` mantém os dois extras.

Para mexer na interface sem modelo nenhum — a tela responde na hora, com um
professor de mentira:

```bash
TC_PROVIDER=fake uv run teacher-charles --reload
```

O modelo é aquecido em segundo plano na subida, mandando **exatamente** o
prefixo que um turno real manda — persona, ajuste de nível e prompt do modo.
Medido com o modelo descarregado (§5.5): sem aquecimento, a primeira palavra do
primeiro turno leva 19,1s (4,1s carregando o modelo, 15,1s avaliando o prompt);
com aquecimento, 0,9s. Desligue com `TC_LLM_WARMUP=false` se quiser ver a
diferença.

### Já tem outro modelo baixado?

Não precisa baixar o `qwen3:4b` só para experimentar. Aponte para o que você tem:

```bash
TC_LLM_MODEL=qwen2.5:3b-instruct-q4_K_M uv run teacher-charles
```

Um 3B dá para desenvolver. Para uso de verdade, 4B é o default por um motivo: nesta
classe de hardware o gargalo é banda de memória, e a vazão cai rápido com o tamanho.
Medido nesta máquina (Intel Core Ultra 5 135U, **Ollama em CPU pura**): **9,3 tok/s num
3B e 4,0 num 7B**, o que põe um 4B em ~7–8 tok/s e um 8B em ~3,5. Com streaming, 4B é
tolerável para texto; 8B não é.
Ver [docs/plano-fase-1.md](docs/plano-fase-1.md) §1.3 para o método da medição.

**A iGPU Intel muda metade dessa conta — ver [a seção da iGPU](#a-igpu-intel-opcional-e-recomendada).**
A avaliação de prompt fica 13× mais rápida, e é ela que domina a espera pela primeira
palavra; a geração empata num 3B e ganha 50% num 7B, mas sem chegar ao piso do
tolerável.

## Configuração

Tudo por variáveis de ambiente com prefixo `TC_`, ou por um `.env` na raiz.
Copie o exemplo e ajuste:

```bash
cp .env.example .env
```

Os defaults estão em [`src/teacher_charles/config.py`](src/teacher_charles/config.py).
O mais útil: `TC_LLM_MODEL`, `TC_OLLAMA_URL`, `TC_PORT`, `TC_DB_PATH`.

Da voz: `TC_VOICE_ENABLED`, `TC_STT_MODEL`, `TC_TTS_VOICE`, `TC_TTS_MODEL`,
`TC_STT_CPU_THREADS`. Cada default tem uma medição atrás dele — ver a seção
"Falar com o Charles".

### A iGPU Intel: opcional, e recomendada

Numa máquina com gráficos Intel integrados (Arc / Meteor Lake e parentes), o
Ollama 0.35 já traz um backend Vulkan no pacote — não é preciso compilar
`llama.cpp` nem instalar IPEX-LLM. Duas variáveis ligam:

```bash
OLLAMA_VULKAN=1 OLLAMA_IGPU_ENABLE=1 ollama serve
```

Precisa só do driver Mesa com o ICD Vulkan da Intel (`intel_icd.json`, que vem
no `mesa-vulkan-drivers`) e de acesso a `/dev/dri/renderD128`. Para conferir
que pegou, o log da subida traz `library=Vulkan ... type=iGPU`, e o `/api/ps`
passa a reportar `size_vram` diferente de zero.

Medido nesta máquina, com `qwen2.5:3b-instruct-q4_K_M`:

| | CPU pura | iGPU Vulkan |
|---|---|---|
| Geração, 3B | 6,2 tok/s | 5,6 tok/s — **empata** |
| Geração, 7B | 2,8 tok/s | 4,2 tok/s — **1,5×** |
| Avaliação de prompt (~1k tokens, 3B) | 12,8 tok/s | **168 tok/s** |
| Primeira palavra do turno | 7,4s | **0,9s** |
| Turno de voz até ouvir a 1ª palavra | 6,3s | **4,0s** |

**O que a iGPU compra é avaliação de prompt** — e como o prompt do turno tem
~900 tokens (persona, nível, modo, histórico), é ela que domina a espera pela
primeira palavra. Por isso o ganho no turno é bem maior que o ganho em qualquer
número de vazão isolado.

Geração é outra história: num 3B empata (os dois batem no mesmo teto de banda
de memória — a Arc integrada usa a mesma DRAM que a CPU), e num 7B ganha 50%,
porque aí os 14 threads do chip é que ficam para trás. Mesmo assim os 4,2 tok/s
do 7B seguem abaixo do piso de ~7–8 tok/s: **a iGPU encolhe a punição por
modelo grande, não a remove**, e o default de 4B-class continua valendo.

Nada no produto muda: o provider continua sendo o `ollama`. Se o servidor da
iGPU estiver noutra porta, basta `TC_OLLAMA_URL`.

O método e a íntegra dos números estão em
[docs/plano-fase-1-5.md](docs/plano-fase-1-5.md) §7, e o script que os produz é
`scripts/measure_igpu.py`.

## Testes

O loop de desenvolvimento é este, e ele não precisa de modelo nenhum:

```bash
uv run pytest -m "not integration"
```

```bash
uv run ruff check . && uv run ruff format --check .
```

São ~500 testes em poucos segundos: o `health` é exercitado com transporte HTTP
falso e o chat com um provider `fake`, que devolve respostas roteirizadas e
guarda o que lhe foi pedido.

Os marcados `integration` batem num Ollama de verdade — eles verificam o
contrato (o prompt chega, o JSON volta, o turno fecha), nunca a qualidade do
modelo, que é assunto dos scripts de medição:

```bash
uv run pytest -m integration
```

Eles se pulam sozinhos quando não há Ollama, então `uv run pytest` sem filtro
fica verde numa máquina sem nada instalado — só demora, porque com Ollama de pé
carrega modelo.

É essa propriedade que torna o CI possível: o
[workflow](.github/workflows/ci.yml) roda lint, formatação e a suíte sem
modelo a cada push no `main` e a cada pull request, num runner que não tem
Ollama nem 2 GB de pesos. O que o CI **não** faz é medir qualidade do modelo —
isso é trabalho dos `scripts/measure_*.py`, que levam minutos e cujos números
vão revisados para o plano, não para um check verde.

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

## Quanto demora

Medido nesta máquina (Intel Core Ultra 5 135U, CPU pura) com
`qwen2.5:3b-instruct-q4_K_M`, numa sessão de seis turnos alternando frase com
erro e frase correta — `uv run python scripts/measure_latency.py`:

| | |
|---|---|
| Primeira palavra | **2,7s** de mediana, e **não cresce** ao longo da sessão |
| Turno completo, frase correta | ~4s |
| Turno completo, frase com erro | 12–16s, porque inclui a análise |
| Análise, isolada | ~10s, **depois** da resposta — o aluno lê enquanto ela corre |
| Primeiro turno, com o modelo aquecido | 0,9s |

Dois detalhes de implementação estão por trás desses números, e os dois saíram
de medição, não de intuição (§5.5):

- **O aquecimento manda o mesmo prefixo que o turno manda.** Há uma lista só
  (`TurnRunner.stable_prefix`) justamente para as duas não divergirem — elas já
  divergiram, e custaram 11,6s no primeiro turno.
- **A dica curada vai dentro da fala do aluno, não como mensagem de sistema.**
  O template de chat junta as mensagens de sistema no topo, então uma dica
  "no fim da lista" era colada logo depois da persona e invalidava o cache de
  todo o histórico: a primeira palavra ia de 2,8s para 9,1s em seis turnos. Na
  fala do aluno, ela fica de fato no fim do prompt — e o modelo ainda passou a
  usá-la melhor.

## O nível do aluno

O perfil tem um nível CEFR (A2 a C2) e ele **faz diferença**: entra no prompt
como segunda mensagem de sistema, logo depois da persona, vindo de
`prompts/partials/level/`. Trocar o nível muda o registro da resposta — medido
no modelo real, a frase do professor vai de 6,5 para 10,0 palavras entre A2 e
C1, e o erro que ele escolhe corrigir muda (plano §5.3).

Segunda mensagem, nunca interpolado na persona: o Ollama cacheia o prefixo do
prompt, e um ajuste dentro da persona jogaria esse cache fora a cada turno.

O nível de partida é o que o aluno declara no cabeçalho da página. Dali em
diante ele se ajusta sozinho a partir de uma **rubrica determinística** sobre o
texto de cada turno — comprimento de frase, palavra longa, conectivos e
estruturas avançadas, menos uma penalidade por erro com regra curada. Nada disso
é perguntado ao modelo: a mesma frase dá sempre o mesmo número, e cada número é
conferível à mão.

```bash
uv run python scripts/measure_level.py              # matriz de confusão, sem modelo nenhum
uv run python scripts/measure_level.py --hysteresis # quantos turnos até o nível andar
uv run python scripts/measure_level.py --reply      # a mesma fala em cada nível, no modelo real
```

Duas regras moldam o ajuste, e as duas vêm de medição:

| | |
|---|---|
| **Um turno nunca move o nível** | Turno com menos de 12 palavras não é evidência de nada, e o nível só anda quando 4 turnos contados seguidos apontam para o mesmo lado — um degrau por vez. Sem isso o aluno veria o registro do professor oscilar a cada mensagem. |
| **C2 é autodeclarado** | A rubrica acerta 78% dos níveis e nunca erra por dois degraus, mas não distingue C1 de C2 em texto de chat. Então ela não estima C2, e não move quem está nele. |

O que o aluno declarou fica guardado separado do nível em vigor
(`self_declared_level` vs `cefr_level`), e toda mudança vira linha em
`level_change` com o motivo. É o que permite responder "por que o Charles mudou
o jeito de falar comigo".

O nome no cabeçalho troca de perfil: são vários alunos numa máquina, cada um com
nível e histórico próprios. Não é multiusuário com login — isso fica fora da
Fase 1.

## Os três modos

A barra acima da conversa troca o modo, e cada um muda **três** coisas — só a
primeira é texto:

| | conversation | correction | drill |
|---|---|---|---|
| Prompt de comportamento | `prompts/modes/conversation.md` | `…/correction.md` | `…/drill.md` |
| Correções estruturadas por turno | até 2 | até **5** | até 2 |
| Regra treinada | — | — | tirada do error log do aluno |

O teto por modo não é configuração: no modo de correção o aluno abriu o modo
para receber os erros, e devolver dois de cinco seria esconder dele o que ele
veio buscar.

**O drill treina um erro que o aluno cometeu.** A regra sai de uma consulta ao
`correction` agregado por `rule_id` — a mais frequente que ainda existe na base,
evitando as treinadas há pouco. Dali vêm a explicação revisada e as formas
certas; do banco vêm as frases que o próprio aluno escreveu. O modelo recebe
isso pronto; ele não escolhe a regra nem inventa o exemplo.

Uma regra por sessão, não por turno: drill que troca de regra a cada mensagem é
questionário, e bloco estável no prompt é prefixo cacheado. Trocar de modo abre
uma sessão nova — a transcrição de um drill não é continuação da conversa
livre — e "new conversation" mantém o modo.

Se o aluno ainda não errou nada com regra curada, o drill **diz isso** e oferece
conversar. Não existe exercício genérico de reserva.

```bash
uv run python scripts/measure_modes.py --turns 3
```

Medido com o modelo real: o pedido do exercício precisa ficar no **fim** do
prompt. Com ele só no bloco do topo, o 3B corrigia a fala do aluno e ia embora
conversar — nenhum exercício em três turnos (plano §5.4).

## Falar com o Charles

A voz é opcional e vem desligada: o chat por texto funciona inteiro sem ela, e
ligá-la custa ~400 MB de wheels (`ctranslate2`, `onnxruntime`) mais ~120 MB de
pesos. Para ligar:

```bash
uv sync --extra dev --extra voice
```

```bash
uv run python scripts/fetch_voice_models.py
```

```bash
TC_VOICE_ENABLED=true uv run teacher-charles
```

O Whisper se baixa sozinho na primeira transcrição; só o Kokoro precisa do
script, porque vem de um release do GitHub e não de um hub com cache próprio.

Na página aparecem duas coisas: um botão **Hold to talk** e um interruptor
**Charles speaks**. Segure o botão, fale, solte — e a transcrição cai **no
campo de texto**, não na conversa. Se o clique for curto, o microfone fica
travado ligado até o clique seguinte.

### Por que a transcrição não vira turno direto

Porque o STT erra. Um erro de reconhecimento que vira turno faz o Charles
corrigir um erro que o aluno **não cometeu** — o pior defeito possível neste
produto, e exatamente a classe de falha que a base curada existe para evitar do
lado do modelo. Com a transcrição no campo, o erro vira um ajuste de uma
palavra antes de mandar.

Custa um pressionar de tecla. Compra a única salvaguarda que não depende de
qual modelo de reconhecimento está carregado.

### Quanto custa, medido

`uv run python scripts/measure_voice.py` — mesma máquina das outras medições:

| | |
|---|---|
| Transcrever uma fala de 2,6s (`base.en`, int8, 8 threads) | **0,91s** |
| Sintetizar uma sentença (Kokoro int8) | **RTF 0,54** — meio segundo de máquina por segundo de fala |
| Até o aluno **ouvir** a primeira palavra | **5,4–7,4s** |
| Depois disso | Fala contínua: a síntese da sentença seguinte cabe dentro da reprodução da atual |

Os 5,4–7,4s são contra uma meta de 1–2s, e isso está registrado como dívida,
não escondido. **A maior fatia é o modelo de linguagem** — 3,0–4,2s até a
primeira sentença fechar —, não o áudio: o reconhecimento e a síntese somados
custam menos que ele sozinho.

**A alavanca era a iGPU Intel, e ela foi puxada.** Com o backend Vulkan ligado
o turno de voz cai para **4,0s** e a relação se inverte: o LLM entrega a
primeira sentença em 1,48s e o áudio soma 2,51s, então agora é o áudio a maior
fatia. A meta de 1–2s segue não atingida. Números e método em
[docs/plano-fase-1-5.md](docs/plano-fase-1-5.md) §7.

Dois defaults saíram dessa medição e contrariam o que o plano previa:

- **`base.en`, não `small.en`.** O `small.en` leva 2,4s numa fala curta e fura
  sozinho o orçamento; o `base.en` leva 0,91s.
- **`TC_STT_CPU_THREADS=8`**, porque o default do `ctranslate2` deixa 30% na
  mesa — e `TC_TTS_CPU_THREADS=0`, porque no `onnxruntime` limitar **piora**.

### O que não foi medido

O áudio das tabelas é **sintético**, gerado pelo próprio Kokoro: fala limpa,
sem sotaque, sem ruído. Isso mede latência honestamente, mas mede exatidão no
caso mais fácil que existe. **A taxa de erro com um brasileiro falando num
microfone de notebook é maior, e nenhum número deste repositório a conhece.**

Medir isso precisa da sua voz:

```bash
uv run python scripts/measure_voice.py --record 6 --say "what you just said"
```

## A tela do que você erra

`/progress` responde a pergunta que a E4 passou a tornar possível: **no que
este aluno mais erra?** Cada correção com regra curada virou linha em SQLite, e
a tela agrega por regra, da mais frequente para a menos:

- quantas vezes, e quando foi a última;
- **as frases do próprio aluno** que caíram naquela regra — erro agregado em
  abstrato não ensina ninguém;
- a explicação e a forma certa, vindas da base curada (não do que o modelo
  disse naquele turno);
- o histórico de mudança de nível, com o motivo de cada uma.

Correções sem regra curada aparecem só como contagem no rodapé. Elas não são
agregáveis — o modelo escreve a explicação que quiser, diferente a cada vez — e
pela política da §5.2 nem chegam ao aluno. O número existe para medir quanto a
base ainda não cobre.

É a mesma consulta que alimenta o drill (E6): o que o aluno vê na tela é o que
o professor vai treinar com ele.

## Medições

Nenhum número deste README é estimativa. Cada um sai de um script que você pode
rodar:

| | |
|---|---|
| `scripts/measure_analysis.py` | Precisão da correção estruturada: recall, falso positivo, JSON degradado (§5.2) |
| `scripts/measure_level.py` | Matriz de confusão da rubrica de nível, e o custo da histerese (§5.3) |
| `scripts/measure_modes.py` | Os três modos contra o modelo real, com o drill andando (§5.4) |
| `scripts/measure_latency.py` | Latência por turno, decomposta com as métricas do servidor (§5.5) |
| `scripts/measure_voice.py` | STT, TTS e o orçamento do turno de voz (plano 1.5 §2 e §3.1) |
| `scripts/measure_igpu.py` | Vazão na iGPU Intel contra a CPU pura, e a prova de que a GPU está no caminho (plano 1.5 §7) |

Os três primeiros precisam de Ollama com o modelo baixado; o
`measure_level.py` sem argumento roda sem modelo nenhum. O `measure_voice.py`
precisa dos pesos de voz, e só o `--budget` precisa também do Ollama. O
`measure_igpu.py` precisa de **dois** servidores de pé, um com o backend
Vulkan — ele diz o comando quando não encontra o segundo.

## Quando dá errado

| Sintoma | O que é |
|---|---|
| `/health` responde 503 | Deliberado: o serviço subiu mas não está pronto para ensinar. O corpo diz qual das três coisas falhou — banco, modelo ou base contrastiva. |
| "o Ollama não conhece o modelo" | Falta `ollama pull <modelo>`. O `/health` lista o que o servidor tem. |
| "não deu para falar com o Ollama" | O serviço não está de pé. `ollama serve`, ou `python3 scripts/check_env.py` para confirmar. |
| Primeira palavra demora ~3s, as seguintes também | É o esperado nesta classe de hardware com um 3B (§5.5). Se o primeiro turno demorar muito mais, o aquecimento não rodou: veja `TC_LLM_WARMUP`. |
| Turno de 15s em frase com erro | A análise custa ~8-10s e roda **depois** da resposta. O aluno lê enquanto ela corre; a anotação aparece quando fica pronta. |
| A correção não aparece | Só vai ao aluno correção com regra curada (§5.2). Frase sem regra na base não vira correção estruturada — vira, no máximo, correção na prosa. |
| A linha de IPA não aparece embaixo das falas | Veja o interruptor **Phonetics** no cabeçalho; se ele nem existe, o `/health` diz por quê no bloco `phonetics` (espeak que não carregou, ou `TC_PHONETICS_ENABLED=false`). |
| O microfone não aparece na página | A voz vem desligada. `uv sync --extra voice`, baixe os pesos e suba com `TC_VOICE_ENABLED=true`. |
| A voz está ligada mas o Charles não fala | Veja `/health`: o bloco `voice` diz qual motor não carregou e por quê. Peso ausente é o caso comum. |
| O Charles escreve mas o navegador não toca som | O navegador bloqueia áudio sem gesto do usuário. Clique em qualquer lugar da página e mande outro turno. |
| `TypeError: open() got an unexpected keyword argument 'metadata_errors'` | PyAV 19 no ambiente. O `pyproject.toml` fixa `av<19`; rode `uv sync --extra voice` de novo. |
| Testes falham com modelo inesperado | Um `.env` na raiz vaza para a suíte. `tests/conftest.py` limpa `TC_*`, mas scripts de medição leem o `.env`. |

## Onde fica cada coisa

```
data/        base de erros típicos de brasileiros — VERSIONADA, é conteúdo do produto
  contrastive/ os itens curados em YAML; schema/ o JSON Schema gerado deles
prompts/     persona e instruções em .md, versionados, recarregados sem reiniciar
src/         o código
  llm/         providers: Ollama (local), OpenAI-compat (nuvem), fake (testes)
  voice/       as duas pontas da Fase 1.5: Whisper (ouvir), Kokoro (falar), fake
    segment.py   fluxo de pedacos -> sentencas falaveis, e a limpeza de markdown
  teaching/    a estrutura pedagógica — é esta pasta que é o produto
    turn.py      orquestra o turno: resposta, análise e sinais de nível
    analyzer.py  fala do aluno -> correção estruturada e verificada
    contrastive.py  carrega e consulta as regras curadas
    level.py     fala do aluno -> nível CEFR estimado, com histerese
    modes.py     os três modos; o drill escolhe a regra no error log do aluno
    phonetics.py texto -> IPA do espeak: a linha de pronúncia de cada fala
  store/       SQLite: esquema em .sql, migrações numeradas
  web/         página única (conversa) e a tela /progress; JS e CSS sem build step
docs/        análise, plano e decisões de arquitetura
scripts/     check_env.py (só stdlib, roda antes do uv sync) e as medições
```

Editar um arquivo de `prompts/` vale no turno seguinte, sem reiniciar o servidor.

Duas coisas **nunca** entram no repo:

- **Pesos de modelo.** Ficam onde o Ollama os guarda (`~/.ollama`); os de voz,
  em `$XDG_CACHE_HOME/teacher-charles/voice` e no cache do `huggingface_hub`.
  São cache, não dado pessoal — e por isso não ficam junto do banco do aluno.
- **Progresso do aluno.** Fica em `$XDG_DATA_HOME/teacher-charles/charles.db`.
  É dado pessoal. `data/` guarda conhecimento curado, nunca dados de aluno.

## Licença

Ver [LICENSE](LICENSE).
