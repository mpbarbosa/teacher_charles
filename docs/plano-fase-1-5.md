# Teacher Charles — Fase 1.5: voz

> **Estado: completa.** O aluno fala, o Charles ouve, responde e fala de volta,
> tudo local. A meta de latência que a Fase 1 projetou — 1–2 s até a primeira
> palavra — **não foi atingida**: são 5,4–7,4 s nesta máquina. O que foi
> atingido é a outra metade do requisito, e ela não era menos difícil: depois
> da primeira palavra, a fala sai contínua.

A Fase 1 moveu a voz para cá com um argumento (§4 do plano da Fase 1) e uma
condição (§1.3): *voz é latência e integração, não ensino*, e por isso ela
**começa por uma medição, não por integração**. Este documento é essa medição,
e o que ela mudou no desenho.

---

## 1. O que entrou

| | |
|---|---|
| **STT** | `faster-whisper` em int8 na CPU, com o VAD do Silero como filtro de entrada |
| **TTS** | Kokoro-82M exportado para ONNX, int8, na CPU |
| **Segmentação** | Fala por sentença: a síntese da sentença *n+1* corre dentro da reprodução da *n* |
| **Turn-taking** | *Push-to-talk* — segura, fala, solta |
| **Rotas** | `POST /api/stt`, `POST /api/tts`, `GET /api/voice`, quadros `sentence` no `/api/chat` |
| **Interface** | Botão de microfone e interruptor "Charles speaks", sem build step |

Desligada por default (`TC_VOICE_ENABLED=false`) e atrás de um extra
(`uv sync --extra voice`): o chat por texto da Fase 1 funciona inteiro sem
nada disto, e ninguém deveria baixar 400 MB de wheels para rodar a suíte.

### O que ficou fora, e por quê

**O VAD como turn-taking.** O plano §4 nomeava o Silero VAD para o servidor
decidir que o aluno terminou de falar. O que entrou foi o mesmo Silero, mas
como *filtro* dentro do `faster-whisper`: ele corta o silêncio gravado antes e
depois da fala — tempo de transcrição pago por nada — e tira o combustível da
alucinação em silêncio, que no Whisper sai como `"Thank you."` ou
`"Thanks for watching!"`.

Quem decide que o aluno terminou é o dedo dele no botão. É menos natural, e é
a escolha certa para a primeira versão: zero falso disparo, zero *barge-in*
para depurar, e o caminho de áudio fica medido antes de ganhar uma máquina de
estados. A dependência que o turn-taking automático precisa já está no
processo.

---

## 2. A medição que abre a fase

Tudo abaixo roda com `uv run python scripts/measure_voice.py`. Máquina: a da
§1.3 do plano da Fase 1 — Intel Core Ultra 5 135U, 14 threads, sem CUDA.

### 2.1 TTS: o Kokoro acompanha a reprodução com folga

O número que decide a Fase 1.5 não é o tempo de síntese, é o **fator de tempo
real** (RTF): segundos de máquina por segundo de áudio gerado. Acima de 1,0 a
síntese não acompanha a reprodução e a voz engasga entre sentenças, por mais
rápido que o modelo de linguagem seja.

Resposta de professor de verdade, quatro sentenças, 14,8 s de áudio:

| Modelo | Threads | Carga + aquecimento | 1ª sentença | RTF mediano | RTF pior |
|---|---|---|---|---|---|
| **int8** (92 MB) | **auto** | 2,85 s | 1,45 s | **0,54** | 0,65 |
| int8 | 4 | 2,07 s | 1,83 s | 0,72 | 0,81 |
| fp32 (325 MB) | auto | 2,24 s | 1,47 s | 0,62 | 0,66 |
| fp32 | 4 | 2,11 s | 1,97 s | 0,78 | 0,88 |

Duas coisas saíram daqui e viraram default:

**O int8 ganha nas duas pontas.** Mais rápido (0,54 contra 0,62) *e* 3,5×
menor. Não houve troca a fazer, o que é raro o bastante para merecer registro:
a quantização do Kokoro não custou velocidade nem tamanho.

**Limitar threads no TTS piora.** 0,54 com o default do onnxruntime contra
0,72 com 4 threads. Daí `TC_TTS_CPU_THREADS=0`, que significa "não mexa". É o
oposto do que aconteceu no STT — ver abaixo —, e por isso os dois defaults
estão em lados opostos da mesma configuração.

### 2.2 STT: o `small.en` do plano não cabe no orçamento

Falas de aluno de três tamanhos, int8, feixe 1. **O STT roda com o aluno
parado esperando**: ele largou o botão e não há streaming para esconder nada.
É latência pura, ao contrário do texto do professor.

Com o default de threads do ctranslate2:

| Modelo | fala de 2,6 s | de 4,8 s | de 8,6 s | WER |
|---|---|---|---|---|
| tiny.en | 0,77 s | 0,78 s | 0,79 s | 0% |
| base.en | 1,19 s | 1,18 s | 1,17 s | 0% |
| small.en | 3,37 s | 3,51 s | 3,41 s | 0% |

Com 8 threads:

| Modelo | fala de 2,6 s | de 4,8 s | de 8,6 s |
|---|---|---|---|
| tiny.en | 0,56 s | 0,63 s | 0,67 s |
| **base.en** | **0,91 s** | 0,88 s | 1,07 s |
| small.en | 2,40 s | 2,72 s | — |

E com 14 threads o `small.en` **piora** para 2,52 s: metade são hyperthreads
que disputam a mesma unidade de execução.

**O default saiu de `small.en` para `base.en`.** 2,4 s do `small.en` furam
sozinhos a meta de 1–2 s do turno inteiro; 0,91 s do `base.en` cabem. O plano
da Fase 1 nomeava o `small.en` sem tê-lo medido — era estimativa, como os
20–25 tok/s que a E1 corrigiu para 9,3.

**E `TC_STT_CPU_THREADS=8`, não o default.** O ctranslate2 deixa 30% na mesa
sozinho.

### 2.3 O que esta medição **não** diz

O áudio das tabelas acima é **sintético**: gerado pelo próprio Kokoro. Isso
mede latência honestamente — o Whisper leva o mesmo tempo em áudio difícil —
mas mede exatidão no caso mais fácil que existe: fala limpa, sem sotaque, sem
ruído, sem microfone de notebook no meio.

**Os 0% de WER são um piso, não uma expectativa.** A taxa de erro que um aluno
brasileiro vai ver é maior, e nenhum número deste repositório a conhece. É
exatamente a diferença entre `tiny.en`, `base.en` e `small.en` que o áudio
sintético não consegue mostrar — os três acertam o caso fácil.

Medir isso precisa de voz humana:

```bash
uv run python scripts/measure_voice.py --record 6 --say "what you said" \
    --stt-models tiny.en,base.en,small.en
```

Fica registrado como a dívida principal desta fase. Ela tem um anteparo —
§3.2 — mas anteparo não é medição.

---

## 3. O que a medição mudou no desenho

### 3.1 O orçamento do turno, por inteiro

Quatro execuções de `measure_voice.py --budget`, que passa pelo `TurnRunner`
de verdade — persona, nível, modo, dica curada, prefixo aquecido — e não por
uma chamada direta ao provider. Modelo `qwen2.5:3b-instruct-q4_K_M`.

| | pior | melhor |
|---|---|---|
| STT (4,8 s de áudio) | 1,56 s | 1,24 s |
| LLM até o 1º token | 2,54 s | 2,03 s |
| LLM até **fechar a 1ª sentença** | 4,21 s | 3,03 s |
| TTS da 1ª sentença | 2,28 s | 0,86 s |
| **até o aluno ouvir a 1ª palavra** | **7,35 s** | **5,44 s** |

**A meta de 1–2 s não foi atingida, e não é alcançável com este arranjo.** A
maior fatia é o LLM, e não o áudio: 3,0–4,2 s dos 5,4–7,4 s. O STT e o TTS
somados custam menos que o modelo de linguagem sozinho.

Isso não é surpresa — é a §1.3 do plano da Fase 1 acontecendo. Ela dizia, antes
de qualquer código de voz: *"a 7 tok/s, 15 tokens levam ~2s, antes do TTS"*, e
registrava a iGPU ociosa como **a** alavanca da Fase 1.5. A medição confirma o
diagnóstico e aponta o mesmo lugar: enquanto o Ollama oficial rodar 100% em
CPU nesta máquina, o resto é detalhe.

> **A §7 puxou essa alavanca, e corrige este parágrafo em dois pontos.** O
> ganho é real (6,3 s → 4,0 s no turno), mas **não vem de vazão de geração** —
> vem de avaliação de prompt, que a iGPU faz 13× mais rápido. E depois dele a
> conclusão abaixo se inverte: o LLM entrega a 1ª sentença em 1,48 s e o áudio
> soma 2,51 s, então **passa a ser o áudio a maior fatia**. Ver §7.4 e §7.5.

O que *não* vale a pena mexer, e a medição diz por quê:

- **Cortar a primeira sentença na vírgula** para começar a falar antes. Ganha
  ~1 s dos 7, e cobra um artefato de prosódia em toda abertura de turno: o
  Kokoro põe entonação de fim de frase no fragmento. Um sétimo da espera não
  paga uma voz que soa errada a cada turno.
- **Trocar o `base.en` pelo `tiny.en`.** Economiza 0,35 s e gasta exatidão —
  justamente a dimensão que a §2.3 não conseguiu medir. Errar aqui é pior que
  esperar.

### 3.2 A transcrição vai para o campo de texto, não para o turno

Esta é a decisão de produto da fase, e ela vem do que a §2.3 **não** pôde
medir.

O STT erra. Um erro de STT que vira turno direto faz o Charles corrigir um erro
que o aluno não cometeu — o pior defeito possível neste produto, e exatamente a
classe de falha que a E3 e a E4 existem para evitar do lado do modelo. Seria
incoerente gastar 43 itens curados e uma política de "sem regra, sem correção"
para depois deixar o pipeline de áudio injetar erros fantasma na entrada.

Então a transcrição cai no campo de texto, o aluno lê, conserta uma palavra se
precisar, e manda. Custa um pressionar de tecla. Compra a única salvaguarda que
existe contra um pipeline de áudio que não tem como estar sempre certo — e
compra sem depender de qual modelo de STT está carregado.

Efeito colateral bom: isso torna a escolha `tiny`/`base`/`small` menos crítica
do que ela seria, o que é o jeito certo de conviver com uma dívida de medição.

### 3.3 O áudio não vem pelo SSE

Os quadros `sentence` do `/api/chat` carregam índice e texto. O áudio a página
pede ao `/api/tts` num pedido próprio, por sentença.

Duas razões, e a segunda é a que importa:

1. Áudio em SSE teria que ir em base64 — 33% a mais de bytes num quadro que já
   atravessa o caminho crítico do texto.
2. A síntese passaria a correr **dentro** do mesmo gerador que repassa os
   pedaços do professor, atrasando a letra para adiantar o som. Em pedido
   separado, a síntese da sentença 1 corre enquanto o modelo escreve a 2 — que
   é a única razão de o TTS incremental existir.

### 3.4 O fim do pedaço fecha sentença

O montador (`voice/segment.py`) aceita fronteira de sentença com espaço depois
**ou** no fim do que já chegou. Esperar o espaço seria mais seguro — `It takes
3.` pode virar `3.5 hours` — mas custaria um token de atraso em *toda*
sentença, e a 7–8 tok/s o token vale ~130 ms, pagos no único ponto do turno em
que o aluno espera em silêncio.

O risco se paga sem o atraso porque tem forma conhecida: o caso ruim é um
dígito antes do ponto, e só esse espera. Abreviação (`Mr.`) e inicial (`J. R.`)
são decididas pelo que está **antes** do ponto, que sempre já chegou.

---

## 4. Onde a voz encostou no código da Fase 1

Pouco, e foi o teste do desenho da E2. O `teaching/turn.py` nasceu agnóstico
quanto à origem do texto, e a aposta era que a voz entraria como **duas
pontas**, não como reescrita do meio. A aposta se pagou:

| Módulo | O que mudou |
|---|---|
| `teaching/` | **Nada.** Nem um arquivo. |
| `llm/` | **Nada.** |
| `store/` | **Nada** — nenhuma migração nova. |
| `voice/` | Novo: contratos, dois motores, dois dublês, o montador de sentenças |
| `app.py` | Três rotas, os quadros `sentence`, voz no `lifespan` e no `/health` |
| `config.py` | As variáveis `TC_STT_*` / `TC_TTS_*` e `resolved_voice_dir` |
| `web/` | `voice.js` novo; `app.js` ganhou um ramo de quatro linhas |

O `voice/` não importa `teaching/`, e `config` não importa `voice` — a mesma
regra de sentido único que vale para `llm`.

---

## 5. Riscos conhecidos desta fase

| Risco | Como lidamos |
|---|---|
| **Exatidão do STT com sotaque brasileiro é desconhecida** | A transcrição vai para o campo de texto, onde o aluno corrige (§3.2). O anteparo é bom; a medição continua faltando, e `--record` existe para quem quiser fazê-la. |
| **5,4–7,4 s até a primeira palavra** | Registrado, não escondido. A alavanca é o LLM, e é a mesma que a §1.3 da Fase 1 já apontava: a iGPU Intel ociosa, via llama.cpp com Vulkan/SYCL ou IPEX-LLM. O `openai_compat` da E1 já serve um llama.cpp local, então trocar é configuração. |
| **`av<19` fixado** | O `decode_audio` do `faster-whisper` 1.2 chama `av.open(metadata_errors=...)`, que o PyAV 19 removeu; com ele instalado, **toda** transcrição estoura antes de o modelo ver um sample. O teto sai quando o `faster-whisper` falar com a API nova. |
| **O Whisper alucina em silêncio** | VAD do Silero ligado, `no_speech_threshold`, e um piso de bytes que devolve texto vazio sem nem carregar o modelo. |
| **Dois runtimes nativos no mesmo processo** | Cada motor serializa com lock próprio e roda em thread: nada de código nativo síncrono no event loop, e nunca duas inferências disputando a mesma CPU. |

---

## 6. O que vem depois

- ~~**A iGPU.**~~ **Medida — ver §7.** Saiu da lista: o Ollama 0.35 já traz o
  backend Vulkan, e ligá-lo leva o turno de voz de 6,3 s para 4,0 s e a
  primeira palavra do chat de 7,4 s para 0,9 s, sem uma linha de código. Deixa
  de ser trabalho futuro e vira configuração recomendada.
- **Turn-taking sem botão**, com o Silero que já está no processo.
- **Avaliação de pronúncia por fonemas** (Fase 2, como na análise inicial): é o
  que transforma "o aluno falou" em "o aluno falou **bem**", e é a razão de a
  voz existir num professor de inglês.
- **Medir o STT com voz humana brasileira** — a dívida da §2.3.

---

## 7. A iGPU, aberta e medida

A §6 listava a iGPU como "a alavanca única que move os 7 s", e a §1.3 do plano
da Fase 1 a registrava desde o começo como promessa não verificada. Esta seção
verifica. **A alavanca existe, é grande, e não custa uma linha de código** — mas
ela move um lugar diferente do que a §3.1 supunha.

### 7.1 O que mudou desde a §1.3

A §1.3 dizia: *"não há build com suporte a Arc/SYCL no pacote oficial"*. Isso
valia para a versão medida; **não vale mais**. O Ollama 0.35 instalado nesta
máquina já traz `lib/ollama/vulkan/libggml-vulkan.so`, e o binário expõe duas
variáveis que ligam o backend:

```bash
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_VULKAN=1 OLLAMA_IGPU_ENABLE=1 ollama serve
```

Não foi preciso compilar `llama.cpp`, nem instalar IPEX-LLM, nem oneAPI. As
dependências já estavam na máquina: Mesa 26.0.8 com o ICD `intel_icd.json`
(ANV) e `/dev/dri/renderD128` acessível por ACL de sessão. O log do servidor
confirma a enumeração:

```
msg="inference compute" id=0 library=Vulkan name=Vulkan0
  description="Intel(R) Graphics (MTL)" type=iGPU total="22.5 GiB" available="9.3 GiB"
```

Dos três caminhos que a §1.3 imaginava — Vulkan, SYCL, IPEX-LLM — o primeiro
chegou pronto no pacote. **Os outros dois não foram testados**, e isto é
limitação desta medição, não conclusão sobre eles: é possível que um kernel de
SYCL extraia da Arc mais do que o Vulkan extrai na geração, que é onde o ganho
é pequeno (§7.3). O que pesa contra ir atrás disso é o preço, não a física —
qualquer um dos dois é um segundo runtime para instalar e manter, contra duas
variáveis de ambiente no binário que já está na máquina.

### 7.2 Como se prova que a GPU está trabalhando

"Compilou com Vulkan" não é "está usando a GPU". O `intel_gpu_top` desta
máquina pede `CAP_PERFMON` e não roda sem root, então a prova vem de três
lugares que rodam como usuário comum:

| | CPU | iGPU Vulkan |
|---|---|---|
| `size_vram` do `/api/ps` | 0,00 GB | **2,16 GB** |
| `drm-engine-render` do `fdinfo` do i915 | 0% ocupado | **66% ocupado** |
| `drm-resident-system0` | 0,00 GB | **2,42 GB** |
| `/sys/class/drm/card1/gt_act_freq_mhz` | 800 MHz (parado) | **1900 MHz** |

O `size_vram=0.00GB` era exatamente o sintoma que a §1.3 usou para concluir
"roda 100% em CPU". Agora ele reporta o modelo inteiro.

### 7.3 Vazão bruta: prompt ganha 13×, geração quase não se move

```bash
uv run python scripts/measure_igpu.py --reps 3 --cpu-url http://127.0.0.1:11436
```

Os dois servidores de pé ao mesmo tempo, mesma máquina, mesmo estado de carga,
pelo `eval_count`/`eval_duration` do próprio servidor — que é como a §1.3 mediu
os 9,3 tok/s.

**3B** — `qwen2.5:3b-instruct-q4_K_M`, 3 repetições, mediana:

| | CPU | iGPU Vulkan | |
|---|---|---|---|
| **geração** | 6,2 tok/s | **5,6 tok/s** | 0,91× — *empate* |
| prompt, 117 tok | 17,7 tok/s | 142,5 tok/s | 8,0× |
| prompt, 1085 tok | 12,8 tok/s | 168,1 tok/s | 13,1× |
| prompt, 2097 tok | 12,1 tok/s | 169,5 tok/s | 14,0× |

**7B** — `--model qwen2.5:7b-instruct-q4_K_M --reps 1`, ponto de confirmação e
não amostra (o lado da CPU leva ~10 min por repetição):

| | CPU | iGPU Vulkan | |
|---|---|---|---|
| **geração** | 2,8 tok/s | **4,2 tok/s** | **1,50× — ganho** |
| prompt, 117 tok | 6,9 tok/s | 89,2 tok/s | 12,9× |
| prompt, 1085 tok | 5,6 tok/s | 100,6 tok/s | 17,9× |
| prompt, 2097 tok | 5,5 tok/s | 94,6 tok/s | 17,2× |

Avaliar prompt é multiplicação de matrizes em lote, limitada por compute, e aí
a Arc ganha uma ordem de grandeza nos dois modelos. **Geração é a parte
interessante, e ela não se comporta igual nos dois.**

No 3B empata, que é o que a §1.3 fazia esperar: ela já dizia que nesta classe
de hardware *"o gargalo é banda de memória, não VRAM"*, e gerar um token lê os
pesos inteiros. A Arc integrada usa **a mesma DRAM** da CPU, então trocar quem
lê não aumenta a banda.

No 7B **não empata: ganha 50%.** Isso diz que a 3B a CPU ainda não estava
estrangulada — ela e a iGPU batem no mesmo teto de banda. A 7B, com o dobro de
pesos por token, os 14 threads de um chip U-series ficam para trás
(descompressão de Q4 e GEMV são trabalho de ALU, não só de leitura) enquanto a
iGPU, com mais unidades em paralelo, se mantém mais perto do teto de banda.

A consequência prática é modesta, e vale dizê-la com o número na mão: **4,2
tok/s continua abaixo dos ~7–8 tok/s que a §1.3 chamou de piso do tolerável.**
A iGPU encolhe a punição por modelo grande; não a remove. Um 7B passa de
inviável (2,8) a ruim (4,2), não a utilizável.

### 7.4 A surpresa: a primeira palavra é limitada por *prompt*, não por geração

A §3.1 lia o orçamento do turno como "a maior fatia é o LLM", e daí concluía,
implicitamente, que o que faltava era vazão de geração. **Estava errado**, e a
medição do turno mostra por quê:

```bash
TC_OLLAMA_URL=http://127.0.0.1:11435 uv run python scripts/measure_latency.py
```

| | CPU (hoje) | iGPU (hoje) | §5.5 (CPU, registrado) |
|---|---|---|---|
| **primeira palavra**, mediana | 7,4 s | **0,9 s** | 2,7 s |
| turno completo, mediana | 20,5 s | 9,7 s | 4,1 s |
| análise isolada, mediana | 20,3 s | 15,6 s | 10,1 s |
| aquecimento da subida | 60,7 s | **8,3 s** | — |

A conta que explica os 8×: mesmo com **88–95% do prompt em cache**, os 5–12%
que sobram de um prompt de ~900 tokens são 50–100 tokens para avaliar. A 12–17
tok/s isso custa 4–8 s; a 170 tok/s custa 0,6–0,9 s. **A espera pela primeira
palavra quase não tem geração dentro dela** — tem o pedaço do prompt que o
cache não cobriu. É o lugar exato onde a iGPU ganha 13×, e é por isso que o
ganho do turno é maior que o ganho de qualquer número de vazão isolado.

A mesma conta explica o que **não** melhora: a análise cai só 1,3× (20,3 s →
15,6 s), porque ela *gera* 63–83 tokens de JSON e o prompt dela tem 314–325
tokens. Análise é geração; geração empata. A dívida da §5.5 — "a análise custa
~10 s por turno com erro" — **continua aberta, e a iGPU não a fecha.**

> **Ressalva de honestidade.** A CPU de hoje está pior que a da §5.5: 7,4 s de
> primeira palavra contra 2,7 s, e 6,2 tok/s de geração contra os 9,3 da §1.3.
> A máquina está com carga de desktop (load 2,3, navegador e editor abertos),
> e as medições antigas foram feitas com ela mais quieta. Por isso **os dois
> lados foram medidos hoje, juntos**: a razão é o número sólido, não o
> absoluto. Vale notar que os 0,9 s da iGPU foram medidos *nessa mesma máquina
> carregada* — ou seja, abaixo dos 2,7 s que a §5.5 registrou com a máquina
> quieta.

### 7.5 O turno de voz: 6,3 s → 4,0 s, e o gargalo troca de lado

```bash
TC_OLLAMA_URL=http://127.0.0.1:11435 uv run python scripts/measure_voice.py --budget
```

| | CPU (hoje) | iGPU (hoje) | §3.1 (CPU, registrado) |
|---|---|---|---|
| STT (4,8 s de áudio) | 1,18 s | 1,22 s | 1,24–1,56 s |
| LLM até o 1º token | 2,21 s | **0,51 s** | 2,03–2,54 s |
| LLM até fechar a 1ª sentença | 3,63 s | **1,48 s** | 3,03–4,21 s |
| TTS da 1ª sentença | 1,48 s | 1,29 s | 0,86–2,28 s |
| **até ouvir a 1ª palavra** | **6,29 s** | **3,99 s** | 5,44–7,35 s |

**A frase de encerramento da §3.1 se inverte.** Ela dizia: *"O áudio coube; o
modelo de linguagem é que não."* Com a iGPU, o LLM entrega a primeira sentença
em 1,48 s e o áudio soma 2,51 s (STT 1,22 + TTS 1,29). **Agora é o áudio que
não cabe.**

A meta de 1–2 s continua não atingida, e agora por um motivo diferente: mesmo
com o LLM em zero, STT + TTS já custariam 2,5 s. O que ainda estava aberto no
LLM foi colhido; o que sobra está nas duas pontas de áudio, e a §3.1 já
registrou que os atalhos óbvios ali (cortar a sentença na vírgula, trocar o
`base.en` pelo `tiny.en`) custam prosódia e exatidão por pouco mais de um
segundo. Isso não mudou.

### 7.6 Ligar no produto não precisou de código nenhum

O plano supunha que a troca passaria por `TC_PROVIDER=openai_compat` apontando
para um `llama-server` — o caminho que a E1 deixou pronto. **Não foi preciso.**
O backend da iGPU é o do próprio Ollama, então o `OllamaProvider` serve sem
mudar uma linha, e basta a URL:

```bash
TC_OLLAMA_URL=http://127.0.0.1:11435 uv run teacher-charles
```

Conferido: um turno completo no navegador, com a correção citando
`calque.age-have` da base contrastiva como sempre, e `uv run pytest -m "not
integration" -q` verde (591 passed, 8 skipped) sem tocar em nada. O
`openai_compat` continua existindo para o que ele sempre foi — nuvem opcional e
llama.cpp, se o Ollama sair do caminho.

### 7.7 O que fazer com isso

**A alavanca vale, e o preço é um `ollama serve` com duas variáveis de
ambiente.** Não é um segundo runtime para manter: é o mesmo binário, o mesmo
modelo, o mesmo provider — o que elimina o principal argumento contra puxá-la.
O resultado negativo que esta investigação admitia como possível não apareceu.
O que apareceu foi um ganho que mora num lugar diferente do esperado:

- **O que a iGPU compra é latência de prompt**, e o produto é feito de prompt:
  persona + nível + modo + histórico são ~900 tokens por turno. Daí os 8× na
  primeira palavra, maiores que qualquer número de vazão isolado.
- **O default de 4B-class da §1.3 continua certo.** Não pelo motivo que a §7.3
  parecia dar no 3B — o 7B *ganha* 50% em geração —, mas porque o ganho não
  chega ao suficiente: 4,2 tok/s num 7B segue abaixo do piso de ~7–8 tok/s.
  Vale medir um 4B antes de assumir o número extrapolado; não foi feito aqui.
- **A dívida da análise (~10 s) não é tocada** (1,3×: ela *gera* 63–83 tokens
  de JSON) e passa a ser a maior fatia isolada do turno completo. O caminho
  continua sendo o que a §5.5 apontou — um modelo melhor só para ela —, agora
  com a nota de que a iGPU quase não ajuda ali.
- **O turno de voz melhora 36% e a meta de 1–2 s continua fora**, agora
  barrada pelo áudio (§7.5) e não mais pelo LLM.

O que isto deixa para a §6: a iGPU **sai** da lista de "o que vem depois" e
entra como configuração recomendada; o turn-taking, a pronúncia e a medição de
STT com voz brasileira continuam onde estavam.
