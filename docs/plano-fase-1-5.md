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

- **A iGPU.** É a alavanca única que move os 7 s, e ela está ociosa desde a
  Fase 1.
- **Turn-taking sem botão**, com o Silero que já está no processo.
- **Avaliação de pronúncia por fonemas** (Fase 2, como na análise inicial): é o
  que transforma "o aluno falou" em "o aluno falou **bem**", e é a razão de a
  voz existir num professor de inglês.
- **Medir o STT com voz humana brasileira** — a dívida da §2.3.
