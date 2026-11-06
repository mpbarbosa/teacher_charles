# Teacher Charles — Fase 2: pronúncia e repetição espaçada

> **Estado: aberta.** A medição que decide a pronúncia está feita (§2) e já
> moveu o desenho em cinco lugares (§3). O resultado é bom em dois eixos e
> inconclusivo no terceiro — e o terceiro é justamente a dívida que a Fase 1.5
> deixou aberta (§2.3 do plano 1.5). Por isso a primeira entrega de código
> **não** é o avaliador: é pagar essa dívida, porque é ela que decide se o
> avaliador existe.

A análise inicial §3 chamou a avaliação de pronúncia de *"a parte com mais
engenharia"*, e a Fase 1 a empurrou para cá duas vezes — uma no escopo (§4,
"Fora, explicitamente") e outra no conteúdo, quando a E3 adiou
`data/contrastive/pronunciation.yaml` porque *"não há o que casar em texto: th
e vogal epentética são erros de fala"* (§5.1). Agora há áudio. O arquivo entra
nesta fase, e com ele a razão de a voz existir num professor de inglês:
transformar *"o aluno falou"* em *"o aluno falou **bem**"*.

A fase abre como a 1.5 abriu — **por uma medição, não por integração**. Deu
certo lá (quatro mudanças de desenho antes de haver código para jogar fora) e
aqui era ainda mais necessário, porque a pergunta que decide a pronúncia não
está em documentação nenhuma:

> **O piso de erro do reconhecedor é menor que o erro que ele precisa
> detectar?**

Se o reconhecedor erra 20% dos fonemas numa fala *correta*, ele não pode
acusar uma troca de /θ/ por /t/: o sinal fica dentro do ruído e o aluno recebe
correção de erro que não cometeu. É a mesma classe de falha que a E4 existe
para evitar do lado do modelo de linguagem, e que a §3.2 da Fase 1.5 evita no
STT mandando a transcrição para o campo de texto. **Aqui não há campo de texto
onde se esconder**: ou o piso é baixo, ou a feature não existe.

---

## 1. O que a fase tem que entregar

| | |
|---|---|
| **Avaliação de pronúncia** | O aluno fala, e o retorno diz *qual fonema* saiu errado e por quê — citando regra curada, não prosa do modelo |
| **Repetição espaçada** | O drill escolhe a regra por **agendamento**, não por frequência no error log: o que o aluno acertou três vezes sai da fila, o que ele erra volta |
| **`pronunciation.yaml`** | A base contrastiva de fala que a E3 adiou, na categoria que o schema já tem |

As duas metades não se cruzam no código, e isso é de propósito: a pronúncia
acrescenta uma fonte de sinal (`pronunciation_signal`), e a repetição espaçada
consome as fontes que existirem (`correction` hoje, mais a nova depois). Quem
for implementar uma não precisa esperar a outra.

---

## 2. A medição que abre a fase

Tudo abaixo roda com `uv run python scripts/measure_phonemes.py`. Máquina: a
da §1.3 do plano da Fase 1 — Intel Core Ultra 5 135U, 14 threads, sem CUDA.

### 2.0 A pilha, e por que ela não custa um wheel

O achado que tornou esta medição barata veio antes dela: **metade da pilha de
pronúncia já estava instalada**, pelo extra `voice` da Fase 1.5.

| Peça | De onde vem | Custo novo |
|---|---|---|
| G2P (texto → IPA) | `phonemizer` 3.6 + `espeakng_loader`, que o `kokoro-onnx` traz | **zero** |
| Runtime | `onnxruntime` 1.31, que o Kokoro já usa | **zero** |
| Reconhecedor de fonema | `onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX`, int8 | 318 MB de peso |
| Síntese controlada | Kokoro com `is_phonemes=True` | **zero** |

É o mesmo padrão do Silero na Fase 1.5 — *"a dependência que o turn-taking
automático precisa já está no processo"* (§1 do plano 1.5). Aqui a dependência
que a pronúncia precisava também já estava, e por acidente: o Kokoro precisa
de um G2P para falar, e o G2P que ele traz é o mesmo que o avaliador precisa
para saber o que o aluno *deveria* ter dito.

Três consequências que decidiram a escolha do modelo:

- **O caminho oficial não entra.** `transformers` + `torch` acrescentaria ~2 GB
  de wheels a um projeto que fixou o interpretador em 3.12 justamente para
  caber nos wheels que já tem (§1.3). O ONNX exportado roda no runtime que já
  está no processo.
- **int8 é o padrão da casa.** Whisper roda int8, Kokoro roda int8. 318 MB
  contra 1.264 MB do fp32 — e a §2.2 da Fase 1.5 já estabeleceu que nesta
  máquina o tamanho do peso é orçamento, não preferência.
- **O alfabeto casa.** Este modelo emite IPA do espeak, que é exatamente o que
  o `phonemizer` desta máquina produz. Sem isso seria preciso um mapa entre
  dois inventários de fonema, e todo mapa desses é uma fonte de erro
  indistinguível de erro do modelo — que é o que esta medição não pode se
  permitir.

### 2.1 O reconhecedor cabe no orçamento, com folga

```bash
uv run python scripts/measure_phonemes.py --rtf
```

Carga do modelo: **1,84 s**.

| áudio | mediana de 3 | RTF |
|---|---|---|
| 2,4 s | 0,42 s | **0,17** |
| 5,0 s | 0,94 s | **0,19** |
| 9,1 s | 1,47 s | **0,16** |

**Mais rápido que o STT que já está no caminho.** A §7.5 mediu o Whisper
`base.en` em 1,22 s para 4,8 s de áudio — RTF ~0,25. O reconhecedor de fonema
é um modelo *large* (318 MB contra 2,5 MB de índice + peso do `base.en`), e
ainda assim ganha, porque não decodifica autorregressivamente: uma passada
convolucional mais CTC guloso, sem beam e sem modelo de linguagem.

A ausência de beam é desenho, não economia. Um decodificador que sabe quais
sequências de fonema são plausíveis em inglês **conserta a pronúncia do aluno
antes de ela chegar ao avaliador** — o mesmo defeito que a análise inicial §3
aponta no Whisper, que transforma "tink" em "think" sem avisar. Aqui o erro é
o produto; qualquer coisa que o suavize é dano.

### 2.2 O piso: 0,061 de PER em fala correta

```bash
uv run python scripts/measure_phonemes.py --floor
```

Oito frases com os erros típicos de brasileiro, sintetizadas pelo Kokoro a
partir da **transcrição de fonemas correta** que o `phonemizer` deu, e
comparadas com essa mesma transcrição.

| regra | PER em fala correta |
|---|---|
| `pron.th-unvoiced` | 0,00 |
| `pron.th-voiced` | 0,06 |
| `pron.epenthesis-initial` | 0,06 |
| `pron.epenthesis-final` | 0,00 |
| `pron.vowel-length` | 0,00 |
| `pron.h-dropping` | 0,09 |
| `pron.ing-alveolar` | 0,06 |
| `pron.r-initial` | 0,08 |
| **mediana** | **0,061** |

6% de erro de fonema é piso baixo, e o que ele é feito de importa mais que o
número: as discordâncias são `/ɔːɹ/→/ɜː/`, `/ɪɹ/→/ɪ/`, `/ɐ/→/æ/` — rótico e
vogal reduzida, **as três coisas em que duas transcrições do mesmo inglês
correto legitimamente divergem**. Não é o modelo falhando; é o espeak
transcrevendo um inglês e o Kokoro falando outro.

Isso tem consequência de desenho, e ela está na §3.4.

### 2.3 O sinal: 6 dos 8 erros nomeados pelo fonema certo

```bash
uv run python scripts/measure_phonemes.py --discriminate
```

O mesmo par, agora com a substituição injetada **no nível do fonema**: a
transcrição correta com `θ → t`, `iː → ɪ`, `h → ∅`, e o Kokoro lendo as duas.
Mesma voz, mesma prosódia; o que muda entre os dois lados é o fonema trocado,
e só ele. É o que permite atribuir a diferença de PER à troca e não ao
falante.

"Pegou?" não é o PER subir. É o alinhamento conter **a operação exata** que a
troca produz no papel — e o gabarito vem das duas transcrições, não de eu
nomear à mão o que esperar.

| regra | PER correto | PER errado | Δ | pegou? |
|---|---|---|---|---|
| `pron.th-unvoiced` | 0,00 | 0,05 | +0,05 | **sim** — `/θ/→/t/` |
| `pron.th-voiced` | 0,06 | 0,19 | +0,12 | **sim** — `/ð/→/d/` |
| `pron.epenthesis-initial` | 0,06 | 0,06 | +0,00 | não — ouviu `+/ɐ/`, gabarito `+/ɪ/` |
| `pron.epenthesis-final` | 0,00 | 0,06 | +0,06 | **sim** — `+/i/` |
| `pron.vowel-length` | 0,00 | 0,18 | +0,18 | **sim** — `/iː/→/ɪ/` |
| `pron.h-dropping` | 0,09 | 0,73 | +0,64 | **sim** — `-/h/` |
| `pron.ing-alveolar` | 0,06 | 0,06 | +0,00 | **não** |
| `pron.r-initial` | 0,08 | 0,31 | +0,23 | **sim** — `/ɹ/→/h/` |
| **mediana** | **0,061** | **0,119** | | **6 de 8** |

Seis de oito, e não só detectados: **localizados e nomeados**. `/θ/ came out as
/t/` é ensino; `82% accuracy` é placar. O backtrace do alinhamento é o que
transforma um no outro, e é a razão de esta medição reportar a operação e não
só a distância.

As duas que escaparam escaparam por razões diferentes, e a diferença é
desenho:

- **`epenthesis-initial`** é quase-acerto. O reconhecedor **ouviu a vogal
  inserida** — reportou `+/ɐ/` onde o gabarito dizia `+/ɪ/`. A classe do erro
  está certa; a identidade da vogal, errada. Ver §3.4.
- **`ing-alveolar`** é ponto cego de verdade: `/ŋ/ → /n/` passou batido, o
  modelo ouviu `/ŋ/` nos dois lados. Defensável — *speakin'* é variante nativa
  de inglês, e um modelo treinado em fala espontânea multilíngue aprendeu a
  tratá-la como tal. O custo é que esse item **não pode ser curado como regra
  de pronúncia nesta pilha**, e a §3.2 diz o que fazer com isso.

**O primeiro número desta tabela foi errado, e vale registrar por quê.** A
primeira rodada reportou os dois itens de epêntese como não detectados. O
defeito era o critério: ele só sabia olhar substituição, e **vogal epentética
é inserção**. Metade da lista de erros de brasileiro é inserção
(*isport*, *bigi*), então o critério estava cego para metade do problema que a
fase existe para resolver. O conserto foi tirar o gabarito da minha mão e
derivá-lo das duas transcrições — ver o docstring de `ops()`.

### 2.4 O proxy de sotaque: o alinhamento desaba

```bash
uv run python scripts/measure_phonemes.py --accent
```

O par controlado responde *"o reconhecedor vê a troca?"*. Não responde
*"o reconhecedor vê a troca na fala de um aluno?"*, e a diferença é a fase
inteira. O proxy mais próximo que esta máquina produz sem um humano: uma voz
brasileira do Kokoro (`pm_alex`) lendo o **texto inglês** pelo G2P de
português — a fonologia do português aplicada à ortografia do inglês.

| regra | PER com o proxy | gabarito nomeado? |
|---|---|---|
| `pron.th-unvoiced` | 0,58 | sim |
| `pron.th-voiced` | 1,00 | não |
| `pron.epenthesis-initial` | 0,25 | não |
| `pron.epenthesis-final` | 0,12 | não |
| `pron.vowel-length` | 0,59 | não |
| `pron.h-dropping` | 0,91 | não |
| `pron.ing-alveolar` | 0,76 | não |
| `pron.r-initial` | 0,77 | sim |
| **mediana** | **0,676** | **2 de 8** |

**O PER alto não é a notícia; a notícia é o que ele faz com o alinhamento.**
A 0,676, dois de cada três fonemas discordam — e quando tudo é erro, nada é
*o* erro. Os erros que aparecem são plausíveis (`/ð/→/t/`, `/ð/→/d/`,
`/h/→/ɾ/`, `/iː/→/æ/`: exatamente a interferência que se espera), mas vêm
afogados em dez outros, e um avaliador que despejasse essa lista no aluno
estaria pior que calado.

**Ressalva de honestidade, e ela é grande.** Este proxy é mais severo que o
aluno do produto. Ler ortografia inglesa pelo G2P de português produz algo
mais perto de um brasileiro que nunca estudou inglês — *"They brought the
other brother"* sai com PER 1,00, o que não é sotaque, é outra língua. O
público da §1.1 é **B1 em diante**, e um B1 produz algo muito mais perto do
inglês.

Então o que as duas medições fazem é **colocar o aluno real entre dois
colchetes** — 0,12 no par controlado, 0,68 no proxy — **sem dizer onde dentro
deles ele cai.** E a viabilidade do avaliador depende exatamente disso.

### 2.5 O que esta medição não diz

O áudio das três tabelas é **sintético**, e aqui isso dói mais que na §2.3 da
Fase 1.5. Lá o sintético media latência honestamente (o Whisper leva o mesmo
tempo em áudio difícil) e media exatidão no caso mais fácil que existe. Aqui a
exatidão *é* o produto.

Fala humana tem coarticulação, hesitação e um sotaque que **não é um conjunto
de substituições discretas**: um brasileiro que troca /θ/ por /t/ também
encurta a vogal seguinte e muda o ritmo da frase. O par controlado é o **teto**
do desempenho, não a expectativa. O proxy é um **limite inferior pessimista**,
não o aluno.

`--audio fala.wav --say "what I said"` existe para fazer a medição que
importa, e é a E9.

---

## 3. O que a medição mudou no desenho

### 3.1 A pronúncia não entra no caminho da primeira palavra

O orçamento do turno de voz já está estourado: **3,99 s até ouvir a primeira
palavra** com a iGPU ligada, contra a meta de 1–2 s (§7.5 do plano 1.5), e a
§7.5 registrou que agora o gargalo é o áudio — STT 1,22 s + TTS 1,29 s
custariam 2,5 s mesmo com o LLM em zero.

Acrescentar 0,94 s de reconhecimento de fonema *dentro* desse caminho levaria
o turno a ~5 s e pioraria a única coisa que a Fase 1.5 entregou bem. Então:

> **O reconhecimento de fonema roda depois do turno, em paralelo com a fala do
> professor, e o retorno de pronúncia aparece quando aparecer.**

Isso é mais que uma otimização; é uma escolha pedagógica que a medição torna
gratuita. Interromper o aluno com fonética no meio da conversa é o oposto do
que a E4 estabeleceu (*"Conversa primeiro"*, §5.2 da Fase 1). O retorno de
pronúncia é material de revisão, não de interrupção — e é por isso que ele
pode esperar 0,94 s sem que ninguém perceba.

### 3.2 Erro de pronúncia sem regra curada não chega ao aluno

É a conclusão nº 1 da E4 (§5.2 da Fase 1), e a §2.4 mostra que aqui ela não é
só coerência de arquitetura — é a diferença entre um avaliador e um gerador de
ruído. A 0,676 de PER, reportar todo fonema discordante é despejar quarenta
erros numa frase de oito palavras.

O avaliador **só reporta operação de alinhamento que casa com uma assinatura
de `pronunciation.yaml`.** Tudo o mais é descartado, inclusive quando está
certo. Mesma lógica, mesma razão, e agora com um número que a justifica.

Consequência imediata para a E10: **um item de `pronunciation.yaml` precisa de
teste de discriminação, não só de exemplo.** `pron.ing-alveolar` é a prova —
ele é um erro real, bem documentado, e esta pilha **não o ouve** (§2.3). Um
item que não sobrevive ao `--discriminate` não entra na base, do mesmo jeito
que na E3 um padrão que não generalizava quebrava a suíte.

### 3.3 O portão de confiança: calar é resposta

Quando o PER da elocução inteira passa de um teto, o reconhecedor **não está
entendendo o aluno**, e nada que ele diga sobre fonemas individuais tem valor.
A resposta certa é não dizer nada sobre pronúncia naquele turno.

É a mesma salvaguarda da §3.2 da Fase 1.5 (a transcrição vai para o campo de
texto porque o STT erra) e da E4 (frase sem candidato não vira pergunta ao
modelo). A §2.4 dá a ordem de grandeza do teto: o par controlado fica em 0,12,
o proxy em 0,68. O valor exato é da E9 — ele **não pode** ser calibrado em áudio
sintético, porque é exatamente o número que separa "aluno com sotaque" de
"reconhecedor perdido", e o sintético não tem nem um nem outro.

### 3.4 A epêntese é classe, não fonema

A §2.2 mostra que o piso é feito de rótico e vogal reduzida, e a §2.3 mostra o
reconhecedor ouvindo `+/ɐ/` onde o gabarito dizia `+/ɪ/`. As duas coisas são o
mesmo fato: **a identidade exata de uma vogal não é confiável nesta pilha; a
presença de uma vogal onde não devia haver é.**

Então o retorno ao aluno se formula na classe, não no símbolo:

- bom: *"you added a vowel before /sp/ — it's **sport**, not **isport**"*
- ruim: *"you said /ɐ/ where the expected phoneme was /ɪ/"*

O segundo é mais preciso e mais errado — precisão que a medição não sustenta.
E o primeiro é melhor pedagogia de qualquer jeito, o que torna esta uma
restrição que não custa nada. A assinatura de um item de `pronunciation.yaml`
é portanto um **conjunto de operações aceitáveis** (`+/ɪ/`, `+/ɐ/`, `+/i/`,
`+/ə/` todas contam como epêntese), não uma operação única.

### 3.5 A iGPU da §7 é o que paga a pronúncia

A §7.5 da Fase 1.5 tirou o LLM do caminho crítico pela iGPU, e a §5 do mesmo
plano guarda uma invariante que parecia barrar esta fase: *"nunca duas
inferências disputando a mesma CPU"*. Com o LLM na CPU, rodar o reconhecedor de
fonema junto com o turno seria exatamente isso.

Com o LLM na iGPU, **os 14 threads ficam livres durante a geração** — e o
reconhecedor de fonema é o primeiro candidato a gastá-los. A §7.7 dizia que o
que a iGPU compra é latência de prompt; esta fase mostra que ela compra uma
segunda coisa, que a §7 não contou: **espaço de CPU para um terceiro modelo.**

A invariante continua valendo e ganha a redação precisa: nada de duas
inferências disputando **o mesmo dispositivo**. Pronúncia e TTS, os dois na
CPU, continuam serializando com lock próprio.

---

## 4. Escopo

### Dentro

- Reconhecedor de fonema atrás de contrato, em `voice/phonemes.py` — recebe
  bytes, devolve fonemas. **Não importa `teaching/`**, pela regra de sentido
  único da Fase 1.5.
- Avaliador em `teaching/pronunciation.py` — recebe fonemas ouvidos + texto
  esperado, devolve erros **com `rule_id`**. É pedagogia, logo mora em
  `teaching/`.
- `data/contrastive/pronunciation.yaml`, na categoria que o schema já tem.
- `POST /api/pronunciation`, e o retorno na UI **depois** do turno.
- `pronunciation_signal` no SQLite, e agregação por fonema no `/progress`.
- `teaching/schedule.py`: repetição espaçada sobre `correction` +
  `pronunciation_signal`; o drill escolhe a regra por agendamento.
- Extra `pronunciation` no `pyproject`, desligado por default
  (`TC_PRONUNCIATION_ENABLED=false`) — mesma política do extra `voice`.

### Fora, explicitamente

- **Turn-taking sem botão.** Está na §6 do plano 1.5, não na Fase 2 da análise
  inicial, e não encosta em nada daqui: é máquina de estados de áudio, não
  ensino. Entra quando entrar, e a dependência (Silero) continua no processo.
- **Nota de pronúncia, placar, percentual.** A §3.4 diz por quê: a pilha
  sustenta classe de erro, não medida fina. Um número de 0 a 100 seria
  precisão inventada.
- **Treino de prosódia e ritmo.** O alinhamento é de fonema; entoação e
  *stress* exigem outro modelo e outra medição.
- **Drill com várias regras por sessão.** A §5.4 da Fase 1 o adiou para cá
  junto com a repetição espaçada, mas a razão de uma regra por sessão era
  cache de prefixo, e isso não mudou. Fica fora até alguém medir o custo.
- **Embeddings na base contrastiva.** Continua item adiado (§3 da Fase 1): a
  base segue na casa dos 100 itens.

---

## 5. Ordem das entregas

A numeração continua do E7 da Fase 1 — a 1.5 foi entregue em bloco, sem
entregas numeradas.

| | Entrega | Pronto quando |
|---|---|---|
| **E8** ✅ | **Medição de abertura** — `scripts/measure_phonemes.py` com os quatro modos; piso, sinal e proxy medidos | **Feito.** Piso 0,061, sinal 0,119 com 6 de 8 erros nomeados pelo fonema certo, proxy 0,676 com 2 de 8. Ver §2 |
| **E9** 🚧 | **A dívida da §2.3 paga** — gravações de falante brasileiro B1+ lendo as frases curadas, PER medido com `--audio`, teto do portão da §3.3 calibrado | Há ≥10 elocuções humanas medidas; o PER delas e a fração de gabaritos nomeados estão registrados aqui; **e a decisão de seguir ou parar está escrita com o número na mão** |
| **E10** | **`pronunciation.yaml`** — a base de fala que a E3 adiou; ~15 itens, cada um com assinatura de operações aceitáveis (§3.4) | Os itens carregam e validam pelo schema; cada um tem teste; **nenhum item entra sem passar no `--discriminate`** (§3.2) |
| **E11** | **O reconhecedor e o avaliador** — `voice/phonemes.py` atrás de contrato como o `whisper.py`, `teaching/pronunciation.py` casando alinhamento com regra | `POST /api/pronunciation` devolve só erros com `rule_id`, cala acima do portão, e **`teaching/turn.py` não muda uma linha** |
| **E12** | **Pronúncia no produto** — retorno na UI depois do turno, `pronunciation_signal` persistido, `/progress` agregando por fonema | O aluno fala, conversa normalmente, e o retorno de pronúncia aparece sem atrasar a fala do professor (§3.1) |
| **E13** | **Repetição espaçada** — `teaching/schedule.py`, e o drill escolhendo a regra por agendamento | A regra do drill vem do agendamento e não da contagem; regra acertada *n* vezes seguidas sai da fila e volta depois do intervalo; um teste mede o intervalo |

**A E9 é portão, não etapa.** É a única entrega da fase cujo resultado pode
cancelar as outras, e ela vem primeiro por isso. A §2.4 colocou o aluno real
entre 0,12 e 0,68 sem dizer onde; se ele estiver perto de 0,68, o avaliador
não é viável nesta pilha e a fase vira outra coisa (um `small` em vez do
`large`, *forced alignment* com o texto conhecido em vez de reconhecimento
livre, ou pronúncia fora do escopo). Escrever E10–E12 antes disso é escrever
código que a medição pode jogar fora — exatamente o que a abertura por medição
existe para evitar.

A **E13 não depende da E9** e pode correr em paralelo: ela consome
`correction`, que existe desde a primeira migração.

---

## 6. Riscos conhecidos desta fase

| Risco | Como lidamos |
|---|---|
| **O aluno real pode cair perto do 0,676 do proxy** | É o portão da E9, e ele vem antes de qualquer código de produto. A decisão fica escrita com o número |
| **Áudio sintético calibra o portão errado** | Não calibra: a §3.3 proíbe calibrar o teto em sintético, de propósito. O valor sai da E9 ou não sai |
| **Erro real de brasileiro reportado como erro que ele não cometeu** | Duas camadas: só operação com `rule_id` chega ao aluno (§3.2), e acima do portão o avaliador cala (§3.3) |
| **Precisão inventada no retorno** | O retorno fala em classe de erro, não em símbolo de vogal (§3.4) |
| **`pron.ing-alveolar` não é o único ponto cego** | Todo item passa pelo `--discriminate` antes de entrar na base (§3.2), e o que não passa fica documentado como ponto cego em vez de virar falso negativo silencioso |
| **Três modelos nativos no mesmo processo** | O lock por motor da Fase 1.5 continua, e a §3.5 explica por que a CPU agora caberia: o LLM saiu dela |
| **318 MB a mais de peso** | Extra desligado por default, como a voz. Quem não quer pronúncia não baixa nada |

---

## 7. O que vem depois

- **Turn-taking sem botão**, com o Silero que já está no processo (§6 do plano
  1.5, inalterado).
- **Prosódia e ritmo** — o que o alinhamento de fonema não vê.
- **A dívida da análise (~10 s por turno com erro)**, que a §7.7 da Fase 1.5
  registrou como a maior fatia isolada do turno e que a iGPU quase não toca. O
  caminho continua sendo um modelo melhor só para ela.
- **Mobile, multiusuário, login, instalador** — depois da Fase 2, como a §4 da
  Fase 1 já dizia.
