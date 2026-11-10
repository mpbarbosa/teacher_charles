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

**Ressalva medida depois, e ela é sobre o exemplo e não sobre a regra.** O
`pron.ing-alveolar` deixou de ser a prova: em fala humana esta pilha **ouve** a
troca — `/ŋ/→/n/` nomeado duas vezes numa tomada limpa (§3.5) —, e o que não a
distinguia era a renderização do Kokoro no par sintético. O teste de
discriminação continua sendo condição de entrada, pela razão escrita acima; o
que muda é que a fonte humana desempata a sintética, e um item reprovado só pelo
par entra com a razão registrada em vez de ficar de fora.

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

**O teto nunca foi calibrado, e a §3.5 mede por que não deve ser.** Em fala
humana as duas faixas se sobrepõem: uma leitura sem erro de pronúncia nenhum
dá PER 0,364, e uma com o erro curado presente dá 0,188. O portão continua
sendo "calar é resposta", e o que o dispara é a **operação de alinhamento**
casada com regra curada — não um número agregado. Ler esta subseção como um
pedido de teto é perseguir um valor que os dados dizem não existir.

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

### 3.5 Há um portão antes do portão, e ele é de sinal

A §3.3 põe um teto no PER para o avaliador calar quando não está entendendo o
aluno. A primeira gravação de verdade feita para a E9 mostrou que **falta um
portão antes desse**, e que ele não é de fonema nenhum.

A gravação saiu cravada no fundo da escala: 0,54% das amostras saturadas, RMS
0,201 contínuo nos 5 s sem as pausas que a fala tem, 36% da energia abaixo de
100 Hz e picos em 8–31 Hz. O reconhecedor respondeu a isso devolvendo fonema
nenhum, e o script reportou **PER 1,000 com os 13 fonemas "comidos"** — número
que o modelo produziu honestamente e que não significa nada sobre pronúncia.

A causa era o dispositivo, não o software: `Capture` em +30,00 dB (máximo) e
`Internal Mic Boost` em +12,00 dB (máximo) — **+42 dB** num microfone interno,
que satura com a respiração. Vale registrar o que *não* era: trocar `arecord`
por `pw-record` não muda nada, porque os dois saem do mesmo mixer ALSA. A
captura pelo PipeWire veio pior (27% de amostras saturadas, RMS 0,571).

Então o script agora **recusa o arquivo com o diagnóstico** antes de carregar o
modelo, por cinco perguntas: silêncio, transiente de abertura, saturação,
fração de energia abaixo de 100 Hz e contraste de fala. É a mesma salvaguarda do piso de bytes do
`voice/whisper.py`, que devolve texto vazio sem carregar o Whisper em vez de
deixá-lo alucinar `"Thank you."` em silêncio — e é a mesma lógica da §3.3 um
degrau acima: um PER de áudio quebrado iria para uma tabela indistinguível de
medição.

A consequência para a E11 é direta: **o diagnóstico de sinal é do `voice/`, não
do `teaching/`.** Ele responde "este áudio é áudio?", que é pergunta de quem
recebe bytes, e o avaliador nunca deve receber um turno que não passou por ele.

#### A lista mudou ao ser usada, e cresceu

As gravações seguintes corrigiram duas das quatro, e as duas correções são do
mesmo tipo: **a estatística descrevia o arquivo, e a pergunta era sobre a
elocução.**

A quarta pergunta original era **offset DC do arquivo inteiro**, com teto em
0,02 — e ela recusou uma gravação boa, por 0,031. O offset era real e a leitura
estava errada. Medido bloco a bloco, ele é um **degrau de DC que decai por
metade a cada ~0,1 s**: média +0,585 no primeiro décimo de segundo, +0,073 em
0,4 s, +0,002 em 1,0 s, contra 0,004 do silêncio do arquivo. É o conversor
abrindo, não um viés do conversor — cortado o primeiro segundo, a média do que
resta é +0,0007. E "consertar" subtraindo a média global era pior que a recusa:
empurrava o silêncio de 0,004 para 0,031 e afogava toda estatística de nível
depois dela.

A pergunta certa usa o que distingue as duas coisas: **fala tem média zero, e um
degrau não.** A recusa olha a média dos primeiros 0,3 s, e um aluno que comece a
falar imediatamente não dispara nada. E como o transiente entra em *toda*
gravação deste dispositivo, o lugar de resolver é na gravação, não no
diagnóstico: o `--record` captura um segundo a mais e corta `LEAD_IN_SECONDS =
1,0 s` — um segundo e não meio porque é onde o rabo chega ao piso de ruído.

A segunda correção é no **aviso de nível baixo**, que existe para um corpus de
dez elocuções não ser gravado baixo e isso só se descobrir depois de medir as
dez. Ele olhava o RMS do arquivo, e o transiente o punha em 0,123 — doze vezes
o teto do aviso — para uma fala que estava em 0,07. Agora o nível sai da
**mediana dos blocos de 100 ms que são fala**, e os blocos de fala saem de um
limiar ancorado no **piso de ruído** (percentil 10 dos blocos). As duas âncoras
óbvias erram, e as duas erraram aqui: o **pico**, porque o transiente põe o
limiar acima da própria fala e a fala toda vira silêncio; a **mediana**, porque
só serve se o silêncio for maioria — numa gravação meio a meio ela cai no vão
entre os dois modos e deixa a maior parte dos blocos de fala de fora.

O aviso também diz **de quanto é o buraco**, em dB. A primeira versão sugeria
`Capture 70%` fixo, que nesta máquina são +4,5 dB — menos da metade dos ~8 dB
que faltavam na gravação que o disparou. Conselho que não cobre o buraco custa
uma gravação para ser descoberto, e um corpus de dez não tem essa folga. Junto
vai a relação sinal-ruído da fala, porque ela separa as duas alavancas: ganho
amplifica o ruído junto e a deixa onde estava, falar mais perto a melhora.

Saber onde a fala está deu um aviso novo de graça: o script diz quando ela vai
**até o fim do arquivo**, sinal de que o último fonema pode ter ficado fora da
janela. Uma elocução cortada no fim devolve um PER que parece medição — a mesma
classe de erro, num grau que não dá para recusar, porque daqui de dentro não há
como distinguir "a frase acabou junto com a janela" de "a janela acabou antes da
frase". Quem gravou sabe, e é a quem o aviso se dirige.

Por fim, a nota de ganho **não** sai nas recusas que não são de ganho: ela manda
baixar o ganho, e no transiente isso é o contrário do remédio — a fala ali já
estava baixa.

#### A primeira elocução humana, e o que o corte fixo fez com ela

Com o ponto de operação acima, a primeira gravação de voz humana brasileira
lendo uma frase curada saiu assim — `gravacoes/e9-r-initial.json`, md5
`d98fb571`, 5,0 s:

```
esperado  /ð ə  ɹ ɛ d  k ɑːɹ  ɪ z  ɹ ɛ d i/
ouvido    /     ɹ ɛ d  k ɑːɹ  ɪ z  ɹ ɛ d i/     PER 0,154
```

**Onze fonemas de onze exatos**, e os dois erros são a deleção de `/ð ə/`. Isso
é o primeiro sinal real de que o reconhecedor lê o inglês deste falante — a
pergunta que a E9 existe para responder —, mas **não é ainda o número da E9**,
porque a deleção provavelmente é do script e não do falante: a fala começa em
0,0 s, e o corte de transiente da época levava 1,0 s fixo do começo. A mensagem
da gravação pedia "fale agora, sem esperar", isto é, pedia para falar dentro do
segundo que o corte joga fora.

Dois reparos saíram disso, e o segundo é a lição geral:

* O corte passou a ser **por decaimento, não por duração**: varre blocos de
  10 ms e para no primeiro que não é dominado por DC (`DC_DOMINANCE`), com teto
  de 1,0 s. Medido, corta 900 ms de um transiente de tamanho real e 90 ms de um
  começo limpo — contra 1000 ms sempre.
* Ganhou um aviso **simétrico** ao de truncamento, para fala encostada no
  *início*. Um fonema inicial perdido vira deleção no alinhamento, idêntica a
  "o aluno não pronunciou" — e em `pron.epenthesis-initial`, cuja regra é sobre
  o começo da sílaba, teria invalidado a medição sem deixar rastro.

O par de avisos é a forma geral do problema que a §3.5 vem perseguindo: **toda
borda do arquivo é um lugar onde o artefato mente sobre a elocução.**

E o corte por decaimento errou uma vez mais, de um jeito que só fala real
mostra: ele exigia que o degrau começasse na **amostra zero**. Numa gravação o
bloco de 0 ms saiu limpo — média +0,0004 — porque o dispositivo ainda não havia
rabeado; o degrau começou em 10 ms e cravou em −1,0 até 110 ms. A versão que
parava no primeiro bloco não dominado por DC parava no bloco 0, cortava 0 ms, e
o diagnóstico recusava o arquivo que o corte deveria ter salvado. Agora o corte
**procura o início** do degrau numa janela de 100 ms (`ONSET_WINDOW_SECONDS`) e
segue dali enquanto o bloco for dominado por DC. Provado nos quatro sinais:
transiente negativo cravado (recusado → 870 ms cortados → limpo), degrau
positivo sintético começando em 30 ms (recusado → 1000 ms → limpo), e duas
gravações limpas, que perdem 30 ms e 50 ms e continuam limpas — o corte não come
fala.

#### As primeiras elocuções humanas, e o portão abre

Seis gravações, das quais o diagnóstico recusou uma e marcou duas com aviso de
borda. As três limpas são a medição:

| regra | arquivo (md5) | esperado → ouvido | PER |
|---|---|---|---|
| `pron.r-initial` | `e9-novo` (`af87b8d1`) | `/ð ə ɹ ɛ d k ɑːɹ ɪ z ɹ ɛ d i/` → idêntico | **0,000** |
| `pron.r-initial` | `e9-r-initial` (`5907fa81`) | `/ɛ/`→`/oʊ/`, `/t/` inserido | 0,154 |
| `pron.th-voiced` | `e9-th-voiced` (`b704dc72`) | `/ɔː/`→`/oʊ/`, `/ɪ/`→`/ɚ/`, **`/ð/`→`/d/`** | 0,188 |
| `pron.th-voiced` | `e9-th-voiced` (`3ba35169`) | `/ɔː/`→`/ɑː/`, `/ɪ/`→`/ə/` — os quatro `/ð/` certos | 0,125 |
| `pron.th-unvoiced` | `e9-th-unvoiced` (`30d91003`) | `/s/`→`/z/` em "this is" — o `/θ/` certo | **0,053** |
| `pron.th-unvoiced` | `e9-th-unvoiced` (`523e87f7`) | réplica independente, transcrição idêntica | **0,053** |

Três coisas, e a terceira é a que decide a fase.

**Fala humana correta dá PER 0,000.** Treze fonemas de treze, exatos — *abaixo*
do piso sintético de 0,061. O piso sintético media duas máquinas discordando
(Kokoro contra o reconhecedor); uma pessoa lendo com clareza casa com o G2P
exatamente. O piso de verdade deste caminho é mais baixo que o que a §2.2
estimou, não mais alto.

**O erro curado foi detectado, nomeado e localizado.** A frase de
`pron.th-voiced` tem `/ð/` quatro vezes — *They*, *the*, *other*, *brother* — e
saiu `/ð/` em três e **`/d/` em "other"**, que é exatamente a substituição que o
item curado prevê. Não é um PER alto dizendo "algo está errado": é o alinhamento
apontando qual fonema, em qual palavra. É o que a §2.3 pedia e o proxy de
sotaque não conseguiu entregar.

**E a margem é larga, não estreita.** Esta subseção dizia o contrário uma versão
atrás, baseada na elocução de 0,154 — e estava errada, porque aquela elocução
continha dois desvios reais, não era fala correta. Com a leitura limpa da mesma
frase em 0,000, o quadro é:

| | PER |
|---|---|
| **humano, fala correta** | **0,000** |
| piso sintético | 0,061 |
| sinal sintético | 0,119 |
| **humano, erro curado presente** | **0,188** |
| proxy de sotaque sintético | 0,676 |

Piso e sinal **em fala humana real** ficam separados por 0,188 — margem maior
que a sintética (0,061 a 0,119), não menor. O proxy de sotaque de 0,676 era
pessimista: não é o que a fala de um brasileiro B1+ faz com este reconhecedor.

#### O gabarito carrega escolhas que não são erro, e isso muda a E10

A segunda leitura de `pron.th-voiced` é o achado mais útil até aqui, porque os
dois "erros" que sobraram **não são erros de pronúncia**:

* `/ɔː/`→`/ɑː/` em "brought" — a fusão *cot/caught*, corriqueira em inglês
  americano. O G2P escreve `/ɔː/`; quem fala com a vogal fundida diz `/ɑː/`.
* `/ɪ/`→`/ə/` em "the" — as duas pronúncias normais do artigo, e `/ðə/` é a
  comum mesmo antes de vogal.

E a terceira, que apareceu depois e muda o desenho, está logo abaixo.

Fonemicamente a leitura está certa, e o PER de 0,125 é feito inteiro de
**variantes do gabarito**. Duas consequências de projeto, as duas para a E10:

1. O gabarito precisa de **conjuntos de equivalência** — `/ɔː/ ≡ /ɑː/`,
   `/ðɪ/ ≡ /ðə/`, e o que mais aparecer nas dez — ou o avaliador dirá a um
   falante correto que ele errou. É a mesma falha que a E3/E4 existem para
   evitar do lado do modelo, e a razão pela qual `POST /api/stt` não grava
   turno: **corrigir erro que o aluno não cometeu é o pior resultado possível.**
2. O portão da §3.3 **não pode ser um teto de PER.** As duas leituras da mesma
   frase dão 0,188 e 0,125 — delta de 0,063 —, e o que as distingue não é o
   valor: é que uma tem a operação `/ð/`→`/d/` em "other" e a outra não. O
   avaliador deve decidir **por operação de alinhamento**, com a confiança
   local, e nunca por PER agregado. Agregado, erro curado e variante de
   gabarito moram na mesma faixa.

#### Fala conectada: o gabarito não tem como estar certo entre palavras

A elocução de `pron.th-unvoiced` saiu com PER 0,053 — um erro em dezenove — e o
erro é `/s/`→`/z/` em **"this is"**. Isso não é desvio de brasileiro: `/s/`
final entre vogais sonoriza na fala fluente, e `/ðɪz ɪz/` é o que um nativo diz
também. É a terceira variante de gabarito da §3.5, e é de **tipo diferente**
das duas primeiras:

| variante | tipo | expressável como |
|---|---|---|
| `/ɔː/ ≡ /ɑː/` | fusão vocálica | par de fonemas |
| `/ðɪ/ ≡ /ðə/` | redução de átona | par de fonemas |
| `/s/ → /z/` entre vogais | **assimilação entre palavras** | regra de contexto |

As duas primeiras cabem num conjunto de equivalência; a terceira não. O
gabarito vem do G2P sobre o **texto escrito**, que transcreve palavra por
palavra e não modela o que acontece nas fronteiras quando se fala corrido. Um
par `/s/ ≡ /z/` incondicional seria pior que o problema: deixaria passar
sonorização onde ela é erro de verdade. A E10 precisa de equivalência
**condicionada ao contexto**, não de uma tabela de pares.

E há uma tendência nos cinco pontos que reforça a §3.3: o piso **não é
constante**. Fala correta deu 0,000 numa frase de 13 fonemas, 0,053 numa de 19 e
0,125 numa de 16 com mais fronteiras — porque mais fronteira de palavra significa
mais oportunidade de assimilação. Um teto global de PER teria de ser frouxo o
bastante para a frase mais longa, e aí não detectaria erro nenhum na mais curta.
**Mais um argumento para decidir por operação de alinhamento, nunca por PER
agregado.**

#### A medição se replica, e isso é o que um portão precisa

A frase de `pron.th-unvoiced` foi gravada duas vezes, independentemente:

| | gravado | duração | fala em | bytes | PER |
|---|---|---|---|---|---|
| A | 17:00:43 | 6,02 s | 0,4–2,9 s | 192 684 | 0,053 |
| B | 17:01:21 | 6,09 s | 0,9–3,8 s | 194 924 | 0,053 |

Áudios diferentes — md5, tamanho e posição da fala na janela todos distintos —
e **transcrição fonêmica byte a byte idêntica.** Variância zero nos dezenove
fonemas.

Isso vale mais que um ponto a mais na contagem, por três razões:

1. **Um portão precisa de medida reprodutível.** Sem réplica, um PER baixo pode
   ser sorte do alinhamento; com ela, não é.
2. Confirma que o `/s/`→`/z/` é **sistemático** na fala conectada deste
   falante, não lapso — o que sustenta lê-lo como variante de gabarito e não
   como erro.
3. Confirma que o `/θ/` correto é estável, e portanto que a hipótese curada de
   `pron.th-unvoiced` de fato não se aplica a ele.

A réplica é barata (uma gravação a mais) e é o único jeito de distinguir medida
de coincidência. Vale repetir pelo menos uma frase de cada tipo de operação —
substituição, inserção, deleção — antes de fechar a E9.

Faltam cinco elocuções para as dez do critério, e nada disso é definitivo com
seis.

#### O corpus se grava numa execução, não em oito

O `--corpus` grava e mede as oito frases curadas de uma vez. A razão não é
conveniência: **esta máquina reexibiu saída antiga para comando repetido** ao
longo de toda a sessão da E9 — gravação que não acontecia enquanto a saída
dizia que sim, e nem nome de arquivo único com `$(date)` escapava. Oito
invocações são oito chances de a gravação não ocorrer sem a saída denunciar;
uma é uma.

Três propriedades que a execução única dá de graça, e que o critério da E9
precisa:

* O `--say` **sai da lista curada** em vez de ser digitado. Descasamento entre
  o que foi lido e o que foi declarado já custou três gravações aqui.
* Cada arquivo é nomeado pela regra, e o relatório pelo md5 do áudio.
* A tabela final é a que a E9 pede: PER por regra e **se o erro previsto foi
  nomeado** — a verificação é a presença da operação `("sub", de, para)` no
  alinhamento, não um limiar de PER, pela razão da subseção anterior.

Ele **pula o que já está medido e limpo**, e "limpo" é o critério da E9: passou
pelo diagnóstico e não tem aviso de borda. Relatório com aviso não conta — foi
assim que um `/ð ə/` comido pelo corte quase entrou na tabela como erro do
aluno. O que ele pula, ele **completa**: o gabarito do relatório antigo é
preenchido ali mesmo, sem regravar nada, pela razão da subseção da passada
correta. Mas a hipótese que fecharia a Fase 2 — "o reconhecedor não lê brasileiro,
e sem alinhamento não há feedback" — **não se sustentou nos primeiros dados
reais.**

#### A quinta pergunta é contraste, e ela fecha a alavanca do ganho

Com o transiente cortado e o ganho subido para +20,25 dB, o aviso de nível
ficou calado e o script publicou **PER 3,462** — 45 erros em 13 fonemas, de uma
transcrição de 65 fonemas de salada em sete segundos. O aviso calou por um
atalho, não por aprovação: sem blocos de fala detectados ele caía numa medição
do arquivo inteiro, e o arquivo inteiro passava o teto que a fala não passaria.

O perfil explica o resto. O RMS por bloco de 100 ms ficou entre 0,009 e 0,050
**do começo ao fim**, sem nenhuma queda a um piso: razão percentil 90 / percentil
10 de **2,8**, contra ~12 na gravação que transcreveu limpo. Pico em −17,2 dBFS,
nada saturado. Um arquivo assim não tem elocução dentro — tem o microfone
ouvindo a sala, e o reconhecedor fazendo o que CTC faz com ruído: sílabas
plausíveis, nenhuma delas dita por ninguém.

Daí a quinta pergunta, e ela é a mais fundamental das cinco porque **não depende
de nível nenhum**: uma elocução tem pausa, a sala não. O teto é
`SPEECH_CONTRAST_MIN = 4`, entre os dois valores medidos — e **provisório**:
dois pontos, nenhum deles de conteúdo conhecido. Quem o calibra é a E9, com
elocuções de frase sabida, onde dá para dizer se um arquivo recusado era
mesmo ilegível ou só abafado.

E ela fecha a alavanca do ganho, que a §3.5 vinha puxando desde a primeira
gravação. De +11,25 dB para +20,25 dB, o piso de ruído subiu de 0,0026 para
0,0137 — **mais** que os 9 dB de ganho — e o contraste não melhorou, caiu. Era
previsível e está dito acima: ganho analógico levanta sinal e ruído juntos. A
consequência para a E9 é que o ganho está resolvido e o que falta não é
elétrico: **distância do microfone e silêncio na sala.** O ponto de operação
que já funcionou nesta máquina é +11,25 dB falando perto.

**O ponto de operação da E9, para a gravação ser reproduzível.** Um corpus cujas
condições de captura não estão escritas não é corpus, porque não dá para
remedir quando o limiar mudar. Nesta máquina:

| | valor |
|---|---|
| `Capture` | 70% — passo 44 de 63, +15,75 dB |
| `Internal Mic Boost` | 0 — +0,00 dB |
| microfone | o interno, falando perto |
| janela | `--record 7`, falando desde o primeiro instante (o primeiro segundo é descartado) |

Com isso: contraste 15,71, pico −11,0 dBFS, 42 blocos de fala em 70, e as cinco
perguntas caladas. É o melhor sinal da série e o número a reproduzir nas dez
elocuções — variar o ganho entre elas tornaria os PER incomparáveis.

#### The correct pass, checked by machine: eight clean utterances, one gabarito named

The "erro previsto nomeado?" column was `—` for every take on disk, and it is
the one column the gate reads. The reason was mechanical: the signature is
computed inside `run_audio`, so only a take recorded **after** that calculation
existed carried one — and the only route the script offered to the rest was
`--redo`, which records over the take. That spends someone's voice on a number
nobody has to speak for, and destroys a clean utterance of a corpus that needs
ten.

The gabarito is not a measurement. `_error_signature` phonemizes the curated
phrase, applies the swap to the transcription and aligns the two: it needs the
G2P and the recognizer's vocabulary, and **no audio at all**. So the `--corpus`
now writes it into the clean report it skips, at that report's own path, and the
column answers over audio that already held the answer — including the two takes
whose WAV a later recording of the same rule overwrote, because `esperado` and
`ouvido` are in the report and that is everything the alignment needs. `--audio`
gained `--rule` for the same reason: `main` was the one caller that never passed
the rule along, so a one-off measurement could not carry a gabarito either.

Counting changed too, and it was wrong in the direction that asks someone to
speak again. The loop opened **one** report per phrase, reported five clean
utterances with eight on disk — and the take it dropped for `pron.th-voiced` was
the one carrying the curated `/ð/→/d/`. The criterion counts utterances, not
rules; ten over eight phrases means replicas, and §3.5 above is where replicas
earned their place.

| regra | áudio (md5) | PER | gabarito nomeado? |
|---|---|---|---|
| `pron.r-initial` | `af87b8d1` | **0,000** | não |
| `pron.th-unvoiced` | `30d91003` | 0,053 | não |
| `pron.th-unvoiced` | `523e87f7` | 0,053 | não |
| `pron.th-voiced` | `3ba35169` | 0,125 | não |
| `pron.r-initial` | `5907fa81` | 0,154 | não |
| `pron.th-voiced` | `b704dc72` | 0,188 | **sim** — `sub /ð/→/d/` |
| `pron.epenthesis-initial` | `c8f85cb5` | 0,312 | não |
| `pron.h-dropping` | `728ff6cb` | 0,364 | não |
| **mediana** | | **0,139** | **1 de 8** |

**One of eight is not a verdict on the evaluator, and that is exactly why the
`--errado` pass exists.** In the correct pass a "não" sums two opposite things:
the evaluator missed the error, and the error was not there. Here all seven are
the second, and the reports say so phoneme by phoneme — the speaker produced all
five `/h/` of *He has a house here*, inserted no vowel before `/sp/` in *Sport*,
and kept `/θ/` and `/ɹ/` in the four takes of the two sentences that test them.
A base that is never exercised cannot be judged by how often it fires.

**The one "sim" is the first curated error named by machine on human speech.**
§3.5 read `/ð/`→`/d/` in *other* off the transcription by hand; now the
alignment is intersected with a gabarito derived from the two transcriptions,
and the operation that matched is `sub /ð/→/d/` — the one the item describes,
not one of the collateral operations the swap also produces. That is the
difference between a number and an answer the evaluator could give a student.

**And the two bands now overlap, which settles §3.3 by data instead of by
argument.** The clean take at **0,364** has no curated error in it at all: its
four disagreements are `/ɐ/→/æ/`, `/z/→/h/`, `/s/→/z/` and `/ɪɹ/→/iː/` —
reduced vowel, resegmentation across a word boundary, inter-word voicing
assimilation and the rhotic, which are the three categories §2.2 and §3.5
already named as legitimate divergence between two transcriptions of correct
English. The take **with** a real curated error sits at 0,188, **0,176 below
it**. A PER ceiling loose enough to spare this correct speaker would never fire
on the erroneous take. §3.5 reached that conclusion from a 0,063 delta between
two readings of one sentence; the gap is now wider, across sentences, and
pointing the wrong way. **The gate is per alignment operation. There is no
aggregate ceiling to calibrate, and E9 should stop looking for one.**

What the gate still owes, and neither item needs new code:

* **Two clean utterances** — one, after the recording session below. Three phrases have no clean take —
  `pron.epenthesis-final`, `pron.vowel-length` and `pron.ing-alveolar` — and all
  three were measured only with the **low-level** warning, at the old operating
  point. Recording those three at the operating point of §3.5 (`Capture` 70%,
  boost 0, speaking close) takes the corpus to eleven. Their PERs at the old
  point were 0,063, 0,059 and 0,177, so nothing suggests they will move the
  picture; they are owed because the criterion counts clean ones.
* **The `--errado` pass**, eight utterances, never recorded. It is what turns
  the seven "não" above from ambiguous into informative.

#### The ninth clean utterance, and two messages that cost takes

Three takes recorded at the operating point of §3.5, with the gain set to 70%
(+15,75 dB) and the boost off. One of them counts.

| regra | áudio (md5) | PER | o que saiu |
|---|---|---|---|
| `pron.vowel-length` | `52561607` | **0,059** | **limpa.** `/ʊ/→/uː/` em *full*; os dois `/iː/` de *beach* e *people* certos |
| `pron.epenthesis-final` | `f27e7ed6` | 0,062 | aviso de borda — fala em 0,0 s. Não conta |
| `pron.ing-alveolar` | `4d742010` | — | **recusada:** contraste de fala 1,8× contra o mínimo de 4 |

**Nine clean utterances of the ten the criterion asks for**, median PER of the
clean set **0,125**, and still one curated error named — `sub /ð/→/d/`. The new
clean take is the second lowest of the nine, and it is a *rejection* of its own
curated hypothesis done properly: both `/iː/` came out long, so `sub /iː/→/ɪ/`
is absent, and the single disagreement is `/ʊ/→/uː/` in *full* — a vowel the
`--discriminate` floor of §2.2 already named as unreliable in this stack.

**The refusal is the fifth question doing its job on real speech, and it is the
calibration point §3.5 asked for.** That subsection left
`SPEECH_CONTRAST_MIN = 4` provisional, on two files of unknown content, and
said E9 would calibrate it with utterances of a known phrase — this is one:
1,8×, a known phrase, refused before the model loaded. What it does **not**
settle is whether that file was unreadable or merely muffled, and that cannot
be settled from inside: the way to find out is a re-take of the same phrase at
the same operating point, not a lower threshold.

**Two messages in the script asked for the opposite of what the diagnostic
wants, and both have cost takes.** They are the same defect in two places —
a number or an instruction that describes the file rather than the utterance,
which is the thread §3.5 has been pulling since the first recording.

* **`record` said "fale agora, sem esperar".** True when the trim took a fixed
  1,0 s; false since it became a decay search, which stops at the first block
  not dominated by DC — 40 ms on a clean start. Speaking instantly buys no
  audio and trips the head-cut warning, which excludes the take. §3.5 records
  this message eating a `/ð ə/` once; the second time it cost
  `pron.epenthesis-final` at PER 0,062, thrown out with the curated error
  absent and a `/ɡ/` deleted mid-phrase, nowhere near the cut. The prompt now
  asks for the instant of silence the diagnostic wants.
* **The headline median counted takes the criterion excludes.** The same run
  printed `PER mediano: 0,094` over ten measured rows while its nine clean ones
  sat at 0,125 — a number no table here could be built from. `limpas` had kept
  "measured" and "counts" apart since it existed; the median had not. It is now
  the median of the clean takes, and with nothing clean it says so instead of
  printing one.

**One clean utterance left**, and two phrases can supply it: `epenthesis-final`
needs only the beat of silence the prompt now asks for, and `ing-alveolar`
needs a quieter room — gain is not the lever, as §3.5 measured. The `--errado`
pass is still unrecorded, and is still what turns the eight "não" above from
ambiguous into informative.

#### The tenth and eleventh utterances, and the room that was the lever

Two runs at the operating point of §3.5 — capture at 70% (+15,75 dB), boost
off, the same gain the nine takes before these were recorded at. Three takes,
and two of them count.

| regra | áudio (md5) | PER | o que saiu |
|---|---|---|---|
| `pron.epenthesis-final` | `37fba6fa` | 0,312 | **limpa.** 630 ms de transiente cortados; `/ɡ/→/iː/` no lugar da epêntese prevista |
| `pron.ing-alveolar` | `27daf04f` | — | **recusada:** contraste de fala 1,6× contra o mínimo de 4 |
| `pron.ing-alveolar` | `365e3cfb` | 0,529 | **limpa**, mesma frase e mesmo ganho, sala mais silenciosa. `/ŋ/→/n/` nomeado, duas vezes |

**Eleven clean utterances against the ten the criterion asks for**, median PER
of the clean set **0,154**, and **two curated errors named** — `sub /ð/→/d/` on
`th-voiced` and `sub /ŋ/→/n/` on `ing-alveolar`.

**The room was the lever, and that is now measured instead of supposed.** The
refusal of `ing-alveolar` was left ambiguous on purpose: 1,8× did not say
whether the file was unreadable or merely muffled, and the subsection above
wrote that the way to find out was a re-take of the same phrase at the same
operating point, not a lower threshold. That is what happened, twice. The first
re-take fell to 1,6× and was refused again; the second, same gain and a quieter
room, passed. The threshold never moved, and the fifth question refused twice
what was not a measurement and let through what was.

**`/ŋ/→/n/` is the second gabarito named in human speech, and it came from the
noisiest clean utterance in the set.** Its 0,529 sits between the signal of
§2.3 (0,119) and the proxy of §2.4 (0,676), and the alignment around it is
noisy — `/ɹ/→/v/` in *reading*, `/æ/→/ɪ/` in *and*. The curated operation came
out anyway, on both `-ing` endings of the phrase. That is exactly what §3.2
asks for and nothing beyond it: the operation carrying a `rule_id` is what
reaches the student, the rest of the alignment is not.

**`epenthesis-final` did not confirm its own hypothesis, and that is not the
same as silence.** The curated hypothesis is `/ɡ/→/ɡi/` — the final plosive
followed by an `/i/`. What the recognizer heard in *big* was `/ɡ/→/iː/`: the
vowel is there, the plosive is gone. A neighbour of the hypothesis, and still a
"não" in the `previsto?` column, because the signature of §3.4 matches
operations and not neighbourhoods. The honest reading is §6's: without the
`--errado` pass there is no separating "the evaluator did not find it" from
"the error was not there", and on this utterance both readings stay open.

**The gate of §2.4 is answered, and the answer is 0,154.** The risk that opened
this phase — the real student landing near the proxy's 0,676 — did not happen:
eleven human utterances, measured through the diagnostic rather than taken on
faith, put the median at 0,154. That is 2,5× the floor of 0,061 and well below
the proxy, on the side of the scale where §3.3 can operate. **E9 asks for the
decision written with the number in hand, and the `--errado` pass is what is
missing to write it:** two gabaritos named out of eleven show the evaluator
names an error when the error is there, but it is the wrong pass that says
whether the nine remaining "não" are absent errors or a blind evaluator. That
is what is left of the deliverable.

#### The wrong pass, and the half of the base that stays silent

Eight takes with the curated error committed on purpose, at the same operating
point as the correct pass — capture at 70% (+15,75 dB), close to the
microphone, one phrase per take. Seven measured, one refused. The column that
decides here is `nomeou?`, not the PER: what the pass asks is whether the
evaluator names the curated substitution **when the substitution is really
there**.

| regra | áudio (md5) | PER certo | PER errado | nomeou? |
|---|---|---|---|---|
| `pron.th-unvoiced` | `89bc4375` | 0,053 | 0,105 | **sim** — `sub /θ/→/t/` |
| `pron.th-voiced` | `eaa0d509` | 0,125 | 0,438 | **sim** — `sub /ð/→/d/` |
| `pron.epenthesis-initial` | `454a1ff6` | 0,312 | 0,438 | não |
| `pron.epenthesis-final` | `dce38bbc` | 0,312 | 0,875 | não |
| `pron.vowel-length` | `e2841d24` | 0,059 | 0,706 | não |
| `pron.h-dropping` | `c3e16004` | 0,364 | 0,818 | **sim** — `del /h/` |
| `pron.ing-alveolar` | `690c26c7` | 0,529 | 0,235 | não |
| `pron.r-initial` | `4e427a30` | 0,000 | recusada | — |

**Três de sete nomeados.** Four of the seven measured takes carry warnings
(level or edge), so they are worth less as PER; the naming column does not
depend on the level in the same way, and it is the column this pass exists for.

**The ambiguity of §6 is resolved, and only for three rules.** The correct pass
left "previsto? não" summing two opposite things — the evaluator did not find
the error, and the error was not there. For `th-unvoiced`, `th-voiced` and
`h-dropping` the answer is now the first one: deliberately said *tink*, *de
udder brudder* and *e as a ouse*, the alignment carried exactly the operation
the rule describes. Their "não" in the correct pass means the speaker did not
make the error.

**For the other four the evaluator is blind, and that is the finding.**
`epenthesis-initial`, `epenthesis-final`, `vowel-length` and `ing-alveolar` did
**not** produce their curated operation even with the error committed on
purpose. It is not that nothing changed: `epenthesis-final` went from 0,312 to
0,875 and `vowel-length` from 0,059 to 0,706, so the recognizer heard something
very different. It just did not name it as the curated substitution. Under
§3.2 an operation without a `rule_id` never reaches the student, so these four
would be **silent in the product** no matter how plainly the student made the
error — a false negative by construction, not by threshold.

**This is the human half of the `--discriminate` guard, and it is stricter.**
§2.3 measured 6 of 8 named on synthetic speech, where the substitution is exact
and everything else is identical between the two sides of the pair. On a human
voice at the same operating point it is 3 of 7. The gap is the measurement
§2.5 warned about: synthetic agreement is a ceiling, not a forecast. §3.2 already
forbids an item entering `pronunciation.yaml` without passing `--discriminate`;
this pass says the gate has to be the human one, and that it currently rejects
half the curated set.

**One inversion worth recording rather than explaining.** `ing-alveolar`'s wrong
take scored **lower** than its correct one — 0,235 against 0,529 — and the
gabarito matched in the *correct* take and not in the wrong one. The likely
reading is that the 0,529 take was the noisy recording rather than the wrong
one being better speech; it is not settled here, and a re-take of both at the
same distance is what would settle it.

**`pron.r-initial` has no wrong-pass data.** It was refused for speech contrast,
the only one of the eight, so the rule with the best correct take (PER 0,000)
is the one with nothing to pair it against. It needs a re-take before any
decision mentions it.

**O que isto decide.** The two numbers E9 asked for both exist now: eleven
clean utterances with median PER **0,154**, and **3 of 7** curated errors named
when deliberately committed. Read together they do not support a clean "go" or
a clean "stop". They support a **narrower phase**: the evaluator is viable for
the rules it can name — the two `th` and `h-dropping` — and is not viable, in
this stack and for this speaker, for the four that stayed silent. The decision
line and the scope of E10 follow from that, and are the remaining item of E9.

#### The re-takes, and the third reading of "não"

The decision of §3.7 was written with three rules named, and three of those four
"blind" verdicts rested on takes the diagnostic had flagged as worth less. Five
phrases were re-recorded for that reason — the `--corpus --errado` skips what is
already clean, so it offered exactly the soft ones.

| regra | áudio (md5) | PER | o que saiu |
|---|---|---|---|
| `pron.th-unvoiced` | `ecd00a0c` | 0,053 | o erro **não foi cometido**: `/aɪ θ ɪ ŋ k/`, *think* e não *tink* |
| `pron.epenthesis-initial` | `54094864` | 0,438 | **`ins /ɪ/` nomeado** — `/f ɔːɹ ɪ s t ɹ ɛ s/`, a vogal epentética na transcrição |
| `pron.epenthesis-final` | `4eb5fcb4` | — | recusada: contraste de fala 2,7× |
| `pron.vowel-length` | `74426f7a` | — | recusada: transiente de abertura, degrau de DC nos primeiros 0,3 s |
| `pron.r-initial` | `7354ffc3` | 0,615 | **`sub /ɹ/→/h/` nomeado** — `/ɛ h æ d/` onde se esperava `/ð ə ɹ ɛ d/` |

**São cinco regras nomeadas, não três.** `epenthesis-initial` saiu do silêncio e
`pron.r-initial`, que não tinha par nenhum, entrou nomeado. Somadas às três de
antes — `th-unvoiced`, `th-voiced`, `h-dropping` — a base admissível pela §3.7
dobra menos uma: **5 de 8**.

**A sexta não conta, e por que ela não conta importa mais que o número.** Uma
varredura por *todas* as tomadas acha `ins /i/` em `epenthesis-final`, na tomada
`d2d3b42e` — a de PER **2,500**, gravada na sala vazia, cujo `ouvido` é
`/ɑ ʃ k a m t o t o t o t o i a k u.../`: 40 erros em 16 fonemas. Uma inserção
que casa naquilo é o alinhamento achando coincidência, não o avaliador nomeando
erro. Contá-la seria exatamente o "número que parece medição e não é" que a §3.3
recusa. As duas tomadas honestas da regra foram uma recusa e um 0,875 não
nomeado.

**Três das cinco nomeações vêm de tomada com aviso**, as duas novas entre elas.
O aviso degrada o PER, não o fato de a operação ter sido encontrada — mas uma
confirmação limpa de `epenthesis-initial` e de `r-initial` é o que faria o 5 de 8
ficar sólido em vez de apenas melhor fundamentado que o 3 de 7.

**E a passada ganhou uma terceira leitura de "não", que ela não foi feita para
separar.** A tomada nova de `th-unvoiced` saiu com PER 0,053 e `/θ/` no lugar: o
falante disse *think*, certo, numa tomada cujo objetivo era errar. O instrumento
não tem como saber: ele compara com os fonemas da frase **correta**, então uma
tomada "errada" pronunciada certa é indistinguível de uma tomada certa. Logo um
"não" pode significar três coisas, e não duas — o avaliador é cego, o erro não
estava lá, **ou o erro deliberado não foi deliberado**. Quem for crescer a base
pela §3.7 precisa olhar o `ouvido` da tomada antes de aceitar um "não", e não só
a coluna.

*As duas confirmações que esta subseção pede continuam pendentes, mas o 5 de 8
não: a subseção seguinte aplica a §3.4, mede quatro tomadas mais e leva a
contagem a 8 de 8 — sem mover nenhuma coluna de limpas.*

#### The class the signature declares, and the eighth rule

The subsection above closed at 5 of 8, and said what would make that number
solid: a clean confirmation of the two new namings. What arrived instead was
cheaper and moved the count further — **§3.4 applied, and four more takes.**

**Two of the four "blind" verdicts were the gabarito's symbol, not the
evaluator's hearing.** The initial-epenthesis signature asks for `ins /ɪ/`
because that is how espeak renders the swap; `454a1ff6` was heard as `ins /i/`,
arguably the better transcription of what a Brazilian inserts. The long vowel is
the same case in substitution: signature `sub /iː/→/ɪ/`, heard `sub /iː/→/i/`, in
both *beach* and *people*. §3.4 had already declared the signature to be a
**set** of acceptable operations (`+/ɪ/`, `+/ɐ/`, `+/i/`, `+/ə/`) and nothing was
reading it as one. The synthetic pair of §2.3 missed the same item the same way,
hearing `+/ɐ/`, so one class fixes both sources.

`widen()` applies it, and **only on the heard side**: the origin keeps coming
from the G2P over the curated phrase, and the positions and operation types keep
coming out of the two transcriptions. A deletion has no heard side and passes
through intact. Three properties, and the third authorizes the rest:

* **It does not touch what is correct.** Only what counts as the *curated error*
  moves; the alignment, the PER and what counts as correct English stay where
  they were. The equivalence sets of this same §3.5 (`/ɔː/ ≡ /ɑː/`) are a
  different table, for a different question, and they are E10's.
* **It stays out of `run_pairs`.** §2's tables were measured with the strict
  intersection; widening it there would redefine them in silence.
* **It does not fire on correct speech.** Across the **30** correct-pass takes
  that carry a signature — eleven of them clean — **not one** matched by class.
  That was the obvious risk (`/ɐ/` is one of the three phonemes the floor of
  §2.2 is made of) and it did not materialize in the only speech there is.

**`--rescore` answers the column with no microphone and no model**, which is why
none of this cost anyone a word. The signature is not a measurement — it is the
curated phrase through the G2P with the swap applied — and `esperado` and
`ouvido` are in every measured report. It fills the signature in even on
**warned** reports, which `--corpus` skips on purpose: the warning excludes the
utterance from the count, not the transcription it carries. Eighteen reports were
filled in; a refusal gets none, because there is no alignment to match against.

Then four more takes, all with the level warning, none clean:

| regra | áudio (md5) | PER | o que saiu |
|---|---|---|---|
| `pron.epenthesis-final` | `aa8e741a` | 0,250 | `+/i/` depois do `/ɡ/` de *big* **e** `+/ɪ/` depois do `/b/` de *club* |
| `pron.epenthesis-initial` | `1d64e713` | 0,250 | `+/ɪ/` antes de `/sp/` **e** de `/st/` — o símbolo exato, nas duas posições |
| `pron.r-initial` | `fc11b48b` | 0,308 | `/ɹ/→/h/` **nomeado** outra vez, e a melhor tomada que esta frase deu |
| `pron.vowel-length` | `6cf76eb7` | 1,000 | casou `/iː/→/i/` **por classe**, num alinhamento onde nada alinhou |

**`pron.epenthesis-final` is named, and it was the rule this pass had written off
twice.** The subsection above is right that `ins /i/` inside the 2,500 salad is
the alignment finding a coincidence. This take is the opposite: 0,250, the best
that phrase has produced in the deliberate pass, with the inserted vowel
appearing **twice** — `+/i/` after the `/ɡ/` of *big*, the signature's own
symbol, and `+/ɪ/` after the `/b/` of *club*, the class. One alignment naming one
rule by symbol and by class, in two words, is §3.4's argument without the
argument.

**And the `/iː/→/i/` at PER 1,000 is the second coincidence, which is the same
lesson the salad taught.** The transcription of *The beach was full of people*
reads `d ə b i tʃ i ɪ s f uː l o ɡ w a s f l o f i p uː`: the class operation is
in there, and so is nearly every other operation, because nothing aligned. It is
discarded for the reason the subsection above discards the salad — and
`vowel-length`'s honest naming is the 0,706 take, by class, not this one.

**A third coincidence sits in the *correct* pass, and it names the shape of
signature that is vulnerable.** `13008743` transcribes as six phonemes against
eleven — the take ends after *He has a* — and `del /h/` matched because the last
two `/h/` are not there to be dropped. A take that stops early deletes everything
after it, so **a deletion signature is the one shape a truncated take satisfies
for free**, and `h-dropping` is the only curated rule whose signature is a
deletion. It needs no new gate: the diagnostic had already said so, in that
take's second warning — *"a fala vai até 6.9s de 7.0s, encostada no fim do
arquivo"*. Three coincidences, three shapes, one rule: **the fraction that counts
is the one over the clean takes**, which is why the corpus table now prints it.

**One caveat for E11 that is about attribution rather than hearing.** In
`454a1ff6` the epenthetic vowel is audible before **both** clusters — the
transcription reads `i s p ɔ ɾ t z ɡ uː d f ɔ i s t ɹ ɛ s` — and the aligner
resolved only the first as `ins /i/`, booking the second as `sub /ɔːɹ/→/i/` over
the vowel of *for* next door. The same phrase in `1d64e713` resolved both as
insertions. What reaches the student is the **operation**, not the audible vowel,
so an epenthesis charged to a neighbour is one the evaluator cannot report. It
costs nothing while one match names the rule; it is the reason a count of
occurrences is the wrong thing for E12 to show.

**So all eight rules are named, and the number that matters is now the shape of
the ladder rather than its height:**

| how the rule was named | regras |
|---|---|
| clean deliberate take | `th-voiced`, `h-dropping` |
| clean take, but in the **correct** pass — error unmonitored, never produced on demand | `ing-alveolar` |
| warned deliberate take, **exact** symbol | `th-unvoiced`, `epenthesis-initial`, `epenthesis-final`, `r-initial` |
| warned deliberate take, **by class** only | `vowel-length` |

`ing-alveolar` is the row the subsection above left unresolved, and the ladder
resolves it in the direction that was already recorded there: its signature
matched the **correct** take and not the deliberate one, because the deliberate
one came out `/ŋ/`. The rule is named — twice in one phrase, on a clean take —
and what is missing is a deliberate take in which the error is actually
committed. That is the third reading of "não" from the subsection above,
confirmed on a second rule.

The corpus behind the ladder: **64 reports, 46 with a signature, 18 refused.**
Correct pass 37 takes — 11 clean, 19 warned, 7 refused, median of the clean
**0,154**. Deliberate pass 27 takes — 3 clean, 13 warned, 11 refused, median of
the clean **0,438**. **No round moved a clean column**, which is the reason the
decision below reads them: the warned rows keep growing, `--rescore` re-answers
every column at any moment with no microphone and no model, and the fractions the
criterion reads have stood at 2 of 11 and 2 of 3 since the eleventh clean
utterance.

### 3.6 A iGPU da §7 é o que paga a pronúncia

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

### 3.7 A decisão da E9: segue, e mais estreita

A entrega pede a decisão escrita com o número na mão. São dois números, e eles
não dizem a mesma coisa.

**O primeiro é o que abriu a fase, e ele passa.** Onze elocuções humanas limpas,
mediana de PER **0,154** — 2,5× o piso de 0,061 e bem longe do proxy de 0,676
da §2.4, onde o alinhamento desaba e nada é *o* erro. O risco da §6 que podia
cancelar as outras entregas não se realizou: esta pilha ouve este falante. A
alternativa que a §5 previa para o caso ruim — `small` em vez de `large`,
*forced alignment* em vez de reconhecimento livre, ou pronúncia fora do escopo —
não precisa ser acionada por causa do PER.

**O segundo é o que decide o escopo, e ele mudou três vezes sem mudar de
direção.** Esta subseção nasceu com **3 de 7**, foi para **5 de 8** com as
regravações, e está em **8 de 8** desde que a §3.4 passou a ser aplicada e
quatro tomadas novas entraram. As duas correções que fizeram a diferença não são
do avaliador: `vowel-length` e a primeira tomada de `epenthesis-initial` eram
**o símbolo do gabarito**, não a audição — o reconhecedor ouviu `+/i/` e
`/iː/→/i/` onde o espeak prevê `+/ɪ/` e `/iː/→/ɪ/`, e a §3.4 já declarava a
assinatura como *conjunto*. O `epenthesis-final` entrou pela última rodada, com
`+/i/` e `+/ɪ/` numa tomada de 0,250 — não pela de 2,500, que a §3.5 descarta
com razão.

**Mas a altura do número vale menos que a forma da escada, e é ela que decide o
escopo.** Duas regras são nomeadas em tomada limpa da passada errada
(`th-voiced`, `h-dropping`); uma só na passada **correta**, porque o erro
deliberado nunca saiu (`ing-alveolar`); quatro em tomada com aviso e símbolo
exato (`th-unvoiced`, `epenthesis-initial`, `epenthesis-final`, `r-initial`); e
uma só por classe, e só com aviso (`vowel-length`). O aviso degrada o PER, não o
fato de a operação ter sido achada — mas **cinco das oito regras só foram
nomeadas em tomada que o critério não conta**, e é isso, não o 8 de 8, que a
E10 tem de resolver antes de virar item.

**A decisão: seguir, com a base admitida pelo teste humano e não pelo
sintético.** Em concreto:

1. **A E10 muda de critério de entrada.** A §3.2 exigia passar no
   `--discriminate`; passa a exigir também **ser nomeada por voz humana, em
   tomada que o diagnóstico não tenha avisado**. O sintético continua valendo
   como primeiro filtro barato — ele reprova cedo o que nem no caso ideal
   funciona —, mas quem decide é o humano, porque é ele que mede o que o aluno
   vai receber. A §2.3 deu 6 de 8 em fala sintética; em fala humana são 8 de 8
   nomeadas e **3 de 8 em tomada limpa**, e é a segunda fração que é o critério.
2. **A E10 nasce pequena, e o plano para de prometer ~15 itens.** Oito regras
   são candidatas e **três** passam hoje pelo critério do item 1 — `th-voiced`,
   `h-dropping` e `ing-alveolar`, esta última pela passada correta. As outras
   cinco pedem uma tomada limpa, que é gravação e não pilha: a §3.5 mede o
   rendimento em 14 utilizáveis de 64, e dez das onze recusas da passada errada
   são contraste de fala. A base admitida é a que passar no teste, e o número
   sai da medição, não do alvo.
3. **Nenhuma regra fica documentada como cega, e é uma mudança desta revisão.**
   Com a §3.4 aplicada e a última rodada medida, não sobrou regra sem nomeação —
   o que sobrou é nomeação que o critério não conta. A diferença importa porque
   ponto cego é veredito sobre a pilha e aviso é veredito sobre a gravação: o
   primeiro não se conserta gravando de novo, e o segundo é exatamente o que uma
   sala silenciosa conserta. A §6 mantém a linha do `pron.ing-alveolar`, que
   continua sendo o item cujo ponto cego era da **síntese**, desempatado pela
   fonte humana.
4. **A E11 e a E12 seguem como escritas.** Nada nelas depende do tamanho da
   base: o contrato é devolver só erro com `rule_id` e calar acima do portão, e
   isso é igual com três regras ou com quinze. Duas ressalvas da §3.5 são da
   E11, não da base: assinatura de **deleção** casa de graça numa tomada
   truncada, e epêntese cobrada da vogal vizinha é epêntese que o avaliador não
   consegue relatar — daí o retorno citar a regra e nunca uma contagem de
   ocorrências.
5. **Um "não" não basta para condenar uma regra, e um "sim" não basta para
   admitir uma.** O "não" tem três leituras e não duas — avaliador cego, erro
   ausente, ou erro deliberado que não foi cometido, o que aconteceu em
   `th-unvoiced` e em `ing-alveolar`. O "sim" tem duas: operação achada, ou
   coincidência de alinhamento, que é o que as tomadas de PER 2,500 e 1,000
   produziram e o que uma assinatura de deleção produz numa tomada cortada. Nos
   dois sentidos a resposta é a mesma: olhe o `ouvido` da tomada, não a coluna.

**O que reverteria esta decisão.** Se o conjunto admitido não passar de quatro
ou cinco regras depois das confirmações limpas e de a E10 testar os itens
novos — o caso provável se a sala não melhorar, porque hoje são três —, o
produto de pronúncia é fino demais para o custo dos 318 MB e do
terceiro modelo nativo no processo, e a §4 volta a valer com "pronúncia fora do
escopo". O número que decide isso é o mesmo desta subseção, medido do mesmo
jeito; não é preciso instrumento novo.

**O que esta decisão não autoriza.** Ela não calibra teto de PER — a §3.3
continua valendo, e o portão continua sendo por operação de alinhamento. E ela
vale para **um** falante: a §2.5 já dizia que uma voz não é a população, e a
primeira coisa que uma segunda voz pode derrubar é justamente a escada — tanto o
8 de 8 nomeadas quanto o 3 de 8 limpas.

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
| **E9** ✅ | **A dívida da §2.3 paga** — gravações de falante brasileiro B1+ lendo as frases curadas (`--record`), PER medido, portão da §3.3 decidido | Há ≥10 elocuções humanas **que passam pelo diagnóstico de sinal da §3.5**; o PER delas e a fração de gabaritos nomeados estão registrados aqui; **e a decisão de seguir ou parar está escrita com o número na mão**. Não há teto de PER a calibrar: §3.5 mede as duas faixas se sobrepondo em fala humana, e o portão é por operação de alinhamento. **Feito.** Onze elocuções limpas, mediana 0,154; passada `--errado` com a operação curada nomeada em **8 de 8** regras e em **3 de 8** por tomada limpa (§3.5). A decisão está na §3.7: **segue, com a base admitida pelo teste humano** — a E10 muda de critério de entrada, exige nomeação em tomada **limpa**, e nasce com três regras enquanto as outras cinco esperam gravação, não pilha |
| **E10** | **`pronunciation.yaml`** — a base de fala que a E3 adiou; os itens que o teste humano admitir, cada um com assinatura de operações aceitáveis (§3.4) | Os itens carregam e validam pelo schema; cada um tem teste; **nenhum item entra sem passar no `--discriminate` (§3.2) e sem ser nomeado por voz humana em tomada que o diagnóstico não avisou** (§3.7). O número de itens sai da medição, não do alvo — e `pron.ing-alveolar` entra pela passada correta, com a razão registrada, porque ali o ponto cego era da síntese (§3.5) |
| **E11** | **O reconhecedor e o avaliador** — `voice/phonemes.py` atrás de contrato como o `whisper.py`, `teaching/pronunciation.py` casando alinhamento com regra | `POST /api/pronunciation` devolve só erros com `rule_id`, cala acima do portão, e **`teaching/turn.py` não muda uma linha** |
| **E12** | **Pronúncia no produto** — retorno na UI depois do turno, `pronunciation_signal` persistido, `/progress` agregando por fonema | O aluno fala, conversa normalmente, e o retorno de pronúncia aparece sem atrasar a fala do professor (§3.1) |
| **E13** ✅ | **Repetição espaçada** — `teaching/schedule.py`, e o drill escolhendo a regra por agendamento | **Feito.** A fila é ordenada pela agenda (`Scheduler.queue`), três acertos seguidos graduam a regra e a tiram da fila por sete dias, e `test_a_escada_mede_os_intervalos_em_dias` mede o intervalo que o aluno recebe. Ver §7 |

**A E9 é portão, não etapa.** É a única entrega da fase cujo resultado pode
cancelar as outras, e ela vem primeiro por isso. A §2.4 colocou o aluno real
entre 0,12 e 0,68 sem dizer onde; se ele estiver perto de 0,68, o avaliador
não é viável nesta pilha e a fase vira outra coisa (um `small` em vez do
`large`, *forced alignment* com o texto conhecido em vez de reconhecimento
livre, ou pronúncia fora do escopo). Escrever E10–E12 antes disso é escrever
código que a medição pode jogar fora — exatamente o que a abertura por medição
existe para evitar.

**E a E9 respondeu: 0,154, não 0,676.** A fase segue como escrita, com o
`large`; o que ela estreitou é a base da E10, não a pilha. A decisão, com as
condições que a reverteriam, está na §3.7.

**O IPA na tela entrou fora da tabela**, sem número de entrega: ele não estava
no plano, e numerá-lo aqui tomaria um rótulo que o autor da fase pode querer
para outra coisa. É `teaching/phonetics.py`, e o que ele faz é escrever a
transcrição fonética embaixo de toda fala — a do aluno e a do professor.

Ele **não depende da E9**, e a distinção é a mesma que o portão protege: a linha
mostra como a palavra **escrita** se pronuncia, que é tabela; a E11 julgaria
como o aluno a **falou**, que é medição sobre áudio. A primeira não corre
nenhum dos riscos da segunda. O G2P é o mesmo que a §2 usou para o gabarito e o
mesmo que o Kokoro usa para falar — daí a propriedade que nenhum outro daria de
graça: o símbolo que o aluno lê é o som que ele ouve. O `phonemizer` e o
`espeakng_loader` saíram do extra `voice` e viraram dependência base, porque
nada ali é áudio. E o partial `partials/app/phonetics.md` mantém explícito para
o modelo que a linha é do texto, não uma leitura do aluno.

**A fila de curadoria também entrou fora da tabela**, e pelo mesmo motivo: não
estava no plano, e numerá-la tomaria um rótulo que o autor da fase pode querer
para outra coisa. É a migração `005_unverified.sql`, a tabela
`unverified_finding` e o `scripts/curation_queue.py`, e o que ela faz é
**guardar a correção que o verificador da E4 barrou** em vez de só contá-la. Até
aqui `Analysis.unverified` dizia *quantos* achados foram descartados por não
citarem regra de `data/contrastive/`, e o trecho ia para `log.debug`: a medida
de quanto a base não cobre existia, a lista não. Cada linha descartada é um erro
no texto real de um aluno, já localizado no trecho — que é a parte cara de
escrever item novo.

Ela **não depende da E9**, e a razão é a do IPA uma camada acima: o que entra
ali é achado do analisador sobre **texto**, não alinhamento sobre áudio. Nada
do que a fila guarda chega ao aluno — é exatamente a gramática que o verificador
acabou de recusar, e `Analysis.as_dict` não a expõe, pelo princípio que a §3.2
aplica à pronúncia: material de curadoria não é material de aula. Por isso quem
lê a fila é um script de linha de comando, e não a tela de progresso.

**E ela nasceu com um ponto cego, que é a assimetria da §6 deslocada um nível.**
Com `require_curated_rule` ligado, a análise não roda quando a base não levanta
candidato nenhum, então toda linha da fila vem de um turno em que a base **já
acertou alguma outra regra**: o buraco mais fundo — o erro que ela não enxerga
de jeito nenhum — é invisível por construção. E não dá para fechá-lo com dado,
porque sem chamar o modelo "inglês correto" e "erro que a base não vê" produzem
a mesma ausência de candidato, como "não errou" e "não tentou" produzem a mesma
ausência de correção no drill. `TC_REQUIRE_CURATED_RULE=false` inverte a sonda,
ao custo da latência que o default protege — rodada de curadoria deliberada, não
uso normal.

A **E13 não depende da E9** e pode correr em paralelo: ela consome
`correction`, que existe desde a primeira migração. Foi o que aconteceu — ela
entrou com a E9 ainda aberta, e a §7 registra o que isso custou: a fonte
`pronunciation_signal` que a §4 previa não existe ainda, então a agenda nasceu
com uma fonte só.

---

## 6. Riscos conhecidos desta fase

| Risco | Como lidamos |
|---|---|
| **O aluno real pode cair perto do 0,676 do proxy** | ~~É o portão da E9~~ **Não caiu:** mediana 0,154 em onze elocuções limpas, e a decisão está escrita (§3.7). O risco volta inteiro para cada novo falante, e é a única condição da §3.7 que reabre o portão |
| **Áudio sintético calibra o portão errado** | Não calibra: a §3.3 proíbe calibrar o teto em sintético, de propósito. O valor sai da E9 ou não sai |
| **Erro real de brasileiro reportado como erro que ele não cometeu** | Duas camadas: só operação com `rule_id` chega ao aluno (§3.2), e acima do portão o avaliador cala (§3.3) |
| **Precisão inventada no retorno** | O retorno fala em classe de erro, não em símbolo de vogal (§3.4) |
| **`pron.ing-alveolar` não é o único ponto cego** | Todo item passa pelo `--discriminate` antes de entrar na base (§3.2), e o que não passa fica documentado como ponto cego em vez de virar falso negativo silencioso. E o ponto cego pode ser da **síntese**: este item é nomeado em fala humana e não no par do Kokoro (§3.5), então a fonte humana desempata a sintética |
| **Três modelos nativos no mesmo processo** | O lock por motor da Fase 1.5 continua, e a §3.6 explica por que a CPU agora caberia: o LLM saiu dela |
| **Gravação quebrada reportada como medição** | Cinco perguntas de sinal antes de o modelo carregar, e recusa com diagnóstico em vez de PER (§3.5). A primeira gravação da E9 caiu exatamente nisso |
| **318 MB a mais de peso** | Extra desligado por default, como a voz. Quem não quer pronúncia não baixa nada |
| **Acerto no drill não distingue "não errou" de "não tentou"** (E13) | Dívida aberta, não resolvida. O aluno que responde *"yes"* três vezes gradua a regra. Não há guarda determinística hoje — o analisador devolve erros, não "usou a estrutura" — e um piso inventado (contagem de palavras, lista de respostas triviais) seria número sem medição. O que existe é o dado: `rule_review.turn_id` aponta para o turno, então dá para medir quantos acertos registrados não eram tentativa. Pinado em `test_uma_resposta_vazia_de_conteudo_conta_como_acerto` |
| **A escada de intervalos não é medida** (E13) | Admitido no código. Não havia histórico de revisão para calibrar escada nenhuma antes de a E13 existir, então os valores são Leitner clássico. É por isso que `rule_review` guarda `interval_days` na linha em vez de confiar na constante: quando houver histórico, a escada se recalibra contra o que o aluno de fato recebeu |

---

## 7. A E13, e o que ela trocou

A entrega está em `teaching/schedule.py`, `store/migrations/004_schedule.sql` e
`tests/test_schedule.py`. Três decisões merecem registro, porque nenhuma das
três é óbvia no arquivo que a implementa.

### 7.1 O sinal que faltava não era "errou mais", era "acertou desde então"

Até a E6 a fila do drill era um ranking de frequência. O problema não era de
implementação: **frequência é um número que só sobe.** Ela sabe o que o aluno
errou e não sabe o que ele aprendeu, então a regra mais corrigida do histórico
continua sendo a mais corrigida depois de o aluno acertar dez vezes, e o drill
nunca a solta. Nenhum ajuste de ordenação resolve isso, porque o dado que
faltava não estava no banco.

A troca está medida num teste e não numa frase: em
`test_a_fila_e_a_agenda_e_nao_a_contagem`, a regra com cinco erros está graduada
e a com um erro nunca foi treinada. A contagem devolveria a primeira; a agenda
devolve a segunda. Com a ordenação revertida para `-times`, **quatro** testes
ficam vermelhos — incluindo o do rodízio da E6, que já existia.

### 7.2 A assimetria do sinal é a decisão de desenho da entrega

Não existe sinal de "o aluno usou a estrutura e acertou": o analisador devolve
erros. Então acerto aqui é *ausência de erro num lugar onde o erro teria
aparecido*, e esse lugar é o drill — o exercício pediu a estrutura e esperou a
resposta, logo a oportunidade foi construída. Fora do drill, turno limpo não é
nada: pode ser que o aluno não tenha usado a estrutura.

| | conta como |
|---|---|
| drill, sobre a regra em treino, sem correção | **acerto** |
| qualquer modo, correção pela regra | **erro** |
| conversa, sem correção | **nada** |
| **análise que não rodou** | **nada** |

A última linha é a guarda que mais importa, e é a §3.3 outra vez: acerto é a
*ausência* de correção, então análise que não rodou produz a mesma ausência e
graduaria a regra por falha de infraestrutura. Quando o avaliador não está
entendendo, calar é a resposta.

O que essa assimetria **não** resolve está na §6: dentro do drill, "não errou" e
"não tentou" continuam sendo a mesma observação.

### 7.3 O rodízio não foi removido — foi rebaixado, e o intervalo o substituiu

O `DRILL_ROTATION` da E6 existia para a fila não travar na regra mais errada. A
agenda torna isso desnecessário por um caminho mais barato: o degrau zero da
escada vale **meio dia**, não zero. Com zero, a regra recém-errada vence na hora
e volta a ser a mais vencida na sessão seguinte, para sempre — era exatamente o
que o rodízio remendava. Com meio dia ela volta amanhã e outra vencida é
treinada hoje. Interleaving passa a ser consequência da agenda, não exceção
colada em cima dela.

O rodízio sobrevive porque cobre o único caso que a agenda não vê: **regra
escolhida numa sessão que terminou sem resposta** não gera revisão, então sem
ele seria escolhida outra vez. Virou termo de ordenação em vez de filtro, o que
também apagou o `fresh or counts` da E6 — a degradação agora é a própria chave.

### 7.4 Uma fonte, não duas

A §4 especifica `teaching/schedule.py` como repetição espaçada sobre
`correction` **+** `pronunciation_signal`. A segunda fonte é da E12, atrás do
portão da E9, e não existe. A §5 já havia reconciliado isso ao dizer que a E13
não depende da E9 *porque* consome `correction` — e é o que foi feito: a agenda
nasceu com uma fonte. `Scheduler.review` recebe `rule_id`, acerto e origem, e
não sabe de onde veio a evidência, então a E12 pendura `pronunciation_signal`
acrescentando um valor a `rule_review.source` e nada mais.

---

## 8. O que vem depois

- **Turn-taking sem botão**, com o Silero que já está no processo (§6 do plano
  1.5, inalterado).
- **Prosódia e ritmo** — o que o alinhamento de fonema não vê.
- **A dívida da análise (~10 s por turno com erro)**, que a §7.7 da Fase 1.5
  registrou como a maior fatia isolada do turno e que a iGPU quase não toca. O
  caminho continua sendo um modelo melhor só para ela.
- **Mobile, multiusuário, login, instalador** — depois da Fase 2, como a §4 da
  Fase 1 já dizia.
