# Medição: a aula de palavra

A única menção a vocabulário nos planos é a §4.3 da `analise-inicial.md` —
*repetição espaçada para vocabulário* —, e é o único item daquela lista de
quatro que nunca foi entregue. Os outros três estão de pé: perfil com nível em
SQLite, modos explícitos, base contrastiva.

Esta medição abre o assunto do jeito que a 1.5 e a 2 abriram os seus: medindo
antes de desenhar. A pergunta não é *o professor sabe a palavra*. É **quem
responde** quando o aluno pergunta por uma — porque o app tem base curada,
agenda de revisão e um G2P determinístico, e a medição mostra que uma pergunta
de vocabulário não consulta nenhum dos três.

Roda com `uv run python scripts/measure_vocab.py` (`--no-model` para a metade
determinística, `--runs 3` para amostrar a variância). A pontuação tem suíte
própria em `tests/test_measure_vocab.py`, e o motivo está na §5.

## 1. O método

Dez palavras, uma pergunta cada — `What does "<palavra>" mean?` —, em conversa
livre, com **sessão nova por palavra**: o histórico de uma mudaria a resposta
da seguinte, e o que se mede é a palavra.

O corte da lista é deliberado: **cinco palavras têm item curado na base** e
cinco não. Sem esse corte, uma coluna vazia na base seria consequência da
escolha das palavras, e não da base ser cega a uma pergunta.

Modelo: `qwen2.5:3b-instruct-q4_K_M`. O `qwen3:4b` configurado não está baixado
nesta máquina, e o 3B é a classe abaixo do que o produto entrega — a §5 diz o
que isso limita. Três varreduras, mais duas rodadas anteriores: **49 respostas**.

## 2. A metade determinística: a base não é consultada

Sem modelo nenhum, em milissegundos:

| palavra | itens curados levantados | no corpus |
|---|---|---|
| fodder | — | 0/492 |
| actually | — | 3/492 |
| pretend | — | 0/492 |
| eventually | — | 0/492 |
| comprehensive | — | 0/492 |
| realize | — | 2/492 |
| thorough | — | 0/492 |
| schedule | — | 0/492 |
| cannon fodder | — | 0/492 |
| house | — | 0/492 |

**43 itens curados, zero levantados em 10 de 10 palavras** — inclusive nas
cinco que têm item próprio. A base casa `patterns` contra o texto do aluno, e
uma pergunta não traz erro nenhum para casar. Com `require_curated_rule`
ligado a análise nem roda. Então a resposta sai sem nada curado dentro, sempre.

A coluna do corpus é o outro lado: `fodder` não aparece uma vez nos 492
enunciados do episódio importado. Nem `house`, nem `apartment` — eles dizem
*place*.

O que essa coluna permite concluir é mais estreito do que parece, e foi um
teste que mostrou: a primeira versão da suíte exigia colocação de todo item
ausente do corpus, e `eventually`, `pretend` e `comprehensive` quebraram.
Estão certos — são palavras B1/B2 comuns, e faltar num episódio de 22 minutos
é tamanho de amostra, não raridade. **492 enunciados são evidência de que uma
palavra rara é rara; não são evidência sobre uma palavra comum.** Para
`fodder`, que é C1, a ausência soma com o nível e com o sentido figurado: o
que serve ao aluno é `fodder for`, não celeiro.

## 3. O que o modelo respondeu

Contado por máquina, sobre as 49 respostas:

| sinal | resultado |
|---|---|
| mencionou a armadilha PT das 5 palavras curadas | **0 de 25 tentativas** |
| mencionou pronúncia, de qualquer forma | **0 de 49** |
| digitou IPA na prosa | **0 de 49** |
| usou português | **0 de 49** |
| alegou ouvir o aluno | **0 de 49** |
| rotulou registro, frequência ou nível | **1 de 49** |
| exemplos inventados por resposta | 1 a 5 |
| latência da resposta, com modelo quente | 5,0 a 18,2 s |

As três últimas guardas do produto seguram: a persona não escorrega para o
português, a parcial de capacidades impede a alegação de escuta, e a parcial de
fonética parece ser o motivo de ele não digitar transcrição. **O que falha é o
conteúdo.**

`fodder`, quatro tentativas completas da mesma pergunta:

| | o que disse do sentido figurado | veredito |
|---|---|---|
| 1 | *material for discussion or criticism* | certo |
| 2 | *something used to make someone angry or upset*, com o exemplo *"That argument was all about fodder."* | **errado**, e o exemplo não é inglês |
| 3 | *material used for argument or discussion, especially when it's meant to cause disagreement* | derivando para o errado |
| 4 | só o sentido agrícola; *a common term in farming or animal care* | o sentido vivo omitido |

`cannon fodder`, cinco tentativas: duas certas (*expendable*), duas erradas
(*soldiers used as targets to practice shooting*) e uma metade. **Cara ou
coroa, com o mesmo prompt e o mesmo modelo.**

E o controle: `house` — palavra sem nada a ensinar — inventou conversa nas
cinco tentativas (*Do you have any questions about what a house is used for?*).
Nenhuma disse que não havia o que ensinar.

## 4. O que a medição decide no desenho

1. **A aula não pode sair do modelo.** Uma em quatro no `fodder`, duas em cinco
   no `cannon fodder`. E aqui não há rede: o verificador da E4 só aceita achado
   que cite regra curada, e não existe regra curada para o sentido de uma
   palavra. O item tem que ser curado, como a gramática é.
2. **O item de vocabulário não cabe no `ContrastiveItem`.** O schema exige
   `wrong` e `right` com `minItems: 1`, e o padrão do `id` admite quatro
   categorias. Uma palavra que o aluno não sabe **não tem `wrong`** — ele não
   errou, ele não tem a palavra. Arquivo próprio e schema próprio, não uma
   quinta categoria.
3. **O IPA é a única metade pronta.** Sai do espeak, já vale para a bolha do
   professor (`Phonemized.role`), e 0 de 49 respostas tentaram competir com ele.
4. **O gatilho não pode ser erro.** 43 itens, zero levantados. São dois
   gatilhos: o aluno perguntar — que é onde as respostas erradas acontecem hoje
   — e a agenda oferecer, que é a máquina da E13 com a chave generalizada.
5. **Os exemplos saem do corpus, derivados e não copiados.** Dos inventados,
   dois não são inglês: *"That argument was all about fodder."* e *"I will call
   you eventually when I finish my work."*
6. **"Nada a ensinar" tem que ser resposta possível.** O controle do `house`
   mostra que hoje não é.

## 5. O que esta medição não diz

- **O modelo é uma classe abaixo do que o produto entrega.** O `qwen3:4b` não
  está baixado aqui; o 7B e o 8B presentes não foram sondados. E não se pode
  supor que um modelo maior resolve: a §5 do plano da Fase 1 registra um 7B
  obedecendo *demais* ao prompt do drill. Nos dois sentidos, não medido.
- **Uma formulação só.** Não foram perguntados *how do I pronounce X*, *how do
  I use X in a sentence*, nem o caso em que o aluno **usa** a palavra errado em
  vez de perguntar. A primeira importa em especial: foi numa pergunta de
  pronúncia que o Charles ofereceu *pat* contra *bat* como contraste de vogal,
  então 0 de 49 transcrições inventadas **não** é evidência de que ele nunca
  invente.
- **O veredito de sentido é meu, não da máquina.** As contagens da §3 são
  determinísticas e testadas; a coluna *certo/errado* da tabela do `fodder` é
  julgamento. Não há gabarito determinístico para uma definição — é justamente
  por isso que o item precisa ser curado.
- **As dez palavras são escolha minha**, cinco curadas de propósito. Não é
  amostra do que um aluno pergunta.
- **A primeira rodada mediu um timeout, não uma palavra.** O script não aquecia,
  e a carga do modelo dentro do turno estourou os 120 s de `request_timeout_s`:
  `fodder` saiu com 0 palavras e a tabela não disse por quê. O script agora
  aquece pelo `stable_prefix`, como o `app.py`, e captura `Failed`. Não é bug
  do produto, que aquece na subida — mas o aquecimento **engole erro por
  política**, e com ele falhando a primeira pergunta do aluno morre em dois
  minutos, calada.

## 6. O que a medição produziu, e o que ainda falta

**Entregue:** a base curada. `teaching/vocabulary.py` com o modelo e o loader,
`data/vocab/items.yaml` com os sete primeiros itens, o schema gerado em
`data/schema/vocab-item.schema.json` (`scripts/dump_vocab_schema.py`) e
`tests/test_vocabulary.py` com teste por item — cada exemplo tem que usar a
palavra, cada glosa tem que estar em inglês, e cada `rule_ids` tem que apontar
para item que existe na base contrastiva.

Os sete são as palavras que a medição viu falhar. Cinco delas são a ponte:
`actually`, `eventually`, `realize`, `pretend` e `comprehensive` citam o item
contrastivo que já existia e nunca era mencionado. `fodder` lidera pelo sentido
figurado, com o agrícola em segundo. `cannon fodder` existe para impedir a
leitura que o modelo improvisou em duas de cinco tentativas. **Não há item de
`house`** — era o controle, e a ausência é a resposta.

Três guardas de código, cada uma morta pelo seu próprio teste numa rodada de
mutação: o `id` tem que ser a palavra em slug, `extra="forbid"` recusa um
`ipa:` no YAML (a regra do G2P único, escrita onde pode ser violada), e uma
contagem de corpus maior que a varredura é erro de curadoria.

**Falta o gatilho**, e sem ele esta base é conteúdo sem caminho — o mesmo
estado em que a base contrastiva ficou entre a E3 e a E4. São duas pontas: a
pergunta do aluno levantar a palavra, e a agenda da E13 oferecê-la com a chave
generalizada de `rule_id` para "item de revisão". Nenhuma das duas existe, e o
`app.py` ainda não carrega a base.

Fica **sem número de entrega**: não está em nenhum plano, e numerar aqui
tomaria um rótulo que o autor da Fase 2 pode querer para outra coisa — mesma
razão da linha de IPA, registrada sob a tabela da §5 do `plano-fase-2.md`.
Esta medição não altera aquele plano.
