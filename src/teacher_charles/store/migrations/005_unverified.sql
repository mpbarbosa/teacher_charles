-- A fila de curadoria: as correcoes que o verificador descartou.
--
-- A E4 barra toda correcao que nao cita regra curada (`_Verifier.verify`), e
-- isso e o anteparo contra o problema no 1 da analise inicial -- o modelo
-- pequeno inventar gramatica. Mas o achado descartado nao era so ruido: ele
-- aponta, no texto real de um aluno, um erro que a base de
-- `data/contrastive/` nao cobre. Ate aqui restava `analysis.unverified`, um
-- contador, e o span e a correcao propostos iam para `log.debug` -- isto e, a
-- medida de quanto a base nao cobre existia e a LISTA nao.
--
-- Esta tabela e aquela lista. Ela nao muda uma linha do que chega ao aluno: o
-- descarte continua acontecendo antes, e `Analysis.as_dict` nao expoe estas
-- linhas. O que entra aqui e material de curadoria, lido por
-- `scripts/curation_queue.py` por quem escreve YAML -- nunca pelo aluno.
--
-- ## O ponto cego, que e melhor estar escrito aqui do que ser redescoberto
--
-- Com `require_curated_rule` ligado (o default), a analise NAO RODA quando a
-- base nao levanta candidato nenhum (`Analyzer.analyze`, e a economia e
-- medida: cada chamada joga o prompt da conversa fora do cache). Sem chamada
-- nao ha achado, e sem achado nao ha linha aqui. Logo:
--
--   toda linha desta tabela vem de um turno em que a base JA ACERTOU alguma
--   outra regra, e o modelo viu um erro a mais ao lado dela.
--
-- O buraco mais fundo -- o turno com erro que a base nao enxerga de jeito
-- nenhum -- e invisivel por construcao, e nao da para fechar com dado: sem
-- rodar o modelo, "ingles correto" e "erro que a base nao ve" produzem a mesma
-- ausencia de candidato, que e a assimetria da §6 do plano da Fase 2 outra vez.
-- Para ir atras dele a tabela serve, mas com `TC_REQUIRE_CURATED_RULE=false`:
-- ai a analise roda em todo turno, tudo o que o modelo propoe sem regra cai
-- aqui, e a fila passa a cobrir o inverso. Isso e rodada de curadoria
-- deliberada -- custa a latencia que o default protege --, nao uso normal.
CREATE TABLE unverified_finding (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    -- O turno do ALUNO que contem o erro, como em `correction` -- e nao a
    -- resposta do professor.
    turn_id     INTEGER NOT NULL REFERENCES turn(id) ON DELETE CASCADE,
    -- O trecho exato do aluno, como ele escreveu. E o que vai virar `wrong:`
    -- no YAML, entao precisa ser literal.
    span        TEXT    NOT NULL,
    -- O mesmo span depois de `contrastive.normalize` e `lower()`. Guardado, e
    -- nao calculado na consulta, por duas razoes: o `lower()` do SQLite e so
    -- ASCII, e a agregacao tem que usar a MESMA normalizacao do matcher -- a
    -- chave aqui e o texto contra o qual o padrao novo vai ser escrito.
    span_key    TEXT    NOT NULL,
    -- O que o modelo propos no lugar. Vira candidato a `right:`.
    corrected   TEXT    NOT NULL,
    error_type  TEXT    NOT NULL,
    -- A prosa do modelo. Guardada apesar de ser exatamente a gramatica
    -- improvisada que o verificador barrou: para o curador ela diz o que o
    -- modelo ACHOU que era a regra, e isso encurta a escrita do item. Nota de
    -- curadoria, no mesmo espirito de `ContrastiveItem.notes`, e pelo mesmo
    -- motivo nunca sai para o aluno.
    explanation TEXT    NOT NULL,
    -- As regras que a base levantou NESTE turno, separadas por espaco, ou ''.
    -- Nao e decoracao: span novo ao lado de `calque.age-have` costuma ser
    -- padrao a alargar no item que ja existe, e span sozinho costuma ser item
    -- novo. Guardado desnormalizado porque esta tabela nao se junta com nada
    -- -- quem le e um script de curadoria, nao o caminho da requisicao.
    candidates  TEXT    NOT NULL DEFAULT '',
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- A fila e lida agregada por span: o que mais aparece primeiro.
CREATE INDEX idx_unverified_finding_key ON unverified_finding(span_key);
CREATE INDEX idx_unverified_finding_turn ON unverified_finding(turn_id);
