-- Fase 2, E13: repeticao espacada.
--
-- Ate aqui o drill escolhia a regra por FREQUENCIA no error log (E6): a mais
-- corrigida primeiro, com um rodizio para nao repetir. Frequencia e um numero
-- que so sobe -- ela sabe o que o aluno errou e nao sabe o que ele aprendeu.
-- Consequencia medida no proprio desenho da E6: a regra mais errada do
-- historico continua sendo a mais errada depois de o aluno acertar dez vezes,
-- e o drill nunca a solta.
--
-- Entao o sinal que faltava nao e "errou mais", e "acertou desde entao". Duas
-- tabelas, e a divisao entre elas e a mesma de `level_signal` / `level_change`:
--
--   `rule_review`   -- o evento, uma linha por revisao, inclusive as que nao
--                      mudaram degrau. Sem as linhas cruas nao da para
--                      recalibrar a escada de intervalos depois, e escada que
--                      ninguem mede e chute.
--   `rule_schedule` -- o estado derivado: em que degrau a regra esta e quando
--                      ela volta. E o que a fila do drill consulta.
--
-- O estado e derivavel das revisoes, e isso e de proposito: `rule_schedule`
-- pode ser reconstruida de `rule_review` quando a escada mudar.

-- Uma revisao: o aluno teve a chance de aplicar a regra, e acertou ou nao.
--
-- `source` e a parte que nao se adivinha olhando a tabela, e ela carrega a
-- assimetria do sinal:
--
--   'drill' -- o aluno respondeu a um exercicio SOBRE esta regra. Aqui a
--              AUSENCIA de correcao e evidencia de acerto, porque a
--              oportunidade foi construida pelo exercicio: o drill pediu a
--              estrutura e esperou a resposta.
--   'lapse' -- o aluno errou por esta regra em qualquer outro lugar. Isto e
--              evidencia negativa, e vale em qualquer modo.
--
-- Nao existe 'hit' fora do drill, e e por isso que a tabela registra a origem:
-- um turno de conversa sem correcao NAO prova acerto nenhum -- prova que o
-- aluno nao usou a estrutura, ou usou e ninguem reparou. Contar isso como
-- acerto graduaria a regra pelo silencio e e precisao inventada, a mesma
-- classe de erro que a §3.4 do plano da Fase 2 proibe no retorno de pronuncia.
CREATE TABLE rule_review (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id    INTEGER NOT NULL REFERENCES student(id) ON DELETE CASCADE,
    -- Regra de data/contrastive/*.yaml. Aqui NUNCA e NULL, ao contrario de
    -- `correction.rule_id`: revisao sem regra curada nao tem o que agendar.
    rule_id       TEXT    NOT NULL,
    -- O turno do ALUNO que produziu a evidencia, como em `correction`.
    turn_id       INTEGER NOT NULL REFERENCES turn(id) ON DELETE CASCADE,
    hit           INTEGER NOT NULL CHECK (hit IN (0, 1)),
    source        TEXT    NOT NULL CHECK (source IN ('drill', 'lapse')),
    -- O degrau antes e depois, e o intervalo que saiu dele. Guardados na linha
    -- porque a escada e constante de codigo: quando ela mudar, as linhas
    -- antigas continuam dizendo que intervalo o aluno de fato recebeu.
    box_before    INTEGER NOT NULL,
    box_after     INTEGER NOT NULL,
    interval_days REAL    NOT NULL,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_rule_review_student ON rule_review(student_id, rule_id, id DESC);

-- O estado de uma regra na fila deste aluno.
--
-- `due_at` e o que a fila le, e um instante e nao um contador -- ao contrario
-- de `student.seen_seq`, que e contador porque empate dentro do mesmo
-- milissegundo quebrava a troca de perfil. Aqui o empate nao existe: o
-- intervalo mais curto da escada e de meio dia.
--
-- Nao ha linha para regra nunca revisada, e a ausencia e informacao: regra que
-- o aluno errou e nunca treinou e a mais urgente da fila, na frente de
-- qualquer uma com `due_at` vencido. Criar linha "vazia" no primeiro erro
-- apagaria essa distincao.
CREATE TABLE rule_schedule (
    student_id     INTEGER NOT NULL REFERENCES student(id) ON DELETE CASCADE,
    rule_id        TEXT    NOT NULL,
    -- Degrau na escada de intervalos de teaching/schedule.py. Acerto sobe um,
    -- erro volta para zero.
    box            INTEGER NOT NULL DEFAULT 0,
    -- Acertos CONSECUTIVOS. E o que gradua a regra, e zera no primeiro erro --
    -- por isso nao da para derivar de `reviews - lapses`.
    streak         INTEGER NOT NULL DEFAULT 0,
    reviews        INTEGER NOT NULL DEFAULT 0,
    lapses         INTEGER NOT NULL DEFAULT 0,
    due_at         TEXT    NOT NULL,
    last_review_at TEXT    NOT NULL,
    created_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (student_id, rule_id)
);

-- A fila do drill: as regras deste aluno ordenadas por vencimento.
CREATE INDEX idx_rule_schedule_due ON rule_schedule(student_id, due_at);
