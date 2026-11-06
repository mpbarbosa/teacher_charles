-- Fase 1, E5: perfil e nivel CEFR.
--
-- O nivel do aluno deixa de ser uma coluna que ninguem le e passa a mudar o
-- prompt da resposta. Com isso aparecem tres perguntas que o esquema da E0 nao
-- respondia, e cada uma virou tabela ou coluna aqui:
--
--   "de onde veio o nivel em vigor?"  -> student.level_source
--   "o que o aluno disse quando chegou?" -> student.self_declared_level
--   "por que ele mudou?"             -> level_signal + level_change
--
-- `level_signal` guarda os sinais de TODO turno, inclusive os que nao contaram
-- como evidencia (`counted = 0`): sem as linhas descartadas nao da para medir
-- se o corte de MIN_WORDS esta no lugar certo.

-- O nivel que o aluno declarou. Fica separado de `cefr_level` porque este
-- ultimo anda sozinho a partir dos sinais, e perder o ponto de partida tira a
-- unica referencia externa que existe.
ALTER TABLE student ADD COLUMN self_declared_level TEXT
    CHECK (self_declared_level IS NULL
           OR self_declared_level IN ('A2', 'B1', 'B2', 'C1', 'C2'));

-- 'self' = o nivel em vigor e o que o aluno declarou; 'auto' = os sinais o
-- moveram. O aluno pode perguntar por que o Charles mudou de registro, e esta
-- coluna com `level_change` sao a resposta.
ALTER TABLE student ADD COLUMN level_source TEXT NOT NULL DEFAULT 'self'
    CHECK (level_source IN ('self', 'auto'));

-- Qual perfil esta em uso. Sao varios alunos numa maquina, nao multiusuario
-- com login (plano §4): quem escreve o nome no perfil passa a ser o aluno
-- ativo, e o ativo e o de maior `seen_seq`.
--
-- Contador, e nao marca de tempo, porque "o mais recente" por relogio empata:
-- a primeira versao usava datetime('now') (resolucao de segundo), a segunda
-- strftime com milissegundo, e um teste de troca-e-volta de perfil quebrou nas
-- duas -- as duas escritas caiam no mesmo instante. O contador nao depende de
-- relogio nenhum e ordena sempre.
ALTER TABLE student ADD COLUMN seen_seq INTEGER NOT NULL DEFAULT 0;

UPDATE student
   SET self_declared_level = cefr_level
 WHERE self_declared_level IS NULL;

UPDATE student SET seen_seq = id WHERE seen_seq = 0;

CREATE INDEX idx_student_seen_seq ON student(seen_seq DESC);

-- Os sinais de um turno, como dado. O mesmo motivo de `correction` existir:
-- nivel que o professor estima de cabeca nao e agregavel, nao e depuravel e
-- nao sobrevive ao reinicio.
CREATE TABLE level_signal (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id             INTEGER NOT NULL REFERENCES turn(id) ON DELETE CASCADE,
    words               INTEGER NOT NULL,
    sentences           INTEGER NOT NULL,
    long_words          INTEGER NOT NULL,
    connectives         INTEGER NOT NULL,
    structures          INTEGER NOT NULL,
    -- Erros com regra curada neste turno. Correcao sem regra nao entra: e a
    -- mesma politica da E4 (plano §5.2).
    errors              INTEGER NOT NULL DEFAULT 0,
    score               REAL    NOT NULL,
    estimated_level     TEXT    NOT NULL
                        CHECK (estimated_level IN ('A2', 'B1', 'B2', 'C1', 'C2')),
    -- 0 quando o turno foi curto demais para ser evidencia.
    counted             INTEGER NOT NULL DEFAULT 1 CHECK (counted IN (0, 1)),
    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_level_signal_turn ON level_signal(turn_id);

-- O historico do nivel. Uma linha por mudanca, com o motivo em texto.
CREATE TABLE level_change (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id  INTEGER NOT NULL REFERENCES student(id) ON DELETE CASCADE,
    from_level  TEXT    NOT NULL CHECK (from_level IN ('A2', 'B1', 'B2', 'C1', 'C2')),
    to_level    TEXT    NOT NULL CHECK (to_level   IN ('A2', 'B1', 'B2', 'C1', 'C2')),
    source      TEXT    NOT NULL CHECK (source IN ('self', 'auto')),
    reason      TEXT    NOT NULL DEFAULT '',
    -- Ate que sinal esta mudanca consumiu. A janela de histerese so olha os
    -- sinais DEPOIS dela: sem isto, os mesmos quatro turnos que moveram o
    -- nivel continuariam na janela e o moveriam de novo no turno seguinte --
    -- quatro turnos levariam o aluno de A2 a C1 em seis.
    at_signal_id INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_level_change_student ON level_change(student_id, id DESC);
