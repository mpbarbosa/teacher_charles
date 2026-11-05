-- Fase 1, E0: esquema inicial.
--
-- O principio da Fase 1 e que correcao é dado estruturado, nao prosa. Por isso
-- `correction` existe desde a primeira migracao: sem correcoes contaveis nao ha
-- error log (E7), exercicio gerado do erro real (E6) nem repeticao espacada
-- (Fase 2).

CREATE TABLE student (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    -- Nivel CEFR. O produto atende B1+; A2 existe para o aluno que se
    -- autodeclara abaixo do alvo, e C2 para o teto da escala.
    cefr_level  TEXT    NOT NULL DEFAULT 'B1'
                CHECK (cefr_level IN ('A2', 'B1', 'B2', 'C1', 'C2')),
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE session (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id  INTEGER NOT NULL REFERENCES student(id) ON DELETE CASCADE,
    mode        TEXT    NOT NULL DEFAULT 'conversation'
                CHECK (mode IN ('conversation', 'correction', 'drill')),
    started_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    ended_at    TEXT
);

CREATE INDEX idx_session_student ON session(student_id, started_at DESC);

CREATE TABLE turn (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES session(id) ON DELETE CASCADE,
    role        TEXT    NOT NULL CHECK (role IN ('student', 'teacher')),
    content     TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_turn_session ON turn(session_id, id);

CREATE TABLE correction (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id     INTEGER NOT NULL REFERENCES turn(id) ON DELETE CASCADE,
    -- Trecho exato do aluno que motivou a correcao.
    span        TEXT    NOT NULL,
    corrected   TEXT    NOT NULL,
    error_type  TEXT    NOT NULL,
    -- Regra citada de data/contrastive/*.yaml. NULL quando a base nao tinha
    -- regra para o caso e a correcao veio so do modelo -- distincao que a E4
    -- precisa medir.
    rule_id     TEXT,
    explanation TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_correction_turn ON correction(turn_id);
CREATE INDEX idx_correction_rule ON correction(rule_id) WHERE rule_id IS NOT NULL;
