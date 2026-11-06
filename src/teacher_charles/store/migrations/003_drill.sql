-- Fase 1, E6: os tres modos, e o drill vindo do error log real.
--
-- `session.mode` existe desde a E0 e ninguem lia. Agora le: o modo escolhe o
-- prompt (prompts/modes/) e, no drill, tambem a regra que vai ser treinada.
--
-- A regra treinada fica na SESSAO, nao no turno, por duas razoes. Pedagogica:
-- um drill que troca de regra a cada mensagem nao e drill, e um questionario.
-- Tecnica: bloco estavel dentro da sessao e prefixo de prompt cacheado -- uma
-- regra por turno jogaria o cache do Ollama fora toda vez (plano §1.4).
--
-- NULL em sessao de drill e um estado legitimo: aluno sem erro registrado
-- ainda nao tem o que treinar, e o produto diz isso em vez de inventar
-- exercicio generico.
ALTER TABLE session ADD COLUMN drill_rule_id TEXT;

-- Para a rotacao: a proxima sessao de drill evita as regras treinadas nas
-- ultimas, para o aluno nao receber a mesma tres vezes seguidas.
CREATE INDEX idx_session_drill ON session(student_id, id DESC)
    WHERE drill_rule_id IS NOT NULL;
