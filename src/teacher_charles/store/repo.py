"""Acesso aos dados.

O que a E0 precisa para provar que o banco funciona de ponta a ponta, o que a
E2 usa para persistir histórico de conversa e o que a E4 usa para persistir
correção. Agregações do error log são da E7.
"""

from __future__ import annotations

import sqlite3

from .models import (
    CEFR_LEVELS,
    LEVEL_SOURCES,
    MODES,
    Correction,
    LevelChange,
    LevelSignal,
    RuleCount,
    Session,
    Student,
    Turn,
)

# O próximo número da fila de "visto por último".
#
# Contador e não relógio: ``datetime('now')`` tem resolução de segundo e
# ``strftime`` com ``%f`` de milissegundo, e as duas versões quebraram o mesmo
# teste — trocar de perfil e voltar acontece dentro do mesmo milissegundo, os
# dois registros empatam e o desempate por id devolve sempre o mesmo aluno.
# Continua uma instrução autônoma, que é o que a conexão compartilhada exige.
NEXT_SEEN = "(SELECT COALESCE(MAX(seen_seq), 0) + 1 FROM student)"


def _check_level(cefr_level: str) -> None:
    if cefr_level not in CEFR_LEVELS:
        raise ValueError(f"nível CEFR inválido: {cefr_level!r} (use um de {CEFR_LEVELS})")


class StudentRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get_or_create(self, name: str, cefr_level: str = "B1") -> Student:
        _check_level(cefr_level)

        row = self._conn.execute("SELECT * FROM student WHERE name = ?", (name,)).fetchone()
        if row is not None:
            return Student.from_row(row)

        self._conn.execute(
            f"""
            INSERT INTO student (name, cefr_level, self_declared_level, seen_seq)
            VALUES (?, ?, ?, {NEXT_SEEN})
            """,
            (name, cefr_level, cefr_level),
        )
        row = self._conn.execute("SELECT * FROM student WHERE name = ?", (name,)).fetchone()
        return Student.from_row(row)

    def get(self, student_id: int) -> Student:
        row = self._conn.execute("SELECT * FROM student WHERE id = ?", (student_id,)).fetchone()
        if row is None:
            raise LookupError(f"aluno {student_id} não existe")
        return Student.from_row(row)

    def active(self) -> Student | None:
        """O perfil em uso: o visto por último.

        São vários alunos numa máquina, não multiusuário com login (plano §4).
        Quem escreve o nome no perfil passa a ser o ativo, e o ativo continua
        ativo quando a página é reaberta.
        """
        row = self._conn.execute(
            "SELECT * FROM student ORDER BY seen_seq DESC, id DESC LIMIT 1"
        ).fetchone()
        return Student.from_row(row) if row is not None else None

    def touch(self, student_id: int) -> None:
        """Marca o perfil como o ativo."""
        self._conn.execute(
            f"UPDATE student SET seen_seq = {NEXT_SEEN} WHERE id = ?",
            (student_id,),
        )

    def set_level(self, student_id: int, cefr_level: str, source: str = "self") -> Student:
        """Põe o nível em vigor, dizendo de onde ele veio.

        ``source='self'`` move também ``self_declared_level``: o aluno que
        corrige o próprio nível está redefinindo o ponto de partida, e deixar o
        antigo ali faria o histórico mentir. ``source='auto'`` nunca toca nele —
        é a referência externa contra a qual a estimativa é medida.
        """
        _check_level(cefr_level)
        if source not in LEVEL_SOURCES:
            raise ValueError(f"origem de nível inválida: {source!r} (use um de {LEVEL_SOURCES})")

        if source == "self":
            cur = self._conn.execute(
                """
                UPDATE student
                   SET cefr_level = ?, self_declared_level = ?, level_source = 'self',
                       updated_at = datetime('now')
                 WHERE id = ?
                """,
                (cefr_level, cefr_level, student_id),
            )
        else:
            cur = self._conn.execute(
                """
                UPDATE student
                   SET cefr_level = ?, level_source = 'auto', updated_at = datetime('now')
                 WHERE id = ?
                """,
                (cefr_level, student_id),
            )
        if cur.rowcount == 0:
            raise LookupError(f"aluno {student_id} não existe")

        return self.get(student_id)

    def all(self) -> list[Student]:
        rows = self._conn.execute("SELECT * FROM student ORDER BY name").fetchall()
        return [Student.from_row(row) for row in rows]

    def count(self) -> int:
        return self._conn.execute("SELECT count(*) AS n FROM student").fetchone()["n"]


class SessionRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def start(
        self,
        student_id: int,
        mode: str = "conversation",
        drill_rule_id: str | None = None,
    ) -> int:
        if mode not in MODES:
            raise ValueError(f"modo inválido: {mode!r} (use um de {MODES})")
        cur = self._conn.execute(
            "INSERT INTO session (student_id, mode, drill_rule_id) VALUES (?, ?, ?)",
            (student_id, mode, drill_rule_id),
        )
        return int(cur.lastrowid)

    def get(self, session_id: int) -> Session:
        row = self._conn.execute("SELECT * FROM session WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise LookupError(f"sessão {session_id} não existe")
        return Session.from_row(row)

    def current(self, student_id: int) -> Session | None:
        row = self._conn.execute(
            """
            SELECT * FROM session
            WHERE student_id = ? AND ended_at IS NULL
            ORDER BY id DESC LIMIT 1
            """,
            (student_id,),
        ).fetchone()
        return Session.from_row(row) if row is not None else None

    def drilled_rules(self, student_id: int, limit: int = 3) -> list[str]:
        """As regras das últimas sessões de drill, da mais recente para trás.

        É o que impede o aluno de receber a mesma regra três vezes seguidas:
        quem escolhe a próxima evita estas.
        """
        rows = self._conn.execute(
            """
            SELECT drill_rule_id AS rule_id FROM session
            WHERE student_id = ? AND drill_rule_id IS NOT NULL
            ORDER BY id DESC LIMIT ?
            """,
            (student_id, limit),
        ).fetchall()
        return [row["rule_id"] for row in rows]

    def current_or_start(self, student_id: int, mode: str = "conversation") -> int:
        """A sessão aberta do aluno, ou uma nova.

        O aluno não deve precisar "criar uma sessão" para começar a falar;
        abrir a página basta.
        """
        row = self._conn.execute(
            """
            SELECT id FROM session
            WHERE student_id = ? AND ended_at IS NULL
            ORDER BY id DESC LIMIT 1
            """,
            (student_id,),
        ).fetchone()
        if row is not None:
            return int(row["id"])
        return self.start(student_id, mode)

    def switch_mode(
        self,
        student_id: int,
        mode: str,
        drill_rule_id: str | None = None,
    ) -> int:
        """Encerra a sessão aberta e começa outra no modo pedido.

        Modo é propriedade da sessão desde a E0, e trocar de modo é trocar de
        atividade: a transcrição de um drill não é continuação da conversa
        livre. O histórico antigo fica no banco — é o material do error log.
        """
        current = self.current(student_id)
        if current is not None:
            self.end(current.id)
        return self.start(student_id, mode, drill_rule_id)

    def end(self, session_id: int) -> None:
        self._conn.execute(
            "UPDATE session SET ended_at = datetime('now') WHERE id = ?",
            (session_id,),
        )


class TurnRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def append(self, session_id: int, role: str, content: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO turn (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        return int(cur.lastrowid)

    def history(self, session_id: int, limit: int = 50) -> list[Turn]:
        """Os ``limit`` turnos mais recentes, em ordem cronológica."""
        rows = self._conn.execute(
            "SELECT * FROM turn WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [Turn.from_row(r) for r in reversed(rows)]


class CorrectionRepo:
    """As correções, como dado contável.

    O princípio da Fase 1 em uma tabela: a correção que o aluno recebe não é só
    prosa na resposta do professor, é linha agregável. Sem isto não há error log
    (E7), drill a partir do erro real (E6) nem repetição espaçada (Fase 2).

    ``turn_id`` aponta para o turno **do aluno** — o turno que contém o erro —,
    não para a resposta do professor.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(
        self,
        turn_id: int,
        *,
        span: str,
        corrected: str,
        error_type: str,
        explanation: str,
        rule_id: str | None = None,
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO correction (turn_id, span, corrected, error_type, rule_id, explanation)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (turn_id, span, corrected, error_type, rule_id, explanation),
        )
        return int(cur.lastrowid)

    def for_turn(self, turn_id: int) -> list[Correction]:
        rows = self._conn.execute(
            "SELECT * FROM correction WHERE turn_id = ? ORDER BY id",
            (turn_id,),
        ).fetchall()
        return [Correction.from_row(row) for row in rows]

    def for_session(self, session_id: int) -> list[Correction]:
        rows = self._conn.execute(
            """
            SELECT c.* FROM correction c
            JOIN turn t ON t.id = c.turn_id
            WHERE t.session_id = ?
            ORDER BY c.id
            """,
            (session_id,),
        ).fetchall()
        return [Correction.from_row(row) for row in rows]

    def count(self) -> int:
        return self._conn.execute("SELECT count(*) AS n FROM correction").fetchone()["n"]

    def top_rules(self, student_id: int, limit: int = 5) -> list[RuleCount]:
        """As regras que mais corrigiram este aluno, da mais frequente para baixo.

        Só correção com regra curada entra — ``rule_id IS NULL`` é correção que
        veio só do modelo, e treinar o aluno em cima de gramática improvisada
        seria pior que não treinar (plano §5.2). É esta consulta que torna o
        drill da E6 um drill do erro **dele**: sem ela, só restaria exercício
        genérico.

        A agregação é do aluno, não da sessão: o erro de ontem conta.
        """
        rows = self._conn.execute(
            """
            SELECT c.rule_id                       AS rule_id,
                   count(*)                        AS times,
                   -- O exemplo mais recente, para o drill mostrar ao aluno as
                   -- palavras dele e não um exemplo de livro.
                   (SELECT c2.span FROM correction c2
                     JOIN turn t2 ON t2.id = c2.turn_id
                     JOIN session s2 ON s2.id = t2.session_id
                    WHERE s2.student_id = ? AND c2.rule_id = c.rule_id
                    ORDER BY c2.id DESC LIMIT 1) AS last_span,
                   (SELECT c2.corrected FROM correction c2
                     JOIN turn t2 ON t2.id = c2.turn_id
                     JOIN session s2 ON s2.id = t2.session_id
                    WHERE s2.student_id = ? AND c2.rule_id = c.rule_id
                    ORDER BY c2.id DESC LIMIT 1) AS last_correction
              FROM correction c
              JOIN turn t ON t.id = c.turn_id
              JOIN session s ON s.id = t.session_id
             WHERE s.student_id = ? AND c.rule_id IS NOT NULL
             GROUP BY c.rule_id
             ORDER BY times DESC, max(c.id) DESC
             LIMIT ?
            """,
            (student_id, student_id, student_id, limit),
        ).fetchall()
        return [RuleCount.from_row(row) for row in rows]

    def spans_for_rule(self, student_id: int, rule_id: str, limit: int = 3) -> list[str]:
        """As frases do próprio aluno que caíram nesta regra, da mais recente."""
        rows = self._conn.execute(
            """
            SELECT DISTINCT c.span AS span, max(c.id) AS last_id
              FROM correction c
              JOIN turn t ON t.id = c.turn_id
              JOIN session s ON s.id = t.session_id
             WHERE s.student_id = ? AND c.rule_id = ?
             GROUP BY c.span
             ORDER BY last_id DESC
             LIMIT ?
            """,
            (student_id, rule_id, limit),
        ).fetchall()
        return [row["span"] for row in rows]

    def count_with_rule(self) -> int:
        """Quantas correções têm regra curada por trás.

        A razão entre este número e ``count()`` é o que a E4 mede: quanto da
        correção vem da base e quanto vem do modelo solto.
        """
        return self._conn.execute(
            "SELECT count(*) AS n FROM correction WHERE rule_id IS NOT NULL"
        ).fetchone()["n"]


class LevelRepo:
    """Os sinais de nível e o histórico de mudança.

    Mesmo motivo de ``CorrectionRepo`` existir: nível que o professor estima de
    cabeça não é agregável nem depurável. Aqui ele é linha — com os números que
    o produziram do lado, para a rubrica poder ser recalibrada depois sobre
    turnos reais em vez de amostras escritas à mão.

    O repositório não conhece a rubrica: recebe números já calculados. Quem
    calcula é ``teaching/level.py``, porque é regra pedagógica.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def record(
        self,
        turn_id: int,
        *,
        words: int,
        sentences: int,
        long_words: int,
        connectives: int,
        structures: int,
        errors: int,
        score: float,
        estimated_level: str,
        counted: bool,
    ) -> int:
        _check_level(estimated_level)
        cur = self._conn.execute(
            """
            INSERT INTO level_signal (
                turn_id, words, sentences, long_words, connectives, structures,
                errors, score, estimated_level, counted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                turn_id,
                words,
                sentences,
                long_words,
                connectives,
                structures,
                errors,
                score,
                estimated_level,
                1 if counted else 0,
            ),
        )
        return int(cur.lastrowid)

    def recent_levels(self, student_id: int, limit: int = 4) -> list[str]:
        """Os níveis estimados mais recentes do aluno, do mais novo ao mais velho.

        Três filtros, e cada um corrige um jeito de a janela mentir:

        - só turnos que contaram como evidência;
        - **através das sessões**, porque a evidência é do aluno e não da
          conversa de hoje — quem começa conversa nova não volta à estaca zero;
        - só o que veio **depois da última mudança de nível**. Sem isto os
          mesmos quatro turnos que moveram o nível continuariam na janela e o
          moveriam outra vez no turno seguinte: quatro turnos bons levariam o
          aluno de A2 a C1 em seis.
        """
        rows = self._conn.execute(
            """
            SELECT ls.estimated_level AS level
              FROM level_signal ls
              JOIN turn t ON t.id = ls.turn_id
              JOIN session s ON s.id = t.session_id
             WHERE s.student_id = ?
               AND ls.counted = 1
               AND ls.id > COALESCE(
                   (SELECT MAX(at_signal_id) FROM level_change WHERE student_id = ?), 0
               )
             ORDER BY ls.id DESC
             LIMIT ?
            """,
            (student_id, student_id, limit),
        ).fetchall()
        return [row["level"] for row in rows]

    def last_signal_id(self, student_id: int) -> int:
        """O id do sinal mais recente do aluno, ou 0.

        É o que uma mudança de nível declarada pelo aluno grava em
        ``at_signal_id``: declarar o próprio nível também zera a janela, senão
        a evidência acumulada antes da declaração a desfaria no turno seguinte.
        """
        row = self._conn.execute(
            """
            SELECT COALESCE(MAX(ls.id), 0) AS last
              FROM level_signal ls
              JOIN turn t ON t.id = ls.turn_id
              JOIN session s ON s.id = t.session_id
             WHERE s.student_id = ?
            """,
            (student_id,),
        ).fetchone()
        return int(row["last"])

    def signals_for_turn(self, turn_id: int) -> LevelSignal | None:
        row = self._conn.execute(
            "SELECT * FROM level_signal WHERE turn_id = ? ORDER BY id DESC LIMIT 1",
            (turn_id,),
        ).fetchone()
        return LevelSignal.from_row(row) if row is not None else None

    def log_change(
        self,
        student_id: int,
        from_level: str,
        to_level: str,
        *,
        source: str = "auto",
        reason: str = "",
        at_signal_id: int = 0,
    ) -> int:
        _check_level(from_level)
        _check_level(to_level)
        if source not in LEVEL_SOURCES:
            raise ValueError(f"origem de nível inválida: {source!r} (use um de {LEVEL_SOURCES})")
        cur = self._conn.execute(
            """
            INSERT INTO level_change (
                student_id, from_level, to_level, source, reason, at_signal_id
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (student_id, from_level, to_level, source, reason, at_signal_id),
        )
        return int(cur.lastrowid)

    def changes(self, student_id: int, limit: int = 20) -> list[LevelChange]:
        """As mudanças mais recentes, do mais novo ao mais velho."""
        rows = self._conn.execute(
            "SELECT * FROM level_change WHERE student_id = ? ORDER BY id DESC LIMIT ?",
            (student_id, limit),
        ).fetchall()
        return [LevelChange.from_row(row) for row in rows]
