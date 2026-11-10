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
    REVIEW_SOURCES,
    Correction,
    GapCount,
    LevelChange,
    LevelSignal,
    RuleCount,
    RuleReview,
    RuleSchedule,
    Session,
    Student,
    Turn,
    UnverifiedFinding,
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

    def progress(self, student_id: int) -> dict[str, int]:
        """Os números de cabeçalho do error log (E7).

        Uma consulta só, porque são três contagens sobre a mesma junção e
        porque todo método de repositório aqui é uma instrução autônoma — a
        conexão é compartilhada e não abre transação no caminho da requisição.
        """
        row = self._conn.execute(
            """
            SELECT count(*)                                        AS corrections,
                   count(DISTINCT c.rule_id)                       AS rules,
                   sum(CASE WHEN c.rule_id IS NULL THEN 1 ELSE 0 END) AS without_rule
              FROM correction c
              JOIN turn t ON t.id = c.turn_id
              JOIN session s ON s.id = t.session_id
             WHERE s.student_id = ?
            """,
            (student_id,),
        ).fetchone()
        return {
            "corrections": row["corrections"] or 0,
            "rules": row["rules"] or 0,
            "without_rule": row["without_rule"] or 0,
        }

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
                    ORDER BY c2.id DESC LIMIT 1) AS last_correction,
                   max(c.created_at)               AS last_at
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

    def rule_history(self, student_id: int, rule_id: str, limit: int = 20) -> list[str]:
        """Quando o aluno errou por esta regra, da mais recente para trás.

        Datas, não contagem: é o que permite ao aluno ver se um erro é coisa
        de ontem ou de três semanas atrás — a diferença entre "ainda erro
        isso" e "já passou".
        """
        rows = self._conn.execute(
            """
            SELECT c.created_at AS at
              FROM correction c
              JOIN turn t ON t.id = c.turn_id
              JOIN session s ON s.id = t.session_id
             WHERE s.student_id = ? AND c.rule_id = ?
             ORDER BY c.id DESC
             LIMIT ?
            """,
            (student_id, rule_id, limit),
        ).fetchall()
        return [row["at"] for row in rows]

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


class ScheduleRepo:
    """A fila da repetição espaçada (E13).

    Duas tabelas, e a divisão entre elas é a de ``LevelRepo``: ``rule_review``
    guarda o evento, ``rule_schedule`` guarda o estado que a fila consulta. O
    estado é derivável dos eventos de propósito — quando a escada de intervalos
    mudar, ``rule_schedule`` pode ser reconstruída a partir de ``rule_review``.

    O repositório **não conhece a escada**: recebe degrau e vencimento já
    calculados, como ``LevelRepo`` recebe o nível já estimado. Quem calcula é
    ``teaching/schedule.py``, porque intervalo de revisão é regra pedagógica.

    **As duas escritas de uma revisão não são uma transação**, e isso é a regra
    da conexão compartilhada, não descuido: nada no caminho de uma requisição
    abre transação de várias instruções. A ordem é escolhida por isso — o evento
    entra primeiro, porque é dele que o estado se reconstrói. Perder o segundo
    passo deixa uma revisão sem efeito na fila; perder o primeiro deixaria a
    fila andada sem registro de por quê.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get(self, student_id: int, rule_id: str) -> RuleSchedule | None:
        """O agendamento desta regra, ou ``None`` se ela nunca foi revisada.

        ``None`` não é "agendamento zerado": é a regra que o aluno errou e nunca
        treinou, que a fila põe na frente de qualquer uma vencida.
        """
        row = self._conn.execute(
            "SELECT * FROM rule_schedule WHERE student_id = ? AND rule_id = ?",
            (student_id, rule_id),
        ).fetchone()
        return RuleSchedule.from_row(row) if row is not None else None

    def for_student(self, student_id: int) -> dict[str, RuleSchedule]:
        """Tudo o que está agendado para este aluno, por ``rule_id``.

        Uma consulta só: quem ordena a fila precisa do estado de todas as
        candidatas ao mesmo tempo, e são poucas (o error log entra com cinco).
        """
        rows = self._conn.execute(
            "SELECT * FROM rule_schedule WHERE student_id = ?",
            (student_id,),
        ).fetchall()
        return {row["rule_id"]: RuleSchedule.from_row(row) for row in rows}

    def log_review(
        self,
        student_id: int,
        rule_id: str,
        turn_id: int,
        *,
        hit: bool,
        source: str,
        box_before: int,
        box_after: int,
        interval_days: float,
    ) -> int:
        """Registra a revisão. Primeira das duas escritas, e a que importa mais."""
        if source not in REVIEW_SOURCES:
            raise ValueError(f"origem de revisão inválida: {source!r} (use um de {REVIEW_SOURCES})")
        cur = self._conn.execute(
            """
            INSERT INTO rule_review (
                student_id, rule_id, turn_id, hit, source,
                box_before, box_after, interval_days
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student_id,
                rule_id,
                turn_id,
                1 if hit else 0,
                source,
                box_before,
                box_after,
                interval_days,
            ),
        )
        return int(cur.lastrowid)

    def save(
        self,
        student_id: int,
        rule_id: str,
        *,
        box: int,
        streak: int,
        due_at: str,
        reviewed_at: str,
        lapsed: bool,
    ) -> None:
        """Põe o estado derivado no lugar. Uma instrução, como todas as daqui.

        ``reviews`` e ``lapses`` são acumulados pelo próprio SQL em vez de
        lidos-e-escritos em Python: ler para depois escrever seriam duas
        instruções, e duas requisições podem se intercalar entre elas.
        """
        self._conn.execute(
            """
            INSERT INTO rule_schedule (
                student_id, rule_id, box, streak, reviews, lapses, due_at, last_review_at
            ) VALUES (?, ?, ?, ?, 1, ?, ?, ?)
            ON CONFLICT(student_id, rule_id) DO UPDATE SET
                box            = excluded.box,
                streak         = excluded.streak,
                reviews        = rule_schedule.reviews + 1,
                lapses         = rule_schedule.lapses + excluded.lapses,
                due_at         = excluded.due_at,
                last_review_at = excluded.last_review_at
            """,
            (student_id, rule_id, box, streak, 1 if lapsed else 0, due_at, reviewed_at),
        )

    def reviews(
        self,
        student_id: int,
        rule_id: str | None = None,
        limit: int = 50,
    ) -> list[RuleReview]:
        """As revisões mais recentes, da mais nova para trás.

        É o material de recalibrar a escada: o intervalo que o aluno de fato
        recebeu está na linha, não na constante de hoje.
        """
        if rule_id is None:
            rows = self._conn.execute(
                "SELECT * FROM rule_review WHERE student_id = ? ORDER BY id DESC LIMIT ?",
                (student_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT * FROM rule_review
                 WHERE student_id = ? AND rule_id = ?
                 ORDER BY id DESC LIMIT ?
                """,
                (student_id, rule_id, limit),
            ).fetchall()
        return [RuleReview.from_row(row) for row in rows]

    def count(self) -> int:
        return self._conn.execute("SELECT count(*) AS n FROM rule_review").fetchone()["n"]


class UnverifiedFindingRepo:
    """A fila de curadoria: o que o verificador barrou, agregado por span.

    O avesso de ``CorrectionRepo``. Lá ficam as correções que passaram pela
    garantia da E4; aqui, as que foram barradas por não citar regra curada —
    que é o mesmo que dizer: os erros que ``data/contrastive/`` ainda não vê.
    Antes disto a medida existia (``analysis.unverified``, um contador) e a
    lista não: o span ia para ``log.debug``.

    Nada daqui chega ao aluno, e não é por descuido de rota: é gramática que o
    modelo pequeno improvisou, exatamente o que a E4 existe para barrar. Quem
    lê é ``scripts/curation_queue.py``, por quem escreve YAML.

    ``turn_id`` aponta para o turno **do aluno**, como em ``correction``.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(
        self,
        turn_id: int,
        *,
        span: str,
        span_key: str,
        corrected: str,
        error_type: str,
        explanation: str,
        candidates: str = "",
    ) -> int:
        """Registra um achado barrado.

        ``span_key`` chega pronto, e não é preguiça: ``store/`` não importa
        ``teaching/`` — a dependência corre num sentido só —, e a chave tem de
        sair da MESMA normalização do matcher, que mora lá
        (``contrastive.span_key``). Derivá-la aqui com o ``lower()`` do SQLite,
        que é só ASCII, daria duas normalizações divergentes para a única
        coisa que esta tabela precisa agregar direito.
        """
        cur = self._conn.execute(
            """
            INSERT INTO unverified_finding
                   (turn_id, span, span_key, corrected, error_type, explanation, candidates)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (turn_id, span, span_key, corrected, error_type, explanation, candidates),
        )
        return int(cur.lastrowid)

    def for_turn(self, turn_id: int) -> list[UnverifiedFinding]:
        rows = self._conn.execute(
            "SELECT * FROM unverified_finding WHERE turn_id = ? ORDER BY id",
            (turn_id,),
        ).fetchall()
        return [UnverifiedFinding.from_row(row) for row in rows]

    def backlog(self, limit: int = 20) -> list[GapCount]:
        """A fila, do buraco mais frequente para baixo.

        Sem ``student_id``: a base contrastiva é conteúdo de produto, não
        progresso pessoal, e o span que dois perfis da máquina produziram é o
        mesmo buraco — separar por aluno faria a fila ordenar errado. É a
        única agregação daqui que não é do aluno, e é de propósito.

        A junção pelo ``max(id)`` em vez de coluna solta ao lado do agregado:
        o SQLite deixaria passar, mas por uma particularidade dele, e consulta
        que depende de particularidade do motor é consulta que quebra calada
        na próxima vez que alguém mexer no ``GROUP BY``.
        """
        rows = self._conn.execute(
            """
            SELECT u.span_key    AS span_key,
                   g.times       AS times,
                   u.span        AS last_span,
                   u.corrected   AS last_correction,
                   u.error_type  AS last_type,
                   u.explanation AS last_explanation,
                   u.candidates  AS last_candidates,
                   u.created_at  AS last_at
              FROM unverified_finding u
              JOIN (SELECT span_key, count(*) AS times, max(id) AS last_id
                      FROM unverified_finding
                     GROUP BY span_key) g
                ON g.last_id = u.id
             ORDER BY g.times DESC, u.id DESC
             LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [GapCount.from_row(row) for row in rows]

    def count(self) -> int:
        return self._conn.execute("SELECT count(*) AS n FROM unverified_finding").fetchone()["n"]

    def distinct_spans(self) -> int:
        """Quantos buracos distintos, que é o tamanho da fila de curadoria.

        Diferente de ``count()``: dez turnos com ``"I have 30 years"`` são dez
        linhas e um item de YAML a escrever.
        """
        return self._conn.execute(
            "SELECT count(DISTINCT span_key) AS n FROM unverified_finding"
        ).fetchone()["n"]
