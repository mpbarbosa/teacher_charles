"""Repetição espaçada: quando uma regra volta ao drill (E13).

Até a E6 o drill escolhia a regra por **frequência** no error log: a mais
corrigida primeiro, com um rodízio para não repetir. O problema não é de
implementação, é do sinal — frequência é um número que só sobe. Ela sabe o que
o aluno errou e não sabe o que ele aprendeu, então a regra mais errada do
histórico continua sendo a mais errada depois de o aluno acertar dez vezes, e o
drill nunca a solta.

O que faltava não é "errou mais", é **"acertou desde então"**. Com isso a fila
deixa de ser um ranking e passa a ser uma agenda.

## O que conta como acerto, e a assimetria que isso impõe

Não existe um sinal de "o aluno usou a estrutura e acertou": o analisador
devolve erros, não acertos. Então acerto aqui é **ausência de erro num lugar
onde o erro teria aparecido**, e esse lugar é o drill: o exercício pediu a
estrutura e esperou a resposta, logo a oportunidade foi construída.

Daí a assimetria, que é a decisão de desenho desta entrega:

- **No drill, sobre a regra em treino**: turno sem correção pela regra é acerto.
- **Em qualquer lugar**: correção pela regra é erro, e vale em qualquer modo.
- **Fora do drill, turno sem correção não é nada.** Não prova acerto — prova que
  o aluno não usou a estrutura, ou usou e ninguém reparou.

Contar o silêncio da conversa como acerto graduaria regras por omissão, e seria
precisão inventada: a mesma classe de erro que a §3.4 do plano da Fase 2 proíbe
no retorno de pronúncia, onde o retorno fala em classe de erro e não em símbolo
de vogal.

### O ponto cego que isto deixa, e por que não o remendei

Dentro do drill, "não errou" e "não tentou" são a mesma observação. O aluno que
responde *"yes"* a três exercícios gradua a regra sem ter usado a estrutura uma
única vez.

Não há guarda determinística para isso hoje: o analisador devolve erros, não
"o aluno usou a estrutura", e qualquer piso inventado aqui — contagem de
palavras, lista de respostas triviais — seria número sem medição, que é
exatamente o que esta base de código recusa. O que existe é o **dado para
resolver depois**: ``rule_review.turn_id`` aponta para o turno, então dá para
voltar e ler o que o aluno escreveu em cada acerto registrado e medir quantos
não eram tentativa. Está anotado como dívida na §6 do plano da Fase 2, e
pinado em ``test_uma_resposta_vazia_de_conteudo_conta_como_acerto`` — o teste
existe para o ponto cego ser visível na suíte em vez de surpresa em produção.

## A escada, e por que o primeiro degrau não é zero

``INTERVALS_DAYS`` é uma escada de Leitner: acerto sobe um degrau, erro volta ao
degrau zero. O intervalo de um degrau é quanto tempo a regra fica fora da fila.

O degrau zero vale **meio dia**, e não zero, e é isso que substitui o rodízio da
E6. Com zero, a regra que o aluno acabou de errar vence na hora e volta a ser a
mais vencida da fila na sessão seguinte — o aluno receberia a mesma regra para
sempre e nunca veria a segunda. Era exatamente isso que ``DRILL_ROTATION``
remendava, filtrando as últimas treinadas. Com meio dia o problema não precisa
de remendo: a regra errada volta no dia seguinte, e nesse meio tempo outra
vencida é treinada. Interleaving é consequência da agenda, não uma exceção
colada em cima dela.

O rodízio continua existindo, mas rebaixado ao que ele sabe fazer: ele é o
único sinal disponível quando a regra foi **escolhida e ainda não respondida**
— sessão de drill aberta e abandonada não gera revisão, então a agenda não tem
o que dizer sobre ela. Virou termo de ordenação em vez de filtro (ver
``Scheduler.queue``).

## Uma sessão pode graduar uma regra

``GRADUATED_STREAK`` são três acertos **consecutivos**, não três sessões. Numa
sessão de drill o aluno responde a vários exercícios sobre a mesma regra, e três
acertos seguidos ali graduam. É deliberado e simétrico com o erro, que também
conta na hora: a agenda mede respostas, não dias de calendário.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from ..store.models import RuleCount, RuleSchedule
from ..store.repo import ScheduleRepo, SessionRepo

# A escada, em dias. Índice = degrau (``RuleSchedule.box``).
#
# O degrau 0 é reaprendizado: meio dia, não zero — ver o módulo. Daí em diante a
# escada é aproximadamente ×2,3, que é o espaçamento clássico de Leitner e o que
# põe o terceiro acerto numa semana.
#
# Os valores não são medidos, e isso está dito aqui porque o resto do projeto
# mede: não há histórico de revisão para calibrar escada nenhuma antes de a E13
# existir. É por isso que `rule_review` guarda `interval_days` na linha em vez
# de confiar nesta tupla — quando houver histórico, a escada se recalibra contra
# o que o aluno de fato recebeu.
INTERVALS_DAYS: tuple[float, ...] = (0.5, 1.0, 3.0, 7.0, 16.0, 35.0)

# Acertos consecutivos que tiram a regra da fila. Três é o que a §1 do plano da
# Fase 2 diz, e com esta escada corresponde a uma semana de folga.
GRADUATED_STREAK = 3

# Quantas regras recém-escolhidas a ordenação desempata para trás.
#
# Herdado da E6, onde era filtro. Aqui é só desempate, e serve ao caso que a
# agenda não cobre: regra escolhida numa sessão que não gerou resposta.
DRILL_ROTATION = 3

# O formato de data do esquema. É o de ``datetime('now')`` do SQLite, e não ISO
# 8601 com fuso, para que `due_at` possa ser comparado e ordenado no SQL junto
# com as outras colunas de data do banco.
STAMP = "%Y-%m-%d %H:%M:%S"


def stamp(moment: datetime) -> str:
    """O instante no formato do esquema, em UTC."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).strftime(STAMP)


def now() -> datetime:
    return datetime.now(UTC)


def interval_for(box: int) -> float:
    """O intervalo deste degrau, em dias. Degrau acima do teto fica no teto."""
    if box < 0:
        box = 0
    return INTERVALS_DAYS[min(box, len(INTERVALS_DAYS) - 1)]


def next_box(box: int, hit: bool) -> int:
    """O degrau depois desta revisão.

    Acerto sobe um e para no teto; erro volta ao zero — Leitner puro. O erro não
    desce "um degrau" de propósito: quem erra a regra que acertou três vezes não
    está um passo atrás, está reaprendendo.
    """
    if not hit:
        return 0
    return min(box + 1, len(INTERVALS_DAYS) - 1)


def is_graduated(schedule: RuleSchedule | None) -> bool:
    """A regra saiu da fila por acerto repetido?

    ``None`` é regra nunca revisada, que é o contrário de graduada.
    """
    return schedule is not None and schedule.streak >= GRADUATED_STREAK


@dataclass(frozen=True, slots=True)
class Review:
    """O resultado de uma revisão, com o próximo vencimento já calculado.

    Valor puro: ``grade`` o produz sem tocar no banco, e é o que permite testar
    a escada sem montar aluno, sessão e turno.
    """

    rule_id: str
    hit: bool
    source: str
    box_before: int
    box_after: int
    streak: int
    interval_days: float
    reviewed_at: str
    due_at: str

    @property
    def graduated(self) -> bool:
        return self.streak >= GRADUATED_STREAK

    def as_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "hit": self.hit,
            "source": self.source,
            "streak": self.streak,
            "interval_days": self.interval_days,
            "due_at": self.due_at,
            "graduated": self.graduated,
        }


def grade(
    rule_id: str,
    schedule: RuleSchedule | None,
    *,
    hit: bool,
    source: str = "drill",
    at: datetime | None = None,
) -> Review:
    """Aplica a escada a uma revisão. Função pura, sem banco e sem relógio oculto.

    ``schedule`` ``None`` é a primeira revisão desta regra: começa no degrau
    zero, que é onde um erro também a põe.
    """
    moment = at or now()
    box_before = schedule.box if schedule is not None else 0
    streak_before = schedule.streak if schedule is not None else 0

    box_after = next_box(box_before, hit)
    streak = streak_before + 1 if hit else 0
    days = interval_for(box_after)

    return Review(
        rule_id=rule_id,
        hit=hit,
        source=source,
        box_before=box_before,
        box_after=box_after,
        streak=streak,
        interval_days=days,
        reviewed_at=stamp(moment),
        due_at=stamp(moment + timedelta(days=days)),
    )


@dataclass(frozen=True, slots=True)
class Queued:
    """Uma candidata na fila do drill, com o motivo de estar onde está.

    O motivo vai junto porque a tela promete ao aluno uma afirmação verificável
    ("drilling: age with to be, you did it 3 times"), e porque uma fila cuja
    ordem não se explica é indepurável.
    """

    rule_id: str
    times: int
    due: bool
    schedule: RuleSchedule | None
    recently_drilled: bool = False

    @property
    def graduated(self) -> bool:
        return is_graduated(self.schedule)

    @property
    def streak(self) -> int:
        return self.schedule.streak if self.schedule is not None else 0

    @property
    def due_at(self) -> str:
        """Vazio para regra nunca revisada — e vazio ordena antes de qualquer data."""
        return self.schedule.due_at if self.schedule is not None else ""

    def as_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "times": self.times,
            "due": self.due,
            "streak": self.streak,
            "graduated": self.graduated,
            "due_at": self.due_at,
            "reviews": self.schedule.reviews if self.schedule is not None else 0,
            "lapses": self.schedule.lapses if self.schedule is not None else 0,
        }


class Scheduler:
    """Lê e escreve a agenda, e ordena a fila do drill.

    Não escolhe a regra: devolve a fila ordenada, e quem escolhe é o
    ``DrillPlanner`` — que ainda é quem sabe filtrar por base contrastiva e
    montar o alvo com as frases do aluno.
    """

    def __init__(self, conn: sqlite3.Connection, rotation: int = DRILL_ROTATION) -> None:
        self._schedules = ScheduleRepo(conn)
        self._sessions = SessionRepo(conn)
        self._rotation = rotation

    # --- a agenda ---------------------------------------------------------- #

    def review(
        self,
        student_id: int,
        rule_id: str,
        turn_id: int,
        *,
        hit: bool,
        source: str = "drill",
        at: datetime | None = None,
    ) -> Review:
        """Registra uma revisão e move a regra na agenda.

        Duas escritas e nenhuma transação, pela regra da conexão compartilhada;
        o evento vai primeiro porque é dele que o estado se reconstrói (ver
        ``ScheduleRepo``).
        """
        outcome = grade(
            rule_id,
            self._schedules.get(student_id, rule_id),
            hit=hit,
            source=source,
            at=at,
        )
        self._schedules.log_review(
            student_id,
            rule_id,
            turn_id,
            hit=outcome.hit,
            source=outcome.source,
            box_before=outcome.box_before,
            box_after=outcome.box_after,
            interval_days=outcome.interval_days,
        )
        self._schedules.save(
            student_id,
            rule_id,
            box=outcome.box_after,
            streak=outcome.streak,
            due_at=outcome.due_at,
            reviewed_at=outcome.reviewed_at,
            lapsed=not outcome.hit,
        )
        return outcome

    # --- a fila ------------------------------------------------------------ #

    def queue(
        self,
        student_id: int,
        candidates: list[RuleCount],
        at: datetime | None = None,
    ) -> list[Queued]:
        """As candidatas em ordem de urgência, a mais urgente primeiro.

        A chave de ordenação, e cada termo paga por si:

        1. **Vencida antes de não vencida.** É a agenda, e é o que esta entrega
           troca pela contagem: regra graduada não está vencida e vai para o fim
           da fila mesmo sendo a mais errada do histórico.
        2. **Não recém-escolhida antes de recém-escolhida.** O rodízio da E6,
           rebaixado a desempate. Cobre o que a agenda não vê: regra escolhida
           numa sessão que terminou sem resposta não tem revisão, então sem este
           termo ela seria escolhida outra vez.
        3. **Vencimento mais antigo primeiro**, com vazio na frente — regra
           nunca revisada vem antes de qualquer vencida, porque dela não há
           nenhuma evidência de aprendizado.
        4. **Mais frequente primeiro.** A contagem da E6, agora como último
           desempate: ela decide entre duas regras igualmente urgentes, e não
           mais qual é urgente.
        5. ``rule_id``, para a ordem ser determinística em teste.

        Nunca devolve lista vazia tendo candidata: fila sem nada vencido
        devolve a de vencimento mais próximo. Repetir a regra é melhor que não
        treinar nada — a mesma escolha que a E6 fez com o rodízio.
        """
        moment = stamp(at or now())
        schedules = self._schedules.for_student(student_id)
        recent = set(self._sessions.drilled_rules(student_id, limit=self._rotation))

        queued = [
            Queued(
                rule_id=count.rule_id,
                times=count.times,
                schedule=schedules.get(count.rule_id),
                due=self._is_due(schedules.get(count.rule_id), moment),
                recently_drilled=count.rule_id in recent,
            )
            for count in candidates
        ]
        queued.sort(
            key=lambda item: (
                not item.due,
                item.recently_drilled,
                item.due_at,
                -item.times,
                item.rule_id,
            )
        )
        return queued

    @staticmethod
    def _is_due(schedule: RuleSchedule | None, moment: str) -> bool:
        """Regra nunca revisada está sempre vencida: é a que mais precisa de treino."""
        if schedule is None:
            return True
        return schedule.due_at <= moment

    def due_count(
        self,
        student_id: int,
        candidates: list[RuleCount],
        at: datetime | None = None,
    ) -> int:
        """Quantas candidatas estão vencidas agora.

        Zero é resposta honesta e diferente de "não há o que treinar": o aluno
        acertou tudo o que devia e as regras voltam depois do intervalo.
        """
        return sum(1 for item in self.queue(student_id, candidates, at=at) if item.due)

    def schedule_for(self, student_id: int, rule_id: str) -> RuleSchedule | None:
        return self._schedules.get(student_id, rule_id)


__all__ = [
    "DRILL_ROTATION",
    "GRADUATED_STREAK",
    "INTERVALS_DAYS",
    "Queued",
    "Review",
    "Scheduler",
    "grade",
    "interval_for",
    "is_graduated",
    "next_box",
    "now",
    "stamp",
]
