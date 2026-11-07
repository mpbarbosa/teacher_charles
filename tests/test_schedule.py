"""Testes da repetição espaçada (E13). Nenhum usa modelo de verdade.

O critério de pronto da E13 tem três metades, e os testes estão agrupados por
elas:

1. **A regra do drill vem do agendamento e não da contagem.** É a troca que a
   entrega faz, e a que mais fácil se desfaz sem ninguém notar: basta alguém
   "consertar" a ordenação de ``Scheduler.queue`` para a fila voltar a ser um
   ranking de frequência. Ver `test_a_fila_e_a_agenda_e_nao_a_contagem`.
2. **Regra acertada n vezes seguidas sai da fila e volta depois do intervalo.**
3. **Um teste mede o intervalo.** É `test_a_escada_mede_os_intervalos_em_dias`,
   e ele mede o que o aluno recebe (a distância entre revisão e vencimento),
   não a constante — a constante medindo a si mesma não provaria nada.

E uma quarta coisa, que não está no critério mas é a decisão de desenho desta
entrega: a **assimetria do sinal**. Turno limpo só é acerto no drill. Os testes
de `--- O que conta como acerto ---` existem para que ninguém a remova por
simetria aparente.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from teacher_charles.config import REPO_ROOT
from teacher_charles.llm import FakeProvider
from teacher_charles.prompts import PromptLibrary
from teacher_charles.store import db
from teacher_charles.store.models import RuleSchedule
from teacher_charles.store.repo import (
    CorrectionRepo,
    ScheduleRepo,
    SessionRepo,
    StudentRepo,
    TurnRepo,
)
from teacher_charles.teaching import Analyzer, TurnRunner
from teacher_charles.teaching.contrastive import load_base
from teacher_charles.teaching.modes import DrillPlanner, drill_block
from teacher_charles.teaching.schedule import (
    GRADUATED_STREAK,
    INTERVALS_DAYS,
    STAMP,
    Scheduler,
    grade,
    interval_for,
    is_graduated,
    next_box,
    stamp,
)

AGE_RULE = "calque.age-have"
QUESTION_RULE = "calque.make-question"
DEPEND_RULE = "calque.depend-of"

# Relógio fixo. A agenda é feita de datas, e teste de data com relógio de
# verdade falha sozinho uma vez por ano.
AT = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def base():
    return load_base(REPO_ROOT / "data" / "contrastive")


@pytest.fixture
def conn(tmp_path):
    connection = db.init_db(tmp_path / "schedule.db")
    yield connection
    connection.close()


@pytest.fixture
def student(conn):
    return StudentRepo(conn).get_or_create("ana", cefr_level="B1")


@pytest.fixture
def real_prompts():
    return PromptLibrary(REPO_ROOT / "prompts")


def record_error(conn, student_id: int, rule_id: str, span: str = "I have 35 years") -> int:
    """Um erro real no histórico: turno do aluno, correção com regra curada."""
    session_id = SessionRepo(conn).current_or_start(student_id)
    turn_id = TurnRepo(conn).append(session_id, "student", span)
    CorrectionRepo(conn).add(
        turn_id,
        span=span,
        corrected="I am 35 years old",
        error_type="calque",
        explanation="curada",
        rule_id=rule_id,
    )
    return turn_id


def a_turn(conn, student_id: int) -> int:
    session_id = SessionRepo(conn).current_or_start(student_id)
    return TurnRepo(conn).append(session_id, "student", "whatever the student wrote")


def days_between(earlier: str, later: str) -> float:
    """Quantos dias separam dois instantes do esquema."""
    start = datetime.strptime(earlier, STAMP).replace(tzinfo=UTC)
    end = datetime.strptime(later, STAMP).replace(tzinfo=UTC)
    return (end - start).total_seconds() / 86400


def as_schedule(review, student_id: int = 1) -> RuleSchedule:
    """O estado que esta revisão deixaria, para alimentar a próxima."""
    return RuleSchedule(
        student_id=student_id,
        rule_id=review.rule_id,
        box=review.box_after,
        streak=review.streak,
        reviews=1,
        lapses=0 if review.hit else 1,
        due_at=review.due_at,
        last_review_at=review.reviewed_at,
    )


# --- A escada de intervalos ------------------------------------------------ #


def test_a_escada_mede_os_intervalos_em_dias():
    """O terceiro critério de pronto da E13: um teste mede o intervalo.

    E mede o que o aluno **recebe** — a distância entre a revisão e o
    vencimento, lida das duas datas —, não o valor da constante. Afirmar
    ``INTERVALS_DAYS[2] == 3.0`` seria a constante medindo a si mesma.

    Note onde a escada começa na prática. Um acerto a partir do zero já sobe
    para o degrau 1, então a sequência de acertos é ``INTERVALS_DAYS[1:]``. O
    degrau 0 (meio dia) é o de reaprendizado, e só o erro leva até ele — é o que
    o teste do lapso abaixo mede.
    """
    review = grade(AGE_RULE, None, hit=True, at=AT)
    assert days_between(review.reviewed_at, review.due_at) == 1.0

    received = [days_between(review.reviewed_at, review.due_at)]
    moment = AT
    for _ in range(len(INTERVALS_DAYS) + 1):
        moment = moment + timedelta(days=review.interval_days)
        review = grade(AGE_RULE, as_schedule(review), hit=True, at=moment)
        received.append(days_between(review.reviewed_at, review.due_at))

    esperado = list(INTERVALS_DAYS[1:])
    # O teto da escada não some: quem continua acertando continua recebendo o
    # maior intervalo, em vez de sair da agenda para sempre.
    esperado += [INTERVALS_DAYS[-1]] * (len(received) - len(esperado))

    assert received == esperado


def test_o_lapso_cai_no_degrau_de_reaprendizado_e_nao_no_zero():
    """Meio dia, e não "vence agora" — é isto que substitui o rodízio da E6.

    Com intervalo zero a regra recém-errada voltaria a ser a mais vencida da
    fila na sessão seguinte, para sempre, e o aluno nunca veria a segunda regra.
    Meio dia a traz de volta amanhã e deixa outra vencida ser treinada hoje.
    """
    review = grade(AGE_RULE, None, hit=False, at=AT)

    assert review.box_after == 0
    assert days_between(review.reviewed_at, review.due_at) == INTERVALS_DAYS[0] == 0.5
    assert days_between(review.reviewed_at, review.due_at) > 0, (
        "intervalo zero traria a mesma regra para sempre"
    )


def test_o_erro_volta_ao_degrau_zero_e_nao_um_degrau_atras():
    """Quem erra a regra que acertou três vezes está reaprendendo, não um passo atrás."""
    graduada = RuleSchedule(
        student_id=1,
        rule_id=AGE_RULE,
        box=3,
        streak=3,
        reviews=3,
        lapses=0,
        due_at=stamp(AT),
        last_review_at=stamp(AT),
    )

    review = grade(AGE_RULE, graduada, hit=False, at=AT)

    assert review.box_after == 0
    assert review.streak == 0
    assert not review.graduated


def test_o_teto_da_escada_nao_estoura():
    assert next_box(len(INTERVALS_DAYS) - 1, hit=True) == len(INTERVALS_DAYS) - 1
    assert interval_for(len(INTERVALS_DAYS) + 10) == INTERVALS_DAYS[-1]
    assert interval_for(-1) == INTERVALS_DAYS[0]


# --- Sair da fila ---------------------------------------------------------- #


def test_tres_acertos_seguidos_graduam_a_regra():
    """O segundo critério de pronto: n acertos seguidos e a regra sai da fila."""
    review = grade(AGE_RULE, None, hit=True, at=AT)
    for _ in range(GRADUATED_STREAK - 1):
        assert not review.graduated
        review = grade(AGE_RULE, as_schedule(review), hit=True, at=AT)

    assert review.streak == GRADUATED_STREAK
    assert review.graduated
    assert days_between(review.reviewed_at, review.due_at) == 7.0, (
        "com esta escada, três acertos valem uma semana de folga"
    )


def test_um_acerto_isolado_nao_gradua():
    """Simétrico com o nível (E5), onde nenhum turno sozinho move o aluno."""
    assert not grade(AGE_RULE, None, hit=True, at=AT).graduated


def test_regra_nunca_revisada_nao_esta_graduada():
    assert not is_graduated(None)


async def test_a_graduada_sai_da_fila_e_volta_depois_do_intervalo(conn, student, base):
    """As duas metades do segundo critério, medidas na fila de verdade."""
    record_error(conn, student.id, AGE_RULE)
    scheduler = Scheduler(conn)
    planner = DrillPlanner(conn, base)

    review = None
    for _ in range(GRADUATED_STREAK):
        review = scheduler.review(student.id, AGE_RULE, a_turn(conn, student.id), hit=True, at=AT)
    assert review.graduated

    # Logo depois de graduar: na fila, mas vencida não.
    fila = planner.queue(student.id, at=AT)
    assert [item.rule_id for item in fila] == [AGE_RULE]
    assert fila[0].due is False
    assert fila[0].graduated is True
    assert planner.due_count(student.id, at=AT) == 0

    # Um dia antes do vencimento, continua fora.
    assert planner.due_count(student.id, at=AT + timedelta(days=6)) == 0

    # Passado o intervalo, volta — "sai da fila" é folga, não exclusão.
    depois = planner.queue(student.id, at=AT + timedelta(days=8))
    assert depois[0].due is True


async def test_nada_vencido_ainda_devolve_a_mais_proxima(conn, student, base):
    """Fila em dia não é fila vazia: repetir é melhor que não treinar nada.

    A mesma escolha que a E6 fez com o rodízio ("preferência, não impedimento"),
    e a razão de ``pick`` não devolver ``None`` aqui: ``None`` significa "não há
    o que treinar", e isso é outra coisa.
    """
    record_error(conn, student.id, AGE_RULE)
    Scheduler(conn).review(student.id, AGE_RULE, a_turn(conn, student.id), hit=True, at=AT)

    planner = DrillPlanner(conn, base)
    assert planner.due_count(student.id, at=AT) == 0
    target = planner.pick(student.id, at=AT)
    assert target is not None and target.rule_id == AGE_RULE


# --- A fila é a agenda, não a contagem ------------------------------------- #


async def test_a_fila_e_a_agenda_e_nao_a_contagem(conn, student, base):
    """O primeiro critério de pronto da E13, e o teste central da entrega.

    A regra que o aluno mais errou está graduada; a que ele errou uma vez nunca
    foi treinada. A contagem da E6 devolveria a primeira — ela tem cinco erros
    contra um — e devolveria **para sempre**, porque frequência é um número que
    só sobe. A agenda devolve a segunda.
    """
    for _ in range(5):
        record_error(conn, student.id, AGE_RULE)
    record_error(conn, student.id, QUESTION_RULE, span="make a question")

    scheduler = Scheduler(conn)
    for _ in range(GRADUATED_STREAK):
        scheduler.review(student.id, AGE_RULE, a_turn(conn, student.id), hit=True, at=AT)

    planner = DrillPlanner(conn, base)
    contagem = planner.candidates(student.id)
    assert contagem[0].rule_id == AGE_RULE, "a contagem continua apontando a mais errada"

    target = planner.pick(student.id, at=AT)
    assert target is not None
    assert target.rule_id == QUESTION_RULE, "quem ordena a fila é a agenda, não a contagem"


async def test_a_nunca_revisada_vem_antes_da_vencida(conn, student, base):
    """Vencida é regra com evidência; nunca revisada não tem nenhuma.

    Cai do ``due_at`` vazio, que ordena antes de qualquer data — e é por isso que
    ``ScheduleRepo.get`` devolve ``None`` em vez de um agendamento zerado.
    """
    record_error(conn, student.id, AGE_RULE)
    record_error(conn, student.id, QUESTION_RULE, span="make a question")
    # A regra do meio errou e já venceu há muito; a outra nunca foi treinada.
    Scheduler(conn).review(
        student.id, AGE_RULE, a_turn(conn, student.id), hit=False, at=AT - timedelta(days=30)
    )

    fila = DrillPlanner(conn, base).queue(student.id, at=AT)

    assert [item.rule_id for item in fila] == [QUESTION_RULE, AGE_RULE]
    assert all(item.due for item in fila)


async def test_a_contagem_desempata_entre_igualmente_urgentes(conn, student, base):
    """A frequência da E6 não foi jogada fora: foi rebaixada a último desempate."""
    record_error(conn, student.id, QUESTION_RULE, span="make a question")
    for _ in range(3):
        record_error(conn, student.id, AGE_RULE)

    fila = DrillPlanner(conn, base).queue(student.id)

    assert all(item.due and item.schedule is None for item in fila)
    assert [item.rule_id for item in fila] == [AGE_RULE, QUESTION_RULE]


async def test_o_rodizio_desempata_o_que_a_agenda_nao_ve(conn, student, base):
    """Sessão de drill escolhida e abandonada não gera revisão.

    É o único caso que a agenda não cobre, e a razão de ``DRILL_ROTATION`` ter
    sobrevivido à E13 — rebaixado de filtro a termo de ordenação. Sem ele a
    mesma regra seria escolhida outra vez, porque nada aconteceu com ela.
    """
    for _ in range(3):
        record_error(conn, student.id, AGE_RULE)
    record_error(conn, student.id, QUESTION_RULE, span="make a question")

    planner = DrillPlanner(conn, base)
    sessions = SessionRepo(conn)

    first = planner.pick(student.id)
    assert first is not None and first.rule_id == AGE_RULE
    sessions.switch_mode(student.id, "drill", first.rule_id)

    second = planner.pick(student.id)
    assert second is not None
    assert second.rule_id == QUESTION_RULE


async def test_a_fila_e_por_aluno(conn, base):
    """A agenda de um aluno não mexe na fila de outro."""
    students = StudentRepo(conn)
    ana = students.get_or_create("ana")
    joao = students.get_or_create("joao")
    record_error(conn, ana.id, AGE_RULE)
    record_error(conn, joao.id, AGE_RULE)

    scheduler = Scheduler(conn)
    for _ in range(GRADUATED_STREAK):
        scheduler.review(ana.id, AGE_RULE, a_turn(conn, ana.id), hit=True, at=AT)

    planner = DrillPlanner(conn, base)
    assert planner.due_count(ana.id, at=AT) == 0
    assert planner.due_count(joao.id, at=AT) == 1


async def test_sem_candidata_a_fila_e_vazia_e_pick_e_none(conn, student, base):
    planner = DrillPlanner(conn, base)
    assert planner.queue(student.id) == []
    assert planner.pick(student.id) is None
    assert planner.due_count(student.id) == 0


# --- O que conta como acerto ---------------------------------------------- #


def runner_for(conn, prompts, base, planner, errors):
    """Um turno com analisador de dublê que devolve sempre os ``errors`` dados."""
    provider = FakeProvider(
        replies="Sure!",
        replies_by_system={"You find mistakes": json.dumps({"errors": errors})},
    )
    return TurnRunner(
        provider=provider,
        prompts=prompts,
        conn=conn,
        analyzer=Analyzer(provider=provider, prompts=prompts, base=base),
        planner=planner,
    )


async def drill_session(conn, student, base, rule_id: str) -> int:
    return SessionRepo(conn).switch_mode(student.id, "drill", rule_id)


async def test_turno_de_drill_sem_correcao_e_acerto(conn, student, base, real_prompts):
    """A ausência de correção é evidência de acerto **aqui**, e só aqui.

    Porque a oportunidade foi construída: o exercício pediu a estrutura e
    esperou a resposta.
    """
    record_error(conn, student.id, AGE_RULE)
    planner = DrillPlanner(conn, base)
    session_id = await drill_session(conn, student, base, AGE_RULE)

    runner = runner_for(conn, real_prompts, base, planner, errors=[])
    async for _ in runner.run(session_id, "I am 35 years old.", student):
        pass

    schedule = ScheduleRepo(conn).get(student.id, AGE_RULE)
    assert schedule is not None
    assert schedule.streak == 1
    assert schedule.lapses == 0


async def test_turno_de_conversa_sem_correcao_nao_e_acerto(conn, student, base, real_prompts):
    """A assimetria do sinal, e o teste que impede alguém de "simetrizá-la".

    Turno limpo em conversa livre não prova acerto: prova que o aluno não usou a
    estrutura, ou usou e ninguém reparou. Contar isso graduaria regras pelo
    silêncio — precisão inventada, a mesma classe de erro que a §3.4 do plano da
    Fase 2 proíbe no retorno de pronúncia.
    """
    record_error(conn, student.id, AGE_RULE)
    planner = DrillPlanner(conn, base)
    session_id = SessionRepo(conn).switch_mode(student.id, "conversation")

    runner = runner_for(conn, real_prompts, base, planner, errors=[])
    async for _ in runner.run(session_id, "The weather is nice today.", student):
        pass

    assert ScheduleRepo(conn).get(student.id, AGE_RULE) is None
    assert ScheduleRepo(conn).count() == 0


async def test_o_erro_conta_em_qualquer_modo(conn, student, base, real_prompts):
    """ "O que ele erra volta" (§1 do plano) não é só no drill."""
    record_error(conn, student.id, AGE_RULE)
    planner = DrillPlanner(conn, base)
    session_id = SessionRepo(conn).switch_mode(student.id, "conversation")

    runner = runner_for(
        conn,
        real_prompts,
        base,
        planner,
        errors=[
            {
                "span": "I have 35 years",
                "type": "calque",
                "correction": "I am 35 years old",
                "rule_id": AGE_RULE,
                "explanation": "ignorada: a curada vence",
            }
        ],
    )
    async for _ in runner.run(session_id, "I have 35 years.", student):
        pass

    schedule = ScheduleRepo(conn).get(student.id, AGE_RULE)
    assert schedule is not None
    assert schedule.streak == 0
    assert schedule.lapses == 1

    (review,) = ScheduleRepo(conn).reviews(student.id, AGE_RULE)
    assert review.hit is False
    assert review.source == "lapse", "fora do drill, o erro entra como lapso"


async def test_errar_a_regra_em_treino_zera_o_acerto(conn, student, base, real_prompts):
    """Quem erra a regra do drill no próprio drill errou — e não ganha acerto junto."""
    record_error(conn, student.id, AGE_RULE)
    scheduler = Scheduler(conn)
    scheduler.review(student.id, AGE_RULE, a_turn(conn, student.id), hit=True, at=AT)
    scheduler.review(student.id, AGE_RULE, a_turn(conn, student.id), hit=True, at=AT)

    planner = DrillPlanner(conn, base)
    session_id = await drill_session(conn, student, base, AGE_RULE)
    runner = runner_for(
        conn,
        real_prompts,
        base,
        planner,
        errors=[
            {
                "span": "I have 35 years",
                "type": "calque",
                "correction": "I am 35 years old",
                "rule_id": AGE_RULE,
                "explanation": "ignorada",
            }
        ],
    )
    async for _ in runner.run(session_id, "I have 35 years.", student):
        pass

    schedule = ScheduleRepo(conn).get(student.id, AGE_RULE)
    assert schedule.streak == 0
    assert schedule.box == 0
    assert schedule.reviews == 3

    ultima = ScheduleRepo(conn).reviews(student.id, AGE_RULE)[0]
    assert ultima.hit is False
    assert ultima.source == "drill", "no drill, o erro da regra em treino é revisão do drill"


async def test_sem_analise_nao_ha_revisao(conn, student, base, real_prompts):
    """A guarda que importa mais: análise que não rodou não gradua ninguém.

    Acerto no drill é a *ausência* de correção, então analisador ausente ou
    estourado produz a mesma ausência. Contá-la graduaria a regra por falha de
    infraestrutura — e a §3.3 do plano da Fase 2 já decidiu o que fazer quando o
    avaliador não está entendendo: calar.
    """
    record_error(conn, student.id, AGE_RULE)
    planner = DrillPlanner(conn, base)
    session_id = await drill_session(conn, student, base, AGE_RULE)

    # Sem `analyzer`: o turno da E2, conversa e nada mais.
    runner = TurnRunner(
        provider=FakeProvider(replies="Sure!"),
        prompts=real_prompts,
        conn=conn,
        planner=planner,
    )
    async for _ in runner.run(session_id, "I am 35 years old.", student):
        pass

    assert ScheduleRepo(conn).count() == 0
    assert ScheduleRepo(conn).get(student.id, AGE_RULE) is None


async def test_uma_resposta_vazia_de_conteudo_conta_como_acerto(conn, student, base, real_prompts):
    """O ponto cego conhecido da E13, pinado aqui em vez de escondido.

    "yes" não comete o calque, então nenhum padrão da base casa, então o turno
    conta como acerto. O aluno que responde qualquer coisa três vezes gradua a
    regra sem ter usado a estrutura uma única vez.

    Não há guarda determinística para isto hoje: o analisador devolve erros, não
    "o aluno usou a estrutura", e qualquer piso que eu inventasse aqui (contagem
    de palavras, lista de respostas triviais) seria número sem medição — o que
    esta base de código evita de propósito.

    O que existe é o dado para resolver depois: ``rule_review.turn_id`` aponta
    para o turno, então dá para voltar e ver o que o aluno de fato escreveu em
    cada acerto registrado. Está anotado como dívida na §6 do plano da Fase 2.
    Este teste falha de propósito no dia em que alguém puser a guarda — e aí é
    para trocá-lo, não para remendar o código.
    """
    record_error(conn, student.id, AGE_RULE)
    planner = DrillPlanner(conn, base)
    session_id = await drill_session(conn, student, base, AGE_RULE)

    runner = runner_for(conn, real_prompts, base, planner, errors=[])
    async for _ in runner.run(session_id, "yes", student):
        pass

    schedule = ScheduleRepo(conn).get(student.id, AGE_RULE)
    assert schedule is not None and schedule.streak == 1


async def test_correcao_sem_regra_curada_nao_entra_na_agenda(conn, student, base, real_prompts):
    """Mesma política da E6 e da §5.2: sem regra curada não há o que agendar."""
    planner = DrillPlanner(conn, base)
    session_id = SessionRepo(conn).switch_mode(student.id, "conversation")
    runner = runner_for(
        conn,
        real_prompts,
        base,
        planner,
        errors=[
            {
                "span": "There have been",
                "type": "grammar",
                "correction": "There has been",
                "rule_id": None,
                "explanation": "inventada pelo modelo",
            }
        ],
    )
    async for _ in runner.run(session_id, "There have been many changes.", student):
        pass

    assert ScheduleRepo(conn).count() == 0


# --- O que o aluno vê ----------------------------------------------------- #


def test_o_bloco_do_drill_conta_o_acerto_ao_modelo(base):
    """Metade do valor de repetição espaçada é o aluno saber que está melhorando."""
    from teacher_charles.teaching.modes import DrillTarget

    schedule = RuleSchedule(
        student_id=1,
        rule_id=AGE_RULE,
        box=2,
        streak=2,
        reviews=2,
        lapses=0,
        due_at=stamp(AT),
        last_review_at=stamp(AT),
    )
    target = DrillTarget(item=base.get(AGE_RULE), times=3, schedule=schedule)

    block = drill_block(target)

    assert "2 times in a row" in block
    assert target.as_dict()["streak"] == 2
    assert target.as_dict()["graduated"] is False


def test_o_bloco_nao_inventa_acerto_que_nao_houve(base):
    from teacher_charles.teaching.modes import DrillTarget

    block = drill_block(DrillTarget(item=base.get(AGE_RULE), times=3))

    assert "in a row" not in block
    assert "3 times" in block, "a contagem de erros continua lá"


# --- O repositório -------------------------------------------------------- #


def test_o_estado_acumula_revisoes_e_lapsos(conn, student):
    """``reviews`` e ``lapses`` somam no SQL, não em Python.

    Ler para depois escrever seriam duas instruções nesta conexão
    compartilhada, e duas requisições podem se intercalar entre elas.
    """
    schedules = ScheduleRepo(conn)
    for box, lapsed in ((1, False), (0, True), (1, False), (0, True)):
        schedules.save(
            student.id,
            AGE_RULE,
            box=box,
            streak=0 if lapsed else 1,
            due_at=stamp(AT),
            reviewed_at=stamp(AT),
            lapsed=lapsed,
        )

    schedule = schedules.get(student.id, AGE_RULE)
    assert schedule.reviews == 4
    assert schedule.lapses == 2


def test_a_revisao_guarda_o_intervalo_que_o_aluno_recebeu(conn, student):
    """A escada é constante de código; a linha é o registro histórico.

    É o que permitirá recalibrá-la depois contra o que o aluno de fato recebeu,
    em vez de contra a constante de hoje — mesmo motivo de ``level_signal``
    guardar os números que produziram a estimativa.
    """
    turn_id = a_turn(conn, student.id)
    review = Scheduler(conn).review(student.id, AGE_RULE, turn_id, hit=True, at=AT)

    (row,) = ScheduleRepo(conn).reviews(student.id, AGE_RULE)
    assert row.turn_id == turn_id
    assert row.hit is True
    assert row.box_before == 0
    assert row.box_after == 1
    assert row.interval_days == review.interval_days == 1.0


def test_origem_de_revisao_invalida_e_recusada(conn, student):
    with pytest.raises(ValueError, match="origem de revisão inválida"):
        ScheduleRepo(conn).log_review(
            student.id,
            AGE_RULE,
            a_turn(conn, student.id),
            hit=True,
            source="telepatia",
            box_before=0,
            box_after=1,
            interval_days=1.0,
        )


def test_a_agenda_e_apagada_com_o_aluno(conn, student):
    """``ON DELETE CASCADE`` só vale com ``PRAGMA foreign_keys``, que ``connect`` liga."""
    Scheduler(conn).review(student.id, AGE_RULE, a_turn(conn, student.id), hit=True, at=AT)
    assert ScheduleRepo(conn).count() == 1

    conn.execute("DELETE FROM student WHERE id = ?", (student.id,))

    assert ScheduleRepo(conn).count() == 0
    assert ScheduleRepo(conn).get(student.id, AGE_RULE) is None


def test_o_vencimento_ordena_como_texto(conn, student):
    """``due_at`` usa o formato do SQLite de propósito: a fila compara strings.

    ISO 8601 com fuso ("2026-01-01T12:00:00+00:00") quebraria a comparação
    lexicográfica contra as outras colunas de data do banco, que vêm de
    ``datetime('now')``.
    """
    cedo = stamp(AT)
    tarde = stamp(AT + timedelta(days=30))
    assert cedo < tarde
    assert stamp(AT) == "2026-01-01 12:00:00"
    assert "" < cedo, "regra nunca revisada ordena antes de qualquer vencimento"
