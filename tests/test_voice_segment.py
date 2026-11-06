"""O montador de sentenças e a limpeza para fala (Fase 1.5).

Nenhum peso de modelo entra aqui, de propósito: é a parte da voz que decide
**o que** vai ser falado, e é onde o comportamento interessante mora. Os
motores são testados à parte, e em `-m integration`.
"""

from __future__ import annotations

import pytest

from teacher_charles.voice.segment import (
    MIN_SENTENCE_CHARS,
    SentenceAssembler,
    for_speech,
    split_sentences,
)


class TestForSpeech:
    """A marcação que a persona usa na tela não pode chegar ao sintetizador."""

    def test_remove_enfase_de_markdown(self):
        # A persona usa *to be* para destacar a estrutura. Na voz, o Kokoro
        # leria os asteriscos.
        assert for_speech("age uses *to be*") == "age uses to be"
        assert for_speech("you said **wrong**") == "you said wrong"
        assert for_speech("the word `have`") == "the word have"

    def test_remove_marca_de_citacao(self):
        # O exemplo de correção da persona vem como blockquote.
        assert for_speech("> You said this.") == "You said this."

    def test_preserva_as_aspas(self):
        # As aspas são a correção: tirá-las apagaria a distinção entre o que o
        # aluno disse e o que o professor propõe.
        assert for_speech('You said "I have 30 years."') == 'You said "I have 30 years."'

    def test_junta_linhas_numa_so(self):
        assert for_speech("first line\n\nsecond line") == "first line second line"

    def test_texto_sem_marcacao_atravessa_intacto(self):
        assert for_speech("Good morning, how are you?") == "Good morning, how are you?"


class TestSplitSentences:
    def test_divide_nas_tres_pontuacoes(self):
        assert split_sentences("Good morning! How are you? I am fine today.") == [
            "Good morning!",
            "How are you?",
            "I am fine today.",
        ]

    def test_ponto_dentro_das_aspas_fecha_a_sentenca(self):
        # `You said "I have 30 years."` é UMA sentença, e o ponto dela está
        # dentro das aspas. Cortar antes da aspa deixaria a aspa órfã na
        # sentença seguinte.
        assert split_sentences('You said "I have 30 years." So, tell me more about it.') == [
            'You said "I have 30 years."',
            "So, tell me more about it.",
        ]

    @pytest.mark.parametrize(
        "text",
        [
            "Mr. Silva is my boss and I like him.",
            "It takes 3.5 hours to get there by car.",
            "J. R. Tolkien wrote that book a long time ago.",
            "Bring a coat, an umbrella, etc. and we can go.",
        ],
    )
    def test_nao_divide_em_abreviacao_decimal_nem_inicial(self, text):
        assert split_sentences(text) == [for_speech(text)]

    def test_sentenca_curta_sai_junto_com_a_seguinte(self):
        # Uma chamada de síntese para 300 ms de áudio gastaria o custo fixo
        # inteiro, e soaria picada na transição.
        assert split_sentences("Oh! That is interesting news.") == ["Oh! That is interesting news."]

    def test_sentenca_curta_mas_acima_do_minimo_sai_sozinha(self):
        assert len("Good morning!") >= MIN_SENTENCE_CHARS
        assert split_sentences("Good morning! Tell me about work.") == [
            "Good morning!",
            "Tell me about work.",
        ]

    def test_texto_sem_pontuacao_vira_uma_sentenca(self):
        assert split_sentences("no punctuation at all here") == ["no punctuation at all here"]

    def test_texto_vazio_nao_rende_sentenca(self):
        assert split_sentences("") == []
        assert split_sentences("   \n  ") == []


class TestSentenceAssembler:
    """O caminho do turno: pedaços chegando, sentenças saindo."""

    @staticmethod
    def drain(pieces: list[str]) -> tuple[list[str], str | None]:
        assembler = SentenceAssembler()
        out: list[str] = []
        for piece in pieces:
            out += assembler.push(piece)
        return out, assembler.flush()

    def test_sentenca_sai_assim_que_fecha_e_nao_no_fim(self):
        assembler = SentenceAssembler()
        assert assembler.push("Good morning! ") == ["Good morning!"]
        # A seguinte ainda não fechou: nada sai.
        assert assembler.push("How was your") == []

    def test_o_fim_do_pedaco_tambem_fecha_sentenca(self):
        # Escolha de latência: esperar o espaço custaria um token de atraso em
        # toda sentença, no único ponto do turno em que o aluno espera calado.
        assembler = SentenceAssembler()
        assert assembler.push("I am doing fine.") == ["I am doing fine."]

    def test_digito_antes_do_ponto_espera_o_pedaco_seguinte(self):
        # O único caso indecidível: `It takes 3.` pode ser fim de sentença ou
        # o começo de `3.5 hours`. Esse espera; os outros não.
        assembler = SentenceAssembler()
        assert assembler.push("The meeting takes 3.") == []
        assert assembler.push("5 hours in total. ") == ["The meeting takes 3.5 hours in total."]

    def test_digito_antes_do_ponto_no_fim_do_turno_sai_no_flush(self):
        # Esperar não pode virar engolir: se o professor parou ali, a sentença
        # sai pelo flush.
        assembler = SentenceAssembler()
        assert assembler.push("We talked for about 2.") == []
        assert assembler.flush() == "We talked for about 2."

    def test_pedacos_que_cortam_no_meio_da_palavra(self):
        got, tail = self.drain(["I am ", "doin", "g fine today. ", "And y", "ou?"])
        assert got == ["I am doing fine today."]
        assert tail == "And you?"

    def test_flush_devolve_o_rabo_sem_pontuacao(self):
        got, tail = self.drain(["Tell me about your week"])
        assert got == []
        assert tail == "Tell me about your week"

    def test_flush_esvazia_o_buffer(self):
        assembler = SentenceAssembler()
        assembler.push("something left over")
        assert assembler.flush() == "something left over"
        assert assembler.flush() is None

    def test_pending_mostra_o_que_ainda_nao_fechou(self):
        assembler = SentenceAssembler()
        assembler.push("Good morning! How was")
        assert assembler.pending == "How was"

    def test_um_pedaco_pode_fechar_duas_sentencas(self):
        # O provider real entrega pedaços de tamanho arbitrário; dois fins de
        # sentença podem chegar juntos.
        assembler = SentenceAssembler()
        assert assembler.push("That is good news. I am happy for you. ") == [
            "That is good news.",
            "I am happy for you.",
        ]

    def test_markdown_e_limpo_na_saida_do_montador(self):
        # A limpeza acontece na borda da fala, e o histórico gravado continua
        # sendo o que o modelo escreveu.
        assembler = SentenceAssembler()
        assert assembler.push("Age uses *to be* in English. ") == ["Age uses to be in English."]

    def test_pedaco_vazio_nao_faz_nada(self):
        assembler = SentenceAssembler()
        assert assembler.push("") == []
        assert assembler.pending == ""

    def test_resposta_tipica_do_professor_em_pedacos_de_quatro(self):
        # O formato que a persona pede, fatiado como o provider fatia.
        reply = (
            'You said "I have 30 years." In English, age uses *to be*: '
            '"I am 30 years old." So, what did you do on your birthday?'
        )
        pieces = [reply[i : i + 4] for i in range(0, len(reply), 4)]
        got, tail = self.drain(pieces)
        assert got == [
            'You said "I have 30 years."',
            'In English, age uses to be: "I am 30 years old."',
            "So, what did you do on your birthday?",
        ]
        # A última fechou no fim do último pedaço, então não sobrou rabo.
        assert tail is None
