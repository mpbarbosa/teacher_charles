"""Reading a subtitle file into cues and utterances.

Every fixture here is dialogue written for this test file. None of it comes
from the material in `corpus/`, and that is not squeamishness: `corpus/` is
ignored precisely so that somebody else's dialogue stays out of a public
repository, and a test fixture is committed like any other file. A reader for
third-party text whose own tests smuggle that text into git would defeat the
convention it was built to serve.
"""

from __future__ import annotations

from teacher_charles.subtitles import (
    MAX_CONTINUATION_GAP_MS,
    parse_srt,
    read_srt,
    utterances,
)


def srt(*blocks: str) -> str:
    return "\n\n".join(block.strip("\n") for block in blocks) + "\n"


class TestParsing:
    """The file as the file has it, before any reassembly."""

    def test_reads_index_timing_and_text(self):
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:03,500
Could you pass me the salt?
"""
            )
        )
        assert len(cues) == 1
        assert cues[0].index == 1
        assert cues[0].start_ms == 1_000
        assert cues[0].end_ms == 3_500
        assert cues[0].text == "Could you pass me the salt?"
        assert cues[0].is_speech

    def test_joins_the_lines_of_one_cue(self):
        # A cue breaks where the screen runs out, not where the sentence does.
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:03,000
I have been waiting here
since eight this morning.
"""
            )
        )
        assert cues[0].text == "I have been waiting here since eight this morning."

    def test_tolerates_bom_and_crlf(self):
        # What a release actually ships, and what the import normalises away.
        raw = "﻿1\r\n00:00:01,000 --> 00:00:02,000\r\nGood morning.\r\n"
        cues = parse_srt(raw)
        assert len(cues) == 1
        assert cues[0].text == "Good morning."

    def test_tolerates_a_dot_before_the_milliseconds(self):
        cues = parse_srt(
            srt(
                """
1
00:00:01.250 --> 00:00:02.750
It is nearly ready.
"""
            )
        )
        assert cues[0].start_ms == 1_250
        assert cues[0].end_ms == 2_750

    def test_pads_a_short_fraction(self):
        # `.5` is half a second, not five milliseconds.
        cues = parse_srt(
            srt(
                """
1
00:00:01,5 --> 00:00:02,25
Almost there.
"""
            )
        )
        assert cues[0].start_ms == 1_500
        assert cues[0].end_ms == 2_250

    def test_survives_a_missing_cue_number(self):
        cues = parse_srt(
            srt(
                """
00:00:01,000 --> 00:00:02,000
Nobody numbered this one.
"""
            )
        )
        assert len(cues) == 1
        assert cues[0].index == 1

    def test_a_blank_line_inside_a_cue_does_not_truncate_it(self):
        # The timecode is the anchor, so the text after a stray blank line is
        # still this cue's text. Splitting on blank lines would lose it.
        cues = parse_srt("1\n00:00:01,000 --> 00:00:04,000\nFirst half of it\n\nand the rest.\n")
        assert len(cues) == 1
        assert cues[0].text == "First half of it and the rest."

    def test_strips_markup_and_override_blocks(self):
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:02,000
{\\an8}<i>Please</i> sit <b>down</b>.
"""
            )
        )
        assert cues[0].text == "Please sit down."

    def test_keeps_a_brace_that_is_not_an_override(self):
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:02,000
Type {name} in the box.
"""
            )
        )
        assert "{name}" in cues[0].text

    def test_an_empty_cue_is_dropped(self):
        cues = parse_srt("1\n00:00:01,000 --> 00:00:02,000\n\n")
        assert cues == []

    def test_garbage_yields_no_cues_and_does_not_raise(self):
        assert parse_srt("") == []
        assert parse_srt("not a subtitle file at all\n") == []

    def test_reads_a_file(self, tmp_path):
        path = tmp_path / "sample.en.srt"
        path.write_text(
            srt(
                """
1
00:00:01,000 --> 00:00:02,000
Read from disk.
"""
            ),
            encoding="utf-8",
        )
        assert read_srt(path)[0].text == "Read from disk."


class TestWhatNobodySaid:
    """Text in the file that is not speech. The verdict is recorded, not dropped."""

    def test_a_music_cue_is_music(self):
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:03,000
♪ and the chorus goes on ♪
"""
            )
        )
        assert cues[0].kind == "music"
        assert not cues[0].is_speech

    def test_a_cue_that_is_only_a_sound_is_sound(self):
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:02,000
[a door slams]
"""
            )
        )
        assert cues[0].kind == "sound"
        assert not cues[0].is_speech

    def test_an_aside_inside_speech_is_stripped_and_the_line_survives(self):
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:03,000
(laughing) I had no idea you were coming.
"""
            )
        )
        assert cues[0].kind == "speech"
        assert cues[0].text == "I had no idea you were coming."

    def test_an_aside_is_gone_from_the_utterance_too(self):
        # The regression that the aggregate run over the real file exposed.
        # `Cue.text` had the aside stripped while `Cue.parts` did not, and
        # the utterances are built from `parts` -- so asserting only on
        # `Cue.text` declared this fixed while it leaked.
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:03,000
(laughing) I had no idea you were coming.
"""
            )
        )
        assert cues[0].text == "I had no idea you were coming."
        assert [u.text for u in utterances(cues)] == ["I had no idea you were coming."]

    def test_parts_are_stripped_like_the_text_is(self):
        # `parts` is public and the utterances are built from it, so it has
        # to agree with `text`. The utterance-level assertion above does not
        # cover this: the strip after joining a run would hide a leak here.
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:04,000
- [sighs] Fine, you win.
- I always do.
"""
            )
        )
        assert cues[0].parts == ("Fine, you win.", "I always do.")
        assert all("[" not in part for part in cues[0].parts)

    def test_an_aside_split_across_two_cues_is_still_removed(self):
        # Neither cue can see the aside: each holds half a bracket pair. It
        # only closes once the run is joined, which is why the join strips
        # asides a second time.
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:03,000
I was going to call you [a phone
""",
                """
2
00:00:03,100 --> 00:00:05,000
rings in the distance] but I forgot.
""",
            )
        )
        spoken = utterances(cues)
        assert len(spoken) == 1
        assert "[" not in spoken[0].text and "]" not in spoken[0].text
        assert spoken[0].text == "I was going to call you but I forgot."

    def test_a_paren_inside_a_square_aside_does_not_defeat_it(self):
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:03,000
[sighs (heavily)] Fine, I will do it.
"""
            )
        )
        assert [u.text for u in utterances(cues)] == ["Fine, I will do it."]

    def test_an_unclosed_bracket_keeps_the_line(self):
        # Erring towards keeping a line that might be speech, because the
        # opposite error silently shrinks the corpus.
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:02,000
[unclosed and then some words
"""
            )
        )
        assert cues[0].kind == "speech"

    def test_non_speech_is_absent_from_the_utterances(self):
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:02,000
[a door slams]
""",
                """
2
00:00:02,100 --> 00:00:04,000
Somebody is at the door.
""",
            )
        )
        assert [u.text for u in utterances(cues)] == ["Somebody is at the door."]


class TestSpeakers:
    """Who said it, where the file bothers to say."""

    def test_reads_an_all_caps_label(self):
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:03,000
NARRATOR: The lights went out.
"""
            )
        )
        spoken = utterances(cues)
        assert spoken[0].speaker == "NARRATOR"
        assert spoken[0].text == "The lights went out."

    def test_a_label_with_no_speech_yields_no_utterance(self):
        # The SDH convention: the name announces who speaks next, and the
        # words are in the following cue. Keeping `NARRATOR:` would put a
        # line ending in a colon into the corpus.
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:02,000
NARRATOR:
""",
                """
2
00:00:02,100 --> 00:00:05,000
The street was completely empty.
""",
            )
        )
        spoken = utterances(cues)
        assert [u.text for u in spoken] == ["The street was completely empty."]

    def test_a_lowercase_colon_is_not_a_label(self):
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:03,000
I said: we are leaving now.
"""
            )
        )
        spoken = utterances(cues)
        assert spoken[0].speaker is None
        assert spoken[0].text == "I said: we are leaving now."

    def test_two_dashes_are_two_speakers(self):
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:04,000
- Are you ready to go?
- Not quite yet.
"""
            )
        )
        assert cues[0].parts == ("Are you ready to go?", "Not quite yet.")
        assert [u.text for u in utterances(cues)] == [
            "Are you ready to go?",
            "Not quite yet.",
        ]

    def test_a_dash_on_the_second_line_only_is_still_a_speaker_change(self):
        # Some releases give the dash to the incoming speaker and not to the
        # outgoing one. Counting dashes read this as one part and left the
        # dash sitting inside the text.
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:04,000
I thought it was today.
- It was yesterday.
"""
            )
        )
        assert cues[0].parts == ("I thought it was today.", "It was yesterday.")
        assert all("-" not in part for part in cues[0].parts)

    def test_a_continuation_line_stays_with_its_dash(self):
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:04,000
- I never finished telling you
what actually happened.
"""
            )
        )
        assert cues[0].parts == ("I never finished telling you what actually happened.",)

    def test_one_dash_is_a_continuation_not_a_speaker(self):
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:03,000
- and then we left without saying goodbye.
"""
            )
        )
        assert cues[0].parts == ("and then we left without saying goodbye.",)


class TestReassembly:
    """A cue is not an utterance. This is the part that is lossy."""

    def test_a_sentence_split_across_cues_comes_back_whole(self):
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:03,000
I have been meaning to tell you
""",
                """
2
00:00:03,100 --> 00:00:05,000
that the meeting was moved.
""",
            )
        )
        spoken = utterances(cues)
        assert len(spoken) == 1
        assert spoken[0].text == ("I have been meaning to tell you that the meeting was moved.")
        assert spoken[0].cue_indexes == (1, 2)
        assert spoken[0].spans_cues

    def test_a_complete_cue_stands_alone(self):
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:03,000
The train leaves at six.
""",
                """
2
00:00:03,100 --> 00:00:05,000
We should hurry.
""",
            )
        )
        spoken = utterances(cues)
        assert [u.text for u in spoken] == ["The train leaves at six.", "We should hurry."]
        assert not spoken[0].spans_cues

    def test_a_scene_change_closes_the_sentence(self):
        # Without the gap rule the last cue before a cut swallows the first
        # cue after it, which in edited dialogue is most cuts.
        gap = MAX_CONTINUATION_GAP_MS + 1_000
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:03,000
I was about to say
""",
                f"""
2
00:00:{3 + gap // 1000:02d},100 --> 00:00:30,000
something else entirely.
""",
            )
        )
        spoken = utterances(cues)
        assert len(spoken) == 2

    def test_an_abbreviation_does_not_close_the_sentence(self):
        # `Mr.` at the end of a cue is the oracle's exception, not a boundary.
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:03,000
I was speaking to Mr.
""",
                """
2
00:00:03,100 --> 00:00:05,000
Almeida about the lease.
""",
            )
        )
        spoken = utterances(cues)
        assert len(spoken) == 1
        assert spoken[0].text == "I was speaking to Mr. Almeida about the lease."

    def test_a_digit_before_the_dot_waits_for_the_next_cue(self):
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:03,000
The whole trip took 3.
""",
                """
2
00:00:03,100 --> 00:00:05,000
5 hours in the end.
""",
            )
        )
        spoken = utterances(cues)
        assert len(spoken) == 1
        assert "3. 5 hours" in spoken[0].text or "3.5 hours" in spoken[0].text

    def test_a_short_sentence_is_not_glued_to_the_one_after_it(self):
        # The 12-character floor in the segmenter is a synthesis-cost rule:
        # "Yes." alone would spend a whole TTS call on 300 ms of audio.
        # Nothing is synthesised here, so that floor would only bury a real
        # utterance inside the next one.
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:05,000
Yes. I will be there at noon.
"""
            )
        )
        assert [u.text for u in utterances(cues)] == [
            "Yes.",
            "I will be there at noon.",
        ]

    def test_two_short_answers_stay_two_utterances(self):
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:02,000
- Did you?
- Yes.
"""
            )
        )
        assert [u.text for u in utterances(cues)] == ["Did you?", "Yes."]

    def test_two_sentences_in_one_cue_become_two_utterances(self):
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:05,000
We arrived late. The door was already locked.
"""
            )
        )
        assert [u.text for u in utterances(cues)] == [
            "We arrived late.",
            "The door was already locked.",
        ]

    def test_a_trailing_fragment_is_not_lost(self):
        # Dialogue is interrupted; a run that never punctuates still happened.
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:03,000
But I thought you said
"""
            )
        )
        spoken = utterances(cues)
        assert len(spoken) == 1
        assert spoken[0].text == "But I thought you said"

    def test_the_span_covers_the_cues_it_came_from(self):
        cues = parse_srt(
            srt(
                """
1
00:00:01,000 --> 00:00:03,000
It was the first time
""",
                """
2
00:00:03,100 --> 00:00:06,000
anybody had asked me that.
""",
            )
        )
        spoken = utterances(cues)
        assert spoken[0].start_ms == 1_000
        assert spoken[0].end_ms == 6_000
