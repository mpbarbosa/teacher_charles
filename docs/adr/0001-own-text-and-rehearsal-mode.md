# 0001 — Lessons about your own text, and rehearsing them aloud

- Status: accepted
- Date: 2026-11-20
- Code: `teaching/lessons.py` (`OwnText`, `RehearsalBlock`, `rehearsal_plan`,
  `read_local_cues`), `app.py` (`rehearsal_view`), `web/static/rehearsal.js`,
  `web/templates/lessons.html`
- Tests: `tests/test_lessons.py`, sections "a text nobody published",
  "the rehearsal plan", "the rehearsal on the page"

## Context

A student wanted to rehearse a five-minute pitch for a recorded job interview:
five blocks with target times, a pace in words per minute, a stopwatch per
block, a full run-through, the text hidden behind cue words, and a recording
of their own voice to play back. A standalone page did this well, but it sat
outside the app and carried the whole script inline.

Two things in the lesson model stood in the way of making it a lesson:

1. `Source` requires a published URL, outlet, byline and date. A pitch has
   none of them.
2. A lesson analyses a text; nothing in it said how to *say* the text — which
   paragraph is which block, and how long each should take.

And one thing in the repository shaped every answer: it is public. The
script names real people and its author's career, so it must never be
committed, and the lesson must not say whose it is.

## Decision

**An own text is a second kind of source, not `Source` with optional fields.**
`OwnText` has `kind: own`, a generic title and a word count, and
`extra="forbid"` refuses a `url:` — inventing an address for a text nobody
published would be a citation to nothing. `Lesson.source` is a discriminated
union whose tag defaults to `published`, so every existing lesson loads
unchanged and keeps every field required. The text lives only in
`corpus/lessons/<slug>.txt`, outside version control, and the citation budget
holds as for any source: the lesson is public even when the text is not.

**The structure goes in the YAML; the cue words stay with the copy.** The
lesson's optional `rehearsal` list gives one block per paragraph of the copy,
in order: a title and a target in seconds (absent = untimed, skipped by the
run-through). That is curated structure, in the app, per the project's
principle. The cue words shown while the text is hidden are *words of the
text*, so in the YAML they would be a second, unbudgeted channel for source
wording — and for this lesson, private facts. `RehearsalBlock` therefore has
no `keywords` field (`extra="forbid"` refuses one), and the cue words live in
`corpus/lessons/<slug>.cues.txt`, one line per paragraph. Without that file,
each sentence is cued by its first three words, so every lesson with a copy
can be rehearsed.

**A plan that does not fit the copy is set aside, not stretched.** Blocks pair
with paragraphs by position; one paragraph more or less would shift every
target after it, and a wrong number on screen is worse than none. On a count
mismatch the page falls back to untimed paragraphs and says why, as it already
does when the copy's length is off. A cue file *shorter* than the copy is
accepted (trailing blank lines are what an editor trims first); a longer one
is set aside.

**Nothing of a rehearsal reaches the server.** Times, pace and run-throughs are
kept per lesson in the browser's `localStorage`: no level, schedule or
correction reads a rehearsal time, so it is not evidence the app teaches from,
and `charles.db` stays the record of what is. Recordings are blobs in the
page's memory, played through object URLs and never sent — not even to
`/api/stt`, because Whisper's reading of a minute of speech would look like a
check, which it is not (the word takes beside the vocabulary already say why).

**The voice and the IPA are the reader's.** A block's "Hear it" is the full-text
reader in `lesson.js` limited to one paragraph, so it uses the same Kokoro
voice and lights the same sentences and IPA; the word cards keep working on
the text. The rehearsal asks the reader for silence (a `tc:silence` event)
when a run-through or a recording starts.

## Consequences

- Run-through history does not follow the student to another browser or
  machine. Moving it to the database is possible later, but it would be the
  first thing stored that nothing in the teaching reads.
- A clone shows an own-text lesson with no text and a sentence saying where to
  put it. That is the same arrangement as every lesson whose source is not on
  the machine.
- A cue file and a plan can drift from an edited copy in ways a count cannot
  see (a paragraph replaced, not added). The page then shows the wrong cue
  words for that block; the remedy is editing the cue file, which sits beside
  the copy.
- The microphone's behaviour was exercised in the app with a synthetic audio
  stream standing in for a microphone; no human recording has been made
  through it yet.

## Amendment — Follow me (2026-11-20)

- Code: `voice/follow.py`, `app.py` (`POST /api/follow`),
  `web/static/follow.js`
- Tests: `tests/test_voice_follow.py`; `tests/test_lessons.py`,
  `test_follow_me_is_there_only_with_a_voice`

The student asked for the word they are saying to be lit while they read the
text aloud, as the reader lights the word Charles is saying. Nothing in the
browser can tell which word is being said, so this is the one part of the
rehearsal that sends the student's voice to the server — which the decision
above says nothing of a rehearsal does. Three ways were weighed: the local
Whisper; Chrome's `SpeechRecognition`, faster but sending the audio to Google
unless an on-device model is present, for a script that names real people;
and a cursor moving at the chosen pace, private but showing where the student
*should* be, not what they said. The student chose the local Whisper.

**The voice goes to this app's server and dies in the request.** While
"Follow me" is on, the page keeps the last 3.5 s of the microphone as raw
samples and, after each answer, sends them as 16 kHz PCM to `/api/follow`
with the words around the light. The timing Whisper (`tiny.en`) hears them
and `voice/follow.py` finds the last heard word that is surely in the text.
The route records nothing and **returns only that index** — not what Whisper
heard — so this decision's reason for keeping recordings off `/api/stt` still
holds: nothing on the page can read as a transcript, let alone as a check. A
stretch whose last second is quiet is not sent at all. "Record yourself" is
unchanged: its take never leaves the page.

**The match is anchored, not greedy.** Letter by letter, as the reader's
timings are, but from the longest run the two sides share (ten letters at
least, two or three words), followed forward only while each next run starts
close to the last. Over a long text a few letters in common are chance, and
the furthest match would drag the light paragraphs ahead.

Measured by standing Kokoro's reading of the pitch in for the microphone, a
window every second: 55 of 55 windows placed on the first paragraph, a median
of 0 words from the word being said and 4 at worst (Whisper writes "two
hundred thousand" as digits, which share no letters with the text, so the
light waits there); with the whole 482-word text searched every time, as
while the light is lost, 84 of 84 placed and 2 words off at worst. A pass is
~430 ms on this CPU, so the light moves about once a second, roughly a word
behind the voice.

Consequences:

- Like the rest of the voice, this needs the `voice` extra; without it the
  button is not drawn.
- Synthetic speech is clean, so these numbers are a ceiling. On 2026-11-21
  the student read the pitch aloud through a real microphone, with Follow me
  on, and the light followed them. That was seen, not measured: no window was
  counted against the word being said, so how far off the light runs on a
  Brazilian accent read live is still unknown — the same open debt as
  `/api/stt` (Phase 1.5 plan §2.3).
