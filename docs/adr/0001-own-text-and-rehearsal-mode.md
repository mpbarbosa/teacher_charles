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
