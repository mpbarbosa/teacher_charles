# corpus/ — third-party source material

Authentic English that somebody else wrote: subtitles, transcripts, book
excerpts. It is listening and reading material for the student, and raw
material for curation — a real line that a real speaker said is the kind of
evidence `data/contrastive/` is built from.

**Nothing in here is versioned, and that is not an oversight.** This is a
fourth data destination, next to the three in `CLAUDE.md`, and confusing it
with them is the same category error:

| | lives in | why |
|---|---|---|
| model weights | `~/.ollama`, voice caches | re-downloadable, hundreds of MB |
| student progress | `$XDG_DATA_HOME/teacher-charles/` | personal data |
| curated knowledge | `data/contrastive/*.yaml` | **versioned** — it is our content, reviewed item by item |
| third-party material | `corpus/`, this folder | somebody else's copyright |

The repository is public. A subtitle file is the complete dialogue of an
episode, owned by whoever made the episode — committing it would publish it
from here, which is redistribution regardless of how the file was obtained.
So `.gitignore` holds `corpus/*` with an exception for this README: the rule
travels with the clone, the material does not. Whoever runs the project puts
their own copies in place.

Note the asymmetry with `data/contrastive/`. A curated item is a rule we wrote,
and a derived observation — *this construction appears 12 times in 30 minutes
of sitcom dialogue* — is ours to version too. The 30 minutes of dialogue are
not. Derive into `data/`, never copy into it.

## Layout

```
corpus/
  subtitles/
    <show>/
      <show>-<season>x<episode>.en.srt
```

Normalize on the way in: strip the UTF-8 BOM, LF line endings, lower-case
slug with no spaces, so a path can be written in a script without quoting.

```bash
mkdir -p corpus/subtitles/<show>
sed -e '1s/^\xEF\xBB\xBF//' -e 's/\r$//' /path/to/original.srt \
  > corpus/subtitles/<show>/<show>-1x01.en.srt
```

## What is on this machine

Not a manifest to fetch from — a record of what the paths below meant, so a
script that reads one of them can be understood without the file present.

| path | content | cues | runtime |
|---|---|---|---|
| `subtitles/friends/friends-1x01.en.srt` | *Friends*, season 1 episode 1, English, from a BluRay release | 365 | 22m38s |

Imported 2026-11-10 from a local archive supplied by the person running the
project. Rights are theirs to hold; this repo asserts none.

## What was looked at and not imported

The rule above says what may live here. This says what was held against it and
turned down, because the next person to find a good free source will ask the
same question, and the answer is not "no" — it is *which half*.

**BBC Learning English — Pronunciation**
(`bbc.co.uk/learningenglish/features/pronunciation`), read 2026-11-10. Two
series: *Tim's Pronunciation Workshop*, 75 episodes on connected speech
(assimilation, elision, gemination, linking and intrusive `/r/`, schwa, the
weak forms of *and*, *can*, *was*, *have to*, *from*, *been*, *the*), each page
carrying the **full transcript as text** and a Takeaways PDF; and *The Sounds
of English*, 45 videos, one per sound. The video is an embedded player — there
is no audio or subtitle file to fetch. Generic learner material: nothing on it
is aimed at Portuguese speakers.

Nothing from it was imported, for three separate reasons, and only the first
is about rights:

1. **It is not material the person running the project holds.** The table
   above records a local archive, which is the condition under which a copy
   sits in this folder at all. A site is not that, and the page text is the
   BBC's. The asymmetry does not change: an observation we write down, with
   the source credited, is ours to version.
2. **It does not unblock the E9 gate.** What the gate needs is a Brazilian
   human voice (§2.5 of `docs/plano-fase-2.md`). Clean RP read by a presenter
   would give a second *floor*, next to the 0.061 of the synthetic audio — not
   the expectation the 0.676 proxy stands in for. More material of the easiest
   kind measures nothing that is open.
3. **The variant disagrees with the loudspeaker.** `DEFAULT_LANGUAGE` in
   `teaching/phonetics.py` is `en-us`. Linking `/r/` and intrusive `/r/` — two
   of the thirty phenomena — are facts about a *non-rhotic* accent and do not
   exist in what espeak renders. A rule taken from them would have the screen
   teaching against the speaker, which is the one failure the single-G2P rule
   exists to prevent. Every item from a source like this has to be filtered by
   variant before anything else, and about a third of this list goes.

What survives is a **reading list, not a corpus**. It points at a kind of
curated item the base has no schema for today: `data/contrastive/` holds
grammar and lexis, and these are cross-word pronunciation facts — *sandwich*
said as *samwich*, the `/d/` elided and then the `/n/` assimilated before
`/w/`. The IPA line cannot produce them and will not learn to: espeak
phonemizes **per word**, so what the student reads is the citation form, which
in that example is the one thing a fluent speaker does not say. That is a real
gap, it is independent of E9 the way E13 and the IPA line were, and closing it
is curation — one item with a test each, written here, citing nothing verbatim.

The 45-unit sound inventory is the half worth skipping: espeak already names
every phoneme the evaluator would score, so the videos add a label somebody
would be writing anyway.
