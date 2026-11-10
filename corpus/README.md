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
