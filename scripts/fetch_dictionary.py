#!/usr/bin/env python3
"""Builds the word cards' lookup file from two open lexicons, once per machine.

    uv run --with wn==1.1.1 python scripts/fetch_dictionary.py
    uv run --with wn==1.1.1 python scripts/fetch_dictionary.py --keep   # keep wn's 270 MB

Downloads Open English WordNet 2024 (definitions, examples, irregular forms)
and OpenWN-PT 1.0 (the Portuguese), joins them sense by sense through the
Interlingual Index, and writes what the card reads to
``Settings.resolved_dictionary_path`` — ``$XDG_CACHE_HOME/teacher-charles/
dictionary/`` by default. Why these two, and why a file of our own, is in
``teaching/dictionary.py``.

``wn`` is passed with ``--with`` and is not a project dependency, extra or
not: it is needed here, for about a minute, and never at runtime. Its working
data goes to a scratch directory beside the output and is deleted at the end
unless ``--keep`` is given.

Nothing here is versioned, and nothing downloaded enters the repository: the
lexicons are re-downloadable, which makes the output a cache, like the voice
weights (``scripts/fetch_voice_models.py``).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from teacher_charles.config import Settings  # noqa: E402
from teacher_charles.teaching.dictionary import (  # noqa: E402
    MAX_PORTUGUESE,
    BuiltWord,
    Sense,
    write_lookup,
)

# By URL and not by the ``wn`` project id: the index's first URL for OEWN is
# en-word.net, which answered 503 when this was written, and ``wn.download``
# does not fall through to the mirror. GitHub's release is the same file.
ENGLISH = (
    "oewn:2024",
    "https://github.com/globalwordnet/english-wordnet/releases/download/"
    "2024-edition/english-wordnet-2024.xml.gz",
)
PORTUGUESE = (
    "own-pt:1.0.0",
    "https://github.com/own-pt/openWordnet-PT/releases/download/v1.0.0/own-pt.tar.gz",
)


def ili_of(synset) -> str | None:
    ili = synset.ili
    if ili is None:
        return None
    return getattr(ili, "id", ili)


def portuguese_by_ili(wn, spec: str) -> dict[str, list[str]]:
    """Each interlingual sense id to its Portuguese words, without repeats.

    One pass over the Portuguese lexicon, instead of ``synset.translate`` per
    English sense: that is an index lookup each, 120 thousand times.
    """
    out: dict[str, list[str]] = {}
    for synset in wn.Wordnet(spec).synsets():
        key = ili_of(synset)
        if not key:
            continue
        words = out.setdefault(key, [])
        for lemma in synset.lemmas():
            lemma = lemma.replace("_", " ").strip()
            if lemma and lemma.lower() not in (w.lower() for w in words):
                words.append(lemma)
    return out


def built_words(wn, spec: str, portuguese: dict[str, list[str]]):
    """Every English word with its senses in WordNet's order, ready to write."""
    english = wn.Wordnet(spec)
    for word in english.words():
        senses = []
        for synset in word.synsets():
            examples = synset.examples()
            key = ili_of(synset)
            senses.append(
                Sense(
                    definition=(synset.definition() or "").strip(),
                    example=examples[0].strip() if examples else None,
                    portuguese=tuple((portuguese.get(key, []) if key else [])[:MAX_PORTUGUESE]),
                )
            )
        senses = [s for s in senses if s.definition]
        if senses:
            yield BuiltWord(
                lemma=word.lemma(),
                pos=word.pos,
                forms=tuple(f for f in word.forms() if f != word.lemma()),
                senses=tuple(senses),
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--keep", action="store_true", help="keep wn's working data")
    args = parser.parse_args()

    try:
        import wn
    except ImportError:
        print(
            "wn is not installed. Run this script as:\n"
            "  uv run --with wn==1.1.1 python scripts/fetch_dictionary.py",
            file=sys.stderr,
        )
        return 2

    target = Settings().resolved_dictionary_path
    scratch = target.parent / "wn-data"
    # wn creates its data directory with a bare mkdir, parents not included,
    # so on a machine with no cache folder for the app yet it fails before
    # downloading anything.
    scratch.mkdir(parents=True, exist_ok=True)
    wn.config.data_directory = scratch
    print(f"lexicons → {scratch}")

    started = time.monotonic()
    for spec, url in (ENGLISH, PORTUGUESE):
        if wn.lexicons(lexicon=spec):
            print(f"  {spec}: already downloaded")
            continue
        print(f"  {spec}: downloading and importing (about half a minute)…", flush=True)
        wn.download(url, progress_handler=None)

    print("joining English and Portuguese by interlingual sense…", flush=True)
    portuguese = portuguese_by_ili(wn, PORTUGUESE[0])
    count = write_lookup(
        target,
        built_words(wn, ENGLISH[0], portuguese),
        meta={"english": ENGLISH[0], "portuguese": PORTUGUESE[0]},
    )
    size = target.stat().st_size / 1024 / 1024
    print(f"wrote {count} words to {target} ({size:.0f} MB, {time.monotonic() - started:.0f}s)")

    if not args.keep:
        # wn keeps its connection pool open; the files go regardless, and the
        # process is about to end.
        shutil.rmtree(scratch, ignore_errors=True)
        print(f"removed {scratch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
