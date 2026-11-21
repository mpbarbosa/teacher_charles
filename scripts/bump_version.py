#!/usr/bin/env python3
"""The control over the release version: read it, check it, move it.

The release is written in three places and they must agree —
``pyproject.toml``, ``__version__`` in ``src/teacher_charles/__init__.py``,
and this package's entry in ``uv.lock``. What the *running* code calls itself
is calculated from that release and the git history after it
(``src/teacher_charles/semver.py``); this script is the only thing that moves
the release itself.

    uv run python scripts/bump_version.py
        Where things stand: the release, its commit, the commits since, the
        ``Version-Bump`` trailers among them and the version they lead to.

    uv run python scripts/bump_version.py --check
        Exit 1 unless the three files carry the same valid version. CI-able;
        ``tests/test_semver.py`` runs the same check on every push.

    uv run python scripts/bump_version.py auto       # what the trailers say
    uv run python scripts/bump_version.py patch|minor|major
    uv run python scripts/bump_version.py 1.0.0      # a version by name
        Rewrites the three files. ``--dry-run`` only says what it would write.

It **does not commit and does not tag**: the checkout is shared (CLAUDE.md),
and a bare ``git commit`` here would carry whatever other sessions staged. It
prints the commit to run, with an explicit pathspec. The commit that changes
the ``version =`` line *is* the release commit — that is how the calculation
finds it — so the bump goes in a commit of its own, with nothing else in it.

``auto`` never leaves 0.x: a ``major`` trailer counts as a ``minor`` before
1.0.0. Naming ``major`` (or ``1.0.0``) here is how 1.0.0 is reached.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from teacher_charles.build_stamp import read_history
from teacher_charles.semver import TRAILER, Bump, Version, build_version

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "teacher-charles"

PYPROJECT = "pyproject.toml"
INIT = "src/teacher_charles/__init__.py"
LOCK = "uv.lock"

_PYPROJECT_LINE = re.compile(r'^(version = ")([^"]*)(")$', re.M)
_INIT_LINE = re.compile(r'^(__version__ = ")([^"]*)(")$', re.M)
_LOCK_ENTRY = re.compile(
    r'^(\[\[package\]\]\nname = "' + re.escape(PACKAGE) + r'"\nversion = ")([^"]*)(")$', re.M
)

FILES: dict[str, re.Pattern[str]] = {
    PYPROJECT: _PYPROJECT_LINE,
    INIT: _INIT_LINE,
    LOCK: _LOCK_ENTRY,
}


@dataclass(frozen=True)
class Found:
    path: str
    version: str | None  # None: the line is not there


def read_versions(root: Path = ROOT) -> list[Found]:
    found = []
    for rel, pattern in FILES.items():
        path = root / rel
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        m = pattern.search(text)
        found.append(Found(rel, m[2] if m else None))
    return found


def check(root: Path = ROOT) -> list[str]:
    """What is wrong with the release as written; empty when nothing is."""
    problems = []
    found = read_versions(root)
    for f in found:
        if f.version is None:
            problems.append(f"{f.path}: no version line")
            continue
        try:
            Version.parse(f.version)
        except ValueError:
            problems.append(f"{f.path}: {f.version!r} is not a semantic version")
    values = {f.version for f in found if f.version is not None}
    if len(values) > 1:
        listed = ", ".join(f"{f.path}={f.version}" for f in found)
        problems.append(f"the files disagree: {listed}")
    # tomllib is the parser pip and uv use; the regex above is only how the
    # file gets rewritten, so the two must see the same line.
    pyproject = root / PYPROJECT
    if pyproject.exists():
        declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
        here = next(f.version for f in found if f.path == PYPROJECT)
        if declared != here:
            problems.append(f"{PYPROJECT}: [project] says {declared}, the line says {here}")
    return problems


def current(root: Path = ROOT) -> Version:
    problems = check(root)
    if problems:
        raise SystemExit("bump_version: " + "; ".join(problems))
    return Version.parse(read_versions(root)[0].version or "")


def target(spec: str, release: Version, root: Path = ROOT) -> Version:
    """The version ``spec`` names, given the current release."""
    if spec == "auto":
        history = read_history(root)
        bump = history.bump if history is not None else Bump.PATCH
        return release.core.bump(bump) if release.is_release else release.core
    kind = Bump.parse(spec)
    if kind is not None:
        return release.bump(kind, stable=True)
    new = Version.parse(spec)
    if new.build:
        raise ValueError("a release carries no build metadata")
    return new


def write(new: Version, root: Path = ROOT) -> None:
    for rel, pattern in FILES.items():
        path = root / rel
        text = path.read_text(encoding="utf-8")
        updated, n = pattern.subn(lambda m: f"{m[1]}{new}{m[3]}", text, count=1)
        if n != 1:
            raise SystemExit(f"bump_version: {rel}: no version line to rewrite")
        path.write_text(updated, encoding="utf-8")


def status(root: Path = ROOT) -> str:
    release = current(root)
    history = read_history(root)
    lines = [f"release      {release}"]
    if history is None:
        lines.append("history      unreadable (no git, shallow clone, or no release commit)")
        return "\n".join(lines)
    running = build_version(release, history.distance, history.bump)
    lines += [
        f"release at   {history.release_sha[:7]}",
        f"since        {history.distance} commit(s)",
        f"trailers     {', '.join(history.trailers) or f'none (no {TRAILER}: → patch)'}",
        f"this build   {running}",
        f"next (auto)  {target('auto', release, root)}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "spec", nargs="?", help="auto | patch | minor | major | an explicit X.Y.Z[-pre]"
    )
    parser.add_argument("--check", action="store_true", help="exit 1 unless the files agree")
    parser.add_argument("--dry-run", action="store_true", help="say what would be written")
    args = parser.parse_args(argv)

    if args.check:
        problems = check()
        for p in problems:
            print(p, file=sys.stderr)
        if not problems:
            print(f"ok: {read_versions()[0].version} in {', '.join(FILES)}")
        return 1 if problems else 0

    if args.spec is None:
        print(status())
        return 0

    release = current()
    try:
        new = target(args.spec, release)
    except ValueError as err:
        print(f"bump_version: {err}", file=sys.stderr)
        return 2
    if new <= release:
        print(f"bump_version: {new} does not come after {release}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(f"{release} → {new} (dry run, nothing written)")
        return 0

    write(new)
    print(f"{release} → {new} in {', '.join(FILES)}")
    print("\nCommit it alone — that commit becomes the release commit:\n")
    print(f'  git commit -m "Releases {new}" -- {" ".join(FILES)}')
    print(f"  git tag -a v{new} -m v{new}      # optional; the calculation does not need it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
