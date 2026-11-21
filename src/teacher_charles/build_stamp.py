"""The commit this server is running, for the line at the bottom of every page.

Read **once, when the app is built**, and never per request: what a page should
name is the code this process loaded, and a commit made while the server is up
does not change that — it only changes what ``git log`` says. Reading it per
request would have the footer name a commit the running code is not.

Asked of ``git`` itself rather than parsed out of ``.git``: in a worktree
``.git`` is a file pointing elsewhere, refs may be packed, and ``git`` already
knows all of that. A machine with no ``git``, or a copy that is not a checkout
at all, gets an empty stamp and the page shows the version alone — the same
rule as the warm-up: a footer is never a reason for startup to fail.

The datetime is the **commit's**, not the start time's: it dates the code, and
two servers on the same commit then show the same line.

The version beside the commit is the **build's**, not the release's
(``semver.py``): the release written in ``pyproject.toml``, plus how far past
it this commit is and how big the trailers say the change has been. Same rule
again — a history git cannot read leaves the release standing alone, never an
error.

Top-level and stdlib-only, like ``stress_shift.py``: it is about the install,
not about teaching, and ``config.py`` must stay free to import nothing heavy.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import __version__
from .semver import TRAILER, Bump, Version, build_version, largest_bump

GIT_TIMEOUT_S = 2.0
SHORT_SHA = 7
# The release lives here; the commit that last changed this line is the
# release commit. ``:(top)`` because ``git -C`` runs inside the package.
MANIFEST = ":(top)pyproject.toml"
RELEASE_LINE = r'^version = "'


@dataclass(frozen=True, slots=True)
class BuildStamp:
    sha: str | None = None
    committed_at: datetime | None = None
    # Uncommitted changes to tracked files: the code running is then *not*
    # the commit named, and the footer says so instead of vouching for it.
    dirty: bool = False
    # The build's semantic version (semver.py); empty when unknown, and the
    # page then shows the release.
    version: str = ""

    @property
    def known(self) -> bool:
        return self.sha is not None and self.committed_at is not None

    @property
    def short_sha(self) -> str:
        return (self.sha or "")[:SHORT_SHA]

    @property
    def when(self) -> str:
        """``2026-11-19 14:32 -03:00``: the commit's own clock and offset."""
        if self.committed_at is None:
            return ""
        return self.committed_at.strftime("%Y-%m-%d %H:%M %:z")

    @property
    def iso(self) -> str:
        return self.committed_at.isoformat() if self.committed_at else ""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=GIT_TIMEOUT_S,
    ).stdout.strip()


@dataclass(frozen=True, slots=True)
class History:
    """Where the code stands relative to the last release."""

    release_sha: str
    distance: int
    # The ``Version-Bump`` trailer values since the release, as written.
    trailers: tuple[str, ...]

    @property
    def bump(self) -> Bump:
        return largest_bump(self.trailers)


def read_history(root: Path) -> History | None:
    """The release commit, the commits after it and their bump trailers.

    ``None`` when the history cannot say: no git, not a checkout, a manifest
    no commit ever set a version in, or a **shallow clone** — there the
    oldest commit present shows every line as added, the version line
    included, and would pass for the release commit.
    """
    try:
        if _git(root, "rev-parse", "--is-shallow-repository") == "true":
            return None
        release = _git(root, "log", "-1", "--format=%H", "-G", RELEASE_LINE, "--", MANIFEST)
        if not release:
            return None
        span = f"{release}..HEAD"
        distance = int(_git(root, "rev-list", "--count", span))
        raw = _git(
            root, "log", f"--format=%(trailers:key={TRAILER},valueonly,separator=%x0A)", span
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    trailers = tuple(line.strip() for line in raw.splitlines() if line.strip())
    return History(release_sha=release, distance=distance, trailers=trailers)


def calculate_version(root: Path, release: str = __version__) -> str:
    """The build's version: the release, or the ``-dev.N`` on the way past it."""
    history = read_history(root)
    if history is None:
        return release
    try:
        base = Version.parse(release)
    except ValueError:
        return release
    return str(build_version(base, history.distance, history.bump))


def read_build_stamp(root: Path) -> BuildStamp:
    """The commit checked out at ``root``, or an empty stamp if there is none."""
    try:
        sha, _, stamp = _git(root, "log", "-1", "--format=%H %cI").partition(" ")
        committed_at = datetime.fromisoformat(stamp)
    except (OSError, subprocess.SubprocessError, ValueError):
        return BuildStamp()
    try:
        # Untracked files are left out on purpose: a stray scratch file is
        # not code the server runs, and counting it would mark every
        # developer checkout dirty.
        dirty = bool(_git(root, "status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.SubprocessError):
        dirty = False
    return BuildStamp(
        sha=sha, committed_at=committed_at, dirty=dirty, version=calculate_version(root)
    )
