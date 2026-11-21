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

Top-level and stdlib-only, like ``stress_shift.py``: it is about the install,
not about teaching, and ``config.py`` must stay free to import nothing heavy.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

GIT_TIMEOUT_S = 2.0
SHORT_SHA = 7


@dataclass(frozen=True, slots=True)
class BuildStamp:
    sha: str | None = None
    committed_at: datetime | None = None
    # Uncommitted changes to tracked files: the code running is then *not*
    # the commit named, and the footer says so instead of vouching for it.
    dirty: bool = False

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
    return BuildStamp(sha=sha, committed_at=committed_at, dirty=dirty)
