"""Semantic versions (semver.org 2.0.0), and how a running build gets one.

Two versions exist and they are not the same thing:

- the **release**, written by hand in ``pyproject.toml`` and mirrored in
  ``__version__`` — it moves only when ``scripts/bump_version.py`` moves it;
- the **build's**, calculated at startup from the release and the history
  after it (``build_stamp.py`` reads the history; this module only does the
  arithmetic). On the release commit itself the two are equal. Past it, the
  build is a pre-release of the *next* version: ``0.2.0-dev.5`` is the fifth
  commit on the way to ``0.2.0``, and SemVer's own precedence puts it after
  ``0.1.0`` and before ``0.2.0`` — which is the truth about that code.

What the next version is depends on how big the change was, and only a person
knows that. The commit message says it with a trailer::

    Version-Bump: minor

``major``, ``minor`` or ``patch``; the largest one since the release wins, and
a history with none is a ``patch``. A trailer and not a prefix in the subject
because this repo's subjects are prose ("Lights the word…"), and a trailer is
what ``git`` already knows how to read out of a message.

**Before 1.0.0 a ``major`` from a trailer counts as a ``minor``** (SemVer §4:
``0.y.z`` promises nothing, and the cargo convention reads the minor as the
breaking digit there). Leaving 0.x is a decision about the product, so it is
taken by name — ``bump_version.py major`` or ``bump_version.py 1.0.0`` — and
never falls out of a commit message.

Top-level and stdlib-only, like ``build_stamp.py``.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Iterable
from dataclasses import dataclass

# The regex semver.org publishes, with named groups.
_SEMVER = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<pre>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+(?P<build>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)

DEV = "dev"
TRAILER = "Version-Bump"


class Bump(enum.IntEnum):
    """How big a change is. Ordered, so the largest of several is ``max``."""

    PATCH = 1
    MINOR = 2
    MAJOR = 3

    @classmethod
    def parse(cls, text: str) -> Bump | None:
        """``"Minor"`` → ``MINOR``; anything else is ``None``, not an error:
        a typo in a trailer must not take the footer down."""
        return cls.__members__.get(text.strip().upper())


@dataclass(frozen=True, slots=True)
class Version:
    major: int
    minor: int
    patch: int
    prerelease: tuple[str, ...] = ()
    build: tuple[str, ...] = ()

    @classmethod
    def parse(cls, text: str) -> Version:
        m = _SEMVER.match(text.strip())
        if m is None:
            raise ValueError(f"not a semantic version: {text!r}")
        pre, build = m["pre"], m["build"]
        return cls(
            int(m["major"]),
            int(m["minor"]),
            int(m["patch"]),
            tuple(pre.split(".")) if pre else (),
            tuple(build.split(".")) if build else (),
        )

    def __str__(self) -> str:
        text = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            text += "-" + ".".join(self.prerelease)
        if self.build:
            text += "+" + ".".join(self.build)
        return text

    @property
    def is_release(self) -> bool:
        return not self.prerelease

    @property
    def core(self) -> Version:
        """The version without pre-release or build: what it is on its way to."""
        return Version(self.major, self.minor, self.patch)

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.major, self.minor, self.patch)

    def bump(self, kind: Bump, *, stable: bool = False) -> Version:
        """The next release after this one.

        ``stable=False`` is the automatic path: before 1.0.0 a ``MAJOR`` is
        read as a ``MINOR`` (see the module docstring). ``stable=True`` is a
        person naming the bump, and a ``major`` then means 1.0.0.
        """
        if kind is Bump.MAJOR and self.major == 0 and not stable:
            kind = Bump.MINOR
        if kind is Bump.MAJOR:
            return Version(self.major + 1, 0, 0)
        if kind is Bump.MINOR:
            return Version(self.major, self.minor + 1, 0)
        return Version(self.major, self.minor, self.patch + 1)

    def _precedence(self) -> tuple:
        # §11: a pre-release sorts before its release; identifiers compare
        # numerically when numeric, and numeric ones sort before the others.
        # Build metadata takes no part in precedence.
        pre = tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in self.prerelease)
        return (self.major, self.minor, self.patch, not self.prerelease, pre)

    def __lt__(self, other: Version) -> bool:
        return self._precedence() < other._precedence()

    def __le__(self, other: Version) -> bool:
        return self._precedence() <= other._precedence()

    def __gt__(self, other: Version) -> bool:
        return self._precedence() > other._precedence()

    def __ge__(self, other: Version) -> bool:
        return self._precedence() >= other._precedence()


def largest_bump(values: Iterable[str]) -> Bump:
    """The largest bump the trailers name; ``PATCH`` when they name none."""
    found = [b for b in map(Bump.parse, values) if b is not None]
    return max(found, default=Bump.PATCH)


def build_version(release: Version, distance: int, bump: Bump) -> Version:
    """The version of the code ``distance`` commits after ``release``.

    On the release commit it is the release. Past it, it is the
    ``dev.<distance>`` pre-release of the version those commits are heading
    to. No build metadata: the commit's sha travels beside it on the page.
    """
    if distance <= 0:
        return release
    if not release.is_release:
        # After ``1.0.0-rc.1``: ``1.0.0-rc.1.dev.3``, which sorts after it.
        # Re-targeting ``1.0.0-dev.3`` would sort *before* the rc (§11).
        return Version(*release.core.as_tuple(), (*release.prerelease, DEV, str(distance)))
    target = release.core.bump(bump)
    return Version(*target.as_tuple(), (DEV, str(distance)))
