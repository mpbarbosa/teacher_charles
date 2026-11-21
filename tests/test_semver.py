"""Semantic versions: the arithmetic (semver.py), the history (build_stamp.py)
and the control over the release (scripts/bump_version.py)."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from teacher_charles import __version__
from teacher_charles.build_stamp import calculate_version, read_history
from teacher_charles.semver import Bump, Version, build_version, largest_bump

ROOT = Path(__file__).resolve().parent.parent
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="no git on this machine")


# --- The arithmetic --------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    ["0.1.0", "1.0.0-rc.1", "1.0.0-alpha.beta", "1.0.0+20130313144700", "1.0.0-0.3.7+exp.sha"],
)
def test_parse_and_print_round_trip(text):
    assert str(Version.parse(text)) == text


@pytest.mark.parametrize("text", ["1.0", "01.0.0", "1.0.0-", "1.0.0-01", "v1.0.0", "1.0.0+"])
def test_what_semver_org_refuses_is_refused(text):
    with pytest.raises(ValueError):
        Version.parse(text)


def test_precedence_follows_section_11():
    # The example chain in semver.org §11, in order.
    chain = [
        "1.0.0-alpha",
        "1.0.0-alpha.1",
        "1.0.0-alpha.beta",
        "1.0.0-beta",
        "1.0.0-beta.2",
        "1.0.0-beta.11",
        "1.0.0-rc.1",
        "1.0.0",
    ]
    versions = [Version.parse(v) for v in chain]
    assert versions == sorted(versions)
    assert all(a < b for a, b in zip(versions, versions[1:], strict=False))


def test_build_metadata_takes_no_part_in_precedence():
    a, b = Version.parse("1.0.0+a"), Version.parse("1.0.0+b")
    assert not a < b and not b < a


@pytest.mark.parametrize(
    ("start", "kind", "stable", "expected"),
    [
        ("1.2.3", Bump.PATCH, False, "1.2.4"),
        ("1.2.3", Bump.MINOR, False, "1.3.0"),
        ("1.2.3", Bump.MAJOR, False, "2.0.0"),
        # Before 1.0.0 an automatic major is a minor; a named one leaves 0.x.
        ("0.4.2", Bump.MAJOR, False, "0.5.0"),
        ("0.4.2", Bump.MAJOR, True, "1.0.0"),
    ],
)
def test_bump(start, kind, stable, expected):
    assert str(Version.parse(start).bump(kind, stable=stable)) == expected


def test_the_largest_trailer_wins_and_none_is_a_patch():
    assert largest_bump([]) is Bump.PATCH
    assert largest_bump(["patch", "Minor", "patch"]) is Bump.MINOR
    # A typo is ignored, never raised: the footer must not fall over on it.
    assert largest_bump(["mayor", "patch"]) is Bump.PATCH


def test_on_the_release_commit_the_build_is_the_release():
    release = Version.parse("0.1.0")
    assert build_version(release, 0, Bump.MINOR) == release


def test_past_it_the_build_is_a_dev_prerelease_of_the_next_one():
    release = Version.parse("0.1.0")
    build = build_version(release, 5, Bump.MINOR)

    assert str(build) == "0.2.0-dev.5"
    # SemVer's own order states the truth: after the release, before the next.
    assert release < build < Version.parse("0.2.0")
    assert build < build_version(release, 6, Bump.MINOR)


def test_past_a_prerelease_release_the_build_sorts_after_it():
    rc = Version.parse("1.0.0-rc.1")
    build = build_version(rc, 3, Bump.PATCH)

    assert str(build) == "1.0.0-rc.1.dev.3"
    assert rc < build < Version.parse("1.0.0")


# --- The history ------------------------------------------------------------ #


class _Repo:
    def __init__(self, path: Path):
        self.path = path

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.path), "-c", "user.name=t", "-c", "user.email=t@t", *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def commit(self, message: str, **files: str) -> None:
        for name, text in files.items():
            (self.path / name).write_text(text, encoding="utf-8")
            self.git("add", name)
        self.git("commit", "-q", "--allow-empty", "-m", message)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    r = _Repo(tmp_path / "repo")
    r.path.mkdir()
    r.git("init", "-q")
    r.commit("first", **{"pyproject.toml": '[project]\nname = "x"\nversion = "0.1.0"\n'})
    return r


@needs_git
def test_on_the_release_commit_nothing_is_added(repo):
    history = read_history(repo.path)

    assert history is not None
    assert history.distance == 0
    assert calculate_version(repo.path, "0.1.0") == "0.1.0"


@needs_git
def test_commits_without_trailers_head_for_a_patch(repo):
    repo.commit("Fixes a typo")
    repo.commit("Fixes another")

    assert calculate_version(repo.path, "0.1.0") == "0.1.1-dev.2"


@needs_git
def test_a_trailer_raises_the_target(repo):
    repo.commit("Adds lessons\n\nBody.\n\nVersion-Bump: minor")
    repo.commit("Fixes a typo\n\nVersion-Bump: patch")

    history = read_history(repo.path)
    assert sorted(history.trailers) == ["minor", "patch"]
    assert calculate_version(repo.path, "0.1.0") == "0.2.0-dev.2"


@needs_git
def test_a_trailer_before_the_release_does_not_count(repo):
    repo.commit("Breaks things\n\nVersion-Bump: minor")
    repo.commit("Releases 0.2.0", **{"pyproject.toml": '[project]\nversion = "0.2.0"\n'})
    repo.commit("Fixes a typo")

    history = read_history(repo.path)
    assert history.distance == 1
    assert history.trailers == ()
    assert calculate_version(repo.path, "0.2.0") == "0.2.1-dev.1"


@needs_git
def test_a_manifest_edit_elsewhere_is_not_a_release(repo):
    repo.commit(
        "Adds a dependency",
        **{"pyproject.toml": '[project]\nname = "x"\nversion = "0.1.0"\ndependencies = []\n'},
    )

    assert read_history(repo.path).distance == 1


@needs_git
def test_a_shallow_clone_leaves_the_release_alone(repo, tmp_path):
    for n in range(3):
        repo.commit(f"commit {n}")
    shallow = tmp_path / "shallow"
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", f"file://{repo.path}", str(shallow)],
        check=True,
        capture_output=True,
    )

    assert read_history(shallow) is None
    assert calculate_version(shallow, "0.1.0") == "0.1.0"


def test_no_git_at_all_leaves_the_release_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))

    assert calculate_version(tmp_path, "0.1.0") == "0.1.0"


# --- The control over the release ------------------------------------------- #


@pytest.fixture
def bump_version():
    script = ROOT / "scripts" / "bump_version.py"
    spec = importlib.util.spec_from_file_location("bump_version", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bump_version"] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        del sys.modules["bump_version"]


def test_this_repo_writes_one_release_in_three_places(bump_version):
    """The guard CI runs: pyproject, ``__version__`` and uv.lock agree."""
    assert bump_version.check(ROOT) == []
    assert [f.version for f in bump_version.read_versions(ROOT)] == [__version__] * 3


def _copy_release_files(bump_version, dest: Path) -> Path:
    for rel in bump_version.FILES:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, dest / rel)
    return dest


def test_a_disagreement_is_reported(bump_version, tmp_path):
    root = _copy_release_files(bump_version, tmp_path)
    init = root / bump_version.INIT
    init.write_text('__version__ = "9.9.9"\n', encoding="utf-8")

    problems = bump_version.check(root)
    assert any("disagree" in p for p in problems)


def test_write_moves_all_three_and_only_the_package_entry_in_the_lock(bump_version, tmp_path):
    root = _copy_release_files(bump_version, tmp_path)
    lock_before = (root / bump_version.LOCK).read_text(encoding="utf-8")

    bump_version.write(Version.parse("7.8.9"), root)

    assert bump_version.check(root) == []
    assert {f.version for f in bump_version.read_versions(root)} == {"7.8.9"}
    lock_after = (root / bump_version.LOCK).read_text(encoding="utf-8")
    changed = [
        (a, b)
        for a, b in zip(lock_before.splitlines(), lock_after.splitlines(), strict=True)
        if a != b
    ]
    assert changed == [(f'version = "{__version__}"', 'version = "7.8.9"')]


@pytest.mark.parametrize(
    ("spec", "expected"),
    [("patch", "0.1.1"), ("minor", "0.2.0"), ("major", "1.0.0"), ("0.3.0-rc.1", "0.3.0-rc.1")],
)
def test_a_named_target(bump_version, spec, expected):
    assert str(bump_version.target(spec, Version.parse("0.1.0"))) == expected


def test_a_release_with_build_metadata_is_refused(bump_version):
    with pytest.raises(ValueError):
        bump_version.target("0.2.0+abc", Version.parse("0.1.0"))
