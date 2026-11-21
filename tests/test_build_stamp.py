"""The commit line at the bottom of every page (build_stamp.py)."""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

import teacher_charles.app as app_module
from teacher_charles.app import create_app
from teacher_charles.build_stamp import BuildStamp, read_build_stamp
from teacher_charles.config import Settings
from teacher_charles.llm import FakeProvider

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="no git on this machine")

SHA = "0123456789abcdef0123456789abcdef01234567"
WHEN = datetime(2026, 11, 19, 14, 32, 5, tzinfo=timezone(timedelta(hours=-3)))

PAGES = ["/", "/progress", "/phonetics", "/lessons", "/lessons/interview-video-pitch"]


def _repo(path, monkeypatch):
    """A one-commit repository, on a commit date fixed to the second."""
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(path.parent))
    monkeypatch.setenv("GIT_COMMITTER_DATE", "2026-11-19T14:32:05-03:00")
    path.mkdir()
    (path / "tracked.txt").write_text("one\n", encoding="utf-8")

    def git(*args):
        subprocess.run(
            ["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@t", *args],
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    git("add", "tracked.txt")
    git("commit", "-q", "-m", "first")
    sha = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return sha


@needs_git
def test_names_the_commit_and_its_own_date(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    sha = _repo(repo, monkeypatch)

    stamp = read_build_stamp(repo)

    assert stamp.known
    assert stamp.sha == sha
    assert stamp.short_sha == sha[:7]
    # The commit's clock and offset, not the server's start time.
    assert stamp.when == "2026-11-19 14:32 -03:00"
    assert stamp.iso == "2026-11-19T14:32:05-03:00"
    assert not stamp.dirty


@needs_git
def test_a_changed_tracked_file_marks_it_dirty(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _repo(repo, monkeypatch)
    (repo / "tracked.txt").write_text("two\n", encoding="utf-8")

    assert read_build_stamp(repo).dirty


@needs_git
def test_an_untracked_file_does_not(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _repo(repo, monkeypatch)
    (repo / "scratch.txt").write_text("notes\n", encoding="utf-8")

    assert not read_build_stamp(repo).dirty


@needs_git
def test_a_directory_that_is_not_a_checkout_gets_an_empty_stamp(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))

    stamp = read_build_stamp(tmp_path)

    assert stamp == BuildStamp()
    assert not stamp.known


def test_no_git_at_all_gets_an_empty_stamp(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))

    assert read_build_stamp(tmp_path) == BuildStamp()


# --- On the pages ---------------------------------------------------------- #


def _client(tmp_path, monkeypatch, stamp: BuildStamp) -> TestClient:
    monkeypatch.setattr(app_module, "read_build_stamp", lambda _root: stamp)
    settings = Settings(
        _env_file=None,
        db_path=tmp_path / "stamp.db",
        provider="fake",
        llm_warmup=False,
        corpus_dir=tmp_path / "corpus",
    )
    return TestClient(create_app(settings, provider=FakeProvider()))


@pytest.mark.parametrize("page", PAGES)
def test_every_page_ends_with_the_commit_and_its_datetime(tmp_path, monkeypatch, page):
    with _client(tmp_path, monkeypatch, BuildStamp(sha=SHA, committed_at=WHEN)) as client:
        response = client.get(page)

    assert response.status_code == 200
    html = response.text
    footer = html[html.rindex("<footer") :]
    assert f'<code title="{SHA}">0123456</code>' in footer
    assert '<time datetime="2026-11-19T14:32:05-03:00">2026-11-19 14:32 -03:00</time>' in footer
    assert "+dirty" not in footer


def test_a_dirty_checkout_says_so(tmp_path, monkeypatch):
    stamp = BuildStamp(sha=SHA, committed_at=WHEN, dirty=True)
    with _client(tmp_path, monkeypatch, stamp) as client:
        assert "+dirty" in client.get("/").text


def test_without_a_commit_the_version_stands_alone(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, BuildStamp()) as client:
        html = client.get("/progress").text

    footer = html[html.rindex("<footer") :]
    assert "build-stamp" in footer
    assert "<code" not in footer
    assert "<time" not in footer


def test_the_stamp_is_read_once_not_per_request(tmp_path, monkeypatch):
    calls = []

    def counting(root):
        calls.append(root)
        return BuildStamp(sha=SHA, committed_at=WHEN)

    monkeypatch.setattr(app_module, "read_build_stamp", counting)
    settings = Settings(
        _env_file=None, db_path=tmp_path / "s.db", provider="fake", llm_warmup=False
    )
    with TestClient(create_app(settings, provider=FakeProvider())) as client:
        for page in ("/", "/progress", "/"):
            client.get(page)

    assert len(calls) == 1
