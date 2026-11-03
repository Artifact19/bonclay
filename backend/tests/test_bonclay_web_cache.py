"""./bonclay start clears the web build cache (frontend/.next) when HEAD is not the commit the last web start recorded
(.bonclay/web.head), and `./bonclay clean-web` deletes it unless the web app is running.

Each test copies the script into a temporary git repository (the script finds its own root), so no real frontend, web
server or port is involved. `_clear_web_cache` is the hidden subcommand that runs the same check `start` makes before
`next dev`; a live web app is faked with a `sleep` process whose pid is in .bonclay/web.pid."""
import shutil
import subprocess
from pathlib import Path

import pytest

from test_bonclay_env_settings import Home

pytestmark = pytest.mark.skipif(shutil.which("bash") is None or shutil.which("git") is None,
                                reason="needs bash and git")

CLEARED = "Cleared the web build cache (code changed since the last start)"


def git(root: Path, *args):
    r = subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false",
                        *args], cwd=root, capture_output=True, text=True, check=True)
    return r.stdout.strip()


@pytest.fixture
def repo(tmp_path):
    h = Home(tmp_path)
    (h.root / "frontend").mkdir()
    git(h.root, "init", "-q")
    git(h.root, "commit", "-q", "--allow-empty", "-m", "first")
    return h


def make_cache(h: Home) -> Path:
    cache = h.root / "frontend" / ".next"
    (cache / "static").mkdir(parents=True)
    (cache / "static" / "old.css").write_text("body { color: red; }")
    return cache


def record(h: Home, sha: str):
    (h.root / ".bonclay").mkdir(exist_ok=True)
    (h.root / ".bonclay" / "web.head").write_text(sha + "\n", encoding="utf-8")


def recorded(h: Home) -> str:
    return (h.root / ".bonclay" / "web.head").read_text(encoding="utf-8").strip()


def test_no_cache_means_nothing_to_clear_and_head_is_recorded(repo):
    head = git(repo.root, "rev-parse", "HEAD")
    out = repo.run("_clear_web_cache")
    assert CLEARED not in out
    assert recorded(repo) == head


def test_cache_from_an_older_commit_is_cleared(repo):
    old = git(repo.root, "rev-parse", "HEAD")
    record(repo, old)
    git(repo.root, "commit", "-q", "--allow-empty", "-m", "second")
    cache = make_cache(repo)
    out = repo.run("_clear_web_cache")
    assert not cache.exists()
    assert CLEARED in out
    assert recorded(repo) == git(repo.root, "rev-parse", "HEAD")


def test_cache_is_kept_when_head_is_the_recorded_commit(repo):
    record(repo, git(repo.root, "rev-parse", "HEAD"))
    cache = make_cache(repo)
    out = repo.run("_clear_web_cache")
    assert cache.exists() and (cache / "static" / "old.css").exists()
    assert CLEARED not in out


def test_first_start_with_this_script_clears_an_existing_cache(repo):
    cache = make_cache(repo)  # no web.head yet: the cache cannot be shown to match the code, so it goes
    out = repo.run("_clear_web_cache")
    assert not cache.exists()
    assert CLEARED in out
    assert recorded(repo) == git(repo.root, "rev-parse", "HEAD")


def test_outside_a_git_checkout_the_cache_is_left_alone(tmp_path):
    h = Home(tmp_path)  # no git repository here; pytest's temp dir sits inside this repo, so git must not climb above it
    cache = make_cache(h)
    out = h.run("_clear_web_cache", env={"GIT_CEILING_DIRECTORIES": str(h.root.parent)})  # git may not step up out of h.root
    assert cache.exists()
    assert CLEARED not in out
    assert not (h.root / ".bonclay" / "web.head").exists()


def test_clean_web_deletes_the_cache_when_the_web_app_is_not_running(repo):
    cache = make_cache(repo)
    out = repo.run("clean-web")
    assert not cache.exists()
    assert "Deleted frontend/.next" in out


def test_clean_web_without_a_cache_says_so(repo):
    out = repo.run("clean-web")
    assert "No web build cache (frontend/.next) to delete" in out


def test_clean_web_refuses_while_the_web_app_runs(repo):
    cache = make_cache(repo)
    sleeper = subprocess.Popen(["sleep", "30"])  # stands in for a running next dev
    try:
        (repo.root / ".bonclay").mkdir(exist_ok=True)
        (repo.root / ".bonclay" / "web.pid").write_text(str(sleeper.pid), encoding="utf-8")
        out = repo.run("clean-web")
        assert cache.exists(), "the cache was deleted while the web app was running"
        assert "The web app is running" in out
    finally:
        sleeper.kill()
        sleeper.wait()
