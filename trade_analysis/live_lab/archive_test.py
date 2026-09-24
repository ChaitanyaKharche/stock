"""Does the end-of-session archive commit the record, and refuse everything else?

    python -m trade_analysis.live_lab.archive_test
    pytest trade_analysis/live_lab/archive_test.py

This runs against REAL throwaway git repositories -- `git init` in a temp directory, with
a bare repo as its origin. Mocking git here would test my model of git, and the two
failures that matter are both git's behaviour, not mine: what a rejected push does, and
what `git add` does when handed a pathspec.

The checks split into "does the job" and "cannot do harm", and the second half is the
reason the file is long. This code runs unattended at 16:00 on a machine holding API keys
in `.env`, and its whole justification is protecting a record it must not be able to
damage.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from . import archive


def _run(*args, cwd, check=True):
    p = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True)
    if check and p.returncode != 0:
        raise AssertionError(f"{args} failed in {cwd}: {p.stdout}{p.stderr}")
    return p


def _repo(with_origin=True) -> tuple[Path, Path | None]:
    """A repo with a committed baseline, a lab dir, an .env, and optionally an origin."""
    root = Path(tempfile.mkdtemp(prefix="archive_test_"))
    work = root / "work"
    work.mkdir()
    _run("git", "init", "-q", "-b", "main", cwd=work)
    _run("git", "config", "user.email", "t@t", cwd=work)
    _run("git", "config", "user.name", "t", cwd=work)
    (work / "live_lab_data").mkdir()
    (work / "live_lab_data" / "session_ledger.jsonl").write_text('{"date":"base"}\n')
    (work / "code.py").write_text("print('code')\n")
    # The thing that must never be committed by this module.
    (work / ".env").write_text("THETA_KEY=super-secret-do-not-commit\n")
    _run("git", "add", "-A", cwd=work)
    _run("git", "commit", "-qm", "baseline", cwd=work)

    bare = None
    if with_origin:
        bare = root / "origin.git"
        # -b main matters: without it the bare repo's HEAD points at refs/heads/master,
        # so `git clone` of it checks out an empty tree even though main exists. That
        # cost a debugging round here, and it is the same shape as the bugs this repo
        # keeps finding -- the clone SUCCEEDS, silently, with nothing in it.
        _run("git", "init", "-q", "--bare", "-b", "main", str(bare), cwd=root)
        _run("git", "remote", "add", "origin", str(bare), cwd=work)
        _run("git", "push", "-q", "-u", "origin", "main", cwd=work)
    return work, bare


_REAL_REPO = archive.REPO


def _with_repo(work):
    """Point the module at this repo. It resolves REPO at import time.

    Restored by `_restore_repo` at the end of the run. Leaving `archive.REPO` pointing at
    a deleted temp directory would make any later caller in the same process -- another
    test, an interactive session -- silently archive nothing, which is precisely the
    class of failure this module exists to prevent.
    """
    archive.REPO = work


def _restore_repo():
    archive.REPO = _REAL_REPO


def _log(work, n=1) -> list[str]:
    p = _run("git", "log", f"-{n}", "--format=%s", cwd=work)
    return p.stdout.strip().splitlines()


def _touch_session(work, text="line\n"):
    f = work / "live_lab_data" / "trades.jsonl"
    f.write_text(f.read_text() + text if f.exists() else text)


# ------------------------------------------------------------------- does the job

def test_commits_and_pushes_the_session_record():
    work, bare = _repo()
    _with_repo(work)
    _touch_session(work)
    r = archive.archive_session("2026-09-17")
    assert r["committed"] is True, r
    assert r["pushed"] is True, r
    assert _log(work)[0] == "Live lab record for 2026-09-17"
    # Actually landed on the remote, not just claimed.
    remote = _run("git", "log", "-1", "--format=%s", "main", cwd=bare).stdout.strip()
    assert remote == "Live lab record for 2026-09-17", remote


def test_an_empty_session_produces_no_commit():
    """A holiday or an aborted session has nothing to say, and a stream of empty
    commits would make the log useless for finding the day something happened."""
    work, _ = _repo()
    _with_repo(work)
    before = _run("git", "rev-parse", "HEAD", cwd=work).stdout
    r = archive.archive_session("2026-09-17")
    assert r["committed"] is False
    assert r["reason"] == "nothing to archive", r
    assert _run("git", "rev-parse", "HEAD", cwd=work).stdout == before


def test_the_index_is_left_clean_when_there_was_nothing_to_do():
    """`git add` ran, so the no-op path must unstage what it staged. Leaving the lab dir
    staged would make the NEXT hand-made commit silently include it."""
    work, _ = _repo()
    _with_repo(work)
    archive.archive_session("2026-09-17")
    assert _run("git", "diff", "--cached", "--quiet", cwd=work,
                check=False).returncode == 0, "left something staged"


def test_a_commit_happens_with_no_remote_at_all():
    """The commit is what makes the record durable, and it must not need a network."""
    work, _ = _repo(with_origin=False)
    _with_repo(work)
    _touch_session(work)
    r = archive.archive_session("2026-09-17")
    assert r["committed"] is True, r
    assert r["pushed"] is False, r
    assert "local only" in r["reason"], r


def test_works_with_no_git_identity_configured():
    """At 16:00 unattended, "Committer identity unknown" would lose the day's record.
    Identity is passed per-invocation with -c."""
    work, _ = _repo()
    _with_repo(work)
    _run("git", "config", "--unset", "user.email", cwd=work)
    _run("git", "config", "--unset", "user.name", cwd=work)
    _touch_session(work)
    r = archive.archive_session("2026-09-17", push=False)
    assert r["committed"] is True, r
    author = _run("git", "log", "-1", "--format=%an", cwd=work).stdout.strip()
    assert author == archive.BOT_NAME, author


# ------------------------------------------------------------------ cannot do harm

def test_never_commits_code_or_secrets():
    """The one that matters most. This runs unattended on a machine holding API keys,
    and `git add -A` on such a machine is how a credential reaches a public remote."""
    work, _ = _repo()
    _with_repo(work)
    _touch_session(work)
    (work / "code.py").write_text("print('MODIFIED, must not be committed')\n")
    (work / ".env").write_text("THETA_KEY=rotated-secret\n")
    (work / "new_secret.txt").write_text("also must not be committed\n")

    r = archive.archive_session("2026-09-17", push=False)
    assert r["committed"] is True, r
    files = _run("git", "show", "--name-only", "--format=", "HEAD",
                 cwd=work).stdout.split()
    assert files, "committed nothing"
    for f in files:
        assert f.startswith("live_lab_data/"), f"committed {f} from outside the lab dir"
    # And the edits outside are still sitting in the working tree, untouched.
    assert _run("git", "status", "--porcelain", cwd=work).stdout.count("code.py") == 1


def _race(work, bare, *, path="elsewhere.jsonl", text="from the other clone\n",
          subject="other clone wins the race"):
    """A second clone pushes first, so `work`'s next push is non-fast-forward.

    `path` decides whether the recovery merge is clean. A different file merges without
    trouble; the SAME file the lab is appending to produces a real conflict.
    """
    other = work.parent / "other"
    _run("git", "clone", "-q", str(bare), str(other), cwd=work.parent)
    _run("git", "config", "user.email", "o@o", cwd=other)
    _run("git", "config", "user.name", "o", cwd=other)
    (other / "live_lab_data" / path).write_text(text)
    _run("git", "add", "-A", cwd=other)
    _run("git", "commit", "-qm", subject, cwd=other)
    _run("git", "push", "-q", "origin", "main", cwd=other)
    return other


def _archive_fast(day, **kw):
    """archive_session without the real sleeps."""
    real = archive.PUSH_BACKOFF
    try:
        archive.PUSH_BACKOFF = (0, 0)
        return archive.archive_session(day, **kw)
    finally:
        archive.PUSH_BACKOFF = real


def _no_merge_in_progress(work):
    p = _run("git", "rev-parse", "--verify", "--quiet", "MERGE_HEAD",
             cwd=work, check=False)
    return p.returncode != 0


# ------------------------------------------- the one recovery, added 2026-09-24

def test_a_rejected_push_recovers_by_merging_and_keeps_the_other_side():
    """The remote moved ahead. Merge it, push, and lose nothing from either side.

    The assertion that matters is not "the push succeeded" -- it is that the other
    clone's commit is STILL REACHABLE afterwards. A force-push would also make the push
    succeed, and would silently delete someone else's session record.
    """
    work, bare = _repo()
    _with_repo(work)
    _race(work, bare)

    _touch_session(work)
    r = _archive_fast("2026-09-17")
    assert r["committed"] is True, r
    assert r["pushed"] is True, r
    assert r["reason"] == "pushed after merging the remote", r

    subjects = _log(work, 6)
    assert "Live lab record for 2026-09-17" in subjects
    assert "other clone wins the race" in subjects, "the other side was discarded"
    # and it survived on the REMOTE, which is the thing a force-push would have wiped
    remote = _run("git", "log", "-6", "--format=%s", "main", cwd=bare).stdout
    assert "other clone wins the race" in remote, f"remote was overwritten:\n{remote}"
    assert "Live lab record for 2026-09-17" in remote
    assert _no_merge_in_progress(work)
    _restore_repo()


def test_the_recovery_never_rebases_or_rewrites_history():
    """Both sides' exact commits must still be ancestors of the new head.

    THEIR sha catches a force-push. OUR sha is the one that catches a rebase -- rebase
    replays our commit onto theirs, so theirs survives untouched and a test that only
    checked theirs would pass against a rebase. An earlier version of this test did
    exactly that and proved nothing.
    """
    work, bare = _repo()
    _with_repo(work)
    other = _race(work, bare)
    theirs = _run("git", "rev-parse", "HEAD", cwd=other).stdout.strip()

    _touch_session(work)
    r = _archive_fast("2026-09-17")
    assert r["pushed"] is True, r
    ours = r["sha"]                      # captured at commit time, before any recovery

    for sha, whose in ((theirs, "theirs"), (ours, "ours")):
        exists = _run("git", "cat-file", "-e", sha, cwd=work, check=False)
        assert exists.returncode == 0, f"{whose} commit {sha} no longer exists: rewritten"
        anc = _run("git", "merge-base", "--is-ancestor", sha, "HEAD",
                   cwd=work, check=False)
        assert anc.returncode == 0, f"{whose} commit {sha} is not an ancestor: rewritten"
    # a merge keeps both lines, so the head must have two parents
    parents = _run("git", "rev-list", "--parents", "-1", "HEAD",
                   cwd=work).stdout.split()
    assert len(parents) == 3, f"head is not a merge commit: {parents}"
    _restore_repo()


def test_the_recovery_never_invokes_a_destructive_git_command():
    """Assert the contract on the argv, because behaviour cannot catch --force here.

    By the time the recovery pushes, the merge has already taken the remote's work in,
    so force-pushing a MERGE COMMIT loses nothing in this scenario and every behavioural
    assertion still passes. It would still destroy a commit that landed between the
    fetch and the push. A mutant that added --force passed the whole rest of this file,
    which is what a decorative test looks like.

    So this one reads what was actually run.
    """
    work, bare = _repo()
    _with_repo(work)
    _race(work, bare)
    _touch_session(work)

    calls = []
    real_git = archive._git

    def spy(*args, **kw):
        calls.append(args)
        return real_git(*args, **kw)

    archive._git = spy
    try:
        r = _archive_fast("2026-09-17")
    finally:
        archive._git = real_git
    assert r["pushed"] is True, r

    # Token checks, not substring checks on a joined string. The merge is invoked as
    # `-c user.name=... -c user.email=... merge --no-edit FETCH_HEAD`, so it does not
    # start with "merge", and a commit message containing the word "clean" would trip a
    # naive substring ban. An earlier version of this test did both.
    assert any("merge" in a for a in calls), f"no merge was run: {calls}"
    BANNED = {"--force", "-f", "--force-with-lease", "rebase", "--amend",
              "--hard", "-A", "--all", "clean", "filter-branch", "reflog"}
    for a in calls:
        bad = BANNED & set(a)
        assert not bad, f"destructive git command: {' '.join(a)!r} uses {sorted(bad)}"
    _restore_repo()


def test_a_conflicting_merge_is_aborted_and_the_commit_stays_local():
    """The old behaviour, preserved for the case that actually needs it.

    Both sides appended to the same file, so the merge conflicts. Nothing may be pushed,
    nothing on the remote may change, and the repo must NOT be left mid-merge -- that
    would drop the next session into a broken working tree at 09:30.
    """
    work, bare = _repo()
    _with_repo(work)
    _race(work, bare, path="trades.jsonl", text="their conflicting line\n",
          subject="conflicting write")

    _touch_session(work, "our conflicting line\n")
    r = _archive_fast("2026-09-17")
    assert r["committed"] is True, r
    assert r["pushed"] is False, r
    assert _no_merge_in_progress(work), "left the repo mid-merge"
    remote = _run("git", "log", "-1", "--format=%s", "main", cwd=bare).stdout.strip()
    assert remote == "conflicting write", f"remote changed: {remote}"
    assert _log(work)[0] == "Live lab record for 2026-09-17"
    _restore_repo()


def test_the_recovery_is_skipped_when_tracked_files_are_modified():
    """Never run a merge across someone's uncommitted edits."""
    work, bare = _repo()
    _with_repo(work)
    _race(work, bare)
    (work / "code.py").write_text("print('half-finished edit')\n")

    _touch_session(work)
    r = _archive_fast("2026-09-17")
    assert r["committed"] is True, r
    assert r["pushed"] is False, r
    assert (work / "code.py").read_text() == "print('half-finished edit')\n"
    assert _no_merge_in_progress(work)
    _restore_repo()


def test_the_recovery_is_not_attempted_when_the_remote_is_unreachable():
    """A dead network is not fixed by merging, and this machine's network is the reason
    this file exists. Must fall through quietly rather than hang or raise.

    The NO-FETCH assertion is the one that earns its place. Without the non-fast-forward
    guard the outcome is identical -- the fetch just fails too and the result dict is the
    same -- so every outcome-based assertion here passes against the unguarded version.
    What the guard actually buys is not correctness but TIME: it avoids a fetch that can
    block for the full 180-second timeout, at 16:00, on the travelling laptop whose
    network is why this module exists. Only the argv shows that.
    """
    work, _bare = _repo()
    _with_repo(work)
    _run("git", "remote", "set-url", "origin",
         str(work.parent / "no-such-remote.git"), cwd=work)
    _touch_session(work)

    calls = []
    real_git = archive._git

    def spy(*args, **kw):
        calls.append(args)
        return real_git(*args, **kw)

    archive._git = spy
    try:
        r = _archive_fast("2026-09-17")
    finally:
        archive._git = real_git

    assert r["committed"] is True, r
    assert r["pushed"] is False, r
    assert "local only" in r["reason"], r
    assert _no_merge_in_progress(work)
    assert not any("fetch" in a for a in calls), \
        f"fetched anyway on a dead network: {[' '.join(a) for a in calls]}"
    _restore_repo()


def test_setting_recover_from_non_ff_false_restores_the_old_behaviour():
    work, bare = _repo()
    _with_repo(work)
    _race(work, bare)
    _touch_session(work)
    real = archive.RECOVER_FROM_NON_FF
    try:
        archive.RECOVER_FROM_NON_FF = False
        r = _archive_fast("2026-09-17")
    finally:
        archive.RECOVER_FROM_NON_FF = real
    assert r["committed"] is True, r
    assert r["pushed"] is False, r
    remote = _run("git", "log", "-1", "--format=%s", "main", cwd=bare).stdout.strip()
    assert remote == "other clone wins the race"
    _restore_repo()


def test_refuses_to_push_from_a_detached_head():
    """Committed locally is right; guessing a branch to push to is not."""
    work, _ = _repo()
    _with_repo(work)
    _touch_session(work)
    sha = _run("git", "rev-parse", "HEAD", cwd=work).stdout.strip()
    _run("git", "checkout", "-q", sha, cwd=work)
    r = archive.archive_session("2026-09-17")
    assert r["committed"] is True, r
    assert r["pushed"] is False, r
    assert "detached" in r["reason"], r


def test_never_raises_when_the_directory_is_not_a_repo():
    """An archive step that can kill a session trades the thing being protected for the
    protection."""
    d = Path(tempfile.mkdtemp(prefix="archive_test_norepo_"))
    _with_repo(d)
    r = archive.archive_session("2026-09-17")
    assert r["committed"] is False
    assert r["pushed"] is False
    assert r["reason"], "no reason recorded"


def test_never_raises_when_git_is_missing_entirely():
    """A PATH without git must degrade, not explode."""
    work, _ = _repo()
    _with_repo(work)
    _touch_session(work)
    real = archive._git
    try:
        archive._git = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no git"))
        r = archive.archive_session("2026-09-17")
    finally:
        archive._git = real
    assert r["committed"] is False
    assert "swallowed" in r["reason"] or r["reason"], r


def test_autostart_archives_by_default_and_ignores_the_result():
    """Two things at once: the safeguard must be on by default, because the hole it
    exists to prevent was caused by a manual step; and its outcome must never reach the
    session's exit code, because that reports the session, not the bookkeeping."""
    import inspect

    from . import autostart as A
    src = inspect.getsource(A.main)
    assert "archive_session" in src, "autostart never archives"
    assert "--no-archive" in inspect.getsource(A.main), "archiving is not opt-out"
    i = src.index("archive_session")
    tail = src[i:]
    assert "rc =" not in tail.split("return rc")[0], \
        "the archive result can reach the session exit code"
    assert src.index("note_finished") < i, \
        "archive runs before the terminal ledger line it is supposed to commit"


import pytest


@pytest.fixture(autouse=True)
def _repo_isolation():
    """Restore archive.REPO after every check, whichever runner is driving."""
    yield
    _restore_repo()


CHECKS = [(n, f) for n, f in sorted(globals().items())
          if n.startswith("test_") and callable(f)]


def main() -> int:
    ok = True
    print("=" * 78)
    print("END-OF-SESSION ARCHIVE -- REAL GIT REPOSITORIES")
    print("=" * 78)
    for name, fn in CHECKS:
        try:
            fn()
            print(f"  [PASS] {name[5:].replace('_', ' ')}")
        except AssertionError as exc:
            ok = False
            print(f"  [FAIL] {name[5:].replace('_', ' ')}\n         {exc}")
        except Exception as exc:                              # noqa: BLE001
            ok = False
            print(f"  [FAIL] {name[5:].replace('_', ' ')}\n         raised {exc!r}")
        finally:
            _restore_repo()
    print(f"\n  {len(CHECKS)} checks")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    print("\n  A session that exists on one disk is one disk failure from never having "
          "happened.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
