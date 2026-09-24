"""Commit and push the day's forward-test record, so it cannot exist on one disk only.

WHY THIS EXISTS
---------------
On 2026-09-16 the repository's committed record stopped at 2026-09-08 while the session
ledger, the trades and the daily files for six later sessions existed only on the lab
machine. `research/live_lab_coverage_audit.md` has the detail. Worse, the coverage tool
reported `usable 7/7 = 100.0%` throughout, because its denominator was the ledger itself.

The gate is 60 sessions. Six is 10% of the entire experiment, and it is the one input the
project cannot buy back: the options arm is already dead with ThetaData lapsed to
`Options: FREE`, and CI width scales as sigma/sqrt(n_sessions), with sessions, not trades.
A session that exists only on a travelling laptop is a session one disk failure from
never having happened.

Committing by hand is not a plan. It was the plan, and it produced a six-session hole.

WHAT IT WILL NOT DO, AND WHY EACH MATTERS
-----------------------------------------
  * **Never raise.** Every path is wrapped and swallowed. An archive step that can kill a
    session is strictly worse than no archive step -- it would trade the thing being
    protected for the protection.
  * **Never commit anything but the lab data directory.** Explicit pathspec, never
    `git add -A`. The repo holds API keys in `.env`, a `local_data/` tree and a Theta
    Terminal path; a broad add on an unattended machine at 16:00 is how a credential
    reaches a public remote.
  * **Never push to a branch it was not told to push to**, and never force.
  * **Never amend, rebase or rewrite.** The record is append-only by design.

ONE RECOVERY, ADDED 2026-09-24, AND ITS LIMITS
----------------------------------------------
The push failed non-fast-forward on four consecutive sessions (2026-09-21 to 09-24)
because a research session pushed to the same branch while the lab was running. Each
time the record sat on the travelling laptop until it was pushed by hand. Depending on
someone remembering is the failure this file exists to prevent.

So after a rejected push, and **only** when git says the remote moved ahead, there is
exactly one recovery: fetch, **merge**, push once. It is deliberately narrow:

  * **Merge only.** Never rebase, never force, never amend. A merge commit leaves every
    existing commit reachable, so the other side's work cannot be lost.
  * **Only for a non-fast-forward rejection.** A dead network or a permissions error is
    not fixed by merging, and this machine's network is the reason the file exists.
  * **Refuses on a dirty tree.** If tracked files differ from HEAD, the merge is skipped
    rather than run across someone's uncommitted edits.
  * **Aborts on conflict**, and leaves no merge in progress. The outcome then is exactly
    the old behaviour: commit local, remote untouched, next push carries both.
  * **One attempt.** No loop. If the second push also loses a race, it waits for
    tomorrow.

Set `RECOVER_FROM_NON_FF = False` to get the old terminal behaviour back.
  * **Never commit an empty change.** A holiday or an aborted session leaves nothing to
    say, and a stream of empty commits would make the log useless for finding the day
    something actually happened.

THE PUSH IS ALLOWED TO FAIL
---------------------------
This runs at 16:00 on a machine whose network is the reason this file exists. So the
commit and the push are separate outcomes: the commit is what makes the record durable
against losing the working tree, and it happens with no network at all. The push is
best-effort with a short backoff, and its failure is logged rather than retried into the
evening.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Identity passed per-invocation with `-c`, so a machine with no configured git user
# still commits rather than failing with "Committer identity unknown" at 16:00.
BOT_NAME = "live-lab-archive"
BOT_EMAIL = "live-lab-archive@localhost"
PUSH_BACKOFF = (2, 8, 20)      # seconds; three tries, then leave it for tomorrow
RECOVER_FROM_NON_FF = True     # set False for the pre-2026-09-24 terminal behaviour
# What git says when the remote has moved ahead. Recovery is attempted for THIS failure
# and nothing else -- merging does not fix a dead network or a rejected credential.
_NON_FF = ("non-fast-forward", "fetch first", "rejected")


def _git(*args, timeout=120) -> tuple[int, str]:
    """Run git in the repo. Returns (rc, combined output). Never raises."""
    try:
        p = subprocess.run(("git", "-C", str(REPO)) + args, capture_output=True,
                           text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except Exception as exc:                                 # noqa: BLE001
        return 127, repr(exc)


def _branch() -> str | None:
    rc, out = _git("rev-parse", "--abbrev-ref", "HEAD")
    if rc != 0 or not out or out == "HEAD":
        return None                 # detached; pushing from here is never what was meant
    return out


def _merging() -> bool:
    """True when a merge is in progress. Asked via git, not by looking for .git/MERGE_HEAD,
    because .git is a file rather than a directory in a worktree."""
    rc, _ = _git("rev-parse", "--verify", "--quiet", "MERGE_HEAD")
    return rc == 0


def _tracked_changes() -> bool:
    """True when tracked files differ from HEAD, or the question could not be answered.

    Conservative on purpose: an unreadable answer counts as dirty, so the merge is
    skipped rather than run over someone's uncommitted work.
    """
    rc, _ = _git("diff", "--quiet", "HEAD")
    return rc != 0


def _merge_remote_and_retry(target: str, log) -> bool:
    """One conservative recovery from a non-fast-forward push: fetch, merge, push once.

    Returns True only if the push finally succeeded. Every early exit leaves the
    repository exactly as it was: commit local, remote untouched, no merge in progress.
    """
    if _tracked_changes():
        log("[archive] recovery skipped: tracked files are modified")
        return False

    rc, out = _git("fetch", "origin", target, timeout=180)
    if rc != 0:
        tail = out.splitlines()[-1][:140] if out else ""
        log(f"[archive] recovery skipped: fetch failed: {tail}")
        return False

    rc, out = _git("-c", f"user.name={BOT_NAME}", "-c", f"user.email={BOT_EMAIL}",
                   "merge", "--no-edit", "FETCH_HEAD")
    if rc != 0 or _merging():
        tail = out.splitlines()[-1][:140] if out else ""
        log(f"[archive] recovery gave up, merge was not clean: {tail}")
        if _merging():
            _git("merge", "--abort")
            log("[archive] merge aborted; commit stays local")
        return False

    rc, out = _git("push", "origin", f"HEAD:refs/heads/{target}", timeout=180)
    if rc != 0:
        tail = out.splitlines()[-1][:140] if out else ""
        log(f"[archive] recovery push failed: {tail}")
        return False
    return True


def archive_session(day, lab_dir="live_lab_data", push: bool = True,
                    branch: str | None = None, log=print) -> dict:
    """Commit `lab_dir` for `day` and try to push. Returns a result dict; never raises.

    `result["committed"]` and `result["pushed"]` are separate on purpose: a commit with a
    failed push is the expected outcome on a bad network and is NOT a failure of this
    step. The data is durable either way.
    """
    result = {"committed": False, "pushed": False, "reason": "", "sha": ""}
    try:
        rel = Path(lab_dir).as_posix()

        rc, out = _git("rev-parse", "--is-inside-work-tree")
        if rc != 0:
            result["reason"] = f"not a git work tree: {out[:120]}"
            log(f"[archive] {result['reason']}")
            return result

        # Only the lab data. Anything staged by hand outside it is left alone, which is
        # why this stages an explicit pathspec rather than using the index as found.
        rc, out = _git("add", "--", rel)
        if rc != 0:
            result["reason"] = f"git add failed: {out[:200]}"
            log(f"[archive] {result['reason']}")
            return result

        rc, _ = _git("diff", "--cached", "--quiet", "--", rel)
        if rc == 0:
            result["reason"] = "nothing to archive"
            log("[archive] nothing to archive")
            _git("reset", "--quiet", "--", rel)
            return result

        rc, stat = _git("diff", "--cached", "--shortstat", "--", rel)
        msg = (f"Live lab record for {day}\n\n"
               f"Written by trade_analysis/live_lab/archive.py at the end of the "
               f"session.\n{stat.strip()}\n")
        rc, out = _git("-c", f"user.name={BOT_NAME}", "-c", f"user.email={BOT_EMAIL}",
                       "commit", "-m", msg, "--", rel)
        if rc != 0:
            result["reason"] = f"git commit failed: {out[:200]}"
            log(f"[archive] {result['reason']}")
            return result
        result["committed"] = True
        _, result["sha"] = _git("rev-parse", "--short", "HEAD")
        log(f"[archive] committed {result['sha']}  {stat.strip()}")

        if not push:
            result["reason"] = "push disabled"
            return result

        target = branch or _branch()
        if not target:
            result["reason"] = "detached HEAD; committed locally, not pushing"
            log(f"[archive] {result['reason']}")
            return result

        for i, wait in enumerate((0,) + PUSH_BACKOFF):
            if wait:
                time.sleep(wait)
            rc, out = _git("push", "origin", f"HEAD:refs/heads/{target}", timeout=180)
            if rc == 0:
                result["pushed"] = True
                log(f"[archive] pushed to {target}")
                return result
            log(f"[archive] push attempt {i + 1} failed: {out.splitlines()[-1][:140]}"
                if out else f"[archive] push attempt {i + 1} failed")

        # The remote moving ahead is the one rejection a merge can fix, and it is the one
        # that actually happened, four sessions running. Anything else falls straight
        # through to the terminal message below.
        if RECOVER_FROM_NON_FF and out and any(h in out.lower() for h in _NON_FF):
            log("[archive] remote moved ahead; one merge-and-retry")
            if _merge_remote_and_retry(target, log):
                result["pushed"] = True
                result["reason"] = "pushed after merging the remote"
                log(f"[archive] pushed to {target} after merging remote")
                return result

        # Deliberately terminal. The commit is local and safe; the next session's push
        # carries it. Rebasing a research record unattended is not worth the risk.
        result["reason"] = ("pushed nothing: commit is local only, "
                            "will go with the next successful push")
        log(f"[archive] {result['reason']}")
        return result
    except Exception as exc:                                 # noqa: BLE001
        result["reason"] = f"archive raised and was swallowed: {exc!r}"
        log(f"[archive] {result['reason']}")
        return result


def main(argv=None) -> int:
    """Run the archive by hand. Useful for backfilling a gap from the lab machine."""
    import argparse
    import datetime as dt

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--day", default=None, help="label for the commit; default today ET")
    ap.add_argument("--lab-dir", default="live_lab_data")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--branch", default=None)
    args = ap.parse_args(argv)

    day = args.day
    if day is None:
        try:
            from .clock import now_et
            day = now_et().date().isoformat()
        except Exception:                                    # noqa: BLE001
            day = dt.date.today().isoformat()

    r = archive_session(day, args.lab_dir, push=not args.no_push, branch=args.branch)
    print(f"\n  committed={r['committed']} pushed={r['pushed']} {r['reason']}")
    # 0 whenever the data is durable. An unpushed commit is not a failure of this step.
    return 0 if (r["committed"] or r["reason"] == "nothing to archive") else 1


if __name__ == "__main__":
    raise SystemExit(main())
