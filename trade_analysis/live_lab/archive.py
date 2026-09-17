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
  * **Never push to a branch it was not told to push to**, and never force. The default
    is the current branch, and a rejected push is left rejected: two sessions' commits
    are still both present locally and the next successful push carries them. Automated
    conflict resolution on a research record is not a thing I am willing to write.
  * **Never amend, rebase or rewrite.** The record is append-only by design.
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
