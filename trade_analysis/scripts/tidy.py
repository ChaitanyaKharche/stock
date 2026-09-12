"""Remove build clutter from an ALLOWLIST. Never sweep by age alone.

    python -m trade_analysis.scripts.tidy                 # dry run, always
    python -m trade_analysis.scripts.tidy --days 14
    python -m trade_analysis.scripts.tidy --apply         # actually delete

WHY THIS IS AN ALLOWLIST AND NOT AN AGE RULE
--------------------------------------------
The request was "delete files not in use older than 7-10 days". Scanned before running,
that rule would have deleted 2.3 GB of irreplaceable data:

    Desktop/data/raw/option_quote_1m_0dte    22 days   667 MB
    Desktop/data/raw/stock_ohlc_1m           22 days   1.4 GB
    research_data/.../data_opt.parquet       37 days   252 MB

All three are past any sensible cutoff and all three are "not in use" by mtime. The 0DTE
option archive CANNOT be re-downloaded -- the ThetaData options entitlement is on the FREE
tier and 403s on every quote endpoint -- and the Vilkov panel is licensed
academic-replication-only. **mtime is not a proxy for value.** This project has already
lost work to exactly that assumption once: the /scratch purge on Discovery destroyed the
original TFT model-definition source, which is why models/tft_model.py is a reconstruction.

So the design is inverted. Nothing is deletable unless it matches ALLOW, and nothing under
DENY is deletable even if it matches ALLOW. Age is a secondary filter applied after both,
never a primary criterion.

DENY WINS. It is checked first and it is checked against the RESOLVED absolute path, so a
symlink or a `..` cannot walk out of it.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Deletable. Directory names are matched anywhere in the tree; glob patterns are matched
# relative to the repo root. Everything here is either regenerated automatically or is a
# transient artifact of running something.
ALLOW_DIR_NAMES = ["__pycache__", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints"]
ALLOW_GLOBS = [
    "*.out", "*.err",                       # Slurm logs that land in the submission dir
    ".playwright-mcp/*",                    # browser-automation page snapshots
    "**/*.pyc", "**/*.pyo",
]

# NEVER deletable, checked first, checked on the resolved path. These are the inputs the
# project cannot rebuild, plus the durable record of the forward test.
DENY_PARTS = [
    ROOT / "data",                          # derived frames (expensive to rebuild)
    ROOT / "live_lab_data",                 # the frozen forward test's durable record
    ROOT / "trained_models",
    ROOT / "models",
    ROOT / "research",
    ROOT / "venv",
    ROOT / ".git",
    Path.home() / "Documents" / "research_data",          # Vilkov panel, licensed
    Path.home() / "Desktop" / "data",                     # raw option + stock archives
]


def denied(p: Path) -> Path | None:
    """Return the DENY root that protects p, or None. Resolved, so symlinks cannot escape."""
    try:
        rp = p.resolve()
    except OSError:
        return None
    for d in DENY_PARTS:
        try:
            rp.relative_to(d.resolve())
            return d
        except (ValueError, OSError):
            continue
    return None


def size_of(p: Path) -> int:
    if p.is_file():
        try:
            return p.stat().st_size
        except OSError:
            return 0
    tot = 0
    for dp, _, fns in os.walk(p, onerror=lambda e: None):
        for fn in fns:
            try:
                tot += (Path(dp) / fn).stat().st_size
            except OSError:
                pass
    return tot


def candidates(days: int) -> tuple[list[Path], dict[str, int]]:
    """Returns (deletable, {deny_root: refusal_count}).

    Three things this gets right that the first version did not:

    * **The glob pass prunes DENY too.** `**/*.pyc` globs the entire tree including venv/,
      so the first version considered and then refused 14,149 files one at a time and
      printed the refusals. A tidiness tool that emits 14k lines of noise is not tidy.
      Refusals are now COUNTED per deny-root, not listed.
    * **Nested matches are de-duplicated.** `__pycache__` and `__pycache__/x.pyc` both
      matched, and both were size-counted, so the total double-counted every byte. A path
      whose ancestor is already scheduled is dropped.
    * **Refusals are reported by full deny root**, because two different roots here are
      both named `data` (the repo's and Desktop's) and the short name was ambiguous.
    """
    cutoff = time.time() - days * 86400
    keep, refused, seen = [], {}, set()

    def consider(p: Path):
        if p in seen:
            return
        seen.add(p)
        d = denied(p)
        if d is not None:
            refused[str(d)] = refused.get(str(d), 0) + 1
            return
        try:
            if p.stat().st_mtime > cutoff:
                return
        except OSError:
            return
        keep.append(p)

    for dp, dirnames, _ in os.walk(ROOT, onerror=lambda e: None):
        cur = Path(dp)
        if denied(cur) is not None:
            refused[str(denied(cur))] = refused.get(str(denied(cur)), 0) + 1
            dirnames[:] = []          # do not descend into protected trees at all
            continue
        for dn in list(dirnames):
            if dn in ALLOW_DIR_NAMES:
                consider(cur / dn)
    for g in ALLOW_GLOBS:
        for p in ROOT.glob(g):
            if denied(p) is not None:
                continue              # counted by the walk; do not re-list per file
            consider(p)

    # drop anything already covered by a scheduled ancestor, so sizes do not double-count
    sched = set(keep)
    return [p for p in keep if not any(a in sched for a in p.parents)], refused


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=10,
                    help="only remove allowlisted items older than this (default 10)")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete. Without it this is a dry run and always was.")
    args = ap.parse_args(argv)

    keep, refused = candidates(args.days)
    print(f"  repo root : {ROOT}")
    print(f"  mode      : {'APPLY -- WILL DELETE' if args.apply else 'DRY RUN'}")
    print(f"  age filter: older than {args.days} days")
    print(f"  allowlist : dirs {ALLOW_DIR_NAMES}")
    print(f"              globs {ALLOW_GLOBS}")

    if refused:
        print(f"\n  DENY roots that blocked matches (counts, not a list):")
        for root, n in sorted(refused.items(), key=lambda kv: -kv[1]):
            print(f"    {n:>6} blocked under {root}")

    if not keep:
        print("\n  nothing to remove. The tree is already tidy.")
        return 0

    total = 0
    print(f"\n  {len(keep)} item(s) eligible:")
    for p in sorted(keep):
        s = size_of(p)
        total += s
        rel = p.relative_to(ROOT) if ROOT in p.parents or p.parent == ROOT else p
        print(f"    {str(rel):<60} {s / 1024:8.1f} KB")
    print(f"\n  total {total / 1024 / 1024:.1f} MB")

    if not args.apply:
        print("\n  DRY RUN -- nothing deleted. Re-run with --apply if this list is right.")
        return 0

    removed = 0
    for p in keep:
        if denied(p) is not None:            # re-check at the moment of deletion
            print(f"    REFUSING {p} -- DENY matched on the second check")
            continue
        try:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
            removed += 1
        except OSError as e:
            print(f"    could not remove {p}: {e}")
    print(f"\n  removed {removed} of {len(keep)} item(s), {total / 1024 / 1024:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
