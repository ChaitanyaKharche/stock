"""Keep the files that exist in BOTH projects byte-identical.

    python huggingface_space/sync_shared.py            # report drift, change nothing
    python huggingface_space/sync_shared.py --push     # repo root  ->  huggingface_space
    python huggingface_space/sync_shared.py --pull     # huggingface_space -> repo root

Run it before every upload. It exits non-zero when the folder is not deployable, so it
works as a pre-upload gate rather than something you have to remember to read.

The Space and the live lab are deliberately separate deployments: the Space installs from
its own minimal requirements.txt and must not import the lab, and nothing done to make a
demo look good may reach back and perturb a frozen forward test. So the overlap is held as
COPIES, not imports or symlinks.

Copies drift. That is the entire cost of this arrangement, and the only defence is a tool
that notices. Run this before uploading to Hugging Face.

Direction is never inferred. `--push` and `--pull` are explicit because guessing which
side is authoritative is how one gets silently overwritten.
"""
from __future__ import annotations

import argparse
import filecmp
import hashlib
import shutil
import sys
from pathlib import Path

SPACE = Path(__file__).resolve().parent
ROOT = SPACE.parent

# (path relative to repo root, path relative to huggingface_space, git-tracked in Space?)
#
# trained_models is GIT_IGNORED on the Space side: the same 19 MB of weights is already
# tracked at the repo root, and git keeps blobs forever, so committing them twice would
# double them in every clone permanently. The Space copy is a build artifact this script
# regenerates -- which only works if something refuses to call the folder deployable when
# it is missing. That is what UPLOAD_REQUIRED is for.
SHARED = [
    ("local_data", "local_data", True),
    ("trained_models", "trained_models", False),
    ("trade_analysis/live_lab/indicators.py", "trade_analysis/lab_indicators.py", True),
    # The US equity calendar. The Space needs it to say WHICH session a signal is from --
    # without it a Labor Day visitor sees Friday's tape presented as today's. Held as a
    # copy for the same reason as the rest: the Space must build with nothing from the
    # parent repo. It is the file most likely to drift, because holidays get appended to
    # the root copy each year and nothing would otherwise notice the Space's copy aging.
    ("trade_analysis/bulk_download/trading_days.py", "trade_analysis/trading_days.py", True),
]

# Present on disk before uploading, whether or not this repo versions them.
UPLOAD_REQUIRED = ["trained_models", "local_data", "trade_analysis/lab_indicators.py"]

# Files the ROOT keeps but the Space deliberately does NOT ship, so their absence is a
# decision rather than drift. All five fail `torch.load(weights_only=True)` -- they are full
# pickles, not plain tensors -- so tft_model.py could never load them anyway, and they are
# exactly the files Hugging Face flags without a "Safe" badge. Shipping 8 MB of
# unloadable pickles as the only security warnings on a public repo is strictly worse than
# not shipping them.
#
# The root keeps them because trade_analysis/models/tft_backtest.py reads
# tft_AMZN_e200_.pth directly. Do not "fix" this by deleting them there.
SPACE_EXCLUDES = {
    "trained_models": {
        "tft_AMZN_e200_.pth", "tft_MSFT_e200_.pth", "tft_SPY_e200_.pth",
        "tft_TSLA_e200_.pth", "tft_model.pth",
    }
}

# lab_indicators.py carries a provenance banner the lab's own copy must not have, so it is
# compared on the CODE below the banner rather than byte-for-byte.
BANNER_EXEMPT = {"trade_analysis/lab_indicators.py"}


def _digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def _body(p: Path) -> str:
    """Everything after the module docstring, so a differing banner does not read as drift."""
    txt = p.read_text(encoding="utf-8")
    parts = txt.split('"""')
    return '"""'.join(parts[2:]) if len(parts) >= 3 else txt


def compare(root_rel: str, space_rel: str) -> tuple[str, list[str]]:
    a, b = ROOT / root_rel, SPACE / space_rel
    if not a.exists():
        return "MISSING_ROOT", []
    if not b.exists():
        return "MISSING_SPACE", []

    if a.is_dir():
        excl = SPACE_EXCLUDES.get(space_rel, set())
        cmp = filecmp.dircmp(str(a), str(b))
        # left_only minus the deliberate exclusions: a file the root has and the Space
        # intentionally omits is not drift.
        diffs = (list(cmp.diff_files)
                 + [f for f in cmp.left_only if f not in excl]
                 + list(cmp.right_only))
        # dircmp is shallow by default: same size + mtime counts as equal. Re-check the
        # ones it passed by content, because a same-size edit is exactly the drift that
        # matters and exactly the one it would miss.
        for name in cmp.common_files:
            if name in diffs:
                continue
            if _digest(a / name) != _digest(b / name):
                diffs.append(name)
        return ("OK" if not diffs else "DRIFT"), sorted(diffs)

    if space_rel in BANNER_EXEMPT:
        return ("OK" if _body(a) == _body(b) else "DRIFT"), []
    return ("OK" if _digest(a) == _digest(b) else "DRIFT"), []


def copy(root_rel: str, space_rel: str, push: bool) -> None:
    a, b = ROOT / root_rel, SPACE / space_rel
    src, dst = (a, b) if push else (b, a)
    if space_rel in BANNER_EXEMPT:
        print(f"    SKIP {space_rel} (banner differs by design; re-copy by hand)")
        return
    if src.is_dir():
        excl = SPACE_EXCLUDES.get(space_rel, set()) if push else set()
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file() and f.name not in excl:
                shutil.copy2(f, dst / f.name)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print(f"    copied {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--push", action="store_true", help="repo root -> huggingface_space")
    g.add_argument("--pull", action="store_true", help="huggingface_space -> repo root")
    args = ap.parse_args()

    mode = "PUSH" if args.push else "PULL" if args.pull else "CHECK"
    print("=" * 82)
    print(f"SHARED-FILE SYNC  [{mode}]   root={ROOT}")
    print("=" * 82)

    drift = 0
    for root_rel, space_rel, tracked in SHARED:
        status, files = compare(root_rel, space_rel)
        label = root_rel if root_rel == space_rel else f"{root_rel}  <->  {space_rel}"
        if not tracked:
            label += "   [git-ignored here; regenerated, not committed]"
        print(f"  [{status:<13}] {label}")
        for f in files[:10]:
            print(f"        - {f}")
        if len(files) > 10:
            print(f"        ... and {len(files) - 10} more")
        if status != "OK":
            drift += 1
            if args.push or args.pull:
                copy(root_rel, space_rel, args.push)

    # Deployability is a separate question from drift. A fresh clone has no
    # trained_models/ on the Space side at all -- correctly, it is git-ignored -- and
    # that is not "drift", it is "not built yet". Uploading in that state would ship a
    # Space whose TFT silently degrades to _default_prediction() for every symbol, which
    # is precisely the class of quiet fallback this project already got burned by once.
    print("\n  UPLOAD READINESS")
    missing = [p for p in UPLOAD_REQUIRED if not (SPACE / p).exists()]
    for p in UPLOAD_REQUIRED:
        here = SPACE / p
        n = len(list(here.iterdir())) if here.is_dir() else (1 if here.exists() else 0)
        print(f"    [{'OK     ' if here.exists() else 'MISSING'}] {p}"
              + (f"  ({n} files)" if here.is_dir() else ""))
    if missing:
        print("    -> NOT deployable. Run:  python huggingface_space/sync_shared.py --push")

    if mode == "CHECK":
        print(f"\n  {drift} of {len(SHARED)} shared paths differ."
              + ("  Run with --push or --pull." if drift else "  No drift."))
    else:
        print(f"\n  {mode} complete; re-run without a flag to confirm.")
    return 1 if ((mode == "CHECK" and drift) or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
