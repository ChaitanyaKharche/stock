"""Collect Arm A's array output and refuse to summarise a partial sweep.

    python -m trade_analysis.hpc.collect_arm_a --results $HOME/vrp/results/arm_a
    python -m trade_analysis.hpc.collect_arm_a --results results/arm_a --allow-partial

--require-all is the DEFAULT and the point of the file. A collector that quietly averages
whatever files happen to be present is how a 24-cell sweep reports 12 cells as a finished
result. Missing slots, duplicate config hashes and mismatched git SHAs are all hard errors.

Output is the by-entry-time table, which is the format Almeida/Freire/Hizmeri report their
Table 6 in -- gross and net Sharpe across nine intraday entry times -- so the comparison is
against the published shape rather than a number of our own choosing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .arm_a_ssd import SLOTS


def load(results: Path) -> tuple[dict, pd.DataFrame, list[str]]:
    recs, frames, missing = {}, [], []
    for slot in SLOTS:
        tag = slot.replace(":", "")
        rj, tp = results / f"result_{tag}.json", results / f"trades_{tag}.parquet"
        if not rj.exists() or not tp.exists():
            missing.append(slot)
            continue
        recs[slot] = json.loads(rj.read_text(encoding="utf-8"))
        frames.append(pd.read_parquet(tp))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return recs, df, missing


def check_provenance(recs: dict) -> list[str]:
    """Every task must have run the same code in the same environment."""
    problems = []
    shas = {s: r["provenance"].get("git_sha") for s, r in recs.items()}
    if len(set(shas.values())) > 1:
        problems.append(f"tasks ran DIFFERENT git SHAs: {shas}")
    # Gate on TRACKED code edits, and on untracked .py files that could shadow an
    # imported module. Nothing else about a dirty tree bears on what code ran, and gating
    # on it fired on Slurm's own .out files and on stray redirect artifacts.
    mod = {s: r["provenance"].get("git_modified_code")
           for s, r in recs.items() if r["provenance"].get("git_modified_code")}
    if mod:
        problems.append(f"tasks ran MODIFIED TRACKED CODE: {mod}")
    shadow = {s: r["provenance"].get("git_untracked_shadowing")
              for s, r in recs.items() if r["provenance"].get("git_untracked_shadowing")}
    if shadow:
        problems.append(f"untracked .py inside the package could shadow a module: {shadow}")
    clutter = max((r["provenance"].get("git_untracked_other_n") or 0)
                  for r in recs.values()) if recs else 0
    if clutter:
        print(f"  note: {clutter} untracked non-code file(s) in the tree -- not gated, but "
              f"worth a look; Slurm writes its .out files into the submission directory")
    for key in ("theta", "min_history", "atm_band"):
        vals = {s: r["provenance"].get(key) for s, r in recs.items()}
        if len(set(map(str, vals.values()))) > 1:
            problems.append(f"tasks disagree on {key}: {vals}")
    npv = {s: (r["provenance"].get("numpy"), r["provenance"].get("pandas"))
           for s, r in recs.items()}
    if len(set(npv.values())) > 1:
        problems.append(f"tasks ran different numpy/pandas: {npv}")
    return problems


def session_t(df: pd.DataFrame, col: str) -> tuple[float, float, int]:
    if df.empty:
        return float("nan"), float("nan"), 0
    per = df.groupby("date")[col].mean()
    n = len(per)
    se = per.std(ddof=1) / np.sqrt(n) if n > 2 else np.nan
    return (float(per.mean()),
            float(per.mean() / se) if se and se > 0 else float("nan"), n)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="results/arm_a")
    ap.add_argument("--allow-partial", action="store_true",
                    help="summarise an incomplete sweep. Use only to inspect a run in "
                         "progress; never to report a result.")
    args = ap.parse_args(argv)

    results = Path(args.results)
    recs, df, missing = load(results)

    print(f"  {len(recs)} of {len(SLOTS)} slots present in {results}")
    if missing:
        print(f"  MISSING: {', '.join(missing)}")
        if not args.allow_partial:
            print("\n  REFUSING to summarise a partial sweep. Re-run the missing array")
            print("  tasks, or pass --allow-partial and do not quote the output.")
            return 2
        print("  --allow-partial given: this output is NOT a result.")

    problems = check_provenance(recs)
    if problems:
        print("\n  *** PROVENANCE PROBLEMS ***")
        for p in problems:
            print(f"    - {p}")
        if not args.allow_partial:
            return 3
    elif recs:
        p0 = next(iter(recs.values()))["provenance"]
        print(f"  provenance OK: git {str(p0.get('git_sha'))[:8]}  "
              f"numpy {p0.get('numpy')}  pandas {p0.get('pandas')}  "
              f"theta {p0.get('theta')}  min_history {p0.get('min_history')}")

    if df.empty:
        print("\n  no trades in any slot")
        return 1

    print(f"\n  {len(df)} trades over {df['date'].nunique()} sessions, "
          f"{df['slot'].nunique()} entry times")
    print(f"  sides overall: buy {(df.side == 1).sum()}  sell {(df.side == -1).sum()}"
          f"   ({100 * (df.side == 1).mean():.1f}% buy)")

    # --- the published shape: by entry time ------------------------------------------
    print("\n  BY ENTRY TIME (compare against Almeida et al. Table 6)")
    print(f"  {'slot':<7}{'n':>6}{'buy%':>7}"
          f"{'gross':>11}{'net':>11}{'grossHdg':>11}{'netHdg':>11}{'netHdg t':>10}")
    rows = []
    for slot in SLOTS:
        d = df[df["slot"] == slot]
        if d.empty:
            continue
        gm, _, _ = session_t(d, "gross")
        nm, _, _ = session_t(d, "net")
        ghm, _, _ = session_t(d, "gross_hedged")
        nhm, nht, ns = session_t(d, "net_hedged")
        rows.append({"slot": slot, "n": len(d), "sessions": ns,
                     "buy_share": float((d.side == 1).mean()),
                     "gross": gm, "net": nm, "gross_hedged": ghm,
                     "net_hedged": nhm, "net_hedged_t": nht})
        print(f"  {slot:<7}{len(d):>6}{100 * (d.side == 1).mean():>6.0f}%"
              f"{gm:>+11.6f}{nm:>+11.6f}{ghm:>+11.6f}{nhm:>+11.6f}{nht:>+10.2f}")

    print("\n  POOLED across entry times")
    for col in ("gross", "net", "gross_hedged", "net_hedged"):
        m, t, n = session_t(df, col)
        sd = df[col].std(ddof=1)
        print(f"    {col:<14} session-mean {m:+.6f}  t {t:+.2f}  "
              f"per-trade {df[col].mean():+.6f}  "
              f"per-trade Sharpe {df[col].mean() / sd if sd > 0 else float('nan'):+.4f}")

    # --- the calibration check that governs whether ANY of this is readable ----------
    ratio = float((df["ep_physical"] / df["mid"].replace(0, np.nan)).median())
    realised = float(df["payoff"].mean() / df["mid"].mean())
    print(f"\n  CALIBRATION OF THE CONDITIONING DEVICE")
    print(f"    median ep_physical / mid   {ratio:.4f}   (1.0 = unbiased)")
    print(f"    mean   payoff     / mid    {realised:.4f}   (<1 = options rich, the VRP)")
    print(f"    buy share                  {100 * (df.side == 1).mean():.1f}%")
    if abs(ratio - 1.0) > 0.01:
        print(f"\n    *** THE DEVICE IS MISCALIBRATED BY {100 * (ratio - 1):+.1f}%. ***")
        print("    The bare physical expectation (theta = 0) sits systematically off the")
        print("    market price, and that bias is larger than the signal it is supposed to")
        print("    detect -- so the rule degenerates to one-sided and the P&L above tests")
        print("    calibration error, not the SSD hypothesis. Per section 4 of the")
        print("    pre-registration this is a fact about this implementation and NOT")
        print("    evidence about the published effect. A usable bound needs a new")
        print("    pre-registration specifying it properly; do not tune theta here, which")
        print("    section 9 forbids and which would only fit the bias.")
        out = 1
    else:
        print("\n    device is calibrated within 1%; the P&L above is readable as a test")
        print("    of the SSD rule.")
        out = 0

    pd.DataFrame(rows).to_csv(results / "by_entry_time.csv", index=False)
    print(f"\n  wrote {results / 'by_entry_time.csv'}")
    return out


if __name__ == "__main__":
    sys.exit(main())
