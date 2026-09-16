"""Reduce the sweep's shards into the single verdict §8 of the pre-registration commits to.

    python -m trade_analysis.momo_sweep.collect --results $HOME/momo/results

This is the file `submit_sweep.sbatch` has documented in its header since the day it was
written, and which did not exist. Without it the 500 shards are 500 unreadable `.npz`.

WHAT IS EXACT HERE AND WHAT IS NOT -- read this before quoting anything
----------------------------------------------------------------------
**The SPA verdict is exact over the whole 2.8M-cell grid.** Hansen's statistic is a MAXIMUM
over cells, and a maximum is associative: each shard wrote its own observed max and its own
per-resample max vector, so taking `max` across shards reproduces the number a single
process with 34 GB of RAM would have computed. `stats_test.py` asserts this equality rather
than assuming it. The one thing that makes it true is that every shard must bootstrap the
SAME sessions in the SAME order with the SAME resample indices -- so this file refuses to
reduce shards whose `session_hash`, `spread`, `block` or `reps` disagree. That check is the
load-bearing part of the file.

**The effective trial count and StepM are NOT exact**, and both are anti-conservative in a
way that has to be stated:

  * `n_eff` is measured on the retained top cells (`TOP_KEEP` per shard), because the full
    (sessions x 2.8M) matrix was never written and a 2.8M-square correlation matrix cannot
    be decomposed anyway. Top cells resemble each other MORE than two cells drawn at random
    from the grid, so their participation ratio is SMALLER, so the MinBTL floor derived from
    it is LOWER, so the gate is WEAKER than the truth. The floor at the nominal cell count
    is therefore printed alongside as the conservative bracket. The pre-registration says
    the measured value governs (§4b), and it does -- but a winner that clears the measured
    floor and fails the nominal one is a borderline result, not a clean pass, and it is
    labelled that way below.
  * StepM runs over the retained candidates only, and only in the branch where SPA already
    rejected. A cell with a modest mean but a tiny variance could in principle carry a large
    t and be missed by a retention rule ranked on the mean. This cannot affect the SPA
    verdict, which never used the retention set.

Nothing here re-runs the grid, re-costs a cell, or re-ranges an axis. §9 prohibits all
three, and a collector that could do them is how the prohibition gets broken by accident.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .stats import (boot_indices, effective_trials, minbtl_sharpe_floor, spa_pvalue,
                    studentize)

TRADING_DAYS = 252.0
ALPHA = 0.05


def load_shards(results: Path, expect: int | None) -> tuple[list[dict], dict]:
    """Every shard, with the integrity checks that make the reduction legitimate."""
    paths = sorted(results.glob("shard_*.npz"))
    if not paths:
        raise SystemExit(f"FATAL: no shard_*.npz under {results}")

    shards, meta = [], {}
    for p in paths:
        z = np.load(p, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        d["_path"] = p
        shards.append(d)

    # ---- all shards must describe the SAME experiment ---------------------------------
    # A mismatch here is not a warning. If two shards bootstrapped different sessions, or
    # different resample indices, then `max` across their boot vectors is a maximum over
    # two unrelated null distributions and the p-value it produces is meaningless -- while
    # looking entirely normal. This is the failure the sbatch guards at submit time and
    # that has to be guarded again at reduce time, because a partial re-run with a changed
    # flag is exactly how the two diverge.
    for key in ("session_hash", "spread", "block", "reps"):
        vals = {str(s[key].item() if s[key].ndim == 0 else s[key]) for s in shards}
        if len(vals) != 1:
            raise SystemExit(
                f"FATAL: shards disagree on `{key}`: {sorted(vals)[:4]}\n"
                f"  These shards are not the same experiment and MUST NOT be pooled.\n"
                f"  Re-run the disagreeing shards with identical flags.")
        meta[key] = sorted(vals)[0]

    n_sess = {int(s["session"].shape[0]) for s in shards}
    if len(n_sess) != 1:
        raise SystemExit(f"FATAL: shards cover different session counts: {sorted(n_sess)}")
    meta["n_sessions"] = n_sess.pop()
    meta["n_shards_found"] = len(shards)
    meta["sessions"] = shards[0]["session"]

    # ---- completeness ------------------------------------------------------------------
    # "Ran 500 of 800 shards and reported a p-value over the part that ran" is the specific
    # accident the sbatch header calls out.
    #
    # BE PRECISE ABOUT WHY, because an earlier version of this comment said the p-value is
    # "biased DOWN, making a null look better supported", and that is not right. Hansen's
    # statistic and its bootstrap null are BOTH maxima over the same set of cells, so
    # dropping cells shrinks both and the result stays a VALID SPA test -- of a SMALLER
    # FAMILY. The defect is scope, not bias: a p-value over 300 of 500 shards supports
    # "no cell among the 1.7M we ran beats not trading", never the sec.8 sentence about all
    # 2,822,400. Worse, the direction is not even fixed -- if the best cell survives the
    # truncation, its observed max is unchanged while the bootstrap max falls, so the test
    # OVER-rejects relative to the full grid; if the best cell was in a dropped shard, the
    # test loses power instead. Neither is a conclusion about the grid.
    ids = sorted(int(p.name[6:11]) for p in paths)
    if expect is None:
        expect = ids[-1] + 1
    missing = sorted(set(range(expect)) - set(ids))
    meta["expected"] = expect
    meta["missing"] = missing
    return shards, meta


def reduce_spa(shards: list[dict]) -> dict:
    """The exact grid-wide Hansen maxima, by reduction across shards."""
    t_obs_max = max(float(s["t_obs_max"]) for s in shards)
    best = max(shards, key=lambda s: float(s["t_obs_max"]))
    out = {"t_obs_max": t_obs_max, "argmax_cell": int(best["argmax_cell"])}
    for name in ("consistent", "upper", "lower"):
        stacked = np.vstack([s[f"boot_max_{name}"] for s in shards])
        bm = stacked.max(axis=0)                      # elementwise across shards
        out[f"boot_max_{name}"] = bm
        out[f"p_{name}"] = spa_pvalue(t_obs_max, bm)
    return out


def stepm(series: np.ndarray, cells: np.ndarray, idx: np.ndarray,
          alpha: float = ALPHA, max_steps: int = 40) -> list[tuple[int, float]]:
    """Romano-Wolf stepdown: which cells survive, not merely whether any does.

    Holm is deliberately not used. At 2.8M cells it divides alpha by 2.8M and could not
    detect anything; StepM exploits the cell correlation the bootstrap already models.
    (Holm stays correct for the live lab's family of 13, which is weakly correlated.)
    """
    alive = np.arange(series.shape[1])
    survivors: list[tuple[int, float]] = []
    for _ in range(max_steps):
        if alive.size == 0:
            break
        D = series[:, alive]
        mean, se, bm = studentize(D, idx)
        t = mean / se
        # Null for the max over the cells still in contention, recentred on their own means.
        null = np.max((bm - mean[None, :]) / se[None, :], axis=1)
        crit = float(np.quantile(null, 1.0 - alpha))
        win = t > crit
        if not win.any():
            break
        for j in np.where(win)[0]:
            survivors.append((int(cells[alive[j]]), float(t[j])))
        alive = alive[~win]
    return survivors


def describe(c) -> str:
    return (f"trail={c.trail_min:>3}m sigma={c.sigma_mult:.2f} exit={c.time_exit:>3}m "
            f"adx={c.adx_min:>4.1f} dmi={c.dmi_tf:<3} macd={int(c.macd_gate)} "
            f"win={c.win_start // 60:02d}:{c.win_start % 60:02d}-"
            f"{c.win_end // 60:02d}:{c.win_end % 60:02d} {c.direction:<5} "
            f"ema9={c.ema9_min if c.ema9_min == c.ema9_min else 0:>4.1f} "
            f"cap={c.max_per_day} gref={c.gate_ref:<8} norm={c.sig_norm:<7} "
            f"vol={c.vol_regime}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="results/momo_sweep")
    ap.add_argument("--expect", type=int, default=None,
                    help="shard count that SHOULD be present (default: max id + 1)")
    ap.add_argument("--allow-missing", action="store_true",
                    help="reduce anyway. The verdict is then over a SUBSET of the grid "
                         "and is biased DOWN; the report says so.")
    ap.add_argument("--neff-cells", type=int, default=1500,
                    help="cells sampled from the retained set for the participation ratio")
    ap.add_argument("--stepm-cells", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260913)
    a = ap.parse_args(argv)

    shards, meta = load_shards(Path(a.results), a.expect)
    n = meta["n_sessions"]
    years = n / TRADING_DAYS

    print("=" * 78)
    print("  MOMO_CHASE SWEEP -- COLLECTED VERDICT")
    print("=" * 78)
    print(f"    shards found        {meta['n_shards_found']} of {meta['expected']}")
    print(f"    sessions            {n}  ({years:.2f} years)  "
          f"{meta['sessions'][0]} .. {meta['sessions'][-1]}")
    print(f"    session hash        {meta['session_hash']}  (identical in every shard)")
    print(f"    cost charged        ${float(meta['spread']):.4f}/share round trip")
    print(f"    bootstrap           B={int(float(meta['reps']))}, "
          f"stationary, block={float(meta['block']):.1f}")

    if meta["missing"]:
        m = meta["missing"]
        print(f"\n    *** {len(m)} SHARDS MISSING: "
              f"{m[:10]}{' ...' if len(m) > 10 else ''}")
        if not a.allow_missing:
            print("    Refusing to reduce. This would be a valid SPA test of a SMALLER")
            print("    FAMILY, not a weaker test of the whole grid -- so its p-value cannot")
            print("    support the sec.8 sentence about all 2,822,400 cells. Re-run the")
            print("    missing shards, or pass --allow-missing for a provisional read.")
            return 2
        print("    --allow-missing given: the verdict below covers ONLY the cells in the")
        print("    shards present. Quote it as such; it is not a statement about the grid.")

    cell_idx = np.concatenate([s["cell_index"] for s in shards])
    net_mean = np.concatenate([s["net_mean"] for s in shards])
    net_sd = np.concatenate([s["net_sd"] for s in shards])
    n_tr = np.concatenate([s["n_trades"] for s in shards])
    brk = np.concatenate([s["breakeven"] for s in shards])
    print(f"    cells reduced       {cell_idx.size:,}")

    # ---- the SPA verdict (exact) -------------------------------------------------------
    spa = reduce_spa(shards)
    print()
    print("-" * 78)
    print("  HANSEN SPA  (exact over the reduced grid; max is associative)")
    print("-" * 78)
    print(f"    observed max studentized statistic   t = {spa['t_obs_max']:+.4f}")
    for name in ("lower", "consistent", "upper"):
        mark = "  <-- GOVERNS" if name == "consistent" else ""
        print(f"    p({name:<10}) = {spa[f'p_{name}']:.4f}{mark}")
    gap = spa["p_upper"] - spa["p_lower"]
    print(f"    upper-lower gap {gap:+.4f} -- a wide gap means the verdict is sensitive")
    print(f"    to how borderline cells are recentred.")
    p_gov = spa["p_consistent"]
    spa_ok = p_gov < ALPHA

    # ---- the winner --------------------------------------------------------------------
    from .grid import build_grid
    from .engine import v0_cell
    grid = build_grid()
    w = int(spa["argmax_cell"])
    where = int(np.where(cell_idx == w)[0][0])
    sharpe = (float(net_mean[where]) / float(net_sd[where]) * np.sqrt(TRADING_DAYS)
              if net_sd[where] > 0 else 0.0)
    print()
    print("-" * 78)
    print("  BEST CELL BY STUDENTIZED STATISTIC")
    print("-" * 78)
    print(f"    cell #{w:,}")
    print(f"      {describe(grid[w])}")
    print(f"    net mean/session    ${float(net_mean[where]):+.4f}")
    print(f"    trades              {int(n_tr[where]):,}")
    print(f"    annualised Sharpe   {sharpe:+.3f}   (net, per-session series)")
    print(f"    breakeven spread    ${float(brk[where]):.4f}/share   "
          f"(charged ${float(meta['spread']):.4f})")
    if float(brk[where]) < float(meta["spread"]):
        print("    -> breakeven is BELOW the real spread: this cell does not pay for its")
        print("       own execution. That is a COST failure, not an edge (sec.8 row 3).")

    # ---- gate 2: MinBTL floor ----------------------------------------------------------
    rng = np.random.default_rng(a.seed)
    top_series = np.hstack([s["top_series"] for s in shards])
    top_cells = np.concatenate([s["top_cells"] for s in shards])
    k = min(a.neff_cells, top_series.shape[1])
    pick = rng.choice(top_series.shape[1], size=k, replace=False)
    n_eff = effective_trials(top_series[:, pick])
    floor_meas = minbtl_sharpe_floor(n_eff, years)
    floor_nom = minbtl_sharpe_floor(float(cell_idx.size), years)
    print()
    print("-" * 78)
    print("  SECOND GATE -- MinBTL SHARPE FLOOR  (sec.4b)")
    print("-" * 78)
    print(f"    effective trials (measured, {k} retained cells)  {n_eff:,.1f}")
    print(f"    nominal cells                                    {cell_idx.size:,}")
    print(f"    floor at measured n_eff    Sharpe >= {floor_meas:.3f}   <-- GOVERNS (sec.4b)")
    print(f"    floor at nominal count     Sharpe >= {floor_nom:.3f}   "
          f"(conservative bracket)")
    print(f"    winner's Sharpe            {sharpe:+.3f}")
    print("    n_eff is measured on RETAINED TOP cells, which resemble each other more")
    print("    than random grid cells do, so this floor is a LOWER bound -- see header.")
    minbtl_ok = sharpe >= floor_meas
    print(f"    {'CLEARS' if minbtl_ok else 'FAILS'} the governing floor"
          + ("" if not minbtl_ok else
             (" and the conservative bracket too" if sharpe >= floor_nom
              else ", but NOT the conservative bracket -- borderline, not a clean pass")))

    # ---- §7: the whole distribution, not just the top ----------------------------------
    print()
    print("-" * 78)
    print("  DISTRIBUTION ACROSS ALL CELLS  (sec.7 -- no survivor without it)")
    print("-" * 78)
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    pct = np.percentile(net_mean, qs)
    print("    net $/session percentiles")
    for q, v in zip(qs, pct):
        print(f"      p{q:<3} {v:+.4f}")
    pos = int((net_mean > 0).sum())
    print(f"    cells with positive net mean  {pos:,}/{cell_idx.size:,} "
          f"({100 * pos / cell_idx.size:.1f}%)")
    print(f"    cells whose breakeven beats the real spread  "
          f"{int(np.nansum(brk > float(meta['spread']))):,}")

    want = repr(v0_cell())
    v0i = next((i for i, c in enumerate(grid) if repr(c) == want), None)
    if v0i is not None and (np.where(cell_idx == v0i)[0]).size:
        j = int(np.where(cell_idx == v0i)[0][0])
        rank = int((net_mean > net_mean[j]).sum()) + 1
        print(f"    FROZEN V0 (cell #{v0i:,}): net ${float(net_mean[j]):+.4f}/session, "
              f"breakeven ${float(brk[j]):.4f}, rank {rank:,}/{cell_idx.size:,} "
              f"({100 * rank / cell_idx.size:.1f}th pct)")

    # ---- StepM, only if SPA rejected ---------------------------------------------------
    print()
    print("-" * 78)
    print("  StepM SURVIVOR SET")
    print("-" * 78)
    if not spa_ok:
        print(f"    Not run. SPA p = {p_gov:.4f} >= {ALPHA}, so there is no rejection to")
        print("    decompose, and sec.8 closes the question at this row. Running a stepdown")
        print("    anyway would be searching a family the global test just failed to")
        print("    reject -- the exact move the pre-registration exists to prevent.")
    else:
        o = np.argsort(-net_mean)[:a.stepm_cells]
        sel = np.array([i for i in o if (top_cells == cell_idx[i]).any()])
        if sel.size == 0:
            print("    No top-ranked cell has a retained series; cannot run StepM.")
        else:
            cols = [int(np.where(top_cells == cell_idx[i])[0][0]) for i in sel]
            S = top_series[:, cols]
            idx = boot_indices(n, int(float(meta["reps"])), float(meta["block"]))
            surv = stepm(S, cell_idx[sel], idx)
            print(f"    candidates {sel.size} (retained, top by net mean)")
            print(f"    survivors  {len(surv)} at alpha={ALPHA}")
            for ci, t in surv[:20]:
                print(f"      cell #{ci:<10,} t={t:+.3f}  {describe(grid[ci])}")

    # ---- §8 --------------------------------------------------------------------------
    print()
    print("=" * 78)
    print("  DECISION  (sec.8, committed before the run)")
    print("=" * 78)
    if not spa_ok:
        print(f"    SPA p = {p_gov:.4f} >= 0.05 on discovery, NET.")
        print("    => No parameterization of the momo-chase form beats not trading, net of")
        print(f"       costs, anywhere in {cell_idx.size:,} cells. MOMO_CHASE is CLOSED as a")
        print("       standalone strategy. The held-out set is NEVER touched (sec.9).")
    elif not minbtl_ok:
        print(f"    SPA p = {p_gov:.4f} < 0.05, but the winner's Sharpe {sharpe:+.3f} is")
        print(f"    below the MinBTL floor {floor_meas:.3f} for a search this size.")
        print("    => sec.4b is explicit that BOTH gates are required and either alone is")
        print("       insufficient. NOT a survivor. The holdout is not touched.")
    else:
        print(f"    SPA p = {p_gov:.4f} < 0.05 AND Sharpe {sharpe:+.3f} >= "
              f"{floor_meas:.3f}.")
        print("    => NOT A RESULT -- A HYPOTHESIS. The StepM survivor set above goes to")
        print("       the held-out set (2022-2026, QQQ AND SPY) exactly ONCE. Only a")
        print("       same-sign, net-positive survivor there is worth anything, and even")
        print("       then it earns a new forward-test pre-registration, not a live")
        print("       promotion (sec.8 row 2, sec.9).")
    print()
    print("    Prohibited from here (sec.9): no cell added/removed/re-ranged, no re-run at a")
    print("    different cost to rescue a cell, no DSR or PBO promoted over this verdict,")
    print("    no addition to the frozen live lab under any outcome.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
