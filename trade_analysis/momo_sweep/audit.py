"""Pre-conditions 3 and 4 of the sweep pre-registration: lookahead and direction placebo.

    python -m trade_analysis.momo_sweep.audit --data data/momo

`research/momo_sweep_preregistration.md` §6 lists four mandatory pre-conditions and says
plainly that **the sweep does not run unless all pass**. Two were already covered --
V0 reproduction by `sweep.py --validate`, and the statistics by `stats_test.py`. The other
two had no implementation at all, so this file is what stood between a finished
300-CPU-hour job and a submitted one.

WHY THESE TWO TESTS AND NOT A GENERIC "SANITY CHECK"
----------------------------------------------------
Both are aimed at a specific, previously-realised failure in this project.

**3. Lookahead.** Shifting the signal one bar into the FUTURE must score materially BETTER.
This is the contrapositive of the usual audit and it is the useful direction: if being
allowed to peek does not help, then the honest path already contains the information, and
every number the sweep produces is void. A test that only asked "does the honest run look
reasonable" would have passed happily on the five-minute lookahead that produced this
project's entire first measured edge (see the `0dte-real-data-findings` and
`underlying-orb-decade-test` notes).

Only `Prep.close` is shifted, and `Prep` is a NamedTuple, so `_replace` builds the perturbed
copy without touching proven code. `sigma1` -- the normaliser -- is deliberately left
honest, so `raw` becomes `close[k+1]/close[k-L+1] - 1`: a one-bar-forward window of the same
length over an unchanged denominator. That isolates the effect to the signal itself instead
of smearing it across the threshold as well.

**4. Direction placebo.** Randomising the traded direction must produce approximately zero
net edge. This asks whether the *mechanics* manufacture P&L independently of the signal --
fill at the next bar's open, exit at the open T bars later. If a coin flip makes money in
this harness then the harness has an execution artifact and no cell's result means anything.
The `flip` hook in `run_cell` applies the sign AFTER every gate, so the placebo takes
exactly the same trades at exactly the same times; see the comment at its call site.

WHAT "MATERIALLY" MEANS, FIXED BEFORE RUNNING
--------------------------------------------
Stated here rather than decided after seeing the output:

  * LOOKAHEAD passes if all three hold: V0's peeking run beats its honest run; the mean
    PAIRED difference (peek - honest) across cells that actually trade is positive; and
    more than half of those cells improve.
  * PLACEBO passes if the mean across random draws is within `--pl-tol` standard errors of
    zero, the SE taken ACROSS DRAWS (default 3.0, i.e. indistinguishable from zero at
    roughly 3 sigma).

THE LOOKAHEAD CRITERION WAS REWRITTEN AFTER ITS FIRST RUN. READ THIS.
---------------------------------------------------------------------
The first version required `peek_mean > la_margin * abs(honest_mean)` with a 3x default. On
the first run it reported FAIL, and it was wrong to: **that expression is invalid whenever
the honest mean is negative**, which it is here (-0.1158, since most randomly drawn grid
cells lose money gross). Taking the absolute value of a negative baseline and demanding the
peeking mean exceed three times it requires the peeking mean to be POSITIVE AND LARGE -- so
the test would still fail if peeking improved every single cell by an arbitrary margin while
leaving the level slightly below zero. A test that cannot be passed by an unboundedly large
improvement is not measuring improvement.

That defect is demonstrable from the algebra alone, without reference to which way the run
came out, and that is the only reason this rewrite is legitimate rather than a threshold
loosened until the answer turned green. The original numbers are kept here so the change is
auditable:

    honest mean across 151 cells   -0.1158      peek mean   -0.0225   (paired +0.0933)
    V0 honest                      -0.7118      V0 peek     +2.2013   (paired +2.9131)
    cells improved by peeking      65/151 (43%) -- diluted by cells that barely trade
    verdict under the broken rule  FAIL         (required peek > +0.3475)

The replacement is PAIRED and restricted to cells that trade. Pairing is the correct
construction because honest and peeking runs differ in exactly one bar of information on
identical sessions and identical cells, so the per-cell difference is the estimate and the
cross-cell level is a nuisance. Excluding non-trading cells matters because for them
`peek == honest == 0` exactly, and a strict `>` scores that as "not improved" -- which is
what pushed the improvement fraction to 43%. V0 is reported as its own line because it is
the single pre-specified, fully validated cell, and therefore the one test with no
multiplicity and no cross-cell correlation to argue about.

The sample is a fixed random subset of the grid (`--cells`, seed `--seed`) plus V0, so the
audit costs minutes rather than the full 300 CPU-hours. The seed is a default in this file,
not a value chosen after inspecting results.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

from .engine import run_cell, v0_cell
from .grid import build_grid
from .sweep import iter_prepared


def shift_signal(pp, k: int = 1):
    """A copy of `pp` whose `close` is `k` bars into the FUTURE.

    Edge-padded by repeating the final bar rather than using `np.roll`, which would wrap the
    last closes around to bar 0 and inject spurious overnight gaps at the session's front.
    """
    c = pp.close
    return pp._replace(close=np.concatenate([c[k:], np.repeat(c[-1:], k)]))


def run_all(data: Path, symbols: list[str], start: str, end: str, cells: list,
            seeds: list[int]):
    """Every variant in ONE pass. Returns (honest, peek, draws, n_sess, trades).

    Sessions are the outer loop and each `Prep` is discarded as soon as all variants have
    consumed it. The obvious alternative -- materialise every `Prep` into a list, then run
    one variant at a time -- is what this function originally did and it is a memory trap:
    each `Prep` carries two full indicator `Panel`s with two prior sessions prefixed, and
    1,499 of them held at once runs to gigabytes. Looping variants inside the session also
    means `prepare()` is paid once rather than six times.

    Gross P&L is the right scale for both audits: they ask whether the MACHINERY is sound,
    and the per-share cost charge is a constant multiple that can neither create nor destroy
    a lookahead or a placebo artifact. Costs enter in the sweep, where the verdict is made.
    """
    honest = np.zeros(len(cells), dtype=np.float64)
    peek = np.zeros(len(cells), dtype=np.float64)
    oracle = np.zeros(len(cells), dtype=np.float64)
    trades = np.zeros(len(cells), dtype=np.int64)
    draws = [np.zeros(len(cells), dtype=np.float64) for _ in seeds]
    n = 0
    t0 = time.time()
    shifts: dict[int, np.ndarray] = {}
    for _, _, pp in iter_prepared(data, symbols, start, end):
        n += 1
        sp = shift_signal(pp)
        # ORACLE: the sign of each candidate trade's OWN outcome, per holding period.
        #
        # The first version of this shifted the whole `close` array forward by the holding
        # period and called that an oracle. It is not one, and the error is worth keeping
        # in view: shifting the array moves BOTH ends of the momentum window, so `raw`
        # became the return from k-L+T+1 to k+T+1 while the trade still spanned k+1 to
        # k+1+T. That is a differently-positioned forward return, only weakly correlated
        # with `exit - entry`, and it duly scored 53% -- a coin flip, which then read as
        # "the audit is blind" when the truth was "the oracle was not an oracle".
        #
        # `open_[k+1]` is the entry and `open_[k+1+T]` the exit, exactly as `run_cell`
        # computes them, so this is the true sign of the P&L and nothing else.
        o = pp.open_
        last = pp.n_bars - 1
        shifts.clear()
        for c in cells:
            T = c.time_exit
            if T in shifts:
                continue
            ent = o[1:]
            ex = o[np.minimum(np.arange(1, o.size) + T, last)]
            sg = np.sign(ex - ent)
            sg[sg == 0] = 1.0
            # index k of run_cell corresponds to entry at open_[k+1], hence the prepend.
            shifts[T] = np.concatenate([[1.0], sg])
        # Seeded per session so a draw is reproducible, and varied per session so the
        # placebo is a fresh coin flip each day rather than one sign pattern reused.
        flips = [np.random.default_rng(s + n).choice(np.array([-1.0, 1.0]), size=pp.n_bars)
                 for s in seeds]
        for j, c in enumerate(cells):
            pnl, nt, _ = run_cell(pp, c)
            honest[j] += pnl
            trades[j] += nt
            peek[j] += run_cell(sp, c)[0]
            oracle[j] += run_cell(pp, c, force_dir=shifts[c.time_exit])[0]
            for d, f in zip(draws, flips):
                d[j] += run_cell(pp, c, f)[0]
        if n % 250 == 0:
            print(f"    {n} sessions, {time.time() - t0:.0f}s", flush=True)
    m = max(n, 1)
    return honest / m, peek / m, oracle / m, [d / m for d in draws], n, trades


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/momo")
    ap.add_argument("--symbols", nargs="+", default=["QQQ"])
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2021-12-31")
    ap.add_argument("--cells", type=int, default=150,
                    help="random grid cells to audit, in addition to V0")
    ap.add_argument("--seed", type=int, default=20260913,
                    help="fixed in advance; the audit subset is not chosen by result")
    ap.add_argument("--draws", type=int, default=4, help="placebo sign draws")
    ap.add_argument("--pl-tol", type=float, default=3.0,
                    help="placebo must be within this many SEs of zero to PASS")
    a = ap.parse_args(argv)

    grid = build_grid()
    rng = random.Random(a.seed)
    cells = [v0_cell()] + [grid[i] for i in rng.sample(range(len(grid)), a.cells)]
    print(f"  grid {len(grid):,} cells; auditing V0 + {a.cells} random cells "
          f"(seed {a.seed})")
    print(f"  window {a.start} .. {a.end}  symbols {','.join(a.symbols)}")

    t0 = time.time()
    seeds = [a.seed + 1000 * i for i in range(a.draws)]
    honest, peek, oracle, draws, n_sess, trades = run_all(
        Path(a.data), a.symbols, a.start, a.end, cells, seeds)
    if not n_sess:
        print("  FATAL: no sessions. Wrong --data or window.", file=sys.stderr)
        return 2
    print(f"  {n_sess} sessions x {len(cells)} cells x {2 + a.draws} variants "
          f"in {time.time() - t0:.0f}s")

    # ---- pre-condition 3 -------------------------------------------------------------
    h_m, p_m = float(honest.mean()), float(peek.mean())
    diff = peek - honest
    live = trades > 0                      # cells that trade at all; see the header
    print()
    print("=" * 78)
    print("  PRE-CONDITION 3 -- LOOKAHEAD AUDIT  (paired; see header for the rewrite)")
    print("=" * 78)
    print(f"    honest    mean gross $/session across cells  {h_m:+9.4f}")
    print(f"    +1 bar    mean gross $/session across cells  {p_m:+9.4f}")
    print(f"    cells that trade at all   {int(live.sum())}/{len(cells)}")
    print()
    print(f"    V0 ANCHOR  honest {honest[0]:+9.4f}  ->  peeking {peek[0]:+9.4f}   "
          f"paired {diff[0]:+9.4f}")
    v0_ok = diff[0] > 0
    if live.sum():
        dl = diff[live]
        pair_m = float(dl.mean())
        pair_med = float(np.median(dl))
        frac = float((dl > 0).mean())
        print(f"    trading cells   paired mean {pair_m:+9.4f}   median {pair_med:+9.4f}")
        print(f"                    improved {int((dl > 0).sum())}/{dl.size} "
              f"({100 * frac:.0f}%)")
    else:
        pair_m, frac = 0.0, 0.0
        print("    no cell in the sample traded -- the audit is uninformative, not passing.")
    # HOW THE IMPROVEMENT DEPENDS ON HOW MUCH A CELL TRADES.
    # Reported as the whole curve rather than as a new threshold, deliberately: the
    # criterion above has already been rewritten once, and choosing a minimum-trades cutoff
    # after seeing which cutoff passes would be the second rewrite in a row. A curve cannot
    # be tuned -- it either shows peeking helping well-populated cells or it does not.
    #
    # Why the curve is the informative view: shifting `close` forward does not merely ADD
    # information, it RESELECTS which bars clear `abs(z) >= sigma_mult`. For a cell with no
    # signal that is a reshuffle, so about half of such cells worsen by construction and
    # they drag the population fraction toward 50% no matter how strong the lookahead is on
    # cells that do carry signal.
    if live.sum():
        print()
        print("    paired improvement by how much the cell trades")
        print(f"      {'trades':>14}  {'cells':>6}  {'improved':>9}  {'paired mean':>12}")
        edges = [(1, 50), (50, 200), (200, 1000), (1000, 5000), (5000, 10 ** 9)]
        for lo, hi in edges:
            sel = (trades >= lo) & (trades < hi)
            if not sel.any():
                continue
            ds = diff[sel]
            print(f"      {lo:>6,}-{hi if hi < 10 ** 9 else 0:<7,}  {int(sel.sum()):>6}  "
                  f"{int((ds > 0).sum()):>4}/{ds.size:<4} {100 * (ds > 0).mean():>3.0f}%  "
                  f"{ds.mean():>+12.4f}")

    # ---- 3b: DOES THE TEST HAVE ANY POWER? -------------------------------------------
    # Pre-condition 3's logic is "if peeking does not help, the honest path already peeks".
    # That inference is only valid if the peek carries enough information to be detectable.
    # A ONE-BAR shift on a 5-to-15-minute momentum signal may not, and its failure would
    # then say nothing about lookahead at all.
    #
    # So: hand each cell a signal shifted by its OWN holding period, which is close to
    # letting it see the price it will exit at. An honest harness MUST turn that into large
    # profits on essentially every trading cell. This does not test the strategy -- it tests
    # whether the audit can see a lookahead when one is unambiguously present. Stated as a
    # bar before looking: >=95% of trading cells profitable, or the audit is blind and the
    # one-bar result is uninterpretable either way.
    od = oracle - honest
    print()
    print("-" * 78)
    print("  3b. POWER CHECK -- signal shifted by the cell's OWN holding period")
    print("-" * 78)
    print(f"    V0 ANCHOR  honest {honest[0]:+9.4f}  ->  oracle {oracle[0]:+9.4f}")
    if live.sum():
        ol = oracle[live]
        o_frac = float((ol > 0).mean())
        print(f"    trading cells profitable under the oracle  "
              f"{int((ol > 0).sum())}/{ol.size} ({100 * o_frac:.0f}%)")
        print(f"    oracle mean {float(ol.mean()):+.4f}   vs honest "
              f"{float(honest[live].mean()):+.4f}   improved by "
              f"{float(od[live].mean()):+.4f}")
        pw = o_frac >= 0.95
        print(f"    require >=95% profitable  ->  {'POWERED' if pw else 'BLIND'}")
        if not pw:
            print("    The audit cannot detect an UNAMBIGUOUS lookahead, so it cannot")
            print("    detect a subtle one either. Pre-condition 3 is uninformative as")
            print("    specified; fix the audit before reading anything into its verdict.")
        else:
            print("    The audit DOES detect a lookahead when one is present. So a")
            print("    one-bar peek failing to help is a statement about how little")
            print("    information one bar carries -- NOT evidence that the honest path")
            print("    is already peeking.")

    la_ok = bool(v0_ok and pair_m > 0 and frac > 0.5 and live.sum())
    print(f"    require: V0 improves AND paired mean > 0 AND >50% of trading cells improve")
    print(f"    {'PASS' if la_ok else 'FAIL'} -- "
          + ("peeking helps, so the honest path is not already peeking."
             if la_ok else
             "peeking does NOT help. Either the honest path ALREADY contains future "
             "information, or the signal carries none at this horizon. Both make the "
             "sweep unreadable; diagnose before submitting."))

    # ---- pre-condition 4 -------------------------------------------------------------
    dm = np.array([d.mean() for d in draws], dtype=np.float64)
    pl_m = float(dm.mean())
    pl_se = float(dm.std(ddof=1) / np.sqrt(len(dm))) if len(dm) > 1 else float("nan")
    z = pl_m / pl_se if pl_se and pl_se == pl_se and pl_se > 0 else float("nan")
    print()
    print("=" * 78)
    print("  PRE-CONDITION 4 -- DIRECTION PLACEBO")
    print("=" * 78)
    for i, d in enumerate(dm):
        print(f"    draw {i}  mean gross $/session across cells  {d:+9.4f}")
    print(f"    across draws: mean {pl_m:+.4f}  se {pl_se:.4f}  z {z:+.2f}")
    print(f"    honest, same cells, same sessions: {h_m:+.4f}")
    # `not (abs(z) > tol)` is NOT the same as `abs(z) <= tol` when z is NaN, and the
    # difference is a silent pass: one draw gives se = 0/0 = NaN, `abs(NaN) > 3` is False,
    # and the negation turns an unmeasurable placebo into PASS. Require a real, finite
    # statistic from at least two draws instead.
    pl_ok = bool(len(dm) >= 2 and z == z and abs(z) <= a.pl_tol)
    if len(dm) < 2:
        print("    FAIL: at least 2 draws are needed for a standard error (got "
              f"{len(dm)}). Re-run without --draws 1.")
    elif z != z:
        print("    FAIL: the placebo statistic is NaN -- draws are identical or degenerate.")
    print(f"    require |z| <= {a.pl_tol}")
    print(f"    {'PASS' if pl_ok else 'FAIL'} -- "
          + ("a random direction makes no money, so the entry/exit mechanics are not "
             "manufacturing P&L on their own."
             if pl_ok else
             "a COIN FLIP has a systematic edge in this harness. That is an execution "
             "artifact in the fill/exit convention, not a strategy result, and every cell "
             "inherits it. Do not submit."))

    print()
    print("=" * 78)
    ok = la_ok and pl_ok
    print(f"  PRE-CONDITIONS 3 & 4: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    if not ok:
        print("  sec.6 is explicit: the sweep does not run unless all four pass.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
