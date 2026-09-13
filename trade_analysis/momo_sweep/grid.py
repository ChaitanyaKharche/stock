"""The frozen parameter universe. Enumerated here so it can be counted before it is run.

    python -m trade_analysis.momo_sweep.grid            # print the size and the axes

WHY THE WHOLE GRID, AND WHY THAT IS THE HONEST CHOICE RATHER THAN THE GREEDY ONE
--------------------------------------------------------------------------------
The instinct is that testing a million parameterizations is worse science than testing
three. It is the opposite, and the reason is precise.

`momo_v2.py` tested three variants. They were picked by looking at an AUC table first. That
is a search too -- it just has an UNKNOWN size, because nobody can say how many variants
would have been considered had the first three looked different. A correction requires the
denominator, and a hand-picked shortlist does not have one.

An exhaustive grid has a known denominator by construction. Every cell is run, every cell
is reported, and the best cell is judged against the distribution of the best cell UNDER THE
NULL across the whole universe. That is what White's Reality Check and Hansen's SPA are for,
and they are only valid if the universe was fixed before the results were seen. Hence this
file, hence it being importable and countable, and hence the sbatch script refusing to run
if `--array` disagrees with `len(build_grid())`.

ADDING CELLS IS NEARLY FREE, WHICH IS THE ACTUAL ARGUMENT FOR SPENDING THE COMPUTE
----------------------------------------------------------------------------------
Under Bonferroni or Holm, a millionth cell would divide alpha by a million and the sweep
would be unable to find anything. Under a bootstrap-based correction it does not, because
neighbouring cells are ~0.99 correlated -- `trail_min=15` and `trail_min=20` trade almost
the same minutes -- and the bootstrap resamples the JOINT distribution, so a near-duplicate
adds almost nothing to the null distribution of the maximum. The effective number of
independent trials is far smaller than the nominal count.

That is why a fine grid is affordable here and a Holm correction over the same grid would
not be. It is also why the grid is dense along continuous axes (thresholds, lookbacks) and
sparse along structural ones (which gates exist): density is cheap, structure is not.

WHAT IS DELIBERATELY *NOT* AN AXIS
----------------------------------
`macd=(9,17,9)` and `dmi_len=14` are fixed. They were calibrated to the trader's BEHAVIOUR
in the journal study and never to P&L, so they are measurements rather than free parameters.
Sweeping them would spend the correction budget on turning a constant into a fitted value.
"""
from __future__ import annotations

import itertools
import sys
from typing import Iterable

from .engine import Cell

NAN = float("nan")

# --- continuous axes: dense, because density is cheap under a bootstrap correction --------
TRAIL_MIN = [5, 10, 15, 20, 30, 45, 60]
SIGMA_MULT = [0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0]
TIME_EXIT = [5, 10, 15, 25, 40, 60, 90]
ADX_MIN = [0.0, 15.0, 20.0, 25.0, 30.0]                # 0.0 = gate off
EMA9_MIN = [NAN, 0.3, 0.71]                            # NAN = gate off

# --- structural axes: sparse, because each one genuinely doubles the hypothesis space -----
DMI_TF = ["off", "1m", "5m"]
MACD_GATE = [False, True]
DIRECTION = ["chase", "fade"]
GATE_REF = ["trade", "momentum"]
MAX_PER_DAY = [0, 3]                                   # 0 = cooldown only, as momo_v2 runs

# Minutes from midnight, exchange-local. 570 = 09:30, 955 = 15:55.
# NOTE the interaction, stated so it is not discovered later as a surprise: `run_cell` scans
# only k < n_bars - time_exit - 2, so a 90-minute hold cannot enter after ~14:28 no matter
# what window is requested. Long holds therefore have a narrower EFFECTIVE window than short
# ones. This is correct -- it is what prevents an exit from being clamped to the close and
# silently becoming a different holding period -- but it means `time_exit` and `window` are
# not orthogonal, and any per-axis marginal must be read with that in mind.
WINDOWS = [
    (570, 955),      # whole session
    (570, 630),      # opening hour only
    (600, 780),      # 10:00-13:00
    (630, 870),      # 10:30-14:30  <- the frozen MOMO_CHASE window
    (780, 955),      # 13:00-15:55, the afternoon
]


def _axes() -> dict[str, Iterable]:
    return {
        "trail_min": TRAIL_MIN, "sigma_mult": SIGMA_MULT, "time_exit": TIME_EXIT,
        "adx_min": ADX_MIN, "dmi_tf": DMI_TF, "macd_gate": MACD_GATE,
        "window": WINDOWS, "direction": DIRECTION, "ema9_min": EMA9_MIN,
        "max_per_day": MAX_PER_DAY, "gate_ref": GATE_REF,
    }


def build_grid() -> list[Cell]:
    """Every cell, de-duplicated, in a deterministic order.

    DE-DUPLICATION MATTERS FOR THE STATISTICS, not for the runtime. `gate_ref` distinguishes
    nothing when `direction == "chase"` (the trade sign and the momentum sign are the same
    number), and it distinguishes nothing when every gate is off (there is no gate left to
    reference). Leaving those duplicates in would inflate the nominal family size with cells
    that are bit-identical to another cell -- which does not bias the bootstrap, but does
    make the reported "N models tested" a lie, and that number is quoted in the write-up.
    """
    seen, out = set(), []
    for tr, sg, te, ax, dt_, mg, (ws, we), di, e9, mx, gr in itertools.product(
            TRAIL_MIN, SIGMA_MULT, TIME_EXIT, ADX_MIN, DMI_TF, MACD_GATE,
            WINDOWS, DIRECTION, EMA9_MIN, MAX_PER_DAY, GATE_REF):
        gates_on = mg or dt_ != "off" or e9 == e9
        if gr == "momentum" and (di == "chase" or not gates_on):
            continue                       # bit-identical to the gate_ref="trade" cell
        c = Cell(trail_min=tr, sigma_mult=sg, time_exit=te, adx_min=ax, dmi_tf=dt_,
                 macd_gate=mg, win_start=ws, win_end=we, direction=di, ema9_min=e9,
                 max_per_day=mx, gate_ref=gr)
        # NaN != NaN, so Cell is not hashable-by-value for the ema9 axis; key on the repr.
        k = repr(c)
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out


def main(argv=None) -> int:
    cells = build_grid()
    print(f"  MOMO_CHASE sweep grid: {len(cells):,} unique cells\n")
    tot = 1
    for name, vals in _axes().items():
        v = list(vals)
        tot *= len(v)
        show = ", ".join(str(x) for x in v)
        print(f"    {name:<12} {len(v):>3}   {show[:78]}")
    print(f"\n    nominal product {tot:,}  ->  {len(cells):,} after de-duplication")
    # Compared by repr, NOT by `==`. v0_cell() carries ema9_min=NaN and NaN != NaN, so a
    # plain `cells.index(v0_cell())` reports the anchor cell MISSING while it is sitting in
    # the list -- a false alarm on the one check that is supposed to catch a real one.
    from .engine import v0_cell
    want = repr(v0_cell())
    v0 = next((i for i, c in enumerate(cells) if repr(c) == want), None)
    if v0 is None:
        print("\n  *** WARNING: the frozen MOMO_CHASE (V0) is NOT in this grid. ***")
        print("      The sweep would then have no anchor to the already-measured result.")
        return 1
    print(f"\n    the frozen MOMO_CHASE (V0) is cell #{v0:,} -- it IS in the grid")
    est = len(cells) / 3731.0
    print(f"    estimated {est:,.0f} CPU-hours at the measured 3,731 cells/CPU-hour")
    return 0


if __name__ == "__main__":
    sys.exit(main())
