"""Pack 3,993 per-session pickles into two .npz files, for speed and for portability.

    python -m trade_analysis.momo_sweep.pack --out data/momo

WHY NOT JUST SHIP THE PICKLES
-----------------------------
Two reasons, and the second is the one that matters.

SPEED: the sweep evaluates a large parameter grid, and every cell re-reads the same bars.
`pickle.load` on 3,993 small files costs ~30 s per pass; one memory-mapped .npz costs
~0.2 s. Over a grid of any size that difference is the whole job.

PORTABILITY, WHICH IS THE REAL REASON: a pickle is a program, not a document. It encodes
the exact classes of the objects inside it -- here `datetime.datetime` -- and unpickling
runs import machinery against whatever interpreter opens it. These files were written by
the laptop's Python; the cluster runs a DIFFERENT interpreter with different library
versions (see [[python-environment-split]]: the laptop itself has two, on numpy 2.3.5 and
2.5.3). Shipping pickles to the cluster is shipping a format whose fidelity depends on the
reader. `.npz` is plain arrays with a documented header and no code path.

REPRODUCING `journal_zone_wf.sessions()` EXACTLY
-----------------------------------------------
The sweep's V0 cell must reproduce the frozen MOMO_CHASE number that `momo_v2.py` computes
from `sessions()`. If this packer silently admits a different set of sessions, V0 will
disagree and the disagreement will look like an engine bug instead of a data bug. So the
two filters are copied deliberately, not re-derived:

    * filename must be `{SYMBOL}_*.pkl`, iterated in `sorted()` order
    * `len(bars) >= 300` -- this DROPS HALF DAYS. 122 of QQQ's 2,779 files have fewer than
      300 one-minute bars (early closes: July 3rd, Thanksgiving Friday, Christmas Eve).
      Keeping them would add sessions whose 13:00 close makes a 25-minute time exit run off
      the end of the array.

A pickle that fails to load is SKIPPED silently by `sessions()`. Here it is counted and
reported, because "the cluster run used 40 fewer sessions than the laptop" is exactly the
kind of difference that is invisible until two numbers fail to match.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "live_lab_data" / "bars_cache"
MIN_BARS = 300                      # identical to journal_zone_wf.sessions()


def pack_symbol(symbol: str, cache: Path) -> dict:
    """Flat arrays plus an offsets index, because sessions are ragged (390 vs 210 bars)."""
    days, offs, bad = [], [0], []
    ts, o, h, lo, c, v = [], [], [], [], [], []

    for fn in sorted(os.listdir(cache)):
        if not fn.startswith(symbol + "_") or not fn.endswith(".pkl"):
            continue
        day = fn[len(symbol) + 1:-4]
        try:
            bars = pickle.load(open(cache / fn, "rb"))
        except Exception as e:                                     # noqa: BLE001
            bad.append((day, repr(e)[:60]))
            continue
        if not isinstance(bars, list) or len(bars) < MIN_BARS:
            continue
        for b in bars:
            t = b["ts"]
            ts.append(t.hour * 60 + t.minute)      # minutes from midnight, exchange-local
            o.append(b["open"]); h.append(b["high"]); lo.append(b["low"])
            c.append(b["close"]); v.append(b["volume"])
        days.append(day)
        offs.append(len(ts))

    if bad:
        print(f"  *** {len(bad)} pickle(s) FAILED to load and were dropped ***")
        for d, e in bad[:5]:
            print(f"      {d}  {e}")

    return {
        "day": np.array(days, dtype="U10"),
        "off": np.array(offs, dtype=np.int64),
        "tod": np.array(ts, dtype=np.int16),
        # float64 throughout. float32 holds ~7 significant digits and a QQQ print is
        # already 5 of them ($718.09); the sweep then takes DIFFERENCES of those prices to
        # get returns of ~1e-4, where float32 rounding is a material fraction of the signal.
        "open": np.array(o, dtype=np.float64),
        "high": np.array(h, dtype=np.float64),
        "low": np.array(lo, dtype=np.float64),
        "close": np.array(c, dtype=np.float64),
        "volume": np.array(v, dtype=np.float64),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--out", default="data/momo")
    ap.add_argument("--symbols", nargs="+", default=["QQQ", "SPY"])
    a = ap.parse_args(argv)

    cache = Path(a.cache)
    if not cache.is_dir():
        print(f"FATAL: bar cache not found at {cache}", file=sys.stderr)
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    for sym in a.symbols:
        d = pack_symbol(sym, cache)
        n = len(d["day"])
        if n == 0:
            print(f"  {sym}: no sessions -- skipped")
            continue
        p = out / f"{sym}.npz"
        np.savez_compressed(p, **d)
        span = f"{d['day'][0]} -> {d['day'][-1]}"
        mb = p.stat().st_size / 1024 / 1024
        print(f"  {sym}: {n} sessions  {len(d['close']):>9,} bars  {span}  "
              f"{mb:.1f} MB  -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
