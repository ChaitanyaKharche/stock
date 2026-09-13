"""Run a shard of the MOMO_CHASE grid over every session, or verify the engine first.

    python -m trade_analysis.momo_sweep.sweep --validate            # DO THIS FIRST
    python -m trade_analysis.momo_sweep.sweep --shard 0 --nshards 200 --out results/

THE VALIDATE MODE IS NOT OPTIONAL CEREMONY
------------------------------------------
`--validate` runs exactly one cell -- the frozen MOMO_CHASE -- and compares it to the
numbers `momo_v2.py` printed on 2026-09-13 from the ORIGINAL pickles through the ORIGINAL
loader: **n=8,598 trades, +$0.0300/trade**.

Two independent things have to be right for that to reproduce, and each is a realistic way
to be silently wrong:

  * the PACKING (`pack.py`) must admit the same sessions -- same >=300-bar filter, same
    sorted order, same prices at full precision
  * the ENGINE (`engine.py`) must apply the same gates in the same order with the same
    indicator values, including the `k % 5 == 4` bucket-close rule and the 25-minute
    cooldown

A grid of a few hundred thousand cells will always contain something that looks good. The
only protection against that something being an artifact is knowing that the machinery
reproduces a number computed a completely different way. If validate fails, the sweep does
not run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np

from .engine import Cell, prepare, run_cell, v0_cell

# What momo_v2.py printed on 2026-09-13, QQQ+SPY, for the V0 (as-frozen) variant.
V0_EXPECTED_N = 8598
V0_EXPECTED_MEAN = 0.03          # printed to 2dp; tolerance below reflects that


def load_sessions(npz_path: Path) -> list[tuple[str, list[dict]]]:
    """Rebuild the list-of-dicts shape that `Panel` and `bars_5m` consume.

    The dicts are rebuilt rather than the indicator code being rewritten around arrays. That
    costs a little memory and buys exact fidelity with the proven implementation -- see the
    header of engine.py.
    """
    z = np.load(npz_path)
    days, off, tod = z["day"], z["off"], z["tod"]
    o, h, lo, c, v = z["open"], z["high"], z["low"], z["close"], z["volume"]
    out = []
    for i, day in enumerate(days):
        a, b = int(off[i]), int(off[i + 1])
        d = dt.date.fromisoformat(str(day))
        bars = [
            {"ts": dt.datetime(d.year, d.month, d.day, int(tod[j]) // 60, int(tod[j]) % 60),
             "open": float(o[j]), "high": float(h[j]), "low": float(lo[j]),
             "close": float(c[j]), "volume": float(v[j])}
            for j in range(a, b)
        ]
        out.append((str(day), bars))
    return out


def iter_prepared(data_dir: Path, symbols: list[str]):
    """Yield (symbol, day, Prep) with two prior sessions seeding the indicators.

    Sessions are the outer loop, so each Prep is built once and handed to every cell.
    """
    from .engine import PREFIX
    for sym in symbols:
        p = data_dir / f"{sym}.npz"
        if not p.exists():
            print(f"  WARNING: {p} missing -- skipping {sym}", file=sys.stderr)
            continue
        S = load_sessions(p)
        print(f"  {sym}: {len(S)} sessions", flush=True)
        for i in range(PREFIX, len(S)):
            day, sess = S[i]
            prior = [S[i - j][1] for j in range(PREFIX, 0, -1)]
            try:
                yield sym, day, prepare(sess, prior)
            except Exception:                                      # noqa: BLE001
                continue


def validate(data_dir: Path, symbols: list[str]) -> int:
    c = v0_cell()
    tot, n, t0 = 0.0, 0, time.time()
    for _, _, pp in iter_prepared(data_dir, symbols):
        p, k = run_cell(pp, c)
        tot += p
        n += k
    mean = tot / n if n else float("nan")
    dur = time.time() - t0

    print("\n" + "=" * 78)
    print("  ENGINE VALIDATION -- frozen MOMO_CHASE (V0) from the PACKED data")
    print("=" * 78)
    print(f"    trades   {n:>10,}   expected {V0_EXPECTED_N:,}")
    print(f"    $/trade  {mean:>+10.4f}   expected ~{V0_EXPECTED_MEAN:+.2f}")
    print(f"    gross    {tot:>+10.2f}")
    print(f"    elapsed  {dur:>10.1f}s for ONE cell over all sessions")

    ok_n = (n == V0_EXPECTED_N)
    ok_m = abs(mean - V0_EXPECTED_MEAN) < 0.005      # momo_v2 printed only 2 decimals
    if ok_n and ok_m:
        print("\n  PASS -- packing and engine reproduce momo_v2.py. The sweep may run.")
        return 0
    print("\n  FAIL -- do NOT run the sweep.")
    if not ok_n:
        print(f"    trade count differs by {n - V0_EXPECTED_N:+,}. A count mismatch is a")
        print("    GATE or SESSION-ADMISSION difference, not a rounding difference: check")
        print("    pack.py's >=300-bar filter and the k%5==4 bucket rule first.")
    if not ok_m:
        print(f"    mean differs by {mean - V0_EXPECTED_MEAN:+.4f} with the SAME trade count,")
        print("    which points at prices or fills rather than gates -- check float dtype")
        print("    in pack.py and the next-bar open indexing in engine.run_cell.")
    return 1


def run_shard(data_dir: Path, symbols: list[str], shard: int, nshards: int,
              out_dir: Path) -> int:
    from .grid import build_grid
    cells = build_grid()
    mine = [(i, c) for i, c in enumerate(cells) if i % nshards == shard]
    if not mine:
        print(f"  shard {shard} of {nshards} has no cells (grid is {len(cells)})")
        return 0

    keys, sess_ids = [i for i, _ in mine], []
    pnl_rows, cnt_rows = [], []
    t0 = time.time()
    for _, day, pp in iter_prepared(data_dir, symbols):
        sess_ids.append(day)
        pr = np.zeros(len(mine), dtype=np.float64)
        cn = np.zeros(len(mine), dtype=np.int32)
        for j, (_, c) in enumerate(mine):
            pr[j], cn[j] = run_cell(pp, c)
        pnl_rows.append(pr)
        cnt_rows.append(cn)
        if len(sess_ids) % 250 == 0:
            print(f"    {len(sess_ids)} sessions, {time.time() - t0:.0f}s", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"shard_{shard:05d}.npz"
    np.savez_compressed(
        p,
        cell_index=np.array(keys, dtype=np.int64),
        session=np.array(sess_ids, dtype="U16"),
        # (n_sessions, n_cells): session-level sums. Per-trade rows do not fit -- see the
        # aggregation note in engine.py -- and session level is the correct unit for the
        # clustered inference anyway.
        pnl=np.vstack(pnl_rows).astype(np.float64),
        n_trades=np.vstack(cnt_rows).astype(np.int32),
    )
    print(f"  wrote {p}  cells={len(mine)}  sessions={len(sess_ids)}  "
          f"{time.time() - t0:.0f}s")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/momo")
    ap.add_argument("--symbols", nargs="+", default=["QQQ", "SPY"])
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default="results/momo_sweep")
    a = ap.parse_args(argv)

    d = Path(a.data)
    if not d.is_dir():
        print(f"FATAL: packed data not found at {d}. Run "
              f"`python -m trade_analysis.momo_sweep.pack` first.", file=sys.stderr)
        return 2
    if a.validate:
        return validate(d, a.symbols)
    return run_shard(d, a.symbols, a.shard, a.nshards, Path(a.out))


if __name__ == "__main__":
    sys.exit(main())
