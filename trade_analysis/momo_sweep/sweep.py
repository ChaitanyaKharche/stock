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


TOD_WINDOW = 14          # sessions averaged for the time-of-day sigma (Zarattini uses 14)
VOL_WINDOW = 20          # sessions the prior day's range is ranked within


def iter_prepared(data_dir: Path, symbols: list[str],
                  start: str | None = None, end: str | None = None):
    """Yield (symbol, day, Prep) with two prior sessions seeding the indicators.

    Sessions are the outer loop, so each Prep is built once and handed to every cell.

    THE TWO CROSS-SESSION QUANTITIES ARE STRICTLY BACKWARD-LOOKING, AND THAT IS THE WHOLE
    RISK IN THIS FUNCTION. `tod_sig` averages the previous TOD_WINDOW sessions and
    `vol_rank` ranks the PREVIOUS session's range inside the VOL_WINDOW sessions before it.
    Neither ever touches the session being traded. This project's single worst bug was a
    five-minute lookahead that produced an entire measured edge, and a rolling normaliser
    that accidentally includes today is the same bug wearing a different hat -- it would
    make the threshold easier to clear on exactly the days that turned out to move.
    Buffers are therefore appended AFTER the session is yielded, never before.

    `start`/`end` gate which sessions are TRADED, not which are read. Sessions before
    `start` still fill the indicator prefix and the rolling normaliser buffers -- that is
    past information and using it is correct. Sessions after `end` are never reached at all.
    This asymmetry is the point: the pre-registration's discovery window is 2016-2021 and
    the 2022-2026 holdout must not be touched, while a 2022 session's tod_sig legitimately
    depends on late-2021 sessions. Without this gate the sweep silently consumes the holdout
    and the out-of-sample test stops existing.
    """
    from .engine import PREFIX
    from .grid import TRAIL_MIN

    for sym in symbols:
        p = data_dir / f"{sym}.npz"
        if not p.exists():
            print(f"  WARNING: {p} missing -- skipping {sym}", file=sys.stderr)
            continue
        S = load_sessions(p)
        n_tradeable = sum(1 for d, _ in S
                          if (start is None or d >= start) and (end is None or d <= end))
        print(f"  {sym}: {len(S)} sessions loaded, {n_tradeable} inside "
              f"[{start or 'begin'} .. {end or 'end'}]", flush=True)

        hist: list[dict[int, np.ndarray]] = []      # per-session |L-min return| by bar
        rng_hist: list[float] = []                  # per-session high-low range, % of close

        for i in range(PREFIX, len(S)):
            day, sess = S[i]
            if end is not None and day > end:
                break                               # holdout: not read, not just not traded
            prior = [S[i - j][1] for j in range(PREFIX, 0, -1)]
            close = np.array([b["close"] for b in sess], dtype=np.float64)
            n = len(close)

            tod_sig = {}
            if len(hist) >= TOD_WINDOW:
                for L in TRAIL_MIN:
                    # Ragged sessions: average only over prior sessions long enough to have
                    # a value at each bar, so a short day cannot shrink the whole profile.
                    cols = [h[L] for h in hist[-TOD_WINDOW:] if len(h[L]) >= n]
                    if cols:
                        tod_sig[L] = np.mean(np.vstack([c[:n] for c in cols]), axis=0)

            vol_rank = float("nan")
            if len(rng_hist) > VOL_WINDOW:
                w = np.array(rng_hist[-VOL_WINDOW - 1:-1])   # excludes the prior day itself
                vol_rank = float((w < rng_hist[-1]).mean())

            if start is None or day >= start:
                try:
                    yield sym, day, prepare(sess, prior, tod_sig, vol_rank)
                except Exception:                                  # noqa: BLE001
                    pass

            # AFTER yielding -- today's own bars must not be in today's normaliser.
            cur = {}
            for L in TRAIL_MIN:
                a = np.zeros(n, dtype=np.float64)
                if n > L:
                    a[L:] = np.abs(close[L:] / close[:-L] - 1.0)
                cur[L] = a
            hist.append(cur)
            hi = max(b["high"] for b in sess)
            lo = min(b["low"] for b in sess)
            rng_hist.append((hi - lo) / close[-1] if close[-1] > 0 else 0.0)


def validate(data_dir: Path, symbols: list[str]) -> int:
    c = v0_cell()
    tot, n, sh, t0 = 0.0, 0, 0.0, time.time()
    for _, _, pp in iter_prepared(data_dir, symbols):
        p, k, s = run_cell(pp, c)
        tot += p
        n += k
        sh += s
    mean = tot / n if n else float("nan")
    dur = time.time() - t0

    print("\n" + "=" * 78)
    print("  ENGINE VALIDATION -- frozen MOMO_CHASE (V0) from the PACKED data")
    print("=" * 78)
    print(f"    trades   {n:>10,}   expected {V0_EXPECTED_N:,}")
    print(f"    $/trade  {mean:>+10.4f}   expected ~{V0_EXPECTED_MEAN:+.2f}")
    print(f"    gross    {tot:>+10.2f}")
    print(f"    elapsed  {dur:>10.1f}s for ONE cell over all sessions")
    if sh > 0:
        # The number that decides everything. Gross dollars per SHARE traded IS the
        # breakeven round-trip spread: below it the cell is net negative. QQQ and SPY are
        # penny-wide, so anything under $0.01 here loses money no matter how good the
        # t-statistic looks.
        be = tot / sh
        print(f"    breakeven spread  ${be:>+8.4f}/share  "
              f"(QQQ/SPY trade at ~$0.01 -- need > that)")

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


TOP_KEEP = 200          # per-shard session series retained; fixed before any result is seen


def run_shard(data_dir: Path, symbols: list[str], shard: int, nshards: int,
              out_dir: Path, start: str | None = None, end: str | None = None,
              spread: float = 0.01, block: float = 5.0, reps: int = 10000) -> int:
    from .grid import build_grid
    cells = build_grid()
    mine = [(i, c) for i, c in enumerate(cells) if i % nshards == shard]
    if not mine:
        print(f"  shard {shard} of {nshards} has no cells (grid is {len(cells)})")
        return 0

    keys, sess_ids = [i for i, _ in mine], []
    pnl_rows, cnt_rows, sh_rows = [], [], []
    t0 = time.time()
    for _, day, pp in iter_prepared(data_dir, symbols, start, end):
        sess_ids.append(day)
        pr = np.zeros(len(mine), dtype=np.float64)
        cn = np.zeros(len(mine), dtype=np.int32)
        sr = np.zeros(len(mine), dtype=np.float64)
        for j, (_, c) in enumerate(mine):
            pr[j], cn[j], sr[j] = run_cell(pp, c)
        pnl_rows.append(pr)
        cnt_rows.append(cn)
        sh_rows.append(sr)
        if len(sess_ids) % 250 == 0:
            print(f"    {len(sess_ids)} sessions, {time.time() - t0:.0f}s", flush=True)

    # ---- reduce HERE, not in the collector -------------------------------------------
    # The naive thing is to write the (sessions x cells) matrix and let the collector do
    # the statistics. At 5,645 cells x 1,499 sessions that is 68 MB per shard and 34 GB
    # across 500 -- for a quantity that is a MAXIMUM over cells, and a maximum reduces.
    # Each shard therefore computes its own bootstrap maxima and writes a (reps,) vector;
    # the collector takes the max across shards and gets the exact grid-wide answer.
    # Per-cell summaries are kept (they are small) so survivors can be named afterwards.
    from .stats import boot_indices, spa_shard_max

    G = np.vstack(pnl_rows)                       # gross, (sessions, cells)
    SH = np.vstack(sh_rows)
    CN = np.vstack(cnt_rows)
    NET = G - SH * spread                         # the only series the verdict may use

    idx = boot_indices(len(sess_ids), reps, block)
    red = spa_shard_max(NET, idx, sess_ids)

    tot_sh = SH.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Gross dollars per share traded IS the breakeven round-trip spread.
        breakeven = np.where(tot_sh > 0, G.sum(axis=0) / tot_sh, np.nan)

    # Keep the full session series for this shard's best cells only, so a survivor can be
    # re-examined without re-running the grid. TOP_KEEP is fixed in advance; selecting it
    # after seeing the collector's output would be choosing what to keep by result.
    order = np.argsort(-NET.mean(axis=0))[:TOP_KEEP]

    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"shard_{shard:05d}.npz"
    np.savez_compressed(
        p,
        cell_index=np.array(keys, dtype=np.int64),
        session=np.array(sess_ids, dtype="U16"),
        session_hash=np.array(red["session_hash"]),
        spread=np.array(spread), block=np.array(block), reps=np.array(reps),
        net_mean=NET.mean(axis=0), net_sd=NET.std(axis=0, ddof=1),
        gross_mean=G.mean(axis=0), n_trades=CN.sum(axis=0).astype(np.int64),
        shares=tot_sh, breakeven=breakeven,
        t_obs_max=np.array(red["t_obs_max"]),
        argmax_cell=np.array(keys[red["argmax"]], dtype=np.int64),
        boot_max_consistent=red["boot_max_consistent"],
        boot_max_upper=red["boot_max_upper"],
        boot_max_lower=red["boot_max_lower"],
        top_cells=np.array([keys[i] for i in order], dtype=np.int64),
        top_series=NET[:, order].astype(np.float64),
    )
    print(f"  wrote {p}  cells={len(mine)}  sessions={len(sess_ids)}  "
          f"t_max={red['t_obs_max']:+.3f}  {time.time() - t0:.0f}s")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/momo")
    ap.add_argument("--symbols", nargs="+", default=["QQQ", "SPY"])
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default="results/momo_sweep")
    # Defaults are the pre-registration's DISCOVERY window. They are defaults rather than
    # something the caller must remember, because the failure mode of forgetting them is
    # silent: the sweep would run over the holdout too and still print a plausible number.
    ap.add_argument("--start", default="2016-01-01",
                    help="first session TRADED (earlier sessions still seed indicators)")
    ap.add_argument("--end", default="2021-12-31",
                    help="last session traded. The 2022+ holdout is not read at all.")
    ap.add_argument("--spread", type=float, default=0.01,
                    help="round-trip cost per SHARE. QQQ/SPY are penny-wide.")
    # `block` and `reps` MUST be identical across shards or the cross-shard max is not the
    # grid-wide max. They are explicit arguments rather than derived per-shard for exactly
    # that reason -- a per-shard optimal_block_length would differ by shard and silently
    # break the reduction. The collector refuses if the shards disagree.
    ap.add_argument("--block", type=float, default=5.0,
                    help="stationary-bootstrap mean block length, in sessions")
    ap.add_argument("--reps", type=int, default=10000)
    a = ap.parse_args(argv)

    d = Path(a.data)
    if not d.is_dir():
        print(f"FATAL: packed data not found at {d}. Run "
              f"`python -m trade_analysis.momo_sweep.pack` first.", file=sys.stderr)
        return 2
    if a.validate:
        # Validation deliberately spans EVERYTHING, holdout included. It is not a result --
        # it reproduces one already-published number to prove the machinery agrees with
        # momo_v2.py, which ran on the full sample. Restricting it to discovery would make
        # the anchor un-checkable.
        return validate(d, a.symbols)
    return run_shard(d, a.symbols, a.shard, a.nshards, Path(a.out), a.start, a.end,
                     a.spread, a.block, a.reps)


if __name__ == "__main__":
    sys.exit(main())
