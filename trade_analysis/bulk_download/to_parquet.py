"""Convert the raw csv.gz archive into typed, query-ready Parquet.

The raw tree stays exactly as the API returned it - that is the archive, and it
is what makes every later decision re-checkable. This module produces the
*working* copy, and does three things the raw CSV cannot:

  TYPES. Everything in a CSV is a string. Prices read as float64 and sizes as
  int64 is a 3-5x size reduction on top of what gzip already did, and it stops
  a downstream join silently comparing "532.000" to 532.0.

  TIMEZONE. The API returns naive local timestamps ("2024-08-07T09:30:00.000")
  which are Eastern. This machine is not on Eastern. Leaving them naive shifts
  every session window by hours - the single most expensive class of bug in this
  project's history - so they are localized explicitly here, once, and every
  consumer reads tz-aware data.

  CONSOLIDATION. The sub-minute layers are one file per symbol-session: a decade
  of 1s quotes across 104 symbols is ~250,000 files. Parquet on that many tiny
  files is slower than the CSV it replaced. Sessions are therefore folded into
  one file per symbol-month, which is the granularity a study actually scans.

Idempotent: a partition is rebuilt only if a source file is newer than the
output, so re-running after an incremental download is cheap.
"""
from __future__ import annotations

import argparse
import gzip
import io
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from . import layers as layers_mod
from .config import ET, PARQUET_DIR, RAW_DIR

# Columns that are prices (float), sizes/counts (int), and exchange/condition
# codes (small int). Declared rather than inferred: pandas will happily read an
# all-zero bid column as int64 in one month and float64 in the next, and then
# the two parquet files cannot be concatenated without a cast.
FLOAT_COLS = {"open", "high", "low", "close", "vwap", "bid", "ask", "price",
              "strike", "market_bid", "market_ask", "market_price",
              "implied_vol", "underlying_price"}
INT_COLS = {"volume", "count", "bid_size", "ask_size", "size", "sequence",
            "open_interest", "n_trades"}
SMALL_INT_COLS = {"bid_exchange", "ask_exchange", "exchange", "bid_condition",
                  "ask_condition", "condition", "ext_condition1",
                  "ext_condition2", "ext_condition3", "ext_condition4"}
TIME_COLS = {"timestamp", "trade_timestamp", "quote_timestamp", "last_trade",
             "created", "underlying_timestamp"}


def _read_csv_gz(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    if not raw.strip():
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(raw), low_memory=False)


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col in TIME_COLS:
            ts = pd.to_datetime(df[col], errors="coerce")
            # tz_localize only applies to a naive series; the API is naive ET.
            # ambiguous/nonexistent handling matters on DST boundaries - without
            # it the two DST days a year raise and kill the whole conversion.
            if getattr(ts.dt, "tz", None) is None:
                ts = ts.dt.tz_localize(ET, ambiguous=True,
                                       nonexistent="shift_forward")
            df[col] = ts
        elif col in FLOAT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        elif col in INT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif col in SMALL_INT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")
        elif col == "right":
            # "CALL"/"PUT" -> C/P as a category: this column repeats for
            # millions of rows and is the single biggest storage win available.
            df[col] = (df[col].astype(str).str[0].str.upper()
                       .astype("category"))
        elif col in ("symbol", "root"):
            df[col] = df[col].astype("category")
        elif col == "expiration":
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date.astype(str)
    return df


def _partition_key(layer: layers_mod.Layer, file_key: str) -> str:
    """Which parquet file a raw chunk belongs to.

    Monthly layers map 1:1. Daily layers are folded to their month - see the
    consolidation note in the module docstring.
    """
    return file_key[:7]     # YYYY-MM for both cases


def convert_layer(layer_name: str, symbols: list[str] | None = None,
                  force: bool = False) -> dict:
    layer = layers_mod.get(layer_name)
    src_root = RAW_DIR / layer.dir_name
    if not src_root.exists():
        print(f"  {layer_name}: nothing downloaded yet")
        return {"layer": layer_name, "files": 0, "rows": 0}

    # Group every raw file by (symbol, target parquet partition).
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in src_root.rglob("*.csv.gz"):
        sym = path.parent.parent.name
        if symbols and sym not in symbols:
            continue
        # filename is SYMBOL_KEY.csv.gz
        file_key = path.name[len(sym) + 1:].replace(".csv.gz", "")
        groups[(sym, _partition_key(layer, file_key))].append(path)

    n_files = n_rows = 0
    total = len(groups)
    for i, ((sym, part), paths) in enumerate(sorted(groups.items()), 1):
        dest = PARQUET_DIR / layer.dir_name / sym / f"{sym}_{part}.parquet"
        if not force and dest.exists():
            newest = max(p.stat().st_mtime for p in paths)
            if dest.stat().st_mtime >= newest:
                continue
        frames = []
        for p in sorted(paths):
            try:
                df = _read_csv_gz(p)
            except Exception as exc:                          # noqa: BLE001
                # A corrupt member must not abort a 10-year conversion. Report
                # it and continue; the raw file is still on disk to inspect.
                print(f"\n    SKIP unreadable {p.name}: "
                      f"{type(exc).__name__}: {exc}")
                continue
            if not df.empty:
                frames.append(df)
        if not frames:
            continue
        out = _coerce(pd.concat(frames, ignore_index=True))
        # The symbol is in the path, not the rows, for stock layers - add it so
        # a concatenated multi-symbol frame is self-describing.
        if "symbol" not in out.columns:
            out["symbol"] = pd.Categorical([sym] * len(out))
        sort_cols = [c for c in ("timestamp", "trade_timestamp", "expiration",
                                 "strike", "right") if c in out.columns]
        if sort_cols:
            out = out.sort_values(sort_cols).reset_index(drop=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(dest, compression="zstd", index=False)
        n_files += 1
        n_rows += len(out)
        if i % 20 == 0 or i == total:
            sys.stdout.write(f"\r    {i:,}/{total:,} partitions  "
                             f"{n_rows:,} rows written   ")
            sys.stdout.flush()
    print(f"\r  {layer_name}: wrote {n_files:,} parquet files, {n_rows:,} rows"
          f"{' ' * 20}")
    return {"layer": layer_name, "files": n_files, "rows": n_rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", nargs="*", default=None,
                    help="default: every layer with raw data on disk")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--force", action="store_true",
                    help="rebuild partitions even if they look current")
    args = ap.parse_args()

    names = args.layers
    if not names:
        names = [l.name for l in layers_mod.LAYERS.values()
                 if (RAW_DIR / l.dir_name).exists()]
        if not names:
            print(f"no raw data under {RAW_DIR}")
            return
    print(f"converting {len(names)} layer(s) -> {PARQUET_DIR}")
    for name in names:
        convert_layer(name, args.symbols, args.force)


if __name__ == "__main__":
    main()
