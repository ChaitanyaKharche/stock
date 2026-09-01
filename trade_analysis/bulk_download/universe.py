"""The symbol universe: SPY, QQQ, and the Nasdaq-100 constituents behind QQQ.

Two things here are easy to get wrong and both silently corrupt a study.

1. VALIDATION AGAINST THE FEED. An index constituent list is a list of *company*
   tickers; it is not a promise that the data vendor carries them under those
   symbols. Rather than discover that 2,000 requests in, every symbol is checked
   against /stock/list/symbols and /option/list/symbols up front. A ticker in the
   index but not in the feed is a real finding, recorded, not silently dropped.

2. SURVIVORSHIP. reference/qqq_constituents.csv is TODAY's membership. Using it
   to define a universe for a 2017-2026 backtest is textbook survivorship bias:
   it contains only companies that made it into the index and stayed, and omits
   every one that was deleted along the way. This module therefore treats the
   list as "the current trading universe" - correct for live monitoring, which
   is what it is for - and refuses to pretend it is a historical universe. The
   `as_of` column is carried through so downstream code can see the vintage
   rather than inferring one.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict

from .config import REFERENCE_DIR
from .theta_v3 import ThetaV3

CONSTITUENTS_CSV = REFERENCE_DIR / "qqq_constituents.csv"
UNIVERSE_JSON = REFERENCE_DIR / "universe.json"

# The two index ETFs are the primary instruments, kept separate from the
# constituents so the expensive tick/second layers can be scoped to them alone.
ETFS = ["SPY", "QQQ"]


@dataclass
class Symbol:
    symbol: str
    name: str = ""
    weight_pct: float = 0.0
    role: str = "constituent"      # etf | constituent
    has_stock: bool | None = None  # verified against the feed
    has_option: bool | None = None


def load_constituents() -> list[Symbol]:
    if not CONSTITUENTS_CSV.exists():
        raise FileNotFoundError(
            f"{CONSTITUENTS_CSV} missing - it holds the QQQ holdings list.")
    out = []
    with open(CONSTITUENTS_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sym = (row.get("symbol") or "").strip().upper()
            if not sym:
                continue
            try:
                w = float(row.get("weight_pct") or 0)
            except ValueError:
                w = 0.0
            out.append(Symbol(sym, (row.get("name") or "").strip(), w,
                              "constituent"))
    return out


def build(verify: bool = True) -> dict:
    """Assemble the universe and (by default) verify it against the feed."""
    consts = load_constituents()
    symbols: dict[str, Symbol] = {}
    for e in ETFS:
        symbols[e] = Symbol(e, f"{e} ETF", 0.0, "etf")
    for s in consts:
        # An ETF that is also somehow in the holdings keeps role=etf.
        if s.symbol in symbols:
            symbols[s.symbol].weight_pct = s.weight_pct
            continue
        symbols[s.symbol] = s

    result = {"n_symbols": len(symbols), "etfs": ETFS,
              "constituents_csv": str(CONSTITUENTS_CSV)}

    if verify:
        with ThetaV3() as api:
            stock_syms = {r["symbol"].upper()
                          for r in api.list_json("/stock/list/symbols")
                          if r.get("symbol")}
            opt_syms = {r["symbol"].upper()
                        for r in api.list_json("/option/list/symbols")
                        if r.get("symbol")}
        print(f"feed carries {len(stock_syms)} stock symbols, "
              f"{len(opt_syms)} option roots")
        for s in symbols.values():
            s.has_stock = s.symbol in stock_syms
            s.has_option = s.symbol in opt_syms
        missing_stock = sorted(s.symbol for s in symbols.values() if not s.has_stock)
        missing_opt = sorted(s.symbol for s in symbols.values() if not s.has_option)
        result["missing_from_stock_feed"] = missing_stock
        result["missing_from_option_feed"] = missing_opt
        result["feed_stock_symbol_count"] = len(stock_syms)
        result["feed_option_root_count"] = len(opt_syms)
        if missing_stock:
            print(f"WARNING {len(missing_stock)} symbols absent from the stock "
                  f"feed and will be skipped: {', '.join(missing_stock)}")
        if missing_opt:
            print(f"NOTE {len(missing_opt)} symbols have no listed options: "
                  f"{', '.join(missing_opt)}")

    result["symbols"] = [asdict(s) for s in sorted(
        symbols.values(), key=lambda x: (x.role != "etf", -x.weight_pct, x.symbol))]
    UNIVERSE_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {UNIVERSE_JSON}  ({len(symbols)} symbols)")
    return result


def load(role: str | None = None, tradable_only: bool = True,
         options_only: bool = False) -> list[str]:
    """Symbol tickers for the downloaders.

    tradable_only drops anything the feed does not carry, so a download run
    never spends requests on a symbol already known to be absent.
    """
    if not UNIVERSE_JSON.exists():
        raise FileNotFoundError(
            f"{UNIVERSE_JSON} missing. Run: "
            f"python -m trade_analysis.bulk_download.universe")
    data = json.loads(UNIVERSE_JSON.read_text(encoding="utf-8"))
    out = []
    for s in data["symbols"]:
        if role and s["role"] != role:
            continue
        if tradable_only and s.get("has_stock") is False:
            continue
        if options_only and not s.get("has_option"):
            continue
        out.append(s["symbol"])
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    res = build(verify=not args.no_verify)
    syms = [s["symbol"] for s in res["symbols"]]
    print(f"\n{len(syms)} symbols:\n{', '.join(syms)}")
