"""Download the small reference datasets everything else depends on.

These are cheap (megabytes, minutes) and disproportionately useful, because two
of them prevent wasted work at a scale that dwarfs their own cost:

  /stock/list/dates/{trade,quote} tells you EXACTLY which sessions a symbol has
  data for. Without it, a 1s layer over 104 symbols x 2400 sessions issues
  ~250,000 requests, of which every one before a symbol's listing date is a
  guaranteed empty. ARM, CRWV, NBIS, SPCX and friends listed within the last few
  years, so that is tens of thousands of pointless calls. With it, the task list
  can be intersected against reality before a single byte moves.

  /option/list/expirations is what makes any option request constructible at
  all: /option/history/ohlc rejects expiration=*, so a per-contract pull needs
  the actual expiration list rather than a guess at the weekly cycle.

Also grabbed because they are free and answer questions that otherwise get
approximated: the exchange holiday calendar (so a gap can be attributed rather
than assumed) and the risk-free rate series (needed to price or value any option
position, and the only alternative is hardcoding a number that goes stale).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import date

from . import universe
from .config import REFERENCE_DIR
from .theta_v3 import ThetaV3

DATES_DIR = REFERENCE_DIR / "available_dates"
EXPIRY_DIR = REFERENCE_DIR / "expirations"


def download_available_dates(api: ThetaV3, symbols: list[str],
                             request_types=("trade", "quote")) -> dict:
    """Per symbol, the exact sessions the feed holds. See module docstring."""
    DATES_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for rt in request_types:
        got = {}
        for i, sym in enumerate(symbols, 1):
            rows = api.list_json(f"/stock/list/dates/{rt}", symbol=sym)
            dates = sorted({str(r["date"]) for r in rows if r.get("date")})
            got[sym] = dates
            sys.stdout.write(f"\r  {rt}: {i}/{len(symbols)} {sym:<6} "
                             f"{len(dates):,} dates    ")
            sys.stdout.flush()
        path = DATES_DIR / f"stock_{rt}_dates.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(got, fh)
        n = sum(len(v) for v in got.values())
        print(f"\r  {rt}: {len(got)} symbols, {n:,} symbol-dates -> {path.name}"
              f"{' ' * 20}")
        out[rt] = {"symbols": len(got), "symbol_dates": n, "path": str(path)}
    return out


def download_expirations(api: ThetaV3, symbols: list[str]) -> dict:
    EXPIRY_DIR.mkdir(parents=True, exist_ok=True)
    got = {}
    for i, sym in enumerate(symbols, 1):
        exps = api.list_expirations(sym)
        if exps:
            got[sym] = exps
        sys.stdout.write(f"\r  {i}/{len(symbols)} {sym:<6} "
                         f"{len(exps):,} expirations    ")
        sys.stdout.flush()
    path = EXPIRY_DIR / "option_expirations.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(got, fh)
    n = sum(len(v) for v in got.values())
    print(f"\r  {len(got)} roots, {n:,} expirations -> {path.name}{' ' * 24}")
    return {"roots": len(got), "expirations": n, "path": str(path)}


def download_strikes(api: ThetaV3, symbols: list[str],
                     expirations: dict, max_per_symbol: int | None = None) -> dict:
    """Strike lists per (root, expiration).

    Deliberately opt-in: SPY alone has ~1000 expirations, so this is one request
    per expiration per root - tens of thousands of calls for a list that the
    quote data itself already contains. Worth it only when contracts must be
    enumerated WITHOUT downloading their data (e.g. to drive /option/history/ohlc,
    which rejects expiration=*).
    """
    out = {}
    for sym in symbols:
        exps = expirations.get(sym, [])
        if max_per_symbol:
            exps = exps[-max_per_symbol:]      # most recent are most useful
        per = {}
        for j, exp in enumerate(exps, 1):
            per[exp] = api.list_strikes(sym, exp)
            sys.stdout.write(f"\r  {sym}: {j}/{len(exps)} expirations    ")
            sys.stdout.flush()
        out[sym] = per
    path = EXPIRY_DIR / "option_strikes.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(out, fh)
    total = sum(len(v) for per in out.values() for v in per.values())
    print(f"\r  {total:,} strike entries -> {path.name}{' ' * 30}")
    return {"strike_entries": total, "path": str(path)}


def download_misc(api: ThetaV3) -> dict:
    """Holiday calendar + risk-free rates. Both free, both tiny."""
    out = {}
    years = list(range(2016, date.today().year + 2))
    holidays = {}
    for y in years:
        try:
            rows = api.list_json("/calendar/year_holidays", year=y)
            if rows:
                holidays[y] = rows
        except Exception as exc:                              # noqa: BLE001
            # Not fatal - trading_days.py has its own table and falls back to
            # pandas_market_calendars. This is a cross-check, not a dependency.
            print(f"  calendar {y}: {type(exc).__name__}: {exc}")
    if holidays:
        p = REFERENCE_DIR / "exchange_holidays.json"
        p.write_text(json.dumps(holidays, indent=2, default=str), encoding="utf-8")
        print(f"  holidays: {len(holidays)} years -> {p.name}")
        out["holidays"] = str(p)

    try:
        rates = api.list_json("/interest_rate/history/eod",
                              start_date="20160101",
                              end_date=date.today().strftime("%Y%m%d"))
        if rates:
            p = REFERENCE_DIR / "interest_rates_eod.json"
            p.write_text(json.dumps(rates, default=str), encoding="utf-8")
            print(f"  interest rates: {len(rates):,} rows -> {p.name}")
            out["interest_rates"] = str(p)
    except Exception as exc:                                  # noqa: BLE001
        print(f"  interest rates unavailable: {type(exc).__name__}: {exc}")

    for name, path in (("stock_symbols", "/stock/list/symbols"),
                       ("option_roots", "/option/list/symbols")):
        rows = api.list_json(path)
        syms = sorted({r["symbol"] for r in rows if r.get("symbol")})
        p = REFERENCE_DIR / f"{name}.json.gz"
        with gzip.open(p, "wt", encoding="utf-8") as fh:
            json.dump(syms, fh)
        print(f"  {name}: {len(syms):,} -> {p.name}")
        out[name] = str(p)
    return out


def load_available_dates(request_type: str = "quote") -> dict[str, set[str]]:
    """For the downloaders: {symbol: {YYYY-MM-DD, ...}} or {} if not fetched."""
    path = DATES_DIR / f"stock_{request_type}_dates.json.gz"
    if not path.exists():
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        raw = json.load(fh)
    out = {}
    for sym, dates in raw.items():
        norm = set()
        for d in dates:
            s = str(d)
            norm.add(s if "-" in s else f"{s[:4]}-{s[4:6]}-{s[6:8]}")
        out[sym] = norm
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-dates", action="store_true")
    ap.add_argument("--skip-expirations", action="store_true")
    ap.add_argument("--strikes", action="store_true",
                    help="also enumerate strikes (expensive, see docstring)")
    ap.add_argument("--strikes-recent", type=int, default=60,
                    help="with --strikes, only the N most recent expirations")
    ap.add_argument("--symbols", nargs="*", default=None)
    args = ap.parse_args()

    symbols = args.symbols or universe.load()
    opt_symbols = args.symbols or universe.load(options_only=True)
    summary = {}
    with ThetaV3() as api:
        print(f"reference data for {len(symbols)} symbols -> {REFERENCE_DIR}")
        print("\n[misc: calendar, rates, symbol lists]")
        summary["misc"] = download_misc(api)

        if not args.skip_dates:
            print(f"\n[available session dates per symbol]")
            summary["dates"] = download_available_dates(api, symbols)

        if not args.skip_expirations:
            print(f"\n[option expirations for {len(opt_symbols)} roots]")
            summary["expirations"] = download_expirations(api, opt_symbols)

        if args.strikes:
            print(f"\n[option strikes, {args.strikes_recent} most recent expirations]")
            with gzip.open(EXPIRY_DIR / "option_expirations.json.gz", "rt",
                           encoding="utf-8") as fh:
                exps = json.load(fh)
            summary["strikes"] = download_strikes(api, universe.ETFS, exps,
                                                  args.strikes_recent)

    p = REFERENCE_DIR / "reference_manifest.json"
    p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
