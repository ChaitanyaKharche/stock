"""Measure what this account can actually download, and how far back.

Run this FIRST and re-run it after any subscription change. Everything else in
the package reads its output (reference/entitlements.json) instead of hardcoding
a start date, because the two facts it establishes are the ones most often
guessed wrong:

  1. ENTITLEMENT. A 403 and an empty response look identical in a directory
     listing. Only the 403 body distinguishes "we are not paying for this" from
     "this genuinely has no rows". Guessing here is how you end up with an
     archive that looks complete and has a tier-shaped hole in it.

  2. HISTORY FLOOR, per layer and interval - NOT per subscription. These differ
     sharply: minute bars reach back years further than second bars, which reach
     further than ticks. A study that assumes one floor for all of them either
     silently truncates or spends hours requesting empty years.

The floor is found by binary search over trading days, which needs ~12 probes
per layer instead of the ~2500 a linear scan would take. Probes are status-only
(see ThetaV3.probe) so a floor search costs kilobytes, not the gigabytes the
same requests would return in full.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta

from .config import REFERENCE_DIR, TIERS
from .manifest import Manifest
from .theta_v3 import ThetaV3
from .trading_days import nearest_trading_day, trading_days

# Layers to test. `probe_args` is a callable taking a date and returning the
# request kwargs, so the same binary search drives both stock and option paths
# despite their different parameter shapes.
#
# SPY is the probe symbol throughout: it is the most liquid, longest-listed
# instrument available, so a floor found with SPY is the FLOOR OF THE FEED, not
# of the symbol. Any other ticker can only be listed later, never earlier - so a
# per-symbol floor must still be handled at download time (a NO_DATA before a
# symbol's IPO is expected and gets recorded, not retried).
LAYERS = {
    # ---- stock (STANDARD) ----
    "stock.ohlc.1m":     dict(kind="stock", endpoint="ohlc",  interval="1m"),
    "stock.ohlc.1s":     dict(kind="stock", endpoint="ohlc",  interval="1s"),
    "stock.quote.1m":    dict(kind="stock", endpoint="quote", interval="1m"),
    "stock.quote.1s":    dict(kind="stock", endpoint="quote", interval="1s"),
    "stock.quote.tick":  dict(kind="stock", endpoint="quote", interval=None),
    "stock.trade.tick":  dict(kind="stock", endpoint="trade", interval=None),
    "stock.eod":         dict(kind="stock", endpoint="eod",   interval=None),
    # ---- option (VALUE) ----
    "option.quote.1m":   dict(kind="option", endpoint="quote", interval="1m"),
    "option.quote.1s":   dict(kind="option", endpoint="quote", interval="1s"),
    "option.quote.tick": dict(kind="option", endpoint="quote", interval=None),
    "option.ohlc.1m":    dict(kind="option", endpoint="ohlc",  interval="1m"),
    "option.trade.tick": dict(kind="option", endpoint="trade", interval=None),
    "option.open_interest": dict(kind="option", endpoint="open_interest", interval=None),
    "option.eod":        dict(kind="option", endpoint="eod",   interval=None),
}

PROBE_SYMBOL = "SPY"
# A recent, unremarkable session used to answer "is this layer available at
# all". Must be a date every layer plausibly covers, so the entitlement test is
# never confounded with the floor test.
KNOWN_GOOD = date(2024, 8, 7)
SEARCH_FLOOR = date(2004, 1, 2)   # earlier than any plausible feed start


def _probe_one(api: ThetaV3, spec: dict, day: date, expirations: list[str]):
    """One status-only probe. Returns (status_code, body_snippet)."""
    ds = day.strftime("%Y%m%d")
    if spec["kind"] == "stock":
        return api.probe(f"/stock/history/{spec['endpoint']}", symbol=PROBE_SYMBOL,
                         start_date=ds, end_date=ds, interval=spec["interval"],
                         venue="utp_cta")

    # Options need an expiration that actually existed on `day`. Using a fixed
    # expiration would make every probe outside its life look like NO_DATA and
    # the binary search would converge on nonsense. Pick the first expiration
    # on or after the probe date - i.e. the front-month/0DTE contract for that
    # day, which is the one most likely to have data if any does.
    exp = next((e for e in expirations if e >= day.isoformat()), None)
    if exp is None:
        return 472, "no expiration on/after probe date"
    kw = dict(symbol=PROBE_SYMBOL, expiration=exp.replace("-", ""), date=ds,
              interval=spec["interval"])
    if spec["endpoint"] in ("quote", "ohlc"):
        # A single ATM-ish strike would need spot for that day, which we don't
        # have during a probe. strike_range=2 keeps the payload small while
        # still guaranteeing real contracts.
        kw["strike_range"] = 2
    if spec["endpoint"] == "eod":
        kw = dict(symbol=PROBE_SYMBOL, expiration=exp.replace("-", ""),
                  start_date=ds, end_date=ds)
    return api.probe(f"/option/history/{spec['endpoint']}", **kw)


def _classify(code: int, body: str) -> str:
    if code == 403:
        return "NOT_ENTITLED"
    if code == 200:
        return "OK" if body.strip() and "\n" in body.strip() else "NO_DATA"
    if code in (472, 404):
        return "NO_DATA" if "<html" not in body.lower() else "BAD_PATH"
    if code == 400:
        return "BAD_REQUEST"
    return "ERROR"


def find_floor(api: ThetaV3, layer: str, spec: dict, expirations: list[str],
               verbose: bool = True) -> dict:
    """Binary-search the earliest trading day this layer returns rows for."""
    # Step 1: is it available at all, and is the path even right?
    code, body = _probe_one(api, spec, KNOWN_GOOD, expirations)
    status = _classify(code, body)
    if status in ("NOT_ENTITLED", "BAD_PATH", "BAD_REQUEST"):
        return {"layer": layer, "available": False, "first_date": None,
                "reason": status, "note": body.strip()[:200]}
    if status != "OK":
        # Not entitled and not present on a known-good day: report rather than
        # search, since a search over an unavailable layer just burns probes.
        return {"layer": layer, "available": False, "first_date": None,
                "reason": f"no data on known-good {KNOWN_GOOD}", "note": body[:200]}

    # Step 2: bracket. Walk back by doubling until a probe comes up empty, so
    # the search space is bounded before bisecting it.
    days = trading_days(SEARCH_FLOOR, KNOWN_GOOD)
    lo, hi = 0, len(days) - 1          # lo = unknown/empty side, hi = known-good
    step = 64
    probe_count = 1
    known_good_idx = hi
    while True:
        idx = max(0, known_good_idx - step)
        code, body = _probe_one(api, spec, days[idx], expirations)
        probe_count += 1
        if _classify(code, body) == "OK":
            known_good_idx = idx
            if idx == 0:
                lo = -1
                break
            step *= 2
        else:
            lo = idx
            break
    hi = known_good_idx

    # Step 3: bisect between the empty lo and the good hi.
    while hi - lo > 1:
        mid = (lo + hi) // 2
        code, body = _probe_one(api, spec, days[mid], expirations)
        probe_count += 1
        if _classify(code, body) == "OK":
            hi = mid
        else:
            lo = mid

    first = days[hi]
    if verbose:
        print(f"    floor {first}  ({probe_count} probes)")
    return {"layer": layer, "available": True, "first_date": first.isoformat(),
            "reason": "measured", "note": f"binary search, {probe_count} probes"}


def _flush(out: dict):
    """Merge-and-write entitlements.json.

    Merges rather than overwrites so a partial re-probe (e.g. `--layers
    option.eod`) cannot delete results measured in an earlier run.
    """
    dest = REFERENCE_DIR / "entitlements.json"
    merged: dict = {}
    if dest.exists():
        try:
            merged = json.loads(dest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            merged = {}
    layers = merged.get("layers", {})
    layers.update(out.get("layers", {}))
    merged.update(out)
    merged["layers"] = layers
    dest.write_text(json.dumps(merged, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", nargs="*", default=None,
                    help="subset of layer names; default all")
    ap.add_argument("--no-floor", action="store_true",
                    help="only test availability, skip the floor search")
    args = ap.parse_args()

    layers = args.layers or list(LAYERS)
    out = {"tiers": TIERS, "probe_symbol": PROBE_SYMBOL, "layers": {}}
    man = Manifest()

    with ThetaV3() as api:
        print("listing SPY option expirations (needed to probe option layers)...")
        expirations = api.list_expirations(PROBE_SYMBOL)
        print(f"  {len(expirations)} expirations, {expirations[0]} -> {expirations[-1]}")
        out["spy_expirations"] = {"count": len(expirations),
                                  "first": expirations[0], "last": expirations[-1]}

        for layer in layers:
            spec = LAYERS[layer]
            print(f"\n[{layer}]")
            if args.no_floor:
                code, body = _probe_one(api, spec, KNOWN_GOOD, expirations)
                st = _classify(code, body)
                rec = {"layer": layer, "available": st == "OK",
                       "first_date": None, "reason": st, "note": body[:200]}
                print(f"    {st}: {body[:120]!r}")
            else:
                rec = find_floor(api, layer, spec, expirations)
                if not rec["available"]:
                    print(f"    UNAVAILABLE ({rec['reason']}): {rec['note'][:140]}")
            out["layers"][layer] = rec
            man.record_entitlement(layer, rec["available"], rec["first_date"],
                                   f"{rec['reason']}: {rec['note']}")
            # Flush after EVERY layer, not at the end. A full probe takes tens
            # of minutes - the option tick layers are slow to generate
            # server-side - and an interrupted run previously lost the entire
            # file even though the manifest had already recorded each result.
            _flush(out)

        out["client_stats"] = {"live_calls": api.live_calls, "retries": api.retries}

    _flush(out)
    print(f"\nwrote {REFERENCE_DIR / 'entitlements.json'}")

    print("\n{:<24} {:<12} {}".format("LAYER", "FIRST DATE", "STATUS"))
    print("-" * 62)
    for layer, rec in out["layers"].items():
        print("{:<24} {:<12} {}".format(
            layer, rec["first_date"] or "-",
            "available" if rec["available"] else f"NO ({rec['reason']})"))


def load_entitlements() -> dict:
    """Read the measured entitlements. Downloaders call this rather than
    assuming a floor, so a subscription change is picked up by re-probing.

    The MANIFEST is the source of truth, not the JSON file: the manifest is
    written per layer as the probe runs, whereas the JSON is only written when
    the whole probe finishes. Since a full probe takes a while (the option tick
    layers are slow to generate server-side), reading only the JSON means a
    download run started mid-probe sees nothing and skips every layer. The JSON
    is still merged in as a human-readable export and a fallback if the manifest
    has been cleared.
    """
    merged: dict = {"layers": {}}
    path = REFERENCE_DIR / "entitlements.json"
    if path.exists():
        merged.update(json.loads(path.read_text(encoding="utf-8")))

    from .manifest import Manifest
    for row in Manifest().entitlements():
        merged["layers"][row["layer"]] = {
            "layer": row["layer"],
            "available": bool(row["available"]),
            "first_date": row["first_date"],
            "reason": "manifest",
            "note": row["note"] or "",
        }
    if not merged["layers"]:
        raise FileNotFoundError(
            "no entitlement data yet. Run:  python -m "
            "trade_analysis.bulk_download.probe_entitlements")
    return merged


def layer_floor(layer: str, default: str | None = None) -> str | None:
    ent = load_entitlements()["layers"].get(layer)
    if not ent or not ent.get("available"):
        return default
    return ent.get("first_date") or default


if __name__ == "__main__":
    main()
