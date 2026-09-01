"""Pull the complete Robinhood OPTION ORDER history, with execution timestamps.

WHY THIS EXISTS
---------------
The account-activity CSV export is date-only. It cannot answer any question about
entry timing, exit timing, or retest-versus-breakout behaviour, and it omits
cancelled/rejected orders entirely. The order endpoint carries all of it:

    created_at            when the order was placed
    updated_at            when it reached its final state
    legs[].executions[]   per-fill timestamp, price, quantity
    type                  limit / market
    price                 the limit price
    time_in_force         gfd / gtc
    state                 filled / cancelled / rejected

`created_at` vs the execution `timestamp` is what distinguishes chasing a move
from resting a bid at a level -- i.e. a breakout entry from a retest entry.

CREDENTIALS
-----------
This script NEVER takes credentials as arguments and never stores them. It calls
robin_stocks' interactive login, which prompts you and handles MFA. Run it
yourself; nobody else needs to see your password.

After it finishes, robin_stocks leaves a session token cached at
    ~/.tokens/robinhood.pickle
Treat that like a credential. Delete it when you are done:
    rm ~/.tokens/robinhood.pickle

NOTE: robin_stocks is an UNOFFICIAL client. Robinhood's terms restrict automated
access. This reads only your own data and places no orders, but the decision to
use it is yours.

USAGE
-----
    pip install robin_stocks
    python research/pull_order_history.py

Writes research/order_history_raw.json (gitignored). No trading action is taken:
this script only issues GETs.
"""
import json
import sys
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent / "order_history_raw.json"

try:
    import robin_stocks.robinhood as rh
except ImportError:
    sys.exit("robin_stocks not installed.  Run:  pip install robin_stocks")


def main():
    print("Logging in -- you will be prompted. Nothing is stored by this script.")
    rh.login()  # interactive: prompts for username, password, MFA

    print("Fetching all option orders (this pages through your full history)...")
    orders = rh.orders.get_all_option_orders()
    if not orders:
        sys.exit("No option orders returned. Is this the right account?")
    print("  %d orders returned" % len(orders))

    # Resolve each leg's option instrument URL -> strike / expiry / right.
    # Cached so each unique contract costs exactly one request.
    cache = {}
    n_new = 0
    for i, o in enumerate(orders, 1):
        for leg in o.get("legs", []):
            url = leg.get("option")
            if not url:
                continue
            if url not in cache:
                try:
                    cache[url] = rh.helper.request_get(url)
                    n_new += 1
                except Exception as e:
                    cache[url] = {"_error": str(e)[:120]}
            leg["_instrument"] = cache[url]
        if i % 100 == 0:
            print("  resolved instruments for %d/%d orders (%d unique contracts)"
                  % (i, len(orders), n_new), flush=True)

    OUT.write_text(json.dumps(orders, indent=1), encoding="utf-8")

    # Summary only -- no credential, no token, no account number printed.
    states = Counter(o.get("state") for o in orders)
    syms = Counter(o.get("chain_symbol") for o in orders)
    dated = sorted(o["created_at"] for o in orders if o.get("created_at"))
    n_exec = sum(len(leg.get("executions") or [])
                 for o in orders for leg in o.get("legs", []))

    print("\n" + "=" * 62)
    print("WROTE %s" % OUT)
    print("=" * 62)
    print("orders                : %d" % len(orders))
    print("unique contracts      : %d" % len(cache))
    print("individual executions : %d" % n_exec)
    print("date range (created)  : %s  ->  %s" % (dated[0][:10], dated[-1][:10]))
    print("states                : %s" % dict(states))
    print("symbols               : %s" % dict(syms.most_common(12)))
    print("\nReminder: delete the cached session token when finished:")
    print("    rm ~/.tokens/robinhood.pickle")


if __name__ == "__main__":
    main()
