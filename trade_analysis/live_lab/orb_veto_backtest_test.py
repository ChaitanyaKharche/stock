"""Does the veto's arithmetic do what the pre-registration says?

    python -m trade_analysis.live_lab.orb_veto_backtest_test
    pytest trade_analysis/live_lab/orb_veto_backtest_test.py

`analyse()` is where the experiment can go wrong quietly. It cannot crash -- it will
happily return a confident contrast while counting trades the veto could never have
suppressed, or dropping the days that make the denominator honest. So each check below
pins one clause of §4 or §6 with synthetic trades whose right answer is known by
construction. No archive, no feed.

The checks that matter most are the two that make the result FALSE rather than merely
imprecise:

  * a trade entered before T must not enter the contrast -- otherwise this measures a
    filter nobody can implement;
  * a day on which the family did not trade must stay in the denominator at 0.0 --
    otherwise the sample is conditioned on the family having fired, which correlates with
    expansion, and any veto looks good.
"""
from __future__ import annotations

import sys

from .orb_veto_backtest import analyse, boot_diff, verdict_text


def _t(day, pnl, at="12:00:00", setup="IMB"):
    return {"day": day, "setup": setup, "direction": "long",
            "entry_ts": f"{day}T{at}", "pnl": pnl}


def _days(n, prefix="2026-01"):
    """n distinct day keys; enough of them that boot_diff will run (needs >= 8/arm)."""
    return [f"{prefix}-{i + 1:02d}" for i in range(n)]


# --------------------------------------------------------------- §4 the restriction

def test_a_trade_entered_before_T_is_excluded_from_the_contrast():
    """A veto decided at 11:00 cannot suppress a 09:36 entry.

    Counting it would measure a filter that cannot be implemented -- and it would do so
    silently, with a plausible-looking CI.
    """
    days = _days(20)
    vetoed = {d: (i % 2 == 0) for i, d in enumerate(days)}
    # Every veto day carries a big LOSS, but at 09:36 -- before T.
    trades = [_t(d, -500.0, at="09:36:00") for d in days if vetoed[d]]
    res = analyse(trades, vetoed)
    assert res["mean_D_veto"] == 0.0, res["mean_D_veto"]
    assert res["n_trades_post_T"] == 0, res["n_trades_post_T"]
    assert res["pnl_removed_by_veto"] == 0.0, "pre-T P&L was attributed to the veto"


def test_a_trade_entered_after_T_does_enter_the_contrast():
    days = _days(20)
    vetoed = {d: (i % 2 == 0) for i, d in enumerate(days)}
    trades = [_t(d, -500.0, at="11:00:01") for d in days if vetoed[d]]
    res = analyse(trades, vetoed)
    assert res["n_trades_post_T"] == 10, res["n_trades_post_T"]
    assert res["mean_D_veto"] == -500.0, res["mean_D_veto"]


def test_exactly_T_is_excluded_not_included():
    """`entry_ts > T`, strictly. An 11:00:00 entry is simultaneous with the decision and
    is not something the gate can be shown to have prevented."""
    days = _days(20)
    vetoed = {d: True for d in days}
    res = analyse([_t(d, -100.0, at="11:00:00") for d in days], vetoed)
    assert res["n_trades_post_T"] == 0, "an 11:00:00 entry was counted as post-T"


def test_a_day_with_no_trades_stays_in_the_denominator_at_zero():
    """§4. Dropping it conditions the sample on the family having fired, which correlates
    with expansion -- the selection effect that makes any veto look good."""
    days = _days(20)
    vetoed = {d: (i % 2 == 0) for i, d in enumerate(days)}
    trades = [_t(days[0], -1000.0)]            # one veto day trades; nineteen do not
    res = analyse(trades, vetoed)
    assert res["n_days_classified"] == 20
    assert res["n_veto_days"] == 10 and res["n_tradeable_days"] == 10
    assert res["mean_D_veto"] == -100.0, res["mean_D_veto"]   # -1000 over TEN days


def test_an_unclassified_day_is_excluded_from_everything():
    """A day we could not classify must not be silently counted as tradeable."""
    days = _days(20)
    vetoed = {d: False for d in days}
    trades = [_t(days[0], 10.0), _t("1999-01-01", 99999.0)]
    res = analyse(trades, vetoed)
    assert res["n_trades_total"] == 1, "an unclassified day's trades leaked in"
    assert res["total_unfiltered"] == 10.0


# --------------------------------------------------------------------- §6.2 does it help

def test_filtering_helps_when_the_losses_sit_on_veto_days_after_T():
    days = _days(24)
    vetoed = {d: (i % 2 == 0) for i, d in enumerate(days)}
    trades = ([_t(d, -300.0) for d in days if vetoed[d]]
              + [_t(d, +200.0) for d in days if not vetoed[d]])
    res = analyse(trades, vetoed)
    assert res["improves"] is True
    assert res["mean_diff"] < 0, res["mean_diff"]
    assert res["ci95"][1] < 0, res["ci95"]
    assert res["total_filtered"] > res["total_unfiltered"]
    assert res["pnl_removed_by_veto"] == -3600.0, res["pnl_removed_by_veto"]


def test_filtering_does_not_help_when_pnl_is_the_same_on_both_kinds_of_day():
    """The null. A veto that removes a representative slice cannot improve anything."""
    days = _days(24)
    vetoed = {d: (i % 2 == 0) for i, d in enumerate(days)}
    trades = [_t(d, 100.0) for d in days]
    res = analyse(trades, vetoed)
    assert res["mean_diff"] == 0.0, res["mean_diff"]
    assert res["improves"] is False
    assert res["p_two_sided"] is not None and res["p_two_sided"] > 0.05


def test_the_best_day_cannot_carry_the_result_alone():
    """§6.2. One session's P&L must not be what makes the filter look good."""
    days = _days(24)
    vetoed = {d: (i % 2 == 0) for i, d in enumerate(days)}
    # Flat everywhere, except ONE catastrophic veto day.
    trades = [_t(d, 0.0) for d in days] + [_t(days[0], -50000.0)]
    res = analyse(trades, vetoed)
    assert res["improves"] is True, "removing a 50k loss did not improve the total"
    # days[0] is the WORST day, not the best, so dropping the best must not rescue it.
    assert res["improves_without_best_day"] is True


# ------------------------------------------------------------------ §6.3 the tail check

def test_the_tail_check_flags_a_veto_that_removes_the_winners():
    """The one that can kill this on its own.

    IMB carries 73.9% of its P&L in the top 1% of trades. A veto that removes contained
    days is only useful if it KEEPS those. Here the big winners sit on veto days, so the
    filter must be reported HARMFUL even though it also removes losses.
    """
    days = _days(24)
    vetoed = {d: (i % 2 == 0) for i, d in enumerate(days)}
    trades = [_t(d, -50.0) for d in days]
    trades += [_t(days[0], 100000.0), _t(days[2], 90000.0)]   # both on veto days
    res = analyse(trades, vetoed)
    assert res["top1pct_share_on_veto_days"] == 1.0, res["top1pct_share_on_veto_days"]
    assert set(res["top10_days_vetoed"]) >= {days[0], days[2]}
    assert any("HARMFUL" in line for line in verdict_text(res)), verdict_text(res)


def test_the_tail_check_passes_when_the_winners_are_on_tradeable_days():
    days = _days(24)
    vetoed = {d: (i % 2 == 0) for i, d in enumerate(days)}
    trades = [_t(d, -300.0) for d in days if vetoed[d]]
    trades += [_t(d, 50.0) for d in days if not vetoed[d]]
    trades += [_t(days[1], 100000.0), _t(days[3], 90000.0)]   # tradeable days
    res = analyse(trades, vetoed)
    assert res["top1pct_share_on_veto_days"] == 0.0
    assert res["top10_days_vetoed"] == []
    assert not any("HARMFUL" in line for line in verdict_text(res))


def test_the_verdict_refuses_to_conclude_without_enough_days():
    """Better to say INCONCLUSIVE than to print a contrast from six days."""
    days = _days(4)
    vetoed = {d: (i % 2 == 0) for i, d in enumerate(days)}
    res = analyse([_t(d, 10.0) for d in days], vetoed)
    assert res["mean_diff"] is None
    assert any("INCONCLUSIVE" in line for line in verdict_text(res))


# ------------------------------------------------------------------------- inference

def test_boot_diff_is_clustered_by_day_not_by_trade():
    """Days are the unit. Treating trades as independent inflated significance ~300x in
    the VRP arm; the same error here would be a false positive on the whole hypothesis.

    Twelve days per arm, identical within a day. Resampling DAYS must leave visible
    uncertainty; resampling trades would collapse the CI toward zero width.
    """
    a = {f"a{i}": (100.0 if i % 2 else -100.0) for i in range(12)}
    b = {f"b{i}": 0.0 for i in range(12)}
    obs, lo, hi, p = boot_diff(a, b, reps=2000)
    assert abs(obs - 0.0) < 1e-9, obs
    assert hi - lo > 40.0, f"CI width {hi - lo:.2f} is too tight for 12 clusters"
    assert p > 0.05


def test_boot_diff_refuses_a_thin_arm():
    assert boot_diff({f"a{i}": 1.0 for i in range(3)},
                     {f"b{i}": 0.0 for i in range(20)}) == (None, None, None, None)


def test_boot_diff_is_deterministic():
    """A pre-registered result that changes between runs is not a result."""
    a = {f"a{i}": float(i) for i in range(15)}
    b = {f"b{i}": float(i) * 0.5 for i in range(15)}
    assert boot_diff(a, b, reps=500) == boot_diff(a, b, reps=500)


CHECKS = [(n, f) for n, f in sorted(globals().items())
          if n.startswith("test_") and callable(f)]


def main() -> int:
    ok = True
    print("=" * 78)
    print("ORB VETO -- THE ARITHMETIC OF SECTIONS 4 AND 6")
    print("=" * 78)
    for name, fn in CHECKS:
        try:
            fn()
            print(f"  [PASS] {name[5:].replace('_', ' ')}")
        except AssertionError as exc:
            ok = False
            print(f"  [FAIL] {name[5:].replace('_', ' ')}\n         {exc}")
        except Exception as exc:                              # noqa: BLE001
            ok = False
            print(f"  [FAIL] {name[5:].replace('_', ' ')}\n         raised {exc!r}")
    print(f"\n  {len(CHECKS)} checks on synthetic trades")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    print("\n  A filter credited with P&L it could not have changed is not a filter.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
