# Pre-registration — does a close-based range-expansion veto improve the shares family?

**Status: PRE-REGISTERED, NOT YET RUN.** Written 2026-09-17. Frozen before any number in
§6 is computed. The design came from the trader reading a QQQ 10-minute chart on
2026-09-17 and saying the day was untradeable; it is his hypothesis, stated in his terms,
and then made falsifiable.

## 0. Why this is not a tenth attempt at the same thing

Every prior null tested **which way** or **when to enter**. Eight closed direction. The
2,822,400-cell MOMO_CHASE sweep closed one entry form at p=0.8227. `CrabelStretch`
(`setups.py:557`) already implements opening-range breakout **as an entry**, with a
volatility-scaled trigger.

This tests something none of them did: a **day-level veto**. Not "enter or not, right
now", asked on every bar, but "is today tradeable at all", asked once, early, and applied
to the *existing frozen family* rather than to a new rule.

That distinction matters because a veto is not a fourteenth setup. It adds no parameters
to any setup, changes no entry logic, and cannot invent a trade. It can only remove
trades the frozen thirteen already took. So its effect is measurable as a contrast on a
record that already exists.

`research/remaining_testable_inventory.md` §G1 is the nearest catalogued item, and it is
a different question: pre-open **scalars** regressed against *the journal's* per-trade
P&L, n=266–333, MDE ±$27/trade, flagged there as underpowered. This runs on ~2,650
sessions of mechanical trades instead.

It may still null.

## 1. THE LOOKAHEAD, NAMED FIRST

The trader's rule as spoken — *"it didn't break out of that range during trading hours"* —
is **not decidable until 16:00**. A filter that uses it to suppress a 10:00 entry is
reading the future. This project lost two entire result sets to a one-minute version of
exactly that defect, which supplied 88–95.7% of a measured edge.

So the hypothesis splits, and only one half can ever be traded:

| | definition | decidable at | may gate a trade? |
|---|---|---|---|
| **V-DECIDE** | no 10-min RTH bar has CLOSED outside the reference band by `T` | `T` | **yes** |
| **V-POST** | no 10-min RTH bar closed outside the band all session | 16:00 | **never** |

Both are measured. V-POST is reported for attribution only — to say how often the early
call was right — and is **prohibited** from appearing in any statement about tradeable
performance. If the write-up ever quotes a V-POST return as an achievable one, that is the
failure this section exists to prevent.

## 2. Close-based, and why

Expansion requires a 10-minute bar to **close** outside the band. A touch does not count.

Chosen before seeing any result, for a stated reason: on 2026-09-17 QQQ printed a spike to
~718 on the session's largest volume bar and closed back inside a ~715.0–716.7 band. A
touch rule calls that expansion; a close rule calls it a failed breakout. The repo already
distinguishes these elsewhere — `PDH_PDL_FailedBreak` exists — and a rule that a single
tick can flip is not a rule.

This also fixes the sign of the error. A close rule can only be late, never fooled. A touch
rule can be fooled by one print, and being fooled is the expensive direction when the
instrument is a naked long option.

## 3. The primary specification, frozen. ONE of them.

Only the row marked PRIMARY is tested for the §6 decision. Everything in §7 is secondary
and may never be promoted.

| element | value | where it comes from |
|---|---|---|
| symbol | QQQ | the only symbol with 10.6y of cached NBBO |
| bar interval | **10 minutes**, aligned to 09:30 | the timeframe he actually reads |
| premarket window | 04:00–09:29 ET | `levels_test.get_ext` already fetches and caches this |
| opening range | first 10-min RTH bar, 09:30–09:39 | |
| reference band | `RH = max(PMH, ORH)`, `RL = min(PML, ORL)` | the union, see below |
| buffer | **0.05% of price** | `levels_test.BUFFER`, his own `VWAP_Reclaim` convention |
| decision time `T` | **11:00 ET** | `CrabelStretch`'s window already ends 11:00 |
| expansion | a 10-min close `> RH*(1+buffer)` or `< RL*(1-buffer)` | |
| **V-DECIDE fires** | no expansion in any 10-min bar closing in `(09:39, 11:00]` | PRIMARY |

**The band is the UNION of premarket and opening range, not either alone.** Expansion then
requires a genuinely new extreme for the day rather than merely exceeding whichever of the
two happened to be narrower. This is the conservative choice: it fires the veto less often,
so it is harder for the veto to look good by accident.

**Every number above is either taken from an existing file in this repo or is the
timeframe he reads.** None was chosen by trying alternatives. That is the only defence
against the garden of forking paths available here, because the free parameters (`T`,
interval, buffer, which band) span a space large enough to manufacture any result, and
this project has already spent 2.8M cells proving that.

## 4. The outcome, and the one subtlety that decides the design

Source: `live_lab_data/sharewf_trades.json`, produced by
`python -m trade_analysis.live_lab.sharewf`. Each row carries `day`, `setup`,
`direction`, `entry_ts`, `pnl` at $10,000 notional, real NBBO both sides, next-bar fills.

**Only trades with `entry_ts > T` may enter the contrast.** A veto decided at 11:00 cannot
suppress a 09:36 entry, and crediting it with that P&L would measure a filter that cannot
be implemented. This single restriction is why the primary metric is:

    D(day) = sum of pnl over all 13 setups for trades entered after 11:00 ET

and the contrast is `mean D | V-DECIDE` versus `mean D | not V-DECIDE`.

Days with no post-`T` trades are **retained with D=0**, not dropped. Dropping them would
condition the sample on the family having fired, which is itself correlated with
expansion — the selection effect that would make any veto look good.

## 5. Power, stated before running

~2,650 QQQ sessions, 2016-01-04 → 2026-08-27. The veto's base rate is unknown before
running; at a plausible 30–50% the split is roughly 800–1,300 versus 1,350–1,850 days.

Inference is a **day-clustered stationary bootstrap** on `D`, reusing `sharewf.boot`
(3,000 reps, seed 20260828). Days are the unit; trades within a day are not independent,
and 13 setups on one symbol collapse into one cluster.

The honest statement of what this cannot see: with per-day P&L dispersion as fat as §6.3
expects, a contrast smaller than roughly 0.15σ will not be detectable at n≈2,650, and
that is a limit of the design, **not evidence of absence.**

## 6. Decision rule, committed now

**6.1 Primary.** The veto is real if the day-clustered bootstrap 95% CI on
`mean D | V-DECIDE − mean D | not V-DECIDE` **excludes zero and is negative** — veto-days
earn less.

**6.2 The filtered strategy must actually be better, not merely different.** A negative
contrast is necessary and not sufficient. Also required:

- total P&L of the family with post-`T` entries suppressed on veto-days must exceed the
  unfiltered total, and
- the improvement must survive removing the single best day, and
- the sign must hold in at least 8 of the 11 calendar years.

**6.3 THE TAIL CHECK, AND IT CAN KILL THIS ON ITS OWN.** IntradayMomentumBoundary carries
**73.9% of its entire P&L in the top 1% of trades**; exclude them and p=0.1764. Every
apparent edge in this project so far has lived in its tail.

So the filter is only useful if it removes contained days while **keeping the expansion
days that carry the edge.** Report, before any conclusion:

- the share of total family P&L that falls on veto-days;
- the share of the top 1% of trades by P&L that fall on veto-days;
- what the filtered total becomes if the veto removes even one of the top ten days.

**If the veto removes any material part of the top-1% P&L, it is harmful regardless of
what §6.1 says**, and the finding is that the trader's intuition identifies low-range days
correctly and that low-range days are not where the losses are.

**6.4 Failure is a result.** A powered null here — with the lookahead split honoured, the
cost already in the fills, and CIs on the contrast — is a stronger artifact than a filter
that works. It would say something specific and useful: that the discretionary no-trade
decision, which feels like the most reliable judgement in the book, does not pay.

## 7. Secondary, reported and never promoted

`T` ∈ {10:30, 12:00} · interval ∈ {5m, 15m} · band ∈ {premarket only, opening range only}
· buffer ∈ {0, 0.10%}. **Holm–Bonferroni across the whole secondary family**, and the
primary result stands whatever these do. A secondary cell beating the primary is evidence
about the size of the search space, not about the strategy.

## 8. Prohibited

- Quoting any V-POST number as achievable performance.
- Including trades entered at or before `T` in the contrast.
- Dropping days on which the family did not trade.
- Changing `T`, the interval, the buffer or the band after seeing §6.1 and calling it the
  same experiment.
- Reporting the best of §7 as if it had been the primary.
- Applying this to the live lab. `live_lab_data/` is frozen; this is a backtest over
  `sharewf_trades.json` and touches no config hash. If it survives §6, adopting it live is
  a separate, later decision with its own freeze.

## 9. Scope

QQQ shares only. This says nothing about the options arm: the 0DTE archive is closed and
an ATM 0DTE needs roughly +5 bp of underlying move to clear spread and theta
(`shares_runner.py`), so the veto's relevance there is arithmetically plausible and
**untested**. Do not extend the conclusion to naked calls and puts without its own
experiment.
