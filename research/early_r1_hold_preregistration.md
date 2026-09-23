# Pre-registration — does removing the close-back-inside stop help?

**Frozen 2026-09-23, before the sweep is run.** Executed by
`trade_analysis/live_lab/early_r1_hold.py`. QQQ, 2016-01-04 → 2026-09-19, live feed
required (the stored sweep has no session close price, so this cannot be done offline).

## 0. What prompted this, in one line

[`after_break_results.md`](after_break_results.md) §2: the six-line rule stops out 69.7%
of early R1 breaks on a close back inside R1, and **54.3% of those stopped-out sessions
still close beyond R2 later the same day.**

## 1. The three arms

All long. All on QQQ. R side only — §4 of the results found the S side dead.
"Early" = R1's close-break is the 09:30 or 09:40 bar (the first two 10-minute bars).
Entry is always the **open of the bar AFTER** the break bar, because a close is not
knowable until its bar ends.

| arm | sessions | entry | exit |
|---|---|---|---|
| **A** incumbent | R1 live, breaks early | next bar's open | close back inside R1, else session close |
| **B** proposal | **the same sessions as A** | **the same entry as A** | **session close only. No stop.** |
| **C** drift benchmark | R1 live, did NOT break by 09:40 | 09:50 bar's open | session close |

**All three are UNCAPPED.** The profit cap is settled three times over and is not on trial
here; adding it would confound two exits in one experiment.

### Arm C is the whole point

QQQ went from about $110 to about $740 across this sample. **Any rule that buys and holds
to the close earns that drift.** Without C, "B returns +X bp" is unreadable — it could be
the setup or it could be the decade. C is the same long, the same holding window, on the
days the setup did *not* fire.

**Matched sub-comparison:** arm A/B sessions whose break was at 09:40 enter at the 09:50
open, which is exactly arm C's entry bar. B restricted to those sessions is reported
separately, as the comparison with no entry-time difference at all.

## 2. Exit price, stated precisely

`six_lines.py`'s docstring says the trade exits at **15:55**. Its code exits at
`ten[-1]["close"]` — the 15:50 bar's close, i.e. **the 16:00 session close**. This
experiment matches the **code**, so every arm exits at the session close, and the
docstring elsewhere is imprecise. Recorded because a ten-minute ambiguity in an exit
price is exactly the kind of thing that later turns out to have been the whole result.

## 3. The decision rule, fixed now

**The proposal survives only if BOTH hold:**

1. **(B − C) > 5 bp**, the cost floor an ATM 0DTE needs for spread and theta; and
2. the 95% CI on (B − C) **excludes zero**.

**(B − A)** is reported regardless — it is the price of the stop, and it is the number
§2 predicts is large and positive.

**Split-half requirement:** (B − C) must carry the **same sign** in both halves,
2016-01→2020-12 and 2021-01→2026-08. A sign flip kills it regardless of the pooled p.

Falsified if (B − C) ≤ 5 bp, or its CI spans zero, or the halves disagree. Written up
either way, with equal care, as a null if that is what it is.

## 4. Inference

One trade per session, so sessions are already the unit of observation and no further
clustering is needed. Session bootstrap, **3,000 reps, seed 20260923**, resampling
sessions with replacement. (B − C) uses a two-group bootstrap resampling each group
independently. Reported as mean, 95% CI, and a two-sided p.

## 5. What would make this wrong — checked before the numbers are believed

| risk | handling |
|---|---|
| decade-long uptrend | arm C, plus the split-half requirement |
| **in-sample** | see §6. Not handled. Stated. |
| survivorship | none — every session with a live R1 enters an arm |
| lookahead on entry | entry is the next bar's OPEN; pinned by a test |
| lookahead on exit | exit is the session close, used by all arms equally |
| thin sessions | `< 300` 1-minute RTH bars dropped, same rule as `six_lines.py` |

## 6. THIS RESULT WILL BE IN-SAMPLE, and that is not fixable here

The §1/§2 findings that motivated this experiment were derived from **these same 2,656
QQQ sessions.** Re-running them with a different exit does not make the answer
out-of-sample; it re-uses the data that generated the hypothesis. Whatever number comes
back is therefore an **upper bound** on what to expect live.

The split-half requirement in §3 is a weak guard, not a fix. **The only genuine
out-of-sample test is SPY, which has never been run.** If QQQ passes, SPY must be run
before any of this is traded, and it must pass there too. If SPY is not run, the result
stays provisional permanently.

## 7. What this cannot conclude

- **Nothing about options.** No option price appears. The arms are underlying moves; the
  5 bp floor is a proxy for what an ATM 0DTE costs to get in and out of.
- **Nothing about the downside.** S side is excluded, per the null already measured.
- **Nothing about a falling market.** 2016–2026 QQQ is one long uptrend with short
  interruptions. A positive result here is conditional on that regime.
- A pass does **not** authorise trading this. It authorises running SPY.
