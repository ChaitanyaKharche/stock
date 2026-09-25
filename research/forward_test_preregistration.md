# Pre-registration — Prospective forward test of the full pipeline

**Status: FROZEN 2026-08-28.** Machine-readable copy: `live_lab_data/FREEZE.json`.

Accepted config hashes — the **only** filter ever applied to trades:

| hash | meaning |
|---|---|
| `1f7247d7839d9950` | current; hash covers **definitions only** |
| `1d7c5a9fc71d67fa` | day 1; **definitionally identical** (see §6) |

---

## 1. What is frozen

The 13 setups (3 user-supplied + `MOMO_CHASE` + 9 researched, Appendix A of
`live_lab_specification.md`), their entries, exits, stops, targets, time windows and
per-day caps; the ATM / ATM-1 / ATM+1 selection rule; ATM as the **primary arm**; the
$0.0404/contract/side fee; and the family size **m = 13**.

**Long premium only. No arm can express a short-option position.**

Any change to any of these forks the affected setup into a new `setup_id` with an empty
history. Old trades are never retro-fitted to a new definition.

## 2. What is measured — the whole pipeline, not just the strategy

```
signal -> option selection -> hypothetical fill -> exit -> underlying twin -> option P&L
```

Every stage is recorded, so failure can be **localised** rather than guessed at. The
question this exists to answer:

> **Did the setup correctly predict the underlying, and if so, did ATM/ATM+-1 convert that
> into money?**

Those are independent failures with opposite remedies:

| observed | conclusion | remedy |
|---|---|---|
| direction wrong | the setup has no edge | none — no strike choice rescues a wrong signal |
| direction right, money lost | the edge exists, the 0DTE structure eats it | the instrument: strike, hold, expiry |

**No prior study in this project could distinguish these.** `attribution.py` reports the
funnel, the conversion curve (how far the underlying must travel before an ATM 0DTE option
pays for its own spread and theta), and the arm comparison **conditional on direction being
right** — which is the strike question stated properly.

## 3. Checkpoints — fixed now

| checkpoint | scope | what is permitted |
|---|---|---|
| **50 trades** | total | **DIAGNOSTIC ONLY.** Pipeline health, data quality, outages. **No effect claims.** |
| **100 trades** | total | **DIAGNOSTIC ONLY.** Same. |
| **200 trades** | per setup | **First serious evaluation.** The promotion bar goes live for that setup. |
| **377 trades** | per setup | Powered for a **+10% mean return** at Holm m=13, 80% power |

377 = `((2.89 + 0.84) x 52/10)^2`, from the measured 52% per-trade return SD.

Reading P&L before the 200-trade bar is not forbidden — the dashboard shows it, and
pretending otherwise would be theatre — but **no conclusion may be drawn from it**, and
every row prints `INSUFFICIENT (n=k/200)` until the bar is met.

## 4. Promotion bar — all four, or the setup is not promoted

1. **n >= 200** completed ATM trades
2. clears **Holm** within the family of 13
3. **|effect| > its own achieved MDE** (the underpowered rule, not relaxed)
4. **first half and second half agree in sign**

Anything else prints `INSUFFICIENT`, `no effect (Holm)`, `underpowered for its own effect`,
or `fails the split-half check`. Ranking is by **expectancy with CI and n**, never raw P&L,
because setups fire at wildly different rates.

## 5. THE CLOCK NEVER RESETS

**The trade counter is cumulative from 2026-08-28 across the accepted hashes. There is no
mechanism to exclude a session, a setup, or a date, and none may be added.**

A setup performing badly is a **result**, not a reason to restart its clock. Specifically
forbidden:

- dropping a bad day as "unrepresentative"
- restarting a setup's count after a drawdown
- excluding a regime, a symbol, or a time of day post hoc
- adding a setup mid-experiment without a dated amendment (and the family never shrinks)
- **loosening any parameter to make a setup fire more often** — the two `DEAD` setups
  (`VWAP_2sigma_Fade`, `ThreeBarPlay`) are dead because of thresholds *we* invented, and
  they stay dead

The one legitimate reset is a **definition change**, which forks a new `setup_id` starting
at zero and leaves the old history intact and reportable.

## 6. Day 1 is INCLUDED, and that is the point

**2026-08-28: 27 ATM trades, -$1,201.18.** It counts.

It was collected prospectively, under these exact definitions, with a feed verified
real-time — **zero stale-quote outages across 14,853 API calls**, where a delayed feed
would have flagged every tick.

The only discrepancy is bookkeeping: `git_sha` and `spec_version` were mistakenly inside
the config hash, so a commit touching no setup would have forked every history and reset
the clock — the exact failure §5 forbids. That is fixed; the hash now covers definitions
only. The setups, arms, fees, family and caps are byte-identical between the two hashes.

**Excluding a losing day collected under the frozen rules would be the first instance of
the behaviour this document exists to prevent.** It stays in.

## 7. Known limitations, stated before results

- **Hypothetical fills.** Entry at the NBBO ask, exit at the bid. Historical work in this
  project measured NBBO simulation as **+$6.85/trade optimistic** versus the trader's
  realised cash, composition unknown. Apply that haircut mentally to every P&L here.
- **~3 independent bets, not 13.** The ORB/Crabel/PDH family are all early-session
  directional breakouts; the fades correlate; the continuation setups correlate. Correlated
  positives will look like confirmation and are not. Pairwise overlap is reported.
- **Four setups cannot reach a verdict.** `Gap_Fade` and `EMA_9_20_Pullback` are `SLOW`
  (30+ months); `VWAP_2sigma_Fade` and `ThreeBarPlay` are `DEAD`. All four stay in the
  family and in the Holm correction.
- **`MOMO_CHASE` is a hypothesis, expected to fail.** Its binding gate is the MACD/DMI/ADX
  stack, not the momentum threshold — so a failure is mostly a verdict on indicators that
  nine prior tests already found empty.
- **Paper only.** No order is ever placed. There is no broker client in the package.

## 8. Stated priors, recorded so nothing can be claimed as expected afterwards

I expect **most or all setups to fail**. Nine pre-registered tests in this project found no
predictive content, one of them well-powered (1,542 sessions x 2 symbols, 0/10 advancing).
An independent 2-year QQQ study puts ORB expectancy near zero.

The outcome I consider most likely is **Family B positive, Family A null** for at least one
setup — direction faintly real, the 0DTE structure eating it. Day 1 pointed the other way
(direction 22.2% right, but 83.3% conversion when it was), which is one session and worth
nothing.

**If everything fails, that is a result and the programme ends with an answer instead of a
suspicion.** That is worth more than another year of ambiguity.

---

## Amendment 2026-09-25 — the trader's six lines enter the family

Appended, never substituted. Written before either new setup has traded a live session.

**What is added.** Two setups, appended to the end of `ALL_SETUPS`:

| setup | rule |
|---|---|
| `Six_Lines` | the trader's own level set exactly as specified 2026-09-20 (`six_lines.py`): yesterday's premarket high/low, yesterday's market-hours high/low, today's premarket high/low. First 10-minute close beyond a line not already passed at the open; one trade per session; exits +20 bp \| 10m close back inside the line \| 15:55 |
| `Six_Lines_NoCap` | the same entry with the +20 bp cap removed |

**Why now.** It is the only strategy in this programme that is the trader's rather than a
published one, and the live lab could not run it: both runners were RTH-only and never read
a premarket bar. `levels_live.py` now fetches yesterday's extended session at warmup and
today's premarket at 09:31, and builds the lines with `six_lines.build_six` itself.

**Why the uncapped twin.** The cap has subtracted value on every dataset it was measured on,
including −0.72 bp on this exact level set (`six_lines_results.md` §3). Running both is the
only way to see that on live fills rather than in a backtest.

**What does not change.**
- The 13 original definitions are **byte-identical** to `1f7247d7839d9950`; this is tested
  in `six_lines_setup_test`.
- Their counters continue across the hash change. The new hash, `ccd00ac71f8243e9`, is added
  to `FREEZE.json` beside the old ones, exactly as the day-1 hash was.
- The two new setups start at zero on 2026-09-25.

**Family.** m = 13 → **15**. It only grows, so the Holm bar rises slightly for every setup
and never falls.

**Stated prior, before any live trade.** The six-line entry is a documented null on QQQ
underlying: 2,438 trades, +0.54 bp uncapped, CI [−1.70, +2.87]. An ATM 0DTE needs ~5 bp.
The expected result is a null here too. The one outcome that would be new is the uncapped
twin clearing the cost floor on live fills when the backtest said it cannot.

**Backfill, and why it is not evidence.** The new config is also replayed over
2026-08-28 → 2026-09-24 through the real runner code, with historical bars and quotes
(`live_lab_data/backfill/`).
- It is **never** counted toward any checkpoint or promotion bar.
- It is computed after the fact, so the decision-before-price guarantee in §1 does not
  hold for it, however faithfully the code path is reused.
- It exists to show what the new config would have done, not to shorten the clock.
