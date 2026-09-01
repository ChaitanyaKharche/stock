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
