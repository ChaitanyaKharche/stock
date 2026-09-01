# Results — Setup cross-check

Executed 2026-08-20 per `setup_crosscheck_preregistration.md`. Setup list fixed from
external sources before any number was computed. No definition changed after.

**Funnel:** 443 round trips → **409 evaluated**. 34 excluded as too early in the session for
a 15-minute trailing window. None outcome-related.

## 1. PRIMARY — coverage. Verdict: SATURATED.

| setup | coverage |
|---|---|
| VWAP side | **86.8%** |
| ORB continuation | **64.5%** |
| prior-day level break | 41.1% |
| gap and go | 38.4% |
| VWAP reclaim | 26.2% |
| pullback to 9 EMA | 20.0% |
| RVOL ≥ 1.5 | 10.0% |
| fade / range edge | **1.5%** |
| flag | **0.7%** |
| late-day breakout | **0.0%** |

**At least one setup: 91.4%.** At least half: 14.9%. Matches per trade: 0→35, 1→25, 2→105,
3→94, 4→89, 5→57, 6→3, 7→1.

Pre-registered threshold was **≥90% ⇒ saturated, cross-check structurally uninformative.**
Met. "His trade matched a named setup" is true of nine entries in ten and therefore carries
essentially no information.

**The predicted pattern held, and was committed in advance so it cannot be claimed as a
discovery:** momentum-continuation setups score high (VWAP side 86.8%, ORB continuation
64.5%, gap-and-go 38.4%) and the fade setup scores 1.5%. That is the established chase
result (+0.20σ, 89% of entries) re-measured through a different lens, not new evidence.

**Two zero-coverage findings are genuinely informative about his behaviour:**
- **Late-day breakout: 0.0%.** He never enters at a session extreme after 14:00.
- **Flag: 0.7%.** He essentially never waits for the consolidation bar after a thrust. He
  enters *during* the move, not after it pauses — consistent with sub-second liquidity-taking
  and inconsistent with the retest/pullback story.

## 2. SECONDARY — outcome by setup, Holm across all ten

| setup | n | effect on return | 95% CI | p raw | p Holm |
|---|---|---|---|---|---|
| ORB continuation | 264 | **+9.49%** | [−2.14%, +20.66%] | **0.107** | 0.751 |
| VWAP side | 355 | −4.82% | [−17.20%, +7.05%] | 0.444 | 1.000 |
| VWAP reclaim | 107 | −4.93% | [−17.09%, +8.05%] | 0.458 | 1.000 |
| RVOL | 41 | −5.52% | [−23.52%, +15.11%] | 0.565 | 1.000 |
| pullback 9 EMA | 82 | −3.32% | [−17.01%, +10.83%] | 0.620 | 1.000 |
| prior-day level | 168 | +2.65% | [−9.63%, +14.43%] | 0.671 | 1.000 |
| gap and go | 157 | +2.18% | [−9.89%, +14.19%] | 0.693 | 1.000 |

Flag, fade-edge and late-day breakout were **skipped for cells under 10** — a limitation, but
one caused by him never trading them.

**Setups clearing Holm: 0 of 7.**

## 3. Why the pre-registration mattered here

**ORB continuation came in at +9.49% with a raw p of 0.107.** Search a menu, stop at the
best-looking cell, and that is exactly the number that gets reported as "the ORB continuation
setup works on your trades." It does not survive correction across the ten setups fixed
before looking, and its CI spans zero.

Equally, **VWAP side at 86.8% coverage** would read as strong validation if quoted alone. It
is a tautology: VWAP-side is a momentum condition and he is a momentum chaser, so the
coverage is guaranteed by what he does, not by what works.

## 4. Verdict

Both prongs null. Coverage is saturated, so a setup match carries no information; and no
setup separates his outcomes after correction.

**Power, as pre-stated:** effective n ≈ 334, SE ≈ 2.9pp, and Holm across ten raises the MDE
to roughly **11–12 percentage points** of return. This excludes effects above that and
nothing smaller. A null here is not proof that no setup helps — it is proof that none helps
by more than ~12pp on this sample.

**Seventh consecutive pre-registered null.** Preceded by: strike distance, position size,
within-day ordinal, entry-timing percentile, a 19-feature predictive ceiling, and his own
declared indicator stack.

Sources for the fixed setup list:
[TradingSim](https://www.tradingsim.com/blog/day-trading-setups),
[Bulls on Wall Street](https://www.bullsonwallstreet.com/post/what-is-the-vwap-trading-indicator-and-how-to-use-it-as-a-day-trader),
[Warrior Trading](https://www.warriortrading.com/vwap/).
