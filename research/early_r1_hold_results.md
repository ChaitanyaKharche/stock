# Results — removing the stop is worth nothing, and my reason for expecting otherwise was wrong

**Run 2026-09-23 on the lab machine.** `trade_analysis/live_lab/early_r1_hold.py`, QQQ,
2016-01-04 → 2026-09-19. **2,795 days tried, 1,489 usable, 138 thin. 317 early-break
sessions, 1,172 benchmark sessions.** Executes
[`early_r1_hold_preregistration.md`](early_r1_hold_preregistration.md).

## 1. The verdict: FALSIFIED on all three pre-registered criteria

| arm | mean | 95% CI | p |
|---|---|---|---|
| **A** incumbent (stop on close back inside R1) | −0.14 bp | [−6.84, +6.98] | 0.943 |
| **B** proposal (hold to the close, no stop) | +1.69 bp | [−10.81, +14.65] | 0.827 |
| **C** benchmark (no early break, buy 09:50, hold) | **+4.25 bp** | [−2.10, +10.43] | 0.184 |

| comparison | result | |
|---|---|---|
| **B − A** what the stop costs | **+1.84 bp** [−13.02, +16.81] p=0.829 | not the large number §2 predicted |
| **B − C** what the setup adds | **−2.56 bp** [−17.01, +11.76] p=0.709 | **wrong sign** |
| B − C, matched entry bar | −2.96 bp [−24.58, +18.45] p=0.804 | same |

| pre-registered gate | outcome |
|---|---|
| (B − C) > 5 bp | **FAIL** (−2.56) |
| 95% CI excludes zero | **FAIL** ([−17.01, +11.76]) |
| split-half signs agree | **FAIL** (+4.87 then −9.97) |

The split-half flip is textbook noise: 2016–2020 says +4.87 bp, 2021–2026 says −9.97 bp,
both with intervals several times wider than the estimate.

**And the benchmark beat both treatment arms.** Days where R1 breaks early are, on the
point estimate, slightly *worse* for a long held to the close than days where it does not
break early. Not significant — every interval here spans zero — but it is the opposite of
the story the experiment was built on.

## 2. CORRECTION to `after_break_results.md` §2 — a true number, a false inference

§2 of that document reported, correctly: of the 219 early R1 breaks stopped out on a
close back inside R1, **119 (54.3%) still closed beyond R2 later the same day.** That
number is measured and stands.

**The inference drawn from it does not.** §2 said the stop "is throwing away more than
half of its winners" and called removing it "the single highest-value thing left to
measure". Removing it is worth **+1.84 bp, CI [−13.02, +16.81]** — indistinguishable from
zero, and two-thirds of a cent on a $740 stock.

**Why the inference was wrong:** "R2's close-break happened later that day" and "the
trade would have been profitable at 16:00" are different statements. R2 can break at
11:20 and price can be back below R1 by the close. The held trade exits at the **session
close**, not at the moment R2 breaks. I read a statement about the *path* as a statement
about the *endpoint*.

That is this project's standard bug class — no crash, no error, a true figure supporting
a conclusion it does not support. It was caught only because the claim was turned into a
pre-registered experiment with a decision rule fixed in advance. §2 of
`after_break_results.md` is corrected in place.

**A second thing worth stating so the two documents do not look contradictory:**
`after_break_results.md` §2 quotes a **median** end move of −16.7 bp while arm A here has
a **mean** of −0.14 bp. Both are right. The distribution is right-skewed — a few large
winners drag the mean up off a clearly negative median — exactly as
`six_lines_results.md` found (mean +0.54, median −13.94).

## 3. What survives, and what it means

**`after_break_results.md` §1 survives untouched.** An early R1 break really does reach R2
more often than a late one: 70.7% vs 59.3%, and +11.3 pp with p=0.002 after both groups
are given the same 60-minute window. That is a real fact about how price travels.

**It does not convert into money.** The path is more likely to get there; the 16:00 price
is not. That is the fourth time this project has landed on the same sentence from a
different direction:

| arm | where |
|---|---|
| `_calculate_momentum_score` was a magnitude forced to emit a direction | `vrp_preregistration.md` §0 |
| MFE +34.26 bp against terminal drift indistinguishable from zero | `breakout_options_results.md` §4 |
| 6-line breakout, 2,438 trades, median MFE far above terminal move | `six_lines_results.md` |
| early R1 reaches R2 more often, terminal return unchanged | **here** |

**These levels measure how far price moves, not which way it ends up.** Four independent
measurements now say it. It is the most robust *descriptive* finding in the project, and
it is the reason a directional long-premium expression keeps failing: you can be right
about the travel and still lose, because you pay the variance risk premium for the
convexity and collect nothing for the direction.

## 4. Round numbers, restated

Unchanged and still a clean null — R1/S1 show no affinity for multiples of 1, 5, 10 or 25,
in either price era. Nothing to act on there.

## 5. What is NOT concluded

- **Not that the stop is good.** B − A is +1.84 bp with an interval from −13 to +17. The
  stop is not shown to help either; it is shown not to matter measurably.
- **Nothing about SPY.** Never run, and now there is nothing left to confirm there.
- **Nothing about options or the cap.** No option price appears anywhere in this run.
- **Nothing about a falling market.** 2016–2026 QQQ is one long uptrend.

## 6. Status

**Closed.** The six-line early-break family is now null on entry timing, on the level set,
on the profit cap, and on the stop. The pre-registration's own §6 said a pass would only
authorise running SPY; it did not pass, so SPY is not warranted.
