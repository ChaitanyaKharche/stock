---
task_id: R-2026-10-26-02
trace_id: https://claude.ai/code/session_01TqhCAiwZiZ56SHrYyFXGVT
parent: R-2026-10-26-01
role: experiment-runner
repo_sha: 511f67c
inputs:
  - research/his_rules_preregistration.md@fe7b072
  - trade_analysis/live_lab/his_rules.py@4493d48
  - research/his_rules_SPY_options.json@511f67c  (the run's own output, unedited)
decision_rule: "PRIMARY: SPY 0DTE 2020-2023, arm A, ATM, 2/2/1 ladder, 5 contracts; mean net $ per trade; PASS if 95% CI lower bound > 0 and mean > 0; FAIL otherwise; fewer than 150 trades = UNDERPOWERED, no verdict"
outputs:
  - research/his_rules_results.md
evidence:
  - "all numbers: research/his_rules_SPY_options.json, report block and trades list"
assumptions:
  - "the first run on 2026-10-26 (before the diagnostics commit) printed 0 trades over 700+ days and its JSON was overwritten by the second run; the only code change between them was logging. The likeliest cause is the stock feed returning nothing during the first run (every session thin). Unverified: the first JSON is gone"
open_questions:
  - "the stated filters fire once per 12 sessions; the owner trades 2.8 times a day. Which is his method?"
errors: []
status: done
verified_by: null
budget: {used: "one SPY run on the laptop, ~1 h; 2024 block untouched", cap: "same"}
---

# Results — his rules on SPY 0DTE, 2020–2023: no verdict, and every version lost money

**Run on the laptop 2026-10-26**, `python -m trade_analysis.live_lab.his_rules --symbol SPY
--options --start 2020-01-02 --end 2023-12-29`. 1,042 weekdays tried, 42 thin (holidays),
963 sessions with lines, 381 sessions outside the option archive (it holds 618 of the
1,000). 178 trades in all.

## 1. The verdict, by the rule fixed in advance

**UNDERPOWERED. 41 priced trades on the primary arm; the pre-registration needs 150.** No
PASS, no FAIL. That is the whole formal result.

## 2. What the 41 trades say anyway (reported, not a verdict)

| arm | n | mean $ / trade | 95% CI | p | win rate |
|---|---|---|---|---|---|
| **primary: A, ATM, ladder 2/2/1** | 41 | **−$70.89** | [−$195, +$59] | 0.28 | 37% |
| A, ATM, all out at +50% | 41 | −$89.17 | [−$199, +$25] | 0.13 | |
| A, one strike OTM, ladder | 41 | −$55.71 | [−$148, +$42] | 0.28 | |
| A, one strike ITM, ladder | 41 | −$63.06 | [−$223, +$102] | 0.47 | |
| B (pullback), ATM, ladder | 58 | −$42.17 | [−$124, +$45] | 0.33 | 31% |

In plain terms: five contracts cost about **$660** (median entry $1.32). The average trade
lost **$71, about 11% of what was put in.** Median trade −$187. Total over the 41:
**−$2,906.** Best three +$882, +$644, +$539; worst three −$805, −$742, −$641. Morning
trades −$2,716 over 31; afternoon −$190 over 10. Fees are $0.40 a trade and are not the
story; the median entry spread was 1.06%, tight.

Every arm, every strike, every exit: negative point estimate. None is far enough from
zero to call, and no amount of re-reading these 41 changes that.

## 3. The underlying says the move is there in the middle and gone on average

Arm A, 80 trades, bp of SPY move to the ladder's exits:

| | value |
|---|---|
| mean | **−0.04 bp** (CI [−6.7, +6.5], p = 0.99) |
| median | +3.39 bp |
| win rate | 54% |
| median best excursion before exit | 28.4 bp |
| exits | target legs 117, stop legs 76, EOD 5 |

Same shape as every earlier measurement (`CLAUDE.md`, "MAGNITUDE, NOT SIGN"): the typical
trade goes 28 bp in your favour at some point, the median closes slightly green, the losers
are big enough that the average is zero, and the option converts zero into −11%.

Against the baseline: the first-break-of-the-day rule on QQQ made **+0.54 bp**; your
selected break on SPY made **−0.04 bp**. The size and volume filters did not find better
breaks. (Different symbol and period; the QQQ run below makes it like for like.)

## 4. The filters are not what he does

The funnel, from the run's own counts:

| step | sessions |
|---|---|
| with six lines | 963 |
| R1 or S1 crossed on a close | 596 (62%) |
| ... with entry inside 09:40–10:45 | 375 (39%) |
| ... **and** bar ≥ 1.5× size **and** ≥ 1.5× volume | **61 (6%)** |

The stated rules fire **once every 12 sessions**. The journal says he trades **2.8 times a
day** (`CLAUDE.md`). So either the thresholds are far stricter than his eye, or most of his
real trades do not meet the criteria he gave. Both can be true. Neither is answered by
loosening the numbers after the fact, which the pre-registration forbids; it is answered by
him looking at ten of his own journal entries next to these 61 signals.

## 5. Can the primary question ever reach 150 trades?

No, not with data that exists. The archive gave 41 priced trades from 618 sessions, one
per 15. The held-out 2024 block (151 sessions) would add about 10. There is no other 0DTE
option archive (`CLAUDE.md`). **The SPY option version of this question cannot be settled
by backtest.** What can be:

1. **QQQ underlying, 2016–2026, same rules** (`--symbol QQQ`, bars already proven to exist
   in September). About 2,600 sessions should give ~160 arm-A trades: enough to say whether
   his selection moves the underlying more than the first-break baseline, with a CI. Run it.
2. **The live lab.** If he wants a real-money answer, the rules in this file can go into
   the forward test as a 16th setup by dated amendment, and collect real 0DTE fills on QQQ
   and SPY from the day it lands. At one signal per 12 sessions per symbol that is roughly
   four trades a month; 150 takes three years. That is the honest size of the question.

## 6. A defect to record

The first invocation on 2026-10-26 reported **0 trades** over 700+ days and was allowed to
finish (its JSON was then overwritten by the second run). The second run, after a commit
that only added logging, found 178. The likely cause is the stock feed returning nothing
during the first run (which the code counts as "thin"), but the evidence is gone. Per the
project rule, a number that changes when nothing changed is a bug detector: the script now
prints the first thin sessions and a funnel, so the next silent zero will say why.
