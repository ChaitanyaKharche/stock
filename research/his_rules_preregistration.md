---
task_id: R-2026-10-26-01
trace_id: https://claude.ai/code/session_01TqhCAiwZiZ56SHrYyFXGVT
parent: research/ARCHITECTURE_AND_BUSINESS_REVIEW_2026-10-23.md
role: quantitative-researcher
repo_sha: 1c38610
inputs:
  - "the owner's description of his method, 2026-10-26, quoted verbatim in §1"
  - research/six_lines_results.md@1c38610
  - research/six_lines_preregistration.md@1c38610
  - trade_analysis/live_lab/six_lines.py@1c38610
decision_rule: "see §5; fixed before any data is read"
outputs:
  - research/his_rules_results.md (not yet written)
evidence: []
assumptions:
  - "the thresholds in §3 are my translation of 'sizably big' and 'with volume'; the owner confirms or corrects them before the run"
open_questions:
  - "Q4 was not answered: how a LOSER is exited. §3 uses his 2026-09-20 rule (close back inside the line) until he says otherwise"
  - "the 12:30-13:00 re-entry: same break rule on lines not yet broken, or any break? §3 assumes the former"
errors: []
status: draft, awaiting owner confirmation
verified_by: null
budget: {cap: "one run per arm on the laptop; 2024 SPY block NOT touched"}
---

# Pre-registration — the trader's rules as he described them on 2026-10-26

**Status: DRAFT. Nothing runs until the owner confirms §3 line by line.**

## 0. Why this exists

`six_lines_results.md` tested the six lines with a mechanical entry (first break of the
session, any size, any volume, any time) and found **+0.54 bp a trade uncapped, about 4
cents a share on QQQ, over 2,438 trades and ten years**. The owner's answer: that is not
his method. He takes a *selected* break, with size and volume behind it, inside a time
window, and he scales out of winners. This file writes that selection down so it can be
tested the same way, and so that if it is also null, the null is about what he does.

## 1. His description, verbatim

> yes with volume too, a sizably big candle that breaks the 1st R1 or S1 line, if I can
> check for a pullback from R2/S2 then I wait for the pullback IF for some reason i aint
> able to enter around R1/S1 or if I didnt feel confident enough entering at/near R1/S1 &
> obviously its a lot more nuanced than that too
>
> i only enter if momentum looks food until 10.45am or i skip and come back around
> 12.30-1pm cause then morning finna be ranged and inconclusive
>
> I keep my target at 25-50%, but if momentum looks too strong then I plan my exit, so for
> example lets say I bought 5 lot of QQQ CE ($x strike price) at $1 and if momentum looking
> strong then ill exit 2 lots at $1.5, other 2 at $1.75 and 1 at $2
>
> I used to do my strategy ATM options or somtimes +1 or -1 OTM
>
> It also depends on some days like any report or new to be released by the FED OR maybe
> some major QQQ news

## 2. What cannot be coded, and is therefore excluded

| his words | why it is out |
|---|---|
| "didn't feel confident enough" | no rule; this is the discretionary residue the test measures *around* |
| "momentum looks good / too strong" | replaced by the §3 bar-size and volume thresholds; if he can state it better, say so |
| "FED report, major QQQ news" | an event-day filter is possible (FOMC dates are public) but was not stated as a rule. Recorded as a planned **secondary** split, not part of the primary verdict |
| "a lot more nuanced than that" | if the mechanical version is null and the nuance is the edge, the only test of the nuance is his own journal and the live lab |

## 3. The mechanical translation (owner to confirm each row)

Lines: the six lines exactly as `six_lines.py` builds them. Bars: 10-minute RTH bars from
1-minute data (`resample_10m`), as in the previous test. One position at a time.

| # | rule | translation |
|---|---|---|
| E1 | "a sizably big candle that breaks R1 or S1" | a 10-min bar **closes** beyond R1 (long) or S1 (short), AND its high–low range ≥ **1.5×** the median range of the previous 20 RTH 10-min bars |
| E2 | "with volume too" | AND that bar's volume ≥ **1.5×** the median volume of the same clock-time bar over the previous 20 sessions |
| E3 | entry | next bar's open (as before), long call on an R1 break, long put on an S1 break |
| E4 | "pullback from R2/S2 if I missed R1" | **Arm B only:** if no E1/E2 entry fired and price reaches R2 (S2) within 30 min of the first R1 (S1) close-break, enter on the first bar that touches back to within 5 bp of R1 (S1) and closes on the far side of it. Window still applies |
| T1 | "until 10:45" | entry bars with open time 09:40–10:45 ET only (09:30 bar excluded: its open is the gap) |
| T2 | "come back around 12:30–1pm" | second window 12:30–13:00 ET, same E1/E2 rule, only on a line **not yet broken** that day. One trade per window, max two per day |
| X1 | "target 25–50%" | **Exit arm 1:** sell all at +50% on the option mid, or +25% if the bar after entry is already a loser (his range, two fixed points; neither is tuned) |
| X2 | "exit 2 at 1.5, 2 at 1.75, 1 at 2" | **Exit arm 2 (primary):** 5 contracts; sell 2 at +50%, 2 at +75%, 1 at +100% on the option bid |
| X3 | loser (**Q4 unanswered**) | a 10-min bar closes back inside the broken line → exit all remaining at the bid; else hard exit 15:55 |
| S1 | "ATM, sometimes ±1 OTM" | **ATM primary**; +1 OTM and −1 OTM reported as sensitivity, not as separate verdicts |
| Z1 | size | 5 contracts per entry; fills at the **ask** in, **bid** out; $0.65 a contract a side fees (the lab's number) |

Thresholds 1.5× (E1, E2), 30 min and 5 bp (E4) are set here, once, and are not swept.
If the owner wants different numbers he says so **now**; after the run they are frozen.

## 4. Data

| series | what it answers | source | period |
|---|---|---|---|
| **SPY 0DTE, real option quotes** (primary) | dollars, with real spreads, his exits in option-price terms | `option_quote_1m_0dte/`, laptop only | **2020-01-03 to 2023-12-29**. The 2024 block is held out and is NOT touched by this study |
| QQQ 1-min underlying (secondary) | bp move from entry to exit on the thing he actually trades, no options | the same feed `six_lines.py` used | 2016-01-04 to 2026-09-19, so the result is comparable with the 4-cent baseline |

QQQ **options** cannot be tested: no archive exists (`CLAUDE.md`). SPY is the nearest
instrument with real 0DTE quotes. Option-price exits (X1, X2) are only meaningful on SPY;
on QQQ underlying they are translated as the bp move an ATM delta-0.5 option would need,
as in `six_lines_results.md`.

## 5. Decision rule, fixed now

**Primary verdict:** SPY 0DTE 2020–2023, Arm A entries (E1+E2+E3, T1+T2), Exit arm 2,
ATM, 5 contracts. Statistic: **mean net dollars per trade**, after spread and fees.
Day-clustered bootstrap, 3,000 reps, seed 20260828, 95% CI.

- **PASS:** lower bound of the 95% CI > 0 **and** mean per trade > $0 after costs.
- **FAIL:** otherwise.

**Secondary, reported, no verdict attached:** Arm B; Exit arm 1; ±1 OTM; the QQQ
underlying bp series; FOMC-day vs other days; the same statistic on the 09:40–10:45 window
alone versus the 12:30–13:00 window alone. Seven secondaries, Holm-corrected if any is
quoted as significant.

**Comparison to baseline:** the QQQ secondary is reported next to the +0.54 bp of
`six_lines_results.md` so the question "is the selected break better than the first
break" gets a number with a CI, not a yes/no.

**Minimum sample:** if the primary produces fewer than **150 trades**, the result is
reported as *underpowered* and no verdict is attached either way.

## 6. What a PASS would and would not mean

A PASS says: on SPY, 2020–2023, this mechanical reading of his rules made money after
costs. It does not say QQQ, it does not say 2026, and it does not say *he* makes money,
because "confident enough" and "momentum looks strong" are not in it. If it passes, the
next step is written in advance: the same rules go into the live lab as a dated amendment,
and the 2024 SPY block is spent on exactly one confirmatory run.

A FAIL says: the parts of his method he could write down do not carry the record. Whatever
does is in the parts he could not write down, and the only instrument for those is his
journal and the forward test.

## 7. Where it runs

The SPY archive and the ThetaData feed are on the laptop. The script will be
`trade_analysis/live_lab/his_rules.py`, importing `six_lines.build_six`,
`orb_veto.resample_10m` and the cost model from the existing code, new file, no edit to
proven code. It is written **after** the owner confirms §3, and its tests are written
before it, each shown to fail on a deliberately wrong rule.
