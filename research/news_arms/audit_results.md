# news_arms — gate 2 audit: discovery report (carried) and holdout section

Auditor, 2026-11-11, at commit `e59a9ff`. The discovery report passed on the fifth pass (commit
`62d17f3`; record in `preregistration.md` §22–§23.4). This pass audits the "Holdout, read once"
section of `results.md`, its status line and "Next", §24, and `analysis_{shares,options}_holdout.json`.
Every holdout number was recomputed from `live_lab_backfill/news_arms/2023–2026` with the auditor's
own loader. That recomputation is a verification of the single read and selects nothing. The
auditor edited neither `results.md` nor the code.

**In one sentence: the holdout numbers are all correct, and the main conclusion stands: no
difference on calendar-release days in either arm, and the Fed-day loss did not hold up. Two
statements break rules or misdescribe the data, so this section fails until they are corrected.**
They are:
- C2 is labelled "FAIL", which the pre-registration forbids for an underpowered test (13.6).
- The shock-day regime split is called "flat", but one regime shows a loss whose interval excludes zero.

Both fixes are text only; no rerun is needed. Because this file's last line is the holdout lock,
writing FAIL here shuts the lock again. The one read has already happened and must not be repeated.

## Verdict table

| # | Claim | Verdict | Reason |
|---|---|---|---|
| H1 | C2 options "FAIL (pre-declared underpowered)" (results.md:4, 284, 312) | **FAIL** | 13.6 (preregistration.md:306-309): C2 "is reported with its effect size, CI and achieved MDE, **never as PASS/FAIL**, unless the observed holdout effect itself exceeds the MDE". Observed +0.057 σ < MDE 0.45 σ (`observed_exceeds_mde: false`). The verdict word is the one the design ruled out. |
| H2 | "the within-regime split is flat" for holdout shock days (results.md:306-307) | **FAIL** | `q3.unscheduled_surprise.post_hoc_within_regime`: low +0.15 [−0.02, +0.34] (26), **mid −0.31 [−0.49, −0.13]** (41), high +0.57 [−0.77, +2.95] (10). The mid-regime interval excludes zero and the sign flips across regimes. That is not flat. |
| H3 | "the holdout says it was selection" (Fed row, results.md:298) | **QUESTIONABLE** | The holdout interval [−0.22, +0.10] σ rules out the discovery size (−0.299). It still includes the discovery figure without 2022 (−0.156). It cannot tell chance from a 2022-specific effect. Holdout Fed days (29) are also below the 30-session minimum, so the row is reported only (`tested: false`). |
| H4 | C1/C2 reporting completeness | **QUESTIONABLE** | Required but missing: the power figures that 14.8 says are "stated next to each result" (power at a true 0.25 σ about 50% for C1, lower for C2; 80% power needs about 0.31 σ for C1 and 0.36 σ for C2), and the own-σ standardisation required by 14.Q(5) (C1 +0.033 σ, C2 +0.069 σ; descriptive, no conclusion changes). |
| H5 | "nothing read these files before" (results.md:274-275) | **QUESTIONABLE** | True of every analysis path. But the heartbeat consoles showed holdout daily nets (§19, §21, and §23's 2024-10-01), and the replay and fill-in processes wrote and read them. Say "no analysis read these files before; console exposures are logged in §19, §21 and §23". |
| H6 | Windows and data | **PASS** | Independent loader. Shares 2023-01-03 → 2026-08-27: 908 sessions, 162,374 trades, 38 chunks (including 2024/shares/fill-a and the single 2026 chunk). The only missing open weekdays are the 8 early closes, so 0 lost; nothing after 2026-08-27. Options 2023: 12 chunks, 248 sessions, 5,993 ATM trades; missing only 2023-07-03 and 11-24 (early closes); no 2024/2025 options chunk loaded. Live hashes. |
| H7 | σ = discovery σ (15.5); own σ descriptive | **PASS** | $3,198.66 / $1,032.25 used. Own σ $2,301.98 / $861.54. Holdout B0 −$108.64 / −$61.36. |
| H8 | C1: 166 of 908, +0.024 σ [−0.07, +0.11], p 0.597, SE 0.086, MDE 0.24, pass rule FAIL; +$75/day [−$230, +$360]; single-category +0.04; by year +0.07 / +0.04 / −0.01 / −0.02 | **PASS** | Reproduced: +0.0236, [−0.071, +0.114], p 0.597, SE 0.0859, MDE 0.240; +$75; single-category +0.043. C1's pass rule is applied as pre-registered. Its interval excludes ±0.25 σ, so the null is informative, not just underpowered. |
| H9 | C2 numbers: 48 of 248, +0.057 [−0.17, +0.31], p 0.654, SE 0.161, MDE 0.45, +$59; Holm over two: neither | **PASS** (numbers) | Reproduced. The verdict *word* is H1. |
| H10 | Fed days holdout, shares: 29 sessions, −0.055 σ [−0.22, +0.10], −$176/day [−$690, +$310]; by year −0.01 / −0.08 / +0.02 / −0.21 vs each year's B0; win/day 38% vs 41%; options 8 sessions +0.003 σ; "did not replicate"; "no forward policy experiment is proposed" | **PASS** | Reproduced exactly. "Did not replicate" is correct: the holdout interval excludes the discovery estimate. Proposing no forward experiment follows 8 / 13.9 and is the owner's to weigh. |
| H11 | Fed-morning sentence as of `e59a9ff`: "about $250 a session of raw morning loss … about $150 Fed-specific excess, against $1,145 and $874 in discovery" | **PASS** | Holdout pre-14:00 net on Fed days −$246/session against −$97 on all sessions, an excess of −$150. Discovery −$1,145 / −$874. Both pairs now use the same bases (raw against raw, excess against excess). The earlier "$590" figure matched neither base. "A $2,300-σ arm does that by chance" is loose but consistent with the holdout interval including zero. |
| H12 | financial_stress −0.53 σ (13 days) / −0.55 σ (6 days), never tested; only Q1 interval excluding zero in either arm | **PASS** | Reproduced. It is the only Q1 cell in either arm whose interval excludes zero. |
| H13 | Shock days pooled: −0.04 σ [−0.28, +0.31], 77 sessions, "nothing left of the discovery −0.33 σ" | **PASS** | Reproduced −0.037 [−0.279, +0.309]. The within-regime wording is H2. |
| H14 | Q7: 39 tested cells on shares, 0 BH passes; options 0 tested cells; no shortlist existed to score | **PASS** | 39 cells meet ≥ 30 sessions and ≥ 50 trades by my count; `bh_pass: []`; options none meet the minimums. |
| H15 | First 60 minutes "nothing"; CPI options −0.28 σ on 12 days, CI across zero | **PASS** | Shares +$5 vs −$78, difference +$83 [−$221, +$391]; options +$12 [−$247, +$293]. CPI options −0.279 [−0.70, +0.18]. |
| H16 | Holdout read exactly once; no discovery number changed | **PASS** (as far as provable) | One JSON per arm, `generated` 01:54:40 and 01:54:54 at `62d17f3` (committed 01:54:28), added in one commit (`5ea40e0`). An earlier local run overwritten before commit could not be detected. Discovery JSONs unchanged since `2f04abc`. In results_tables.md the only removed line is the old title. |
| H17 | Information-set conclusions (results.md:311-314) | **PASS** | Follow from H8–H10. The C2 wording there inherits H1. |
| D1 | Discovery report (all claims) | **PASS** (carried) | Fifth pass, `62d17f3`. Nothing in the discovery sections changed since. |
| D2 | Discovery Fed days, placebo, shock days | **QUESTIONABLE** (carried) | Caveats below. The holdout now adds that the Fed row did not replicate. |

## FAIL detail and the fix

- **H1:** replace "FAIL (pre-declared underpowered)" with "**underpowered, no verdict (13.6)**: +0.057 σ
  [−0.17, +0.31], achieved MDE 0.45 σ". Make the same change at results.md:4 and :312, and at
  report.py, which prints PASS/FAIL for C2 from `pass_rule`. The code's `pass_rule` field may stay; the
  report must not render it as the C2 verdict.
- **H2:** replace "the within-regime split is flat" with "post hoc within-regime split: low +0.15
  [−0.02, +0.34], mid −0.31 [−0.49, −0.13], high +0.57 on 10 days [−0.77, +2.95]; the sign flips across
  regimes; descriptive only".

A follow-up pass need only check these lines. No number moves.

## Caveats to repeat verbatim (not blocking)

- **Fed days:** "Exploratory in discovery (failed the multiple-testing correction, half of it 2022);
  on the holdout −0.055 σ [−0.22, +0.10] on 29 sessions, below the 30-session minimum. The holdout
  rules out the discovery size but cannot say whether discovery was chance or a 2022 effect."
- **C1:** "No difference on release days; the interval excludes ±0.25 σ. Power at a true 0.25 σ was
  about 50% under the pass rule."
- **Placebo:** "The pre-registered placebo favours 'effect' by construction. Read it beside the
  in-pool rank."
- **Shock days:** "Descriptive; salience-selected list; unstable in discovery, and pooled near zero on
  the holdout."

## Not run

- §18 acceptance ρ: quoted, not recomputed. It is outside gate 2.
- No placebo or block-bootstrap figures were recomputed for the holdout, and results.md quotes none.
- Whether any analysis touched the holdout outside the committed run: not provable from the files.
  The H16 evidence is timestamps and commit order.

GATE 2 VERDICT: FAIL
