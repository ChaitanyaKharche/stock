# news_arms — gate 2 audit of the discovery report

Auditor, 2026-11-10. Audited: `results.md`, `results_tables.md`, `analysis_{shares,options,q6}_discovery.json`,
`counts_{shares,options}_discovery.json`, `analyze.py`, `analyze_test.py`, `join.py`, `events.py`, raw replay
chunks under `live_lab_backfill/news_arms/2018–2022/`. Nothing in `results.md` or the code was edited.

**In one sentence: gate 2 fails. The holdout lock is already open, and three of the report's explanations are
contradicted by its own data. The core numbers are correct: every headline effect, CI and dollar figure
reproduced independently from the raw replay files.**

## Verdict table

| # | Claim | Verdict | Reason |
|---|---|---|---|
| 1 | Holdout guard: "`analyze --set holdout` refuses until this file carries [the gate-2 pass token]" (results.md:207-208) | **FAIL** | `holdout_allowed()` is a substring test (analyze.py:556), and results.md:208 itself contains that literal token. It returns **True today**, so the holdout can be read before any audit. |
| 2 | F3: "that is the volatility regime, not the shock"; "the regime column already captures this" (results.md:27-32, 128-129) | **FAIL** | Removing each regime's mean leaves −0.362 σ of the −0.379 σ. Within the high regime, shock days are −0.75 σ *against high-regime days*. 2020 carries it, not the regime. |
| 3 | F1: the placebo centre sits above B0 "because the runners lose less on Mondays and Fridays" (results.md:69-71, 192-194) | **FAIL** | Year×weekday mix explains +0.012 of +0.042 σ (shares) and +0.020 of +0.080 σ (options). The rest is the pool excluding the condition's own losing days within each stratum. |
| 4 | "Missing shares sessions are half days dropped by rule" (results.md:185) | **FAIL** | 23 regular 2022 sessions are missing because of vendor HTTP 500s at warm-up, not half days. They include FOMC 2022-11-02 and NFP 2022-08-05 and 2022-10-07. No list, no count, chunk rc 0. |
| 5 | Fed days "same sign in all five years" / "F2 is every year" (results.md:100, 200; by-year column at :85) | **FAIL** | The by-year column is the absolute mean z, not the effect. Against B0, 2020 is **+0.041 σ**, and +0.177 σ against 2020's own mean. 4 of 5 years are negative. |
| 6 | "−0.16 bp of $10,000 notional" per average trade (results.md:51, 102) | **FAIL** | B0 is −$1.59 per trade, which is **−1.6 bp**. It is off by 10×, so "−6.1 bp vs −0.16 bp" overstates the Fed-day gap 38× instead of 3.8×. |
| 7 | "stands aside before 14:00 on Fed days … avoided … about $7,000 a year" (results.md:106-108) | **FAIL** | $6,860/yr is the *whole-day excess*. Standing aside before 14:00 removes the raw pre-14:00 net of **−$8,400/yr**, of which the excess over an average day is **−$6,300/yr**. |
| 8 | Minor misstatements (results.md:55, 52, 151, 152-153, 184) | **FAIL (wording)** | "every Tue/Thu" (26 Tue/Thu before 2022-11-14 were kept: real holiday-shifted and quarter-end expiries); "−$4.8" is −$4.86; "fourteen" setups is 15; ORB_5min is 7th of 14 on Fed days, not "near the top"; "17 shares chunks" is **49**. |
| 9 | F2 Fed days, shares −0.282 σ [−0.55, −0.01], 38 sessions, BH fail in family of 4, candidate only | **QUESTIONABLE** | Reproduced exactly. Half of it is 2022 (without 2022: −0.156 σ [−0.304, −0.001]). A 10-day block bootstrap moves the upper bound to 0.000. One 2022 FOMC session is missing to the outage. |
| 10 | All shares discovery numbers (σ, B0, every cell) | **QUESTIONABLE** | Computed on 1,226 of 1,249 regular sessions. The 23 missing ones cluster in Jul–Nov 2022, a high-volatility stretch, so they are not missing at random. Direction of bias unknown. |
| 11 | Placebo ranks / "selection / chance" verdicts, every table | **QUESTIONABLE** | Implemented as pre-registered (13.3), but excluding the condition's sessions from the pool biases the null *away* from the real effect, which favours "effect". Scheduled flips to chance when the pool includes them (0.965→0.852 shares, 0.9998→0.925 options). FOMC holds (0.961). |
| 12 | F3 downgrade and its post hoc within-regime split | **QUESTIONABLE** | The post hoc label is adequate. But the pre-registered §2 rule already marks shares shock days **"unstable"**: positive vs B0 in 2018, 2019 and 2021, positive in the low tercile. That label is never applied. financial_stress supplies 16 of the 114 days; without it −0.217 σ. |
| 13 | §2 "unstable" rule applied to every table | **QUESTIONABLE** | Not implemented anywhere (no code, no output). `by_year`/`by_regime` (analyze.py:326-331) are absolute means, so they cannot detect an effect sign flip. |
| 14 | "a test asserts none [of the conditioning columns] is a price or P&L field" (results.md:197-198) | **QUESTIONABLE** (as evidence) | analyze_test.py:37-40 checks names in a tuple (analyze.py:58) that no code reads. It would pass on a renamed price column, so it is decoration. The lookahead property itself passes on inspection (row 22). |
| 15 | Audit trail: `repo_sha` `6e91528` in both analysis JSONs | **QUESTIONABLE** | `analyze.py` does not exist at `6e91528`; the run used uncommitted code. A re-run on HEAD reproduces both JSONs bit-for-bit, so the numbers are unaffected. |
| 16 | Q3 "unscheduled_surprise" definition | **QUESTIONABLE** | analyze.py:351-352 also drops 21 shares (11 options) surprise sessions that carry a release. This is not in §6 Q3. Small effect (−0.349 σ with them, −0.379 σ without). |
| 17 | B0 and σ: shares 1,226 sessions / 222,363 trades / σ $3,197 / B0 −$288 (median −$683); options 482 / 9,943 / $1,032 / −$100 (−$214) | **PASS** | Reproduced from the raw daily and trade files by an independent loader. |
| 18 | F1 calendar-release days: shares −0.045 σ [−0.16, +0.08] = −$143 [−$510, +$250]; options −0.091 [−0.26, +0.09] = −$94 [−$270, +$90]; BH fail; "−0.8 bp per $10,000 trade"; 46 sessions/yr | **PASS** | Reproduced (the CI-based null, not the placebo explanation). |
| 19 | Implementation = pre-registered statistic (effect def., σ ddof=1, session bootstrap 3,000 / seed 20260828, placebo 5,000 matched by year×weekday with exclusion, rank rule 14.Q6, minimums 30/50, BH q=0.10 per table, Q7 zero-fill with arm σ, shortlist rule) | **PASS** | Read line by line and reproduced numerically. No placebo needed the stratum fallback. Q7 cell PDH_PDL_FailedBreak\|FOMC −0.0562 σ / −$179.5 reproduced. |
| 20 | Families identical to the pre-committed counts (Q7 66 / 20; H3 4 / 2) | **PASS** | Set equality, not count: Q7 sets equal in both arms, Q1 tested {CPI, FOMC, NFP, PCE} / {NFP, PCE}, descriptive+placebo {trade_tariff} / {}, every cell's (sessions, trades) identical. |
| 21 | 14.11: 270 no-0DTE sessions excluded before σ, B0 and every table; all Tue/Thu before 2022-11-14 | **PASS** | 270 = 132 Tue + 138 Thu, last 2022-11-10, 94/95/81 by year. My σ reproduces only with them dropped first (join.py:90-100). |
| 22 | No conditioning on the session's own prices (15.4); regime from trailing-20 prior sessions; cuts = discovery terciles | **PASS** | Conditions come from event files, regime, year and weekday only. `med20_range_bp` = median of the 20 *prior* sessions (0/2,676 mismatches; 1,534 mismatches if today is included). Cuts 111.2118 / 184.4911 reproduced as terciles of 1,249 discovery QQQ days. |
| 23 | F4 (CPI, NFP, GDP_ADV, PCE both arms), F5 (one BH pass, shortlist empty), F6 (first-60: −$186 [−$577, +$215]; −$88 [−$276, +$109]), F7 (no divergence) | **PASS** | All reproduced. F6 recomputed independently (different seed: [−$585, +$210], [−$276, +$102]). |
| 24 | Fed-day timing: loss before 14:00, "a morning loss" (Q5, 14.4) | **PASS** | Pre/post follows 14.4 (14:00 boundary, entry-based; no session mixes kinds within a category). Of the −$902 excess, −$831 is pre-14:00 and −$592 is 09:30–10:30. Pre-14:00-entered trades realised −$49.5k before 14:00 and +$7.5k after, so this is not a path-vs-endpoint error. |
| 25 | "−$34,300 over five years"; −$902; −6.1 bp vs −0.5 bp; 195 vs 180 trades; win/day 29% vs 39%; options stops 41% vs 35%, RoP −5.4% vs −0.2%, twin −7.9 vs +2.1 bp; low-regime Fed −0.21 [−0.41, −0.01]; F3 table cells | **PASS** (transcription) | 902 × 38 = $34,276. All the others match the JSON. The F2 caveats (row 9) still apply. |
| 26 | Re-runs give identical JSON | **PASS** | Re-ran `run()` for both arms. Identical apart from `generated` / `command` / `repo_sha`. |
| 27 | No holdout file read | **PASS** (code path) | Discovery chunk lists stop at 2022-12-31. `load()` opens only discovery years. No holdout join or analysis file exists. Reads outside the code path cannot be proven absent; the console exposure in §19/§21 is already logged. |

## FAIL detail

**1. Holdout guard is open.** analyze.py:553-556 returns True if the gate-2 pass token (the words GATE 2, colon, PASS) appears anywhere in
results.md, and results.md:208 contains it while describing the guard. Verified:
`python -c "from trade_analysis.news_arms import analyze as A; print(A.holdout_allowed())"` prints `True`.
The test (analyze_test.py:171-175) runs against an empty `tmp_path`, never the committed file, so it passes
on the bug. Violates §2, §11 step 6, 14.3. *Fix:* make the guard require a dedicated anchored line (for example
an anchored whole-line match written only by the auditor, or the verdict line at the end of this file, matched as a whole line), reword results.md:208 so it
does not contain the token, and add a test asserting `holdout_allowed()` is False on the committed results.md
at this commit. That test must fail on the current code.

**2. F3 mechanism is wrong.** The JSON's own within-regime split (`q3.unscheduled_surprise.post_hoc_within_regime.high`)
is −0.750 σ [−1.26, −0.28]: shock days relative to *other high-regime days*. That is the shock, not the regime.
Independent check: subtracting each regime's mean z (low −0.019, mid −0.031, high −0.228) moves the overall
shock effect only from −0.379 to −0.362 σ. What does carry it: 2020. Within high, without 2020, the effect is
−0.307 σ [−0.79, +0.22] (37% of high-regime shock days are 2020), and financial_stress (March 2020) supplies
16 of 114 sessions. *Fix:* F3 reads "unstable by the pre-registered §2 rule (sign flips in 3 of 5 years and
across terciles); carried by 2020; not a candidate". Delete "regime, not the shock" and "a shock rule would
add almost nothing to a regime rule". The verdict "not a candidate" stands, for that reason.

**3. F1 placebo explanation is mostly wrong.** Decomposition (my code, matching the JSON's mean ±0.002):

| | shares | options |
|---|---|---|
| placebo centre as run (pool excludes the condition) | +0.041 | +0.080 |
| year×weekday mix alone (pool includes the condition) | +0.012 | +0.020 |
| left over = within-stratum exclusion | +0.029 | +0.060 |
| rank → verdict, as run | 0.965 → effect | 0.9998 → effect |
| rank → verdict, pool includes the condition | 0.852 → chance | 0.925 → chance |

Wednesdays look bad mostly *because of* the Fed and CPI days on them: all Wednesdays −0.123 σ, non-release
Wednesdays −0.062 σ. Removing the release days from the pool leaves the better days to match against. The
"96th/100th percentile on the losing side" (results.md:68-69) is mostly this artefact. *Fix:* replace the
explanation with the decomposition above and state that the scheduled placebo verdict is not robust to the
pool exclusion. The F1 conclusion (CI includes zero, no difference) is unaffected and passes (row 18).

**4. 23 sessions missing, misdescribed.** Missing from shares discovery: 2022-05-13; 07-15, 18, 19, 20, 21, 22,
25; 08-04, 05, 08, 09; 10-03 to 10-12 (8 sessions); 11-01, 02, 03. Each has a `warmup FeedOutage … HTTP 500`
in `outages.jsonl` and no daily file, and the chunk still exited rc 0. Includes FOMC 2022-11-02, NFP 2022-08-05,
NFP 2022-10-07, and geopolitical 07-25, 08-08, 10-05, 10-07. The 10 thin half days are the only legitimate
drops. Violates 14.1 (exclusions counted and listed) and is the project's standard bug class: a silent gap
with no error. *Fix:* re-replay the 23 sessions and re-run, or, if that is impossible, list them by
date with the reason in results.md and the JSON, and correct :185.

**5. "Every year negative" is false.** Effect vs B0 by year: 2018 −0.246, 2019 −0.055, **2020 +0.041**, 2021 −0.372,
2022 −0.803. The column at results.md:85 (report.py:86, `z_mean`) is the absolute mean z of Fed days that year,
and every one is negative only because the arm loses on an ordinary day (B0 −0.090). Same issue for the options
by-year column. *Fix:* render the by-year and by-regime columns as condition − B0 (or condition − that year's
mean) and restate: "4 of 5 years negative; 2022 is −0.80 σ".

**6. −0.16 bp is −1.6 bp.** 222,363 trades, −$288.48 × 1,226 sessions gives −$1.59 per trade on $10,000. The other
bp figures (−6.1, −0.5, −0.8) are right. *Fix:* correct :51 and :102.

**7. The $7,000 stand-aside figure measures something else.** Per Fed session: pre-14:00 net −$1,105 vs −$275 on
all sessions. A stand-aside rule forgoes the raw −$1,105 × 38 / 5 ≈ **$8,400/yr**; the Fed-specific excess
is −$831 × 38 / 5 ≈ **$6,300/yr**. $7,000 is the whole-day excess. *Fix:* quote one of these two, named,
still as an association (13.2).

**8. Wording.** Correct :55 ("all 270 excluded are Tue/Thu; 26 Tue/Thu before 2022-11-14 had a 0DTE and are
kept"), :52 (−$4.9), :151 (fifteen), :152-153 (VWAP_Reclaim is 2nd by p on Fed days; ORB_5min is 7th of 14),
:184 (49 shares chunks).

## QUESTIONABLE detail, with the caveat to repeat verbatim

**9. Fed days (F2).** Correct as reported: −0.282 σ, CI [−0.554, −0.008], p 0.043, BH threshold 0.025 in the
committed family of 4, single-category −0.254 [−0.52, +0.03] (n 33), sessions are the bootstrap unit (one row
per session, so day-clustering is real). Labelled exploratory/candidate, which is the right label. Fragility
measured: drop 2022 → −0.156 σ [−0.304, −0.001]; three 2022 days (01-26, 06-15, 09-21) are each −2.1 to −2.6 σ;
moving-block bootstrap L = 10 gives [−0.541, 0.000], L = 20 gives [−0.557, +0.022]; one 2022 FOMC session is
missing (row 4). Options: n = 22, underpowered, and the drop-2022 effect is −0.084 σ.
*Caveat:* "Exploratory; fails the pre-registered multiple-testing correction; about half of it is 2022, and
without 2022 the interval barely excludes zero; 4 of 5 years negative, not 5."
*What would change it:* the holdout. There is no Q7 shortlist row for it, so only a separately pre-registered
forward test, as results.md already says.

**10. Missing sessions.** *Caveat:* "Shares discovery is 1,226 of 1,249 regular sessions; 23 mid/late-2022
sessions were lost to a vendor outage and are not in σ, B0 or any cell." *Change:* row 4 fix.

**11. Placebo design.** Excluding the condition's sessions shifts the placebo centre by
Σ_s (k_s/K)·p_s·(mean_non − mean_cond) in each stratum s, always against the real effect's sign, which
makes "effect" verdicts easier to reach. Implemented exactly as 13.3 requires, so this is a design limitation, not
an implementation defect. *Caveat:* "Placebo ranks are anti-conservative by construction; the scheduled
verdict reverses when the condition's sessions stay in the pool." *Change:* report both pools, by dated
amendment before the holdout read.

**12–13. Instability rule.** §2: "a result whose sign flips across terciles or across ≥ 3 of the years is
reported as unstable". Not computed. Applied by hand (effect vs B0): shares shock days are unstable (years +, +, −, +, −;
terciles low +0.118, mid −0.216, high −0.887); shares scheduled is unstable across terciles (mid +0.170);
shares FOMC is stable (all terciles negative, one year flips). *Change:* implement the rule on effects and
print the label.

**14. 15.4 test.** *Change:* derive every mask in `q1`/`q3`/`q5`/`q7` through a function that reads only
`CONDITIONING` keys and fails on any other key, and show the test fails when a price column is introduced.

**15. repo_sha.** *Change:* refuse to write the JSON from a dirty tree, or record `git diff` hash.

**16. Q3 definition.** *Caveat:* "unscheduled_surprise excludes surprise days that also carry a release (21
shares, 11 options) and includes financial_stress rows (16 / 12)." *Change:* state it in the table.

## Checks not run, or not runnable

- **Bias direction of the 23 missing sessions:** unknowable without replaying them. Their P&L does not exist.
- **Whether any holdout file was read outside the code path:** file access times are not reliable on this host.
  I checked the code path and file inventory only.
- **Acceptance (§18) re-verification:** out of scope for gate 2 and not recomputed. results.md's ρ 0.932 / 0.953
  is quoted from §18, not reproduced here.
- **`levels_unavailable` (102), `tick` (12), `flatten_after_close` (28) outages on kept shares sessions:**
  counted, not traced to setups. They can thin Six_Lines / PDH_PDL trades on specific days without dropping
  the session.
- **Power / MDE for C1, C2:** holdout-only. Not applicable to discovery.
- **Cross-arm dependence (Q6):** different windows and session sets per arm; no joint resampling exists, so
  the "difference" column has no CI. Only the pre-registered "both CIs exclude zero, opposite signs" rule was
  checked (no category qualifies).
- **Serial dependence:** lag-1 autocorrelation of daily z is −0.03 (shares) and −0.09 (options); |z| lag-1 is +0.31 and
  +0.14 (volatility clustering). Only the FOMC CI was block-bootstrapped (row 9); other CIs were not.

GATE 2 VERDICT: FAIL
