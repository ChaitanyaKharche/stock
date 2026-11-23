---
task_id: R-2026-11-20-01
trace_id: https://claude.ai/code/session_01TqhCAiwZiZ56SHrYyFXGVT
role: experiment-runner
repo_sha: f440cf5
inputs:
  - live_lab_backfill/news_arms/2018..2022/shares, 2020..2022/options (discovery replays, gate-2 audited in news_arms)
  - live_lab_data/trades.jsonl, live_lab_data/shares/trades.jsonl (frozen forward test, from 2026-08-28)
decision_rule: "none: descriptive. No test, no verdict, no change to the lab."
outputs:
  - research/setup_baseline_results.md
  - research/setup_baseline.json
evidence:
  - "every number below: research/setup_baseline.json, written by trade_analysis/news_arms/setup_baseline.py"
assumptions:
  - "options replay is SPY only; the live options arm trades QQQ and SPY"
  - "the shares rules were written on 2026 microstructure and replayed on 2018-2022 bars"
open_questions:
  - "the holdout (shares 2023-2026-08, options 2023) has never been read per setup; a pick-then-check on it needs its own pre-registration"
errors: []
status: done
verified_by: null
---

# Per-setup baseline: does dropping the losing setups leave a profitable book?

Generated 2026-11-20T22:26:52 by `trade_analysis/news_arms/setup_baseline.py`. **Descriptive.
Not pre-registered, not audited.** The question came from the owner on 2026-11-14: profitable
trades show up in the terminal every day, yet the day ends at −$2,000 to −$3,000, so are good
setups being dragged down by the others, and is $500 a day already achieved?

## The answer in plain words

**No. Over the replayed years there is no profitable group of setups to keep.** Setups that
ended above zero, with their trade counts and dollars per session: shares ThreeBarPlay (342 trades, +0.1);
options IntradayMomentumBoundary (885 trades, +8.6), VWAP_2sigma_Fade (19 trades, +1.1). Choosing the setups that were ahead in the first part of the
history and keeping only those for the rest gives +0 a session on shares (none
qualified) and +18 a session on options. The live totals rest on a few crash days: without
its two largest days the options arm is at -75 a session and the shares arm at
-194.

## Shares arm, replay 2018-01-02 → 2022-12-30

1,249 sessions, 226,678 trades, hash ['7c6414fc549315f3']. Whole book: **-286 a session**.

| setup | trades | total $ | $ per trade | $ per session | win % | 2018 | 2019 | 2020 | 2021 | 2022 |
|---|---|---|---|---|---|---|---|---|---|---|
| ThreeBarPlay | 342 | +69 | +0.20 | +0.1 | 39.8 | -234 | -436 | +213 | -287 | +814 |
| VWAP_2sigma_Fade | 288 | -522 | -1.81 | -0.4 | 54.9 | -795 | -35 | -488 | -3 | +799 |
| Crabel_Stretch | 17,407 | -13,137 | -0.75 | -10.5 | 43.0 | +12,974 | -6,024 | -15,174 | -6,642 | +1,728 |
| IntradayMomentumBoundary | 21,828 | -13,472 | -0.62 | -10.8 | 23.2 | +6,738 | -8,687 | -14,980 | -3,007 | +6,465 |
| EMA_9_20_Pullback | 3,633 | -16,702 | -4.60 | -13.4 | 30.9 | -1,311 | -1,701 | -3,806 | -4,446 | -5,438 |
| Gap_Fade | 6,003 | -17,256 | -2.87 | -13.8 | 46.0 | +1,839 | -466 | -15,085 | -5,915 | +2,372 |
| PDH_PDL_Breakout | 15,405 | -19,255 | -1.25 | -15.4 | 36.5 | +884 | -267 | -18,171 | +3,135 | -4,836 |
| TTM_Squeeze | 18,136 | -19,375 | -1.07 | -15.5 | 42.8 | -5,612 | -6,817 | -3,853 | -4,202 | +1,109 |
| Six_Lines_NoCap | 17,154 | -24,127 | -1.41 | -19.3 | 27.3 | -52 | -1,428 | -15,994 | -8,272 | +1,618 |
| ORB_15min | 16,352 | -24,622 | -1.51 | -19.7 | 36.8 | -6,810 | +44 | -11,228 | +1,941 | -8,569 |
| ORB_5min | 17,010 | -26,338 | -1.55 | -21.1 | 44.5 | -4,547 | +5,034 | -23,952 | +1,714 | -4,587 |
| PDH_PDL_FailedBreak | 12,074 | -28,042 | -2.32 | -22.4 | 39.6 | -4,220 | -1,225 | -1,787 | -5,102 | -15,709 |
| Six_Lines | 17,154 | -37,357 | -2.18 | -29.9 | 59.3 | -4,288 | -4,671 | -15,659 | -7,849 | -4,890 |
| VWAP_Reclaim | 23,941 | -51,690 | -2.16 | -41.4 | 40.8 | -14,187 | -6,710 | -18,993 | -5,027 | -6,772 |
| MOMO_CHASE | 39,951 | -64,856 | -1.62 | -51.9 | 45.4 | -16,749 | -11,486 | -22,700 | -9,969 | -3,952 |

Positive over the whole window (hindsight): ThreeBarPlay, together +0.1 a session.

**Pick-then-check.** First part to 2020-12-31 (748 sessions), second part 501 sessions. Net positive in the first part: **none**. Those setups alone in the second part: **+0 a session**. Whole book in the second part: -187 a session.

## Options arm, replay 2020-01-03 → 2022-12-30

482 sessions, 9,943 trades, hash ['ccd00ac71f8243e9']. Whole book: **-100 a session**.

| setup | trades | total $ | $ per trade | $ per session | win % | 2020 | 2021 | 2022 |
|---|---|---|---|---|---|---|---|---|
| IntradayMomentumBoundary | 885 | +4,119 | +4.65 | +8.6 | 22.0 | -716 | +274 | +4,562 |
| VWAP_2sigma_Fade | 19 | +533 | +28.08 | +1.1 | 63.2 | +46 | +57 | +431 |
| ThreeBarPlay | 15 | -345 | -23.01 | -0.7 | 26.7 | -120 | -190 | -34 |
| PDH_PDL_Breakout | 681 | -422 | -0.62 | -0.9 | 35.5 | -1,614 | +1,606 | -414 |
| TTM_Squeeze | 917 | -1,204 | -1.31 | -2.5 | 41.7 | -204 | -841 | -159 |
| EMA_9_20_Pullback | 180 | -1,606 | -8.92 | -3.3 | 25.0 | -340 | -1,005 | -260 |
| Six_Lines | 745 | -2,047 | -2.75 | -4.2 | 55.6 | -188 | -489 | -1,371 |
| Crabel_Stretch | 711 | -3,296 | -4.64 | -6.8 | 33.6 | -4,171 | -795 | +1,669 |
| MOMO_CHASE | 1,796 | -3,543 | -1.97 | -7.3 | 42.3 | -684 | -1,504 | -1,356 |
| Gap_Fade | 265 | -4,441 | -16.76 | -9.2 | 35.1 | -2,359 | -1,556 | -527 |
| VWAP_Reclaim | 972 | -5,876 | -6.04 | -12.2 | 38.6 | -3,297 | -1,497 | -1,081 |
| Six_Lines_NoCap | 745 | -6,432 | -8.63 | -13.3 | 22.1 | -913 | -258 | -5,262 |
| ORB_15min | 772 | -7,395 | -9.58 | -15.3 | 32.8 | -1,955 | -577 | -4,863 |
| ORB_5min | 712 | -7,720 | -10.84 | -16.0 | 37.8 | -3,126 | +305 | -4,899 |
| PDH_PDL_FailedBreak | 528 | -8,666 | -16.41 | -18.0 | 30.5 | -2,595 | -1,173 | -4,898 |

Positive over the whole window (hindsight): IntradayMomentumBoundary, VWAP_2sigma_Fade, together +9.7 a session.

**Pick-then-check.** First part to 2021-06-30 (234 sessions), second part 248 sessions. Net positive in the first part: IntradayMomentumBoundary, VWAP_2sigma_Fade. Those setups alone in the second part: **+18 a session** (IntradayMomentumBoundary +4,018, VWAP_2sigma_Fade +440 in total). Whole book in the second part: -92 a session.

## Live lab since the freeze

Frozen forward test (`live_lab_data/FREEZE.json`): no setup may be dropped, and the first per-setup evaluation is at 200 trades per setup. Everything in this section is a diagnostic, not an effect claim.

| | options (ATM) | shares |
|---|---|---|
| sessions | 54 | 56 |
| trades | 1,183 | 8,561 |
| total | +20,667 | +10,584 |
| mean per session | +383 | +189 |
| median session | -275 | +4 |
| sessions at +$500 or better | 16 | 17 |
| sessions at −$500 or worse | 25 | 23 |
| three largest days | 2026-11-05 +17,149, 2026-11-09 +7,431, 2026-11-20 +3,461 | 2026-11-05 +14,743, 2026-11-20 +6,293, 2026-11-09 +4,380 |
| per session without the largest 1 | +66 | -76 |
| per session without the largest 2 | -75 | -194 |
| per session without the largest 3 | -145 | -280 |
| trades whose exit was rebuilt from history | 25 | 274 |

Off-ladder option fills: 8 of 1183 live signals were filled on a strike more than 1.5 points from the underlying or with a neighbour arm more than two strikes away ({'2026-11-05': 1, '2026-11-20': 7}). No nearer strike had a usable quote in the chain that minute and the runner took the nearest one that did. Cause not established: both days were 4% selloffs, so a thin or crossed market, strikes not listed that far from the prior close, or a feed gap are all possible. Those rows are in the record as ATM trades.

### Live options, per setup (diagnostic)

Last column leaves out the two largest days, 2026-11-05, 2026-11-09.

| setup | trades | total $ | $ per trade | win % | total without the two largest days |
|---|---|---|---|---|---|
| Crabel_Stretch | 103 | +5,610 | +54.46 | 33.0 | -993 |
| Six_Lines_NoCap | 65 | +4,783 | +73.58 | 24.6 | -606 |
| IntradayMomentumBoundary | 160 | +4,491 | +28.07 | 19.4 | +1,811 |
| ORB_15min | 100 | +4,220 | +42.20 | 43.0 | +1,462 |
| ORB_5min | 72 | +2,486 | +34.53 | 41.7 | -762 |
| VWAP_Reclaim | 108 | +825 | +7.64 | 44.4 | -29 |
| TTM_Squeeze | 86 | +419 | +4.87 | 44.2 | -70 |
| Six_Lines | 65 | +378 | +5.81 | 50.8 | -75 |
| MOMO_CHASE | 220 | +291 | +1.32 | 41.4 | -136 |
| EMA_9_20_Pullback | 15 | +198 | +13.19 | 40.0 | +198 |
| PDH_PDL_Breakout | 94 | +64 | +0.69 | 29.8 | -1,614 |
| VWAP_2sigma_Fade | 4 | -20 | -5.08 | 0.0 | -20 |
| ThreeBarPlay | 1 | -119 | -119.08 | 0.0 | -119 |
| Gap_Fade | 33 | -1,138 | -34.47 | 27.3 | -1,138 |
| PDH_PDL_FailedBreak | 57 | -1,821 | -31.94 | 22.8 | -1,821 |

### Live shares, per setup (diagnostic)

Last column leaves out the two largest days, 2026-11-05, 2026-11-20.

| setup | trades | total $ | $ per trade | win % | total without the two largest days |
|---|---|---|---|---|---|
| Six_Lines_NoCap | 476 | +5,066 | +10.64 | 33.6 | +156 |
| IntradayMomentumBoundary | 853 | +4,322 | +5.07 | 26.1 | -507 |
| Crabel_Stretch | 752 | +4,294 | +5.71 | 43.6 | +175 |
| ORB_15min | 665 | +2,194 | +3.30 | 37.6 | +396 |
| ORB_5min | 709 | +1,246 | +1.76 | 44.0 | -480 |
| Gap_Fade | 307 | +1,101 | +3.59 | 46.6 | +935 |
| VWAP_2sigma_Fade | 11 | -52 | -4.76 | 63.6 | -52 |
| ThreeBarPlay | 9 | -54 | -5.97 | 22.2 | -54 |
| Six_Lines | 476 | -116 | -0.24 | 58.2 | -437 |
| TTM_Squeeze | 640 | -283 | -0.44 | 42.3 | -481 |
| PDH_PDL_Breakout | 666 | -387 | -0.58 | 38.3 | -1,044 |
| EMA_9_20_Pullback | 143 | -912 | -6.38 | 32.2 | -1,072 |
| MOMO_CHASE | 1,484 | -1,801 | -1.21 | 45.8 | -3,083 |
| VWAP_Reclaim | 910 | -1,812 | -1.99 | 41.9 | -2,168 |
| PDH_PDL_FailedBreak | 460 | -2,222 | -4.83 | 39.1 | -2,736 |

## What this does and does not say

- It says the mechanical setups, as frozen, have no positive subset over the replayed years
  that survives being chosen in advance. Seeing a +$300 trade in the terminal is real; so are
  the many small losers beside it, and which trade wins is only known afterwards.
- It does not say anything about the owner's own discretionary trading, which is a different
  record (`journal_reverse_engineering_results.md`).
- It does not use the holdout. A fair forward check of "keep only setup X" exists in the
  data (shares 2023–2026-08, options 2023) and needs its own pre-registration before a read.
- The live per-setup columns are below the lab's own 200-trade checkpoint for every setup
  except those at or past it, and the live totals are tail-dominated; they are not evidence
  for or against any setup.
- Replay limits as in `news_arms/preregistration.md` §9: vendor-corrected bars, fills on the
  minute, no outages; VWAP_Reclaim, ORB_5min and MOMO_CHASE replay with low trade-for-trade
  fidelity.
