# 001 — Trust the clock

**Status:** implemented 2026-10-19 (detection only). Verify live: the next session's log
shows a `-- clock --` section with a `CLOCK:` line.

## Why

On 2026-10-15 the laptop woke from a ten-hour hibernate at 10:46 MST with its clock about
7 seconds fast. Windows set the time from the hardware clock, and the Windows Time service
(start type Manual, "Last Successful Sync Time: unspecified") did not correct it until
15:19:57 MST — a −6,946 ms correction, after the close.

- Both arms reject a stock quote whose age (laptop clock minus exchange timestamp) is over
  5 seconds (`runner.py:52`, `shares_runner.py:73`). Every quote measured 6.1–7.7 s.
- 946 (options) and 1,770 (shares) `stale_quote` rows; zero trades.
- Preflight printed `[OK] … REAL-TIME (age 6.2s)` and "All checks passed": its freshness
  limit is 120 s (`preflight.py:39`).

Full trace: `context_recovery/18_live_lab_daily_review.md`.

## Requirements

1. When preflight runs, it shall measure this machine's clock against an internet time
   server (SNTP, UDP 123) and print a `CLOCK:` line under `-- clock --`.
2. When the clock is 2 s or more off, that line shall be a WARN naming the size and
   direction of the skew and the fix. When the skew is 5 s or more (the arms' quote limit),
   it shall say plainly that the arms will reject every quote.
3. When no time server answers, the line shall be a WARN saying the offset is unverified.
4. The clock line shall never be a FAIL and shall never stop a session.
5. During the session, the supervisor shall measure the clock every 5 minutes. It shall
   log a skew when it starts, at most every 15 minutes while it lasts, and once when it
   ends.
6. The 5 s limit the check describes shall be the same number both runners enforce.

## Touches the frozen record?

No. No trading rule, no config hash, no change to what either arm does with a quote. The
check only measures and logs.

## Acceptance tests

All in `trade_analysis/live_lab/clockcheck_test.py`. Before the change the module did not
exist, so all of them failed at import:

- arithmetic: `test_a_fast_clock_is_reported_as_ahead`, `test_a_slow_clock_is_reported_as_behind`,
  `test_a_short_or_garbage_packet_is_not_a_measurement`;
- server fall-through and never raising: `test_measure_*`;
- the wording thresholds: `test_describe` (6 cases);
- no drift between copies: `test_the_limit_quoted_is_the_one_both_runners_enforce`;
- never a FAIL: `test_preflight_clock_line_is_never_a_fail`;
- the 10-15 case by name: `test_preflight_names_the_2026_10_15_failure`;
- the in-session watch: `test_watch_*`.

Checked live on 2026-10-19 00:22 MST: `measure()` gave −0.50 s against time.windows.com and
−0.48 s against pool.ntp.org. `w32tm /stripchart` gave +0.497 s for the same gap, written
the other way round.

## Rollout

Landed on `main` after the 2026-10-16 session and before 2026-10-19's 05:30 start.

## Out of scope

- **Setting the clock.** That needs admin rights the lab does not have. Owner action:
  set the Windows Time service to Automatic, and after a long sleep run
  `w32tm /resync /force` in an admin PowerShell.
- **Correcting quote ages by the measured offset.** Bar admission runs on the same clock;
  on a fast clock a "corrected" quote would let an arm act on a bar before it has closed.
- **Changing the 120 s delayed-feed check or the 5 s limit.**
