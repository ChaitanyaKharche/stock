# 004 — Survive a network switch

**Status:** implemented 2026-10-19. Verify live: on a day with a network switch, the log
shows either no `[watchdog]` restart while offline, or a "restart budget restored" line
after recovery.

## Why

The commute design of 2026-09-28 (`research/network_resilience.md`) froze and caught up
correctly. But `TerminalWatchdog` had one branch that broke its own rule, "never restart
when the network is merely down":

- The **not-answering** branch restarted the terminal without asking whether the internet
  was up (`autostart.py`, `TerminalWatchdog.check`, before this change).
- On 2026-10-05 (from 14:09 ET) and 2026-10-06 (from 12:26 ET) the network died. The
  terminal stopped answering, was restarted into a network that was still gone, and failed
  six times inside about 20 minutes.
- The cap was per session and permanent, so the watchdog then logged "leaving it for a
  human" every minute until the close: 41 lines on 10-05, 93 on 10-06.

On a commute day (home WiFi → phone hotspot → work WiFi) that sequence turns the first long
gap — a dead zone, a captive-portal login — into a lost afternoon, even after the laptop
reaches the next network.

## Requirements

1. When the terminal is not answering and the internet is unreachable, the watchdog shall
   not restart it and shall not count the check toward a restart.
2. When the internet comes back, a terminal that is still not answering shall be restarted
   after the usual two checks.
3. When the terminal has served history again for 10 consecutive checks, the restart budget
   shall be restored.
4. When the budget is spent, the watchdog shall still try once every 15 minutes until the
   close, instead of never.
5. "Leaving it for a human" shall be logged once per episode, not every minute.

## Touches the frozen record?

No. This is supervisor code: no trading rule, no config hash, no arm behaviour. The arms'
own freeze-and-catch-up (`catchup.py`) is unchanged.

## Acceptance tests

In `trade_analysis/live_lab/watchdog_test.py`. The first five failed on the old code:

- `test_not_answering_while_offline_is_never_restarted`
- `test_not_answering_is_restarted_once_the_network_is_back`
- `test_the_budget_comes_back_after_the_terminal_has_been_healthy`
- `test_past_the_cap_a_slow_retry_still_happens`
- `test_leaving_it_for_a_human_is_logged_once_not_every_minute`
- `test_no_slow_retry_before_its_interval` — a guard that passes on both; it pins that the
  cap still holds inside the 15 minutes

The existing six watchdog tests pass unchanged, including `test_the_restart_cap_holds`.

## Rollout

Landed on `main` after the 2026-10-16 session and before 2026-10-19's 05:30 start. The full
offline suite passed: 1,160 passed and 1 skipped, against 1,133 and 1 before.

## Out of scope

- **A captive portal that answers TCP 443 for any host.** `_internet_up` can mistake it for
  the internet. The budget, its restoration and the slow retry bound the cost; logging in
  to the portal is the owner's step.
- **The 6-restart cap and the 2- and 3-check thresholds.** Unchanged.
- **The host sleeping.** No code runs during Modern Standby (`research/network_resilience.md`).
