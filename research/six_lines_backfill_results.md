# Results — the trader's six lines, replayed over the live lab's first 19 sessions

**Run 2026-09-25**, before either six-line setup had traded a live session.

**This is a BACKFILL, not evidence.**
- It was computed after the fact, so the lab's decide-before-price guarantee does not hold.
- It is never counted toward any checkpoint (amendment of 2026-09-25 in
  `forward_test_preregistration.md`).
- Code: `trade_analysis/live_lab/backfill.py`. Output: `live_lab_backfill/options/`.

## 1. What was run

The new options config, `ccd00ac71f8243e9`, is the 13 original setups plus `Six_Lines` and
`Six_Lines_NoCap`.
- It ran through the **real `LiveLab` runner code** over every session from 2026-08-28 to
  2026-09-24: 19 sessions, QQQ and SPY.
- A history-backed feed and a simulated clock stepped once a minute.
- Warmup ran at 09:07, as it now does live, so the replay **sees the open**. The live record
  did not (`incident_2026-09-24_opening_window.md`).

## 2. Is the replay faithful? Checked three ways before reading any result

| check | result |
|---|---|
| Six-line signals vs `six_lines.first_break_trade` itself, same history | **38 of 38 sessions identical**: same bar, same line, same direction |
| 13 original setups, replay vs live record, on the 15 days the live arm traded | **+$3,344.71 vs +$3,269.11** (+2.3%) |
| Same signal, same strike, 2026-09-23 | 14 of 17 matched signals on the same strike; median entry-price gap 3.9% |

The three large price gaps on 09-23 were all an ATM strike flip: the underlying sat at a
strike midpoint (e.g. 771.48 vs 771.62). Running the same day twice gave identical numbers.

## 3. What the six lines would have done (ATM contract)

| | trades | total | median | win | exits |
|---|---|---|---|---|---|
| `Six_Lines` (his spec, +20 bp cap) | 26 | **−$156.10** | −$35.08 | 9 / 26 | 16 failed · 9 target · 1 eod |
| `Six_Lines_NoCap` | 26 | **+$1,533.90** | −$36.08 | 7 / 26 | 17 failed · 9 eod |

- **The uncapped twin's total is one day.** Its best trade is +$1,040.92, **68%** of the
  total, on 2026-09-21. Both variants have a median trade near −$36; the typical six-line
  trade loses.
- **The cap cost $1,690 here, almost all of it that one day.** That is the direction the cap
  has gone every time it was measured, but 26 trades on 19 sessions do not add a fourth
  measurement. They show what the rule does on a trend day, not an edge.

## 4. What changes in the daily record

On the 15 sessions the live options arm traded:

| | total |
|---|---|
| live record (13 setups, blind to the open) | +$3,269.11 |
| replay, same 13 setups, sees the open | +$3,344.71 |
| + Six_Lines | +$11.32 |
| + Six_Lines_NoCap | +$1,657.32 |
| **replay, new config** | **+$5,013.35** |

- **2 of 15 sessions change sign:** 2026-09-02 (−$126.53 → +$84.14) and 2026-09-16
  (+$88.46 → −$625.02).
- Eight sessions were positive before and eight after.
- The replay also covers **2026-09-08 → 09-11**, when the live options arm was blocked by the
  lapsed entitlement. There the new config loses **−$4,042.71**. So the all-19-session
  replay total is +$970.62. That figure compares against no live number, because the live
  arm did not trade those days.

## 5. What this does not say

- **Nothing about an edge.** 26 trades per variant against a pre-registered bar of 200. The
  2,438-trade QQQ underlying test of this same rule (`six_lines_results.md`) is a null.
- **Nothing that shortens the clock.** The live six-line record starts at zero on 2026-09-25.
