# Incident — host suspended through the close, 2026-09-25

**The laptop slept from 15:39:33 to 16:19:09 ET, despite the suspend guard.** The network
was also down from about 15:57. The 15:55 flatten never ran. On waking, the shares arm closed
its 22 open positions at 16:19, **at after-hours prices**. The options arm had nothing open
and is unaffected (−$963.62 over 20 signals, all closed by 14:39).

## What the record got wrong

- Two XLP longs were "sold" at an after-hours bid of **76.66**, 6.2% below entry, on a day
  XLP did not fall 6%: −$620.34 and −$634.09.
- In total the 22 late closes carry **−$1,478.61** of the day's −$4,306.22.
- The ledger recorded the day **COLLECTED**, a clean session.

## What it would have been

`gap_recovery.py` replays the session through `SharesLab._manage` itself on vendor history.
Each position is rebuilt at its entry and managed only from the moment the gap began.

| | |
|---|---|
| 22 positions, as recorded | −$1,478.61 |
| 22 positions, recovered (best available) | **−$16.26** |
| the two XLP longs | −$1,254.43 → **+$70.92** |

17 of the 22 would simply have closed at the 15:55 flatten. XLV hit its stop at 15:46,
AAPL its target at 15:51, and XLK its bar exit at 15:40.

**Rebuild quality, stated per row in `live_lab_data/shares/trade_corrections.jsonl`:**
- 7 exact
- 10 approximate: exit parameters recomputed on history bars
- 5 estimates: the 15:55 flatten priced at the real NBBO, assuming nothing fired in the gap

Only the 7 exact rows are applied by default (`apply_corrections`). `trades.jsonl` is
untouched.

## Why most rows are not exact

The shares arm never recorded each position's stop, target and trailing rule; they were
recomputed. On history bars only 32 of today's 165 decisions came out identical and 45 did
not re-fire at all. The vendor revises bars after publication (late prints), so ATR, MACD
and VWAP inputs shift by a few percent. **Fixed:** the shares FILL row now records
`stop`, `target`, `time_exit_min`, `bar_exit`, `trailing` and `timeframe`, as options trade
rows always have. From 2026-09-26, rebuilds are exact.

## The self-check, and a correction to it

Every position that closed normally before the gap is replayed too, as a check.
- The first gate required the **same exit rule and the same minute**. It measured 59% (19 of 32
  exact rebuilds) and refused.
- Every miss but one was the same rule, one bar early or late: the revised-bar effect again.
- Same-rule agreement was **97%** (31 of 32), and 100% (71 of 71) for approximate rebuilds.
- The gate now tests the rule. Minute agreement is reported beside it in every run.
- This change was made **after seeing the first gate fail**, and is recorded here for that
  reason.

## What changed

1. **Neither runner takes a fresh quote after 16:00.** The shares arm uses the last
   in-session NBBO and the options arm the last mark. The row is flagged `*_after_close_mark`
   and an outage is recorded.
2. **`gap_recovery.py`** resolves exits for positions open across a gap, and only those.
   - It never opens a position.
   - Output is append-only, idempotent and self-checked.
3. **`autostart` runs it after every session**, in a subprocess, before the archive commit.
   - It retries any day in the last 7 that still needs it, so a day whose recovery failed
     (network still down at 16:00) is completed the next evening.
4. **A day with a suspend during RTH is recorded PARTIAL**, not COLLECTED.
5. Config hashes are unchanged: `ccd00ac71f8243e9` / `7c6414fc549315f3`.

## Not changed, and why

**The 2026-09-25 ledger line stays COLLECTED.** The ledger ranks a day by its best state,
so appending PARTIAL now would not show. It is corrected here instead.

## Operational

**Keep the machine on power with the lid open through 16:00.** The guard asks Windows not to
idle-sleep, but a closed lid or low battery overrides it.
