# Specs

One file per change to the live lab, approved before any code. Research already works
this way (every study in `research/` starts as a pre-registration); this extends the
same discipline to code.

## Rules

1. No change under `trade_analysis/live_lab/` without a spec here.
2. A spec's tests are written first and must fail on the code as it was. A test that
   passes on the bug is decoration.
3. Anything that changes a trading rule is a dated amendment to the pre-registration,
   not a spec. That decision is the owner's.

## Template

```
# NNN — short name

Status:       draft | approved | implemented | verified-live (with the session date)
Why:          the incident or evidence, with dates and file:line
Requirements: one sentence each: "When <something happens>, the <part> shall <do this>."
Touches the frozen record?   config hash / trading rules / both arms / none
Acceptance tests: named tests; each shown to FAIL on the old code
Rollout:      when it lands (after 16:05 ET or a weekend) and which log line proves it
Out of scope: what this spec deliberately does not change
```

## Index

| # | name | status |
|---|---|---|
| 001 | [Trust the clock](001-clock-trust.md) | implemented 2026-10-19, detection only |
| 004 | [Survive a network switch](004-network-switch-watchdog.md) | implemented 2026-10-19 |

Specs 002 (scheduled task after DST, due before 2026-10-30), 003 (options arm half-day
close, due before 2026-11-25), 005 (truthful session label) and 006 (the two frozen-rule
defects) are proposed in `context_recovery/19_next_steps_spec_driven.md` and not yet
written.
