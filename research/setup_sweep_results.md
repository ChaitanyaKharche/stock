# Results — Frozen 24-setup sweep

Executed 2026-08-20 per `setup_sweep_preregistration.md`. Universe frozen before running.
Every setup reported. Held-out set **not touched** — the decision rule never triggered.

## Sample

Discovery window (2025-05-05 → 2025-11-26): **349 round trips, 102 dates.**
Held-out reserve: 69 round trips, 49 dates — untouched.
23 of 24 setups entered the correction family; `entry 14:00–15:30 ET` was excluded with
n=8 on the true side (he barely trades the late afternoon: **8 of 349**).

## All 23 setups, ranked by raw p

| setup | effect on return | n(T) | n(F) | 95% CI | p raw | p Holm | q BH |
|---|---|---|---|---|---|---|---|
| 10m+15m agreement | **−10.6%** | 138 | 211 | [−22.5, +0.6] | 0.066 | 1.000 | 0.916 |
| price > prior close (aligned) | **+9.6%** | 237 | 112 | [−1.7, +21.2] | 0.094 | 1.000 | 0.916 |
| ADX > 25 | −8.3% | 163 | 186 | [−21.0, +3.9] | 0.182 | 1.000 | 0.916 |
| gap aligned | +6.9% | 198 | 151 | [−6.5, +19.6] | 0.284 | 1.000 | 0.916 |
| entry 10:30–14:00 ET | −7.0% | 171 | 178 | [−19.4, +5.9] | 0.285 | 1.000 | 0.916 |
| volume > 1.5× EMA20 | −9.8% | 45 | 304 | [−27.2, +8.8] | 0.293 | 1.000 | 0.916 |
| ATR% > 20d median | +9.2% | 39 | 310 | [−9.4, +26.7] | 0.323 | 1.000 | 0.916 |
| MACD hist aligned > 0 | −5.0% | 240 | 109 | [−18.3, +7.7] | 0.441 | 1.000 | 0.916 |
| entry 09:30–10:30 ET | +4.4% | 163 | 186 | [−8.1, +16.7] | 0.473 | 1.000 | 0.916 |
| RSI(14) aligned > 50 | +4.7% | 285 | 64 | [−9.4, +17.7] | 0.497 | 1.000 | 0.916 |
| opening range cleared (aligned) | +3.0% | 196 | 153 | [−9.2, +15.5] | 0.649 | 1.000 | 0.916 |
| +DI > −DI (aligned) | −3.0% | 270 | 79 | [−17.9, +10.6] | 0.671 | 1.000 | 0.916 |
| ADX > 20 | +2.6% | 243 | 106 | [−14.3, +17.3] | 0.722 | 1.000 | 0.916 |
| volume > EMA20 | −2.2% | 161 | 188 | [−15.9, +12.2] | 0.750 | 1.000 | 0.916 |
| entry 09:00–10:00 LOCAL | +2.0% | 36 | 313 | [−10.9, +18.1] | 0.776 | 1.000 | 0.916 |
| price > VWAP (aligned) | +2.0% | 301 | 48 | [−13.8, +15.7] | 0.783 | 1.000 | 0.916 |
| trailing 15m move aligned > 0 | −2.5% | 311 | 38 | [−22.0, +15.6] | 0.823 | 1.000 | 0.916 |
| 0DTE | +1.3% | 283 | 66 | [−12.5, +14.1] | 0.837 | 1.000 | 0.916 |
| price > day open (aligned) | +1.3% | 254 | 95 | [−11.3, +13.4] | 0.842 | 1.000 | 0.916 |
| day range > prior day range | −1.9% | 38 | 311 | [−21.2, +21.0] | 0.844 | 1.000 | 0.916 |
| DTE ≥ 1 | −1.3% | 66 | 283 | [−14.2, +12.7] | 0.854 | 1.000 | 0.916 |
| upper half of day range (aligned) | +1.1% | 296 | 53 | [−13.9, +14.9] | 0.876 | 1.000 | 0.916 |
| prior-day direction aligned | +0.8% | 176 | 173 | [−12.5, +13.6] | 0.918 | 1.000 | 0.918 |

## Verdict

| | |
|---|---|
| family size | 23 |
| **expected false positives at raw α=0.05** | **1.2** |
| **observed at raw p < 0.05** | **0** |
| survive Holm | **0** |
| survive BH q<0.10 | **0** |

**Not one setup reached even the uncorrected threshold.** That is a stronger outcome than
"nothing survived correction" — the sweep produced *fewer* nominal hits than chance alone
predicts. Every 95% CI in the table contains zero.

Held-out data was never used, because nothing qualified to be tested against it.

## Reading it honestly

**The two closest calls point in opposite directions**, and the nearest is negative: his own
**10m/15m confirmation at −10.6% (p=0.066)**. The second, `price > prior close`, is +9.6%
(p=0.094). Neither is significant, and a coin-flip family of 23 will routinely produce a
best raw p near 0.07.

**Power, as pre-stated:** SE on a balanced split ≈ 6.1pp, uncorrected MDE ≈ **17pp**, and
≈22–24pp after Holm. So the claim is *"no simple setup separates his outcomes by more than
roughly 20 percentage points of return"* — **not** "no setup works." Most observed effects
sit between 1 and 10pp, comfortably inside the noise band. A genuine 5pp edge would be
invisible to this design and to this sample.

**What it does close:** the "we haven't looked hard enough" hypothesis. 23 standard setups —
trend, location, momentum, volatility, participation, time-of-day, structure — were fixed in
advance and all reported. The search was run properly and returned nothing.

## Scope

This tested whether a setup **separates his existing trades**. It did not backtest any setup
as a standalone strategy; that remains unauthorised here and is bounded by the 2020-01-01
options floor.

Seventh consecutive pre-registered null on this record.
