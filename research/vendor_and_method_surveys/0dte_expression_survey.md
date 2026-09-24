# Survey — why the options arm loses, and what the literature says to change

Compiled 2026-09-14 from five web-research passes, prompted by the options arm sitting at
−$1,317.82 over 134 ATM trades / 7 sessions while several underlying signals looked fine.

**Preserved here because the source reports existed only in a scratchpad and in task
notifications.** This project has already lost three surveys to exactly that (`48ebea2`).
Everything below is either quoted from a primary source with a URL, or explicitly flagged.

⚠️ **Status: the three central academic sources are WORKING PAPERS, not peer-reviewed.**
Verified on the authors' own pages (rodrigohizmeri.com/research, gustavobfreire.com/research)
— conference-presented and discussant-reviewed, not journal-refereed.

---

## 1. The break-even hit rate, which is the whole frame

Delta-matched break-even hit rate for a ~30-minute hold, SPY-like, 13% IV, crossing the full
quoted spread both ways (**derived by a research pass, not quoted from a paper** — re-derive
against our own MFE/MAE distribution before relying on it):

| wrapper | break-even hit rate |
|---|---|
| shares | **50.3%** |
| 0.80Δ ITM 0DTE | 53.5% |
| **0.50Δ ATM 0DTE — what the lab trades** | **56.9%** |
| 0.30Δ OTM 0DTE | 59.3% |

**Strike selection has strong, consistent support for going HIGHER delta.** Frazzini &
Pedersen's delta-hedged panel (1996–2010), index calls, 1-month: deep OTM **−37.50%/mo** →
ATM **−7.31%** → deep ITM **−0.57%**. Moving ATM→0.80Δ roughly halves wrapper cost. Wider
ITM spreads are real but do not overturn it — fewer contracts are needed per unit delta.

---

## 2. What the evidence says NOT to do

**Do not trade verticals.** A vertical doubles the legs you cross, making the worst term
worse. And the one decade-long quote-level study of 0DTE structures issued an **August 2026
correction** — the bid-ask half-spread had been charged at **1/100 of true size** (0.022 bp
instead of 2.2 bp). Bear put spread went net Sharpe **+0.30 → −0.73**; iron butterfly/condor
**−0.96 → −2.67**. Upstream conclusion, verbatim: *"No structure retains a materially
positive net Sharpe ratio."* See [[vilkov-0dte-dataset]] — that is our own panel, and our
`cost_model.py` is structurally immune (it subtracts an ask price from a bid price; there is
no spot-normalised `bas` term in the P&L path).

**Do not sell premium.** Four independent measurements agree and this is closed:
- Almeida/Freire/Hizmeri, short ATM call delta-hedged, 1,815 dates 2012–2025: **net Sharpe
  negative at all nine entry times** (−0.010 to −0.042), gross ≈ 0. FOMC days removed:
  still negative at all nine.
- Bevilacqua/Hizmeri, one-month SPX straddle: **+0.638 gross → −0.203 (40 bp cost) →
  −0.728 (65 bp)**.
- Dim/Eraker/Vilkov: the 0DTE VRP is *"only about 0.01%"* per day gross, *"which makes it
  hardly profitable"* after delta-hedging.
- Our own ThetaData cost model: **−$0.0171/trade at t = −5.38**.

**Do not go further OTM.** Break-even rises to 59.3%, and Almeida finds OTM wings run the
*opposite* way to the ATM: *"OTM calls and puts rarely violate the upper bound, while their
prices are below the lower bound reasonably often, i.e., they are generally too cheap"* —
directly adverse to condors, strangles and credit spreads.

---

## 3. The premium is real and still too small

Almeida Table 1, annualized 0DTE VRP by entry time, **all p = 0.000** (bootstrap, 2,500
reps): 1.804, 1.562, 1.540, 1.717, 1.806, 2.205, 1.981, 2.460, **2.956** pct pts — versus
0.558 for 1DTE and 0.812 for 22DTE. It rises ~64% from 10:00 to 14:00.

**Real, large, overwhelmingly significant gross — and smaller than the spread at every
hour.** That is the entire story of the short side.

---

## 4. Alpha decay, documented by the authors auditing themselves

The single best decay evidence found, because it is not a replication dispute — it is the
same team's own numbers across two drafts of one paper.

Short ATM call delta-hedged, **NET**, 13:00 entry:

| draft | sample ends | net Sharpe |
|---|---|---|
| June 2024 | 2023-07-03 | **+0.013** |
| May 2025 | 2025-03-18 | **−0.010** |

Their SSD strategy halved (net 0.262 → 0.122 at 10:00). The prose moved with it:

| | June 2024 | May 2025 |
|---|---|---|
| abstract | "exploiting this mispricing is highly profitable" | "highly profitable before 2022, but dissipates after the daily availability of 0DTEs" |
| stability | "remarkably stable performance over time" | "remarkably stable performance **up to 2022**" |

**Twenty months of out-of-sample data flipped the sign.** Cite the specific draft; the
numbers are not stable across versions.

---

## 5. Why win rate is the wrong statistic — now measured three ways

Beckmeyer/Branger/Gayda Table 4, margin-adjusted returns **net of actual retail fills**,
Cboe transaction-level data, Feb 2021 – Sep 2023:

| structure | mean | **median** | P5 |
|---|---|---|---|
| Iron Condor | **−1.1%** | **+5.5%** | −100% |
| Put Spread | +0.1% | +3.0% | −100% |
| Call Spread | −0.2% | +3.3% | −100% |

Their own reading: *"the average trade (Median) of many strategies is indeed profitable.
Average profits instead are mostly negative, driven by a few negative outliers."*
Derived: roughly **18–33 median winning days erased by one 5th-percentile day.**

Two corroborations from the vendor side: **ORATS**, net of slippage and commissions on QQQ
0DTE condors since Oct 2020 — **81% winners, Sharpe 0.64**. And **Option Alpha**'s own
published figures undermine each other: **70.19% of trades win** while only **49.64% of
traders are profitable.**

⛔ **Do not cite "93% of retail investors lose money."** That is **SEBI, India, futures &
options, FY22–FY24** — not US options, not SPX, not 0DTE.

---

## 6. Cost: published net figures are optimistic relative to what we do

Almeida's cost treatment is *"the worst case scenario"* on its own terms — buy at ask, sell
at bid — but three qualifications make it **optimistic for us**:

1. **Hold-to-expiry**, so the option settles at intrinsic and **the spread is paid once, at
   entry.** Our 25-minute exit crosses **twice**.
2. The delta hedge is **static** — no rebalancing cost charged.
3. **No cost on the underlying leg.**

So our structural friction is roughly **2× the already-negative published numbers**. And
**no study anywhere examines a 0DTE hold shorter than two hours** — Almeida's shortest is
the 14:00 entry, and there is no early-exit or profit-target analysis in any source found.
Our 25-minute window is unstudied territory.

**Where we actually sit on execution, measured:** our median entry spread is **0.93% of
mid** (mean 1.04%), against published retail 0DTE effective spreads of **5.0–6.0% of
premium** (Beckmeyer, *after* Cboe price improvement) and 9.6–12.5% non-retail. We execute
~6× better than the flow the *"more than $90 million of these losses are the result of the
transaction costs"* figure (≈72% of $125M) was measured on. **That headline does not
transfer to us** — locally, spread + fees is **14%** of our loss.

**The SEC DERA lever, if execution ever becomes the binding term:** non-marketable limit
orders cost **$0.021–0.028** all-in versus **$0.05** crossing, and *"realized spreads are
always less than half the quoted spread."* But an unfilled limit is a trade that never
happened, which silently deletes the fast-moving cases where the signal was right — a
selection bug, not an improvement. It needs a fill model first.

---

## 7. Time of day

Almeida restricts to **10:00–14:00** deliberately: *"The bid-ask spread tends to be
relatively stable at its minimum between 10:00 and 14:00"*, while *"volume is higher at
market open and close, these times of the day are also the ones with highest bid-ask
spread."* **There is no published evidence for entries before 10:00 or after 14:00.** Our
arm put 81 of 134 trades in the 09:30–10:59 bucket.

Measured on our own records, though, the window makes no difference: inside 10:00–14:00
−$9.89/trade (median spread 0.93% of mid), outside −$9.74/trade (0.79%). **The literature's
spread-window argument does not reproduce on QQQ/SPY ETF options with a usability filter.**

⚠️ **Theta does NOT follow √T.** Measured on the SPXW panel: `P(T) ∝ T^0.405` (R = 0.994),
because IV *rises* ~12% into the close. Per-hour decay 13.2% at 10:00, 10.2% at 12:00,
22.5% at 14:00, 100% in the final hour. **Discard any "late-day theta is 7× morning" figure
derived from a √T assumption.** And *"0DTE loses 50% in the first two hours"* is false — it
is ~17–24%.

---

## 8. The regime our forward test is running in

Computed from Cboe's own VIX history (n = 9,272 sessions) and our `data/momo/SPY.npz`:

| | daily-0DTE era (2022-05-11 →) | full history 1990–2026 |
|---|---|---|
| mean VIX close | 18.59 | 19.43 |
| days VIX > 30 | **3.6%** | **7.9%** |
| days VIX > 40 | **0.4%** | **2.2%** |

**The level is normal; the TAIL is missing** — under half the usual rate of VIX>30 and a
fifth of VIX>40. Max close-to-close SPY drawdown in the entire daily-0DTE era is
**−18.95%**; there has never been a completed 20% bear market in it. Open→close absolute
move: mean 0.60%, p95 1.66%, p99 2.84%; only 2.5% of sessions exceeded 2.0%.

Since short-premium P&L is decided almost entirely by the left tail, **every 0DTE backtest
sampled from this era — including our forward test — under-samples the only thing that
matters.**

---

## 9. UNVERIFIED — do not build on these without re-sourcing

- **Muravyev & Ni (JFE 2020)**, the *"−1% overnight vs +0.3% intraday"* decomposition. SSRN
  returns 403; the figures reached us via search summaries only. **This was reported to the
  user as "the encouraging finding" and then retracted.** Weakly corroborated by a
  Bevilacqua footnote (intraday Sharpes *"an order of magnitude smaller"*), but the
  magnitudes are unsourced.
- **Bevilacqua/Hizmeri short-ONLY net Sharpe.** Their Table 4 reports net figures for the
  long-short strategy and the benchmark, *not* short-only. Must not be assumed positive.
  Their abstract's ">2.0 after costs" is **VIX futures**; SPX straddles net 0.885 / 0.357.
- **Almeida Table 6 annualization.** The paper never labels those Sharpes annualized — they
  are per-trade. Any ×√252 conversion is arithmetic, not theirs.
- **A post-2022-only Sharpe for the short benchmark.** Almeida shows the 2022 break
  graphically (Figure 9) and does not tabulate it.
- **Da/Goyenko/Zhang intraday option momentum** (Sharpe 4.64, 25-minute holds, 41 bp/day) is
  **gross only** — the paper's sole cost statement is an assertion, not a calculation — and
  it is individual equity options, not SPX or 0DTE.
- **Any tastytrade/tastylive 0DTE numeric claim.** The original article was removed and the
  archive was rate-limited.
- **FINRA retail options P&L.** Does not appear to exist publicly.
- **Formal ES/CVaR estimates and a published maximum single-day loss for short 0DTE.**

## 10. Citations that circulate and are wrong

- **Bevilacqua & Hizmeri is not a 0DTE paper.** "0DTE" appears once in 71 pages, in the
  reference list. All straddles are constant-maturity 1/2/3/6 months. Its 0.638 gross
  Sharpe is a **one-month** straddle. Our memory credited it with a 0DTE result until
  2026-09-14.
- **"Beckmeyer/Branger/Grünthaler, Liquidity Provision in a Zero-Day-to-Expiry World"** does
  not appear to exist. The paper is Beckmeyer/Branger/**Gayda**, SSRN 4404704.
- **Brogaard/Han/Won (SSRN 4426358)** is about underlying **volatility**, not liquidity, and
  Dim/Eraker/Vilkov explicitly contest it.
