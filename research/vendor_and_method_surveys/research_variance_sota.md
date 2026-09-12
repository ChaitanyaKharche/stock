# Intraday realized-variance forecasting and the 0DTE VRP — web research
*Compiled 2026-09-10. VERIFIED = source fetched and read. UNCERTAIN = search snippet / secondary.*
*Claims marked ✅ were independently re-verified by the main session.*

## Bottom line

**The most likely explanation for "implied variance beats HAR-RV-J by 24% in QLIKE" is that the
HAR baseline is misspecified for intraday periodicity, not that the market is a great
forecaster.** 0.411136 / 0.541358 = 0.7594 — a 24.1% QLIKE reduction from one regressor.
Published IV-augmentation gains at daily horizons are single-digit percent. IV mechanically
encodes time-to-close and the diurnal U-shape; a HAR on day/week/month lags does not.
Periodicity-adjusting the HAR is cheap, is validated **on SPY**, and could flip the premise.
Do it before spending cluster time.

Second: **the 0DTE short-premium null is now replicated three times independently**, and one of
those is a correction that reverses a published positive this project has on disk.

---

## 1. Time-series foundation models, Sept 2026

### The one paper that directly answers the question
**VERIFIED.** Brini, *Forecasting Realized Volatility with Time Series Foundation Models*,
arXiv:2607.05291v1 — https://arxiv.org/html/2607.05291v1

9 TSFMs zero-shot vs 8 econometric specs. 50 assets incl. E-mini S&P 500, 2015-01-02 to
2026-01-30, VOLARE 5-min RV. **Horizons h=1, 5, 22 DAYS — no intraday.** QLIKE on variance scale.

QLIKE loss ratio vs Log-HAR (Table 6):

| Model | h=1 | h=5 | h=22 |
|---|---|---|---|
| **TTM** | **0.982** | **0.986** | **0.987** |
| Sundial | 0.998 | 1.084 | 1.182 |
| Moirai-2.0-S | 1.038 | 1.109 | 1.259 |
| Moirai-MoE-S | 1.090 | 1.102 | 1.140 |
| TimesFM-2.5 | 1.086 | 1.201 | 1.331 |
| Chronos-Bolt-S | 1.110 | 1.188 | 1.302 |
| Chronos-Bolt-B | 1.121 | 1.181 | 1.303 |
| Toto | 1.214 | 1.273 | 1.157 |
| Lag-Llama | 1.532 | 1.582 | 1.023 |

MCS inclusion rate (% of 50 assets): TTM 98/96/94 (avg 0.96); Log-HAR 86/90/90 (0.88);
HAR 86/66/54. **Only TTM beats Log-HAR, by 1.3–1.8%, and does not displace it from the MCS.**
Every transformer-scale model loses. A context-length ablation shows accuracy rising
monotonically with context — the models are rediscovering HAR's lookback, not adding information.

### Finance-wide verdict on generic TSFMs
**VERIFIED.** Rahimikia, Ni & Wang, arXiv:2511.18578v1 — https://arxiv.org/html/2511.18578v1
Off-the-shelf: Chronos-large R²_OOS **−1.37%**, TimesFM-500M **−2.80%**, vs CatBoost −0.10%.
Finance-native *pre-training* moves Chronos-small from −77.07% to −3.18%.
Quoted: "Generic time series pre-training does not directly transfer to financial domains."

**VERIFIED (abstract).** Goel, Pasricha, Magris & Kanniainen, arXiv:2505.11163 — TimesFM beats
econometric benchmarks **only after incremental fine-tuning**; zero-shot insufficient.

### Model inventory, licences, contamination

| Model | Params | Licence | Covariates | Contamination, SPY 2020–24 |
|---|---|---|---|---|
| **Chronos-2** | 120M / 28M | **Apache-2.0** | **past + known-future, native** | **Low** |
| TimesFM-2.5 | 200M | Apache-2.0 | XReg | Medium (10% leak, fev-bench) |
| TimesFM-3 | 330M | **non-commercial weights** | multivariate + known-future | Unknown |
| TiRex | 35M xLSTM | NX-AI Community | not documented | Unknown |
| Moirai-2.0 | 11–117M | Apache-2.0 | yes | **High (28% leak)** |
| Toto | 151M | Apache-2.0 | — | Unknown |
| Sundial | 128M | open | — | Unknown |
| TTM | <1M | Apache-2.0 (IBM) | yes (r2) | Low |
| Granite PatchTST-FM-r2 | ~385M | Apache-2.0 / OpenMDW-1.0 | not documented | Unknown |
| **Kronos** | 24.7/102.3/499.2M | open weights | OHLCV native | **DISQUALIFYING** |

- **VERIFIED** Chronos-2 — https://huggingface.co/amazon/chronos-2 ·
  https://arxiv.org/html/2510.15821v1 — encoder-only T5-style, RoPE, alternating time/group
  attention, 21-quantile head, context 8192. **Real-pretraining domains: Energy, Nature,
  Transport, Web, Cloud Ops — no financial domain.** Covariate training entirely synthetic.
  fev-bench win rates: Chronos-2 90.7% / skill 47.3; TiRex 80.8/42.6; TimesFM-2.5 75.9/42.3.
- **VERIFIED** TiRex — https://huggingface.co/NX-AI/TiRex — commercial terms not stated on card.
- **VERIFIED** Granite PatchTST-FM-r2, released 2026-09-09 —
  https://huggingface.co/blog/ibm-research/ibm-releases-sota-granite-time-series
- **UNCERTAIN** TimesFM-3 (2026-08-31, 330M): weights under `timesfm-non-commercial-license-v1.0`,
  code Apache-2.0. https://github.com/google-research/timesfm
- **VERIFIED** Kronos, arXiv:2508.02739 — https://arxiv.org/html/2508.02739v1 — 12B K-line
  records, 45 exchanges, **1-minute through weekly**, **training data through June 2024.**
  **The held-out year is 2024. Kronos saw 1-min US equity bars through June 2024. It cannot be
  honestly evaluated on the held-out set, and overlap with 2020–2023 train/validation is total.**
- **VERIFIED** contamination is measured, not hypothetical: fev-bench
  (https://arxiv.org/html/2509.26468v4) records 10% (TimesFM-2.5) and 28% (Moirai-2.0) leakage.

### What nobody has published
**No benchmark evaluates zero-shot TSFMs on INTRADAY realized variance.** Brini is daily; Goel
daily; Rahimikia daily returns. This setup is unbenchmarked, and the daily priors are poor.

---

## 2. SOTA for realized-variance forecasting, 2025–2026

### Highest-value item: periodicity-adjusted HAR (HARP)
**VERIFIED.** Dumitru, Hizmeri & Izzeldin, *Forecasting the realized variance in the presence of
intraday periodicity*, J. Banking and Finance 170:107342 (2025) —
https://rodrigohizmeri.com/wp-content/uploads/2024/11/dhi_jbf.pdf ·
https://www.sciencedirect.com/science/article/pii/S0378426624002565

- **SPY** plus 30 S&P 500 constituents, 2000–2020, 5,284 days, TickData.
- Periodicity **inflates the variance of RV and biases jump estimators**; both degrade forecasts.
  HARP builds HAR predictors from periodicity-filtered returns.
- **"For SPY, we observe improvements of up to 7% in the forecast losses using HARP models."**
  Stock average 6.6–7.3% at 1-day and 1-week. Losses = MSE and QLIKE. Filtered models
  "consistently rank first" in the Hansen MCS.
- **Directly indicts HAR-RV-J:** SPY jump days fall 324 (6.13%) → 271 (5.12%) filtered; stock
  average collapses 504 → 183. **The J component is partly spurious.**
- Periodicity-adjusting the RV forecast also improves the **variance risk premium** as a return
  predictor, raising R² by up to 50% for the aggregate market.

**VERIFIED.** Christensen, Hounyo & Podolskij — https://arxiv.org/html/2601.16613 — the diurnal
pattern accounts for a significant fraction of intraday volatility variation, but "important
sources of heteroskedasticity remain." Deseasonalising is necessary, not sufficient.

### ML at intraday horizons — two real datapoints, both warnings
**VERIFIED.** Zhang, Zhang, Cucuringu & Qian, JFEC 22(2):492 (2024) —
https://academic.oup.com/jfec/article/22/2/492/7081291 · https://arxiv.org/pdf/2202.08962
Horizons **10/30/65-min**, 93 S&P 500 stocks, Nasdaq ITCH, 2011-07 to 2021-06. QLIKE at 10-min:
LSTM-Aug 0.376 vs HAR-D-Aug 0.453 (~17%); 30-min 0.171 vs 0.227 (~25%); 65-min 0.160 vs 0.186.
**Caveat that kills applicability: "MLPs and LSTMs are only performed under the Universal and
Augmented settings" — they never fit a single-stock neural model, for lack of data.** Intraday
commonality is 74.3% at 65-min vs 35.5% daily. We have one asset.

**VERIFIED.** Brini & Toscano, *SpotV2Net*, arXiv:2401.06249v4 — https://arxiv.org/pdf/2401.06249
**30-minute** spot volatility, 30 DJIA stocks, 2020-06-01 to 2023-05-10, 10,318 estimates.

| QLIKE | HAR-Spot | XGB | LSTM | SpotV2Net-NE | SpotV2Net |
|---|---|---|---|---|---|
| Validation | 0.263 | 0.217 | 0.184 | 0.173 | **0.137** |
| **Test** | **0.343** | 0.429 | 0.359 | 0.354 | **0.286** |

**The single most useful table for this programme.** On validation every ML model beats HAR-Spot.
On test, XGB (0.429) and LSTM (0.359) both **lose** to HAR-Spot (0.343). Only the graph model
exploiting cross-asset spillovers wins. SpotV2Net's own validation→test QLIKE degrades
0.137 → 0.286, a factor of 2.1. A 346/250/151 split sits squarely in that regime.

### Rough vol / path-dependent vol
**UNCERTAIN.** Abi Jaber & Li, *Volatility Models in Practice*, Mathematical Finance (2025) —
https://onlinelibrary.wiley.com/doi/10.1111/mafi.12463 · https://arxiv.org/abs/2401.03345
Rough models **underperform** one-factor Markovian models at 1wk–3mo with equal parameter counts.

**VERIFIED.** Fan, Wang & Ye — https://arxiv.org/html/2604.02743v1 — S&P 500 5-min 2011–2021,
SPX options, OOS 2020-01 to 2021-06, horizons 1–22 **days**. 1-day OOS QLIKE: HAR-RV-RHeston
**0.0403** vs HAR-RV **0.0428** (~6%). HAR-RV-VIX beats HAR-RV but loses to RHeston (DM −2.5446).
**The gain comes from the options data, not from roughness.**

**UNCERTAIN (canonical).** HARQ/HARQ-F, Bollerslev-Patton-Quaedvlieg (2016) —
https://public.econ.duke.edu/~boller/Papers/MV_HARQ_020818.pdf — HARQ sits *inside* Brini's 90%
MCS alongside HAR and Log-HAR: a legitimate benchmark, not an upgrade. Realized quarticity at
intraday sampling is extremely noisy; HARQ's benefit may not survive at 30–60 min origins.

### The result that reframes the whole objective
**VERIFIED.** Pollok, *Predicting Realized Variance Out of Sample: Can Anything Beat The
Benchmark?*, arXiv:2506.07928 — https://arxiv.org/pdf/2506.07928
Verbatim: *"We find, viewed through traditional forecast rankings, it is hard to significantly
and unambiguously beat the benchmark models. However ... alternative volatility forecasts can
lead to economically significant increases to returns, both absolute and risk-adjusted."*
The economic test is a portfolio sort on the **forecast-RV-minus-IV spread** forming ATM
delta-neutral straddles. **QLIKE and P&L are only loosely coupled, in both directions. Measure
the P&L. This is exactly the target-variable error that has burned this project before.**

### Free data that partially relieves "can never be extended"
**VERIFIED.** VOLARE, arXiv:2602.19732 — https://arxiv.org/pdf/2602.19732 ·
archive **https://volare.unime.it**, **CC BY 4.0**. Kibot tick data. Stocks from 2015-01-02;
includes **E-mini S&P 500**. Parkinson, Garman-Klass, realized range, RV, RQ, bipower,
semivariances, median/min RV, realized kernel, realized covariances. Sampling 1-min/5-min.
**Daily rows only — it will not extend the intraday frame**, but it is the only open replacement
for the Oxford-Man Realized Library (discontinued Feb 2023). Use as free independent daily
validation.

---

## 3. Is "IV beats HAR" consistent with the literature?

**Direction: yes. Magnitude: outlier. Framing: malformed.**

**VERIFIED.** Kambouroudis, McMillan & Tsakou, J. Futures Markets (2021) —
https://rahwebdav.swan.ac.uk/repec/pdf/WP2019-03.pdf — 10 indices, QLIKE, Hansen MCS +
Giacomini-White, 16 specifications. Verbatim: **"Across all indices only HAR specifications that
account for implied volatility belong to the MCS."** Also: "the parsimonious HAR-IV specification
yields remarkably lower loss than specifications that exclude implied volatility, but include
simultaneously two or more of the other stylized facts." Vol-of-RV "adds little or nothing."

**UNCERTAIN (canonical).** Busch, Christensen & Nielsen, J. Econometrics 160(1):48–57 (2011) —
https://www.sciencedirect.com/science/article/abs/pii/S0304407610000564 — **in FX, IV completely
subsumes daily/weekly/monthly RV.** In stock and bond markets IV is incremental but does *not*
fully subsume. They propose VecHAR to handle IV endogeneity.

**UNCERTAIN (403).** Michael, Cucuringu & Howison, Quantitative Finance 25(3):443–470 (2025) —
https://www.tandfonline.com/doi/abs/10.1080/14697688.2025.2454623

**Four specific problems with the comparison as specified:**
1. **Magnitude.** Published IV gains at daily horizons ~6–7%; TTM 1.3–1.8%; biggest intraday gains
   anywhere are LSTM-panel 14–25%. **24.1% from one regressor on one asset is at the very top of
   the published range.**
2. **Wrong baseline.** Nobody treats IV-alone-vs-HAR-alone as decisive. The standard benchmark is
   **HAR-IV**. The neural arm should be judged against HAR-IV (or HARP-IV).
3. **Specification threat.** IV at time *t* mechanically encodes remaining time-to-close and the
   diurnal expectation; a HAR-RV-J at intraday origins with no time-of-day term cannot. HARP
   recovers 7% on SPY at *daily* horizons alone; at intraday origins the deterministic diurnal
   component is far larger.
4. **Heavy tails.** t = −2.30 on a 24% loss reduction over ~250 clustered sessions means the loss
   differential is extremely heavy-tailed — a handful of sessions is carrying it. Under SPA/MCS
   with a model sweep, t = −2.30 will not survive. Also "p=0.0222 at h=30 vs p=0.0507 at h=60" is
   a Gelman-Stern error: the difference between those is not itself significant.

---

## 4. 0DTE, 2025–2026

### The VRP is real, positive, and about the size measured here
**VERIFIED.** Almeida, Freire & Hizmeri, *0DTE Asset Pricing* —
https://www.fma.org/assets/docs/Derivatives2025/Almeida.pdf · SSRN 4701401
**CBOE intraday SPX/SPXW 1-minute bid/ask, 2012-01-06 to 2025-03-18, 1,815 dates.** Entries
10:00–14:00 in 30-min steps, held to 16:00. **Annualised 0DTE VRP 1.54%–2.95%**, vs 0.55% (1DTE)
and 0.81% (22DTE). Upside and downside components both significantly positive, **upside exceeding
downside** — the reverse of the 1-month horizon. Relative bid-ask is U-shaped over the day,
"relatively stable at its minimum between 10:00 and 14:00."

### The unconditional short-premium harvest is net-negative — three independent replications
**VERIFIED.** Almeida et al. Table 6, write ATM delta-hedged call, Sharpe by entry time,
gross / net (sell at bid, buy at ask):

| Entry | 10:00 | 10:30 | 11:00 | 11:30 | 12:00 | 12:30 | 13:00 | 13:30 | 14:00 |
|---|---|---|---|---|---|---|---|---|---|
| Gross | 0.002 | −0.016 | −0.015 | −0.002 | 0.003 | 0.015 | 0.017 | 0.016 | 0.010 |
| **Net** | **−0.037** | **−0.042** | **−0.039** | **−0.027** | **−0.023** | **−0.011** | **−0.010** | **−0.012** | **−0.033** |

**VERIFIED.** Bevilacqua & Hizmeri, *Early Birds Get the Vol* —
https://wp.lancs.ac.uk/fofi2026/files/2026/03/FoFI-2026-091-Mattia-Bevilacqua.pdf
Verbatim: *"The benchmark strategy of consistently selling SPX straddles, which is profitable in
gross terms (Sharpe ratio of 0.64), delivers a negative Sharpe ratio once transaction costs are
incorporated."*

**✅ VERIFIED BY THIS SESSION — AND IT TOUCHES THIS REPO.** Vilkov *0DTE Trading Rules*
replication package — https://github.com/vilkovgr/0dte-strategies/blob/main/KNOWN-ISSUES.md
Re-fetched directly 2026-09-10; text confirmed verbatim.

> **2026-08 correction:** bid-ask stored as fraction of spot while returns were in percent of
> spot, so the half-spread was charged at **1/100 of true size — ~0.022 bp instead of 2.2 bp.**

Net Sharpe before → after: put ratio spread +0.84 → **−0.61**; risk reversal +0.44 → +0.10;
bear put spread +0.30 → **−0.73**; strangle/straddle −0.51 → **−0.97**; iron fly/condor
−0.96 → **−2.67**. Stated conclusion: **"No structure retains a materially positive net Sharpe
ratio."** Secondary 2026-05 defect: costs *added* rather than subtracted on short days.

**✅ Local exposure assessed by this session — limited.** The defect is in Vilkov's ANALYSIS code;
`data_opt.parquet` is unaffected. `trade_analysis/backtesting/vilkov_0dte_conditional_backtest.py`
reads only the parquet, imports no upstream analysis code, and applies the same `scale` to `mid`
and `bas` before charging `bas/2` per side plus FEE_BP=0.5 — it does not reproduce the
percent-vs-fraction mix. The VRP cost-model result (gross t=+7.52 / net t=−5.38) came from
ThetaData SPY 0DTE quotes, not this panel, so it is independent.

### What *does* work in the literature is conditioning, not harvesting
- **Almeida et al.:** the **SSD-violation** strategy (buy the delta-hedged ATM option when its
  price violates the risk-averse lower bound, sell when it violates the upper) earns net Sharpe
  **0.101–0.159** across the nine entry times vs −0.010 to −0.042 unconditional. Costs barely dent
  it. The conditioning device is crude — a historical return histogram rescaled by current ATM IV.
  Conditional Sharpe is **higher when 0DTE volume, realized variance and attention are LOW.**
- **Bevilacqua & Hizmeri:** morning VVIX at 10:00 ET predicts next-day variance-asset returns,
  t up to 5.6, adj R² up to 2.6%, **power dies after 11:00 ET**. Long-short at the 75th percentile:
  annualised Sharpe 1.73 SPX straddles, 2.09 VIX futures, 2.87 variance swaps. Net: SPX straddles
  0.89 at 40bp, 0.36 at 65bp. **"Profitability stems from both legs."** Needs intraday VIX-options
  data; sample ends June 2022.

### Have spreads kept tightening into 2025–2026? Volume yes; spreads probably; **premium no**
- **UNCERTAIN (Cboe/press).** 0DTE hit a record **66.2% of total SPX options volume in July 2026**;
  62% YTD through Aug 2026 (https://www.cboe.com/insights/posts/spx-0-dte-options-jump-to-record-62-share-in-august).
  Trajectory 17% (2020) → 37% (2022) → 62% (2026).
- **UNCERTAIN (practitioner).** Concretum, 30-min Cboe quote snapshots Jan 2022–Aug 2026: median
  ATM SPX 0DTE spread **0.20 index points**; 0.10–0.34 bp of strike —
  https://concretumgroup.substack.com/p/spx-0dte-options — **not broken out by year**, so it does
  *not* confirm continued tightening after 2023. No published year-by-year 0DTE relative-spread
  series for 2024–2026 was found. **Treat extrapolation of 0.0575 → 0.0248 (2020–2023) into 2024+
  as unverified.**
- **The countervailing trend, and it is decisive.** Almeida et al.: mispricing is "highly
  profitable before 2022, but **dissipates after the daily availability of 0DTEs**." Edge is larger
  when volume and attention are LOW. **Spreads tighten because volume and attention rise; the
  premium shrinks for the same reason. The net gap may be closing from both ends and converging on
  zero. Any model of only the cost leg is one-sided.**

---

## 5. Evaluation methodology for a sweep on one validation set

**VERIFIED.** `arch` (Sheppard) implements everything needed —
https://bashtage.github.io/arch/multiple-comparison/multiple-comparison-reference.html
- **`SPA`** — Hansen's Superior Predictive Ability: does *any* model beat a designated benchmark,
  correcting for having looked at many. The correct headline test.
- **`StepM`** — stepwise, returns *which* models beat the benchmark under FWER control.
- **`MCS`** — Hansen, Lunde & Nason (2011). T×k loss matrix, `block_size` for the block bootstrap.
- Alternative Python MCS: https://github.com/JLDC/model-confidence-set

**Note (this session): `arch` is NOT currently installed** in the repo venv — statsmodels 0.14.5
and scipy 1.15.3 are present, `arch` and `hyppo` are not.

**Protocol:** (1) SPA, not a pile of pairwise DMs, with **HAR-IV as the benchmark, not HAR**.
(2) Block-bootstrap by **session**; set `block_size` ≥ origins-per-session and verify the effective
sample. (3) MCS on the survivors, reported as a set with p-values, not a winner. (4) Report the
count of specifications examined, including discarded ones.
**UNCERTAIN (paywalled):** Grant (2026), multi-horizon DM extension —
https://onlinelibrary.wiley.com/doi/full/10.1002/for.70150

---

## 6. Contradictions with established project facts

| Established fact | Web evidence | Action |
|---|---|---|
| "IV alone BEATS the HAR family" | Direction consistent. **Magnitude 24.1% is an outlier**; no paper treats IV-alone-vs-HAR-alone as decisive. | Re-benchmark against HAR-IV. Test the periodicity artifact. |
| HAR-RV-J is the benchmark | HARP: periodicity **biases the jump estimator**; SPY jumps 324→271, stock avg 504→183. | Add HARP / HARP-J. ~7% free on SPY. |
| "Net gap closes because spreads tighten" | Almeida: mispricing dissipates after daily expirations; conditional Sharpe higher when volume/attention LOW. | Model both legs; test whether *gross* edge trends. |
| Short ATM 0DTE straddle net-negative | **Replicated three times independently.** | Cost model looks right. **Stop re-testing the unconditional short.** |
| Vilkov 0DTE dataset | **100× cost understatement, corrected 2026-08.** ✅ re-verified. | Local exposure assessed: limited (see §4). |
| "Dataset can never be extended" | True for the intraday options frame. **Not** for daily RV — VOLARE (CC BY 4.0). | Use as free independent daily validation. |
| Buy-only project constraint | Every working strategy here needs **both legs**. | **Open scoping question: does buy-only bind the VRP programme?** |

---

## 7. Ranked: what to try, what not to

### Tier 1 — days, not weeks, before any GPU spend
1. **Periodicity-adjust the HAR baseline and re-run the DM.** Build HARP / HARP-J from
   periodicity-filtered returns, and add an explicit time-to-close term since IV encodes it and
   HAR does not. **Falsifier: if the IV-vs-HARP DM loses most of its magnitude and |t| drops below
   ~1.5, the headline was a specification artifact — the market is not beating us and the neural
   arm's expected value is restored. If the gap survives, it is a much stronger negative and the
   neural arm should stop.**
2. **Re-baseline everything against HAR-IV.** Also run the encompassing regression
   `log RV_{t+h} = a + b·log RV_HAR + c·log IV_t` — the Busch-Christensen-Nielsen "subsumes" test,
   one regression.
3. **Inspect the per-origin QLIKE loss-differential distribution.** How many sessions carry
   t = −2.30? If fewer than ~10, no SPA or MCS will keep it.

### Tier 2 — hours of CPU, no cluster
4. **Zero-shot Chronos-2 with implied variance as a known-future covariate.** The only
   contamination-clean, Apache-2.0, covariate-native model in the field, and the only zero-shot
   experiment that can beat HAR-IV rather than merely HAR. **Expected outcome: it loses. Do it
   anyway** — a few hours closes the "we never tried zero-shot" question permanently.
5. **Add TTM (<1M params, Apache-2.0) as the zero-shot control.** The only TSFM to beat Log-HAR at
   every horizon. If TTM cannot reproduce even a 1–2% edge on this frame, the frame differs from
   the published one and that needs explaining.
6. **VOLARE as a free external validation set.** Validate model *ranking* there so the 151-session
   held-out set is spent on the answer, not on discovering a bug.
7. **Wrap the sweep in `arch` SPA + MCS, block-bootstrapped by session, benchmark = HAR-IV.**

### Tier 3 — only if Tier 1 leaves the neural arm alive
8. **Map QLIKE to P&L explicitly** (Pollok). Build the forecast-RV-minus-IV sort and measure
   straddle P&L directly. Do not infer P&L from QLIKE.
9. **Replicate the conditional structure, not the carry** (Almeida SSD: net 0.101–0.159 vs
   −0.010 to −0.042). Their conditioning device is crude. **If a good RV forecast cannot beat a
   rescaled histogram, the forecasting programme has no economic value even if it wins on QLIKE.**

### Do NOT bother
- **Do not train a bespoke deep model on 346 sessions of one asset.** Zhang et al. could not fit
  single-stock neural models at all; SpotV2Net's univariate LSTM (0.359) and XGB (0.429) both lost
  to HAR-Spot (0.343) on test after winning on validation.
- **Do not use Kronos.** Pretrained through June 2024 on 1-min bars. The held-out year is 2024.
- **Do not quote zero-shot Moirai-2.0 or TimesFM-2.5 as clean** — 28% and 10% measured leakage.
- **Do not build on TimesFM-3 weights** if production is conceivable — non-commercial licence.
- **Do not re-test the unconditional short 0DTE straddle.** Four independent measurements agree.
- **Do not pursue rough volatility as a forecasting upgrade.** Gains come from options data, not
  roughness; Abi Jaber & Li find rough underperforms Markovian.
- **Do not extrapolate spread tightening without modelling the premium trend.** They move together,
  in opposite P&L directions.
- **Do not run a GPU-cluster sweep before Tier 1.** If the IV advantage is a diurnal artifact the
  whole design changes; if it is real the neural arm is very likely dead. Either way the cluster
  time is currently unjustified.
