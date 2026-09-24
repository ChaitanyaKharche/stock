# Omnibus Dependence Detection Under Serial Dependence — Web Research Report

**Date of research:** 2026-09-10
**Scope:** web research only. No repository files were read or modified.
**Verification convention used throughout:**

- **[V]** = VERIFIED — I fetched the source page/abstract myself and the claim comes from that fetched text.
- **[U]** = UNCERTAIN — claim comes from a search-result snippet only; I did not fetch and read the primary source. Treat as a lead, not a fact.

---

## 0. The one-paragraph answer, before the details

There **is** a principled multiple-testing-safe way to ask "is there any exploitable dependence at all" — but almost every method that gets recommended for it dies on your serial-dependence constraint, and the survivors are *far fewer and far more boring* than the literature suggests. The honest short answer is: **cluster/block at the session level, use a small number of pre-specified omnibus statistics evaluated by a session-level block bootstrap or block permutation, wrap the whole programme in e-values so the accounting is additive and dependence-agnostic, and — before running any of it — compute the minimum detectable effect.** With hundreds of sessions and per-trade means smaller than their standard errors, the dominant risk is not a false positive; it is running a beautiful nonparametric omnibus test that had ~5% power all along and reporting its null as evidence of "no signal." That is exactly your project's documented silent-failure mode, dressed in new clothes.

---

## 1. The filter that kills most candidates: what serial dependence actually breaks

### 1.1 Naive permutation is provably broken here — with a named citation

**[V] Romano & Tirlea, "Permutation Testing for Dependence in Time Series" (arXiv:2009.03170).**
https://arxiv.org/abs/2009.03170

Key statements from the fetched abstract:

- Permutation tests are exactly level α in finite samples **only under an i.i.d. null**.
- For testing *uncorrelatedness* in dependent data, standard permutation tests are **"neither exact nor approximately level α in large samples."**
- They document **large Type 3 (directional) errors**: a two-sided test concludes *positive* autocorrelation when the truth is *negative* correlation.
- Their fix retains exact level α when observations are genuinely independent, while adding asymptotic validity under dependence.

**[U]** A companion framing found via search: if the null specifies zero autocorrelation and you permute using the sample first-order autocorrelation as the statistic, Type 1 error can be "shockingly different from the nominal level, even asymptotically." Same source family; I did not fetch the specific page carrying that sentence.

**[U]** The classic genetics-side cautionary paper — *Naive Application of Permutation Testing Leads to Inflated Type I Error Rates*, Genetics 178(1) — makes the general point that when data are **not exchangeable**, permutation Type I error exceeds nominal.
https://academic.oup.com/genetics/article/178/1/609/6062374

**Implication for your existing permutation control.** Your current control (shuffle returns to kill trend, preserve volume/volatility/geometry) is *directionally right* and is a genuinely good idea — but as a plain i.i.d. shuffle it is **not** a valid null for a serially dependent series. Shuffling destroys the autocorrelation that exists under your null too, so the permutation null distribution is too narrow, and anything with a serial-dependence footprint (which is most intraday statistics) will look significant. **The fix is cheap: permute whole sessions, not observations.** Sessions are your natural exchangeable unit. This converts the shuffle into a *block permutation* with block = session, which is precisely the construction the literature validates (see §1.2).

### 1.2 Block permutation is the validated repair, and it is implemented

**[V] Shen, Chung, Mehta, Xu, Vogelstein, "Independence Testing for Temporal Data," Transactions on Machine Learning Research, 2024** (arXiv:1908.06486).
https://arxiv.org/abs/1908.06486

Fetched claims:
- Directly applying standard dependence measures to temporal data **"can inflate the p-value and result in an invalid test."**
- Their procedure = temporal dependence statistic + **block permutation**.
- Claim: **asymptotically valid and universally consistent for testing independence between stationary time series.** Stationarity is explicitly required; the abstract does not spell out the mixing conditions.

**[V] Implementation: `hyppo.time_series`.** Provides `DcorrX` (cross distance correlation) and `MGCX` (cross multiscale graph correlation), both with a `max_lag` parameter testing lags j ∈ {0,…,M}; p-values via permutation; documented as assuming a strictly stationary time series.
https://hyppo.neurodata.io/user_guide/time_series

**[V] `hyppo` maintenance:** latest PyPI version **0.5.2, released 2025-05-24**, supporting Python 3.8–3.14. Actively maintained as of that release.
https://pypi.org/project/hyppo/

*Caveat I could not resolve:* the hyppo docs page says "permutation test" without stating explicitly that the shipped implementation uses **block** permutation rather than plain permutation. **Before trusting `DcorrX`/`MGCX` p-values, read the source and confirm the block scheme, and confirm the block length is settable to your session length.** If it silently plain-permutes, you have imported the exact bug you are trying to avoid. This is a concrete, checkable failure mode — check it first.

### 1.3 Verdict table for §1

| Construction | Survives serial dependence? |
|---|---|
| Plain permutation of observations | **NO** — provably invalid [V] |
| Permutation of whole sessions (block permutation) | **YES**, for stationary series, asymptotically [V] |
| i.i.d. bootstrap | **NO** |
| Session-level cluster / block bootstrap | **YES** |

---

## 2. Nonlinear dependence measures and their 2024–2026 libraries

### 2.1 Distance correlation (dCor)

**Base method.** [V] `dcor` (vnmabus) provides distance correlation, energy statistics, an independence test based on distance covariance, and an energy-distance homogeneity test. Dependencies numpy/numba/scipy/joblib, Python 3.8+.
https://github.com/vnmabus/dcor · https://dcor.readthedocs.io/

**Serial dependence verdict: the base package is i.i.d.-only.** [V] The dcor GitHub page I fetched shows **no** time-series or dependent-data facility; its independence test is a permutation test on the raw sample. Using it directly on your intraday panel is exactly the ~300x effective-n inflation you described.

**The correct time-series version exists.** [U] Davis, Matsui, Mikosch & Wan, *Applications of Distance Correlation to Time Series*, **Bernoulli 24(4A):3087–3116, 2018** — defines the **auto-distance covariance/correlation function (ADCV/ADCF)** for stationary series.
https://arxiv.org/abs/1606.05481 · https://projecteuclid.org/euclid.bj/1522051234

[U] Fokianos & Pitsillou showed the sample ADCV is a **degenerate order-2 V-statistic under the independence null**, and the reference implementation computes simultaneous confidence bands via **independent wild bootstrap**. Implemented in the **R** package `dCovTS` (`ADCF`, `ADCFplot`).
https://journal.r-project.org/articles/RJ-2016-049/RJ-2016-049.pdf · https://rdrr.io/cran/dCovTS/man/ADCF.html

**Practical verdict:** dCor is usable, but only in its ADCF/wild-bootstrap or block-permutation form. `dcor` (Python) gives you the *statistic*; `hyppo.time_series.DcorrX` or `dCovTS` (R) gives you a *valid test*. Do not use `dcor.independence` p-values on your data.

### 2.2 HSIC and kernel independence tests

This is the area with the most genuine 2026 activity, and — unusually — the activity is *directly about serial dependence*.

**[V] Li, Xu & Zhou, "Testing for Serial Independence via Auto Hilbert-Schmidt Independence Criterion," arXiv:2605.22025, submitted 21 May 2026.**
https://arxiv.org/abs/2605.22025
- Proposes **AutoHSIC**: kernel dependence between an observation and its lagged counterpart, for strictly stationary series.
- Built as a **lagged U-statistic from overlapping observations** — degeneracy under the null is central.
- **The limiting null distribution is non-pivotal**, so they develop a **wild bootstrap** for critical values and prove asymptotic validity.
- Extends to residual-based model diagnostics (parameter estimation shifts the null).
- **No code release mentioned in the abstract page.** This is a 4-month-old paper; expect to implement it yourself.

**[V] Diz-Castro, Febrero-Bande & González-Manteiga, "Kernel-based independence and mean independence tests for weakly dependent data," arXiv:2604.28104, submitted 30 April 2026.**
https://arxiv.org/abs/2604.28104
- Unified HSIC framework for independence **and mean independence**, on general topological spaces.
- Asymptotics for stationary ergodic processes under **near epoch dependence (NED)** — a weaker and more realistic condition than strong mixing for financial data.
- No bootstrap scheme and no software surfaced on the abstract page.

**[U]** Older but load-bearing foundations found via search: Chwialkowski & Gretton, *A Kernel Independence Test for Random Processes* (ICML 2014) shows that under absolute regularity / φ-mixing the HSIC null becomes an infinite weighted sum of **dependent** χ² variables (vs. independent χ² in the i.i.d. case); and Chwialkowski et al., *A Wild Bootstrap for Degenerate Kernel Tests* (NeurIPS 2014) is the canonical fix.
http://proceedings.mlr.press/v32/chwialkowski14.pdf · https://arxiv.org/abs/1408.5404

**[U] Maintained aggregated-kernel implementations** — worth knowing because kernel bandwidth choice is a hidden multiple-testing leak: `agginc` implements **HSICAggInc / MMDAggInc / KSDAggInc** (Schrab, Kim, Guedj, Gretton, NeurIPS 2022), numpy + jax.
https://github.com/antoninschrab/agginc
These aggregate over a *collection* of kernels with the multiplicity correction built in, rather than you picking a bandwidth and pretending you didn't look. **But: they are i.i.d.-null tests.** You would have to swap the permutation resampler for a block/wild bootstrap yourself.

**Serial dependence verdict:** HSIC has *the best theoretical coverage of your exact situation* of any method in this report (two 2026 papers explicitly on stationary/NED data), but **zero maintained Python implementation of the dependent-data variant**. You get theory, not a library.

### 2.3 Chatterjee's ξ — attractive, and a trap

Chatterjee's ξ is the most-hyped measure in this space. It has three separate problems for you.

**Problem 1 — low power.** [U] Shi, Drton & Han, *On the power of Chatterjee's rank correlation*, **Biometrika 109(2):317–333, 2022**: compared against Hoeffding's D, Blum–Kiefer–Rosenblatt's R, and Bergsma–Dassios–Yanagimoto's τ*, **ξ is rate sub-optimal** against local rotation and mixture alternatives.
https://academic.oup.com/biomet/article-abstract/109/2/317/6259083 · https://arxiv.org/abs/2008.11619

For a project whose effects are "smaller than their standard errors," picking the *rate-suboptimal* statistic is self-sabotage.

**Problem 2 — the bootstrap fails.** [U] *On the failure of the bootstrap for Chatterjee's rank correlation*, **Biometrika 111(3):1063–…, 2024**.
https://academic.oup.com/biomet/article-abstract/111/3/1063/7600440
[U] The repair: Dette & Kroll, *A simple bootstrap for Chatterjee's rank correlation*, **Biometrika 112(1), 2025** — an **m-out-of-n** bootstrap, consistent whenever asymptotic normality of ξ holds, proved in Kolmogorov and Wasserstein distance.
https://academic.oup.com/biomet/article-abstract/112/1/asae045/7741684 · https://arxiv.org/abs/2308.01027

**Problem 3 — no serial-dependence theory.** Nothing I found establishes ξ's null distribution under a serially dependent stationary process. All the 2024–2026 activity (bias correction arXiv:2508.09040 [U]; multivariate AC extensions arXiv:2512.07443 [U]; rank-based AC coefficient arXiv:2412.02668 [U]; lack of weak continuity arXiv:2410.11418 [U]) is i.i.d.-sample theory.

**Serial dependence verdict: NO.** ξ is a *descriptive* statistic you may compute; it is not a test you can run on this data. Put it on the AVOID list.

### 2.4 Azadkia–Chatterjee CODEC / FOCI

**[V] Montaño & Arrieta-Prieto, "Measuring General Associations in Time Series: An Adaptation and Empirical Evaluation of the CODEC Coefficient in Determining Autoregressive Dynamics," arXiv:2509.06111, 7 Sep 2025.**
https://arxiv.org/abs/2509.06111
- Adapts CODEC-FOCI to pick lags in autoregressive models; compares to Pearson/Spearman.
- Finding: CODEC beats correlation in **nonlinear and nonstationary** settings with larger samples; **Pearson wins in purely linear models**.
- Framed as **"exploratory lag identification," model-free**.
- **Critically: the abstract provides no null distribution, no formal test procedure, and no treatment of serial-dependence validity.** I fetched it specifically to check; it is not there.

**Serial dependence verdict: NO valid test.** CODEC/FOCI is a *screening/ordering* device. Used as a screen it will happily emit a lag ordering from pure noise, because it has no calibrated null. That is textbook silent failure. If you use it at all, use it only inside a session-level block-permutation wrapper that you built and validated yourself.

### 2.5 Mutual information estimators (KSG, MINE, successors)

**[U]** Current state, from search:
- **KSG** is bias-free in low dimensions but suffers increasing variance and numerical instability as dimension grows; hits the curse of dimensionality at roughly d ≳ 20.
- **MINE** does well on Gaussians but degrades when n < Θ(exp(d)).
- 2026 successors: **NMINE** (normalized MI neural estimation, arXiv:2607.27710), neural difference-of-entropies estimator (arXiv:2502.13085).
- [U] A 2026 benchmark, *Towards Diverse and Comprehensive Benchmarks for Mutual Information Estimation* (arXiv:2607.03487), reports "substantial variance in performance, interpretability, and scalability" across estimators: SMILE tracks ground truth closest but with higher variance; InfoNCE is biased downward at high MI but stable; **others diverge during training or fail to capture the correct trends.**
https://arxiv.org/abs/2607.27710 · https://arxiv.org/html/2607.03487v1

**Serial dependence verdict: NO — and worse than that.** None of these have a null distribution at all, let alone one valid under serial dependence. You get a point estimate with unknown bias and no calibrated test. For tiny effects, the bias *is* the effect. **This is the single most dangerous family in this report for your specific failure mode:** a neural MI estimator will return a confidently positive number on pure noise and there is no built-in way for it to fail loudly.

### 2.6 Copula-based measures

**[U]** `pyvinecopulib` is the maintained Python interface to the C++ `vinecopulib` (vine copula models); `VineCopulas` is a newer pure-Python open-source alternative.
https://github.com/vinecopulib/pyvinecopulib
[U] Goodness-of-fit against the **independence copula** is typically done via Rosenblatt transform + Anderson–Darling with a **bootstrap-adjusted p-value**.

**Serial dependence verdict: NO as shipped.** The standard copula GOF bootstrap is i.i.d. You would need to swap in a session-block bootstrap. Copulas also model *contemporaneous* dependence structure, which is not the question you are asking (you are asking about predictive/lagged dependence). Low priority.

---

## 3. Conditional independence testing and causal discovery for time series

### 3.1 The result that should govern this entire section

**[U] Shah & Peters, "The Hardness of Conditional Independence Testing and the Generalised Covariance Measure," Annals of Statistics 48(3):1514–1538, 2020.**
https://arxiv.org/abs/1804.07203 · https://projecteuclid.org/journals/annals-of-statistics/volume-48/issue-3/...

**The no-free-lunch theorem:** if you require Type I error control for *all* absolutely continuous triplets (X, Y, Z), then your CI test **has no power against any alternative.** Every usable CI test therefore buys power by making assumptions, and the assumptions are where it dies.

### 3.2 And the 2025 result that says it is worse in practice

**[V] He, Pogodin, Li, Deka, Gretton & Sutherland, "On the Hardness of Conditional Independence Testing In Practice," NeurIPS 2025 (arXiv:2512.14000, 16 Dec 2025).**
https://arxiv.org/abs/2512.14000
- Argues Shah–Peters **does not explain** the observed practical failures; the real hardness is implementational.
- Focuses on the **KCI test**, and shows the **Generalised Covariance Measure is nearly a special case of KCI.**
- Two concrete failure drivers: (a) **errors in conditional mean embedding estimates substantially inflate Type I error**; (b) **conditioning-kernel selection** — previously overlooked — **improves power but tends to inflate Type I error.**
- Conclusion: an irreducible tension between power and calibration; the practitioner must choose.

**Read this as: every CI test you might reach for has a knob that trades your Type I error for power, and the knob is usually set by a default you didn't choose.**

### 3.3 PCMCI+ / tigramite

**[U]** PCMCI+ (Runge et al., arXiv:2003.03685) adds contemporaneous links to PCMCI and uses **momentary conditional independence (MCI)** conditioning sets specifically to improve **autocorrelation calibration**. Under high autocorrelation it reportedly achieves higher recall, lower false positives, and faster runtime than PC. Implemented in `tigramite`.
https://github.com/jakobrunge/tigramite · https://jakobrunge.github.io/tigramite/ · https://arxiv.org/abs/2003.03685

**[U]** `tigramite` offers CI tests for continuous, discrete and mixed data, including **CMIknn** (k-NN conditional mutual information; Runge, AISTATS 2018). Documentation I found is for **version 5.2**; I could not verify a 2026 release.

**Critics:** I searched specifically for 2025–2026 critiques of PCMCI+ false-positive behaviour on financial data and **found none.** What I found was mostly favourable, including a Treasury-market application reporting PCMCI contributing "just four false positives, all interpretable" vs 10–15 for other detectors [U] (arXiv:2605.30363). **Absence of published critique is not evidence of good behaviour** — PCMCI+ is heavily used in climate science, where signals are large; your regime is the opposite.

**Serial dependence verdict: PARTIAL YES.** PCMCI+ is one of the few methods explicitly *designed* for autocorrelated data. But: (a) it inherits the §3.2 calibration/power tension from whichever CI test you plug in; (b) CMIknn on tiny effects will be dominated by estimator bias (see §2.5); (c) it answers "what is the graph," which is a much harder question than "is there anything," and therefore burns far more power.

### 3.4 The dynamic Generalised Covariance Measure — the strongest technical fit found

**[V] Wieck-Sosa, Haddad & Ramdas, "The dynamic generalized covariance measure for conditional independence testing with nonstationary time series," arXiv:2504.21647 (submitted 30 Apr 2025; latest revision 10 Aug 2026).**
https://arxiv.org/abs/2504.21647 · https://arxiv.org/html/2504.21647v3

Fetched from the full HTML:
- Billed as **"the first framework for conditional independence testing that works with a single realization of a nonstationary nonlinear process."**
- **Statistic:** max ℓp-norm of partial sums of residual products, `S_{n,p} = max_{s} ‖(1/√T) Σ_{t≤s} R̂_t‖_p`, where R̂ contains products of residuals from regressing X and Y on Z at each time t. Both ℓ∞ (sparse alternatives) and ℓ2 (dense alternatives) variants.
- **Null approximation: multiplier bootstrap** (not wild). Estimate local long-run covariance Σ̂ from lag-windowed residual products, simulate R̆ ~ N(0, Σ̂), typically N_sim = 5000, take empirical quantiles. Explicitly chosen because estimating individual local covariances is "extremely challenging in practice."
- **Dependence assumptions:** *not* mixing. Instead **Wu (2005)-style functional dependence** with polynomial decay (β̄_R > 3), a martingale-difference condition on errors, and locally stationary processes handled in a supplement. Explicitly allows heteroskedasticity and error–covariate dependence.
- **Their own indictment of the field:** *"Most conditional independence tests lack Type I error control guarantees outside the iid setting. At most, some methods provide guarantees for stationary mixing processes."*
- Also notes residual-based tests **violate Type I control when errors depend on covariates** (Shah & Peters 2020, Example 1) — which is why they use *products of residuals* rather than testing residual independence.
- **Rate double robustness:** the product of the two regression estimation errors must be `o(T^{-1/2} τ^7 D^{-3/2})` — so you can be slow on one regression if fast on the other.
- **No code repository mentioned.** They benchmark against the R `CondIndTests` package but do not point at their own release.

**Serial dependence verdict: YES — best in class.** This is the only CI test I found whose stated design target is your exact data: single realisation, nonstationary, nonlinear, heteroskedastic, serially dependent. **Cost: you implement it yourself, and the τ^7 in the rate condition tells you the constants are not friendly.**

### 3.5 Transfer entropy

**[U]** Implementations: `IDTxl` (Python, multivariate TE network inference, KSG estimators, surrogate-based statistical gating with automatic stopping) and `JIDT` (Java, multivariate KSG extension).
https://github.com/pwollstadt/IDTxl · https://arxiv.org/abs/1807.10459 · https://lizier.me/joseph/software/

**[U] Surrogate scheme matters enormously.** IDTxl's default surrogate is **permutation of replications while keeping the temporal order of samples intact**; it falls back to permuting samples in time only when there are too few replications. **That default is the correct one for you** — "replications" maps onto "sessions." The fallback is the invalid one.

**[U] The killer critique:** *serial autocorrelation in financial returns induces spurious transfer entropy, because lagged self-similarity mimics genuine information flow.* Plus documented small-sample bias in entropy estimation. The standard repair is **effective transfer entropy** — estimate the shuffled-data bias and subtract it (`RTransferEntropy`, R).
https://www.sciencedirect.com/science/article/pii/S2352711019300779

**Serial dependence verdict: NO in default form; MARGINAL with replication-surrogates + effective-TE bias correction.** Given that your effects are sub-standard-error, a method whose known failure mode is "autocorrelation manufactures the signal" is the wrong tool.

### 3.6 Convergent cross mapping

**[U]** Documented problems, from 2024–2025 sources:
- **"Unacceptable false positives for some dynamics"**; spurious dependencies from coincidental similarity.
- **Observational noise leads to spurious results** — and financial data is nearly all noise.
- **Cannot distinguish direct from mediated causality**: for x ⇒ y ⇒ z, CCM may report x ⇒ z as direct.
- 2025: CCM **misreads bidirectional coupling as unidirectional** when the attractor is symmetric (Chaos 35(10):103147).
https://arxiv.org/abs/2502.03802 · https://pubs.aip.org/aip/cha/article/35/10/103147/3369983/

**Serial dependence verdict: NO.** CCM assumes a deterministic low-dimensional attractor. Intraday equity returns are dominated by stochastic noise. **AVOID.**

### 3.7 GCM / COMETs — the one you can actually pip-install

**[V] `pycomets`** (https://github.com/shimenghuang/pycomets) — Python implementation with `GCM()` and `PCM()` classes and a `.test()` method; plug in any sklearn-style regressor (random forest, linear, kernel ridge, XGBoost). **39 commits total; no last-commit date visible; no stated assumptions about i.i.d. vs dependent data.**
[U] The R sibling `comets` (https://github.com/LucasKook/comets) reportedly covers GCM, PCM, **wGCM, kGCM**; and CRAN has `GeneralisedCovarianceMeasure` (doc dated 21 July 2025) and `weightedGCM`.

**Serial dependence verdict: NO as shipped — but it is the natural scaffold for §3.4.** The dGCM is structurally a GCM with a multiplier bootstrap and long-run covariance. Building dGCM on top of `pycomets`'s regression plumbing is a much smaller job than building it from scratch.

### 3.8 causal-learn

**[U]** `causal-learn` (py-why) ships Fisher-z, missing-value Fisher-z, Chi-square, G-square, and **KCI**. Version **0.1.4.7 released 2026-05-18** — actively maintained.
https://pypi.org/project/causal-learn/ · https://causal-learn.readthedocs.io/en/latest/independence_tests_index/index.html
**All of these assume i.i.d. samples.** KCI in particular is the test §3.2 shows is calibration-fragile. Useful for a *sanity/positive-control harness on synthetic i.i.d. data*, not for your real data.

---

## 4. Multiple-testing-safe discovery — the most important section

### 4.1 e-values and e-BH: the right accounting substrate for a research programme

This is, in my view, **the single best structural fit to your actual problem** — which is not "which test" but "how do I keep score across a programme that has already burned nine hypotheses and cannot get more data."

**[U] Wang & Ramdas, "False discovery rate control with e-values," JRSS-B 84(3):822–852, 2022.**
https://academic.oup.com/jrsssb/article/84/3/822/7056146 · https://arxiv.org/abs/2009.02824

**The property that matters to you: e-BH controls FDR at level q under *arbitrary dependence* between the e-values, with no correction.** Standard BH requires PRDS; e-BH requires nothing. Given that your hypotheses are all computed on the *same* sessions and are therefore massively dependent, this is not a nicety — it is the difference between a valid and an invalid ledger.

Two more properties that fit a long-running programme:
- **e-values merge by averaging and (for independent ones) by multiplication.** You can accumulate evidence across studies rather than re-testing.
- **[U] Anytime-validity.** Adaptively stopped e-processes are still e-values, so you can apply e-BH at *every* time step and continuously monitor. The 2025 refinement — *Anytime-valid FDR control with the stopped e-BH procedure* (arXiv:2502.08539, also in Statistics & Probability Letters) — identifies the condition under which local e-processes are global w.r.t. a common filtration, which is what makes stopping at a data-dependent time legitimate.
https://arxiv.org/abs/2502.08539

**Why this matters specifically for your frozen forward test.** A frozen prospective test with checkpoints at 50/100/200/377 is *exactly* the setting where p-values leak (peeking) and e-processes do not. Re-expressing the live-lab checkpoints as an e-process would let you look whenever you want without inflating error — and would let you *stop early for futility* with a defensible number.

**[U] Related 2024–2026 refinements:** boosting e-BH via conditional calibration (arXiv:2404.17562); choosing the nominal level post-hoc with knockoffs using e-values (arXiv:2511.11166); the e-partitioning principle (arXiv:2504.15946).

**Serial dependence verdict: YES — arbitrary dependence *across hypotheses* is exactly what e-BH handles.** Note the important subtlety: **e-BH's dependence-robustness is across hypotheses, not within a time series.** Each individual e-value must still be a valid e-value under your serially dependent null. You get that from a session-clustered/blocked construction, or from universal inference (§4.5).

### 4.2 Knockoffs for time series (TSKI)

**[V] Chi, Fan, Ing & Lv, "High-Dimensional Knockoffs Inference for Time Series Data," Journal of the American Statistical Association, 2025** (arXiv:2112.09851; JASA 120(551); PubMed 40857500).
https://arxiv.org/abs/2112.09851 · https://www.tandfonline.com/doi/full/10.1080/01621459.2024.2431344

Fetched claims:
- **TSKI** extends model-X knockoffs to time series, "exploiting **subsampling and e-values** to address the difficulty caused by the serial dependence."
- Relaxes the model-X requirement of a **known** covariate distribution — which is untenable for time series — via robust knockoffs.
- Establishes **sufficient conditions for asymptotic FDR control** (asymptotic, not finite-sample).
- **No code availability indicated on the abstract page.**

**[U] Other time-series knockoff work:** DeepLINK-T (LSTM + knockoffs, arXiv:2404.04317); IPAD (arXiv:1809.05032).

**[U] Knockoff critiques you should know before adopting:**
- **Instability from algorithmic randomness.** "Two different runs of the knockoff filter may result in wildly different rejection sets" — a serious reproducibility problem.
- **Derandomized knockoffs** (Ren & Barber 2023) fix this by running the filter repeatedly, converting to relaxed e-values, and applying e-BH to the average.
- **Remaining unsolved problems:** how to choose the derandomization parameter q_kn is unknown and it **directly controls power**; and **power can be severely degraded by correlation among covariates.**
https://arxiv.org/abs/2512.17401 · https://arxiv.org/pdf/2502.18750

**Serial dependence verdict: YES in principle (this is the whole point of TSKI), but with two large practical caveats.** (1) FDR control is *asymptotic*, and your effective n is hundreds of sessions, not thousands. (2) The correlation-kills-power critique bites hard: your features are almost certainly heavily correlated with each other. Knockoffs answer "which of these 200 features matter," which is *not* your question ("does anything matter at all") and is far more power-hungry.

### 4.3 Bailey & López de Prado: DSR, PBO, MinBTL, False Strategy Theorem

**[U] Deflated Sharpe Ratio** (Bailey & López de Prado, *Journal of Portfolio Management* 40(5):94, 2014): corrects for (a) selection bias under multiple testing and (b) non-normal returns. Accounts for the variance of Sharpe estimates, the number of trials, and their **effective independence, typically estimated by clustering** the trials.
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 · https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf

**[U] False Strategy Theorem** (López de Prado & Bailey, SSRN 3221798): the maximum Sharpe over an unknown number of trials is **right-unbounded** — with enough trials there is no ceiling on the best backtest you can find by chance alone.
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3221798

**[U] Minimum Backtest Length (MinBTL):** after only **~7 strategy configurations**, a researcher is expected to find at least one 2-year backtest with annualized Sharpe > 1 when the true out-of-sample Sharpe is 0.
https://stefan-jansen.github.io/machine-learning-for-trading/08_ml4t_workflow/01_multiple_testing/

**Read that number against your programme.** Nine pre-registered nulls is *fewer* trials than the MinBTL threshold — which is actually a point in your favour: your discipline has been better than the field's. But it also means the DSR haircut for N=9 is small, and **DSR will not rescue or condemn anything on its own.**

**[U] Critiques of DSR** (found, but thinner than I would like — I could not locate a substantial peer-reviewed teardown):
- "DSR is still **model-based**: it relies on a normal approximation corrected by skew and kurtosis, relies on an **assumed form for the distribution of SRs under the null**, and **requires a choice of N**."
- The choice of N (number of effectively independent trials) is the whole ballgame and is estimated by clustering — i.e., **by a judgment call the analyst makes after seeing the data.**
- **[U]** López de Prado, Lipton & Zoonekynd, *How to Use the Sharpe Ratio* (SSRN 5520741, 2025) appears to be a recent methodological update. **I could not fetch it — SSRN returned HTTP 403.** Worth retrieving manually.

**[U] CPCV (Combinatorial Purged Cross-Validation):** reported to beat K-Fold, Purged K-Fold and Walk-Forward on both PBO and DSR in a synthetic controlled environment (Arian, Norouzi & Seco, *Knowledge-Based Systems*, 2024). Caveats found: **high computational cost**, and **walk-forward remains the industry standard for realistic trading simulation.**
https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110

**Serial dependence verdict: PARTIAL.** DSR handles multiple testing and non-normality; it does **not** by itself handle serial dependence in the return stream, and its Sharpe standard error assumes something about the autocorrelation structure. Its real weakness for you is **N is a free parameter you choose**, which is the opposite of a loud null.

### 4.4 White's Reality Check / Hansen's SPA / StepM / MCS — the underrated answer

This family is, for your problem, dramatically more useful than knockoffs, and it is **already installed**.

**[V] `arch` 8.0.0, released 2025-10-21**, Production/Stable, Python 3.10–3.14, one maintainer (bashtage), actively maintained.
https://pypi.org/project/arch/

**[V] `arch` multiple comparison procedures** (docs for 8.0.0):
- `SPA(bm_losses, model_losses, seed=..., reps=250)` — Hansen's Test of Superior Predictive Ability, a.k.a. Reality Check / bootstrap data snooper. Null: **no model is better than the benchmark.**
- `StepM(bm_losses, model_losses_df)` — Romano–Wolf stepwise: returns *the set* of models better than the benchmark rather than a yes/no.
- `MCS(losses, size=0.10)` — Model Confidence Set: the set of models statistically indistinguishable from the best, with FWER control.
- All take **losses** (smaller = better).
https://bashtage.github.io/arch/multiple-comparison/multiple-comparison_examples.html

**[V] Caveat found by reading the docs:** the arch multiple-comparison examples **do not discuss serial dependence and generate their example data with `randn()`.** The block-bootstrap machinery is there (the classes accept a `block_size`), but the documentation will not remind you to set it. **If you leave block_size at its default, you have silently reintroduced the i.i.d. assumption.** Concretely: set the block to one session.

**[U] Provenance:** White's Reality Check (1997/2000); Hansen's SPA is "more powerful and less sensitive to poor and irrelevant alternatives than the Reality Check"; Romano–Wolf StepM extends SPA stepwise; Sullivan–Timmermann–White applied RC to 100 years of DJIA with a universe of trading rules. Hsu, Hsu & Kuan extended to a stepwise-SPA.
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=264569 · https://www.kevinsheppard.com/files/teaching/mfe/advanced-econometrics/Sullivan_Timmermann_White.pdf · https://homepage.ntu.edu.tw/~ckuan/pdf/Step-SPA-20090720.pdf

**Serial dependence verdict: YES — this family was *built* for it.** SPA/StepM/MCS use a **stationary or circular block bootstrap internally**, which is precisely the correct null for serially dependent loss series. This is the most mature, most tested, most maintained, and most directly-fit-for-purpose tool in the whole report, and it is the one nobody in the ML dependence-testing literature will tell you about.

**Reframing that makes SPA answer your actual question.** "Is there ANY exploitable dependence at all?" = "**Does the best of my K candidate signals beat a zero-signal benchmark, after correctly accounting for the fact that I looked at K of them and my losses are serially dependent?**" That is *literally the SPA null hypothesis*. You can enter all nine dead hypotheses plus any new ones as columns and get one number.

### 4.5 Universal inference — valid e-values with almost no assumptions

**[U] Wasserman, Ramdas & Balakrishnan, "Universal Inference," PNAS 2020** (arXiv:1912.11436). The **split likelihood ratio test (split LRT)** statistic is an **e-value**, giving a finite-sample-valid test **under virtually no regularity conditions**.
https://arxiv.org/abs/1912.11436

**[U] Known cost:** Strieder & Drton (2022) and Tse & Davison (2022) provide empirical evidence that the split LRT **may be highly conservative** — i.e., it trades power for validity. Countervailing: Shi & Drton (2025) find universal inference achieves the same detection rate as classical methods in Gaussian mixture models, contradicting the "SAVI always loses power" folklore.
https://arxiv.org/abs/2407.19361

**Serial dependence verdict: YES, with the right split.** Split by *session*, and the two halves are (approximately) independent. Universal inference gives you an e-value you can feed straight into e-BH, and it has a genuinely loud null. **But "highly conservative" is a real problem when your effects are sub-standard-error.** Use it as a confirmatory device, never as a screen.

### 4.6 Conformal inference for time series

**[U]** The core problem: conformal coverage guarantees rest on **exchangeability**, which "is fundamentally violated in time series data." Serial dependence, volatility clustering and drift all break it, and naive split conformal can deliver coverage substantially off nominal.
https://arxiv.org/abs/2511.13608 (*A Gentle Introduction to Conformal Time Series Forecasting*)

**[U] Current landscape (2025–2026):**
- **EnbPI** (Xu & Xie, ICML 2021 / IEEE TPAMI) — ensemble batch prediction intervals. https://github.com/hamrel-cxu/EnbPI
- **ACI** (Gibbs & Candès 2021), **AgACI** (Zaffran et al., ICML 2022). https://proceedings.mlr.press/v162/zaffran22a/zaffran22a.pdf
- **SPCI** (Sequential Predictive Conformal Inference, 2023) — time-adaptive residual quantile re-estimation; reported to improve on EnbPI.
- **CPTC** (NeurIPS 2025) — validity without stationarity assumptions. https://github.com/Rose-STL-Lab/CPTC
- **BC-ACI** (arXiv:2604.13253, 2026) — identifies a structural limitation: threshold-only methods adapt the quantile but always centre intervals on the point prediction, so they **cannot correct location bias** and must widen intervals proportionally to it.
- **Rolling-Origin Conformal Prediction under Local Stationarity and Weak Dependence** (arXiv:2605.08422, 2026) — finite-sample guarantees under weak dependence.
- **Benchmark:** *Conformal Prediction Algorithms for Time Series Forecasting: Methods and Benchmarking* (arXiv:2601.18509, 2026).

**[U] Implementation:** `MAPIE` (scikit-learn-contrib), `mapie.regression.TimeSeriesRegressor` with EnbPI; docs current at **1.4.1**; v1 released 2025. Notably, **MAPIE in 2026 added exchangeability tests "to help users verify when MAPIE can be legitimately applied."**
https://mapie.readthedocs.io/en/stable/generated/mapie.regression.TimeSeriesRegressor.html · https://github.com/scikit-learn-contrib/MAPIE

**Serial dependence verdict: NO for your question — right tool, wrong question.** Conformal gives you **calibrated prediction intervals**, not a **test of whether dependence exists**. A conformal interval that always contains zero is weak evidence of nothing, not evidence of no signal. **Do not use conformal as a discovery screen.** The MAPIE exchangeability tests, however, are a genuinely useful *diagnostic*.

---

## 5. Bootstrap and resampling under serial dependence

### 5.1 The most important practical point in this report

**You have a natural cluster: the session.** That means you probably **do not need automatic block-length selection at all.** Resample *whole sessions with replacement* (cluster bootstrap = block bootstrap with block = session, blocks aligned to natural boundaries). This is:
- more defensible than any data-driven block length,
- immune to the ~0.9 within-session autocorrelation you described,
- trivially explainable in a write-up,
- and gives you an honest effective n = **number of sessions**, not number of trades.

Politis–White block-length selection is for when there is **no** natural block. Reaching for it when you have sessions is added complexity that buys nothing — and your own project memory records that added complexity has consistently made things worse.

### 5.2 The methods, and their maintained implementations

**[V] `arch` (v8.0.0, 2025-10-21)** provides `IIDBootstrap`, `StationaryBootstrap`, `CircularBlockBootstrap`, `MovingBlockBootstrap`, `IndependentSamplesBootstrap`.
https://arch.readthedocs.io/en/latest/bootstrap/bootstrap.html · https://pypi.org/project/arch/

**[V] `arch.bootstrap.optimal_block_length(x) -> DataFrame`** returns two columns: **`b_sb`** (stationary bootstrap) and **`b_cb`** (circular bootstrap). Implements **Politis & White (2004)** with the **Patton, Politis & White (2009) correction**. Algorithm: find tuning parameter *m* as the first lag where consecutive autocorrelations fall inside a conservative band; compute autocovariances; then `b_i^OPT = (2g²/d_i × n)^(1/3)` with d_i = 2 (stationary) or 4/3 (circular).
**Documented caveat:** autocovariances are computed using maximum sample length rather than common sampling length, so results **may differ from Patton's reference MATLAB implementation.** Accepts 1D/2D; processes columns independently.
https://arch.readthedocs.io/en/latest/bootstrap/generated/arch.bootstrap.optimal_block_length.html

**[U] `recombinator`** (InvestmentSystems) — iid bootstrap, **moving block, circular block, stationary, and tapered block bootstrap**, plus optimal block-length selection. Latest PyPI **0.0.6.1**. Version number and the absence of visible recent releases suggest **lower maintenance confidence than `arch`**.
https://github.com/InvestmentSystems/recombinator

**[U] Wild cluster bootstrap — `wildboottest`** (py-econometrics): implements the fast algorithms of Roodman et al. (2019) and MacKinnon, Nielsen & Webb (2022), including WCR13/31/33 and WCU13/31/33 variants, CRV1/CRV3 and CRV3-jackknife, non-clustered wild bootstrap (Wu 1986), and the subcluster bootstrap (MacKinnon & Webb 2018). **Limitation: computes bootstrapped p-values only, no confidence intervals.**
https://github.com/py-econometrics/wildboottest

**This is the direct match for "cluster by session."** If your estimand is a regression coefficient (e.g. "does feature X predict return, clustering by session"), `wildboottest` with `cluster = session_id` is the standard, well-tested econometric answer — and it is specifically designed for the **few-clusters** regime, which matters since hundreds of sessions is "few" by wild-bootstrap standards.

**[U] Wild bootstrap for degenerate kernel statistics** (Chwialkowski et al., NeurIPS 2014) is the right resampler for HSIC/dCor-type degenerate U/V-statistics under dependence — a *different* wild bootstrap from the regression one above. Do not confuse them.
https://arxiv.org/abs/1408.5404

### 5.3 Block-length guidance, current state

**[U]** The literature has not moved much: Hall, Horowitz & Jing (1995) subsampling cross-validation; Bühlmann & Künsch; Lahiri; **Politis & White (2004)** spectral-density plug-in; **Patton, Politis & White (2009)** correction. The R package `blocklength` implements `hhj()` and `pwsd()`.
https://mathweb.ucsd.edu/~politis/SBblock-revER.pdf · https://alec-stashevsky.github.io/blocklength/

**Current practical guidance, synthesised:**
1. If a natural block exists (session, day, event), **use it**. Do not optimise.
2. Otherwise use `arch.bootstrap.optimal_block_length` as a starting point.
3. **Always report sensitivity**: rerun at 0.5×, 1×, 2× the chosen block length. If your conclusion flips, you never had one.
4. Prefer the **stationary bootstrap** (random geometric block lengths) over fixed moving blocks — it is less sensitive to a misspecified block length by construction.

---

## 6. "Is there any signal at all" — omnibus screening, 2025–2026

### 6.1 The family the ML literature won't mention: martingale difference hypothesis tests

**This is the closest thing that exists to a purpose-built answer to your question**, it comes from econometrics rather than ML, and it has been mature for twenty years.

The **martingale difference hypothesis (MDH)** — E[r_t | past information] = μ — is *literally* "there is no exploitable directional dependence in this series." An MDH test is an omnibus test of exactly the hypothesis your eight nulls have been attacking one hand-crafted feature at a time.

**[U] Escanciano & Velasco, "Generalized spectral tests for the martingale difference hypothesis," Journal of Econometrics (2006).**
https://www.sciencedirect.com/science/article/abs/pii/S0304407605001417
- Tests based on **second-order statistics are inconsistent against uncorrelated-but-dependent alternatives** — i.e., ordinary autocorrelation tests are blind to exactly the nonlinear predictability you would care about.
- Their fix uses characteristic-function-based dependence measures and a **generalized spectral distribution function**, which **avoids bandwidth choice** (one fewer researcher-degree-of-freedom).
- Considers **dependence at all lags simultaneously**; consistent against general pairwise nonparametric Pitman local alternatives at the parametric rate.

**[U] Escanciano & Velasco, "Testing the martingale difference hypothesis using integrated regression functions," CSDA (2006).**
https://www.sciencedirect.com/science/article/abs/pii/S0167947306002416

**[V] Rolla, "Testing the martingale difference hypothesis using martingale difference divergence function," arXiv:2306.13963 (June 2023, rev. Nov 2023).**
https://arxiv.org/abs/2306.13963
- Builds a **Ljung–Box-type statistic by summing sample MDD over a finite number of lags**.
- MDD = 0 **iff** conditional mean independence — so it detects **nonlinear** serial dependence that autocorrelation misses.
- Establishes the asymptotic null distribution under suitable conditions.
- Includes an **S&P 500 empirical application**.
- Abstract does not mention wild bootstrap or explicit conditional-heteroskedasticity treatment; **no code indicated.**

**[U] MDD provenance:** Shao & Zhang (2014) — MDD(Y|X)=0 iff E(Y|X)=E(Y) a.s. Since the limiting null is **non-pivotal**, the standard practice is **wild bootstrap** for critical values, with proven bootstrap consistency.
https://academic.oup.com/biomet/article-abstract/107/2/331/5716269 · https://arxiv.org/abs/1805.06640

**[U] Automatic Variance Ratio + wild bootstrap** — the simplest credible omnibus screen. Choi (1999) automatic VR (data-driven lag selection), Kim (2009) wild bootstrap version, **"no size distortion in small samples."** Implemented in **R** `vrtest` (`Auto.VR`, `AutoBoot.test`). **No Python port found** — you would port ~50 lines.
https://cran.r-project.org/web/packages/vrtest/vrtest.pdf · https://rdrr.io/cran/vrtest/man/AutoBoot.test.html

**Serial dependence verdict: YES — this whole family is designed for it.** MDH tests assume a stationary series with serial dependence *as the object of study*, not as a nuisance. They have well-defined, loud nulls. And an MDH test is cheap: **it costs you one hypothesis, not nine.**

### 6.2 Foundation models as dependence detectors

**[U] TabPFN-TS** (Hoo et al., arXiv:2501.02945) extends TabPFN-v2 to forecasting via lightweight temporal featurisation; 11M parameters; SOTA on covariate-informed forecasting, competitive univariate, on GIFT-Eval and fev-bench. `tabpfn-time-series` on PyPI; as of v1.1.0 (Dec 2026) the default checkpoint is TabPFN-TS-3.
https://arxiv.org/abs/2501.02945 · https://pypi.org/project/tabpfn-time-series/ · https://github.com/PriorLabs/tabpfn

**What I looked for and did not find:** any published use of TabPFN **as a calibrated independence/dependence test.** I searched specifically. There is adjacent work (uLEAD-TabPFN for dependency-based anomaly detection, arXiv:2604.20255; Drift-Resilient TabPFN) but **nothing that turns TabPFN into a hypothesis test with a null distribution.**

**Serial dependence verdict: NO.** TabPFN is a *predictor*, not a *test*. Using "TabPFN beat the baseline" as evidence of dependence is a backtest, with all the same multiple-testing and dependence problems, plus a pretrained prior over structural causal models that you did not choose and cannot audit. Its prior is fitted on synthetic SCM-generated data — there is no reason to think that prior is calibrated for financial noise. **This is the highest-risk item on the "attractive-sounding" list.**

### 6.3 Learned test statistics — "deep testing"

**[V] Geenens, Lafaye de Micheaux & Zou, "Deep-testing: the case of dependence detection," arXiv:2604.26558, submitted 29 April 2026.**
https://arxiv.org/abs/2604.26558
- The test statistic is **a classification map learned by a deep neural network from simulated data satisfying the null and alternative.**
- Headline result: **highest overall power against nineteen competing methods across a broad range of complex dependence structures.**
- **What the abstract page does NOT say, and I checked:** no calibration methodology, **no statement about serial dependence or non-i.i.d. data**, no code availability, no numeric comparison to dcor/HSIC specifically.

**Serial dependence verdict: UNKNOWN, presume NO.** The framing ("simulate under the null") is actually promising for you — *you can simulate your own null*, e.g. session-block-shuffled data, and train the discriminator on that. That converts it into a legitimate, self-calibrating tool. But it is a 5-month-old paper with no code, and a learned statistic is precisely the kind of thing that fails silently. **Interesting; not for this year.**

### 6.4 Conformal test martingales / e-processes for exchangeability

This is a genuinely underused idea for your live forward test.

**[U]** Conformal test martingales (Vovk et al. 2003; Vovk, *Conformal test martingales for change-point detection*, PMLR 152) are non-negative processes starting at 1 that are martingales **under the exchangeability null**. They aggregate conformal p-values into an online test, generalising CUSUM / Shiryaev–Roberts.
https://proceedings.mlr.press/v152/vovk21b/vovk21b.pdf · https://arxiv.org/abs/2102.10439

**[U]** Vovk et al., **"Conformal e-testing," Pattern Recognition 168 (2025)**.
https://www.sciencedirect.com/science/article/pii/S0031320325005011

**[V] Szabadváry, "Betting on Moments: Legendre Jumper Martingales for Online Exchangeability Testing," arXiv:2606.20859 (18 June 2026, rev. 12 July 2026).**
https://arxiv.org/abs/2606.20859
- Betting functions built from **shifted Legendre polynomials**, betting against the uniformity of conformal p-values.
- Three variants: Simple (detects mean/variance/skew/higher moments), Product (exponential cost — the "jumping tax"), **Variational (mean-field approximation, constant time per step, minimal performance loss).**
- **Guaranteed false-alarm-rate control via test martingales (e-processes).**
- No code mentioned.

**[U]** Also 2026: *Anytime-Valid Distribution Shift Detection via Predictive Rank Martingales* (arXiv:2609.00536); *Distribution-free changepoint localization after sequential change detection* (arXiv:2606.01256).

**Serial dependence verdict: PARTIAL — valid if your exchangeable unit is the session.** Feed the martingale one nonconformity score **per session**, not per trade, and the exchangeability null becomes plausible. Then the martingale's value **is** an e-value you can read off at any time without penalty. **This is a near-perfect fit for the frozen forward test:** it gives you a running number that is honest at every checkpoint, and it has an unmistakably loud null (the martingale is a fair bet; it drifts to zero if there is nothing there).

### 6.5 The benchmark nobody should skip

**[V] Jian Ma, "Evaluating Independence and Conditional Independence Measures," arXiv:2205.07253 (15 May 2022).**
https://arxiv.org/abs/2205.07253
- Benchmarks **16 independence and 16 CI measures** on simulated (normal, copula) and real data.
- **Most measures behaved well on simulated data with correct monotonicity, but performance diverged sharply on complex real data; only a few showed reliable real-world performance.**
- Recommends **copula entropy (CE)** for both roles, citing a distribution-free definition and a consistent nonparametric estimator.
- **Explicitly contains no commentary on time series / dependent data** — I checked for this.

The transferable lesson: **simulation power curves in dependence-measure papers do not survive contact with real data.** Whatever you pick, validate it on *your* data's null (session-block shuffles), not on the paper's Gaussians.

---

## 7. Ranked shortlist — what to actually implement

Ordered by (fit to your problem) × (survives serial dependence) × (maturity) ÷ (implementation cost).

### Tier 0 — Do this before writing any test code

**0. Compute the minimum detectable effect (MDE) at your true effective n.**
No URL required; this is arithmetic. Effective n ≈ number of **sessions**, not trades. Simulate: inject a synthetic edge of known size into your real data (preserving all other structure), then run your intended test at effect sizes spanning 0.05–0.5 SE. Plot power vs effect size.
**If the MDE at 80% power exceeds any plausible real edge, stop. Every test below is then guaranteed to return a null that means nothing**, and running them would be a tenth silent failure. This is the sanity check that can fail, and it should be the first one you run.

### Tier 1 — Implement these

**1. Hansen SPA / Romano–Wolf StepM / MCS via `arch`, with block = session.** [V]
`arch` 8.0.0, released 2025-10-21, actively maintained. https://pypi.org/project/arch/ · https://bashtage.github.io/arch/multiple-comparison/multiple-comparison_examples.html
- *Why #1:* the SPA null **is** your question ("does the best of K candidates beat zero after honest accounting?"), it handles serial dependence natively via block bootstrap, it is mature and pip-installed, and it costs **one** hypothesis for **all nine** dead ideas plus any new ones.
- *Loud null:* yes — the bootstrap distribution is explicit and plottable.
- *Sanity check that can fail:* enter a synthetic column with a known injected edge; SPA must find it. Enter only pure-noise columns; SPA must not.
- **⚠ Set `block_size` explicitly to your session length. The docs will not remind you** (the examples use `randn()`). [V]

**2. Martingale difference hypothesis test (MDD-Ljung-Box, or automatic VR with wild bootstrap).** [V for MDD paper, U for VR]
https://arxiv.org/abs/2306.13963 · https://cran.r-project.org/web/packages/vrtest/vrtest.pdf
- *Why:* the single cheapest honest answer to "is there ANY directional predictability." Built for serially dependent stationary data. MDD detects **nonlinear** conditional-mean dependence that autocorrelation is blind to.
- *Cost:* AVR is ~50 lines to port from R. MDD is a moderate implementation with a wild bootstrap.
- *Loud null:* yes, wild-bootstrap distribution.

**3. Session-level cluster/wild-cluster bootstrap as the universal inference substrate — `wildboottest`.** [U]
https://github.com/py-econometrics/wildboottest
- *Why:* not a discovery method, but **the correct plumbing under everything else.** Roodman et al. (2019) / MacKinnon–Nielsen–Webb (2022) algorithms, designed for the few-clusters regime.
- *Limitation:* p-values only, no CIs.

**4. Re-express the frozen live forward test as an e-process (conformal test martingale, session-level).** [U/V]
https://proceedings.mlr.press/v152/vovk21b/vovk21b.pdf · https://arxiv.org/abs/2606.20859
- *Why:* you have a prospective test with fixed checkpoints at 50/100/200/377. A p-value checked four times is not a p-value. A test martingale is honest at **every** step, permits early futility stopping, and its value is directly an e-value.
- *Loud null:* extremely — a fair bet's wealth process visibly fails to grow.
- *Constraint:* one nonconformity score **per session**.

**5. e-BH as the programme-level ledger.** [U]
https://academic.oup.com/jrsssb/article/84/3/822/7056146 · https://arxiv.org/abs/2502.08539
- *Why:* **FDR control under arbitrary dependence across hypotheses, with no correction** — which is exactly right when all nine of your hypotheses live on the same sessions. Converts "how many hypotheses have we burned?" from a vibe into an arithmetic ledger. Stopped-e-BH (2025) makes checkpoint-time application legitimate.
- *Caveat:* each individual e-value must still be valid under *your* serially dependent null. e-BH fixes cross-hypothesis dependence, not within-series dependence.

### Tier 2 — Worth it if Tier 1 shows anything

**6. Block-permutation dCor / MGC — `hyppo.time_series.DcorrX` / `MGCX`.** [V]
https://hyppo.neurodata.io/user_guide/time_series · https://arxiv.org/abs/1908.06486 (TMLR 2024)
- *Why:* asymptotically valid and universally consistent for stationary series; already implemented; nonlinear.
- **⚠ VERIFY FIRST: read the source and confirm it block-permutes rather than plain-permutes, and that block length is settable.** If it plain-permutes, discard immediately.

**7. Auto-distance correlation function (ADCF) with independent wild bootstrap — R `dCovTS`.** [U]
https://journal.r-project.org/articles/RJ-2016-049/RJ-2016-049.pdf
- *Why:* the properly-derived time-series distance correlation with simultaneous confidence bands. Good as a **visual** omnibus screen across lags.
- *Cost:* R, or port.

**8. Universal inference (split LRT) on a session-level split, as a confirmatory e-value.** [U]
https://arxiv.org/abs/1912.11436
- *Why:* finite-sample-valid under almost no assumptions; produces an e-value directly.
- *Cost:* documented as **possibly highly conservative** — confirmatory only, never a screen.

### Tier 3 — Real, but expensive; only if you have engineering time

**9. dynamic Generalised Covariance Measure (dGCM).** [V]
https://arxiv.org/abs/2504.21647
- *Why:* the only CI test whose stated design target is single-realisation, nonstationary, nonlinear, heteroskedastic, serially dependent data. Multiplier bootstrap null. Rate-double-robust.
- *Cost:* **no code released**; build on `pycomets` (GCM/PCM scaffolding) [V] https://github.com/shimenghuang/pycomets.

**10. AutoHSIC with wild bootstrap.** [V]
https://arxiv.org/abs/2605.22025
- *Why:* purpose-built kernel test for serial independence in strictly stationary series, wild bootstrap with proved asymptotic validity.
- *Cost:* four months old, no code.

**11. TSKI (time-series knockoffs).** [V]
https://arxiv.org/abs/2112.09851 (JASA 2025)
- *Why:* the only FDR-controlling variable-selection method that confronts serial dependence head-on, using subsampling + e-values.
- *Cost:* no code found; FDR guarantee is only **asymptotic**; power degrades badly with correlated covariates [U]; answers a harder question than yours.

---

## 8. AVOID — attractive-sounding methods and why they fail here

| Method | Why it is attractive | Why to avoid it **here** | Source |
|---|---|---|---|
| **Plain permutation of observations** (incl. your current shuffle, if applied per-observation) | Simple, assumption-free-seeming | **Provably invalid** under serial dependence; Type 1 error not level α even asymptotically; large directional (Type 3) errors. **Fix is trivial: permute whole sessions.** | [V] https://arxiv.org/abs/2009.03170 |
| **Chatterjee's ξ** | Fashionable; simple; 0 iff independent; asymptotically normal under independence | **Rate sub-optimal vs D, R, τ\*** (Biometrika 2022) — you cannot afford lost power; **the standard bootstrap provably fails** (Biometrika 2024, needs m-out-of-n); and **no serial-dependence theory exists.** | [U] https://academic.oup.com/biomet/article-abstract/109/2/317/6259083 · https://academic.oup.com/biomet/article-abstract/111/3/1063/7600440 |
| **CODEC / FOCI as a screen** | Model-free, nonlinear, conditional, beats correlation on nonlinear data | **No null distribution and no test procedure** in the time-series adaptation — I checked the paper specifically. Will emit a confident lag ordering from pure noise. Textbook silent failure. | [V] https://arxiv.org/abs/2509.06111 |
| **Neural MI estimators (MINE, SMILE, InfoNCE, NMINE)** | "Measures all dependence"; modern | **No null distribution at all.** 2026 benchmarks report substantial variance, divergence during training, and estimators that "fail to capture the correct trends." Bias ≥ your effect size. Cannot fail loudly. | [U] https://arxiv.org/html/2607.03487v1 · https://arxiv.org/abs/2607.27710 |
| **KSG mutual information on high-dim features** | Nonparametric, classic | Increasing variance and numerical instability with dimension; curse of dimensionality at d ≳ 20; no calibrated null under serial dependence. | [U] search-verified |
| **Convergent cross mapping** | Intuitive; "detects causality in nonlinear systems" | **Unacceptable false positives for some dynamics**; **spurious results under observational noise** (your regime is nearly all noise); cannot separate direct from mediated causation; 2025 result shows it misreads bidirectional coupling as unidirectional under attractor symmetry. Assumes a deterministic low-dim attractor that intraday returns do not have. | [U] https://arxiv.org/abs/2502.03802 · https://pubs.aip.org/aip/cha/article/35/10/103147/3369983/ |
| **Transfer entropy (default settings)** | Directional, nonlinear, information-theoretic | **Serial autocorrelation manufactures spurious transfer entropy** — lagged self-similarity mimics information flow. Plus documented small-sample entropy bias. Only marginally salvageable with replication-surrogates + effective-TE bias subtraction. | [U] https://www.sciencedirect.com/science/article/pii/S2352711019300779 |
| **KCI / causal-learn CI tests on real data** | Nonparametric, in a maintained package | **NeurIPS 2025 result:** conditional-mean-embedding estimation error substantially inflates Type I error, and conditioning-kernel choice trades calibration for power. Plus Shah–Peters: no CI test has power against all alternatives. Fine as a synthetic positive control; not for real data. | [V] https://arxiv.org/abs/2512.14000 · [U] https://arxiv.org/abs/1804.07203 |
| **TabPFN / foundation models as a dependence detector** | Zero-shot, SOTA, no training | **No published calibration as a hypothesis test.** It is a predictor; "it beat the baseline" is a backtest with all the same problems plus an unauditable synthetic-SCM prior. Highest silent-failure risk in this report. | [V] search for such a use returned nothing; https://arxiv.org/abs/2501.02945 |
| **Conformal prediction (EnbPI/ACI) as a discovery screen** | Distribution-free, hot topic, maintained (MAPIE) | **Wrong question.** Conformal calibrates *intervals*, it does not *test for dependence*. And exchangeability — its foundational assumption — is violated by serial dependence, volatility clustering and drift. (The MAPIE 2026 exchangeability tests are a useful *diagnostic*, though.) | [U] https://arxiv.org/abs/2511.13608 · https://github.com/scikit-learn-contrib/MAPIE |
| **PCMCI+ full causal graph discovery** | Designed for autocorrelated data; genuinely good calibration story | Answers "what is the whole graph," which is vastly more power-hungry than "is there anything." Inherits the §3.2 CI-test calibration tension. Validated mostly in climate science where effects are large. **Not wrong — just wasteful of a sample you cannot replace.** | [U] https://arxiv.org/abs/2003.03685 |
| **Deflated Sharpe Ratio as the primary verdict** | Standard in the industry; directly about backtest overfitting | **N (effective independent trials) is a free parameter you choose after seeing the data**, via clustering. Also model-based (assumed null SR distribution, normal-plus-skew-kurtosis). Report it, don't rely on it. Note the MinBTL figure cuts *in your favour*: 9 trials is below the ~7-trials-to-Sharpe-1 threshold, so the DSR haircut is small and uninformative either way. | [U] https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf · https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3221798 |
| **Deep-testing (learned test statistics)** | Highest power vs 19 competitors | 5 months old, no code, **no stated treatment of serial dependence**, and a learned statistic is the archetype of a confident number that measured nothing. Revisit in 2027. | [V] https://arxiv.org/abs/2604.26558 |
| **Politis–White automatic block length** | Principled, implemented in `arch` | Not *wrong*, just **unnecessary** — you have a natural block (the session). Optimising a block length you already know is added complexity, which your own project history says makes things worse. Use it only for sensitivity analysis. | [V] https://arch.readthedocs.io/en/latest/bootstrap/generated/arch.bootstrap.optimal_block_length.html |

---

## 9. Loose ends I could not close

1. **`hyppo`'s actual permutation scheme.** The user guide says "permutation test" without confirming block permutation. **Read `hyppo/time_series/*.py` before trusting any p-value from it.** [V that the docs are silent]
2. **SSRN 5520741** — López de Prado, Lipton & Zoonekynd, *How to Use the Sharpe Ratio* (2025). **SSRN returned HTTP 403.** Likely the most current statement of the DSR position; retrieve manually.
3. **No substantial peer-reviewed teardown of DSR/PBO found.** The critiques I located are practitioner-level ("model-based," "requires a choice of N"). Either the critique literature is thin or my searches missed it.
4. **No 2025–2026 critique of PCMCI+ false positives on financial data found**, despite targeted searching. Treat this as unknown rather than as a clean bill of health.
5. **`tigramite` current version unverified** — docs found are for 5.2; I could not confirm a 2026 release.
6. **No code release located** for: dGCM, AutoHSIC, TSKI, Deep-testing, Legendre jumper martingales. All Tier-3 items are build-it-yourself.
7. **`recombinator` maintenance status uncertain** — PyPI 0.0.6.1, no recent release date confirmed. Prefer `arch`.

---

## 10. The one-page recommendation

**Stop adding dependence measures. Change the unit of analysis and the accounting.**

1. **Make the session the unit.** Everything — permutation, bootstrap, cluster, conformal score, split — happens at session granularity. This alone removes the ~300x effective-n inflation, and it removes it *without* any new method.
2. **Compute the MDE first.** If no plausible edge is detectable at your session count, the programme's question is already answered — not "no signal," but **"this sample cannot distinguish signal from no signal,"** which is a different and more honest conclusion, and one your existing eight nulls do not establish.
3. **Run one SPA.** All nine dead hypotheses plus any live ones as loss columns, block = session. One number, correctly accounting for how many you looked at, valid under serial dependence, in a maintained library you already have.
4. **Run one MDH test.** Cheapest possible omnibus answer to "is there any directional predictability."
5. **Convert the frozen forward test to an e-process.** Then the checkpoints stop costing you error rate and you can stop early for futility with a defensible number.
6. **Keep the ledger in e-values.** e-BH is dependence-agnostic across hypotheses, which is the only honest way to score a programme where every hypothesis shares a sample.

Steps 3–6 together cost roughly **two** hypotheses' worth of the sample, not nine. That is the actual answer to "how do I stop burning the sample."
