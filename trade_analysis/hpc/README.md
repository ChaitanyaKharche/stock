# HPC package — the 0DTE variance risk premium experiment

Everything needed to run the experiment pre-registered in
[`research/vrp_preregistration.md`](../../research/vrp_preregistration.md).

**Read the pre-registration first.** It fixes the split, the target, the benchmark, the
family size and the decision rule before any of this runs. The code here implements it; it
does not get to change it.

---

## Why this question and not another

Eight prior nulls closed **direction** prediction in this project. This is a **variance**
question, which none of them tested, and the distinction is not cosmetic:

- A 0DTE straddle held to expiry pays realised variance against the variance implied by
  the price paid. No directional call is required at any point.
- The variance risk premium — implied systematically exceeding realised — is a documented
  effect. In a 4-session smoke sample of this very dataset, implied exceeded realised at
  **80.1%** of forecast origins.
- The hint came from a bug found on 2026-09-08: `_calculate_momentum_score` is a pure
  **magnitude** (every term `abs()`-wrapped), and the code had been forcing that magnitude
  to emit a directional CALL/PUT. The part of the system measuring something real was
  measuring **size, not sign**, and no experiment had ever tested it on its own terms.

It may still null. §6 of the pre-registration says what to do when it does, and why a
powered null here is worth more than another good-looking backtest.

---

## The data, and one hard constraint

| archive | coverage | note |
|---|---|---|
| `option_quote_1m_0dte/SPY/` | **769 sessions**, 2020–2024 | minute quotes with **real bid/ask** |
| `stock_ohlc_1m/SPY/` | continuous 2016–2026 | stored **per month**, not per day |
| `stock_quote_1m/SPY/` | 2017–2026 | underlying NBBO |

**The option archive cannot be extended.** The ThetaData subscription lapsed to
`Options: FREE` on or before 2026-09-08. Every 0DTE session that will ever exist for this
project exists now. Spending the 2024 held-out block on an exploratory run is
**permanent**, and it is the single reason the split is frozen in advance rather than
chosen later.

Validated on build: `_rv` over a full session returns 8.8% annualised for 2020-01-03,
against 8.7% close-to-close for the same period from an independent source. The
annualisation is right.

---

## Outcome — answered 2026-09-09

**The premium is real and 1.84x too small to trade.** Full write-up in
[`research/vrp_cost_model_results.md`](../../research/vrp_cost_model_results.md).

| | mean/trade | win rate | session-clustered t |
|---|---|---|---|
| GROSS (mid to mid) | **+$0.0205** | 68.6% | **+7.52** |
| NET (bid to ask) | **-$0.0171** | 61.0% | **-5.38** |

Spread cost $0.0376 against a $0.0205 edge. **The mid-fill assumption is the difference
between t=+7.52 and t=-5.38** -- between a strategy and its opposite.

And the trap: 61% of trades win NET, median +$0.04, mean still negative, because **the
worst 1% of trades carry 94% of the loss.** Short gamma would look like it was working for
months.

**The GPU sweep was deliberately not run.** A model would have to beat the premium by
nearly 2x purely to break even, and implied variance already beats HAR at forecasting
realised variance (p=0.0222). The steps below remain reproducible; step 4 is the one that
was judged not worth the compute.

---

## Order of operations

Layout on Discovery, matching the actual clone:

```
~/vrp/rep/      the git clone -- disposable, re-clonable, never holds a result
~/vrp/data/vrp/ the parquet frame -- rsynced up, NOT in git
~/vrp/results/  outputs -- pull these down the day they appear
```

**0. Build the frame on the laptop, then push it up.** The parquet is deliberately not in
git (derived, 37 MB, and git keeps blobs forever), so the clone arrives without it.

```bash
python -m trade_analysis.hpc.build_vrp_dataset --symbol SPY --out data/vrp
```
```bash
rsync -avP data/vrp/ kharche.c@login.explorer.northeastern.edu:~/vrp/data/vrp/
```

**1. Build the environment once, on a compute node** (torch is ~2.5 GB; login nodes are
shared):

```bash
srun --partition=short --time=00:40:00 --mem=16G --pty bash
cd ~/vrp/rep && bash trade_analysis/hpc/setup_env.sh
```

**2. Establish the bar and audit it. CPU only, no GPU requested.**

```bash
sbatch trade_analysis/hpc/submit_baseline.sbatch
```

Runs HAR-RV and HAR-RV-J against naive persistence and the option market's own implied
variance, then re-runs with a deliberately injected one-minute lookahead. **Read the audit
the opposite way to the obvious guess:**

| result | meaning |
|---|---|
| audit **better** than honest | **PASS** — the honest features did not contain that minute |
| audit **same** as honest | **FAIL** — the minute was already inside the honest run |

Measured 2026-09-08 on 623 sessions: honest HAR-RV QLIKE **0.187365**, audit **0.184776**.
Audit better by 1.4% → pass. That gap is also a calibration: one minute of genuine
lookahead is worth ~1.4% QLIKE here, so a model claiming a much larger margin over HAR is
claiming more than cheating with a minute of the future would buy.

This project lost an entire result set to a 1-minute lookahead that produced 95.7% of a
measured edge. One extra CPU job.

**4. Sweep on the cluster** (GPU, ~30 min per cell)

```bash
sbatch --array=0-23 trade_analysis/hpc/submit_sweep.sbatch
```

24 cells: 2 optimisers × 3 widths × 4 seeds. **Four seeds per configuration is the point.**
At ~250 validation sessions the seed-to-seed QLIKE spread is comparable to the effect being
looked for, so a single seed beating HAR is not a finding.

**5. The held-out block.** Only if §6's validation gate passed, and only once.

---

## What is deliberately small

The model is a 2-layer MLP, tens of thousands of parameters. The existing TFT checkpoint
has 398,854 parameters and collapsed to its prior — capacity was never the constraint. With
~368 training sessions, a larger model buys variance, not signal.

Requesting an H100-week for a 3 MB dataset reads as inexperience, not rigour. The sweep is
sized so the *evidence* is strong, not the hardware.

---

## Choices that are not obvious

**QLIKE, not MSE.** Realised variance is itself a noisy estimator of latent variance. MSE
over a noisy proxy rewards predicting the noise; QLIKE is robust to it (Patton 2011) and is
the field standard for exactly that reason. It is both the training loss and the evaluation
metric — training on one and scoring on another optimises the wrong thing. RMSE is reported
alongside, never instead.

**Clustered by session.** Minute observations within a day are ~0.9 correlated at 30-minute
separation. Treating ~300 origins per session as independent inflates the effective sample
~300× and manufactures significance. Diebold–Mariano here uses one loss differential per
**session**: n ≈ 250, not n ≈ 75,000.

**Predict log variance.** Makes a negative variance unrepresentable rather than unlikely.
QLIKE is undefined for one.

**Zero-shot foundation models are tried FIRST, not last.** In 2024 the reflex was to train a
TFT. By 2026, Chronos-2 / TimesFM-2.5 / Moirai-2 are strong enough zero-shot that training
before checking them wastes compute and risks reporting a trained model a free API call
would have beaten. If a zero-shot model wins, §9 requires stating that the problem needed no
bespoke model — not dressing it up as this project's contribution.

**Muon where available.** Orthogonalised updates via Newton–Schulz; native in PyTorch ≥ 2.9,
~2× compute efficiency vs AdamW on compute-optimal LLM training. At this scale the wall
clock is irrelevant. It is in the sweep so the optimiser cannot be the excuse for a null.

---

## Cluster layout — and the purge that already ate this project once

`/scratch` on Discovery is **purged after 45 days of no access.** It has already destroyed
work here: the entire `/scratch/kharche.c/kharche.c/` tree is gone, taking the **original
TFT model-definition source** with it. The checkpoints survived only because they had been
copied into the git repo, which is why `models/tft_model.py` is a reverse-engineered
reconstruction rather than the real thing.

So nothing that matters is allowed to live on scratch:

```bash
# on the cluster -- everything under $HOME, which persists
mkdir -p ~/vrp && cd ~/vrp
git clone <this repo> repo && ln -s repo/trade_analysis trade_analysis
```

```bash
# from the laptop -- push the built frame up (a few hundred MB, not the 3.5 GB raw archive)
rsync -avP data/vrp/ kharche.c@login.explorer.northeastern.edu:~/vrp/data/vrp/
```

```bash
# after the sweep -- pull results down the same day, do not let them wait
rsync -avP kharche.c@login.explorer.northeastern.edu:~/vrp/results/ ./results/
```

Scratch is still the right place for fast I/O *during* a job. It is never the place a
result waits for you.

**Reclaimable space:** `~/.cache/huggingface` was measured at **44 GB**. If home quota is
tight, `du -sh ~/.cache/*` then clear what is not needed — the old cleanup script
(`trade_analysis/scripts/clean_trade_venv.sh`) fails on a stale `huggingface-cli --yes`
flag and reclaimed nothing.

**Never `export HF_TOKEN=...` on a command line.** It lands in `~/.bash_history` in
plaintext on a shared filesystem. Use `huggingface-cli login`, or a `chmod 600` env file.

## Cluster environment

```bash
module load cuda/12.4
python -m venv ~/venvs/vrp && source ~/venvs/vrp/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy pandas pyarrow scipy lightgbm
# optional, for the zero-shot arm:
pip install chronos-forecasting  # or: pip install timesfm
```

`torch>=2.9` gives `torch.optim.Muon` natively; otherwise `pip install muon` or let it fall
back to AdamW (the script reports which it used).

Transfer only `data/vrp/` — parquet, a few hundred MB — not the 3.5 GB raw archive.

---

## Files

| file | role |
|---|---|
| `build_vrp_dataset.py` | raw archives → per-session parquet; the no-lookahead rule lives here |
| `har_baseline.py` | HAR benchmarks, QLIKE, clustered Diebold–Mariano, `--audit` |
| `train_vrp.py` | the neural candidate; QLIKE loss, AdamW or Muon |
| `submit_sweep.sbatch` | 24-cell array job |
