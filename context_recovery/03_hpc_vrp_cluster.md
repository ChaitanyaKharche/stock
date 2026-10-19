# 03 — The HPC cluster leg: the variance-premium experiment and the dead TFT

Written 2026-10-13 by a read-only context-recovery agent. Static reading plus read-only
`git log` / `git show` inside the worktree. Nothing was run, nothing was installed, the
cluster was not contacted, and the main checkout was not touched.

**What I could not see, so everything about it is marked (unverified):** the cluster's
filesystem (`~/vrp/...`), the derived data files (`data/vrp/`, `data/cost_model_trades.parquet`
— both git-ignored, `.gitignore:119`, `.gitignore:139`, and absent from this worktree), and
any raw result files. The numbers below are quoted from tracked `research/*.md` files and
commit messages, never recomputed.

All paths are relative to the repo root unless they start with a drive letter.

Where the seven extra questions are answered:

| question | section |
|---|---|
| 1. Why the cluster; "size, not sign" | §2 |
| 2. The pipeline in order | §3.1–3.7 |
| 3. The verdict in cents; "1.84× too small" | §3.6 |
| 4. The cluster, the purge, every sbatch | §3.8 |
| 5. What happened to the TFT | §3.9 |
| 6. Frozen / pre-registered / do not peek | §6 |
| 7. Still open, including the HF token | §8 |

---

## 1. In one paragraph

This slice is the part of the project that ran on Northeastern's shared computer cluster
("Discovery", scheduled by Slurm — the program that queues jobs). It holds two things.
First, an old neural network (a "Temporal Fusion Transformer", TFT) that was trained there
in 2025 to predict next-day price gaps, later measured, and found to give the same answer
whatever you feed it. Second, the September 2026 experiment built to replace it: instead
of asking *which way* SPY will move, ask *how much* it will move, and whether option
sellers are overpaid for that. The answer was clean: **option buyers do overpay — by about
2 cents per share on a $1.79 straddle — but it costs almost 4 cents in bid-ask spread to
collect it, so the trade loses money.** No GPU model was ever trained for the new
experiment, on purpose. The leg has had no commits since 2026-09-24 and is effectively
closed, with a handful of loose ends listed in §8.

---

## 2. The goal it serves, and how it changed

**Why the project went to the cluster at all.** Twice, for different reasons.

1. **2025, the TFT.** A 398,854-parameter network was trained on a cluster GPU to predict
   the next day's "gap" (direction and size of the next daily move) for seven tickers.
   Root `README.md:204-205` records it as "an earlier HPC training run
   (`srun --gres=gpu:h100:1`)". That needed a GPU because it was a deep model. It did not
   work (§3.9).
2. **2026-09-08, the variance experiment.** The stated question was whether *anything* —
   including a GPU-trained model — forecasts short-horizon price movement size better than
   a six-number linear formula, well enough to trade (`research/vrp_preregistration.md:1`,
   `:144-157`). The GPU was for a 24-run training sweep and for a pretrained forecasting
   model with ~200 million parameters (`trade_analysis/hpc/submit_zeroshot.sbatch:7-9`).

**"Size, not sign" in plain words.** Eight earlier experiments tried to predict *direction*
(up or down) and all failed. On 2026-09-08, while fixing the demo app, the assistant found
that the app's "momentum score" was built entirely from absolute values — it measured *how
much is happening*, never *which way* — and the code was forcing it to say CALL or PUT
anyway (`research/vrp_preregistration.md:13-22`). The pre-registration's own sentence:

> "the part of this system that was measuring something real was measuring size, not sign."
> — `research/vrp_preregistration.md:20-21`

So the new question was about size. "Variance" is the technical word for size of movement
(the average squared price change). An option's price contains the market's guess at future
variance ("implied variance"). What actually happens is "realised variance". If implied is
usually higher than realised, option sellers collect the gap; that gap is the **variance
risk premium (VRP)**. A **straddle** (one call plus one put at the same strike) is the
instrument that pays on size regardless of direction (`trade_analysis/hpc/README.md:17-18`).

Note for whoever picks this up: the owner trades naked long calls and puts, and `CLAUDE.md`
says not to offer straddles. This leg's use to him is explanatory — it measures the premium
a long-option buyer pays — not a trade to hand him.

**How the goal moved, in order** (dates from `git log -- trade_analysis/hpc research/vrp_*`):

| date | goal at that moment | what moved it |
|---|---|---|
| 2025 | train a TFT to predict gaps | — |
| 2026-09-06 | TFT removed from the demo's vote | measured as a constant (`huggingface_space/trade_analysis/enhanced_api.py:540-557`) |
| 2026-09-08 | forecast variance; beat the linear benchmark; then GPU sweep | the "size, not sign" hint |
| 2026-09-08/09 | **is the premium tradeable after the real spread?** | pre-registration §6 demands it; answered: no (`research/vrp_cost_model_results.md`) |
| 2026-09-09/10 | does a free pretrained model beat the benchmark? | "zero-shot first" rule; answered: no, it loses badly |
| 2026-09-11 | is the premium collectable on a *chosen subset* of moments? | literature survey; Arm A failed replication, Arm B "passes but fragile" |
| 2026-09-11 | is there any forecasting room left in this data? | challenged by the owner; answered: no (`ceiling.py`, commit `119ed54`) |
| after 2026-09-11 | nothing — leg dormant | cluster reused for the momentum sweep (another agent's slice) |

---

## 3. How it works, end to end

### 3.1 What goes in

Three raw archives on the laptop, outside the repo, under
`C:\Users\chaitanyakharche\Desktop\data\raw` (`trade_analysis/hpc/build_vrp_dataset.py:28`):

| archive | what | note |
|---|---|---|
| `option_quote_1m_0dte/SPY/` | one file per day of minute-by-minute SPY option quotes for options expiring that same day ("0DTE" = zero days to expiry), with real bid and ask | **769 sessions**: 2020: 159, 2021: 39, 2022: 170, 2023: 250, 2024: 151 (`research/vrp_preregistration.md:34-37`) |
| `stock_ohlc_5m/SPY/` | SPY price bars, 5-minute, one file per **month** | complete 2016–2026 (`build_vrp_dataset.py:40-42`) |
| `stock_ohlc_1m/SPY/` | 1-minute bars | missing 2024-01..07 and cannot be refetched; used only to repair truncated 5-minute months (`build_vrp_dataset.py:104-132`) |

The option archive can never grow: the ThetaData subscription dropped to `Options: FREE`
on or before 2026-09-08 (`research/vrp_preregistration.md:42-46`).

**The split, frozen in advance** (`research/vrp_preregistration.md:48-54`, implemented in
`trade_analysis/hpc/har_baseline.py:69-76`):

| block | dates | sessions that reach the models | use |
|---|---|---|---|
| TRAIN | before 2023-01-01 | 346 | fit anything, unlimited looks |
| VALIDATION | 2023 | 250 | compare models |
| **HELD OUT** | **2024** | **151** | **look exactly once, at the very end, for one winning model only** |

Counts from `research/vrp_baseline_results.md:7-11`. 769 on disk, 747 usable: the first 22
sessions are a warm-up because one input needs 22 prior days (`har_baseline.py:189-195`).

### 3.2 Building the dataset (on the laptop)

`python -m trade_analysis.hpc.build_vrp_dataset --symbol SPY --out data/vrp`
(`build_vrp_dataset.py:3`). For each session it walks the 5-minute bars between 10:00 and
15:00 (`:32`, `:308`) and writes one row per bar — a "forecast origin", the moment a
forecast would be made. Because the code also needs seven bars of history (`:314`), the
first usable origin is 10:05, giving 60 per session **(inferred from `:312-315`; the "60
origins" figure itself is stated at `zeroshot_baseline.py:16`)**.

Each row holds:

- **Inputs known at that moment.** Movement so far today, over the last 30 minutes, the
  previous day, the previous 5 and 22 days (`:276-294`); and from the option quotes: the
  market's implied variance backed out of the at-the-money straddle price, a skew proxy, a
  smile proxy, the straddle's bid-ask spread (`:233-272`).
- **The answer, known only later.** Realised variance over the next 30 minutes
  (`rv_fwd_30`) and next 60 (`:326-337`).

The one rule that matters is at `:310-312`: the history handed to the feature code is
`bars.index < t` — the bar stamped `t` is thrown away, not trimmed, because it contains
prices from after `t`. The targets start at the bar *after* `t` (`:328`). So the bar at
`t` belongs to neither side. This gap is deliberate and is why the pretrained-model arm
forecasts 7 steps ahead, not 6 (`zeroshot_baseline.py:16-32`).

Output: one parquet file per session plus `manifest.json` recording the bar interval
(`:394`, `:405-414`). Per-session files were chosen so a walk-forward cannot accidentally
shuffle across dates (`:12-14`). The frame is ~37 MB and deliberately not in git
(`.gitignore:115-119`).

### 3.3 The scoring vocabulary, one sentence each

- **QLIKE** — the error score used throughout; lower is better; it punishes forecasting
  too *low* harder than too high, and is the field standard because the "true" variance
  is itself measured noisily (`har_baseline.py:14-18`, `:79-83`).
- **Diebold–Mariano (DM) test** — a t-test on the difference in error between two
  forecasts, asking "is A reliably better than B or is the gap noise?" (`har_baseline.py:127-148`).
- **Session clustering** — the 60 forecasts inside one day are near-copies of each other
  (~0.9 correlated), so the test averages each day down to one number first; the honest
  sample is ~250 days, not ~15,000 rows (`har_baseline.py:20-23`, `:132`).
- **Holm correction** — when several models are tested, the p-value bar is tightened so
  one lucky model out of many is not called a finding (`harp_baseline.py:141-149`).
- **The lookahead audit** — deliberately give every model one bar of the future and check
  the score gets *better*. Better = PASS (the honest run did not already contain it). Same
  = FAIL (it did) (`har_baseline.py:25-38`, `:216-224`).

### 3.4 The contestants

| name | what it is | where |
|---|---|---|
| persistence | "the next 30 minutes will look like the last 30" | `har_baseline.py:165-167` |
| implied_variance | the option market's own number, used raw | `har_baseline.py:168-171` |
| **HAR-RV** | a linear formula on five past-movement inputs (last 30 min, since open, yesterday, 5-day, 22-day). "HAR" = heterogeneous autoregressive, the standard benchmark | `har_baseline.py:54`, `:86-124` |
| HAR-RV-J | HAR plus a jump-robust term | `har_baseline.py:55`, `:172-173` |
| HAR-IV | HAR plus the market's implied variance | `harp_baseline.py:165` |
| HAR-TOD / -bins | HAR plus time-of-day terms | `harp_baseline.py:166-167` |
| **HAR-IV-TOD** | both — the final bar to beat | `harp_baseline.py:168` |
| Chronos-Bolt-base, zero-shot | a pretrained general forecasting model from Amazon, used as-is with no training ("zero-shot") | `zeroshot_baseline.py:495`, `submit_zeroshot.sbatch:45` |
| kitchen-sink linear, GBDT | every usable column thrown in; a gradient-boosted tree — diagnostics only | `ceiling.py:107-132` |
| MLP (the "trained model") | a small 2-layer network — **code exists, never run** | `train_vrp.py:115-119` |

All of them import the split, cleaning, QLIKE and DM test from `har_baseline.py` rather
than re-implementing them, so no two models are scored by different rules
(`zeroshot_baseline.py:137-138`, `harp_baseline.py:77-78`, `ceiling.py:58-59`,
`dm_concentration.py:74-75`).

### 3.5 What came out of the forecasting contest (validation = 2023, 250 sessions)

| model | QLIKE | source |
|---|---|---|
| **HAR-IV-TOD** | **0.330489** | `research/vrp_harp_specification.md:17` |
| HAR-IV | 0.335902 | `:18` |
| GBDT, all features | 0.336346 | commit `119ed54` message |
| kitchen-sink linear | 0.342327 | commit `119ed54` message |
| implied_variance alone | 0.411136 | `research/vrp_baseline_results.md:85` |
| HAR-RV | 0.430642 | `:86` |
| HAR-RV-J | 0.430788 | `:87` |
| Chronos-Bolt, 8.5 sessions of history | 0.792147 | `research/PROJECT_REPORT.md:178` |
| Chronos-Bolt, 22 sessions of history | 0.793359 | `research/PROJECT_REPORT.md:179` |
| persistence | 0.803472 | `research/vrp_baseline_results.md:88` |

Reading it:

- The option market's raw number and plain HAR are **statistically tied** (t=−0.55,
  p=0.5832, `research/vrp_baseline_results.md:95`). On a typical day HAR actually wins:
  the market's number is better on only 99 of 250 sessions, so HAR takes 60.4%
  (`research/vrp_phase0_concentration.md:49`, `:60`; "151/250" in commit `485d48d`).
- **Putting the two together beats either alone by about 22%**, robustly: 198 of 250
  sessions, DM t=−2.95, p=0.0035 (`research/vrp_harp_specification.md:34-50`).
- **The 200-million-parameter pretrained model loses to the six-number formula** by a wide
  margin (t=+5.25, p=3.2e-07), and giving it more history changed nothing
  (`research/vrp_cost_model_results.md:105-112`).
- Throwing every column at a tree model makes the forecast *worse* than the bar. The
  conclusion written into the code: "The limiting factor is the DATA, not the model class"
  (`ceiling.py:246`).

The GPU training sweep was therefore never run (`trade_analysis/hpc/README.md:69-72`).

### 3.6 The verdict: is there a premium, and can you collect it?

**Yes there is one, and no you cannot.** `trade_analysis/hpc/cost_model.py` reads the raw
option quotes directly and simulates the simplest possible trade at every origin: sell the
at-the-money straddle, hold 30 minutes, buy it back (`cost_model.py:14-19`). It prices each
trade two ways:

- **gross** = sell at the mid price, buy back at the mid (`:134`) — what a careless
  backtest assumes;
- **net** = sell at the bid, buy back at the ask (`:135`) — what you actually get.

37,517 trades over 618 sessions, 2020–2023; 2024 excluded in code (`cost_model.py:174`).
From `research/vrp_cost_model_results.md:16-20`:

| | mean per trade | win rate | session-clustered t |
|---|---|---|---|
| GROSS (mid → mid) | **+$0.0205** | 68.6% | **+7.52** |
| NET (bid → ask) | **−$0.0171** | 61.0% | **−5.38** |
| spread cost | $0.0376 | | |

In everyday money (prices are per share; one option covers 100 shares, so multiply by 100
for one straddle — that ×100 is my arithmetic, **(inferred)**): the straddle costs about
$179. The premium you would collect is about **$2.05**. Crossing the bid-ask spread in and
out costs about **$3.76**. You end up **−$1.71** per trade.

The document's own words for "1.84× too small":

> "The ATM straddle mid is $1.79 at a 152 bp spread. Round-tripping costs **$0.0376**. The
> gross edge is **$0.0205**. The spread is **1.84× the edge.** You would pay nearly twice
> your edge for the privilege of collecting it."
> — `research/vrp_cost_model_results.md:31-34`

("bp" = basis point = one hundredth of one percent. 152 bp of $1.79 is about 2.7 cents —
the full bid-to-ask width at entry; that multiplication is mine, not the file's.)

And the mid-price trap:

> "it is the difference between `t=+7.52` and `t=−5.38` — between a strategy and its opposite."
> — `research/vrp_cost_model_results.md:37-38`

Two more facts from the same file that a newcomer needs:

- **It would have fooled you for months.** 61% of trades win and the median trade makes
  +$0.04, yet the average is negative, because "The worst 1% of trades — 375 of 37,517 —
  account for 94% of the total net loss" (`:53`). The worst 0.1% loses $2.295 per share,
  57× the median win (`:46`, `:56`).
- **Negative in every year 2020–2023, but shrinking** — from −$0.0346 to −$0.0080 —
  because spreads tightened, not because the premium grew (`:68-78`). The file is explicit
  that this "is not a prediction and it is not permission to trade it" (`:81`).

Why this result matters to a long call/put buyer: he is on the other side of this premium.
`CLAUDE.md` states the consequence — you can be right about the travel, pay the variance
premium for it, and collect nothing for direction.

### 3.7 The conditional follow-up (2026-09-11)

Question: the premium is too small *on average*; is it collectable at *selected* moments?
Pre-registered in `research/conditional_0dte_preregistration.md` before any number was
computed (commit `45eb21a`). Two arms.

**Arm B — condition on our own forecast.** No new backtest: it re-reads the P&L already
measured by `cost_model.py` and keeps only origins where implied variance most exceeds the
HAR-IV forecast (top third, cutoff taken from TRAIN only and applied to 2023 unchanged;
`conditional_vrp.py:59`, `:161-162`, `:196`). Result (`research/conditional_0dte_armB_results.md:11-26`):

| | net per trade | sessions | session t |
|---|---|---|---|
| selected | **+$0.01002** | 73 | +2.05 |
| all 2023 origins | −$0.00797 | 250 | −1.81 |

About +$1 per straddle on 7.5% of moments. It clears all five pre-registered gate
conditions and is **reported as FRAGILE**: dropping the single best session takes t under
2.0, and the worst 0.1% of trades is 16× the median win in a year with no crisis
(`:28-42`). The script exits non-zero on a fragile pass so nobody skims it as a clean one
(`conditional_vrp.py:317-323`, `:330`). A follow-up check showed that "just sell when
implied vol is high" does **not** reproduce it (23 sessions, t=−0.58; commit `d0de4c9`).

**Arm A — replicate a published rule** (Almeida/Freire/Hizmeri) on a different dataset,
the Vilkov SPXW panel (S&P 500 index options, 30-minute bars, 2016-09 → 2024-05). This is
the one job in this leg that ran as a 12-task Slurm array (`submit_arm_a.sbatch:9`,
`arm_a_ssd.py:72-73`). **It failed to replicate, and the failure is the implementation's**:
the rule's reference price sat 2.8% above the market while the thing it was supposed to
detect was 0.6%, so the rule degenerated into "buy almost always" (78.4% buys)
(`research/conditional_0dte_armA_results.md:40-47`). The pre-registration had fixed in
advance that a failed replication "is about our code until proven otherwise"
(`research/conditional_0dte_preregistration.md:147`), and that Arm B alone supports no
claim (`research/conditional_0dte_armB_results.md:92-95`).

So the conditional question is **unresolved**, not answered: one fragile pass that the
pre-registration says not to lean on until the replication works, and a replication that
needs a new, properly specified pre-registration that was never written
(`research/conditional_0dte_armA_results.md:93-99`; no such file exists in `research/`).

### 3.8 The cluster itself

**Names.** The project calls it Discovery; the login host is
`login.explorer.northeastern.edu` (`trade_analysis/hpc/README.md:98`) and Northeastern's
current docs call the cluster Explorer (`research/vendor_and_method_surveys/research_hpc_setup.md:1`).
User `kharche.c`, home `/home/kharche.c` (`submit_arm_a.sbatch:35`).

**Partitions** (queues), as published and checked 2026-09-10/11
(`research_hpc_setup.md:43-58`, `trade_analysis/hpc/README.md:313-321`):

| partition | max time | running jobs per user | note |
|---|---|---|---|
| `short` | 48 h | 50 | CPU only |
| `gpu` | 8 h | **4**, one GPU each | |
| `sharing` | 60 min | 2 | unusable here |

The published 50 is a ceiling, not a promise: a 500-task array later measured only 20
running at once (memory note `northeastern-discovery-hpc.md:33-39`, written during the
momentum sweep — not re-verified).

**GPUs.** H200, A100 and "any GPU" requests all appear in the account's history (memory
note `northeastern-discovery-hpc.md:26-29`); the zero-shot run landed on a V100 (commit
`66be51b`). Root `README.md:205` says the TFT was trained on an H100. The sweep script asks
for any GPU on purpose: "Pinning h100/h200/a100 only buys queue time"
(`submit_sweep.sbatch:18-19`).

**Storage** (`research_hpc_setup.md:656-663`):

| path | size | purged? |
|---|---|---|
| `/home/<user>` | 75 GB | no |
| `/scratch/<user>` | 20 TB | **everything, first Tuesday of every month** |
| `/projects/<name>` | 35 TB per PI | no |

**The purge that ate the TFT source.** `/scratch` is not "deleted after 45 days of no
use". The whole thing is wiped in the monthly maintenance window regardless of how
recently a file was touched (`trade_analysis/hpc/README.md:187-197`, quoting
rc.northeastern.edu/scratch-space-policy, verified 2026-09-11). The project's own notes
said "45 days of no access" until commit `aa80e9b` corrected them — "wrong in the direction
that loses work, because it implies touching a file protects it". By 2026-09-09 the tree
`/scratch/kharche.c/kharche.c/` was gone, and with it what the project believed was the
only copy of the TFT's model-definition code (`trade_analysis/hpc/README.md:199-202`). The
trained weights survived only because they had been committed to git (commit `e17df27`).
See §9 for a complication: a TFT definition does exist in the repo.

**The layout that prevents a repeat** (`trade_analysis/hpc/README.md:83-89`,
`submit_sweep.sbatch:36-47`). Everything under `$HOME`, nothing on scratch:

```
~/vrp/rep/      the git clone — disposable, never holds a result
~/vrp/data/     inputs pushed up from the laptop — never in git
~/vrp/results/  outputs — pull down the day they appear
~/vrp/logs/     Slurm logs — must exist BEFORE sbatch
```

**Environment.** A conda environment named `vrp`, Python 3.11, built once by
`trade_analysis/hpc/setup_env.sh:20-40` **on a compute node, not the login node**
(`:9-13`): numpy, pandas, pyarrow, scipy, then torch from the CUDA 12.4 wheel index. That
resolved to torch 2.6.0+cu124 (commit `9725244`). Chronos is added by
`fetch_zeroshot.sh`, which refuses to run on a login node (`:32-39`) because the login node
killed pip for memory on 2026-09-09 (`:10-15`), installs with `--no-deps` so pip cannot
swap torch (`:51-52`), and prints a warning if torch or numpy moved (`:55-63`). The Arm A
run recorded numpy 2.2.6 / pandas 2.3.2 (`research/conditional_0dte_armA_results.md:3-4`).
`ceiling.py` needs scikit-learn, which `setup_env.sh` does not install — it must have been
added by hand **(inferred)**; `ceiling.py:138-141` says it is in neither local interpreter.

**Exactly how to submit a job** (assembled from `trade_analysis/hpc/README.md:91-142` and
the sbatch headers):

1. Laptop: build the frame (`python -m trade_analysis.hpc.build_vrp_dataset ...`).
2. Laptop → cluster: copy `data/vrp/` to `~/vrp/data/vrp/`. The README says `rsync`
   (`:98`); Git for Windows ships no rsync (`submit_arm_a.sbatch:40`), and `.gitignore:124-126`
   mentions a `vrp_frame.tar.gz` upload tarball, so it was probably tarred and copied
   **(inferred)**. For single files the form that works from PowerShell is `cd` into the
   folder and `scp <bare filename> host:dest` (`submit_arm_a.sbatch:32-43`; it took three
   commits to get that right — `6a0a028`, `8817ccd`).
3. Cluster, once: `git clone` into `~/vrp/rep`; `mkdir -p ~/vrp/logs`; build the env on a
   compute node via `srun --partition=short --time=00:40:00 --mem=16G --pty bash`.
4. Cluster, **from a login node**: `cd ~/vrp/rep && sbatch trade_analysis/hpc/<file>.sbatch`.
5. Same day: copy `~/vrp/results/` back to the laptop.

**Every sbatch file:**

| file | runs | asks for | output |
|---|---|---|---|
| `submit_baseline.sbatch` | `har_baseline` honest, then `--audit` (`:66-72`) | `short`, 4 CPUs, 24 GB, 20 min, **no GPU** (`:16-20`) | `~/vrp/results/baseline_honest.txt`, `baseline_audit.txt`; logs `~/vrp/logs/vrp-baseline_<job>.out/.err` (`:27-28`) |
| `submit_zeroshot.sbatch` | `zeroshot_baseline` at history 512 and 1320, then two audit modes at 1320 (`:96-115`) | `gpu`, 1 GPU, 4 CPUs, 32 GB, 1 h (`:15-20`) | `~/vrp/results/zeroshot_<model>_ctx<N>_honest.txt`, `..._audit_<mode>.txt`; logs `vrp-zeroshot_<job>` |
| `submit_sweep.sbatch` | one `train_vrp` cell per array task: optimiser × width {32,64,128} × seed {0..3}, 120 epochs (`:88-118`) | `gpu`, 1 GPU, 4 CPUs, 16 GB, 30 min (`:16-22`); array range given on the command line, 0-11 or 0-23 (`:4-5`) | `~/vrp/results/vrp_<opt>_w<width>_s<seed>.json`. **Never submitted, by decision.** |
| `submit_arm_a.sbatch` | `arm_a_ssd --slot-index $SLURM_ARRAY_TASK_ID` (`:109-112`) | `short`, 2 CPUs, 24 GB, 1 h, array 0-11 (`:2-9`) | `~/vrp/results/arm_a/trades_<HHMM>.parquet`, `result_<HHMM>.json`; then `collect_arm_a` writes `by_entry_time.csv` |

Shared habits and their reasons: `set -euo pipefail` in all four; the interpreter is called
by absolute path `~/.conda/envs/vrp/bin/python` rather than `conda activate`, because in a
batch job `conda` is often not even a command (`submit_baseline.sbatch:37-45`, commit
`f81f143`); log paths are absolute, because a relative `--output` lands in the git clone
(commit `ecd64e3`); each script checks its inputs exist and fails with the fix printed
before doing any work (`submit_baseline.sbatch:46-61`, `submit_zeroshot.sbatch:47-57`).

### 3.9 What happened to the TFT

**What it predicted.** Daily bars, a 96-day window, ten price/indicator inputs and five
"static" inputs, producing three quantiles of the next day's close-to-close return plus a
three-way UP/DOWN/FLAT "gap" classifier (`trade_analysis/models/tft_model.py:30-45`,
`:72-78`; target built at `huggingface_space/trade_analysis/tft_model.py:383-390`). The
398,854 parameter count checks out by hand from the layer sizes at
`trade_analysis/models/tft_model.py:63-78`.

**Why it was judged dead** — all measured, none inferred from architecture:

- Fed six different tickers' histories, its headline output moved 0.10 on a 0–100 scale;
  no class probability left 1/3 by more than 0.0077
  (`huggingface_space/trade_analysis/enhanced_api.py:542-545`).
- Over 120 sessions it gave one constant direction per ticker (NVDA DOWN 119/119, SPY and
  QQQ UP 119/119) and its hit rate equalled the always-guess-the-majority baseline to the
  decimal (`:545-548`).
- A violent crash and a flat day differ by 0.1 points (`research/PROJECT_REPORT.md:296-305`).
- The saved scaler for the five static inputs has every scale equal to 1.0: those inputs
  were a constant during training (`enhanced_api.py:555-557`). The code agrees — they are
  hard-coded numbers (`huggingface_space/trade_analysis/tft_model.py:344-357`).

It was removed from the demo's decision on 2026-09-06 (`enhanced_api.py:540`). The
replacement shown in the demo is a HAR volatility forecast fitted per ticker on daily bars
(`huggingface_space/trade_analysis/har_forecast.py:1-17`, called at `enhanced_api.py:332`),
which "replaces nothing and votes on nothing".

**The leftovers.** `trained_models/` holds 19 tracked files, about 19 MB, all added in
commit `e17df27` (2026-07-30) and never changed:

| files | size each | what |
|---|---|---|
| `tft_{SPY,QQQ,TSLA,NVDA,MSFT,META}_validated.pth` | ~1.61 MB | bare weights, loadable safely |
| `tft_{...}_validated.joblib` (6) | 1.2 KB | the matching input scalers |
| `tft_{AMZN,MSFT,SPY,TSLA}_e200_.pth`, `tft_model.pth` | ~1.61 MB | full pickles with scalers embedded |
| `tft_QQQ_e200_.pth`, `tft_QQQ_e200.joblib` | 1.61 MB, 1.2 KB | QQQ 200-epoch pair |

**Does anything still use them?** Two things, neither of which trades.

1. The Hugging Face demo loads the six `_validated` checkpoints at startup
   (`enhanced_api.py:113-123`), computes a prediction, and shows it in a collapsed panel
   titled "the 398,854-parameter TFT that collapsed (does not vote)"
   (`huggingface_space/streamlit_app.py:673-680`). The five full-pickle files are
   deliberately not shipped to the Space (`huggingface_space/sync_shared.py:61-75`).
2. `trade_analysis/models/tft_backtest.py:32-40` reads the six `_validated` files plus
   `tft_AMZN_e200_.pth`. That is why the root keeps the pickles — "Do not 'fix' this by
   deleting them there" (`sync_shared.py:68-69`).

Nothing in the live lab references them (searched `trade_analysis/` for `tft`,
`trained_models`, `TRAINED_MODELS_DIR`).

**Did the cluster's output feed anything downstream?** Only as text. The demo's research
tab shows the +7.52 / −5.38 headline and a fill-price slider rebuilt from
`data/cost_model_trades.parquet` (`huggingface_space/tools/build_research_snapshot.py:34-55`,
`:154-158`). No model trained on the cluster is used anywhere.

---

## 4. Map

`trade_analysis/hpc/` — 19 tracked files, **no test file**.

| file | lines | what it is for |
|---|---|---|
| `README.md` | 351 | the run-book, the outcome, the scratch warning, and the 2026-09-11 notes on what should change but was left for the owner |
| `__init__.py` | 0 | empty |
| `build_vrp_dataset.py` | 424 | raw archives → one parquet per session; the no-lookahead rule lives here |
| `har_baseline.py` | 279 | the shared core: load, split, clean, QLIKE, HAR fit, DM test, `--audit` |
| `harp_baseline.py` | 278 | HAR plus implied variance and time-of-day; Holm; found HAR-IV |
| `dm_concentration.py` | 356 | "is this result carried by a handful of days?" — sign test, leave-k-out, bootstrap |
| `zeroshot_baseline.py` | 518 | Chronos-Bolt scored on the same rows; three audit designs documented in the docstring |
| `ceiling.py` | 276 | "is any forecasting room left?" — kitchen-sink linear and a tree model |
| `cost_model.py` | 201 | **the result that closed the leg**: gross vs net short-straddle P&L |
| `conditional_vrp.py` | 345 | Arm B: condition the measured straddle on the forecast; mechanical gate |
| `arm_a_ssd.py` | 316 | Arm A worker: one entry time per array task, with a provenance record |
| `collect_arm_a.py` | 183 | Arm A collector; refuses partial sweeps and mismatched code versions |
| `train_vrp.py` | 212 | the small neural candidate — never run |
| `setup_env.sh` | 64 | build the conda env on a compute node |
| `fetch_zeroshot.sh` | 87 | install Chronos without letting pip move torch; cache weights |
| `submit_baseline.sbatch` | 76 | CPU job: benchmark + audit |
| `submit_zeroshot.sbatch` | 120 | GPU job: Chronos, two histories, two audits |
| `submit_sweep.sbatch` | 118 | GPU array: the training sweep — never submitted |
| `submit_arm_a.sbatch` | 112 | CPU array of 12: Arm A |

`trade_analysis/models/`

| file | what it is for |
|---|---|
| `tft_model.py` (248) | a **reconstruction** of the TFT from the checkpoints' tensor shapes; loader and predictor |
| `tft_backtest.py` (140) | out-of-sample check of the checkpoints via yfinance (needs network) |
| `__init__.py` | empty |

`huggingface_space/trade_analysis/` (another agent owns the Space; listed for the link)

| file | what it is for |
|---|---|
| `tft_model.py` (595) | a full TFT definition, wrapper, training loop and loader — see §9 |
| `train_tft.py` (162) | training entry point: 5 years of daily bars from yfinance |
| `har_forecast.py` (169) | the per-ticker HAR volatility forecast that took the TFT's place on screen |

`research/` — the record

| file | what it is |
|---|---|
| `vrp_preregistration.md` | the frozen design: split, target, 7-model family, decision rule |
| `vrp_baseline_results.md` | benchmark + audit; four silent defects; the corrected headline |
| `vrp_cost_model_results.md` | the verdict |
| `vrp_phase0_concentration.md` | the "option market beats HAR" claim does not exist once corrected |
| `vrp_harp_specification.md` | time-of-day is minor; HAR-IV is the real benchmark |
| `conditional_0dte_preregistration.md` | Arms A and B, gates fixed in advance |
| `conditional_0dte_armA_results.md` | failed replication |
| `conditional_0dte_armB_results.md` | passes its gate, reported fragile |
| `vendor_and_method_surveys/research_hpc_setup.md` | cluster policy survey: purge, partitions, Slurm exit codes, uv |
| `vendor_and_method_surveys/research_variance_sota.md` | literature survey — **briefed with a stale number**, see §7 |
| `vendor_and_method_surveys/research_dependence_methods.md` | survey of dependence tests; its advice: make the session the unit, compute the minimum detectable effect first |

---

## 5. Design decisions and the reason for each

| decision | reason, with evidence |
|---|---|
| Test variance, not direction | eight direction nulls; the magnitude-only momentum score (`research/vrp_preregistration.md:8-30`) |
| Freeze the split before anything runs; 2024 looked at once | the archive cannot be extended, so a wasted look is permanent (`vrp_preregistration.md:42-46`) |
| Temporal split, never random | "adjacent minutes of one day share the same realised variance path" (`vrp_preregistration.md:56-57`) |
| 5-minute bars, not 1-minute | the 1-minute archive is missing 2024-01..07 and cannot be refetched, which capped the held-out block at 5 of 151 sessions; 5-minute is also the literature standard (`build_vrp_dataset.py:38-50`, commit `c1ea49a`) |
| Windows stated in minutes, converted to bars | so "30" means 30 minutes at any interval (`build_vrp_dataset.py:34-36`, `:66-68`) |
| Exclude the bar stamped `t` entirely | a 1-minute lookahead once produced 88–95.7% of a measured edge (`build_vrp_dataset.py:5-10`) |
| Drop zero-priced bars at source | a trading-halt bar on 2020-03-09 made `log(0)`, which silently removed 193 of 769 sessions (`build_vrp_dataset.py:154-164`, commit `80e1d4c`) |
| Closed-form implied vol, not a solver | a solver can fail to converge unattended over 769 sessions (`build_vrp_dataset.py:219-230`) |
| QLIKE, not squared error | squared error on a noisy target rewards predicting the noise (`har_baseline.py:14-18`) |
| One loss number per session in every test | 60 near-identical rows per day would inflate the sample ~300× (`har_baseline.py:20-23`) |
| Predict the logarithm of variance | makes a negative variance impossible (`train_vrp.py:15-17`) |
| Multiply log-space forecasts by exp(s²/2) | exponentiating a log fit gives the median, not the mean; leaving it out cost HAR 20% of its score and created a false "p=0.023" (`har_baseline.py:106-124`, commit `cad54eb`) |
| CPU benchmark and audit before any GPU | "finding that out after a cluster allocation is the expensive way to learn it" (`har_baseline.py:6-9`) |
| Try a pretrained model before training one | otherwise you may report a trained model a free forward pass would have beaten (`zeroshot_baseline.py:3-9`) |
| Give Chronos 22 sessions of history | HAR gets a 22-day input; "Beating a model you under-informed is not a result" (`zeroshot_baseline.py:140-146`, commit `d9ef8f4`) |
| Score a lognormal fitted to Chronos's quantiles | the library's "mean" is its median — identical to six decimals on 14,940 rows (`zeroshot_baseline.py:108-117`, commit `cdafc2d`) |
| Keep the neural model tiny | "capacity was never the constraint"; ~346 training sessions (`trade_analysis/hpc/README.md:146-153`) |
| Four seeds per configuration | seed-to-seed spread is as large as the effect sought (`submit_sweep.sbatch:7-10`) |
| Refuse Muon when absent rather than fall back to AdamW | a silent fallback would have made 12 of 24 cells duplicates under a different label (`train_vrp.py:75-88`, commit `9725244`) |
| Cost model reads raw quotes, sells at bid, buys at ask | pre-registration §6: "A model that only wins gross of spread is a null" (`cost_model.py:6-24`) |
| Cost model hard-excludes 2024 | the held-out block stays sealed (`cost_model.py:174-177`) |
| **Do not run the GPU sweep** | a model would have to beat the premium ~2× just to break even (`README.md:69-77`); and later, every extra feature made forecasts worse (`ceiling.py:233-252`) |
| Everything under `$HOME`, nothing on scratch | the purge already destroyed work once (`submit_sweep.sbatch:36-45`) |
| Never `conda activate` inside a batch job | `conda` is a shell function that non-interactive shells often lack (commit `f81f143`) |
| Absolute Slurm log paths | relative ones wrote 24 stray files into the git clone and tripped a provenance check (commit `ecd64e3`) |
| `short` (CPU) for arrays, not `gpu` | 50 running jobs vs 4; and no GPU work in Arm A at all (`submit_arm_a.sbatch:23-26`) |
| Array over entry times, not sessions | the only shared history is a small table each task can rebuild; and it matches the published table's shape (`arm_a_ssd.py:8-16`) |
| Collector refuses a partial sweep by default | "how a 24-cell sweep reports 12 cells as a finished result" (`collect_arm_a.py:6-8`) |
| Provenance gate on tracked code changes only | the live lab keeps the tree permanently dirty, so a gate on any dirt fired on every task (`arm_a_ssd.py:94-108`, commit `915a512`) |
| New experiments import from `har_baseline`, never edit it | the project's isolation rule (`research/vrp_harp_specification.md:108-114`) |
| Arm B threshold from TRAIN only | computing it on the evaluation year is "the specific lookahead most likely to manufacture a result" (`conditional_vrp.py:20-23`) |
| Do not tune θ in Arm A after seeing the bias | moving a gate after the result voids the arm; and one constant cannot fix a bias that changes sign through the day (`research/conditional_0dte_armA_results.md:68-69`) |
| Vilkov panel kept outside git | licence: academic replication only (`submit_arm_a.sbatch:28-30`) |
| The 2026-09-11 improvements were written down, not applied | "they are the user's calls" (`trade_analysis/hpc/README.md:347-351`) |

Reason not recorded: why the H100 was chosen for the original TFT run, and anything about
how that run was configured (no job script survives in the tree).

---

## 6. Records, formats, invariants, frozen things

**Must not be touched or peeked at:**

1. **The 2024 block of the SPY 0DTE frame — 151 sessions, 9,030 rows.** Opened once, only
   by §6 of `research/vrp_preregistration.md:169-181`, only after a model beats the best
   HAR-family benchmark on 2023 with Holm-corrected p < 0.05. That gate has never passed.
   The conditional pre-registration repeats it: "No outcome here authorises touching 2024"
   (`research/conditional_0dte_preregistration.md:153-154`).
   - **The seal is a convention, not a lock.** The 2024 session files, with their answers
     in them, are built and sit in `data/vrp/` alongside the rest **(inferred from the
     held-out row counts printed in `research/vrp_baseline_results.md:11`)**. The guards
     are: `cost_model.py:174` filters `year < 2024`; `zeroshot_baseline.py:498-499` does
     not offer `heldout` as a choice; every other script scores only
     `parts["validation"]`. Loading the folder into a notebook and calling `.describe()`
     is a peek.
   - No script in `trade_analysis/hpc/` scores on the held-out block at all. Spending it
     needs new code.
2. **The pre-registrations**, kept verbatim: `vrp_preregistration.md` (status line updated
   only, `:3-6`) and `conditional_0dte_preregistration.md`. §9 of the first forbids adding
   a feature after seeing a result, reporting the best of several horizons, and reporting a
   gross number as the headline (`:211-219`).
3. **Arm B's constants**: top tercile, 30-minute horizon, ≥50 sessions, ≥1,000 trades,
   t > 2.0 (`conditional_vrp.py:59-63`). "If you find yourself wanting to change the
   tercile, the horizon or the evaluation window, that is a new pre-registration, not an
   edit to this file" (`:6-8`).
4. **Arm A's constants**: θ = 0, minimum history 250, ATM band 0.0015
   (`arm_a_ssd.py:75-77`).
5. **The raw archives** under `Desktop\data\raw` and the Vilkov panel under
   `Documents\research_data` — irreplaceable; protected by a deny-list in
   `trade_analysis/scripts/tidy.py:53-63` after an age-based cleanup nearly deleted 2.3 GB
   of them (commit `48ebea2`).
6. **`live_lab_data/`** — untouched by this leg by design (`vrp_preregistration.md:208-209`).

**Closed, do not re-run** (per `CLAUDE.md` and the result files): the unconditional short
0DTE straddle; "does implied variance beat HAR" (it does not); the diurnal question.

**Formats.**

- *Frame*, `data/vrp/SPY_<date>.parquet`, one row per origin. Columns written by
  `build_vrp_dataset.py:276-338`: `date`, `t`, `rv_from_open`, `rv_30m`, `rv_5m`,
  `bipower_30m`, `quarticity_30m`, `rv_prev_day`, `rv_prev_5`, `rv_prev_22`,
  `vix_prev_close`, `minutes_since_open`, `minutes_to_close`, `dow`, `spot`, `iv_atm`,
  `iv_var_atm`, `straddle_spread_bp`, `n_strikes`, `risk_reversal`, `butterfly`,
  `rv_fwd_30`, `rv_fwd_60`, `vrp_30`, `vrp_60`. All variances are annualised (`:191-200`).
- *`manifest.json`*: symbol, interval, bars per day/year, horizons, and a per-session row
  count with a reason when zero (`:405-414`).
- *`data/cost_model_trades.parquet`*: `date`, `t`, `strike`, `spot`, `entry_mid`,
  `exit_mid`, `entry_bid`, `exit_ask`, `gross`, `net`, `spread_cost`, `entry_spread_bp`
  (`cost_model.py:130-139`). Prices per share.
- *Arm A*: `trades_<HHMM>.parquet`; `result_<HHMM>.json` with summary and a provenance
  block — git SHA, modified tracked files, package versions, hostname, every `SLURM_*`
  variable, the three constants (`arm_a_ssd.py:129-143`, `:300-311`).
- *Sweep cell* (never produced): JSON with features, parameter count, optimiser, best
  validation QLIKE, table, seed, DM vs benchmark (`train_vrp.py:172-191`).

**Invariants worth protecting.**

- `vrp_30` and `vrp_60` are implied minus the *future* realised value, so they contain the
  answer. They must never be model inputs (`ceiling.py:42-48`, `:61`).
- The 7-step horizon for Chronos (`zeroshot_baseline.py:148`) — verified exact on
  40,541 of 40,541 pairs; 6 matches 3 of 41,310 (`:27-32`).
- HAR-RV on validation must read **0.430642**. `fetch_zeroshot.sh:41-63` treats that
  number as the fingerprint of an unchanged environment.

---

## 7. Failure modes, incidents, traps

**Already bit, with the guard that now exists.**

| what happened | cost | guard |
|---|---|---|
| Scratch wiped | TFT source believed lost | everything under `$HOME`; README warning |
| Purge policy recorded wrongly as "45 days of no access" | would have lost work again | corrected in `hpc/README.md`, `setup_env.sh:15-17`, `submit_sweep.sbatch:39-45` |
| Halt bar → `log(0)` | 193 of 769 sessions silently dropped; the first published benchmark (0.187365) void | `build_vrp_dataset.py:164`; `clean()` prints what cleaning cost (`har_baseline.py:204-207`) |
| Truncated month `SPY_2020-02` | ten sessions at the start of COVID lost | per-day fallback to the 1-minute archive, verified identical (`build_vrp_dataset.py:104-115`) |
| 1-minute archive hole | held-out capped at 5 of 151 | move to 5-minute |
| Missing exp(s²/2) | false headline "option market beats HAR, p=0.0222", committed as `6037995` | `predict_har`; corrected in place, wrong number left visible (`vrp_baseline_results.md:55-77`) |
| **The stale p=0.0222 lived on in an agent memory file for two days** and was used to brief the literature survey | `research_variance_sota.md` spends its length explaining a "24.1% anomaly" that is really 4.5% | `vrp_phase0_concentration.md:29-33`; memory now says trust the repo over itself. **Read that survey knowing its premise was wrong.** |
| scipy-missing fallback used `np.math`, removed in numpy 2 | would have crashed exactly when needed | stdlib `math.erf` (`har_baseline.py:143-147`) |
| Silent AdamW under a "muon" label | 12 duplicate cells | refuse (`train_vrp.py:75-88`); probe (`submit_sweep.sbatch:75-96`) |
| Chronos API renamed `context` → `inputs` | one V100 allocation, 46 seconds in | probe three signatures first (`zeroshot_baseline.py:190-228`) |
| Fed logs, exponentiated the mean | geometric mean scored as a mean | feed levels (`zeroshot_baseline.py:98-106`) |
| Clamped a negative quantile to 1e-12 | forecast of 3.7e+05; RMSE 6.7e+11 | drop, sort, cap sigma (`zeroshot_baseline.py:292-314`) |
| Two audit designs for Chronos that could not be read | days | third design plus a blunt "saturate" control (`zeroshot_baseline.py:41-88`) |
| Concentration ratio divided by a near-zero net | printed 356%, then 185.3% in a second file the same day | report against gross (`dm_concentration.py:169-172`, `conditional_vrp.py:97-105`) |
| Session-mean reported under a per-trade label | headline overstated 4.5× | print both; gate reads per-trade (`conditional_vrp.py:214-219`) |
| pip killed on the login node | chronos missing at job start | compute node only; import check before the GPU is used |
| Slurm logs written into the clone | provenance gate refused a good run | absolute paths |
| Arm A tasks at two different commits | collector refused | re-run at one SHA (`collect_arm_a.py:44-46`) |
| A commit message described a change the diff did not contain (`d9ef8f4`) | — | corrected by a follow-up commit, not rewritten (`97e742a`) |

**Traps still live for whoever touches this next.**

1. **`submit_arm_a.sbatch` kept two hazards that were later fixed only in the momentum
   sweep's script.** (a) It launches with `srun` (`:109`); an array submitted from inside
   an interactive `srun --pty bash` session inherits CPU-binding variables and fails —
   that killed 15 tasks of job 10373735 (commit `3f6179e`). (b) It runs its grid check
   with bare `python` (`:68-72`) *before* the environment block (`:80-91`); if the default
   `python` lacks numpy the job dies blaming the grid (commit `881171c`). Submit from a
   login node, or port those two fixes first.
2. **`submit_arm_a.sbatch` always takes its "fallback" branch.** It looks for `uv.lock`
   (`:80`); none is tracked (`git ls-files` shows only `requirements.txt` and
   `requirements-dev.txt`). So it runs `conda activate` (`:88-89`) — the thing the other
   three scripts say never to do.
3. **A failed "saturate" control does not fail the job.** `zeroshot_baseline.py:481-489`
   prints FAIL and still returns 0. Compare `conditional_vrp.py:330`, which exits non-zero.
4. **An out-of-range sweep task exits 0 having done nothing** (`submit_sweep.sbatch:100-103`).
   `--array=0-23` on an AdamW-only environment gives 12 COMPLETED tasks with no output, and
   there is no collector for the sweep to notice.
5. **Two declared inputs are empty in every session file**: `rv_5m` (one 5-minute return
   is not enough for the two-return minimum, `build_vrp_dataset.py:198`, `:283`) and
   `vix_prev_close` (never filled, `:385-389`). `ceiling.py:74-79` drops them with a
   printed reason; `train_vrp.py:41` drops them silently.
6. **The audit is one *bar*, not one minute.** `har_baseline.py:217`, `:224` and the
   pre-registration say "one minute"; on the 5-minute frame `shift(-1)` is five minutes.
   `harp_baseline.py:189` says "+1 bar" correctly. The comment at
   `build_vrp_dataset.py:310-311` ("covers [t, t+60s)") is likewise left over from 1-minute bars.
7. **Three different audit-gap numbers are in circulation**: 1.4% (1-minute frame, void),
   1.95% (5-minute, before the lognormal fix), 1.91% (after). `trade_analysis/hpc/README.md:124-127`
   still presents the void 0.187365 / 1.4% as the measurement.
8. **The forecast window and the trade window are offset by one bar.** `rv_fwd_30` covers
   the six bars after the excluded bar `t` (`build_vrp_dataset.py:328-335`); the cost model
   trades from `t` to `t+30` (`cost_model.py:114`). Arm B conditions one on the other.
   Small, and nobody has measured whether it matters **(inferred)**.
9. **"Implied variance" here is the market's average to the close, not for the next 30
   minutes.** It is inverted with time-to-close (`build_vrp_dataset.py:229`, `:321`) and
   compared to a 30-minute target; the comment at `har_baseline.py:169` says "scaled to the
   horizon", which the code does not do. HAR-IV's fitted coefficient absorbs the mismatch;
   the raw `implied_variance` row does not **(inferred)**.
10. **`tft_backtest.py` uses the checkpoint file's modification time as the training
    cutoff** (`:46`). In this worktree every checkpoint is stamped 2026-09-14 18:42 — the
    checkout time, not the training time. Its "out-of-sample" window therefore depends on
    when you cloned **(inferred from `:46` and the directory listing)**.
11. **`trade_analysis/hpc/` cannot run from the test environment.** `requirements-dev.txt:38-43`
    has no pyarrow, torch or scikit-learn. There are zero tests for this package
    (`research/PROGRAMME_REVIEW_2026-09-24.md:164-166`).
12. **Half the home quota is cache.** `~/.cache/huggingface` was measured at 44 GB
    (`trade_analysis/hpc/README.md:225`) against a 75 GB home quota — and this layout puts
    everything in home. The old cleanup script fails on a stale flag (`:226-228`).
13. **Short gamma hides.** Any future read of Arm B must remember 2023 was a calm year:
    its median implied variance is 0.42× the training block's
    (`research/conditional_0dte_armB_results.md:34-39`).

---

## 8. Current state

**Done.**

- Frame built, benchmark established and audited, reproduced on the cluster and on the
  laptop line for line (commit `6e139a4`).
- Cost model: decisive. The unconditional premium is real and untradeable for a taker.
- Zero-shot Chronos-Bolt: loses to HAR at both history lengths.
- "Option market beats HAR": retracted. HAR-IV established as the benchmark.
- Diagnostic that the frame has no forecasting room left.
- Arm A run cleanly (12 of 12, one code version) and reported as a failed replication.
- Arm B run and reported as a fragile pass.
- Stale p=0.0222 corrected in the README and the demo (commit `9686c6b`, 2026-09-24 — the
  last commit touching this package).

**Half-done.**

- **The zero-shot "saturate" control has no recorded result.** It was added on 2026-09-10
  (commit `f451625`) and the sbatch runs it, but no tracked file reports its number.
  `research/PROJECT_REPORT.md:198`, `:383` still says the arm is "running" and blocked on
  "the positive-control audit", while `research/vrp_cost_model_results.md:105-112` treats
  the loss as final. Only the weak "recency" control has a number (0.781812 vs 0.793359,
  better by 1.46%; `zeroshot_baseline.py:67-68`).
- **Five of the seven pre-registered models were never evaluated**: Chronos-2 (Bolt was
  used instead), TimesFM-2.5, fine-tuned Chronos-2, a TFT with real static inputs, and
  LightGBM (`research/vrp_preregistration.md:146-154`). The §6 gate has formally never
  been attempted in full.
- **The conditional question.** Arm A needs a new pre-registration with a properly derived
  bound; not written. Arm B's own pre-registration says a pass "warrants a forward test,
  not a claim" (`conditional_vrp.py:314-316`); no forward test exists.
- **The 2026-09-11 recommendations** — drop the Muon axis, move the sweep to `short`, add
  HAR and tree control cells, ≥5 seeds — are recorded and deliberately unapplied
  (`trade_analysis/hpc/README.md:347-351`). `submit_sweep.sbatch` is still the original
  GPU design.

**Open.**

1. **Hugging Face token in `~/.bash_history` on the shared cluster filesystem —
   revocation unconfirmed.** Found 2026-09-09; the owner was told to revoke it (memory
   note `northeastern-discovery-hpc.md:58-60`). Every later document still lists it as
   unconfirmed and the top-priority item: `research/PROJECT_REPORT.md:391-392`,
   `research/PROGRAMME_REVIEW_2026-09-24.md:168-169`, `:357-358`, and `CLAUDE.md`. No file
   I read records a revocation. I could not check the cluster. No value was seen or copied.
2. **A second exposure of the same variable name, in public git history.** Root
   `README.md:246-258` (audited 2026-08-16) lists `HF_TOKEN` as committed in `README.md`
   at commits `f384d81` and `e17df27`, "Rotated? NO", alongside brokerage and data-vendor
   keys. It also names `trade_analysis/slurm-1612669.out` — a Slurm job log that "captured
   exported env vars" — as present in history (added `e17df27`, deleted `e04d237`). I did
   not open those objects. Whether this is the same token as the one on the cluster is not
   recorded. No token-shaped string exists in the current tree (checked by pattern, count 0).
3. Stray files in the cluster clone (`44604`, `747`, `HAR-RV`, old `.out`/`.err`); the
   cleanup command is in commit `ecd64e3`; whether it was run is not recorded.
4. Whether `~/vrp/results/` was ever copied down. No `results/` exists in this worktree
   and none is tracked; the research documents are the only tracked record of the numbers.
5. No tests for `trade_analysis/hpc/`.
6. Stale statements listed in §9.

---

## 9. Contradictions

| # | one source says | another says | which wins |
|---|---|---|---|
| 1 | **The TFT model definition is lost** — "Confirmed permanently unrecoverable" (root `README.md:206`); "a reverse-engineered reconstruction rather than the real thing" (`hpc/README.md:199-202`) | **A complete TFT definition with a training loop is in the repo**: `huggingface_space/trade_analysis/tft_model.py:75-203`, `:405-461`, added in commit `c0c869d` (2026-09-08). Its fallback paths point at the cluster (`:31-32`), and its hard-coded static inputs `2.0, 1.0, 1.0, 2.0, 50.0` (`:344-357`) are exactly the constant `tft_backtest.py:25` says "matches every scaler's training mean" | Unresolved. It looks like the original, or a close relative, survived on the Space side **(inferred)**. Owner question 1 |
| 2 | Reconstruction wiring: static inputs seed the LSTM's starting state; attention runs on the raw embedding (`trade_analysis/models/tft_model.py:84-98`) | Space wiring: static inputs are added to every time step; attention runs on the LSTM output (`huggingface_space/trade_analysis/tft_model.py:175-189`) | Both load the same weights because layer names and shapes match; they compute different things. The reconstruction's own docstring admits the wiring is a best guess (`models/tft_model.py:15-22`) |
| 3 | Reconstruction labels classifier outputs DOWN, FLAT, UP (`models/tft_model.py:247`) | Space unpacks them UP, DOWN, FLAT (`huggingface_space/trade_analysis/tft_model.py:555`) | Unknown. Moot while the model is a constant |
| 4 | Root `README.md:224-228`: "no working training script"; `enhanced_api.py` "no longer exists" | both exist under `huggingface_space/trade_analysis/` | code |
| 5 | Scratch "purges after 45 days of no access": root `README.md:207`, `research/PROJECT_REPORT.md:117`, the memory index line and the memory note's description field | wiped in full, first Tuesday monthly: `hpc/README.md:187-197`, `setup_env.sh:15-17`, `submit_sweep.sbatch:39-45`, the memory note's body | the corrected version |
| 6 | TFT trained on an H100 (root `README.md:205`) | account history shows H200 most often (memory note `:26-29`) | not resolvable from the repo |
| 7 | Pre-registered family: 7 named models, none an MLP (`vrp_preregistration.md:146-154`) | `train_vrp.py:198` offers only `mlp`, and prints "Holm correction across the 7-model family" (`:182-183`) | the pre-registration; the MLP is outside the family |
| 8 | Pre-registration: Chronos-2 | run: Chronos-Bolt-base (`submit_zeroshot.sbatch:45`) | a substitution, not recorded as a deviation anywhere I found |
| 9 | Pre-registration target uses one-minute returns (`vrp_preregistration.md:63`) | frame is 5-minute | documented correction (`vrp_baseline_results.md:110-120`) |
| 10 | Pre-registration features 3 and 16 (5-minute RV; prior-day VIX) | both columns are 100% empty | code; see §7 trap 5 |
| 11 | `setup_env.sh:38-39` and its printed "absent -> AdamW" (`:55`): Muon "falls back to AdamW and says so" | `train_vrp.py:84-88` refuses | code |
| 12 | `hpc/README.md:236-242`: build a `venv` at `~/venvs/vrp` | `setup_env.sh:26` builds conda env `vrp`; all sbatch files use `~/.conda/envs/vrp` | scripts |
| 13 | `hpc/README.md:209`: `git clone <repo> repo` plus a symlink | everything else: `~/vrp/rep` | `rep` (commit `26330e7`: "The clone landed at ~/vrp/rep") |
| 14 | `submit_zeroshot.sbatch:4`: run the fetch on the "login node" | `fetch_zeroshot.sh:5`, `:32-39`: compute node only, refuses on login | the fetch script |
| 15 | Commit `7d32fa4`: compute nodes "have no outbound network" | commit `1f3036d` and `submit_zeroshot.sbatch:59-63`: they do | the later one |
| 16 | `hpc/README.md` steps numbered 0, 1, 2, 4, 5; "Files" table lists 4 files | 19 files; step 3 (zero-shot) missing | stale README |
| 17 | `PROJECT_REPORT.md:171`, `:383`: zero-shot arm "running" | `vrp_cost_model_results.md:105-112`: run and lost | the later text; but see §8 on the missing control |
| 18 | Conditional pre-registration counts Bevilacqua & Hizmeri as a 0DTE replication (`conditional_0dte_preregistration.md:19-22`) | memory note `vilkov-0dte-dataset.md:54-75`: that paper is one-month maturity; the four are Almeida et al., Dim/Eraker/Vilkov, corrected Vilkov, and the project's own cost model | I could not check the paper. Pre-registrations are kept verbatim, so the doc was not edited |
| 19 | Survey: transfers must go through `xfer.discovery.neu.edu` (`research_hpc_setup.md:681-683`); put work in `/projects` (`:41`) | repo copies to the login host and keeps everything in `$HOME` | what was done evidently worked for Arm A **(inferred)**; the survey's advice was not adopted and the reason is not recorded |
| 20 | `PROGRAMME_REVIEW_2026-09-24.md:164`: "18 files" in `trade_analysis/hpc/` | 19 tracked including the empty `__init__.py` | trivial |

One nuance rather than a contradiction: "2024 was never touched" is true of the SPY frame.
Arm A's table, on a different dataset, includes S&P 500 index sessions through 2024-05
(`research/conditional_0dte_armA_results.md:3-5`) — the same calendar days, a different
instrument. And a separate pre-registration lists an options arm on the same 769 SPY
sessions as "RUNNABLE" (`research/breakout_options_preregistration.md:31`) without
mentioning the seal; `trade_analysis/live_lab/breakout_sweep.py:32` says no option price
was used there.

One observation, **(inferred)** and offered only as a question: in the Space's training
loop the loss is computed on the median-quantile output alone
(`huggingface_space/trade_analysis/tft_model.py:430`). The three-way classifier that
produces `gap_probability` and the direction (`:549-559`) is not in the loss. An untrained
three-way classifier gives roughly 1/3 each, and `(up + down) × 100` would then sit near
66.7 — the model's measured output was 63–70 for every ticker in every regime
(`research/PROJECT_REPORT.md:299-302`). If that loop is what trained the checkpoints, the
"collapse" may be a head that was never trained. Not measured; do not quote as a finding.

---

## 10. Questions only the owner can answer

1. Is `huggingface_space/trade_analysis/tft_model.py` the code that trained the
   checkpoints in `trained_models/`? If so the "lost source" story needs correcting in
   three places, and the reconstruction's different wiring is wrong.
2. Has the Hugging Face token found in the cluster's shell history been revoked? Is it the
   same token as the `HF_TOKEN` the root README says is in public git history at
   `f384d81` / `e17df27`? Were the other keys in that table rotated?
3. Is the cluster account still active, and for how long? Everything this leg kept lives
   in a student home directory.
4. Do `data/vrp/` and `data/cost_model_trades.parquet` still exist on the laptop? Were
   `~/vrp/results/` and `~/vrp/data/` ever copied down from the cluster?
5. What did the "saturate" control print? If it was never read, the zero-shot result is
   resting on the weak control only.
6. Is this leg closed for good? Its only unresolved thread is a rule for *selling*
   straddles, and `CLAUDE.md` says you trade naked calls and puts and do not want other
   structures offered.
7. If it is closed, is the 2024 block now free for a different pre-registered question —
   and may the breakout pre-registration's "SPY 0DTE" arm read it?
8. Should the unapplied 2026-09-11 changes to the sweep (drop Muon, CPU partition, control
   cells) be applied, or should `submit_sweep.sbatch` and `train_vrp.py` be marked dead?
9. Keep the TFT panel in the demo and 19 MB of checkpoints in git, or retire them?
10. Should the stale "45 days" wording in the root README and `PROJECT_REPORT.md` be
    corrected in place?
