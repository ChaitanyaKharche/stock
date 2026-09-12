# HPC sweep setup research — Northeastern Explorer, sub-1M-param RV forecasting

Research date: **2026-09-10**. All post-May-2026 facts were fetched live, not recalled.

**Legend**
- **[V]** VERIFIED — I fetched the primary source and the claim is stated there.
- **[V-2nd]** VERIFIED but from a secondary source (search summary / third-party doc), not the vendor's own page.
- **[U]** UNCERTAIN — inference, or the source is a blog / could not be confirmed on a primary page.

**Bottom line up front:** three of your five questions resolve to "stop doing the thing." Muon should be
deleted from the sweep, the GPU should be deleted from the job, and the biggest actual risk in the whole
setup is one you did not ask about — your `/scratch` purge model is wrong in a way that will destroy data
again.

---

## 0. The two facts that change the plan

### 0.1 `/scratch` is NOT a 45-day-since-access purge. It is a total wipe, monthly. **[V]**

Your memory file says "PURGES AFTER 45 DAYS of no access." The currently published policy says something
much worse:

> "All files on /scratch will be purged during the monthly maintenance window (the first Tuesday of the month)"
> — https://rc.northeastern.edu/scratch-space-policy/ (also at https://rc.northeastern.edu/policies-scratch-space-policy/)

There is **no access-time condition**. A file you created and read yesterday is deleted on the first
Tuesday regardless. Quota is "20TB for storage and 20 million files (inodes) per user" and it is "not
backed up." The docs add: **"Do not store files on /scratch."**

Corroborating announcements, all first-Tuesdays: purge scheduled 2026-01-06 and 2026-03-03
(https://rc.northeastern.edu/status-updates/). The RTD best-practices page repeats it:
"Files are not backed up in /scratch and as such important data or scripts need to be transfered quickly
to /projects or /home to be retained"
(https://rc-docs.northeastern.edu/en/explorer-main/_sources/best-practices/scratchpurge.md.txt).

**[U]** One search snippet claimed a "28 day purge policy." I could not find that on any page I fetched;
both live policy pages say monthly/first-Tuesday. Treat 28 days as stale. Either way, 45 is wrong and the
"no access" qualifier is wrong, which is the dangerous part.

**Consequence:** source code, conda envs, and sweep results must live in `/projects`, never `/scratch`.

### 0.2 The `gpu` partition caps you at **4 concurrent jobs**; `short` caps you at **50**. **[V]**

From https://rc.northeastern.edu/partitions:

| Partition | Time (default/max) | Limits | Running jobs/user | Access |
|---|---|---|---|---|
| `short` | 4 h / **48 h** | 1024 cores, 25 TB RAM, `--nodes=2` max | **50** | general |
| `sharing` | 30 min / 60 min | ≤2 nodes | 2 | general |
| `gpu` | 4 h / 8 h | **GPU limit 1** | **4** | general |
| `gpu-short` | 1 h / 2 h | GPU limit 1 | 2 | general |
| `gpu-interactive` | 1 h / 2 h | GPU limit 1 | 2 | general |
| `multigpu` | 12 h / 24 h | 8 GPUs | 4 | **approval required** |
| `long` | 1 d / 5 d | 1024 cores | 25 | **approval + checkpoint justification** |

A 24-cell array on `gpu` runs **6 sequential waves of 4**. The same array on `short` runs in **one wave**.
This alone decides the GPU question before any FLOP argument is made.

H200 request syntax is `--gres=gpu:h200:1`
(https://rc-docs.northeastern.edu/en/explorer-main/gpus/quickstart-h200.html). On `gpu-interactive`,
"requesting more than 1 GPU will cause your request to fail."

---

## 1. Muon optimizer — current state, September 2026

### 1.1 It is real, public, and shipped. Your torch 2.6 diagnosis was right. **[V]**

- Current stable PyTorch is **2.14.0, released 2026-09-02** (https://pypi.org/project/torch/). Release
  history from the same page: 2.13.0 (2026-07-08), 2.12.1 (2026-06-17), 2.12.0 (2026-05-13), 2.11.0
  (2026-03-23), 2.10.0 (2026-01-21), 2.9.1 (2025-11-12), **2.9.0 (2025-10-15)**.
- `docs.pytorch.org/docs/stable/...` redirects to `/docs/2.14/...`, confirming 2.14 is stable. **[V]**
  (Note: https://docs.pytorch.org/docs/versions.html still lists "v2.9.0 (stable)" — that page is stale.
  Do not use it as your version oracle.)
- Muon became **public API in 2.9.0**, not later. Verified in the tag itself:
  https://github.com/pytorch/pytorch/blob/v2.9.0/torch/optim/__init__.py contains
  `from torch.optim._muon import Muon as Muon`, sets `Muon.__module__ = "torch.optim"`, and lists
  `"Muon"` in `__all__`. The `_muon.py` filename is just the implementation module; the symbol is public.

So on torch **2.6.0+cu124** there is no `torch.optim.Muon`, and any code that reached AdamW under a "muon"
label did so through its **own** fallback (`getattr(torch.optim, "Muon", AdamW)`, a `try/except
ImportError`, or a string-dispatch dict with a default). Fix that fallback, not just the torch version —
see §3.4.

### 1.2 Exact API (torch 2.14) **[V]**

https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html

```python
torch.optim.Muon(
    params,
    lr=0.001,
    weight_decay=0.1,
    momentum=0.95,
    nesterov=True,
    ns_coefficients=(3.4445, -4.775, 2.0315),
    eps=1e-07,
    ns_steps=5,
    adjust_lr_fn=None,
)
```

`adjust_lr_fn` accepts `"original"` (default), `"match_rms_adamw"`, and `"spectral_unclamped"`. The
`spectral_unclamped` option is **new in 2.14** **[V-2nd]** (2.14 release-notes summary; the 2.14 doc page
lists all three, which is consistent). `match_rms_adamw` lets Muon "directly reuse the learning rate and
weight decay tuned for AdamW" — useful if you want a fair one-knob comparison.

### 1.3 Correct usage — and a genuinely helpful loud failure **[V]**

The documented rule, quoted from the docs:

> "Muon is an optimizer for 2D parameters of neural network hidden layers. Other parameters, such as bias,
> and embedding, should be optimized by a standard method such as AdamW."

Official example (https://github.com/pytorch/pytorch/blob/main/torch/optim/_muon.py):

```python
muon_params  = [p for p in model.parameters() if p.ndim == 2]
other_params = [p for p in model.parameters() if p.ndim != 2]
optim_muon  = torch.optim.Muon(muon_params, lr=0.02, momentum=0.95)
optim_adamw = torch.optim.AdamW(other_params, lr=3e-4, weight_decay=0.01)
```

**Good news for your fail-loudly requirement:** the constructor **raises `ValueError`** if any parameter
has `ndim != 2`, with message `"Muon only supports 2D parameters whereas we found a parameter with size:
{p.size()}"`. It also raises `RuntimeError` for complex parameters and sparse gradients. So
`Muon(model.parameters())` on an MLP with biases/LayerNorm **cannot** silently no-op — it explodes. That
is the one place in this stack that already behaves the way you want.

Keller Jordan's canonical post (https://kellerjordan.github.io/posts/muon/) adds that **output/classifier
heads** should also stay on AdamW — that exclusion is "purely empirical," while the embedding exclusion is
motivated by modular norm theory. Note that `ndim == 2` filtering does **not** exclude your output head;
if you want the canonical recipe you must exclude it by name.

**[U]** Neither the 2.14 doc page nor the 2.9/2.14 release blogs state `torch.compile` support or
distributed support for Muon. **[V-2nd]** search results say Muon "does not yet support distributed
training" as of 2.9 and that this appears unchanged in 2.14. Irrelevant for you (single device), but do
not assume compile-friendliness.

### 1.4 Does it help at sub-1M params on tabular data? Essentially no — and it costs 3×. **[V]**

This is the decisive evidence, and it is almost exactly your setting: **"Benchmarking Optimizers for MLPs
in Tabular Deep Learning"**, https://arxiv.org/html/2604.15297v1 — 15 optimizers (AdamW, SGD-momentum,
NAdamW, RAdam, ADOPT, Adan, AdaBelief, Cautious AdamW, AdEMAMix, Lion, Signum, SOAP, **Muon**,
Schedule-Free AdamW, AdamW+EMA) across 17 tabular datasets, **including regression** ("for regression, we
normalize the labels for training and use mean squared error").

Findings:

- Muon is the most *consistent* winner: "Muon consistently outperforms AdamW, and thus should be considered
  a strong and practical choice for practitioners and researchers, **if the associated training efficiency
  overhead is affordable**."
- But the effect size on a plain MLP is **+0.32% mean Δscore** over AdamW.
- And the cost is **"about 3.03× more tuning time than AdamW on average (46.9h vs 15.5h total for all 17
  datasets)."**

So: **+0.32% for 3× the compute budget.** On 769 sessions of HAR features, +0.32% is far inside your noise
floor. You would be spending 3× your sweep budget to measure something you cannot resolve.

Supporting evidence that the LLM-scale claims do not transfer:

- Keller Jordan's own headline numbers are **CIFAR-10 3.3 → 2.6 A100-seconds**, **NanoGPT 1.35×**, and
  **1.5B params: 10 vs 13.3 8×H100-hours**. The "2×"-flavoured claims live at the 1B+ end. **[V]**
- An independent study (https://github.com/E0NIA/muon-optimizer-study) found Muon won 2 of 3 small
  pretraining cases (13–17% perplexity, reaching AdamW's best "~1.5–1.7× sooner") but **tied on the
  large-vocab model**, with the explanation that Muon's "advantage scales with the fraction of parameters
  it actually governs" — because embeddings/norms/biases/heads stay on AdamW. **[V]** Caveat from the
  source itself: single-seed runs.
- The one robust small-scale benefit is **LR robustness**: "~8× more LR-robust," best-val shifting 0.08
  across a 16× LR range vs 0.7 for AdamW. **[V]** That is a real property, but it *reduces* the value of a
  sweep rather than justifying a sweep axis.

**Verdict on Area 1: delete the optimizer axis.** Run AdamW. This halves the grid from 24 to 12, removes
the entire torch-2.9-vs-2.6 problem, and removes the duplicate-cell hazard by removing the thing that
duplicated. If you keep Muon for completeness, keep it as **one** extra cell with `adjust_lr_fn="match_rms_adamw"`
so it reuses the AdamW LR, not as a factor crossed with everything else.

### 1.5 The prior you should be arguing with instead **[V]**

Before spending a sweep on this: the realized-volatility literature says a properly fitted **HAR OLS** is
the thing that wins.

- "HARd to beat: The overlooked impact of rolling windows in the era of machine learning"
  (https://arxiv.org/pdf/2406.08041, ScienceDirect S0169207025000597): "Despite extensive hyperparameter
  tuning, ML models fail to surpass the linear benchmark set by HAR when utilizing a refined fitting
  approach for the latter." **[V-2nd]** — I could only read this via search summary; the arXiv PDF would
  not parse for me.
- "Forecasting realized volatility: Does anything beat linear models?"
  (https://www.sciencedirect.com/science/article/abs/pii/S0927539824000598) — 1,445 US stocks; "There is
  no evidence that nonlinear ML models can statistically outperform linear models." **[V-2nd]**
- "Predicting Realized Variance Out of Sample: Can Anything Beat The Benchmark?"
  (https://arxiv.org/abs/2506.07928, submitted 2025-06-09) — finds gains are economic rather than
  statistical, and depend on training *for the portfolio objective* rather than on forecast error. **[V]**

Given that a 398,854-param TFT already collapsed to its prior, a plausible reading is that this is null #9,
not a capacity or optimizer problem. **Put a correctly-fitted HAR OLS and a GBDT in the grid as cells**, so
the null is measured rather than assumed.

---

## 2. Is a GPU correct here? No.

### 2.1 The crossover is documented and you are below it **[V-2nd]**

https://indepth.dev/posts/2006/en/gpu-overhead-why-mnist-trains-faster-on-cpu — MNIST 784→128→10, ~101K
params, Quadro RTX 5000 + i9-10885H:

| Implementation | s/epoch |
|---|---|
| Keras **GPU** | 6.5 |
| Keras **CPU** | 4.4 |
| Pure NumPy (bs 2048) | 0.84 |
| JAX + JIT (bs 4096) | 0.14 |

Single matmul (32×784 @ 784×128): GPU 109 µs vs CPU 272 µs — the GPU wins the *math* and still loses the
*epoch*. Full-epoch CUDA overhead breakdown: context switching 0.87 s, timing/sync 0.57 s, memory
transfers 0.83 s ≈ 2.7 s of pure overhead. At batch 32 the model used "0.00003% of the GPU's capacity"
with 3 of 48 SMs active.

Their rule of thumb: **"For models with fewer than ~500K parameters and batch sizes under 256, a fast CPU
will be both cheaper and faster."** You are targeting well under 1M params on ~769 sessions of tabular
rows. This is a blog, hence [V-2nd], but the mechanism is corroborated everywhere.

### 2.2 The mechanism: launch overhead, not FLOPs **[V-2nd]**

Per-kernel-launch overhead is "roughly 5–15 microseconds." A tiny MLP step is a chain of dozens of
microsecond-scale kernels, so dispatch dominates. Independent confirmation for small transformers:
https://arxiv.org/pdf/2505.06461 finds CPU execution can exceed GPU for sub-1B models with multiple
threads, and that careful threading offsets "GPU kernel launch overheads and memory-transfer bottlenecks."

PyTorch itself acknowledges the regime. From the 2.14 release blog
(https://pytorch.org/blog/pytorch-2-14-release-blog/): **"For models with many small compiled regions,
fixed per-call cost matters more than graph quality."** 2.14 spent effort trimming exactly this
(`compile_wrapper` avoiding `DispatchKeySet` pybind churn, cheaper `torch._dynamo.disable`, guard creation
skipped for unused inputs). **[V]**

### 2.3 `torch.compile` at this scale: not worth it for a sweep **[V-2nd]**

`mode="reduce-overhead"` (CUDA Graphs) is the right mode for small models, and the wins on inference
workloads are real (2.14–2.16× reported on top of SDPA in one benchmark). But compilation is "up to
several minutes" and "only beneficial when you plan to execute many runs of training at once." For a
sweep where **each cell** is a fresh process, you pay the compile cost **per cell**, and for a tiny model
the compile cost plausibly exceeds the entire training run. Sources:
https://lightning.ai/docs/fabric/stable/advanced/compile.html,
https://github.com/pytorch/pytorch/issues/128424 (reduce-overhead: "very long compile time + GPU memory
continuously to grow").

**If** you keep GPU and **if** cells were long enough to amortise, the correct incantation would be
`torch.compile(model, mode="reduce-overhead", fullgraph=True)` — `fullgraph=True` to make graph breaks a
loud error rather than a silent slow path. That `fullgraph=True` discipline is worth keeping in spirit even
without compile.

### 2.4 NVIDIA MPS / GPU sharing: a dead end here **[V-2nd]**

Packing sweep cells onto one GPU with MPS fails on policy, not physics: "only one user on a system may have
an active MPS server," and requests from other users serialize
(https://docs.nvidia.com/deploy/mps/latest/index.html, and the Slurm-users thread). Combined with
Explorer's `gpu` limit of 1 GPU / 4 jobs, there is no way to get sweep-scale concurrency out of the GPU
partitions.

### 2.5 The better use of the same wall-clock: GBDT and TabPFN on CPU **[V-2nd]**

- **GBDT still leads on tabular in 2026**, with the honest caveat that the margin is often small: a 19-algorithm /
  176-dataset comparison found that "for a surprisingly high number of datasets, either the performance
  difference between GBDTs and NNs is negligible, or light hyperparameter tuning on a GBDT is more
  important than choosing between NNs and GBDTs" (https://arxiv.org/pdf/2305.02997).
- **Cost**: on 250–1250-sample datasets, XGBoost trained in **0.31–1.35 s** and LightGBM in **0.39–1.88 s**
  (https://www.bohrium.com/en/blog/tutorials/xgboost-vs-lightgbm/). Also: "with a small dataset, you should
  not expect many threads to scale well (it will negatively scale)" — so give GBDT cells **1–2 cores**, not 16.
  At 769 rows your entire 24-cell grid is **seconds of CPU**, not GPU-hours.
- **TabPFN**: v2 targets ≤10K rows / 500 features and "predicts in well under a second" on small data
  (https://priorlabs.ai/tabpfn-2). **TabPFN-3 released May 2026**, up to 1M rows / 200 features
  (https://priorlabs.ai/technical-reports/tabpfn-3). Caveat for you: the time-series variant TabPFN-TS
  "requires roughly 30× more inference time" (https://arxiv.org/pdf/2501.02945). Licensing/weights need
  checking before you rely on it.

**Verdict on Area 2: run the sweep on `short`, CPU-only, `--cpus-per-task=2`.** You get 50 concurrent tasks
instead of 4, 48 h instead of 8 h, no CUDA/driver/wheel surface at all, and the whole grid finishes in one
wave. Keep one GPU cell as a *control* if you want to prove GPU ≡ CPU numerically, not as the workhorse.

---

## 3. Slurm job arrays done right

### 3.1 How Slurm actually derives the exit code — this is your silent-failure hole **[V]**

https://slurm.schedmd.com/job_exit_code.html:

- For `sbatch`, the recorded exit code is **"the output of the batch script"** — i.e. the shell's exit
  status, which by default is **the exit status of the last command in the script**.
- "Any non-zero exit code will be assumed to be a job failure and will result in a Job State of FAILED with
  a Reason of 'NonZeroExitCode'."
- The **derived exit code** for a multi-step batch job is "set to the value of the highest exit code
  returned by all of the job's steps (`srun` invocations)"; individual step codes are kept in the job step
  record.
- Exit codes are "an 8 bit unsigned number ranging between 0 and 255." Signals are recorded separately and
  displayed after the code, colon-delimited.

**The trap:** a batch script that runs `python train.py` and then `echo done` (or `cp`, or `date`) reports
**COMPLETED** even when Python died. That is exactly the "confident numbers that measured nothing" failure
mode. Two independent fixes, use both:

1. `set -euo pipefail` at the top — `-e` exits on any non-zero, `-u` errors on unset variables,
   `pipefail` makes a pipeline fail if *any* stage fails (otherwise only the last stage's status counts).
2. Launch the payload with **`srun`**, so it becomes a job *step* with its own recorded exit code, visible
   per-step in `sacct`, and folded into the derived exit code as a max.

### 3.2 Array mechanics **[V]**

https://casrai.org/guides/slurm-job-array and https://slurm.schedmd.com/job_array.html:

- Syntax: `--array=0-31`, stride `--array=1-7:2`, list `--array=1,3,5,7`, throttle `--array=0-99%10`.
- Exported vars: `SLURM_ARRAY_JOB_ID`, `SLURM_ARRAY_TASK_ID`, **`SLURM_ARRAY_TASK_COUNT`**,
  `SLURM_ARRAY_TASK_MIN`, `SLURM_ARRAY_TASK_MAX`.
- Output tokens: `%A` = array job id, `%a` = task index.
- Smallest index is 0; max index is `MaxArraySize - 1`. Default `MaxArraySize` is **1001**, max supported
  **4000001**. Check with `scontrol show config | grep MaxArraySize`.
- `sacct -j <jobid>` gives per-task exit codes; `scancel 20_4` cancels one task.
- Watch for `QOSMaxSubmitJobPerUserLimit` — on Explorer the *running* limits are 50 (`short`) / 4 (`gpu`),
  which throttle rather than reject.

### 3.3 Make the sweep verify its own grid size **[U — pattern, not a cited source]**

There is no Slurm feature for this; it is a code discipline. The reliable shape:

1. A single `build_grid.py` is the **only** definition of the grid. It writes `grid.jsonl`, one JSON object
   per cell, plus a `grid.sha256`.
2. `sbatch` is never hand-edited. Submit with
   `sbatch --array=0-$(( $(wc -l < grid.jsonl) - 1 )) sweep.sbatch`.
3. Inside the task, assert three ways: `SLURM_ARRAY_TASK_COUNT` equals `wc -l grid.jsonl`; the task's index
   is `< len(grid)`; and the re-hash of `grid.jsonl` equals `grid.sha256`. Any mismatch → `sys.exit(2)`.
4. A `collect.py` run afterwards asserts that **exactly** `len(grid)` result files exist and that the set of
   `cell_id`s in the results equals the set in the grid. Missing cell → non-zero exit, loudly.
5. Every result file records a hash of the *resolved* config, so two cells that resolved to identical
   hyperparameters are detectable as duplicates after the fact. (This is precisely what would have caught
   the muon→AdamW collapse: 12 cells, 12 distinct labels, 6 distinct config hashes.)

This mirrors published practice: SurvBench records "the git commit hash, SHA-256 of resolved YAML
configuration, ... optimizer, learning rate, batch size, actual epochs run, early-stop epoch, wall-clock
training time, parameter count, and device" in every JSON result
(https://arxiv.org/pdf/2511.11935) **[V-2nd]**. General guidance: "A run should be uniquely identified by a
tuple of (code hash, data hash, config hash, environment hash)"
(https://www.dailydoseofds.com/mlops-crash-course-part-3/) **[V-2nd]**.

### 3.4 The environment assertion that would have saved the last sweep **[U — my code, mechanism verified]**

Put this at the top of the training entrypoint. No `getattr`, no `try/except`, no default.

```python
import os, sys, json, subprocess, importlib.metadata, torch

def build_provenance(cell):
    return {
        "cell_id": cell["cell_id"],
        "config_sha256": cell["config_sha256"],
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True).strip()),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "python": sys.version,
        "packages": {d.metadata["Name"]: d.version
                     for d in importlib.metadata.distributions()},
        "hostname": os.uname().nodename,
        "device": (torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() else "cpu"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_array_task_count": os.environ.get("SLURM_ARRAY_TASK_COUNT"),
        "seed": cell["seed"],
    }

def make_optimizer(name, model, cfg):
    if name == "adamw":
        return [torch.optim.AdamW(model.parameters(), **cfg)]
    if name == "muon":
        # Hard requirement. No fallback. Ever.
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError(
                f"cell requests muon but torch {torch.__version__} has no "
                f"torch.optim.Muon (needs >= 2.9.0). REFUSING to substitute AdamW."
            )
        hidden = [p for n, p in model.named_parameters()
                  if p.ndim == 2 and not n.startswith(("head.", "out."))]
        other  = [p for n, p in model.named_parameters()
                  if not (p.ndim == 2 and not n.startswith(("head.", "out.")))]
        assert len(hidden) + len(other) == len(list(model.parameters()))
        assert hidden, "muon cell has zero 2D hidden params — grid is wrong"
        return [
            torch.optim.Muon(hidden, lr=cfg["lr"], momentum=0.95,
                             adjust_lr_fn="match_rms_adamw"),
            torch.optim.AdamW(other, lr=cfg["lr"], weight_decay=0.01),
        ]
    raise ValueError(f"unknown optimizer {name!r}")
```

Also assert the *parameter budget* explicitly, since "capacity was never the constraint" is now a
pre-registered claim:

```python
n_params = sum(p.numel() for p in model.parameters())
assert n_params < 1_000_000, f"cell {cell['cell_id']} has {n_params} params (> 1M budget)"
```

### 3.5 Seeds and determinism **[V]**

https://docs.pytorch.org/docs/2.14/notes/randomness.html:

- "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or
  different platforms. Furthermore, results may not be reproducible between CPU and GPU executions, even
  when using identical seeds." — so pin the torch version *and* the device in provenance; you cannot
  compare a GPU cell to a CPU cell bit-for-bit.
- Seed all three: `torch.manual_seed(s)`, `random.seed(s)`, `np.random.seed(s)`.
- `torch.use_deterministic_algorithms(True)` makes PyTorch "throw an error if an operation is known to be
  nondeterministic." With `warn_only=True` it degrades to a warning — **do not use `warn_only=True`**, that
  is the silent-failure default in disguise
  (https://docs.pytorch.org/docs/2.14/generated/torch.use_deterministic_algorithms.html).
- GPU only: `torch.backends.cudnn.benchmark = False`, `torch.backends.cudnn.deterministic = True`.
- DataLoader workers need explicit seeding:

```python
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed); random.seed(worker_seed)

g = torch.Generator(); g.manual_seed(0)
DataLoader(ds, batch_size=bs, num_workers=nw, worker_init_fn=seed_worker, generator=g)
```

- "Deterministic operations are often slower than nondeterministic operations."
- **[U]** `CUBLAS_WORKSPACE_CONFIG` (`:4096:8` or `:16:8`, required for CUDA ≥ 10.2) is **not mentioned** on
  either 2.14 page I fetched, and a PyTorch discussion suggests "exporting this CUBLAS_WORKSPACE_CONFIG env
  var is no longer necessary for deterministic mode" in newer versions
  (https://github.com/pytorch/pytorch/pull/162040). Setting it anyway is harmless. On CPU it is moot.

**Seed as a grid axis, not a global constant.** With ~769 sessions and an expected null, a single seed per
cell cannot distinguish "this config is better" from "this seed was luckier." The tabular-MLP benchmark's
own caveat about single-seed runs applies to you too. Budget ≥5 seeds per config; on CPU this is free.

### 3.6 `--requeue`, checkpointing, and signals **[V-2nd]**

- `#SBATCH --requeue` marks the job as safe to restart.
- `#SBATCH --signal=B:USR1@120` sends SIGUSR1 to the batch script (`B:`) 120 s before the wall limit.
- On preemption Slurm sends SIGTERM and typically allows ~2 min to checkpoint.
- `$SLURM_JOB_ID` is preserved across requeues, so a checkpoint keyed on it resumes correctly.
  (https://docs.mila.quebec/examples/good_practices/checkpointing/index.html,
  https://docs.coreweave.com/products/sunk/run_workloads/handle-slurm-signals)

**Honest assessment for your job:** with sub-1M-param cells on `short` (48 h limit) each finishing in
seconds-to-minutes, checkpoint/restart is over-engineering. The one thing you *do* need from this section is
**idempotence**: each task should skip (exit 0) if its result file already exists and its `config_sha256`
matches, so a requeue never double-writes or half-writes. Write to `result.json.tmp`, `fsync`, then
`os.replace` — a partially written JSON is a silent-failure vector.

### 3.7 Ready-to-adapt sbatch — CPU sweep (recommended)

```bash
#!/bin/bash
#SBATCH --job-name=rv_sweep
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-23%50
#SBATCH --requeue
#SBATCH --output=/projects/<proj>/rv/logs/%A_%a.out
#SBATCH --error=/projects/<proj>/rv/logs/%A_%a.err

set -euo pipefail

PROJ=/projects/<proj>/rv
GRID="$PROJ/grid.jsonl"
OUT="$PROJ/results/$SLURM_ARRAY_JOB_ID"
mkdir -p "$OUT"

# ---- fail loudly on grid/array mismatch -------------------------------
NCELLS=$(wc -l < "$GRID")
if [[ "${SLURM_ARRAY_TASK_COUNT}" -ne "${NCELLS}" ]]; then
  echo "FATAL: array size ${SLURM_ARRAY_TASK_COUNT} != grid size ${NCELLS}" >&2
  exit 2
fi
if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${NCELLS}" ]]; then
  echo "FATAL: task ${SLURM_ARRAY_TASK_ID} out of range for ${NCELLS} cells" >&2
  exit 2
fi
sha256sum -c "$PROJ/grid.sha256"          # non-zero exit kills the task

# ---- pin threads: tiny models scale negatively with many threads ------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1

# ---- environment, pinned; see section 4 -------------------------------
export UV_CACHE_DIR="$PROJ/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$PROJ/.venv"
cd "$PROJ"
uv sync --locked --offline        # FAILS if uv.lock does not match pyproject

echo "=== provenance ==="
uv run python -c "import torch,sys; print('torch',torch.__version__,'cuda',torch.version.cuda)"
git -C "$PROJ" rev-parse HEAD
nvidia-smi -L 2>/dev/null || echo "no GPU (expected on short)"
echo "=================="

# ---- srun so sacct records a per-step exit code -----------------------
srun --export=ALL uv run python -u train_cell.py \
     --grid "$GRID" \
     --index "$SLURM_ARRAY_TASK_ID" \
     --out   "$OUT"

# nothing after srun: the script's last command IS the payload
```

Submit and collect:

```bash
NCELLS=$(wc -l < grid.jsonl)
JID=$(sbatch --parsable --array=0-$((NCELLS-1))%50 sweep.sbatch)

# after it drains — exit codes per task, not a vibe check
sacct -j "$JID" --format=JobID%20,State,ExitCode,DerivedExitCode,Elapsed,MaxRSS

# hard gate: this must exit 0 or the sweep did not happen
python collect.py --grid grid.jsonl --results "results/$JID" --require-all
```

### 3.8 If you insist on a GPU control cell

```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1     # or gpu:a100:1, or gpu:1 for any
#SBATCH --time=01:00:00
#SBATCH --array=0-23%4        # %4 mirrors the real 4-job cap; do not fight it
```

and add a hard device assertion so a CPU-fallback cell cannot masquerade as a GPU result:

```python
if cell["device"] == "cuda":
    assert torch.cuda.is_available(), "cell requested cuda but none visible — REFUSING to run on CPU"
```

---

## 4. Reproducible environments on a shared HPC filesystem

### 4.1 Why conda gave you torch 2.6.0+cu124 **[V]**

Explorer's documented conda is **`module load anaconda3/2024.06`**
(https://rc-docs.northeastern.edu/en/explorer-main/_sources/software/packagemanagers/conda.md.txt). A
mid-2024 Anaconda distribution plus conda-channel PyTorch packaging lag is exactly how you land on a 2024-era
torch in 2026. Conda's SAT solver treats your `torch` request as a soft constraint to be satisfied against
whatever channel content exists — **it will happily downgrade rather than fail**. That is the root cause,
and it is structural, not a one-off.

Northeastern's own documented conda rules:
- `conda create --prefix=/<path>/<environment-name> python=3.11`
- `source activate /<path>/<environment-name>`
- "Do NOT automatically initialize conda on startup, as it sometimes interferes with other environments on the HPC" — remove the `conda init` block from `.bashrc`.
- "We recommend avoiding building Conda environments in your `/home`, for its space quota. Instead, Use `/projects`."
- Home-quota page (https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/best-practices/homequota.md): check with `check-quota /home/<username>`, inspect with `du -shc .[^.]* ~/*`, "We advise against using 'pip install' to install packages outside of a conda environment or python virtual environment" (it fills `.local`), and `conda clean --all`.

There is also a filesystem reason to leave conda: "Large Conda environments can easily amount to several
100k individual small files," causing inode exhaustion and metadata storms on parallel filesystems tuned
for >4 MB files (https://docs.mpcdf.mpg.de/bnb/214.html, https://scicomp.ethz.ch/wiki/Conda). **[V-2nd]**
Given /scratch's 20M-inode cap and /home's 75 GB, this is not hypothetical.

### 4.2 `uv` — yes, and PyTorch itself now ships uv instructions **[V]**

- The **official PyTorch 2.9 release blog** gives wheel-variant install instructions using `uv venv` /
  `uv pip install torch` (https://pytorch.org/blog/pytorch-2-9/). That is about as strong a mainstreaming
  signal as exists.
- Astral's own PyTorch integration guide (https://docs.astral.sh/uv/guides/integration/pytorch/) documents
  the indices: `cpu`, `cu118`, `cu126`, `cu128`, `cu130`, `rocm7.2`, `xpu`, and states that with no
  explicit index "PyTorch would be installed from PyPI, which hosts CPU-only wheels for Windows and macOS,
  and GPU-accelerated wheels on Linux (targeting **CUDA 13.0, as of PyTorch 2.11.0**)."
- The pinning pattern, verbatim from that page:

```toml
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = [
  { index = "pytorch-cu130", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchvision = [
  { index = "pytorch-cu130", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
```

`explicit = true` ensures the PyTorch index "is only used for `torch`, `torchvision`, and other
PyTorch-related packages."

- HPC centres are documenting it. Hannover LUIS ships `module load GCCcore/.14.2.0 uv/0.9.22`, warns
  "`$HOME` will fill up fast, and it is quite unsuitable for HPC workloads," and prescribes
  `UV_CACHE_DIR` / `PIP_CACHE_DIR` off `$HOME`
  (https://docs.cluster.uni-hannover.de/doku.php/guide/soft/uv). **[V]** GWDG documents Python
  environments similarly (https://docs.hpc.gwdg.de/software_stacks/compilers_interpreters/python/index.html).
  **[U]** I found no Northeastern RC page mentioning uv — their docs still only cover conda. You would be
  running it as a user-installed tool, which is fine (single static binary) but unsupported by RC.

### 4.3 The resolver-cannot-lie recipe

Belt and braces, because "the resolver silently gave me a different version" is your actual scar:

1. **`uv.lock`, committed to git.** Declare a loose constraint (`torch>=2.9`) and let the lock hold the
   exact resolved version + hashes.
2. **`uv sync --locked`** in the sbatch script. `--locked` makes uv **fail** rather than re-resolve if the
   lock does not match `pyproject.toml`. This is the single most important flag in this document. (Add
   `--offline` on compute nodes so a network hiccup fails loudly instead of silently re-resolving.)
3. **Runtime assertion anyway** (§3.4). A lock guarantees what was *installed*; the assertion guarantees
   what was *imported*. `module load` ordering and a stray `.local/lib` can still shadow.
4. **Record `importlib.metadata` versions into every result JSON** (§3.4). Then a wrong environment is
   detectable *after the fact*, from the artefacts alone.
5. **CUDA index must match the driver.** CUDA 13.x needs driver **≥ 580.65.06**; CUDA 12.x needs ≥ 525
   (https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html,
   https://docs.nvidia.com/deploy/cuda-compatibility/). **[U]** I could not find Explorer's driver version
   documented. **Probe it before pinning**: `srun -p gpu-short --gres=gpu:1 --pty nvidia-smi` and read the
   "Driver Version" field. If < 580, pin `cu128` (which last shipped with torch 2.11) or `cu126`. Since
   the recommendation is CPU anyway, `[[tool.uv.index]] url = ".../whl/cpu"` sidesteps this entirely and
   makes the venv ~10× smaller.

### 4.4 Apptainer — correct, but overkill for you **[V-2nd]**

Apptainer is the HPC-native container runtime (runs in the user namespace, no root daemon, native
`--nv` CUDA passthrough): https://researchcomputing.princeton.edu/support/knowledge-base/apptainer,
https://docs.mpcdf.mpg.de/doc/computing/software/containers.html. The 2026 guide "Twelve quick tips for
designing AI-driven HPC workflows" (https://arxiv.org/abs/2606.07491) explicitly names "containerisation
for environment portability, strategic deployment of job arrays, explicit feedback loop mechanics, and
I/O optimisation for small files" as the system-level bottlenecks worth addressing — which is a neat
summary of areas 3, 4 and 5 of your brief. **[V]** (I could only read the abstract; the PDF would not parse.)

For a pure-Python, CPU-only, single-node job, a `uv.lock` gives you the reproducibility a container would,
at a fraction of the build friction, and you avoid needing a build host with root. **[U — my judgement]**
The container becomes clearly worth it if you later need a specific CUDA/cuDNN stack that fights the
cluster's, or if you want the many-small-files problem solved by construction (a SIF is one big file — the
same trick as tarballing a conda env and extracting to node-local `/tmp`, https://wynton.ucsf.edu/hpc/howto/conda-stage.html).

**[U]** I did not verify that Apptainer is installed on Explorer. Check with `module avail apptainer` /
`module avail singularity` before planning around it.

---

## 5. Data staging and purge-proof workflow

### 5.1 Storage map, Explorer **[V]**

| Path | Quota | Backed up | Purged | Use |
|---|---|---|---|---|
| `/home/<user>` | **75 GB** | not stated | no | "small files such as script files, source code, and software installation files" |
| `/scratch/<user>` | 20 TB / 20M inodes | **no** | **ALL files, first Tuesday monthly** | temporary job output only |
| `/projects/<name>` | 35 TB free per PI (across all projects) | not stated | no | "data that is actively being used for research" |
| Archival (disk/tape) | — | — | — | long-term |

Sources: https://rc.northeastern.edu/data-storage-options, https://rc.northeastern.edu/scratch-space-policy/.

**Cross-cluster gotcha [V-2nd]:** "Folder names such as `/projects/foo` on Explorer and `/work/foo` on
Discovery are mapped together. Accessing and making changes to `/work/foo` on Discovery will be reflected
in `/projects/foo` on Explorer, and vice-versa." Same bytes, two names — so `/work` advice in RC blog posts
and `/projects` advice in the docs refer to the same place.

**Do this:** put source code, `uv.lock`, `grid.jsonl`, the parquet inputs, and all results in
`/projects/<proj>/rv/`. Use `/scratch` only for genuinely disposable intermediates, and treat anything
there as gone on the 1st Tuesday. Given a few GB of parquet, `/projects` alone is sufficient — you may not
need `/scratch` at all, which is the cleanest possible defence.

### 5.2 Transfers **[V]**

https://rc-docs.northeastern.edu/en/explorer-main/datamanagement/transferringdata.html:

- **Transfer node: `<username>@xfer.discovery.neu.edu`.** "The HPC has a dedicated transfer node that you
  must use to transfer data to and from the cluster. You cannot transfer data from any other node."
  (Note the hostname says `discovery` even for Explorer.)
- `scp <filename> <username>@xfer.discovery.neu.edu:/scratch/<username>`
- `rsync -av test-data/ <username>@xfer.discovery.neu.edu:/scratch/<username>`
- Within the cluster, rsync from a **compute** node:
  `srun --partition=short --nodes=1 --ntasks=1 --time=01:05:00 --constraint=ib --pty /bin/bash`
  then `rsync -av /scratch/<user>/src /home/<user>/dst`
- Globus "is highly recommended if you need to transfer large amounts of data."

Practical flags for a few GB of parquet **[V-2nd]** (https://docs.hpc.shef.ac.uk/en/latest/hpc/transferring-files.html,
https://hpcc.umd.edu/kb/filexfer/): `rsync -avP --partial` — `--partial` "makes resumption of interrupted
transfers quicker." Skip `-z`: "If you are on a high bandwidth connection, you may not require compression,
since the data transfer speed may be constrained by the time taken to compress the data" — and parquet is
already compressed. For a verifiable copy add `-c` (checksum) on a final confirming pass; it is slow but it
turns "the file arrived" from an assumption into a check.

### 5.3 Node-local `$TMPDIR` **[U for Explorer]**

I could **not** find Explorer/Discovery documentation for node-local scratch or `$TMPDIR`. Elsewhere the
convention is `/scratch/$SLURM_JOB_USER/$SLURM_JOB_ID` or a job-private `$TMPDIR` auto-cleaned at job end
(https://researchcomputing.princeton.edu/faq/how-do-i-use-local-scratc,
https://wynton.ucsf.edu/hpc/scheduler/using-local-scratch.html). Probe it:

```bash
srun -p short --pty bash -c 'echo "TMPDIR=$TMPDIR"; df -h "$TMPDIR" /tmp; ls -ld /tmp'
```

**But:** a few GB of parquet read **once** at process start into RAM is not an I/O problem, and 24 tasks
reading the same few files will hit page cache. Node-local staging is a solution to a problem you do not
have. **This is the part of Area 5 that does not matter.**

### 5.4 Quota monitoring **[V]**

`check-quota /home/<username>` and `du -shc .[^.]* ~/*`
(https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/best-practices/homequota.md).
Add a cheap guard to the sbatch preamble so a quota-full run fails at submit rather than producing
truncated JSON at the end:

```bash
check-quota "/home/$USER" || true          # informational
AVAIL=$(df -Pk "/projects/<proj>" | awk 'NR==2{print $4}')
[[ "$AVAIL" -gt 1048576 ]] || { echo "FATAL: <1GB free on /projects" >&2; exit 3; }
```

---

## 6. Prioritised changes

**P0 — data safety and the silent-failure class**

1. **Move everything off `/scratch` to `/projects/<proj>/`** — code, env, grid, inputs, results. The purge
   is a total monthly wipe, not 45 days of inactivity. Correct the memory file; this belief has already
   cost you source code once and will again. *(§0.1)*
2. **`set -euo pipefail` + `srun` the payload + nothing after it.** Without this, `sbatch` reports
   COMPLETED whenever the last line of the script succeeds, no matter what Python did. *(§3.1)*
3. **Delete every optimizer fallback.** `hasattr(torch.optim, "Muon")` false → `RuntimeError`, never
   AdamW. Same for `cuda requested but unavailable`. *(§3.4)*
4. **Write a provenance block into every result JSON** — git SHA + dirty flag, torch version,
   `torch.version.cuda`, full `importlib.metadata` map, device name, hostname, all four `SLURM_ARRAY_*`
   vars, seed, resolved-config SHA-256, param count. A wrong environment must be detectable from the
   artefact alone, months later. *(§3.3, §3.4)*
5. **`collect.py --require-all`** that exits non-zero if any grid cell is missing a result, and reports any
   two cells whose `config_sha256` collide. This is the check that would have caught the 12 duplicate
   muon/AdamW cells. *(§3.3)*

**P1 — stop paying for things that do not help**

6. **Drop the optimizer axis; run AdamW.** Best available evidence for your exact setting (MLP, tabular,
   regression, 17 datasets, 15 optimizers) is **+0.32% for 3.03× tuning time**. Grid 24 → 12. The
   torch-2.9 problem evaporates with it. *(§1.4)*
7. **Move the sweep to `--partition=short`, CPU, `--cpus-per-task=2`.** 50 concurrent tasks vs 4, 48 h vs
   8 h, no CUDA surface. Set `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK` — tiny models scale *negatively* with
   many threads. *(§0.2, §2.1, §2.5)*
8. **Switch conda → `uv` with `uv sync --locked --offline`, `uv.lock` committed.** `--locked` makes a
   resolver disagreement a hard failure. Pin the index explicitly (`.../whl/cpu` if CPU-only). *(§4.2, §4.3)*
9. **Reinvest the freed budget in seeds, not hyperparameters.** ≥5 seeds per config. With an expected null
   and ~769 sessions, seed variance is the thing that will otherwise be reported as an effect. *(§3.5)*

**P2 — make the result mean something**

10. **Add a correctly-fitted HAR OLS and a LightGBM/XGBoost cell to the grid.** The RV literature says a
    properly fitted HAR is not beaten by ML; GBDT cells cost ~1 s each. Without these controls a null
    result is uninterpretable. *(§1.5, §2.5)*
11. **Grid self-verification**: `grid.jsonl` + `grid.sha256` as the single source of truth,
    `--array=0-$(($(wc -l < grid.jsonl)-1))`, and in-task assertions against `SLURM_ARRAY_TASK_COUNT`.
    Never hand-edit the `--array` range. *(§3.3)*
12. **Idempotent atomic writes**: skip if a matching result exists; write `.tmp` → `fsync` → `os.replace`.
    Makes `--requeue` safe and kills the partial-JSON vector. *(§3.6)*
13. **Determinism on, `warn_only=False`.** Seed torch/random/numpy; `torch.use_deterministic_algorithms(True)`
    so a nondeterministic op raises instead of warning. Record in provenance that CPU and GPU results are
    not bit-comparable. *(§3.5)*
14. **Probe before pinning**: `scontrol show config | grep MaxArraySize`; `nvidia-smi` driver version if you
    keep any GPU cell; `module avail apptainer`; `echo $TMPDIR` on a compute node. *(§3.2, §4.3, §5.3)*

---

## 7. What turned out NOT to matter

- **Area 1 (Muon) — resolved to "delete it."** The question was well-posed and the answer is clean: it is
  genuinely shipped and public since 2.9.0, the usage pattern is exactly as you described, and it will
  raise `ValueError` rather than silently misbehave on non-2D params. But at your scale it buys +0.32% for
  3× compute. The right move is not to fix the Muon plumbing — it is to remove the axis, which also removes
  the torch-version hazard that motivated the question. **The entire problem dissolves rather than gets solved.**
- **`torch.compile` — skip it.** Per-cell compile cost (up to minutes) exceeds per-cell training time for a
  sub-1M-param model. `mode="reduce-overhead"` would be the right mode if the arithmetic worked; it does not.
- **NVIDIA MPS / GPU packing — not available to you.** One MPS server per user system-wide, and Explorer
  caps you at 1 GPU / 4 jobs regardless. Moot once you are on CPU.
- **Apptainer/Singularity — correct but unnecessary.** A committed `uv.lock` + `uv sync --locked` gives you
  the same guarantee for a pure-Python CPU job without a build host or root. Revisit only if you end up
  needing a CUDA stack that fights the cluster's.
- **Node-local SSD / `$TMPDIR` staging (half of Area 5) — a non-problem.** A few GB of parquet, read once
  into RAM, shared across 24 tasks via page cache. There is no I/O bottleneck to optimise. *(The other half
  of Area 5 — the purge policy — turned out to be the single most important finding in this report.)*
- **`--requeue` / checkpoint-restart machinery — over-engineering.** Cells finish in seconds-to-minutes
  inside a 48 h limit. Keep `--requeue` for free, but invest in **idempotence**, not checkpointing.
- **The `sharing` partition — unusable for this.** 60 min max, 2 running jobs. `short` dominates it on
  every axis.

---

## Appendix: every source cited

**PyTorch / Muon**
- https://pypi.org/project/torch/
- https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html
- https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html (redirects to 2.14)
- https://github.com/pytorch/pytorch/blob/v2.9.0/torch/optim/__init__.py
- https://github.com/pytorch/pytorch/blob/main/torch/optim/_muon.py
- https://github.com/pytorch/pytorch/releases/tag/v2.9.0
- https://pytorch.org/blog/pytorch-2-9/
- https://pytorch.org/blog/pytorch-2-14-release-blog/
- https://docs.pytorch.org/docs/2.14/notes/randomness.html
- https://docs.pytorch.org/docs/2.14/generated/torch.use_deterministic_algorithms.html
- https://github.com/pytorch/pytorch/pull/162040
- https://github.com/pytorch/pytorch/issues/128424
- https://docs.pytorch.org/docs/versions.html (stale — flagged)
- https://kellerjordan.github.io/posts/muon/
- https://github.com/E0NIA/muon-optimizer-study
- https://arxiv.org/html/2604.15297v1 (Benchmarking Optimizers for MLPs in Tabular Deep Learning)

**CPU vs GPU / small models / tabular**
- https://indepth.dev/posts/2006/en/gpu-overhead-why-mnist-trains-faster-on-cpu
- https://arxiv.org/pdf/2505.06461
- https://arxiv.org/pdf/2305.02997 (When Do Neural Nets Outperform Boosted Trees on Tabular Data?)
- https://www.bohrium.com/en/blog/tutorials/xgboost-vs-lightgbm/
- https://priorlabs.ai/tabpfn-2 , https://priorlabs.ai/technical-reports/tabpfn-3
- https://arxiv.org/pdf/2501.02945 (TabPFN-v2 for time series)
- https://lightning.ai/docs/fabric/stable/advanced/compile.html
- https://docs.nvidia.com/deploy/mps/latest/index.html

**Realized variance literature**
- https://arxiv.org/pdf/2406.08041 / https://www.sciencedirect.com/science/article/abs/pii/S0169207025000597
- https://www.sciencedirect.com/science/article/abs/pii/S0927539824000598
- https://arxiv.org/abs/2506.07928

**Slurm**
- https://slurm.schedmd.com/job_exit_code.html
- https://slurm.schedmd.com/job_array.html
- https://casrai.org/guides/slurm-job-array
- https://docs.mila.quebec/examples/good_practices/checkpointing/index.html
- https://docs.coreweave.com/products/sunk/run_workloads/handle-slurm-signals

**Environments**
- https://docs.astral.sh/uv/guides/integration/pytorch/
- https://docs.astral.sh/uv/concepts/indexes/
- https://docs.cluster.uni-hannover.de/doku.php/guide/soft/uv
- https://docs.hpc.gwdg.de/software_stacks/compilers_interpreters/python/index.html
- https://docs.mpcdf.mpg.de/bnb/214.html , https://scicomp.ethz.ch/wiki/Conda
- https://wynton.ucsf.edu/hpc/howto/conda-stage.html
- https://researchcomputing.princeton.edu/support/knowledge-base/apptainer
- https://arxiv.org/abs/2606.07491 (Twelve quick tips for designing AI-driven HPC workflows)
- https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html
- https://docs.nvidia.com/deploy/cuda-compatibility/

**Northeastern RC (primary)**
- https://rc.northeastern.edu/partitions
- https://rc.northeastern.edu/scratch-space-policy/ , https://rc.northeastern.edu/policies-scratch-space-policy/
- https://rc.northeastern.edu/data-storage-options
- https://rc.northeastern.edu/status-updates/
- https://rc-docs.northeastern.edu/en/explorer-main/gpus/quickstart-h200.html
- https://rc-docs.northeastern.edu/en/explorer-main/datamanagement/transferringdata.html
- https://rc-docs.northeastern.edu/en/explorer-main/_sources/best-practices/scratchpurge.md.txt
- https://rc-docs.northeastern.edu/en/explorer-main/_sources/software/packagemanagers/conda.md.txt
- https://github.com/northeastern-rc/rc-public-documentation/blob/master/docs/source/best-practices/homequota.md
- https://rc.northeastern.edu/2025/04/30/all-about-explorer-and-the-new-h200-gpus/

**Provenance practice**
- https://arxiv.org/pdf/2511.11935 (SurvBench per-run JSON provenance)
- https://arxiv.org/html/2505.06558v1 (DataLad / machine-actionable reproducibility for HPC)
- https://www.dailydoseofds.com/mlops-crash-course-part-3/
