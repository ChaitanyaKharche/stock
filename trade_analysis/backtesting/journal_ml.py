"""Methods 1-3 of the journal reverse-engineering study.

Executes research/journal_reverse_engineering_preregistration.md. Every threshold, family
size, model hyperparameter and permutation count in this file was fixed in that document
before anything here was run.

  Q1   SELECTION   is_entry ~ market state          (418 entries vs 3,274 placebos)
  Q2a  OUTCOME     fwd underlying move ~ state      (exit-free, 4 horizons)
  Q2b  OUTCOME     win / option return ~ state      (the money, exit-contaminated)
  M2   per-feature NULL IMPORTANCE (target permutation)
  M3   depth-3 decision tree + permuted-best-leaf control

The leak this file most has to avoid is not lookahead -- it is the feature-availability
leak. Six columns (qty, dte, option spread, trade ordinal, previous trade result) exist only
on rows where he actually traded. Handing those to Q1 would let any model separate entries
from placebos on NaN-ness alone and score AUC 1.0. MARKET is therefore derived by asking
which columns are populated on placebo rows, not by hand.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.dummy import DummyClassifier  # noqa: E402,F401
from sklearn.ensemble import (  # noqa: E402
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: E402

SCRATCH = (r"C:\Users\CHAITA~1\AppData\Local\Temp\claude"
           r"\C--Users-chaitanyakharche-Documents-stock"
           r"\3ad3e562-0324-4d02-bf8b-208ff71f108d\scratchpad")
TABLE = os.path.join(SCRATCH, "journal_features.pkl")
SEED = 20260829
NFOLD = 5
N_PERM_MODEL = 500          # pre-registered
N_PERM_IMPORT = 200         # pre-registered
N_PERM_TREE = 500           # pre-registered
PI_REPEATS_REAL = 20        # pre-registered
PI_REPEATS_NULL = 5         # noisier null => wider => conservative; stated in results
IMPORTANCE_MODEL = "gradboost"   # amendment A1: RandomForest-400 predict cost makes 200
                                 # null refits a ~7h job. Real and null importance use the
                                 # SAME model, so the comparison stays internally valid.

TARGETS = {"is_entry", "win", "opt_ret", "opt_net", "hold_min", "fwd_hishold_bp"}
META = {"symbol", "date", "ts", "direction", "is_etf"}


# ------------------------------------------------------------------------------- models

def clf_models():
    return {
        "logistic": Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler()),
                              ("m", LogisticRegression(max_iter=2000, C=1.0,
                                                       random_state=SEED))]),
        "randomforest": Pipeline([("imp", SimpleImputer(strategy="median")),
                                  ("m", RandomForestClassifier(
                                      n_estimators=400, min_samples_leaf=20,
                                      n_jobs=-1, random_state=SEED))]),
        "gradboost": Pipeline([("m", HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300,
            early_stopping=False, random_state=SEED))]),
    }


def reg_models():
    return {
        "ridge": Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "randomforest": Pipeline([("imp", SimpleImputer(strategy="median")),
                                  ("m", RandomForestRegressor(
                                      n_estimators=400, min_samples_leaf=20,
                                      n_jobs=-1, random_state=SEED))]),
        "gradboost": Pipeline([("m", HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=300,
            early_stopping=False, random_state=SEED))]),
    }


# ------------------------------------------------------------------ out-of-fold machinery

def oof_pred(model, X, y, groups, proba=True):
    """Out-of-fold predictions. Nothing is ever scored on data its model saw."""
    out = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=NFOLD)
    for tr, te in gkf.split(X, y, groups):
        m = _clone(model)
        m.fit(X[tr], y[tr])
        out[te] = m.predict_proba(X[te])[:, 1] if proba else m.predict(X[te])
    return out


def _clone(model):
    from sklearn.base import clone
    return clone(model)


def auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan


def r2_oof(y, p):
    ss = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - p) ** 2)) / ss if ss > 0 else np.nan


# ------------------------------------------------------------------- permutation schemes

def perm_within_session(y, sess_id, rng):
    """Q1 null: shuffle is_entry inside each (symbol, date). Holds the day constant and asks
    only whether the MINUTE he picked was special."""
    out = y.copy()
    for s in np.unique(sess_id):
        m = sess_id == s
        out[m] = rng.permutation(y[m])
    return out


def perm_by_day_block(y, day_id, rng):
    """Q2 null: permute whole day blocks, preserving within-day dependence."""
    days = np.unique(day_id)
    order = rng.permutation(len(days))
    mapping = {d: days[order[i]] for i, d in enumerate(days)}
    out = np.empty_like(y)
    pools = {d: list(np.flatnonzero(day_id == d)) for d in days}
    for d in days:
        src = pools[mapping[d]]
        dst = pools[d]
        take = (src * (len(dst) // len(src) + 1))[:len(dst)]
        out[dst] = y[take]
    return out


# --------------------------------------------------------------------------------- Q1/Q2

def run_classification(name, X, y, groups, sess_id, feats, perm_fn, n_perm, do_null_imp):
    print("\n" + "=" * 92)
    print(name)
    print("=" * 92)
    print("  n=%d  positives=%d (%.1f%%)  groups=%d  features=%d"
          % (len(y), int(y.sum()), 100 * y.mean(), len(np.unique(groups)), X.shape[1]))
    rng = np.random.default_rng(SEED)
    scores = {}
    for mn, mdl in clf_models().items():
        t0 = time.time()
        p = oof_pred(mdl, X, y, groups, proba=True)
        scores[mn] = auc(y, p)
        print("  %-14s OOF AUC %.4f    (%.1fs)" % (mn, scores[mn], time.time() - t0))
    best = max(scores, key=lambda k: scores[k])
    print("  best model: %s  AUC %.4f" % (best, scores[best]))

    mdl = clf_models()[best]
    t0 = time.time()
    null = []
    for i in range(n_perm):
        yp = perm_fn(y, sess_id, rng)
        null.append(auc(yp, oof_pred(mdl, X, yp, groups, proba=True)))
        if (i + 1) % 100 == 0:
            print("    null %d/%d  mean %.4f  (%.0fs)"
                  % (i + 1, n_perm, np.nanmean(null), time.time() - t0), flush=True)
    null = np.asarray(null)
    p_emp = (1 + np.sum(null >= scores[best])) / (1 + len(null))
    print("  permutation null: mean %.4f  sd %.4f  p95 %.4f"
          % (np.nanmean(null), np.nanstd(null), np.nanpercentile(null, 95)))
    print("  REAL %.4f   empirical p = %.4f" % (scores[best], p_emp))
    res = {"scores": scores, "best": best, "auc": scores[best], "p": p_emp,
           "null_mean": float(np.nanmean(null)), "null_sd": float(np.nanstd(null)),
           "null_p95": float(np.nanpercentile(null, 95))}
    if do_null_imp:
        imdl = clf_models()[IMPORTANCE_MODEL]
        print("  (importance model: %s, AUC %.4f -- amendment A1)"
              % (IMPORTANCE_MODEL, scores[IMPORTANCE_MODEL]))
        res["importance"] = null_importance(imdl, X, y, groups, sess_id, feats, perm_fn)
        res["importance_linear"] = null_importance(
            clf_models()["logistic"], X, y, groups, sess_id, feats, perm_fn,
            tag="linear cross-check")
    return res


def null_importance(mdl, X, y, groups, sess_id, feats, perm_fn, tag=""):
    """METHOD 2. Permutation importance on held-out folds, then the same computation on
    label-permuted data to build a per-feature null. A feature counts only if it clears BH."""
    print("\n  METHOD 2 -- per-feature null importance")
    rng = np.random.default_rng(SEED + 1)

    def imp_once(yy, repeats):
        acc = np.zeros(X.shape[1])
        gkf = GroupKFold(n_splits=NFOLD)
        for tr, te in gkf.split(X, yy, groups):
            m = _clone(mdl)
            m.fit(X[tr], yy[tr])
            if len(np.unique(yy[te])) < 2:
                continue
            r = permutation_importance(m, X[te], yy[te], scoring="roc_auc",
                                       n_repeats=repeats, random_state=SEED, n_jobs=-1)
            acc += r.importances_mean
        return acc / NFOLD

    t0 = time.time()
    real = imp_once(y, PI_REPEATS_REAL)
    print("    real importance done (%.0fs)" % (time.time() - t0))
    nulls = np.zeros((N_PERM_IMPORT, X.shape[1]))
    for i in range(N_PERM_IMPORT):
        nulls[i] = imp_once(perm_fn(y, sess_id, rng), PI_REPEATS_NULL)
        if (i + 1) % 25 == 0:
            print("    null importance %d/%d  (%.0fs)"
                  % (i + 1, N_PERM_IMPORT, time.time() - t0), flush=True)
    nm, ns = nulls.mean(0), nulls.std(0)
    z = (real - nm) / np.where(ns > 0, ns, np.nan)
    p = (1 + (nulls >= real).sum(0)) / (1 + N_PERM_IMPORT)
    order = np.argsort(p)
    q = np.minimum.accumulate(
        (p[order] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    bh = np.empty(len(p))
    bh[order] = np.minimum(q, 1.0)
    tab = pd.DataFrame({"feature": feats, "importance": real, "null_mean": nm,
                        "null_sd": ns, "z": z, "p": p, "q_BH": bh})
    tab = tab.sort_values("importance", ascending=False)
    print("\n  %-24s %10s %10s %8s %8s %8s" % ("feature", "import", "null_mu", "z", "p", "qBH"))
    for _, r in tab.head(20).iterrows():
        print("  %-24s %10.5f %10.5f %8.2f %8.3f %8.3f"
              % (r.feature, r.importance, r.null_mean, r.z, r.p, r.q_BH))
    surv = tab[tab.q_BH < 0.05]
    print("\n  features clearing BH q<0.05: %d" % len(surv))
    if len(surv):
        print("   ", ", ".join(surv.feature.tolist()))
    conc = (tab.importance.clip(lower=0).sort_values(ascending=False).head(5).sum()
            / max(tab.importance.clip(lower=0).sum(), 1e-12))
    print("  top-5 share of positive importance: %.1f%%  "
          "(near 5/%d = %.1f%% means evenly spread = learned nothing)"
          % (100 * conc, len(tab), 500.0 / len(tab)))
    return tab


def run_regression(name, X, y, groups, day_id, feats):
    print("\n" + "-" * 92)
    print(name)
    print("  n=%d  mean %.2f  sd %.2f  groups=%d" % (len(y), y.mean(), y.std(),
                                                     len(np.unique(groups))))
    rng = np.random.default_rng(SEED)
    out = {}
    for mn, mdl in reg_models().items():
        p = oof_pred(mdl, X, y, groups, proba=False)
        out[mn] = {"r2": r2_oof(y, p), "auc_sign": auc((y > 0).astype(int), p)}
        print("    %-14s OOF R2 %+.4f   AUC(sign) %.4f" % (mn, out[mn]["r2"],
                                                           out[mn]["auc_sign"]))
    best = max(out, key=lambda k: out[k]["r2"])
    mdl = reg_models()[best]
    null = []
    for _ in range(200):
        yp = perm_by_day_block(y, day_id, rng)
        null.append(r2_oof(yp, oof_pred(mdl, X, yp, groups, proba=False)))
    null = np.asarray(null)
    p_emp = (1 + np.sum(null >= out[best]["r2"])) / (1 + len(null))
    print("    best %s  R2 %+.4f   null mean %+.4f  p=%.4f"
          % (best, out[best]["r2"], null.mean(), p_emp))
    return {"best": best, "r2": out[best]["r2"], "auc_sign": out[best]["auc_sign"],
            "p": p_emp, "null_mean": float(null.mean()), "all": out}


# ------------------------------------------------------------------------------ METHOD 3

def run_tree(name, X, y, groups, sess_id, feats, perm_fn):
    print("\n" + "=" * 92)
    print("METHOD 3 -- depth-3 decision tree: " + name)
    print("=" * 92)
    rng = np.random.default_rng(SEED + 2)
    base = y.mean()

    def best_leaf(yy):
        """Largest out-of-fold rate among leaves holding >=25 held-out rows."""
        gkf = GroupKFold(n_splits=NFOLD)
        agg = {}
        for tr, te in gkf.split(X, yy, groups):
            t = DecisionTreeClassifier(max_depth=3, min_samples_leaf=25,
                                       random_state=SEED)
            Xi = np.nan_to_num(X, nan=0.0)
            t.fit(Xi[tr], yy[tr])
            leaf = t.apply(Xi[te])
            for lf in np.unique(leaf):
                m = leaf == lf
                a = agg.setdefault(lf, [0, 0])
                a[0] += int(yy[te][m].sum())
                a[1] += int(m.sum())
        rates = [(s / n, n) for s, n in agg.values() if n >= 25]
        return max(rates)[0] if rates else np.nan

    real = best_leaf(y)
    t = DecisionTreeClassifier(max_depth=3, min_samples_leaf=25, random_state=SEED)
    t.fit(np.nan_to_num(X, nan=0.0), y)
    print("  full-sample tree (HYPOTHESIS ONLY -- not a rule):")
    for ln in export_text(t, feature_names=list(feats), max_depth=3).splitlines():
        print("    " + ln)
    print("\n  base rate %.3f    best out-of-fold leaf rate %.3f" % (base, real))
    null = np.array([best_leaf(perm_fn(y, sess_id, rng)) for _ in range(N_PERM_TREE)])
    p95 = np.nanpercentile(null, 95)
    p_emp = (1 + np.nansum(null >= real)) / (1 + len(null))
    print("  PERMUTED-LABEL best-leaf distribution (%d fits on pure noise):" % N_PERM_TREE)
    print("    mean %.3f   p50 %.3f   p95 %.3f   max %.3f"
          % (np.nanmean(null), np.nanpercentile(null, 50), p95, np.nanmax(null)))
    verdict = "ABOVE the 95th percentile" if real > p95 else "INSIDE the noise band"
    print("  real %.3f is %s   empirical p = %.4f" % (real, verdict, p_emp))
    return {"real": float(real), "null_p95": float(p95), "p": p_emp,
            "null_mean": float(np.nanmean(null)), "base": float(base)}


# ------------------------------------------------------------------------------------ main

def main():
    df = pd.read_pickle(TABLE)
    df["sess"] = df.symbol + "_" + df.date
    allf = [c for c in df.columns
            if c not in TARGETS and c not in META and not c.startswith("fwd_")
            and c != "sess"]
    plac = df[df.is_entry == 0]
    MARKET = [c for c in allf if plac[c].notna().mean() > 0.99]
    TRADEONLY = [c for c in allf if c not in MARKET]
    print("features total %d   market-state %d   trade-only %d"
          % (len(allf), len(MARKET), len(TRADEONLY)))
    print("  trade-only (excluded from Q1 -- they would leak entry/placebo status):")
    print("   ", ", ".join(TRADEONLY))

    results = {}

    # ---- Q1 SELECTION
    d = df.dropna(subset=MARKET)
    X = d[MARKET].to_numpy(float)
    y = d.is_entry.to_numpy(int)
    groups = d.date.to_numpy()
    sess = d.sess.to_numpy()
    results["Q1"] = run_classification(
        "Q1 SELECTION -- his entry minutes vs same-session placebos",
        X, y, groups, sess, MARKET, perm_within_session, N_PERM_MODEL, do_null_imp=True)
    results["Q1_tree"] = run_tree("Q1 selection", X, y, groups, sess, MARKET,
                                  perm_within_session)

    # ---- Q2 his entries only
    e = df[df.is_entry == 1].copy()
    FE = MARKET + TRADEONLY
    e = e.dropna(subset=FE)
    Xe = e[FE].to_numpy(float)
    ge = e.date.to_numpy()
    print("\n\nQ2 sample: %d entries, %d date clusters, %d features"
          % (len(e), e.date.nunique(), len(FE)))

    print("\n" + "=" * 92)
    print("Q2a OUTCOME, EXIT-FREE -- forward underlying move (aligned, bp)")
    print("=" * 92)
    for h in (5, 10, 15, 25):
        col = "fwd_%dm_bp" % h
        m = e[col].notna().to_numpy()
        results["Q2a_%d" % h] = run_regression(
            "  horizon %d min" % h, Xe[m], e[col].to_numpy(float)[m], ge[m], ge[m], FE)

    print("\n" + "=" * 92)
    print("Q2b OUTCOME, THE MONEY -- his realised option result")
    print("=" * 92)
    yw = e.win.to_numpy(int)
    results["Q2b_win"] = run_classification(
        "  win / loss", Xe, yw, ge, ge, FE, perm_by_day_block, 200, do_null_imp=True)
    results["Q2b_ret"] = run_regression("  option return %", Xe,
                                        e.opt_ret.to_numpy(float), ge, ge, FE)
    results["Q2b_tree"] = run_tree("Q2b win/loss", Xe, yw, ge, ge, FE, perm_by_day_block)

    import pickle
    pickle.dump(results, open(os.path.join(SCRATCH, "journal_ml_results.pkl"), "wb"))
    print("\n\nsaved -> journal_ml_results.pkl")


if __name__ == "__main__":
    main()
