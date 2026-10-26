"""Cross-dataset generalisation FEMTO <-> XJTU-SY: raw vs per-condition normalisation, conformal under shift,
and recalibration with a few labelled target bearings."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdm import eval as ev
from pdm import models, pipeline
from pdm import uncertainty as unc

CAP, SEEDS, K_RECAL, DRAWS = models.CFG["rul_cap_life_fraction"], [0, 1, 2], 3, 10
tf, ef = pipeline.load("femto")
tx, ex = pipeline.load("xjtu")
T = pipeline.add_rul_targets(pd.concat([tf, tx], ignore_index=True))
E = np.vstack([ef, ex])
cols = pipeline.feat_cols(T)
T["bearing"] = T.dataset + "/" + T.asset_id
T["cond_key"] = T.dataset + "/" + T.condition.astype(str)


def per_condition_norm(T, E):
    """Label-free z-score of features and embeddings within each (dataset, condition) group."""
    T, E = T.copy(), E.copy()
    for _, i in T.groupby("cond_key").indices.items():
        X = T.iloc[i][cols].to_numpy(float)
        T.iloc[i, [T.columns.get_loc(c) for c in cols]] = (X - X.mean(0)) / (X.std(0) + 1e-9)
        E[i] = (E[i] - E[i].mean(0)) / (E[i].std(0) + 1e-9)
    return T, E


variants = {"raw": (T, E), "per_condition_norm": per_condition_norm(T, E)}
y_all = T.rul.to_numpy()
rows, shift_rows = [], []
dsets = {"femto": np.where(T.dataset == "femto")[0], "xjtu": np.where(T.dataset == "xjtu")[0]}


def fit_predict(Tv, Ev, mode, tr, te, seed):
    fz = models.Featurizer(mode, cols, seed=seed).fit(Ev, tr)
    m = models.lgbm("reg", seed).fit(fz.transform(Tv, tr), y_all[tr])
    return fz, m, np.clip(m.predict(fz.transform(Tv, te)), 0, CAP)


for vname, (Tv, Ev) in variants.items():
    for src, tgt in (("femto", "xjtu"), ("xjtu", "femto")):
        tr, te = dsets[src], dsets[tgt]
        assert set(T.bearing[tr]).isdisjoint(T.bearing[te])
        for mode in ("sensor", "image", "fused"):
            for seed in SEEDS:
                _, _, p = fit_predict(Tv, Ev, mode, tr, te, seed)
                rows.append({"norm": vname, "setting": f"{src}->{tgt}", "model": mode, "seed": seed,
                             **ev.rul_metrics(y_all[te], p, 100)})
        # in-domain reference: leave-one-bearing-out inside the target dataset, same features (no covariates)
        for mode in ("sensor", "image", "fused"):
            ps = np.zeros(len(te))
            for b in np.unique(T.bearing.to_numpy()[te]):
                itr = te[T.bearing.to_numpy()[te] != b]
                ite = te[T.bearing.to_numpy()[te] == b]
                _, _, p = fit_predict(Tv, Ev, mode, itr, ite, 0)
                ps[np.searchsorted(te, ite)] = p
            rows.append({"norm": vname, "setting": f"{tgt} in-domain LOBO", "model": mode, "seed": 0,
                         **ev.rul_metrics(y_all[te], ps, 100)})
        print("done", vname, src, "->", tgt, flush=True)

res = pd.DataFrame(rows)
pipeline.save_table(res, "generalisation_runs")
gen = res.groupby(["norm", "setting", "model"])[["rmse", "mae", "nasa"]].agg(["mean", "std"])
gen.columns = ["_".join(c) for c in gen.columns]
pipeline.save_table(gen.reset_index(), "generalisation")

# conformal under shift + recalibration (fused model, per-condition norm, 90% abs-residual intervals)
Tv, Ev = variants["per_condition_norm"]
alpha = 0.1
for src, tgt in (("femto", "xjtu"), ("xjtu", "femto")):
    s_idx, t_idx = dsets[src], dsets[tgt]
    s_bear, t_bear = T.bearing.to_numpy()[s_idx], T.bearing.to_numpy()[t_idx]
    for seed in SEEDS:
        cal_b = np.random.default_rng(seed).choice(np.unique(s_bear), 4, replace=False)
        s_tr, s_cal = s_idx[~np.isin(s_bear, cal_b)], s_idx[np.isin(s_bear, cal_b)]
        fz, m, _ = fit_predict(Tv, Ev, "fused", s_tr, t_idx[:1], seed)
        pred = lambda i: np.clip(m.predict(fz.transform(Tv, i)), 0, CAP)
        p_scal, p_t = pred(s_cal), pred(t_idx)
        lo, hi = unc.abs_residual_interval(p_scal, y_all[s_cal], p_t, alpha)
        cv, w = unc.coverage(y_all[t_idx], np.clip(lo, 0, CAP), np.clip(hi, 0, CAP))
        shift_rows.append({"setting": f"{src}->{tgt}", "calibration": "source bearings (no recalibration)", "seed": seed,
                           "coverage": cv, "width": w})
        for d in range(DRAWS):
            rb = np.random.default_rng(100 * seed + d).choice(np.unique(t_bear), K_RECAL, replace=False)
            rmask = np.isin(t_bear, rb)
            lo, hi = unc.abs_residual_interval(p_t[rmask], y_all[t_idx][rmask], p_t[~rmask], alpha)
            cv, w = unc.coverage(y_all[t_idx][~rmask], np.clip(lo, 0, CAP), np.clip(hi, 0, CAP))
            shift_rows.append({"setting": f"{src}->{tgt}", "calibration": f"{K_RECAL} labelled target bearings", "seed": seed,
                               "coverage": cv, "width": w})
sh = pd.DataFrame(shift_rows).groupby(["setting", "calibration"]).agg(
    coverage=("coverage", "mean"), coverage_std=("coverage", "std"), width=("width", "mean")).reset_index()
sh["nominal"] = 1 - alpha
pipeline.save_table(sh, "generalisation_coverage")
pipeline.log_run("cross_dataset", {"seeds": SEEDS, "recal_bearings": K_RECAL})
print(gen.to_string()); print(sh.to_string())
