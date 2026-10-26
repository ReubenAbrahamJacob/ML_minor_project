"""C-MAPSS through the same modules (new adapter only): train/test on official splits, FD001-FD004."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdm import eval as ev
from pdm import models, pipeline
from pdm import uncertainty as unc

CAP = 125.0
rows, cov_rows = [], []
for fd in (1, 2, 3, 4):
    t, emb = pipeline.load(f"cmapss{fd}")
    cols = pipeline.feat_cols(t)
    covars = ["set1", "set2", "set3"]
    tr_all = np.where(t.split == "train")[0]
    te = np.where(t.split == "test")[0]              # last window of each test engine, official RUL labels
    y = t.rul.to_numpy()
    eng = t.asset_id.to_numpy()
    assert set(eng[tr_all]).isdisjoint(eng[te])
    for seed in models.CFG["seeds"]:
        # calibration engines held out of training
        cal_e = np.random.default_rng(seed).choice(np.unique(eng[tr_all]), 20, replace=False)
        cal = tr_all[np.isin(eng[tr_all], cal_e)]
        tr = tr_all[~np.isin(eng[tr_all], cal_e)]
        for mode in models.MODES:
            fz = models.Featurizer(mode, cols, covars, seed=seed).fit(emb, tr)
            m = models.lgbm("reg", seed).fit(fz.transform(t, tr), y[tr])
            p = np.clip(m.predict(fz.transform(t, te)), 0, CAP)
            rows.append({"fd": fd, "model": f"lgbm_{mode}", "seed": seed, **ev.rul_metrics(y[te], p)})
            if mode == "fused":
                lo, hi = unc.abs_residual_interval(m.predict(fz.transform(t, cal)), y[cal], p, 0.1)
                cv, w = unc.coverage(y[te], lo, hi)
                cov_rows.append({"fd": fd, "seed": seed, "nominal": 0.9, "coverage": cv, "width": w})
    print("done FD00", fd, flush=True)
res = pd.DataFrame(rows)
pipeline.save_table(res, "cmapss_runs")
s = res.groupby(["fd", "model"])[["rmse", "mae", "nasa"]].agg(["mean", "std"])
s.columns = ["_".join(c) for c in s.columns]
pipeline.save_table(s.reset_index(), "cmapss_summary")
c = pd.DataFrame(cov_rows).groupby("fd")[["coverage", "width"]].mean().reset_index()
pipeline.save_table(c, "cmapss_coverage")
pipeline.log_run("cmapss", {"rul_cap_cycles": CAP, "window": 30})
print(s.to_string()); print(c.to_string())
