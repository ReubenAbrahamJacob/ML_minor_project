"""Task B on FEMTO: RUL regression, leave-one-bearing-out, 5 seeds, baselines + ablations."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdm import eval as ev
from pdm import models, physics, pipeline, splits

CAP = models.CFG["rul_cap_life_fraction"]
t, emb = pipeline.load("femto")
t = pipeline.add_rul_targets(t)
cols = pipeline.feat_cols(t)
groups, y = t.asset_id.to_numpy(), t.rul.to_numpy()
life = t.groupby("asset_id").life_s.first()
rows, preds = [], []

for tr, te, b in splits.logo(groups):
    assert set(groups[tr]).isdisjoint(groups[te])
    train_life = life.drop(b).to_numpy()
    age = t.age_s.to_numpy()[te]
    out = {}
    out["mean_lifetime"] = np.clip(1 - age / train_life.mean(), 0, CAP)
    rem = physics.WeibullLife(train_life).mean_residual(age)
    out["weibull"] = np.minimum(rem / (age + rem), CAP)
    per_seed = {}
    for seed in models.CFG["seeds"]:
        for mode in models.MODES:
            fz = models.Featurizer(mode, cols, covars=["rpm", "load"], seed=seed).fit(emb, tr)
            m = models.lgbm("reg", seed).fit(fz.transform(t, tr), y[tr])
            per_seed.setdefault(f"lgbm_{mode}", []).append(np.clip(m.predict(fz.transform(t, te)), 0, CAP))
        fz = models.Featurizer("sensor", cols, covars=["rpm", "load"], seed=seed).fit(emb, tr)
        m = models.rf("reg", seed).fit(fz.transform(t, tr), y[tr])
        per_seed.setdefault("rf_sensor", []).append(np.clip(m.predict(fz.transform(t, te)), 0, CAP))
    for name, ps in per_seed.items():
        for seed, p in zip(models.CFG["seeds"], ps):
            rows.append({"bearing": b, "model": name, "seed": seed, **ev.rul_metrics(y[te], p, scale=100)})
        out[name] = ps[0]
    for name in ("mean_lifetime", "weibull"):
        rows.append({"bearing": b, "model": name, "seed": -1, **ev.rul_metrics(y[te], out[name], scale=100)})
    for name, p in out.items():
        preds.append(pd.DataFrame({"bearing": b, "model": name, "life_frac": t.life_frac.to_numpy()[te], "true": y[te], "pred": p}))
    print("done", b, flush=True)

res = pd.DataFrame(rows)
pipeline.save_table(res, "femto_rul_runs")
pipeline.save_table(pd.concat(preds), "femto_rul_predictions_seed0")
per_b = res.groupby(["model", "bearing"])[["rmse", "mae", "nasa"]].mean().reset_index()  # mean over seeds
summ = per_b.groupby("model")[["rmse", "mae", "nasa"]].agg(["mean", "std"])
summ.columns = ["_".join(c) for c in summ.columns]
summ = summ.reset_index().sort_values("rmse_mean")
pipeline.save_table(summ, "femto_rul_summary")
piv = per_b.pivot(index="bearing", columns="model", values="rmse")
pv = [{"comparison": f"lgbm_fused vs {o}", "mean_rmse_diff": float((piv.lgbm_fused - piv[o]).mean()),
       "wilcoxon_p": ev.wilcoxon(piv.lgbm_fused, piv[o])} for o in
      ["lgbm_sensor", "lgbm_image", "lgbm_fused_shuffled", "rf_sensor", "weibull", "mean_lifetime"]]
pipeline.save_table(pd.DataFrame(pv), "femto_rul_wilcoxon")
pr = pd.concat(preds)
pr = pr[pr.model == "lgbm_fused"]
pipeline.save_table(pd.DataFrame([{"life_bin": k, "mae": v} for k, v in ev.error_by_life(pr.life_frac, pr.true, pr.pred).items()]),
                    "femto_error_by_life")
pipeline.log_run("femto_rul", {"split": "leave-one-bearing-out", "target": f"min(1-life_frac, {CAP})"})
print(summ.to_string())
print(pd.DataFrame(pv).to_string())
