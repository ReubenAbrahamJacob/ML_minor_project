"""Quantile LightGBM + split conformal on FEMTO (fused model), plus decision layer simulation.

For every held-out bearing: k calibration bearings are drawn from the others (held out of training too).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdm import decision, models, pipeline, splits
from pdm import uncertainty as unc

CAP, ALPHAS, K_CAL = models.CFG["rul_cap_life_fraction"], models.CFG["alphas"], 4
t, emb = pipeline.load("femto")
t = pipeline.add_rul_targets(t)
cols = pipeline.feat_cols(t)
groups, y = t.asset_id.to_numpy(), t.rul.to_numpy()
life = t.groupby("asset_id").life_s.first()
cov_rows, band_rows, dec_rows = [], [], []

for b in np.unique(groups):
    for seed in (0, 1, 2):
        tr, ca, te = splits.with_calibration(groups, b, K_CAL, seed)
        splits.assert_disjoint(groups[tr], groups[ca], groups[te])
        fz = models.Featurizer("fused", cols, covars=["rpm", "load"], seed=seed).fit(emb, tr)
        X = {k: fz.transform(t, i) for k, i in (("tr", tr), ("ca", ca), ("te", te))}
        point = models.lgbm("reg", seed).fit(X["tr"], y[tr])
        pc, pt = np.clip(point.predict(X["ca"]), 0, CAP), np.clip(point.predict(X["te"]), 0, CAP)
        qm = unc.fit_quantiles(X["tr"], y[tr], ALPHAS, seed)
        for a in ALPHAS:
            lo_m, hi_m = qm[a]
            lo_c, hi_c, lo_t, hi_t = lo_m.predict(X["ca"]), hi_m.predict(X["ca"]), lo_m.predict(X["te"]), hi_m.predict(X["te"])
            ivs = {"raw_quantile": (lo_t, hi_t),
                   "conformal_abs": unc.abs_residual_interval(pc, y[ca], pt, a),
                   "cqr": unc.cqr_interval(lo_c, hi_c, y[ca], lo_t, hi_t, a)}
            for name, (lo, hi) in ivs.items():
                lo, hi = np.clip(lo, 0, CAP), np.clip(hi, 0, CAP)
                cv, w = unc.coverage(y[te], lo, hi)
                cov_rows.append({"bearing": b, "seed": seed, "method": name, "nominal": 1 - a, "coverage": cv, "width": w})
                if a == 0.1 and seed == 0 and name == "cqr":
                    band_rows.append(pd.DataFrame({"bearing": b, "life_frac": t.life_frac.to_numpy()[te], "true": y[te],
                                                   "med": pt, "lo": lo, "hi": hi}))
    print("done", b, flush=True)

cov = pd.DataFrame(cov_rows)
pipeline.save_table(cov, "femto_coverage_runs")
summ = cov.groupby(["method", "nominal"]).agg(coverage=("coverage", "mean"), coverage_std=("coverage", "std"),
                                             width=("width", "mean")).reset_index()
pipeline.save_table(summ, "femto_coverage_summary")
bands = pd.concat(band_rows)
pipeline.save_table(bands, "femto_cqr90_bands")

# decision layer: policies compared on the held-out bearings (seed 0, CQR 90% bands)
mean_life = life.mean()
for b, g in bands.groupby("bearing"):
    r = decision.simulate(g.life_frac.to_numpy(), g.med.to_numpy(), g.lo.to_numpy(), g.hi.to_numpy(), life[b])
    fx = decision.fixed_policy(life[b], mean_life)
    dec_rows.append({"bearing": b, "policy": "uncertainty_aware", **r})
    dec_rows.append({"bearing": b, "policy": "fixed_interval", **fx})
    dec_rows.append({"bearing": b, "policy": "run_to_failure", "cost": decision.COSTS["unplanned_failure"], "failed": True})
    # point-estimate policy: same rule but zero-width interval
    r0 = decision.simulate(g.life_frac.to_numpy(), g.med.to_numpy(), g.med.to_numpy(), g.med.to_numpy(), life[b])
    dec_rows.append({"bearing": b, "policy": "point_estimate", **r0})
dec = pd.DataFrame(dec_rows)
pipeline.save_table(dec, "femto_decision_runs")
ds = dec.groupby("policy").agg(mean_cost=("cost", "mean"), failure_rate=("failed", "mean")).reset_index()
pipeline.save_table(ds, "femto_decision_summary")
pipeline.log_run("femto_uncertainty", {"calibration_bearings": K_CAL, "assumed_costs": decision.COSTS})
print(summ.to_string()); print(ds.to_string())
