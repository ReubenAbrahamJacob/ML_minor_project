"""Data audit + SHAP on the fused FEMTO model + per-modality contribution."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdm import models, pipeline

# ---- data audit (from cached tables; snapshot counts = full lifetimes)
rows = []
for name in ("femto", "xjtu"):
    t, _ = pipeline.load(name)
    for b, g in t.groupby("asset_id"):
        rows.append({"dataset": name, "asset": b, "condition": g.condition.iloc[0], "snapshots": int(g.n_snap.iloc[0]),
                     "lifetime_s": g.life_s.iloc[0], "windows_used": len(g), "channels": 2, "sampling_hz": 25600,
                     "rpm": g.rpm.iloc[0], "load": g.load.iloc[0]})
t, _ = pipeline.load("cwru")
for f, g in t.groupby("file"):
    rows.append({"dataset": "cwru", "asset": g.asset_id.iloc[0], "condition": g.load.iloc[0], "snapshots": len(g),
                 "lifetime_s": np.nan, "windows_used": len(g), "channels": 1, "sampling_hz": 12000, "rpm": g.rpm.iloc[0], "load": g.load.iloc[0]})
for fd in (1, 2, 3, 4):
    t, _ = pipeline.load(f"cmapss{fd}")
    rows.append({"dataset": f"cmapss FD00{fd}", "asset": f"{t.asset_id.nunique()} engines", "condition": np.nan,
                 "snapshots": len(t), "lifetime_s": np.nan, "windows_used": len(t), "channels": 21, "sampling_hz": np.nan,
                 "rpm": np.nan, "load": np.nan})
pipeline.save_table(pd.DataFrame(rows), "data_audit")

# ---- SHAP: train fused model on all but 3 bearings, explain those
t, emb = pipeline.load("femto")
t = pipeline.add_rul_targets(t)
cols = pipeline.feat_cols(t)
test_b = ["Bearing1_3", "Bearing2_3", "Bearing3_3"]
te = np.where(t.asset_id.isin(test_b))[0]
tr = np.where(~t.asset_id.isin(test_b))[0]
fz = models.Featurizer("fused", cols, covars=["rpm", "load"], seed=0).fit(emb, tr)
m = models.lgbm("reg", 0).fit(fz.transform(t, tr), t.rul.to_numpy()[tr])
Xte = fz.transform(t, te)
names = cols + [f"img_pc{i}" for i in range(fz.n_pca)] + ["rpm", "load"]
sv = shap.TreeExplainer(m).shap_values(Xte)
imp = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(0)})
imp["modality"] = np.where(imp.feature.str.startswith("img_pc"), "image (CNN embedding)",
                           np.where(imp.feature.isin(["rpm", "load"]), "operating condition", "sensor features"))
pipeline.save_table(imp.sort_values("mean_abs_shap", ascending=False), "shap_importance")
pipeline.save_table(imp.groupby("modality").mean_abs_shap.sum().reset_index(), "shap_modality")
np.savez(pipeline.PROC / "shap_values.npz", sv=sv, X=Xte, names=names)
print(imp.sort_values("mean_abs_shap", ascending=False).head(10)); print(imp.groupby("modality").mean_abs_shap.sum())
