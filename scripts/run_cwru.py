"""Task A on CWRU: health-state / fault-type classification, leave-one-load-out, 5 seeds."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdm import eval as ev
from pdm import models, pipeline, splits

t, emb = pipeline.load("cwru")
cols = pipeline.feat_cols(t)
# assumption (documented): healthy = Normal, early = 0.007 in faults, severe = 0.014 / 0.021 in faults
t["state"] = np.where(t.fault == "Normal", 0, np.where(t.diameter <= 0.007, 1, 2))
t["ftype"] = t.fault.map({"Normal": 0, "IR": 1, "B": 2, "OR": 3})
tasks = {"state": [0, 1, 2], "ftype": [0, 1, 2, 3]}
groups = t.load.to_numpy()
rows, cms = [], {}
for task, labels in tasks.items():
    y = t[task].to_numpy()
    for tr, te, g in splits.logo(groups):
        assert set(groups[tr]).isdisjoint(groups[te])
        for seed in models.CFG["seeds"]:
            variants = {f"lgbm_{m}": (m, "lgbm") for m in models.MODES}
            variants["rf_sensor"] = ("sensor", "rf")
            for name, (mode, algo) in variants.items():
                fz = models.Featurizer(mode, cols, seed=seed).fit(emb, tr)
                clf = (models.lgbm if algo == "lgbm" else models.rf)("clf", seed)
                clf.fit(fz.transform(t, tr), y[tr])
                p = clf.predict(fz.transform(t, te))
                rows.append({"task": task, "held_out_load": g, "seed": seed, "model": name, **ev.clf_metrics(y[te], p, labels)})
                if name == "lgbm_fused" and seed == 0:
                    cms.setdefault(task, np.zeros((len(labels),) * 2, int))
                    cms[task] += ev.confusion(y[te], p, labels)
    print("done", task, flush=True)
res = pd.DataFrame(rows)
pipeline.save_table(res, "cwru_classification_runs")
summ = res.groupby(["task", "model"]).agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
                                           acc_mean=("accuracy", "mean")).reset_index()
# paired Wilcoxon: fused vs each single modality over (load, seed) pairs
pv = []
for task in tasks:
    r = res[res.task == task].pivot_table(index=["held_out_load", "seed"], columns="model", values="macro_f1")
    for other in ["lgbm_sensor", "lgbm_image", "lgbm_fused_shuffled"]:
        pv.append({"task": task, "comparison": f"fused vs {other}", "mean_diff": float((r.lgbm_fused - r[other]).mean()),
                   "wilcoxon_p": ev.wilcoxon(r.lgbm_fused, r[other])})
pipeline.save_table(summ, "cwru_classification_summary")
pipeline.save_table(pd.DataFrame(pv), "cwru_classification_wilcoxon")
for task, cm in cms.items():
    pipeline.save_table(pd.DataFrame(cm), f"cwru_confusion_{task}")
pipeline.log_run("cwru", {"split": "leave-one-load-out", "seeds": models.CFG["seeds"]})
print(summ.to_string())
print(pd.DataFrame(pv).to_string())
