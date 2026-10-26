"""All report figures, from the CSVs in reports/tables and cached tables. python scripts/make_figures.py"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdm import REPORTS, images, ingest, pipeline

TB, FG = REPORTS / "tables", REPORTS / "figures"
FG.mkdir(parents=True, exist_ok=True)
C = {"sensor": "#4C78A8", "image": "#F58518", "fused": "#54A24B", "shuf": "#B279A2", "base": "#9D9D9D", "rf": "#E45756"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130, "font.size": 9})
rd = lambda n: pd.read_csv(TB / f"{n}.csv")
col = lambda m: C["shuf"] if "shuffled" in m else C["fused"] if "fused" in m else C["image"] if "image" in m else \
    C["sensor"] if "sensor" in m and "rf" not in m else C["rf"] if "rf" in m else C["base"]


def save(name):
    plt.tight_layout(); plt.savefig(FG / f"{name}.png"); plt.close()


# 1 CWRU F1
s = rd("cwru_classification_summary")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for a, task, title in zip(ax, ["state", "ftype"], ["Health state (3 classes)", "Fault type (4 classes)"]):
    d = s[s.task == task].sort_values("macro_f1_mean")
    a.barh(d.model, d.macro_f1_mean, xerr=d.macro_f1_std, color=[col(m) for m in d.model])
    a.set_xlim(0.9, 1.0); a.set_title(title); a.set_xlabel("macro-F1 (leave-one-load-out, 5 seeds)")
save("cwru_f1")
for task in ("state", "ftype"):
    cm = rd(f"cwru_confusion_{task}").to_numpy()
    fig, a = plt.subplots(figsize=(3.4, 3)); a.imshow(cm, cmap="Blues")
    for i in range(len(cm)):
        for j in range(len(cm)):
            a.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    a.set_xlabel("predicted"); a.set_ylabel("true"); a.set_title(f"CWRU {task}: fused, seed 0")
    save(f"cwru_confusion_{task}")

# 2 FEMTO RMSE
s = rd("femto_rul_summary").sort_values("rmse_mean", ascending=False)
fig, a = plt.subplots(figsize=(6, 3.2))
a.barh(s.model, s.rmse_mean, xerr=s.rmse_std, color=[col(m) for m in s.model])
a.set_xlabel("RMSE of remaining life fraction (mean ± std over 17 bearings)"); a.set_title("FEMTO RUL, leave-one-bearing-out")
save("femto_rmse")

# 3 predictions with CQR bands
b = rd("femto_cqr90_bands")
fig, axs = plt.subplots(3, 6, figsize=(15, 6.5), sharex=True, sharey=True)
for a, (name, g) in zip(axs.ravel(), b.groupby("bearing")):
    g = g.sort_values("life_frac")
    a.fill_between(g.life_frac, g.lo, g.hi, color=C["fused"], alpha=0.25, label="90% CQR band")
    a.plot(g.life_frac, g.med, color=C["fused"], lw=1, label="prediction"); a.plot(g.life_frac, g.true, "k--", lw=1, label="true")
    a.set_title(name, fontsize=8)
axs[0, 0].legend(fontsize=6); fig.supxlabel("life fraction"); fig.supylabel("remaining life fraction (capped)")
for a in axs.ravel()[len(b.bearing.unique()):]:
    a.axis("off")
save("femto_rul_bands")

# 4 error vs life fraction
e = rd("femto_error_by_life")
fig, a = plt.subplots(figsize=(4.5, 3)); a.bar(e.life_bin, e.mae, color=C["fused"])
a.set_xlabel("life fraction"); a.set_ylabel("MAE"); a.set_title("FEMTO fused: error vs life fraction"); save("femto_error_by_life")

# 5 calibration plot
c = rd("femto_coverage_summary")
fig, a = plt.subplots(figsize=(4, 3.6)); a.plot([0.75, 1], [0.75, 1], "k:", label="ideal")
for m, mc in zip(["raw_quantile", "conformal_abs", "cqr"], [C["base"], C["sensor"], C["fused"]]):
    d = c[c.method == m]; a.errorbar(d.nominal, d.coverage, yerr=d.coverage_std, marker="o", color=mc, label=m, capsize=2)
a.set_xlabel("nominal coverage"); a.set_ylabel("empirical coverage"); a.legend(fontsize=7); a.set_title("FEMTO calibration (per-bearing mean ± std)")
save("femto_calibration")

# 6 decision cost
d = rd("femto_decision_summary").sort_values("mean_cost")
fig, a = plt.subplots(figsize=(5, 3)); a.bar(d.policy, d.mean_cost, color=C["base"])
for i, (mc, fr) in enumerate(zip(d.mean_cost, d.failure_rate)):
    a.text(i, mc, f"{mc:.1f}\nfail {fr:.0%}", ha="center", va="bottom", fontsize=7)
a.set_ylabel("mean cost per bearing (ASSUMED units)"); a.set_ylim(0, 12); plt.xticks(rotation=15); a.set_title("Maintenance policies")
save("decision_cost")

# 7 SHAP
imp = rd("shap_importance").head(15).iloc[::-1]
fig, a = plt.subplots(figsize=(5, 4)); a.barh(imp.feature, imp.mean_abs_shap, color=[{"sensor features": C["sensor"], "image (CNN embedding)": C["image"], "operating condition": C["base"]}[m] for m in imp.modality])
a.set_xlabel("mean |SHAP|"); a.set_title("Fused FEMTO model: top features"); save("shap_top")
mo = rd("shap_modality")
fig, a = plt.subplots(figsize=(4.5, 2.6)); a.barh(mo.modality, mo.mean_abs_shap, color=C["fused"]); a.set_xlabel("sum of mean |SHAP|")
a.set_title("Per-modality contribution"); save("shap_modality")

# 8 generalisation
g = rd("generalisation")
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for a, nm in zip(axs, ["raw", "per_condition_norm"]):
    d = g[g.norm == nm].pivot(index="setting", columns="model", values="rmse_mean")
    d.plot.bar(ax=a, color=[C["fused"], C["image"], C["sensor"]], rot=20); a.set_title(f"normalisation: {nm}"); a.set_ylabel("RMSE (life fraction)")
save("generalisation")
cv = rd("generalisation_coverage")
fig, a = plt.subplots(figsize=(5, 3)); lab = cv.setting + "\n" + cv.calibration.str.replace(" (no recalibration)", "").str.replace("labelled ", "")
a.bar(range(len(cv)), cv.coverage, yerr=cv.coverage_std, color=C["sensor"]); a.axhline(0.9, color="k", ls=":")
a.set_xticks(range(len(cv))); a.set_xticklabels(lab, fontsize=6); a.set_ylim(0.6, 1); a.set_ylabel("coverage @ 90%"); save("generalisation_coverage")

# 9 C-MAPSS
m = rd("cmapss_summary")
fig, a = plt.subplots(figsize=(6, 3)); m.pivot(index="fd", columns="model", values="rmse_mean").plot.bar(ax=a, rot=0)
a.set_ylabel("RMSE (cycles)"); a.set_title("C-MAPSS test set, same pipeline via new adapter"); a.legend(fontsize=6); save("cmapss_rmse")

# 10 EDA: RMS & kurtosis vs life fraction
t, _ = pipeline.load("femto")
fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
for nm, g_ in t.groupby("asset_id"):
    for a, f in zip(axs, ["c0_rms", "c0_kurt"]):
        a.plot(g_.life_frac, g_[f], lw=0.6, alpha=0.7)
axs[0].set_yscale("log"); axs[0].set_ylabel("horizontal RMS"); axs[1].set_ylabel("horizontal kurtosis")
for a in axs: a.set_xlabel("life fraction")
fig.suptitle("FEMTO: 17 bearings"); save("eda_femto_features")

# 11 signal-derived image grid: Bearing1_1 at 10%, 80%, 100% of life
root = ingest.femto_root() / "Training_set" / "Learning_set" / "Bearing1_1"
files = sorted(root.glob("acc_*.csv"))
fig, axs = plt.subplots(1, 3, figsize=(9, 3.2))
for a, (fr, lab) in zip(axs, [(0.1, "healthy (10% life)"), (0.8, "degrading (80%)"), (0.999, "critical (end)")]):
    x = pd.read_csv(files[int(fr * len(files))], header=None).iloc[:, 4].to_numpy(np.float32)
    a.imshow(images.stft_image(x, 25600).transpose(1, 2, 0), origin="lower"); a.set_title(lab); a.axis("off")
fig.suptitle("Signal-derived STFT images (Bearing1_1), input to the frozen CNN"); save("stft_grid")
print("figures:", len(list(FG.glob("*.png"))))
