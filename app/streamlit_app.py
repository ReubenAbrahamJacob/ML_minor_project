"""Replay a held-out FEMTO bearing: health curve, RUL with conformal band, decision, SHAP. Uses cached CSVs only."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdm import decision

TB = Path(__file__).resolve().parents[1] / "reports" / "tables"
st.title("Bearing RUL replay (FEMTO, held-out bearings)")
bands = pd.read_csv(TB / "femto_cqr90_bands.csv")
bearing = st.selectbox("Bearing (never seen in training)", sorted(bands.bearing.unique()))
g = bands[bands.bearing == bearing].sort_values("life_frac").reset_index(drop=True)
step = st.slider("Replay position (window index)", 0, len(g) - 1, len(g) // 2)
cur = g.iloc[: step + 1]
c1, c2, c3 = st.columns(3)
r = g.iloc[step]
c1.metric("Life fraction used", f"{r.life_frac:.0%}")
c2.metric("RUL median (fraction of life, capped)", f"{r.med:.2f}", f"90% band [{r.lo:.2f}, {r.hi:.2f}]")
act = decision.decide(r.med, r.lo, r.hi)
c3.metric("Recommendation (assumed costs)", act.replace("_", " "))
chart = pd.DataFrame({"true": cur.true.values, "median": cur.med.values, "lower": cur.lo.values, "upper": cur.hi.values},
                     index=cur.life_frac.values)
st.line_chart(chart)
st.caption("Costs in configs/costs.yaml are assumptions. Intervals: CQR, 90% nominal, calibrated on 4 held-out bearings.")
st.subheader("What drives the prediction (SHAP, fused model)")
st.bar_chart(pd.read_csv(TB / "shap_importance.csv").head(12).set_index("feature").mean_abs_shap)
