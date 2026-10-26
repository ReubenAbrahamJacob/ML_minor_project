"""Shared helpers for the experiment scripts."""
import json
import re
import subprocess
import time

import numpy as np
import pandas as pd

from . import PROC, REPORTS, ROOT
from .models import CFG


def load(name: str) -> tuple[pd.DataFrame, np.ndarray]:
    return pd.read_pickle(PROC / f"{name}_table.pkl"), np.load(PROC / f"{name}_emb.npy")


def feat_cols(table: pd.DataFrame) -> list[str]:
    """Sensor-branch feature columns (channel features c0_*, c1_* or C-MAPSS s*_*)."""
    return [c for c in table.columns if re.match(r"^(c\d_|s\d+_)", c)]


def add_rul_targets(t: pd.DataFrame) -> pd.DataFrame:
    """Remaining life fraction, flat at `cap` until life fraction 1-cap (piecewise-linear); 3 health states."""
    t = t.copy()
    cap = CFG["rul_cap_life_fraction"]
    t["rul"] = np.minimum(1 - t.life_frac, cap)
    t["state"] = np.digitize(t.life_frac, CFG["health_bins"])  # 0 healthy, 1 degrading, 2 critical
    return t


def save_table(df: pd.DataFrame, name: str) -> None:
    (REPORTS / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORTS / "tables" / f"{name}.csv", index=False)


def log_run(name: str, extra: dict | None = None) -> None:
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=False, capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except OSError:
        h = ""
    (REPORTS / "runs").mkdir(parents=True, exist_ok=True)
    rec = {"run": name, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "git": h or "uncommitted",
           "config": CFG, **(extra or {})}
    (REPORTS / "runs" / f"{name}.json").write_text(json.dumps(rec, indent=2, default=str))
