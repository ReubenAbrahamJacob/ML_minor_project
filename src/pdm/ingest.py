"""Loaders. Every loader returns (meta DataFrame, signals array [N, channels, samples]).

meta columns: dataset, asset_id, condition, idx, life_frac (nan if not run-to-failure),
rpm, load, life_s (total life of asset in seconds), age_s.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from . import RAW

MAX_WIN = 300


def _pick(n: int, k: int) -> np.ndarray:
    """k evenly spaced indices out of n (keeps the whole life, thins it)."""
    return np.arange(n) if n <= k else np.unique(np.linspace(0, n - 1, k).astype(int))


# ----------------------------------------------------------------------- CWRU
CWRU_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}


def cwru_label(name: str) -> tuple[str, float | None, int]:
    """'IR007_1' -> ('IR', 0.007, 1); 'Normal_2' -> ('Normal', 0.0, 2)."""
    m = re.match(r"(Normal|IR|B|OR)(\d{3})?(?:@\d+)?_(\d)", name)
    if not m:
        raise ValueError(name)
    return m.group(1), (int(m.group(2)) / 1000 if m.group(2) else 0.0), int(m.group(3))


def load_cwru(win: int = 2048, max_win: int = 60) -> tuple[pd.DataFrame, np.ndarray]:
    d = RAW / "cwru"
    cat = pd.read_csv(d / "catalogue.csv")
    rows, sigs = [], []
    for f, label in zip(cat.file, cat.label):
        ftype, diam, load = cwru_label(label)
        m = sio.loadmat(d / "raw_mat" / f)
        key = [k for k in m if k.endswith("DE_time")][0]
        x = m[key].ravel()
        n = len(x) // win
        for i in _pick(n, max_win):
            sigs.append(x[i * win:(i + 1) * win][None])
            rows.append(dict(dataset="cwru", asset_id=label, file=f, fault=ftype, diameter=diam, load=load,
                             rpm=CWRU_RPM[load], idx=int(i)))
    return pd.DataFrame(rows), np.stack(sigs).astype(np.float32)


# ---------------------------------------------------------------------- FEMTO
def femto_root() -> Path:
    return next((RAW / "femto" / "extracted").rglob("FEMTOBearingDataSet"))


FEMTO_COND = {1: (1800, 4000), 2: (1650, 4200), 3: (1500, 5000)}


def load_femto(max_win: int = MAX_WIN) -> tuple[pd.DataFrame, np.ndarray]:
    root = femto_root()
    dirs = sorted((root / "Training_set" / "Learning_set").glob("Bearing*")) + \
        sorted((root / "Validation_Set" / "Full_Test_Set").glob("Bearing*"))
    rows, sigs = [], []
    for b in dirs:
        files = sorted(b.glob("acc_*.csv"))
        n, cond = len(files), int(b.name[7])
        for i in _pick(n, max_win):
            df = pd.read_csv(files[i], header=None, sep=None, engine="python")
            sigs.append(df.iloc[:, 4:6].to_numpy().T)
            rows.append(dict(dataset="femto", asset_id=b.name, condition=cond, idx=int(i), n_snap=n,
                             life_frac=(i + 1) / n, rpm=FEMTO_COND[cond][0], load=FEMTO_COND[cond][1],
                             life_s=n * 10.0, age_s=(i + 1) * 10.0))
    return pd.DataFrame(rows), np.stack(sigs).astype(np.float32)


# ----------------------------------------------------------------------- XJTU
XJTU_COND = {1: (2100, 12), 2: (2250, 11), 3: (2400, 10)}


def load_xjtu(max_win: int = MAX_WIN, seg: int = 2560) -> tuple[pd.DataFrame, np.ndarray]:
    """Centre `seg` samples of each 1.28 s snapshot, so windows match FEMTO's 2560 samples."""
    root = RAW / "xjtu" / "extracted_unar"
    dirs = sorted(p for p in root.rglob("Bearing*_*") if p.is_dir() and any(p.glob("*.csv")))
    rows, sigs = [], []
    for b in dirs:
        files = sorted(b.glob("*.csv"), key=lambda p: int(p.stem))
        n, cond = len(files), int(b.name[7])
        for i in _pick(n, max_win):
            x = pd.read_csv(files[i]).to_numpy().T
            s = (x.shape[1] - seg) // 2
            sigs.append(x[:2, s:s + seg])
            rows.append(dict(dataset="xjtu", asset_id=b.name, condition=cond, idx=int(i), n_snap=n,
                             life_frac=(i + 1) / n, rpm=XJTU_COND[cond][0], load=XJTU_COND[cond][1],
                             life_s=n * 60.0, age_s=(i + 1) * 60.0))
    return pd.DataFrame(rows), np.stack(sigs).astype(np.float32)


# --------------------------------------------------------------------- C-MAPSS
CMAPSS_COLS = ["unit", "cycle"] + [f"set{i}" for i in (1, 2, 3)] + [f"s{i}" for i in range(1, 22)]


def cmapss_dir() -> Path:
    return next((RAW / "cmapss" / "extracted").rglob("CMAPSSData"))


def load_cmapss_raw(fd: int = 1):
    d = cmapss_dir()
    rd = lambda n: pd.read_csv(d / n, sep=r"\s+", header=None, names=CMAPSS_COLS)
    train, test = rd(f"train_FD00{fd}.txt"), rd(f"test_FD00{fd}.txt")
    rul = pd.read_csv(d / f"RUL_FD00{fd}.txt", header=None)[0].to_numpy()
    return train, test, rul
