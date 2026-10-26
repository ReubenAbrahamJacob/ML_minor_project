"""Build cached feature tables + CNN embeddings: python scripts/build_features.py [cwru femto xjtu cmapss]"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdm import PROC, features, images, ingest

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "datasets.yaml").read_text())
MAXW = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "models.yaml").read_text())["max_windows_per_bearing"]


def save(name: str, meta: pd.DataFrame, feats: pd.DataFrame, emb: np.ndarray) -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    pd.concat([meta.reset_index(drop=True), feats], axis=1).to_pickle(PROC / f"{name}_table.pkl")
    np.save(PROC / f"{name}_emb.npy", emb)
    print(f"saved {name}: {len(meta)} windows, {feats.shape[1]} features, emb {emb.shape}", flush=True)


def cwru() -> None:
    meta, sig = ingest.load_cwru()
    g = CFG["cwru"]["geometry"]
    geom = dict(n=g["balls"], d=g["ball_diameter_in"], D=g["pitch_diameter_in"], theta_deg=g["contact_angle_deg"])
    f = features.extract(sig, CFG["cwru"]["sampling_rate"], meta.rpm.to_numpy(), geom)
    emb = images.embed(sig, CFG["cwru"]["sampling_rate"], "cwt", "cwru")
    save("cwru", meta, f, emb)


def rul_dataset(name: str, loader) -> None:
    meta, sig = loader(MAXW)
    f = features.extract(sig, CFG[name]["sampling_rate"], meta.rpm.to_numpy(), None)
    emb = images.embed(sig, CFG[name]["sampling_rate"], "stft", name)
    save(name, meta, f, emb)


def cmapss() -> None:
    """Adapter: sliding 30-cycle windows of 21 sensors -> same schema (features + heatmap embedding)."""
    W = 30
    for fd in (1, 2, 3, 4):
        train, test, rul = ingest.load_cmapss_raw(fd)
        sens = [f"s{i}" for i in range(1, 22)]
        # drop sensors constant in the training set
        sens = [s for s in sens if train[s].std() > 1e-6]
        mu, sd = train[sens].mean(), train[sens].std()
        rows, wins = [], []
        for split, df in (("train", train), ("test", test)):
            for u, g in df.groupby("unit"):
                x = ((g[sens] - mu) / sd).to_numpy(np.float32)
                last = [len(g) - 1] if split == "test" else range(0, len(g), 3)
                for i in last:
                    w = x[max(0, i - W + 1): i + 1]
                    w = np.vstack([np.repeat(w[:1], W - len(w), 0), w])  # pad start by repeating first row
                    wins.append(w)
                    life = len(g) if split == "train" else len(g) + rul[u - 1]
                    r = life - (i + 1)
                    rows.append(dict(dataset=f"cmapss{fd}", asset_id=f"{split}{fd}_{u}", split=split, idx=i,
                                     rul=float(min(r, CFG["cmapss"]["rul_cap_cycles"])), rpm=0, load=0,
                                     set1=g["set1"].iloc[i], set2=g["set2"].iloc[i], set3=g["set3"].iloc[i]))
        arr = np.stack(wins)                                    # [N, W, S]
        meta = pd.DataFrame(rows)
        t = np.arange(W)
        cols = {}
        for j, s in enumerate(sens):
            a = arr[:, :, j]
            cols[f"{s}_last"], cols[f"{s}_mean"], cols[f"{s}_std"] = a[:, -1], a.mean(1), a.std(1)
            cols[f"{s}_slope"] = ((a - a.mean(1, keepdims=True)) * (t - t.mean())).sum(1) / ((t - t.mean()) ** 2).sum()
        emb = images.embed(arr, 0, "heatmap", f"cmapss{fd}")
        save(f"cmapss{fd}", meta, pd.DataFrame(cols), emb)


if __name__ == "__main__":
    todo = sys.argv[1:] or ["cwru", "femto", "xjtu", "cmapss"]
    for k in todo:
        t = time.time()
        {"cwru": cwru, "femto": lambda: rul_dataset("femto", ingest.load_femto),
         "xjtu": lambda: rul_dataset("xjtu", ingest.load_xjtu), "cmapss": cmapss}[k]()
        print(f"{k} done in {time.time() - t:.0f}s", flush=True)
