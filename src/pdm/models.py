"""Feature sets (sensor / image / fused / shuffled-image) and model factories."""
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from . import ROOT

CFG = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
MODES = ["sensor", "image", "fused", "fused_shuffled"]


@dataclass
class Featurizer:
    """Builds X for a modality. Scaler + PCA are fit on training rows only (rule 1)."""
    mode: str
    feat_cols: list[str]
    covars: list[str] = field(default_factory=list)
    n_pca: int = CFG["pca_components"]
    seed: int = 0

    def fit(self, emb: np.ndarray, train_idx: np.ndarray) -> "Featurizer":
        if self.mode == "fused_shuffled":  # mismatched control: embeddings no longer belong to their window
            emb = emb[np.random.default_rng(self.seed).permutation(len(emb))]
        self._emb = emb
        if self.mode != "sensor":
            self.scaler = StandardScaler().fit(emb[train_idx])
            self.pca = PCA(self.n_pca, random_state=self.seed).fit(self.scaler.transform(emb[train_idx]))
        return self

    def transform(self, table: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
        parts = []
        if self.mode != "image":
            parts.append(table.iloc[idx][self.feat_cols].to_numpy(float))
        if self.mode != "sensor":
            parts.append(self.pca.transform(self.scaler.transform(self._emb[idx])))
        if self.covars:
            parts.append(table.iloc[idx][self.covars].to_numpy(float))
        return np.hstack(parts)


def lgbm(task: str, seed: int, **kw):
    p = {**CFG["lgbm"], "random_state": seed, **kw}
    return lgb.LGBMClassifier(**p) if task == "clf" else lgb.LGBMRegressor(**p)


def rf(task: str, seed: int):
    cls = RandomForestClassifier if task == "clf" else RandomForestRegressor
    return cls(n_estimators=150, min_samples_leaf=3, n_jobs=4, random_state=seed)
