"""Metrics (CLAUDE.md section 6)."""
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score, recall_score


def clf_metrics(y, p, labels) -> dict:
    return {"macro_f1": f1_score(y, p, average="macro", labels=labels, zero_division=0),
            **{f"recall_{l}": r for l, r in zip(labels, recall_score(y, p, labels=labels, average=None, zero_division=0))},
            "accuracy": float((np.asarray(y) == np.asarray(p)).mean())}


def confusion(y, p, labels) -> np.ndarray:
    return confusion_matrix(y, p, labels=labels)


def nasa_score(y, p, scale: float = 1.0) -> float:
    """Mean asymmetric NASA score; d = pred - true (late = d > 0) is penalised more. `scale` converts units
    to the original cycle scale (e.g. 100 for life fractions -> percent of life)."""
    d = (np.asarray(p) - np.asarray(y)) * scale
    return float(np.mean(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def rul_metrics(y, p, scale: float = 1.0) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"rmse": float(np.sqrt(np.mean((p - y) ** 2))), "mae": float(np.mean(np.abs(p - y))),
            "nasa": nasa_score(y, p, scale)}


def error_by_life(life_frac, y, p, bins=(0, 0.2, 0.4, 0.6, 0.8, 1.0)) -> dict:
    b = np.digitize(life_frac, bins[1:-1])
    return {f"{bins[i]:.1f}-{bins[i + 1]:.1f}": float(np.mean(np.abs(np.asarray(p)[b == i] - np.asarray(y)[b == i])))
            for i in range(len(bins) - 1) if (b == i).any()}


def wilcoxon(a, b) -> float:
    """Paired Wilcoxon signed-rank p-value on per-bearing metrics."""
    a, b = np.asarray(a), np.asarray(b)
    return float(stats.wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0
