"""Quantile LightGBM + split conformal prediction."""
import numpy as np

from .models import lgbm


def conformal_q(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def abs_residual_interval(pred_cal, y_cal, pred_test, alpha):
    q = conformal_q(np.abs(np.asarray(y_cal) - pred_cal), alpha)
    return pred_test - q, pred_test + q


def cqr_interval(lo_cal, hi_cal, y_cal, lo_test, hi_test, alpha):
    """Conformalised quantile regression (Romano et al. 2019)."""
    q = conformal_q(np.maximum(lo_cal - y_cal, y_cal - hi_cal), alpha)
    return lo_test - q, hi_test + q


def fit_quantiles(Xtr, ytr, alphas, seed):
    """Returns {alpha: (lo_model, hi_model)} with quantiles alpha/2 and 1-alpha/2."""
    return {a: (lgbm("reg", seed, objective="quantile", alpha=a / 2).fit(Xtr, ytr),
                lgbm("reg", seed, objective="quantile", alpha=1 - a / 2).fit(Xtr, ytr)) for a in alphas}


def coverage(y, lo, hi) -> tuple[float, float]:
    y = np.asarray(y)
    return float(((y >= lo) & (y <= hi)).mean()), float(np.mean(hi - lo))
