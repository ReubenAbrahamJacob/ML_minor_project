"""Rolling-bearing mechanics from the course notes (CLAUDE.md section 5)."""
import numpy as np
from scipy import stats

WEIBULL_A, WEIBULL_B = 6.84, 1.17  # course values: ln(1/R) = (L/a)^b, L in multiples of L10


def fault_freqs(fr: float, n: int, d: float, D: float, theta_deg: float = 0.0) -> dict[str, float]:
    """BPFO, BPFI, BSF, FTF in Hz from shaft frequency fr (Hz), ball count n, ball dia d, pitch dia D."""
    c = (d / D) * np.cos(np.radians(theta_deg))
    return {
        "BPFO": n / 2 * fr * (1 - c),
        "BPFI": n / 2 * fr * (1 + c),
        "BSF": D / (2 * d) * fr * (1 - c**2),
        "FTF": fr / 2 * (1 - c),
    }


def l10_millions(C: float, P: float, ball: bool = True) -> float:
    """Rating life in millions of revolutions."""
    return (C / P) ** (3 if ball else 10 / 3)


def l10_hours(C: float, P: float, rpm: float, ball: bool = True) -> float:
    return 1e6 * l10_millions(C, P, ball) / (60 * rpm)


def life_ratio(R: float) -> float:
    """L / L10 at reliability R, computed from a and b (the slide's 8.85 does not follow from them)."""
    return WEIBULL_A * np.log(1 / R) ** (1 / WEIBULL_B)


def l90_coefficient() -> float:
    """Coefficient c in L/L90 = c * [ln(1/R)]^(1/b): c = 1 / ln(1/0.9)^(1/b) = a (= 6.84, not the slide's 8.85)."""
    return 1 / np.log(1 / 0.9) ** (1 / WEIBULL_B)


class WeibullLife:
    """Population Weibull fit to training lifetimes; expected residual life given survival to age t."""

    def __init__(self, lifetimes: np.ndarray):
        self.shape, _, self.scale = stats.weibull_min.fit(np.asarray(lifetimes, float), floc=0)

    def mean_residual(self, age: np.ndarray, n_grid: int = 400) -> np.ndarray:
        age = np.atleast_1d(np.asarray(age, float))
        out = np.empty_like(age)
        for i, t in enumerate(age):
            grid = np.linspace(t, t + 6 * self.scale, n_grid)
            surv = stats.weibull_min.sf(grid, self.shape, scale=self.scale)
            out[i] = np.trapezoid(surv, grid) / max(surv[0], 1e-12)
        return out
