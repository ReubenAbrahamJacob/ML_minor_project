"""Sensor branch: time, frequency and envelope-spectrum features per channel."""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import hilbert

from .physics import fault_freqs

ENV_BANDS = [(0, 100), (100, 300), (300, 600), (600, 1000)]  # Hz, generic (no geometry needed)


def _bandsum(freq: np.ndarray, p: np.ndarray, lo: float, hi: float) -> np.ndarray:
    m = (freq >= lo) & (freq < hi)
    return p[:, m].sum(axis=1)


def channel_features(x: np.ndarray, fs: float, rpm: np.ndarray, geom: dict | None = None) -> dict[str, np.ndarray]:
    """x: [N, L] windows of one channel. Returns {name: [N]}."""
    x = x - x.mean(axis=1, keepdims=True)
    rms = np.sqrt((x**2).mean(axis=1))
    f = {
        "std": x.std(axis=1),
        "rms": rms,
        "skew": stats.skew(x, axis=1),
        "kurt": stats.kurtosis(x, axis=1),
        "crest": np.abs(x).max(axis=1) / (rms + 1e-12),
        "p2p": x.max(axis=1) - x.min(axis=1),
    }
    spec = np.abs(np.fft.rfft(x * np.hanning(x.shape[1]), axis=1)) ** 2
    freq = np.fft.rfftfreq(x.shape[1], 1 / fs)
    tot = spec.sum(axis=1) + 1e-12
    edges = np.linspace(0, fs / 2, 9)
    for i in range(8):
        f[f"band{i}"] = _bandsum(freq, spec, edges[i], edges[i + 1]) / tot
    pn = spec / tot[:, None]
    f["spec_entropy"] = -(pn * np.log(pn + 1e-12)).sum(axis=1) / np.log(pn.shape[1])
    f["spec_centroid"] = (pn * freq).sum(axis=1)
    env = np.abs(hilbert(x, axis=1))
    env = env - env.mean(axis=1, keepdims=True)
    es = np.abs(np.fft.rfft(env, axis=1)) ** 2
    et = es.sum(axis=1) + 1e-12
    for lo, hi in ENV_BANDS:
        f[f"env_{lo}_{hi}"] = _bandsum(freq, es, lo, hi) / et
    if geom is not None:  # fault-frequency energy (+ 2 harmonics) in the envelope spectrum
        for name in ("BPFO", "BPFI", "BSF", "FTF"):
            vals = np.zeros(len(x))
            for k in np.unique(rpm):
                m = rpm == k
                ff = fault_freqs(k / 60, **geom)[name]
                for h in (1, 2, 3):
                    vals[m] += _bandsum(freq, es[m], h * ff * 0.97, h * ff * 1.03)
            f[f"env_{name}"] = vals / et
    return f


def extract(sig: np.ndarray, fs: float, rpm: np.ndarray, geom: dict | None = None) -> pd.DataFrame:
    """sig: [N, C, L] -> DataFrame with columns like 'c0_rms'."""
    cols = {}
    for c in range(sig.shape[1]):
        for k, v in channel_features(sig[:, c].astype(np.float64), fs, rpm, geom).items():
            cols[f"c{c}_{k}"] = v
    return pd.DataFrame(cols)
