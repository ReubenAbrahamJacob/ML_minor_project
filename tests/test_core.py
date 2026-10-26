import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pdm import decision, features, models, physics, splits
from pdm import eval as ev
from pdm import uncertainty as unc


def test_fault_freqs_textbook():
    # CWRU 6205: n=9, d=0.3126, D=1.537 -> published multiples of shaft speed
    f = physics.fault_freqs(1.0, 9, 0.3126, 1.537, 0)
    assert f["BPFO"] == pytest.approx(3.5848, abs=1e-3)
    assert f["BPFI"] == pytest.approx(5.4152, abs=1e-3)
    # course formula gives the ball spin frequency; CWRU's published 4.7135 is 2x that (defect strikes both races)
    assert 2 * f["BSF"] == pytest.approx(4.7135, abs=1e-3)
    assert f["FTF"] == pytest.approx(0.39828, abs=1e-4)


def test_l10_and_weibull_coefficient():
    assert physics.l10_millions(20, 10, ball=True) == pytest.approx(8.0)
    assert physics.l10_millions(20, 10, ball=False) == pytest.approx(2 ** (10 / 3))
    assert physics.life_ratio(0.9) == pytest.approx(1.0, abs=0.01)         # a, b are consistent at R=0.9
    assert physics.l90_coefficient() == pytest.approx(6.84, abs=0.02)
    assert abs(physics.l90_coefficient() - 8.85) > 2                          # slide's 8.85 does not follow
    assert physics.l10_hours(20, 10, 1000) == pytest.approx(1e6 * 8 / 60000)


def test_weibull_residual_life_decreases():
    w = physics.WeibullLife(stats.weibull_min.rvs(2.0, scale=1000, size=200, random_state=0))
    r = w.mean_residual(np.array([0.0, 500.0, 1500.0]))
    assert r[0] > r[1] > r[2] > 0


def test_time_features_synthetic():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 2.0, size=(1, 1, 20000))
    f = features.extract(x, 12000, np.array([1800]), None).iloc[0]
    assert f["c0_rms"] == pytest.approx(2.0, rel=0.05)
    assert f["c0_kurt"] == pytest.approx(0.0, abs=0.2)                      # Gaussian excess kurtosis
    s = np.sin(2 * np.pi * 50 * np.arange(12000) / 12000)[None, None]
    g = features.extract(s, 12000, np.array([1800]), None).iloc[0]
    assert g["c0_crest"] == pytest.approx(np.sqrt(2), rel=0.01)             # sine crest factor
    assert g["c0_rms"] == pytest.approx(1 / np.sqrt(2), rel=0.01)


def test_splits_no_group_overlap_and_deterministic():
    groups = np.repeat(np.arange(8), 10)
    for tr, te, g in splits.logo(groups):
        assert set(groups[tr]).isdisjoint(groups[te])
    a = splits.with_calibration(groups, 3, 2, seed=1)
    b = splits.with_calibration(groups, 3, 2, seed=1)
    assert all((x == y).all() for x, y in zip(a, b))
    splits.assert_disjoint(groups[a[0]], groups[a[1]], groups[a[2]])
    with pytest.raises(AssertionError):
        splits.assert_disjoint(np.array([1, 2]), np.array([2, 3]))


def test_featurizer_fits_on_train_only():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(100, 32))
    emb[50:] += 100                                                         # test rows wildly different
    t = pd.DataFrame({"c0_a": rng.normal(size=100)})
    fz = models.Featurizer("image", ["c0_a"], n_pca=4).fit(emb, np.arange(50))
    assert np.allclose(fz.scaler.mean_, emb[:50].mean(0))                   # scaler never saw test rows
    assert fz.transform(t, np.arange(50, 60)).shape == (10, 4)


def test_conformal_coverage():
    rng = np.random.default_rng(0)
    cal, test = rng.normal(size=2000), rng.normal(size=20000)
    lo, hi = unc.abs_residual_interval(np.zeros(2000), cal, np.zeros(20000), 0.1)
    cov, _ = unc.coverage(test, lo, hi)
    assert cov == pytest.approx(0.9, abs=0.015)


def test_metrics():
    y = np.array([0.5, 0.3]); p = np.array([0.5, 0.3])
    assert ev.rul_metrics(y, p)["rmse"] == 0
    assert ev.nasa_score([10], [20]) > ev.nasa_score([10], [0])            # late predictions penalised more
    assert ev.clf_metrics([0, 1, 1], [0, 1, 1], [0, 1])["macro_f1"] == 1.0


def test_decision_monotone():
    far = decision.decide(0.4, 0.3, 0.4)
    near = decision.decide(0.01, 0.0, 0.03)
    assert far == "continue" and near == "maintain_now"
