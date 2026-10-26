"""Data-dependent checks: skipped automatically if the cached tables are missing."""
import numpy as np
import pytest

from pdm import PROC, pipeline

pytestmark = pytest.mark.skipif(not (PROC / "femto_table.pkl").exists(), reason="run `make features` first")


def test_femto_audit_counts():
    t, emb = pipeline.load("femto")
    assert len(emb) == len(t)
    assert t.asset_id.nunique() == 17
    assert t.groupby("asset_id").life_frac.max().round(6).eq(1.0).all()   # last snapshot = failure
    assert np.isfinite(t[pipeline.feat_cols(t)].to_numpy()).all()


def test_cwru_counts():
    t, emb = pipeline.load("cwru")
    assert len(t) == len(emb) and t.file.nunique() == 64


def test_no_bearing_in_two_folds():
    from pdm import splits
    t, _ = pipeline.load("femto")
    g = t.asset_id.to_numpy()
    for tr, te, b in splits.logo(g):
        assert b not in set(g[tr])
