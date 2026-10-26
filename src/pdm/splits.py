"""Leakage-safe splits: always by group (bearing / engine / load condition), never by window."""
import numpy as np


def logo(groups: np.ndarray):
    """Leave-one-group-out: yields (train_idx, test_idx, held_out_group)."""
    for g in np.unique(groups):
        yield np.where(groups != g)[0], np.where(groups == g)[0], g


def with_calibration(groups: np.ndarray, test_group, k: int, seed: int):
    """Split off k calibration groups (other than the test group); returns (train_idx, calib_idx, test_idx)."""
    others = np.array([g for g in np.unique(groups) if g != test_group])
    cal = np.random.default_rng(seed).choice(others, size=min(k, len(others) - 1), replace=False)
    is_cal = np.isin(groups, cal)
    is_test = groups == test_group
    return np.where(~is_cal & ~is_test)[0], np.where(is_cal)[0], np.where(is_test)[0]


def assert_disjoint(*idx_groups_pairs) -> None:
    """Each arg is an array of group ids for one split; raises if any id appears in two splits."""
    seen: set = set()
    for ids in idx_groups_pairs:
        s = set(np.unique(ids))
        if seen & s:
            raise AssertionError(f"leakage: groups in several splits: {seen & s}")
        seen |= s
