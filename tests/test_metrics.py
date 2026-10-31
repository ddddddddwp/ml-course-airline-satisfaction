"""metrics 单元测试：全部合成数据，不依赖 70 万真实集。"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.eval.metrics import binary_metrics, bootstrap_auc_ci, reliability_curve


def _make_data(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.45).astype("int64")
    logit = np.where(y == 1, 1.6, -1.6) + rng.normal(0.0, 1.0, size=n)
    p = 1.0 / (1.0 + np.exp(-logit))
    return y, p


def test_perfect_prediction_auc_is_one() -> None:
    y = np.r_[np.zeros(50, dtype="int64"), np.ones(50, dtype="int64")]
    p = y.astype("float64")

    m = binary_metrics(y, p)

    assert m["n"] == 100
    assert m["positives"] == 50
    assert m["auc"] == pytest.approx(1.0)
    assert m["accuracy"] == pytest.approx(1.0)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (50, 0, 0, 50)
    assert m["brier"] == pytest.approx(0.0, abs=1e-12)
    assert m["logloss"] == pytest.approx(0.0, abs=1e-9)
    assert m["ece"] == pytest.approx(0.0, abs=1e-12)


def test_constant_prediction_auc_half_and_threshold_metrics() -> None:
    y = np.r_[np.zeros(100, dtype="int64"), np.ones(100, dtype="int64")]
    p = np.full(200, 0.5)

    m = binary_metrics(y, p)

    assert m["auc"] == pytest.approx(0.5)
    assert m["logloss"] == pytest.approx(math.log(2.0), rel=1e-9)
    assert m["brier"] == pytest.approx(0.25)
    assert m["ece"] == pytest.approx(0.0, abs=1e-12)
    # 常数 0.5 在 threshold=0.5 下全部判正
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (0, 100, 0, 100)
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(2.0 / 3.0)


def test_random_prediction_near_half() -> None:
    rng = np.random.default_rng(123)
    y = rng.integers(0, 2, 4000)
    p = rng.random(4000)

    auc = binary_metrics(y, p)["auc"]

    assert 0.45 < auc < 0.55
    assert auc == pytest.approx(float(roc_auc_score(y, p)))


def test_threshold_metrics_exact() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.10, 0.40, 0.60, 0.55, 0.80, 0.90])

    m = binary_metrics(y, p, threshold=0.5)

    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 1, 0, 3)
    assert m["accuracy"] == pytest.approx(5.0 / 6.0)
    assert m["precision"] == pytest.approx(3.0 / 4.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(6.0 / 7.0)


def test_single_class_auc_is_nan_with_warning() -> None:
    y = np.ones(120, dtype="int64")
    p = np.linspace(0.01, 0.99, 120)

    with pytest.warns(UserWarning, match="单一类别"):
        m = binary_metrics(y, p)

    assert math.isnan(m["auc"])
    assert m["positives"] == 120
    assert math.isfinite(m["logloss"])
    # 全是正例：预测为正的样本全部判对，precision=1；recall 约为 0.5
    assert m["precision"] == pytest.approx(1.0)
    assert 0.0 < m["recall"] < 1.0


def test_ece_known_value() -> None:
    # 100 个样本都预测 0.2，但只有 10 个正例 -> 该箱缺口 0.1 -> ECE=0.1
    y = np.zeros(100, dtype="int64")
    y[:10] = 1
    p = np.full(100, 0.2)

    m = binary_metrics(y, p)

    assert m["ece"] == pytest.approx(0.1, abs=1e-9)


def test_reliability_curve_bins_and_counts() -> None:
    p = np.r_[np.full(50, 0.05), np.full(50, 0.95)]
    y = np.r_[
        np.zeros(25, dtype="int64"), np.ones(25, dtype="int64"),
        np.zeros(25, dtype="int64"), np.ones(25, dtype="int64"),
    ]

    rel = reliability_curve(y, p, n_bins=10)

    assert list(rel.columns) == [
        "bin", "bin_lower", "bin_upper", "n", "pred_mean", "y_mean", "abs_gap",
    ]
    assert int(rel["n"].sum()) == 100
    assert int(rel.loc[rel["bin"] == 0, "n"].iloc[0]) == 50
    assert float(rel.loc[rel["bin"] == 0, "pred_mean"].iloc[0]) == pytest.approx(0.05)
    assert float(rel.loc[rel["bin"] == 0, "y_mean"].iloc[0]) == pytest.approx(0.5)
    assert int(rel.loc[rel["bin"] == 5, "n"].iloc[0]) == 0
    assert math.isnan(float(rel.loc[rel["bin"] == 5, "pred_mean"].iloc[0]))


def test_ece_respects_n_bins() -> None:
    y, p = _make_data(n=500, seed=11)

    m = binary_metrics(y, p, n_bins=5)
    rel = reliability_curve(y, p, n_bins=5)
    expected = float(np.nansum(rel["n"] / len(y) * rel["abs_gap"]))

    assert m["ece"] == pytest.approx(expected, abs=1e-12)


def test_reliability_curve_rejects_bad_bins() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        reliability_curve([0, 1], [0.2, 0.8], n_bins=0)


def test_bootstrap_ci_contains_point_estimate() -> None:
    y, p = _make_data(n=600, seed=7)

    low, high, mean = bootstrap_auc_ci(y, p, n_boot=300, seed=42)

    point = float(roc_auc_score(y, p))
    assert low < high
    assert low <= point <= high
    assert low <= mean <= high

    low2, high2, mean2 = bootstrap_auc_ci(y, p, n_boot=300, seed=42)
    assert (low, high, mean) == (low2, high2, mean2)


def test_bootstrap_single_class_returns_nan() -> None:
    with pytest.warns(UserWarning, match="单类"):
        out = bootstrap_auc_ci(np.ones(50, dtype="int64"), np.linspace(0, 1, 50), n_boot=20)

    assert all(math.isnan(v) for v in out)


def test_probability_clipping_warns_and_bounds_brier() -> None:
    y, p = _make_data(n=200, seed=3)
    p = p.copy()
    p[0], p[1] = -0.5, 1.5

    with pytest.warns(UserWarning, match="越界"):
        m = binary_metrics(y, p)

    assert 0.0 <= m["brier"] <= 1.0
    assert 0.0 <= m["logloss"] <= 40.0


def test_all_nan_returns_empty_metrics_with_warning() -> None:
    with pytest.warns(UserWarning):
        m = binary_metrics([np.nan] * 10, [np.nan] * 10)

    assert m["n"] == 0
    assert m["positives"] == 0
    assert math.isnan(m["auc"])
    assert math.isnan(m["logloss"])
    assert math.isnan(m["ece"])
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (0, 0, 0, 0)


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="长度"):
        binary_metrics([0, 1, 0], [0.1, 0.9])


def test_invalid_label_raises() -> None:
    with pytest.raises(ValueError, match="0/1"):
        binary_metrics([0, 1, 2], [0.1, 0.9, 0.5])
