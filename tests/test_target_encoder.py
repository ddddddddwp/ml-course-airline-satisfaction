"""TargetEncoder 的契约测试：折内 fit、unseen→prior、平滑公式、输出列名/类型。

本文件保护的是全项目最容易写出"CV 虚高、LB 不动"的那一行：
`fit()` 只能看到训练折。反例见 test_full_fit_leaks_valid_rows 的注释。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.encode import TargetEncoder


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "Flight Distance": [100, 100, 100, 200, 200, 300, 300, 300, 300],
        "y": [1, 0, 1, 1, 0, 0, 0, 1, 0],
    })


def _encode_valid(df: pd.DataFrame, y: np.ndarray, tr_idx, va_idx, smooth: float = 2.0):
    """模拟一次折内编码：只在训练折 fit，transform 验证折。"""
    te = TargetEncoder("Flight Distance", smooth=smooth).fit(df.iloc[tr_idx], y[tr_idx])
    return te.transform(df.iloc[va_idx]).to_numpy()


def test_fit_uses_train_fold_only():
    """改动验证折的 y 不能影响验证折的编码值。"""
    df = _frame()
    tr_idx = [0, 1, 2, 3, 4]
    va_idx = [5, 6, 7, 8]
    y = df["y"].to_numpy()
    y_poisoned = y.copy()
    y_poisoned[va_idx] = 1 - y_poisoned[va_idx]   # 篡改验证折标签

    before = _encode_valid(df, y, tr_idx, va_idx)
    after = _encode_valid(df, y_poisoned, tr_idx, va_idx)
    np.testing.assert_array_equal(before, after)


def test_full_fit_leaks_valid_rows():
    """反证：如果像 train_final 那样在全量上 fit，验证行的编码会带上它自己的标签。

    所以 SHAP/可解释性必须用 OOF-TE（P3 的 src/eval/interpret.py），
    不能沿用 train_final 的自泄漏编码。
    """
    df = _frame()
    tr_idx = [0, 1, 2, 3, 4]
    va_idx = [5, 6, 7, 8]
    y = df["y"].to_numpy()
    y_poisoned = y.copy()
    y_poisoned[va_idx] = 1 - y_poisoned[va_idx]

    te_a = TargetEncoder("Flight Distance", smooth=2.0).fit(df, y)
    te_b = TargetEncoder("Flight Distance", smooth=2.0).fit(df, y_poisoned)
    assert not np.array_equal(
        te_a.transform(df.iloc[va_idx]).to_numpy(),
        te_b.transform(df.iloc[va_idx]).to_numpy(),
    )


def test_unseen_category_falls_back_to_prior():
    df = _frame()
    te = TargetEncoder("Flight Distance", smooth=2.0).fit(df, df["y"])

    out = te.transform(pd.DataFrame({"Flight Distance": [999, 12345]}))
    assert out.iloc[0] == pytest.approx(te.prior, rel=1e-6)
    assert out.iloc[1] == pytest.approx(te.prior, rel=1e-6)
    assert np.isfinite(out).all()


def test_smoothing_formula():
    """te = (sum_y + m * prior) / (count + m)。"""
    df = pd.DataFrame({
        "Flight Distance": [10, 10, 10, 20, 20, 30],
        "y": [1, 0, 1, 1, 1, 0],
    })
    m = 2.0
    te = TargetEncoder("Flight Distance", smooth=m).fit(df, df["y"])
    prior = float(df["y"].mean())
    assert te.prior == pytest.approx(prior)

    stats = {"10": (2.0, 3), "20": (2.0, 2), "30": (0.0, 1)}
    for key, (s, c) in stats.items():
        expect = (s + m * prior) / (c + m)
        assert te.mapping[key] == pytest.approx(expect, rel=1e-12)
        got = te.transform(pd.DataFrame({"Flight Distance": [int(key)]})).iloc[0]
        assert got == pytest.approx(expect, rel=1e-6)   # float32 输出


def test_transform_output_contract():
    df = _frame()
    te = TargetEncoder("Flight Distance", smooth=2.0).fit(df, df["y"])
    X = df.iloc[[5, 7]].copy()

    out = te.transform(X)
    assert out.name == "Flight Distance__te"
    assert out.dtype == np.float32
    assert out.index.equals(X.index)
    assert len(out) == len(X)


def test_fit_accepts_series_y_and_list_y():
    df = _frame()
    a = TargetEncoder("Flight Distance", smooth=2.0).fit(df, df["y"].to_numpy())
    b = TargetEncoder("Flight Distance", smooth=2.0).fit(df, list(df["y"]))
    assert a.prior == pytest.approx(b.prior)
    assert a.mapping.keys() == b.mapping.keys()
    for k in a.mapping:
        assert a.mapping[k] == pytest.approx(b.mapping[k])
