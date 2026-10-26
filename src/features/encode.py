"""折内目标编码。

这是全项目最容易写出"CV 虚高 0.02~0.05、LB 不动"的地方。
铁律：fit() 只能看到训练折，transform() 才作用于验证折。
"""

from __future__ import annotations

import pandas as pd


class TargetEncoder:
    """带平滑的目标编码：te = (sum_y + m * prior) / (count + m)。"""

    def __init__(self, col: str, smooth: float = 20.0):
        self.col = col
        self.smooth = float(smooth)
        self.mapping: dict[str, float] = {}
        self.prior: float = 0.0

    def fit(self, X: pd.DataFrame, y) -> "TargetEncoder":
        y = pd.Series(y, index=X.index, dtype="float64")
        key = X[self.col].astype(str)
        g = y.groupby(key).agg(["sum", "count"])
        self.prior = float(y.mean())
        self.mapping = (
            (g["sum"] + self.smooth * self.prior) / (g["count"] + self.smooth)
        ).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.Series:
        key = X[self.col].astype(str)
        return (
            key.map(self.mapping)
            .fillna(self.prior)
            .astype("float32")
            .rename(f"{self.col}__te")
        )
