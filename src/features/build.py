"""特征构造（不含任何用到标签的变换，目标编码见 encode.py）。

四个特征组，可通过 --groups 组合：
  base  原始列
  A     语义修正：0 分 = "不适用"，不是"最差"。加 is_zero 标记，原列 0 -> NaN
  B     聚合/交互（隐含 A）：n_zeros / n_fives / 评分均值·最小·标准差 / 舱位×出行目的 /
        总延误 / 延误差值 / 距离分箱
  C     频次编码（Flight Distance、Age）——只用特征，不用标签，全量统计安全
  T     目标编码（Flight Distance）——**必须**在训练折内 fit，见 encode.py

T 单独成组是有意的：它是全项目唯一有泄漏风险的变换，必须能单独开关、
单独出一行消融，才能证明"折内拟合"这件事真的做对了。

B 隐含 A 是有意的：在 0=不适用 的前提下做聚合才有意义，
否则"平均评分"会被 0 拉低，得到的是没有业务含义的数。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CAT_CROSS = ("Type of Travel", "Class")
DEP = "Departure Delay in Minutes"
ARR = "Arrival Delay in Minutes"
DIST = "Flight Distance"


def add_features(df: pd.DataFrame, groups, schema: dict,
                 count_ref: pd.DataFrame | None = None) -> pd.DataFrame:
    g = set(groups)
    g.discard("base")
    if "B" in g:
        g.add("A")
    out = df.copy()
    svc = [c for c in schema["service_cols"] if c in out.columns]

    if "A" in g and svc:
        for c in svc:
            out[f"{c}__is_zero"] = (out[c] == 0).astype("int8")
        out[svc] = out[svc].mask(out[svc] == 0)

    if "B" in g and svc:
        raw = df[svc].to_numpy(dtype="float64")
        out["n_zeros"] = (raw == 0).sum(axis=1).astype("int16")
        out["n_fives"] = (raw == 5).sum(axis=1).astype("int16")

        sub = out[svc]
        out["svc_mean"] = sub.mean(axis=1)
        out["svc_min"] = sub.min(axis=1)
        out["svc_std"] = sub.std(axis=1)

        if all(c in out.columns for c in CAT_CROSS):
            out["travel_class"] = (
                out[CAT_CROSS[0]].astype(str) + "|" + out[CAT_CROSS[1]].astype(str)
            )

        if DEP in out.columns and ARR in out.columns:
            out["total_delay"] = out[DEP].fillna(0) + out[ARR].fillna(0)
            out["arr_minus_dep"] = out[ARR] - out[DEP]
            out["is_delayed"] = (out[DEP] > 0).astype("int8")
            out["delay_log"] = np.log1p(out["total_delay"].clip(lower=0))

        if DIST in out.columns:
            r = out[DIST].rank(method="first")
            out["dist_bin"] = pd.qcut(r, 10, labels=False).astype("float32")

    if "C" in g:
        ref = count_ref if count_ref is not None else df
        for c in (DIST, "Age"):
            if c in out.columns and c in ref.columns:
                vc = ref[c].value_counts()
                out[f"{c}__count"] = out[c].map(vc).fillna(0).astype("float32")

    return out


def prepare_X(df: pd.DataFrame, schema: dict, groups,
              count_ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """构造特征矩阵并统一类别列 dtype。"""
    x = add_features(df, groups, schema, count_ref=count_ref)
    drop = [c for c in (schema["target"], schema["id_col"]) if c and c in x.columns]
    x = x.drop(columns=drop)

    for c in x.columns:
        if x[c].dtype == object:
            x[c] = x[c].astype("category")
    return x
