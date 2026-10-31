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

## 为什么要有 FeatureBuilder

B 组的 `dist_bin` 是**十分位分箱**，边界必须来自一份固定的参照数据（全量 train）。
早期实现是在每个子集上各自 `qcut(rank, 10)`：边界按子集重算，同一位乘客在
OOF（全量 train 上一次算好）与测试集（单独 prepare）里会落进不同的箱子——
特征语义随调用方式漂移，OOF 分数和 submission 就不再是同一件事。

`FeatureBuilder.fit(df)` 把 Flight Distance 的 10 个分位边界固化下来，
`transform(df)` 用 `pd.cut(固定边界)` 落箱：任何子集、任何行顺序、任何调用方
得到的 `dist_bin` 都一致。C 组的 `count_ref`（train+test 频次表）同理由调用方
一次性传入，不做逐子集重算。

## pandas 3 兼容

pandas 3 里字符串列 dtype 是 `str` 而不是 `object`，只判 `dtype == object` 会漏，
把 `str` 列原样喂给 LightGBM 会报
"pandas dtypes must be int, float or bool. Fields: Gender: str..."。
`prepare_X` 统一用 is_object_dtype / is_string_dtype / CategoricalDtype 判断，
非数值列一律 `astype("category")`，category 原样保留。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CAT_CROSS = ("Type of Travel", "Class")
DEP = "Departure Delay in Minutes"
ARR = "Arrival Delay in Minutes"
DIST = "Flight Distance"
AGE = "Age"
COUNT_COLS = (DIST, AGE)
N_DIST_BINS = 10


def normalize_groups(groups) -> list[str]:
    """规范化特征组：去掉 base（base 隐含）、B 补上 A、去重保序。

    返回的列表**不含** "base"；feat_ver 之类的字符串请用原始 groups 生成，
    否则 B 会被写成 "A+B"，历史 run_id 对不上。
    """
    out: list[str] = []
    for g in groups:
        g = str(g).strip()
        if not g or g == "base" or g in out:
            continue
        out.append(g)
    if "B" in out and "A" not in out:
        out.insert(0, "A")
    return out


def dist_edges_from(values, n_bins: int = N_DIST_BINS) -> np.ndarray:
    """在全量参照数据上算 Flight Distance 的分位边界（fit 阶段用一次）。"""
    v = pd.Series(values).dropna().to_numpy(dtype="float64")
    if v.size == 0:
        raise ValueError("[features] Flight Distance 全为空，无法确定 dist_bin 边界")
    edges = np.quantile(v, np.linspace(0.0, 1.0, int(n_bins) + 1))
    # 距离有大量重复值；宁可少几箱，也不能让边界随传入的行子集变化
    edges = np.unique(edges)
    if edges.size < 3:
        raise ValueError(
            f"[features] Flight Distance 只有 {edges.size} 个不同分位点，无法分箱"
        )
    return edges


def make_count_ref(*frames: pd.DataFrame | None, cols=COUNT_COLS) -> pd.DataFrame | None:
    """C 组频次编码的参照表：train + test 全量（只用特征列，不用标签）。

    一份参照表要在 run_cv / predict / diag 里保持完全一致，所以收口到这里。
    """
    parts = []
    for f in frames:
        if f is None or len(f) == 0:
            continue
        use = [c for c in cols if c in f.columns]
        if use:
            parts.append(f[use])
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def add_features(df: pd.DataFrame, groups, schema: dict,
                 count_ref: pd.DataFrame | None = None,
                 dist_edges: np.ndarray | None = None) -> pd.DataFrame:
    """按组追加特征列（不含标签）。base 列始终保留。

    dist_edges：dist_bin 的固定边界（来自 FeatureBuilder.fit 的全量统计）。
    为 None 时退回"按当前 df 的 rank 现算分位"的旧路径，只适合一次性全量调用；
    任何多子集调用（逐折 / train-valid-test 分别处理）都必须传固定边界。
    """
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
            if dist_edges is None:
                r = out[DIST].rank(method="first")
                out["dist_bin"] = pd.qcut(r, N_DIST_BINS, labels=False).astype("float32")
            else:
                out["dist_bin"] = pd.cut(
                    out[DIST],
                    bins=np.asarray(dist_edges, dtype="float64"),
                    labels=False,
                    include_lowest=True,
                ).astype("float32")

    if "C" in g:
        ref = count_ref if count_ref is not None else df
        for c in COUNT_COLS:
            if c in out.columns and c in ref.columns:
                vc = ref[c].value_counts()
                out[f"{c}__count"] = out[c].map(vc).fillna(0).astype("float32")

    return out


def to_category(x: pd.DataFrame) -> pd.DataFrame:
    """把字符串/object 列统一成 category（pandas 3 的 `str` dtype 也算）。"""
    for c in x.columns:
        dt = x[c].dtype
        if isinstance(dt, pd.CategoricalDtype):
            continue
        if pd.api.types.is_object_dtype(dt) or pd.api.types.is_string_dtype(dt):
            x[c] = x[c].astype("category")
    return x


def prepare_X(df: pd.DataFrame, schema: dict, groups,
              count_ref: pd.DataFrame | None = None,
              dist_edges: np.ndarray | None = None) -> pd.DataFrame:
    """构造特征矩阵并统一类别列 dtype。

    dist_edges 见 add_features；多子集场景请改用 FeatureBuilder。
    """
    x = add_features(df, groups, schema, count_ref=count_ref, dist_edges=dist_edges)
    drop = [c for c in (schema["target"], schema["id_col"]) if c and c in x.columns]
    x = x.drop(columns=drop)
    return to_category(x)


class FeatureBuilder:
    """全量统计量的载体：fit 一次、transform 任意子集，特征语义不漂移。

    用法::

        fb = FeatureBuilder(schema, groups=["base", "B", "C"], count_ref=count_ref).fit(df)
        X_tr = fb.transform(df.iloc[tr])      # 折内训练
        X_va = fb.transform(df.iloc[va])      # 折内验证
        X_te = fb.transform(test)             # 测试集

    fit 只记录与标签无关的统计量（当前是 Flight Distance 的 10 分位边界），
    所以它看全量 train 不会造成泄漏。目标编码 T 不在这里，它必须逐折 fit
    （见 encode.py / train.run_cv）。
    """

    def __init__(self, schema: dict, groups=("base",),
                 count_ref: pd.DataFrame | None = None,
                 n_dist_bins: int = N_DIST_BINS):
        self.schema = schema
        self.groups = normalize_groups(groups)
        self.count_ref = count_ref
        self.n_dist_bins = int(n_dist_bins)
        self.dist_edges: np.ndarray | None = None
        self.n_fit_rows = 0
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        if "B" in self.groups and DIST in df.columns:
            self.dist_edges = dist_edges_from(df[DIST], self.n_dist_bins)
        else:
            self.dist_edges = None
        self.n_fit_rows = int(len(df))
        self.fitted = True
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("[features] FeatureBuilder 需要先 fit(全量 train) 再 transform")
        return prepare_X(df, self.schema, self.groups, count_ref=self.count_ref,
                         dist_edges=self.dist_edges)
