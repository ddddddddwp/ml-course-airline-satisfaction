"""特征构造的契约测试（P2：特征一致性 / pandas 3 兼容 / 列数冻结）。

这里锁定三件事：
1. pandas 3 的 `str` dtype 与旧 `object` dtype 都必须变成 category，
   否则 LightGBM 会报 "pandas dtypes must be int, float or bool. Fields: Gender: str..."。
2. 消融矩阵的列数是写进报告的契约：base=22 / A=35 / B=46 / C=24 / C+T=25。
3. FeatureBuilder 在全量上定的 dist_bin 边界，对任意行子集都必须稳定。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.config import load_config
from src.data.schema import detect_schema
from src.features.build import (
    FeatureBuilder,
    make_count_ref,
    normalize_groups,
    prepare_X,
)
from src.features.encode import TargetEncoder

SERVICE = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
]
CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
NUM_COLS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]


def make_df(n: int = 400, string_dtype: str = "str", seed: int = 0) -> pd.DataFrame:
    """造一张与 S6E10 同结构的小表（13 服务列 + 4 数值列 + 4 类别列 + 标记列）。"""
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {"id": np.arange(n)}
    data["Age"] = rng.integers(18, 70, n)
    data["Flight Distance"] = rng.integers(80, 5000, n)
    for c in SERVICE:
        data[c] = rng.integers(0, 6, n)          # 0 分存在，A 组才有意义
    data["Departure Delay in Minutes"] = rng.integers(0, 200, n)
    arr = rng.integers(0, 200, n).astype("float64")
    arr[::17] = np.nan
    data["Arrival Delay in Minutes"] = arr
    data["arr_delay_missing"] = np.isnan(arr).astype("int8")
    data["Gender"] = rng.choice(["Female", "Male"], n)
    data["Customer Type"] = rng.choice(["Loyal Customer", "disloyal Customer"], n)
    data["Type of Travel"] = rng.choice(["Business travel", "Personal Travel"], n)
    data["Class"] = rng.choice(["Business", "Eco", "Eco Plus"], n)
    data["satisfaction"] = (rng.random(n) < 0.45).astype("int8")

    df = pd.DataFrame(data)
    if string_dtype == "object":
        for c in CAT_COLS:
            df[c] = df[c].astype(object)
    elif string_dtype == "string":
        for c in CAT_COLS:
            df[c] = df[c].astype("string")
    elif string_dtype != "str":
        raise ValueError(string_dtype)
    return df


def schema_of(df: pd.DataFrame) -> dict:
    """用真实配置自动探测 schema（与 train/predict 入口一致）。"""
    return detect_schema(df, load_config("configs/base.yaml"), verbose=False)


# --------------------------------------------------------------------------- pandas 3 dtype

@pytest.mark.parametrize("string_dtype", ["str", "object", "string"])
def test_string_dtypes_become_category(string_dtype):
    """str / object / StringDtype 三种字符串列都必须变成 category。"""
    df = make_df(string_dtype=string_dtype)
    X = prepare_X(df, schema_of(df), ["base", "A", "B", "C"])

    for c in X.columns:
        assert not pd.api.types.is_object_dtype(X[c].dtype), f"{c} 还是 object"
        assert not pd.api.types.is_string_dtype(X[c].dtype), f"{c} 还是字符串"

    for c in [*CAT_COLS, "travel_class"]:
        assert isinstance(X[c].dtype, pd.CategoricalDtype), f"{c} 不是 category"


def test_lightgbm_accepts_prepared_matrix():
    """回归测试：修好 dtype 之前这里会抛 'pandas dtypes must be int, float or bool'。"""
    # Windows 上 lightgbm 必须先等 sklearn 加载（OpenMP/DLL 顺序问题，否则 access violation）
    import sklearn  # noqa: F401
    import lightgbm as lgb

    df = make_df(n=300)
    X = prepare_X(df, schema_of(df), ["base", "A", "B", "C"])
    model = lgb.LGBMClassifier(
        n_estimators=5, num_leaves=7, min_child_samples=5,
        verbose=-1, n_jobs=1,
    )
    model.fit(X, df["satisfaction"])          # 不抛异常即通过
    assert model.predict_proba(X.iloc[:5]).shape == (5, 2)


# --------------------------------------------------------------------------- 列数契约

def test_group_column_counts():
    """base=22 / A=35 / B=46 / C=24 / C+T=25（写进报告的消融矩阵契约）。"""
    df = make_df()
    schema = schema_of(df)
    assert schema["service_cols"] == SERVICE, "服务列探测结果与真实数据不一致"

    assert prepare_X(df, schema, ["base"]).shape[1] == 22
    assert prepare_X(df, schema, ["A"]).shape[1] == 35
    assert prepare_X(df, schema, ["B"]).shape[1] == 46
    assert prepare_X(df, schema, ["C"]).shape[1] == 24

    ct = prepare_X(df, schema, ["C", "T"])
    assert ct.shape[1] == 24, "T 组的 __te 列必须由折内 TargetEncoder 追加，prepare_X 不加"
    te = TargetEncoder("Flight Distance", smooth=20.0).fit(ct, df["satisfaction"])
    ct["Flight Distance__te"] = te.transform(ct)
    assert ct.shape[1] == 25


def test_B_implies_A():
    """B 组隐含 A：加不加 A 结果必须逐位相同。"""
    df = make_df()
    schema = schema_of(df)
    b_only = prepare_X(df, schema, ["B"])
    a_b = prepare_X(df, schema, ["A", "B"])

    assert list(b_only.columns) == list(a_b.columns)
    pd.testing.assert_frame_equal(b_only, a_b)
    assert sum(c.endswith("__is_zero") for c in b_only.columns) == len(SERVICE)
    for c in ("n_zeros", "n_fives", "svc_mean", "svc_std", "travel_class",
              "total_delay", "delay_log", "dist_bin"):
        assert c in b_only.columns


def test_normalize_groups():
    assert normalize_groups(["base"]) == []
    assert normalize_groups(["B"]) == ["A", "B"]
    assert normalize_groups(["base", "C", "T"]) == ["C", "T"]
    assert normalize_groups(["B", "C"]) == ["A", "B", "C"]


# --------------------------------------------------------------------------- FeatureBuilder

def test_feature_builder_dist_bin_is_stable_on_subset():
    """全量 fit 后 transform 子集，dist_bin 必须与全量结果逐行一致。"""
    df = make_df(n=500, seed=3)
    schema = schema_of(df)
    fb = FeatureBuilder(schema, ["base", "B", "C"], count_ref=make_count_ref(df)).fit(df)
    full = fb.transform(df)
    assert full["dist_bin"].notna().all()
    assert full["dist_bin"].nunique() == 10

    # 偏斜子集：只取距离最小的 35%。旧实现按子集重算分位，必然与全量不一致。
    sub = df[df["Flight Distance"] < df["Flight Distance"].quantile(0.35)]
    assert len(sub) > 20
    sub_x = fb.transform(sub)
    pd.testing.assert_series_equal(
        sub_x["dist_bin"], full["dist_bin"].loc[sub.index], check_names=False
    )

    # 同一个距离值只能落进同一个箱子（跨子集不变）
    pairs = pd.DataFrame({"fd": df["Flight Distance"], "bin": full["dist_bin"]})
    assert not pairs.drop_duplicates()["fd"].duplicated().any()


def test_prepare_X_without_fixed_edges_drifts_on_subset():
    """反证：不传固定边界的旧路径按子集重算分位，箱子编号不可比（这正是要修的 bug）。"""
    df = make_df(n=500, seed=3)
    schema = schema_of(df)
    full = prepare_X(df, schema, ["B"])
    sub = df[df["Flight Distance"] < df["Flight Distance"].quantile(0.35)]

    sub_x = prepare_X(sub, schema, ["B"])
    assert not np.array_equal(
        sub_x["dist_bin"].to_numpy(),
        full["dist_bin"].loc[sub.index].to_numpy(),
    )


def test_feature_builder_requires_fit():
    df = make_df(n=50)
    fb = FeatureBuilder(schema_of(df), ["base", "B"])
    assert fb.dist_edges is None
    with pytest.raises(RuntimeError):
        fb.transform(df)


def test_feature_builder_without_B_has_no_dist_bin():
    df = make_df(n=50)
    fb = FeatureBuilder(schema_of(df), ["base", "C"]).fit(df)
    assert fb.dist_edges is None
    assert "dist_bin" not in fb.transform(df).columns


def test_count_ref_is_train_plus_test():
    """C 的频次参照是 train+test 全量，不是单看 train。"""
    tr, te = make_df(n=300, seed=1), make_df(n=200, seed=2)
    ref = make_count_ref(tr, te)
    X = prepare_X(tr, schema_of(tr), ["C"], count_ref=ref)

    vc = pd.concat([tr["Flight Distance"], te["Flight Distance"]]).value_counts()
    fd = tr["Flight Distance"].iloc[0]
    assert X["Flight Distance__count"].iloc[0] == pytest.approx(float(vc.loc[fd]))
    assert X["Flight Distance__count"].max() <= len(tr) + len(te)
