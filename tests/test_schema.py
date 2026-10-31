"""列类型探测与目标二值化的回归测试。

重点复现两个 pandas 3 问题：
1. 字符串列 dtype 是 `str`（不是 object），detect_schema 必须仍能找出 4 个类别列；
2. to_binary 必须兼容 bool / 0-1 / 字符串 / 'satisfied' 各种目标列写法。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.schema import detect_schema, to_binary

SERVICE = [
    "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink",
    "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
]
CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]


def _raw_like_df(string_dtype: str = "str") -> pd.DataFrame:
    """结构与 data/raw/train.csv 一致的合成表（60 行足够）。"""
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "id": np.arange(n),
        "Age": rng.integers(18, 70, n),
        "Flight Distance": rng.integers(100, 4000, n),
        "Departure Delay in Minutes": rng.integers(0, 90, n),
        "Arrival Delay in Minutes": rng.integers(0, 90, n).astype("float64"),
    })
    for c in SERVICE:
        df[c] = rng.integers(1, 6, n)
    df["Gender"] = pd.Series(["Female", "Male"] * (n // 2), dtype=string_dtype)
    df["Customer Type"] = pd.Series(
        ["Loyal Customer", "disloyal Customer"] * (n // 2), dtype=string_dtype
    )
    df["Type of Travel"] = pd.Series(
        ["Business travel", "Personal Travel"] * (n // 2), dtype=string_dtype
    )
    df["Class"] = pd.Series(
        ["Business", "Eco", "Eco Plus"] * (n // 3) + ["Business"] * (n % 3), dtype=string_dtype
    )
    df["satisfaction"] = rng.integers(0, 2, n).astype("int8")
    return df


def _cfg() -> dict:
    return {"data": {"target": "satisfaction", "id_col": "id"}, "features": {}}


@pytest.mark.parametrize("string_dtype", ["str", "object", "category"])
def test_detect_schema_finds_four_category_columns(string_dtype: str) -> None:
    df = _raw_like_df(string_dtype)
    schema = detect_schema(df, _cfg(), verbose=False)

    assert set(schema["cat_cols"]) == set(CAT_COLS), (
        f"dtype={string_dtype} 时类别列丢了：{schema['cat_cols']}"
    )
    assert len(schema["cat_cols"]) == 4
    assert schema["target"] == "satisfaction"
    assert schema["id_col"] == "id"


@pytest.mark.parametrize("string_dtype", ["str", "object"])
def test_detect_schema_finds_thirteen_service_columns(string_dtype: str) -> None:
    df = _raw_like_df(string_dtype)
    schema = detect_schema(df, _cfg(), verbose=False)

    assert len(schema["service_cols"]) == 13
    assert set(schema["service_cols"]) == set(SERVICE)
    # 三组互斥：服务列 / 类别列不出现在数值列里
    assert not set(schema["service_cols"]) & set(schema["cat_cols"])
    assert not set(schema["service_cols"]) & set(schema["num_cols"])
    assert not set(schema["cat_cols"]) & set(schema["num_cols"])
    # 数值列保持原有逻辑（id / 目标不进任何组）
    assert "id" not in schema["num_cols"]
    assert "satisfaction" not in schema["num_cols"]


def test_detect_schema_str_dtype_is_not_object() -> None:
    """pandas 3 的回归护栏：str dtype 不是 object，但仍然算文本列。"""
    s = pd.Series(["Female", "Male"], dtype="str")
    assert not pd.api.types.is_object_dtype(s.dtype)
    assert pd.api.types.is_string_dtype(s.dtype)


@pytest.mark.parametrize(
    "series, expected",
    [
        (pd.Series([True, False, True]), [1, 0, 1]),
        (pd.Series([True, False]), [1, 0]),
        (pd.Series([0, 1, 1, 0], dtype="int64"), [0, 1, 1, 0]),
        (pd.Series([0.0, 1.0]), [0, 1]),
        (pd.Series(["satisfied", "neutral or dissatisfied"], dtype="str"), [1, 0]),
        (pd.Series(["satisfied", "neutral or dissatisfied"], dtype="object"), [1, 0]),
        (pd.Series(["Satisfied ", " NEUTRAL OR DISSATISFIED"], dtype="str"), [1, 0]),
        (pd.Series(["yes", "no"], dtype="str"), [1, 0]),
        (pd.Series(["1", "0"], dtype="str"), [1, 0]),
    ],
)
def test_to_binary_accepts_supported_encodings(series: pd.Series, expected: list[int]) -> None:
    out = to_binary(series)
    assert out.tolist() == expected
    assert str(out.dtype) == "int8"


def test_to_binary_maps_two_unknown_labels_by_sorted_order() -> None:
    out = to_binary(pd.Series(["b", "a", "b"], dtype="str"))
    assert out.tolist() == [1, 0, 1]


@pytest.mark.parametrize(
    "series",
    [
        pd.Series([0, 1, 2]),
        pd.Series(["a", "b", "c"], dtype="str"),
        pd.Series([0.0, float("nan")]),
        pd.Series(["satisfied", None], dtype="object"),
    ],
)
def test_to_binary_rejects_non_binary_or_missing(series: pd.Series) -> None:
    with pytest.raises(ValueError):
        to_binary(series)
