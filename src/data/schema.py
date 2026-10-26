"""列类型探测：把原始表分成 服务评分 / 类别 / 数值 三组。

真实数据到货后先跑一次，把 reports/schema.json 看一眼再当真。
配置里显式写了就以配置为准（避免自动探测在边界列上猜错）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def to_binary(s: pd.Series) -> pd.Series:
    """把目标列统一成 0/1，兼容 bool / 0-1 / 'satisfied' 字符串。"""
    if s.dtype == bool:
        return s.astype("int8")
    if pd.api.types.is_numeric_dtype(s):
        u = sorted(s.dropna().unique())
        if not set(u) <= {0, 1}:
            raise ValueError(f"目标列不是二值：{u[:5]}")
        return s.astype("int8")

    v = s.astype(str).str.strip().str.lower()
    known = {
        "satisfied": 1, "neutral or dissatisfied": 0,
        "yes": 1, "no": 0, "true": 1, "false": 0,
        "1": 1, "0": 0,
    }
    u = set(v.dropna().unique())
    if u <= set(known):
        return v.map(known).astype("int8")
    if len(u) != 2:
        raise ValueError(f"目标列不是二值：{u}")
    lo, hi = sorted(u)
    return (v == hi).astype("int8")


def detect_schema(df: pd.DataFrame, cfg: dict, verbose: bool = True) -> dict:
    target = cfg["data"]["target"]
    id_col = cfg["data"].get("id_col")
    skip = {target}
    if id_col:
        skip.add(id_col)

    fcfg = cfg.get("features", {}) or {}
    service = list(fcfg.get("service_cols") or [])
    cat = list(fcfg.get("cat_cols") or [])
    num = list(fcfg.get("num_cols") or [])

    if not service:
        for c in df.columns:
            if c in skip or c in cat:
                continue
            if not pd.api.types.is_integer_dtype(df[c]):
                continue
            nu = df[c].nunique(dropna=True)
            if 3 <= nu <= 7 and df[c].min() >= 0 and df[c].max() <= 6:
                service.append(c)

    if not cat:
        cat = [
            c for c in df.columns
            if c not in skip
            and (df[c].dtype == object or isinstance(df[c].dtype, pd.CategoricalDtype))
        ]

    if not num:
        num = [
            c for c in df.columns
            if c not in skip and c not in service and c not in cat
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    schema = {"target": target, "id_col": id_col,
              "service_cols": service, "cat_cols": cat, "num_cols": num}

    if verbose:
        print(f"[schema] 目标列   : {target}")
        print(f"[schema] 服务评分 : {len(service)} 列  {service[:4]}{' ...' if len(service) > 4 else ''}")
        print(f"[schema] 类别列   : {len(cat)} 列  {cat}")
        print(f"[schema] 数值列   : {len(num)} 列  {num}")
        if len(service) < 3:
            print("[schema][警告] 服务评分列少于 3 个，特征组 A/B 可能失效，请检查配置")
    return schema


def save_schema(schema: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
