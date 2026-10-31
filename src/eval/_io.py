"""P3 评估层共用 IO 与分群工具。

为什么单独抽一层：
- `evaluate` 与 `badcase` 都必须按 OOF 的 `row` 列对齐 train 行，**绝不按 id merge**。
  一旦 OOF 被排序、或 train 的 id 有重复，merge 会静默错位，误差分析就全废了。
- 六个分群口径（Class / Type of Travel / Customer Type / 延误桶 / 距离五分位 /
  Online boarding）只允许有一份实现，否则 report 里的数字会对不上。

对 OOF 的约定（与 src/models/train.py、predict.py 一致）：
    row   训练行号，0..n_train-1，唯一
    y     0/1
    pred  [0, 1] 的概率
可额外带 id，但本模块不依赖它。
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import ROOT

OOF_COLUMNS = ("row", "y", "pred")

# 分群列名（同时用于 evaluate 的 segment_metrics.csv 与 badcase 的汇总表）
SEGMENT_COLUMNS = [
    "Class",
    "Type of Travel",
    "Customer Type",
    "Departure Delay (min)",
    "Flight Distance (quintile)",
    "Online boarding",
]

# 有序分群的展示顺序；其余分群按样本量从大到小
DELAY_ORDER = ("0", "1-15", "16-60", ">60", "缺失")
DISTANCE_ORDER = ("Q1", "Q2", "Q3", "Q4", "Q5", "缺失")
ONLINE_BOARDING_ORDER = tuple(str(i) for i in range(0, 7)) + ("缺失",)

SEGMENT_GROUP_ORDER = {
    "Departure Delay (min)": DELAY_ORDER,
    "Flight Distance (quintile)": DISTANCE_ORDER,
    "Online boarding": ONLINE_BOARDING_ORDER,
}


def _resolve(p: str | Path) -> Path:
    """相对路径按仓库根目录解析（与 src.common.config 一致）。"""
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def resolve_oof_path(cfg: dict, oof: str | Path | None = None) -> Path:
    """定位 OOF：显式 --oof > P2_final_oof.parquet > 最新 run 的 OOF。"""
    if oof:
        p = _resolve(oof)
        if not p.exists():
            raise FileNotFoundError(f"指定的 OOF 不存在：{p}")
        return p

    oof_dir = _resolve(cfg["paths"]["oof_dir"])
    preferred = oof_dir / "P2_final_oof.parquet"
    if preferred.exists():
        return preferred

    if oof_dir.exists():
        candidates = sorted(
            oof_dir.glob("*.parquet"),
            key=lambda q: q.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            print(f"[oof] 未找到 {preferred.name}，回退到最新 run 的 OOF：{candidates[0]}")
            return candidates[0]

    raise FileNotFoundError(
        f"找不到可用的 OOF：{preferred} 不存在\n"
        "  先确认 P2 已生成 OOF（70 万行 5 折，约 1 分钟）：\n"
        '    $env:PYTHONPATH="."; python -m src.models.predict --config configs/base.yaml --groups base,C,T\n'
        "  或显式指定某次 run 的 OOF：\n"
        "    python -m src.eval.evaluate --oof reports/oof/<run_id>.parquet"
    )


def load_train(cfg: dict) -> pd.DataFrame:
    """读 data/processed/train.parquet；缺失时给出可操作提示。"""
    p = _resolve(cfg["paths"]["processed_dir"]) / "train.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"找不到训练集：{p}\n"
            "  先跑 make data（或 python -m src.data.make_dataset --config configs/base.yaml）"
        )
    return pd.read_parquet(p)


def load_oof(
    path: str | Path,
    *,
    n_rows: int | None = None,
    allow_partial: bool = False,
    context: str = "eval",
) -> pd.DataFrame:
    """读取并校验 OOF，返回按 row 升序排列的 [row, y, pred]。

    校验项：约定列、缺失/非数值、y 二值、row 唯一且越界、长度、概率范围。
    概率越界会裁剪并警告；缺失直接报错——OOF 有 NaN 说明生成端已经坏了，
    静默剔除会让 AUC 与报告对不上。
    """
    p = _resolve(path)
    if not p.exists():
        raise FileNotFoundError(f"[{context}] OOF 不存在：{p}")
    try:
        raw = pd.read_parquet(p)
    except Exception as e:  # pragma: no cover - 依赖损坏时才会走到
        raise ValueError(f"[{context}] 读取 OOF 失败：{p}\n  原因：{e}\n  请确认是 parquet 且 pyarrow 正常") from e

    missing = [c for c in OOF_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(
            f"[{context}] OOF 缺少列 {missing}；约定列是 row,y,pred（可带 id）。文件：{p}"
        )
    if len(raw) == 0:
        raise ValueError(f"[{context}] OOF 是空表：{p}")

    out = raw[list(OOF_COLUMNS)].copy()
    for c in OOF_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    values = out.to_numpy(dtype="float64")
    bad = ~np.isfinite(values)
    if bad.any():
        counts = {c: int((~np.isfinite(out[c].to_numpy(dtype="float64"))).sum()) for c in OOF_COLUMNS}
        raise ValueError(
            f"[{context}] OOF 存在缺失/非数值：{counts}。文件：{p}\n"
            "  请回查生成 OOF 的 run（run_cv / predict_test），不要在这里静默删除行。"
        )

    if not out["y"].isin([0.0, 1.0]).all():
        bad_y = sorted(out.loc[~out["y"].isin([0.0, 1.0]), "y"].unique())[:5]
        raise ValueError(f"[{context}] y 必须是 0/1，发现 {bad_y}。文件：{p}")

    if not np.allclose(out["row"], np.round(out["row"])):
        raise ValueError(f"[{context}] row 必须是整数行号。文件：{p}")
    out["row"] = out["row"].astype("int64")
    out["y"] = out["y"].astype("int8")

    if (out["row"] < 0).any():
        raise ValueError(f"[{context}] row 出现负数：{(out['row'] < 0).sum()} 行。文件：{p}")
    dup = int(out["row"].duplicated().sum())
    if dup:
        raise ValueError(
            f"[{context}] row 有 {dup} 个重复值；OOF 每行只能出现一次。文件：{p}\n"
            "  常见原因：多次 run 的 OOF 被拼到一起。"
        )

    if n_rows is not None:
        if (out["row"] >= n_rows).any():
            raise ValueError(
                f"[{context}] OOF 的 row 最大值 {int(out['row'].max())} 超出训练集行数 {n_rows}；"
                "OOF 与 train 不是同一份数据。"
            )
        if len(out) != n_rows:
            if not allow_partial:
                raise ValueError(
                    f"[{context}] OOF 行数 {len(out):,} 与 train {n_rows:,} 不一致。\n"
                    "  常规 OOF 应覆盖全部训练行；若只是合成数据冒烟，请加 --allow-partial。\n"
                    "  否则请回查生成 OOF 的 run（是否采样/截断）。"
                )
            warnings.warn(
                f"[{context}] OOF 仅覆盖 {len(out):,}/{n_rows:,} 行，按 row 对齐后继续（--allow-partial）。",
                stacklevel=2,
            )

    out_of_range = int(((out["pred"] < 0) | (out["pred"] > 1)).sum())
    if out_of_range:
        warnings.warn(
            f"[{context}] {out_of_range} 个预测概率越界，已裁剪到 [0,1]；请检查模型输出。",
            stacklevel=2,
        )
        out["pred"] = out["pred"].clip(0.0, 1.0)

    return out.sort_values("row", kind="mergesort").reset_index(drop=True)


def align_train_features(train: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    """按 OOF.row 取 train 的对应行（位置对齐，不做 id merge）。"""
    if "row" not in oof.columns:
        raise ValueError("[align] OOF 缺少 row 列，无法与 train 对齐")
    rows = oof["row"].to_numpy(dtype="int64")
    if rows.size and int(rows.max()) >= len(train):
        raise ValueError(
            f"[align] row 最大值 {int(rows.max())} 超出 train 行数 {len(train)}，无法对齐"
        )
    if rows.size and int(rows.min()) < 0:
        raise ValueError("[align] row 出现负数，无法对齐")
    return train.iloc[rows].reset_index(drop=True)


def delay_bucket(s: pd.Series) -> pd.Series:
    """出发延误分桶：0（未延误）/ 1-15 / 16-60 / >60（分钟），缺失单列。"""
    vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")
    labels = np.select(
        [np.isnan(vals), vals <= 0, vals <= 15, vals <= 60],
        ["缺失", "0", "1-15", "16-60"],
        default=">60",
    )
    return pd.Series(labels, index=s.index, name="Departure Delay (min)")


def distance_quintile(s: pd.Series) -> pd.Series:
    """Flight Distance 五分位（Q1 最短 ... Q5 最长），缺失单列。"""
    vals = pd.to_numeric(s, errors="coerce")
    ranks = pd.Series(vals, index=s.index).rank(method="first")
    out = pd.Series("缺失", index=s.index, dtype="object")
    valid = ranks.notna()
    n_valid = int(valid.sum())
    if n_valid >= 5:
        q = pd.qcut(ranks[valid], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        out.loc[valid] = q.astype("object")
    elif n_valid > 0:
        # 冒烟小数据不足 5 行时，qcut 会产生重复分位点；统一归 Q1，避免直接抛错
        out.loc[valid] = "Q1"
    return out.astype(str).rename("Flight Distance (quintile)")


def _to_str_series(s: pd.Series) -> pd.Series:
    """数值/类别统一转字符串，缺失值记为“缺失”。"""
    obj = s.astype("object")
    obj = obj.where(s.notna(), "缺失")
    return obj.astype(str)


def build_segment_frame(feats: pd.DataFrame) -> pd.DataFrame:
    """从对齐后的 train 特征构造统一分群表（索引已重置）。"""
    required = [
        "Class",
        "Type of Travel",
        "Customer Type",
        "Departure Delay in Minutes",
        "Flight Distance",
        "Online boarding",
    ]
    missing = [c for c in required if c not in feats.columns]
    if missing:
        raise ValueError(
            f"[segment] 分群缺少列 {missing}；需要 {required}。\n"
            "  请确认传入的是 train 原始列（不是 prepare_X 之后的特征矩阵）。"
        )

    out = pd.DataFrame(index=feats.index)
    out["Class"] = _to_str_series(feats["Class"])
    out["Type of Travel"] = _to_str_series(feats["Type of Travel"])
    out["Customer Type"] = _to_str_series(feats["Customer Type"])
    out["Departure Delay (min)"] = delay_bucket(feats["Departure Delay in Minutes"]).to_numpy()
    out["Flight Distance (quintile)"] = distance_quintile(feats["Flight Distance"]).to_numpy()
    out["Online boarding"] = _to_str_series(feats["Online boarding"].astype("Int64"))
    return out.reset_index(drop=True)


def is_string_column(dtype) -> bool:
    """判断列 dtype 是否是 LightGBM 不接受的字符串类型（pandas 3 的 str / object）。"""
    if isinstance(dtype, pd.CategoricalDtype):
        return False
    return (
        dtype == object
        or isinstance(dtype, pd.StringDtype)
        or pd.api.types.is_string_dtype(dtype)
    )


def ensure_lgbm_dtypes(X: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """兼容兜底：把 pandas 3 的 str/object 列转成 category。

    背景：pandas 3 默认把字符串读成 `str` dtype，LightGBM 只接受
    int/float/bool/category，`prepare_X` 目前只处理 object，因此这里补一刀。
    等 `src/features/build.py` 修好后本函数是 no-op（可保留，不影响结果）。
    """
    out = X.copy()
    converted: list[str] = []
    for c in out.columns:
        dt = out[c].dtype
        if (
            isinstance(dt, pd.CategoricalDtype)
            or pd.api.types.is_numeric_dtype(dt)
            or pd.api.types.is_bool_dtype(dt)
        ):
            continue
        if is_string_column(dt):
            out[c] = out[c].astype("category")
            converted.append(str(c))
    if converted and verbose:
        preview = ", ".join(converted[:8]) + (" ..." if len(converted) > 8 else "")
        print(f"[dtype] prepare_X 输出仍有 str 列，已临时转 category：{preview}")
    return out


def _json_safe(v: Any) -> Any:
    """把 numpy / pandas / NaN 递归转成可 json.dump 的对象。"""
    if isinstance(v, dict):
        return {str(k): _json_safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, np.ndarray):
        return [_json_safe(x) for x in v.tolist()]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return _json_safe(float(v))
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, Path):
        return str(v)
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    return v


def write_json(path: str | Path, obj: Any) -> Path:
    """写 JSON（自动建目录、NaN -> null、中文不转义）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(_json_safe(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return p
