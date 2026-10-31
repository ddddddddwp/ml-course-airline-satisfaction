"""生成特征字典（P1-07 验收项：字段全覆盖）。

做法：真的把特征矩阵跑一遍，把 X 里出现的每一列都写进表里。
这样"全覆盖"是**构造保证**的，不是靠人肉数。

产出：docs/feature-dictionary.md（--out 可改；相对路径按仓库根解析）

用法：
    python scripts/gen_feature_dict.py --config configs/base.yaml
    python scripts/gen_feature_dict.py --out docs/feature-dictionary.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.config import ROOT, load_config, path_of
from src.data.schema import detect_schema
from src.features.build import add_features, prepare_X

# 原始列说明（写报告时可直接抄）
BASE_DESC = {
    "Age": ("int", "乘客年龄 7–85", "保留；树模型可自行分段，线性模型建议分箱"),
    "Flight Distance": ("int", "航段距离（英里），高基数，可作航线代理", "保留 + C 组频次编码 + T 组目标编码"),
    "Departure Delay in Minutes": ("int", "起飞延误（分钟），约 90% 为 0", "保留 + log1p + is_delayed"),
    "Arrival Delay in Minutes": ("float", "到达延误（分钟），0.04% 缺失；**飞行后才可知**", "行内用起飞延误填补 + 缺失标记；业务口径若为「飞行前」需剔除"),
    "arr_delay_missing": ("int8", "到达延误是否缺失", "保留为特征，缺失本身可能携带信息"),
    "Gender": ("str", "性别", "类别编码"),
    "Customer Type": ("str", "常旅客 / 非常旅客", "类别编码"),
    "Type of Travel": ("str", "商务出行 / 个人出行", "类别编码；强信号"),
    "Class": ("str", "舱位 Business / Eco / Eco Plus", "类别编码；最强单列"),
    "satisfaction": ("int8", "目标：是否满意", "不进入特征"),
}

SERVICE_DESC = "0 = 不适用（不是最差）；1–5 评分"

DERIVED_DESC = {
    "__is_zero": ("1 表示该评分填了 0（服务不适用）", "把「不适用」从「最差」里拆出来"),
    "n_zeros": ("一行里评 0 的列数", "整体「不适用」程度"),
    "n_fives": ("一行里评 5 的列数", "整体满意度"),
    "svc_mean": ("13 项评分中 1–5 部分的均值（0 已转为 NaN）", "服务综合水平"),
    "svc_min": ("13 项评分中的最低分", "短板效应"),
    "svc_std": ("13 项评分的标准差", "服务一致性"),
    "travel_class": ("Type of Travel × Class 交叉", "类别交互（线性模型必需）"),
    "total_delay": ("起飞延误 + 到达延误", "总延误"),
    "arr_minus_dep": ("到达延误 − 起飞延误", "空中追回/损失的时间"),
    "is_delayed": ("起飞是否延误 > 0", "延误与否的二值化"),
    "delay_log": ("log1p(总延误)", "压长尾，线性模型友好"),
    "dist_bin": ("Flight Distance 的 10 分位分箱", "距离的非线性"),
    "__count": ("该取值在全量（train+test）出现的次数", "频次编码：只用特征、不用标签，无泄漏"),
    "__te": ("目标编码（平滑 m=20），**只在训练折内拟合**", "高基数特征的均值编码"),
}


def describe(col: str) -> tuple[str, str, str]:
    if col in BASE_DESC:
        return BASE_DESC[col]
    for suffix, (d, why) in DERIVED_DESC.items():
        if col.endswith(suffix) or col == suffix:
            parent = col[: -len(suffix)] if col.endswith(suffix) and col != suffix else ""
            src = f"（基于 {parent}）" if parent else ""
            return (d + src, why, "")
    if col in ("Inflight wifi service", "Cleanliness"):
        return (SERVICE_DESC, "0 单独标记 + 0→NaN", "")
    return (SERVICE_DESC, "0 单独标记 + 0→NaN", "")


def group_of(col: str) -> str:
    if col.endswith("__is_zero"):
        return "A"
    if col.endswith("__count"):
        return "C"
    if col.endswith("__te"):
        return "T"
    if col in {"n_zeros", "n_fives", "svc_mean", "svc_min", "svc_std",
               "travel_class", "total_delay", "arr_minus_dep", "is_delayed",
               "delay_log", "dist_bin"}:
        return "B"
    return "base"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default="docs/feature-dictionary.md")
    args = ap.parse_args()

    cfg = load_config(args.config)
    proc = path_of(cfg, "processed_dir")
    tr_path = proc / "train.parquet"
    if not tr_path.exists():
        raise SystemExit(f"[dict] 找不到 {tr_path}，先跑 make data")
    df = pd.read_parquet(tr_path)
    test = pd.read_parquet(proc / "test.parquet")
    schema = detect_schema(df, cfg, verbose=False)

    cols = [c for c in ("Flight Distance", "Age") if c in test.columns]
    count_ref = pd.concat([df[cols], test[cols]], ignore_index=True) if cols else None

    # 用"全开"的特征矩阵取列清单，保证与实际训练一致
    X = prepare_X(df, schema, ["base", "A", "B", "C"], count_ref=count_ref)
    X["Flight Distance__te"] = 0.0  # T 组在折内生成，这里只登记

    rows = []
    for c in X.columns:
        g = group_of(c)
        dtype = str(df[c].dtype) if c in df.columns else str(X[c].dtype)
        desc, how, _ = describe(c)
        nuniq = df[c].nunique() if c in df.columns else ""
        rows.append({"列名": c, "组": g, "类型": dtype,
                     "唯一值": nuniq, "含义": desc, "处理策略": how})

    table = pd.DataFrame(rows)
    counts = table["组"].value_counts().reindex(["base", "A", "B", "C", "T"]).fillna(0).astype(int)

    lines = [
        "# 特征字典 v1",
        "",
        "> 由 `python scripts/gen_feature_dict.py` 自动生成，列清单直接取自训练时的特征矩阵。",
        "> 改动特征后重新生成即可，**不需要手工维护**——这就是「字段全覆盖」的保证。",
        "",
        "## 汇总",
        "",
        "| 组 | 含义 | 列数 |",
        "|---|---|---:|",
        f"| base | 原始字段 | {counts['base']} |",
        f"| A | 语义修正（0 = 不适用） | {counts['A']} |",
        f"| B | 聚合 / 交互 | {counts['B']} |",
        f"| C | 频次编码 | {counts['C']} |",
        f"| T | 目标编码（折内拟合） | {counts['T']} |",
        f"| **合计** | | **{len(table)}** |",
        "",
        "## 字段明细",
        "",
        "| 列名 | 组 | 类型 | 唯一值 | 含义 | 处理策略 |",
        "|---|---|---|---:|---|---|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| `{r['列名']}` | {r['组']} | {r['类型']} | {r['唯一值']} | {r['含义']} | {r['处理策略']} |"
        )

    lines += [
        "",
        "## 使用约定",
        "",
        "1. `base` 永远可用；A/B/C/T 通过 `--groups` 组合。",
        "2. `B` 隐含 `A`：不做 0→NaN 的语义修正，聚合特征没有业务含义。",
        "3. `T` 的目标编码**只在训练折 fit**，验证折与 test 只 transform。",
        "4. `Arrival Delay in Minutes` 是飞行后才可知的事后特征，业务口径若为「飞行前预测」，必须剔除并报告损失。",
        "",
    ]

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")

    print(f"[dict] {len(table)} 列  base={counts['base']} A={counts['A']} "
          f"B={counts['B']} C={counts['C']} T={counts['T']}")
    print(f"[dict] -> {out}")

    # 自检：X 的每一列都必须在文档里出现
    missing = [c for c in X.columns if f"`{c}`" not in "\n".join(lines)]
    if missing:
        raise SystemExit(f"[dict][错误] 以下列没有写进字典：{missing}")
    print("[dict] 自检通过：特征矩阵的每一列都已登记")


if __name__ == "__main__":
    main()
