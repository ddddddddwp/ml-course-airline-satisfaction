"""P3 误差分析：高置信 FP/FN top-50 + 分群错误率。

定义（写报告时必须原样交代）：
- FP：y=0 且 pred >= 0.8，按 pred 从高到低取前 50（模型最“自信”地把不满意判成满意）；
- FN：y=1 且 pred <= 0.2，按 pred 从低到高取前 50（模型最“自信”地漏掉满意乘客）；
- 分群错误率：0.5 阈值下 (pred>=0.5) != y 的比例，只作描述；主指标仍是 AUC。

对齐方式：
OOF 的 row 就是 train 的行号，因此用 `train.iloc[oof.row]` **按位置对齐**。
绝不按 id merge——OOF 一旦被排序或 id 有重复，merge 会静默错位，整份误差分析作废。

用法：
    $env:PYTHONPATH="."
    python -m src.eval.badcase --config configs/base.yaml
    python -m src.eval.badcase --config configs/base.yaml --top 50 --fp-cut 0.8 --fn-cut 0.2
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.config import ROOT, load_config
from src.eval._io import (
    SEGMENT_COLUMNS,
    SEGMENT_GROUP_ORDER,
    align_train_features,
    build_segment_frame,
    load_oof,
    load_train,
    resolve_oof_path,
)

plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)

MIN_GROUP_N = 50
OUT_COLUMNS = [
    "id", "row", "y", "pred", "error_type",
    "Class", "Type of Travel", "Online boarding",
    "Departure Delay", "Flight Distance", "Age",
]
_DISPLAY_MAP = {
    "Class": "Class",
    "Type of Travel": "Type of Travel",
    "Online boarding": "Online boarding",
    "Departure Delay": "Departure Delay in Minutes",
    "Flight Distance": "Flight Distance",
    "Age": "Age",
}


def _resolve(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def _fmt(v, nd: int = 4) -> str:
    if v is None:
        return "NA"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return "NA"
    return f"{f:.{nd}f}"


def load_badcases(
    oof: pd.DataFrame,
    feats: pd.DataFrame,
    *,
    fp_cut: float = 0.8,
    fn_cut: float = 0.2,
    top: int = 50,
    id_col: str = "id",
) -> pd.DataFrame:
    """从 OOF + 对齐后的 train 行里挑高置信 FP/FN，各取 top N。"""
    if not (0.0 <= float(fn_cut) < float(fp_cut) <= 1.0):
        raise ValueError(
            f"[badcase] 需要 0 <= fn_cut < fp_cut <= 1，收到 fn_cut={fn_cut}, fp_cut={fp_cut}"
        )
    if int(top) < 1:
        raise ValueError(f"[badcase] --top 必须 >= 1，收到 {top}")
    if len(oof) != len(feats):
        raise ValueError(f"[badcase] OOF 行数 {len(oof)} 与特征行数 {len(feats)} 不一致")
    for c in ("row", "y", "pred"):
        if c not in oof.columns:
            raise ValueError(f"[badcase] OOF 缺少列 {c}")

    missing = [out for out, src in _DISPLAY_MAP.items() if src not in feats.columns]
    if missing:
        raise ValueError(
            f"[badcase] train 缺少展示列 {missing}；需要 {list(_DISPLAY_MAP.values())}。\n"
            "  请确认传入的是 train 原始列（不是 prepare_X 输出）。"
        )

    base = pd.DataFrame(
        {
            "row": oof["row"].to_numpy(),
            "y": oof["y"].to_numpy(),
            "pred": oof["pred"].to_numpy(),
        }
    )
    if id_col and id_col in feats.columns:
        base.insert(0, "id", feats[id_col].to_numpy())
    else:
        print(f"[badcase][警告] train 中没有 id 列 {id_col!r}，输出用 row 代替 id。")
        base.insert(0, "id", base["row"].to_numpy())

    for out, src in _DISPLAY_MAP.items():
        base[out] = feats[src].to_numpy()

    def _take(mask: pd.Series, ascending: bool, label: str) -> pd.DataFrame:
        sub = base.loc[mask].sort_values(
            ["pred", "row"], ascending=[ascending, True], kind="mergesort"
        ).head(int(top)).copy()
        sub["error_type"] = label
        return sub

    fp = (base["y"] == 0) & (base["pred"] >= float(fp_cut))
    fn = (base["y"] == 1) & (base["pred"] <= float(fn_cut))
    picked = pd.concat(
        [
            _take(fp, ascending=False, label="FP"),
            _take(fn, ascending=True, label="FN"),
        ],
        ignore_index=True,
    )
    if picked.empty:
        print("[badcase][警告] 没有满足条件的高置信 FP/FN；输出空表（只有表头）。")
        return pd.DataFrame(columns=OUT_COLUMNS)
    return picked[OUT_COLUMNS].reset_index(drop=True)


def segment_error_table(
    y_true,
    y_prob,
    features: pd.DataFrame,
    *,
    threshold: float = 0.5,
    min_n: int = MIN_GROUP_N,
) -> pd.DataFrame:
    """六个分群的错误率表：n / 错误数 / FP / FN / 错误率 / 平均预测 / low_n。"""
    y = np.asarray(y_true, dtype="float64").reshape(-1)
    p = np.asarray(y_prob, dtype="float64").reshape(-1)
    if y.size != p.size:
        raise ValueError(f"[badcase] y 长度 {y.size} 与 pred 长度 {p.size} 不一致")
    if len(features) != y.size:
        raise ValueError(
            f"[badcase] 特征行数 {len(features)} 与 y 长度 {y.size} 不一致；请先按 row 对齐"
        )

    seg = build_segment_frame(features)
    pred_bin = (p >= float(threshold)).astype("int8")
    yy = y.astype("int8")
    rows: list[dict] = []
    for col in SEGMENT_COLUMNS:
        for group, sub in seg.groupby(col, sort=False, dropna=False):
            idx = sub.index.to_numpy()
            yg, pg = yy[idx], pred_bin[idx]
            n = int(idx.size)
            err = yg != pg
            rows.append(
                {
                    "segment": col,
                    "group": str(group),
                    "n": n,
                    "n_error": int(err.sum()),
                    "error_rate": float(err.mean()) if n else float("nan"),
                    "fp": int(((yg == 0) & (pg == 1)).sum()),
                    "fn": int(((yg == 1) & (pg == 0)).sum()),
                    "pred_mean": float(p[idx].mean()) if n else float("nan"),
                    "pos_rate": float(yg.mean()) if n else float("nan"),
                    "low_n": bool(n < int(min_n)),
                }
            )

    table = pd.DataFrame(rows)
    seg_order = {name: i for i, name in enumerate(SEGMENT_COLUMNS)}
    table["_seg_order"] = table["segment"].map(seg_order).fillna(999).astype(int)

    def _group_order(row: pd.Series) -> int:
        order = SEGMENT_GROUP_ORDER.get(row["segment"])
        if order is None:
            return 999
        try:
            return order.index(row["group"])
        except ValueError:
            return 999

    table["_group_order"] = table.apply(_group_order, axis=1)
    table = table.sort_values(
        ["_seg_order", "_group_order", "n"],
        ascending=[True, True, False],
        kind="mergesort",
    ).drop(columns=["_seg_order", "_group_order"])
    return table.reset_index(drop=True)


def _plot_segment_errors(table: pd.DataFrame, path: Path, threshold: float) -> None:
    """2×3 面板：每个分群一条错误率横条；low_n 用灰色标出。"""
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.4))
    axes = axes.reshape(-1)
    for ax, seg_name in zip(axes, SEGMENT_COLUMNS):
        d = table[table["segment"] == seg_name].sort_values(
            "error_rate", ascending=True, kind="mergesort"
        )
        if d.empty:
            ax.set_visible(False)
            continue
        colors = ["#a0aec0" if flag else "#2b6cb0" for flag in d["low_n"]]
        positions = np.arange(len(d))
        ax.barh(positions, d["error_rate"].to_numpy(), color=colors, height=0.72)
        ax.set_yticks(positions, [str(g) for g in d["group"]], fontsize=7)
        top = float(d["error_rate"].max()) if len(d) else 0.0
        for pos, (_, r) in zip(positions, d.iterrows()):
            ax.text(r["error_rate"] + max(top, 0.01) * 0.02, pos,
                    f"{r['error_rate']:.3f} (n={int(r['n']):,})", va="center", fontsize=6)
        ax.set_xlim(0, max(top * 1.35, 0.02))
        ax.set_title(seg_name, fontsize=9)
        ax.set_xlabel(f"error rate @ {threshold:g}", fontsize=7)
    fig.suptitle(
        "P3 badcase: error rate by segment (gray = n<50, low_n; descriptive only, primary metric is AUC)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_summary(
    out: Path,
    bad: pd.DataFrame,
    table: pd.DataFrame,
    oof_path: Path,
    y: np.ndarray,
    p: np.ndarray,
    fp_cut: float,
    fn_cut: float,
    threshold: float,
    top: int,
    n_train: int,
    allow_partial: bool,
) -> None:
    fp_all = int(((y == 0) & (p >= float(fp_cut))).sum())
    fn_all = int(((y == 1) & (p <= float(fn_cut))).sum())
    pred_bin = (p >= float(threshold)).astype("int8")
    err_all = int((pred_bin != y.astype("int8")).sum())
    low_n = int(table["low_n"].sum())

    lines = [
        "# P3 误差分析（badcase）",
        "",
        f"> 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}  ",
        f"> OOF：`{oof_path}`  ｜  样本：{len(y):,} / train {n_train:,}  ",
        f"> 高置信 FP：y=0 且 pred>={fp_cut:g}；高置信 FN：y=1 且 pred<={fn_cut:g}；各取 top {top}。",
        "",
        "> 这里的“错误”是 **0.5 阈值** 下的分类错误，只用于定位问题样本；",
        "> 模型选择与提交仍以 ROC AUC / 概率为准，不能阈值化。",
        "",
    ]
    if allow_partial and len(y) < n_train:
        lines += ["> 注意：本次 OOF 只覆盖训练集子集（合成/冒烟），数字不可写进报告。", ""]

    lines += [
        "## 1. 总体",
        "",
        "| 项 | 值 |",
        "|---|---:|",
        f"| 0.5 阈值错误数 / 错误率 | {err_all:,} / {err_all / max(len(y), 1):.2%} |",
        f"| 全量高置信 FP（pred>={fp_cut:g}） | {fp_all:,} |",
        f"| 全量高置信 FN（pred<={fn_cut:g}） | {fn_all:,} |",
        f"| 本次导出 badcase 数 | {len(bad):,}（FP {int((bad['error_type'] == 'FP').sum()) if len(bad) else 0} / "
        f"FN {int((bad['error_type'] == 'FN').sum()) if len(bad) else 0}） |",
        "",
        "## 2. 高置信错误构成",
        "",
    ]
    if len(bad):
        grp = bad.groupby("error_type", sort=False).agg(
            n=("pred", "size"), pred_mean=("pred", "mean")
        ).reset_index()
        lines += ["| error_type | 条数 | 平均预测 |", "|---|---:|---:|"]
        for _, r in grp.iterrows():
            lines.append(f"| {r['error_type']} | {int(r['n'])} | {_fmt(r['pred_mean'])} |")
    else:
        lines.append("没有满足条件的高置信 FP/FN（检查 fp_cut / fn_cut 是否过严）。")
    lines += [
        "",
        "样例明细见 `badcase_top50.csv`（按置信度排序，不是随机抽样）。",
        "",
        "## 3. 分群错误率",
        "",
        f"完整表：`{len(table)}` 个群，其中 {low_n} 个群 n < {MIN_GROUP_N}（`low_n=True`，不解读）。",
        "",
        "| 分群 | 组 | n | 错误数 | 错误率 | FP | FN | 平均预测 | low_n |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['segment']} | {r['group']} | {int(r['n']):,} | {int(r['n_error']):,} | "
            f"{_fmt(r['error_rate'])} | {int(r['fp']):,} | {int(r['fn']):,} | "
            f"{_fmt(r['pred_mean'])} | {'是' if r['low_n'] else ''} |"
        )
    lines += [
        "",
        "## 4. 读法",
        "",
        "- 高置信错误样本最值得回看原始特征：它们通常是“评分组合与标签矛盾”的边界样本。",
        "- 分群错误率差异可能来自正例率差异（某些舱位/距离段的满意率先验就低），不要直接当模型缺陷。",
        "- `low_n=True` 的群样本不足，错误率一根样本就能跳很多，不写结论。",
        "- 本表是误差定位，不是模型选择依据；模型比较看 AUC 与多 seed 稳定性（stability.md）。",
        "",
    ]
    (out / "badcase_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_badcase(
    cfg: dict,
    *,
    oof_path: str | Path | None = None,
    out_dir: str | Path = "reports/eval",
    top: int = 50,
    fp_cut: float = 0.8,
    fn_cut: float = 0.2,
    threshold: float = 0.5,
    allow_partial: bool = False,
) -> dict:
    """读 OOF + train -> 高置信 badcase CSV + 分群错误率表 + 图。"""
    oof_file = resolve_oof_path(cfg, oof_path)
    train = load_train(cfg)
    oof = load_oof(
        oof_file, n_rows=len(train), allow_partial=allow_partial, context="badcase"
    )
    feats = align_train_features(train, oof)
    y = oof["y"].to_numpy(dtype="float64")
    p = oof["pred"].to_numpy(dtype="float64")

    bad = load_badcases(
        oof, feats, fp_cut=float(fp_cut), fn_cut=float(fn_cut), top=int(top),
        id_col=str(cfg.get("data", {}).get("id_col", "id")),
    )
    table = segment_error_table(y, p, feats, threshold=float(threshold))

    out = _resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "badcase_top50.csv"
    bad.to_csv(csv_path, index=False, encoding="utf-8-sig")
    _plot_segment_errors(table, out / "p3_badcase.png", float(threshold))
    _write_summary(
        out, bad, table, oof_file, y, p,
        fp_cut=float(fp_cut), fn_cut=float(fn_cut),
        threshold=float(threshold), top=int(top),
        n_train=len(train), allow_partial=allow_partial,
    )

    fp_n = int((bad["error_type"] == "FP").sum()) if len(bad) else 0
    fn_n = int((bad["error_type"] == "FN").sum()) if len(bad) else 0
    print(f"[badcase] OOF: {oof_file}")
    print(f"[badcase] 导出 {len(bad):,} 条（FP {fp_n} / FN {fn_n}），top={top}")
    print(f"[badcase] CSV  -> {csv_path}")
    print(f"[badcase] 图   -> {out / 'p3_badcase.png'}")
    print(f"[badcase] 汇总 -> {out / 'badcase_summary.md'}")
    if table["low_n"].any():
        print(
            f"[badcase][提示] {int(table['low_n'].sum())} 个分群 n<{MIN_GROUP_N}，"
            "图中标灰，报告里不要单独解读。"
        )
    return {"csv": str(csv_path), "out": str(out), "bad": bad, "segments": table}


def main() -> None:
    ap = argparse.ArgumentParser(description="P3 误差分析：高置信 FP/FN + 分群错误率")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--oof", default=None)
    ap.add_argument("--out", default="reports/eval")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--fp-cut", type=float, default=0.8)
    ap.add_argument("--fn-cut", type=float, default=0.2)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    try:
        run_badcase(
            cfg,
            oof_path=args.oof,
            out_dir=args.out,
            top=args.top,
            fp_cut=args.fp_cut,
            fn_cut=args.fn_cut,
            threshold=args.threshold,
            allow_partial=args.allow_partial,
        )
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"[badcase] {e}") from None


if __name__ == "__main__":
    main()
