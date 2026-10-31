"""P3 主评估：OOF -> 指标 / 可靠性 / 分群 / 图 / summary.md。

用法：
    $env:PYTHONPATH="."
    python -m src.eval.evaluate --config configs/base.yaml
    python -m src.eval.evaluate --config configs/base.yaml \
        --oof reports/oof/P2_final_oof.parquet --out reports/eval --threshold 0.5

默认行为：
- OOF 先找 reports/oof/P2_final_oof.parquet，找不到就取 reports/oof 下最新的 run OOF；
- 按 OOF 的 row 列与 data/processed/train.parquet **位置对齐**（不做 id merge）；
- 输出 4 个数据文件 + 3 张 300dpi 图。

报告口径：AUC 是主指标；0.5 阈值下的 accuracy / precision / recall / F1 / 混淆矩阵
只作描述，不能用于选模型，更不能把提交概率阈值化。
"""

from __future__ import annotations

import argparse
import math
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import roc_auc_score

from src.common.config import ROOT, git_commit, load_config
from src.eval._io import (
    SEGMENT_COLUMNS,
    SEGMENT_GROUP_ORDER,
    align_train_features,
    build_segment_frame,
    load_oof,
    load_train,
    resolve_oof_path,
    write_json,
)
from src.eval.metrics import binary_metrics, bootstrap_auc_ci, reliability_curve

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
_EPS = 1e-15


def _resolve_out(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def _fmt(v, nd: int = 4) -> str:
    """报告里的数值格式化：NaN/None 统一显示 NA。"""
    if v is None:
        return "NA"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return "NA"
    return f"{f:.{nd}f}"


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    """分群 AUC：单类群返回 NaN，不逐群刷警告。"""
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        return float(roc_auc_score(y, p))


def _safe_logloss(y: np.ndarray, p: np.ndarray) -> float:
    if y.size == 0:
        return float("nan")
    q = np.clip(p, _EPS, 1.0 - _EPS)
    return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))


def segment_metrics(
    y_true,
    y_prob,
    features: pd.DataFrame,
    threshold: float = 0.5,
    min_n: int = MIN_GROUP_N,
) -> pd.DataFrame:
    """六个维度的分群指标：n / 正例率 / AUC / 平均预测 / logloss / 错误率 / low_n。

    features 必须是与 y_true 同长度、同顺序的 **train 原始列**（不是 prepare_X 输出）。
    n < min_n 的群标记 low_n=True；分群 AUC 波动大，低样本群不要写结论。
    """
    y = np.asarray(y_true, dtype="float64").reshape(-1)
    p = np.asarray(y_prob, dtype="float64").reshape(-1)
    if y.size != p.size:
        raise ValueError(f"[segment] y 长度 {y.size} 与 pred 长度 {p.size} 不一致")
    if len(features) != y.size:
        raise ValueError(
            f"[segment] 特征行数 {len(features)} 与 y 长度 {y.size} 不一致；请先按 row 对齐"
        )
    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError(f"[segment] threshold 必须在 [0,1]，收到 {threshold}")

    seg = build_segment_frame(features)
    rows: list[dict] = []
    for col in SEGMENT_COLUMNS:
        for group, sub in seg.groupby(col, sort=False, dropna=False):
            idx = sub.index.to_numpy()
            yy, pp = y[idx], p[idx]
            n = int(idx.size)
            pred_bin = (pp >= float(threshold)).astype("int8")
            rows.append(
                {
                    "segment": col,
                    "group": str(group),
                    "n": n,
                    "pos_rate": float(yy.mean()) if n else float("nan"),
                    "auc": _safe_auc(yy, pp),
                    "pred_mean": float(pp.mean()) if n else float("nan"),
                    "logloss": _safe_logloss(yy, pp),
                    "error_rate": float((pred_bin != yy.astype("int8")).mean()) if n else float("nan"),
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


def _plot_calibration(rel: pd.DataFrame, metrics: dict, path: Path) -> None:
    """可靠性曲线 + 对角线。点越贴近对角线，概率越可信。"""
    d = rel[rel["n"] > 0]
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="#718096", label="perfect calibration")
    if len(d):
        ax.plot(d["pred_mean"], d["y_mean"], "-o", ms=4, lw=1.2, color="#2b6cb0",
                label="OOF (equal-width bins)")
        for _, r in d.iterrows():
            if r["n"] >= 500:
                ax.annotate(f"{int(r['n'])}", (r["pred_mean"], r["y_mean"]),
                            textcoords="offset points", xytext=(4, -9), fontsize=6,
                            color="#4a5568")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title(
        f"P3 calibration (OOF, n={metrics['n']:,}, ECE={metrics['ece']:.4f})"
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_score_dist(y: np.ndarray, p: np.ndarray, path: Path, threshold: float) -> None:
    """按真实标签分组的预测分布（组内占比，避免正负例不均衡造成误读）。"""
    bins = np.linspace(0.0, 1.0, 51)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label, color, name in ((0, "#718096", "y=0"), (1, "#2b6cb0", "y=1")):
        vals = p[y == label]
        if vals.size:
            ax.hist(
                vals,
                bins=bins,
                weights=np.full(vals.size, 1.0 / vals.size),
                alpha=0.55,
                color=color,
                label=f"{name} (n={vals.size:,}, mean={vals.mean():.3f})",
            )
    ax.axvline(threshold, color="#c53030", ls="--", lw=1.0,
               label=f"threshold={threshold:g} (descriptive only)")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Share within true class")
    ax.set_title("P3 OOF score distribution by true label")
    ax.legend(loc="upper center", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion(metrics: dict, path: Path) -> None:
    """0.5 阈值混淆矩阵。标题与脚注都明确：这只是一张描述图，主指标是 AUC。"""
    cm = np.array(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], dtype="float64"
    )
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.imshow(cm, cmap="Blues", vmin=0)
    ax.grid(False)
    total = max(cm.sum(), 1.0)
    for i in range(2):
        for j in range(2):
            count = int(cm[i, j])
            share = count / total
            ax.text(
                j, i, f"{count:,}\n({share:.1%})",
                ha="center", va="center", fontsize=10,
                color="white" if cm[i, j] > cm.max() * 0.55 else "#1a202c",
            )
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["True 0", "True 1"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(
        f"Confusion matrix @ {metrics['threshold']:g} (descriptive only)\n"
        f"Primary metric: ROC AUC = {metrics['auc']:.6f}"
    )
    fig.text(
        0.5, -0.02,
        "Do not threshold submission probabilities: the competition scores probabilities by AUC.",
        ha="center", fontsize=7, color="#4a5568",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_summary(
    out: Path,
    metrics: dict,
    ci: tuple[float, float, float],
    rel: pd.DataFrame,
    seg: pd.DataFrame,
    oof_path: Path,
    cfg: dict,
    n_train: int,
    n_boot: int,
    seed: int,
    n_bins: int,
    allow_partial: bool,
) -> None:
    """生成人读的 summary.md；数字口径与 metrics.json / segment_metrics.csv 一致。"""
    low = int(seg["low_n"].sum()) if len(seg) else 0
    worst = (
        seg[(~seg["low_n"]) & seg["auc"].notna()]
        .sort_values("auc", kind="mergesort")
        .head(5)
    )
    nonempty = rel[rel["n"] > 0]
    max_gap_row = nonempty.loc[nonempty["abs_gap"].idxmax()] if len(nonempty) else None

    lines = [
        "# P3 评估摘要（OOF）",
        "",
        f"> 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}  ",
        f"> 配置：`{cfg.get('_path', 'N/A')}`  ｜  git：`{git_commit() or 'unknown'}`  ",
        f"> OOF：`{oof_path}`",
        "",
        "> **主指标是 ROC AUC。** accuracy / precision / recall / F1 / 混淆矩阵都依赖 0.5 阈值，",
        "> 只用于描述模型在该阈值下的行为，不能用于选模型；提交必须给概率，不能阈值化。",
        "",
        "## 1. 数据与主指标",
        "",
        "| 项 | 值 |",
        "|---|---:|",
        f"| 评估样本数 | {metrics['n']:,} |",
        f"| 正例数 / 正例率 | {metrics['positives']:,} / {metrics['positives'] / max(metrics['n'], 1):.2%} |",
        f"| **ROC AUC** | **{_fmt(metrics['auc'], 6)}** |",
        f"| AUC bootstrap 95% CI | [{_fmt(ci[0], 6)}, {_fmt(ci[1], 6)}]（n_boot={n_boot}, seed={seed}） |",
        f"| LogLoss | {_fmt(metrics['logloss'], 6)} |",
        f"| Brier | {_fmt(metrics['brier'], 6)} |",
        f"| ECE（{n_bins} 箱） | {_fmt(metrics['ece'], 6)} |",
        f"| 阈值 | {metrics['threshold']:g} |",
        f"| Accuracy @ 阈值 | {_fmt(metrics['accuracy'])} |",
        f"| Precision @ 阈值 | {_fmt(metrics['precision'])} |",
        f"| Recall @ 阈值 | {_fmt(metrics['recall'])} |",
        f"| F1 @ 阈值 | {_fmt(metrics['f1'])} |",
        f"| TN / FP / FN / TP | {metrics['tn']:,} / {metrics['fp']:,} / {metrics['fn']:,} / {metrics['tp']:,} |",
        "",
        "> 口径说明：本页 AUC 是把整份 OOF 一次算出来的（pooled）；`reports/exp_log.csv` 的 `cv_mean`",
        "> 是 5 折 AUC 的算术平均，两者通常相差 1e-5 量级，引用时必须写清是哪一种。",
        "",
    ]
    if allow_partial and metrics["n"] < n_train:
        lines += ["> 注意：本次 OOF 只覆盖训练集的一个子集（合成/冒烟数据），数字不可写进报告。", ""]

    lines += [
        "## 2. 概率校准",
        "",
        f"可靠性曲线见 `p3_calibration.png` 与 `reliability.csv`（{n_bins} 个等宽箱）。",
        f"ECE = **{_fmt(metrics['ece'], 6)}**：越小说明预测概率越接近真实频率。",
    ]
    if max_gap_row is not None:
        lines.append(
            f"最大偏差落在 [{max_gap_row['bin_lower']:.1f}, {max_gap_row['bin_upper']:.1f})："
            f"平均预测 {_fmt(max_gap_row['pred_mean'])}，实际正例率 {_fmt(max_gap_row['y_mean'])}，"
            f"缺口 {_fmt(max_gap_row['abs_gap'])}（n={int(max_gap_row['n']):,}）。"
        )
    lines += [
        "",
        "## 3. 分群表现",
        "",
        f"完整表见 `segment_metrics.csv`：{len(seg)} 个群，其中 {low} 个群 n < {MIN_GROUP_N}（`low_n=True`）。",
        "",
        "AUC 最低的 5 个群（只统计 n ≥ 50，避免小样本噪声）：",
        "",
        "| 分群 | 组 | n | 正例率 | AUC | 平均预测 | 错误率 @0.5 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    if len(worst):
        for _, r in worst.iterrows():
            lines.append(
                f"| {r['segment']} | {r['group']} | {int(r['n']):,} | {_fmt(r['pos_rate'])} | "
                f"{_fmt(r['auc'])} | {_fmt(r['pred_mean'])} | {_fmt(r['error_rate'])} |"
            )
    else:
        lines.append("| - | - | - | - | - | - | - |")
    lines += [
        "",
        "## 4. 读法与局限",
        "",
        "- AUC 是排序指标，对阈值与正例率不敏感；阈值指标只说明 0.5 下的取舍。",
        "- 分群 AUC 的样本量与正例率都更小，波动比总体大；`low_n=True` 的群不要下结论。",
        "- OOF 由冻结的 `splits/fold_v1.json` 产出，同一份 OOF 内部分群可比；换划分后不可直接横比。",
        "- 本页只描述“模型在留出折上表现如何”，不回答因果；特征为什么重要请看 `shap_importance.csv`。",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_evaluation(
    cfg: dict,
    *,
    oof_path: str | Path | None = None,
    out_dir: str | Path = "reports/eval",
    threshold: float = 0.5,
    n_bins: int = 10,
    n_boot: int = 200,
    seed: int = 42,
    allow_partial: bool = False,
) -> dict:
    """执行完整评估并返回摘要 dict（供测试 / 上层脚本调用）。"""
    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError(f"[evaluate] --threshold 必须在 [0,1]，收到 {threshold}")
    if int(n_bins) < 1:
        raise ValueError(f"[evaluate] --n-bins 必须 >= 1，收到 {n_bins}")

    oof_file = resolve_oof_path(cfg, oof_path)
    train = load_train(cfg)
    oof = load_oof(
        oof_file,
        n_rows=len(train),
        allow_partial=allow_partial,
        context="evaluate",
    )
    feats = align_train_features(train, oof)
    y = oof["y"].to_numpy(dtype="float64")
    p = oof["pred"].to_numpy(dtype="float64")

    print(f"[evaluate] OOF   : {oof_file}")
    print(f"[evaluate] 样本  : {len(oof):,} / train {len(train):,}  |  正例率 {y.mean():.4%}")
    print("[evaluate] 计算指标与 bootstrap 置信区间 ...")

    metrics = binary_metrics(y, p, threshold=threshold, n_bins=int(n_bins))
    ci = bootstrap_auc_ci(y, p, n_boot=int(n_boot), seed=int(seed))
    rel = reliability_curve(y, p, n_bins=int(n_bins))
    seg = segment_metrics(y, p, feats, threshold=threshold, min_n=MIN_GROUP_N)
    print(
        f"[evaluate] 指标  : AUC={_fmt(metrics['auc'], 6)} "
        f"CI=[{_fmt(ci[0], 6)}, {_fmt(ci[1], 6)}]  LogLoss={_fmt(metrics['logloss'], 6)}"
    )

    out = _resolve_out(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rel.to_csv(out / "reliability.csv", index=False, encoding="utf-8-sig")
    seg.to_csv(out / "segment_metrics.csv", index=False, encoding="utf-8-sig")

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": cfg.get("_path", ""),
        "oof": str(oof_file),
        "n_oof": len(oof),
        "n_train": len(train),
        "partial": bool(len(oof) != len(train)),
        "positive_rate": metrics["positives"] / max(metrics["n"], 1),
        **metrics,
        "auc_ci": {
            "low": ci[0],
            "high": ci[1],
            "mean": ci[2],
            "n_boot": int(n_boot),
            "seed": int(seed),
            "level": 0.95,
        },
        "ece_bins": int(n_bins),
        "min_group_n": MIN_GROUP_N,
        "segment_summary": {
            "n_groups": len(seg),
            "n_low_n": int(seg["low_n"].sum()),
        },
        "git_commit": git_commit(),
    }
    write_json(out / "metrics.json", payload)

    _plot_calibration(rel, metrics, out / "p3_calibration.png")
    _plot_score_dist(y, p, out / "p3_score_dist.png", threshold)
    _plot_confusion(metrics, out / "p3_confusion.png")
    _write_summary(
        out, metrics, ci, rel, seg, oof_file, cfg,
        n_train=len(train), n_boot=int(n_boot), seed=int(seed), n_bins=int(n_bins),
        allow_partial=allow_partial,
    )

    print(f"[evaluate] 输出  -> {out}")
    for name in (
        "metrics.json", "reliability.csv", "segment_metrics.csv", "summary.md",
        "p3_calibration.png", "p3_score_dist.png", "p3_confusion.png",
    ):
        print(f"           - {name}")
    low_n = int(seg["low_n"].sum())
    if low_n:
        print(f"[evaluate][提示] {low_n} 个分群 n<{MIN_GROUP_N}（low_n=True），报告中不要单独解读。")
    return {
        "oof": str(oof_file),
        "out": str(out),
        "metrics": metrics,
        "auc_ci": {"low": ci[0], "high": ci[1], "mean": ci[2]},
        "segments": seg,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="P3 主评估：OOF -> metrics/reliability/分群/图")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--oof", default=None,
                    help="OOF parquet；默认 reports/oof/P2_final_oof.parquet，找不到则取最新 run")
    ap.add_argument("--out", default="reports/eval", help="输出目录（默认 reports/eval）")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="阈值指标/混淆矩阵用；仅作描述，不影响 AUC")
    ap.add_argument("--n-bins", type=int, default=10, help="可靠性曲线分箱数")
    ap.add_argument("--n-boot", type=int, default=200,
                    help="AUC bootstrap 次数（70 万行每 100 次约 20 秒）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-partial", action="store_true",
                    help="允许 OOF 只覆盖部分 train 行（仅合成数据冒烟用）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    try:
        run_evaluation(
            cfg,
            oof_path=args.oof,
            out_dir=args.out,
            threshold=args.threshold,
            n_bins=args.n_bins,
            n_boot=args.n_boot,
            seed=args.seed,
            allow_partial=args.allow_partial,
        )
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"[evaluate] {e}") from None


if __name__ == "__main__":
    main()
