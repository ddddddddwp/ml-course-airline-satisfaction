"""EDA 四张图（P1-03 / P1-21）。

产出 reports/figures/：
  fig1_label_dist.png      目标分布
  fig2_data_quality.png    缺失率 / 每行 0 值个数 / 各评分列 0 值占比
  fig3_delay_dist.png      延误长尾（原始 vs log1p）
  fig4_rating_vs_target.png 评分-满意率曲线（0 = 不适用 的核心证据）

用法：
    python scripts/eda.py                 # 用 configs/base.yaml
    python scripts/eda.py --config configs/toy.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.config import ROOT, load_config, path_of
from src.data.schema import detect_schema

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

DEP = "Departure Delay in Minutes"
ARR = "Arrival Delay in Minutes"


def _save(fig, out: Path, name: str) -> Path:
    p = out / name
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p}  ({p.stat().st_size / 1024:.0f} KB)")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = pd.read_parquet(path_of(cfg, "processed_dir") / "train.parquet")
    schema = detect_schema(df, cfg, verbose=False)
    y = df[schema["target"]].to_numpy()
    svc = [c for c in schema["service_cols"] if c in df.columns]

    fig_dir = ROOT / cfg["paths"]["figures"] if "figures" in cfg["paths"] else ROOT / "reports/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eda] {len(df):,} 行 × {df.shape[1]} 列 -> {fig_dir}")

    # ---------- fig1 标签分布
    vc = pd.Series(y).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.6))
    bars = ax.bar(["0 (neutral or dissatisfied)", "1 (satisfied)"],
                  [vc.get(0, 0), vc.get(1, 0)],
                  color=["#8c8c8c", "#2b6cb0"], width=0.55)
    for b, v in zip(bars, [vc.get(0, 0), vc.get(1, 0)]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n{v / len(y):.2%}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("rows")
    ax.set_title("Target distribution (fairly balanced, no resampling needed)")
    ax.set_ylim(0, vc.max() * 1.18)
    _save(fig, fig_dir, "fig1_label_dist.png")

    # ---------- fig2 数据质量
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    raw_svc = df[svc].to_numpy(dtype="float64")
    n_zeros = (raw_svc == 0).sum(axis=1)
    zero_rate = pd.Series((raw_svc == 0).mean(axis=0), index=svc).sort_values(ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axes[0]
    if len(miss):
        ax.barh([c.replace(" in Minutes", "") for c in miss.index],
                miss.values * 100, color="#c53030")
        for i, v in enumerate(miss.values * 100):
            ax.text(v, i, f" {v:.3f}%", va="center", fontsize=9)
        ax.set_xlim(0, max(miss.values * 100) * 1.6)
    else:
        ax.text(0.5, 0.5, "no missing values", ha="center", va="center")
    ax.set_xlabel("missing rate (%)")
    ax.set_title("Missing values: only one column")

    ax = axes[1]
    ax.hist(n_zeros, bins=range(0, len(svc) + 2), color="#2b6cb0",
            align="left", rwidth=0.85)
    ax.set_xlabel("# of ratings equal to 0 in a row")
    ax.set_ylabel("rows")
    ax.set_title(f"Zero ratings per row (N/A answers)")

    ax = axes[2]
    ax.barh([c.replace(" service", "") for c in zero_rate.index[-8:]],
            zero_rate.values[-8:] * 100, color="#dd6b20")
    ax.set_xlabel("share of 0 (%)")
    ax.set_title("Zero share by rating column")
    _save(fig, fig_dir, "fig2_data_quality.png")

    # ---------- fig3 延误长尾
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, name in ((axes[0], DEP, "Departure Delay in Minutes"),
                          (axes[1], ARR, "Arrival Delay in Minutes")):
        s = df[col].dropna()
        ax.hist(s.clip(upper=s.quantile(0.995)), bins=60, color="#2b6cb0")
        ax.set_yscale("log")
        ax.set_xlabel(name)
        ax.set_ylabel("rows (log)")
        ax.set_title(f"{(s == 0).mean():.1%} are zero, max {s.max():.0f} min")
    _save(fig, fig_dir, "fig3_delay_dist.png")

    # ---------- fig4 评分-满意率（0 = 不适用）
    order = []
    for c in svc:
        r = pd.Series(df[c]).corr(pd.Series(y))
        order.append((c, abs(r) if pd.notna(r) else 0))
    order = [c for c, _ in sorted(order, key=lambda t: -t[1])[:6]]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, c in zip(axes.ravel(), order):
        g = df.groupby(c)[schema["target"]].agg(["mean", "size"])
        ax.plot(g.index, g["mean"], "o-", color="#2b6cb0", markersize=5)
        ax.set_xticks(sorted(df[c].dropna().unique()))
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"{c} rating")
        ax.set_ylabel("satisfied rate")
        ax.set_title(c, fontsize=9)
        if 0 in g.index:
            ax.annotate(f"0 = N/A\n{g.loc[0, 'mean']:.0%} satisfied",
                        xy=(0, g.loc[0, "mean"]), xytext=(0.35, 0.12),
                        textcoords="axes fraction", fontsize=8,
                        arrowprops=dict(arrowstyle="->", color="#c53030"),
                        color="#c53030")
    fig.suptitle("Rating vs satisfaction: non-monotonic, 0 means 'not applicable'", y=1.01)
    _save(fig, fig_dir, "fig4_rating_vs_target.png")

    # ---------- 给报告用的数字
    print("\n[eda] 可直接写进报告的数字：")
    print(f"  正例率            : {y.mean():.2%} ({int(y.sum()):,} / {len(y):,})")
    print(f"  完全重复行        : {int(df.duplicated().sum()):,}")
    if len(miss):
        print(f"  有缺失的列        : {len(miss)} 列，最高 {miss.iloc[0]:.3%} ({miss.index[0]})")
    print(f"  0 值评分占比      : {(raw_svc == 0).mean():.2%}（全表）")
    print(f"  行内含 0 值的中位数: {int(np.median(n_zeros))}")
    print(f"  出发延误 = 0 的比例: {(df[DEP] == 0).mean():.2%}")
    for c in order[:3]:
        g = df.groupby(c)[schema["target"]].mean()
        if 0 in g.index and 1 in g.index:
            print(f"  {c}: 评0 满意率 {g.loc[0]:.1%} vs 评1 {g.loc[1]:.1%}")


if __name__ == "__main__":
    main()
