"""特征重要性 top-20（P1-11）。

用 LGBM 在**全量训练集**上拟合一次取 gain 重要性。注意：
这只能说明"模型怎么用特征"，不能替代消融——重要性高 ≠ 去掉会掉分。

产出（路径取自 --config 的 paths.figures / paths.reports）：
    reports/figures/fig_feat_importance.png
    reports/feature_importance.csv

用法：
    python scripts/plot_importance.py --config configs/base.yaml --groups base,C,T --top 20
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.config import load_config, path_of
from src.data.schema import detect_schema
from src.features.build import prepare_X
from src.features.encode import TargetEncoder
from src.models.train import _make_lgbm

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 9})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--groups", default="base,C,T")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    if args.top < 1:
        raise SystemExit("[imp] --top 必须是 >= 1 的整数")
    cfg = load_config(args.config)
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    proc = path_of(cfg, "processed_dir")
    tr_path = proc / "train.parquet"
    if not tr_path.exists():
        raise SystemExit(f"[imp] 找不到 {tr_path}，先跑 make data")
    df = pd.read_parquet(tr_path)
    test = pd.read_parquet(proc / "test.parquet")
    schema = detect_schema(df, cfg, verbose=False)
    y = df[schema["target"]].to_numpy()

    count_ref = None
    if "C" in set(groups):
        cols = [c for c in ("Flight Distance", "Age") if c in test.columns]
        if cols:
            count_ref = pd.concat([df[cols], test[cols]], ignore_index=True)

    X = prepare_X(df, schema, groups, count_ref=count_ref)

    # T 组：这里必须用 **OOF** 目标编码。
    # 若用全量 train 一次 fit，每行的编码里都含它自己的标签（自泄漏），
    # 重要性会被抬到第一，答辩一问就穿。
    # 注：最终提交模型用全量 fit 是允许的（那时没有"未来折"），但解释重要性不行。
    if "T" in set(groups) and "Flight Distance" in X.columns:
        from src.common.split import load_folds

        folds, _ = load_folds(cfg)
        te_col = np.full(len(X), np.nan)
        for tr, va in folds:
            enc = TargetEncoder("Flight Distance", smooth=20.0).fit(X.iloc[tr], y[tr])
            te_col[va] = enc.transform(X.iloc[va]).to_numpy()
        X["Flight Distance__te"] = te_col
        print("[imp] T 组已用 OOF 目标编码（避免自泄漏）")

    cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
    print(f"[imp] 训练 {len(X):,} 行 × {X.shape[1]} 特征（{len(cat_cols)} 个类别列）...")
    model = _make_lgbm(cfg, cfg["seed"])
    model.fit(X, y)

    # sklearn 的 feature_importances_ 默认是 "split"（被选为分裂点的次数），
    # 会系统性偏向连续高基数特征。这里显式取 "gain"（分裂带来的增益）。
    gains = model.booster_.feature_importance(importance_type="gain")
    imp = (
        pd.Series(gains, index=X.columns, name="gain")
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    total = imp["gain"].sum()
    imp["share"] = imp["gain"] / total if total else 0.0
    out_csv = path_of(cfg, "reports") / "feature_importance.csv"
    imp.to_csv(out_csv, index=False, encoding="utf-8-sig")

    top = imp.head(args.top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(top) + 1.6))
    ax.barh(top["feature"], top["share"] * 100, color="#2b6cb0")
    for i, v in enumerate(top["share"] * 100):
        ax.text(v, i, f" {v:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("share of total gain (%)")
    ax.set_title(f"Feature importance (LightGBM gain), top {len(top)}")
    fig.tight_layout()
    png = path_of(cfg, "figures") / "fig_feat_importance.png"
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)

    print("\n[imp] top-10：")
    for _, r in imp.head(10).iterrows():
        print(f"  {r['feature']:<34} {r['share']:>6.2%}")
    print(f"\n[imp] 图   -> {png}")
    print(f"[imp] 数据 -> {out_csv}")
    print("\n[imp] 提醒：重要性高 ≠ 消融增益大。报告里两件事分开写。")


if __name__ == "__main__":
    main()
