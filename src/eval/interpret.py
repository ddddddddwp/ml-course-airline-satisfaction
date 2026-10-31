"""P3 可解释性：不依赖 shap 包，直接用 LightGBM 原生 TreeSHAP（pred_contrib）。

⚠️ 防自泄漏（答辩必问，代码里必须能指出来）：
- T 组目标编码严格用 **OOF**：每一折只用其余 4 折 fit `TargetEncoder`，再 transform 本折。
  这样每行的 `Flight Distance__te` 都不含自己的标签。
- 绝不能用全量 train 一次 fit 目标编码：那会让 SHAP 把 TE 排到第一，是典型的自泄漏。
- 也刻意**不调用** `src.models.predict.train_final()`：它是给最终提交用的（TE 在全量 train 上
  fit），拿它做解释会高估 TE 重要性。本模块自己按 OOF 口径重建特征与模型。

实现：
1. `prepare_X` 构造 base/C 特征；
2. 按 `splits/fold_v1.json` 逐折 fit TargetEncoder，得到 OOF 的 `Flight Distance__te`；
3. 在完整 train 上训练 `_make_lgbm`（与主模型同参）；
4. 固定 seed 采样 20000 行，`booster.predict(X_sample, pred_contrib=True)`；
5. 去掉最后一列 base value，得到 SHAP 矩阵，输出重要性 CSV 与两张图。

用法：
    $env:PYTHONPATH="."
    python -m src.eval.interpret --config configs/base.yaml --sample 20000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.config import ROOT, load_config
from src.common.split import load_folds
from src.data.schema import detect_schema
from src.eval._io import ensure_lgbm_dtypes
from src.features.build import prepare_X
from src.features.encode import TargetEncoder
from src.models.train import _make_lgbm

plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)

_DIST = "Flight Distance"


def _resolve(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def build_oof_features(cfg: dict, groups: list[str]) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
    """构造与主模型一致的特征矩阵，其中 T 组用 OOF 目标编码（防自泄漏）。"""
    proc = _resolve(cfg["paths"]["processed_dir"])
    tr_path, te_path = proc / "train.parquet", proc / "test.parquet"
    if not tr_path.exists():
        raise FileNotFoundError(f"找不到 {tr_path}；先跑 make data")
    if "C" in set(groups) and not te_path.exists():
        raise FileNotFoundError(
            f"C 组频次编码需要 test.parquet（train+test 全量统计），但 {te_path} 不存在；"
            "先跑 make data，或改用 --groups base,T"
        )

    df = pd.read_parquet(tr_path)
    test = pd.read_parquet(te_path) if te_path.exists() else None
    schema = detect_schema(df, cfg, verbose=False)
    y = df[schema["target"]].to_numpy()

    count_ref = None
    if "C" in set(groups) and test is not None:
        cols = [c for c in (_DIST, "Age") if c in test.columns and c in df.columns]
        if cols:
            count_ref = pd.concat([df[cols], test[cols]], ignore_index=True)

    X = prepare_X(df, schema, groups, count_ref=count_ref)

    if "T" in set(groups) and _DIST in X.columns:
        folds, meta = load_folds(cfg)
        if len(X) != int(meta.get("n_rows", len(X))):
            raise ValueError(
                f"train 行数 {len(X):,} 与冻结划分 {meta.get('n_rows')} 不一致；"
                "先确认 data/processed/train.parquet 与 splits/fold_v1.json 是同一版本。"
            )
        te_col = np.full(len(X), np.nan, dtype="float64")
        for tr, va in folds:
            # fit 只见到训练折；验证折只 transform -> OOF 目标编码，无自泄漏
            enc = TargetEncoder(_DIST, smooth=20.0).fit(X.iloc[tr], y[tr])
            te_col[va] = enc.transform(X.iloc[va]).to_numpy()
        if np.isnan(te_col).any():
            raise ValueError(
                f"OOF 目标编码仍有 {int(np.isnan(te_col).sum()):,} 行缺失；"
                "检查 splits/fold_v1.json 是否覆盖全部训练行。"
            )
        X[_DIST + "__te"] = te_col.astype("float32")
        print(f"[interpret] T 组使用 OOF 目标编码（{len(folds)} 折逐折 fit，避免自泄漏）")

    X = ensure_lgbm_dtypes(X)
    print(f"[interpret] 特征矩阵 {X.shape[0]:,} × {X.shape[1]}（{len(groups)} 组：{'+'.join(groups)}）")
    return X, y, df, schema


def compute_shap(model, X_sample: pd.DataFrame) -> tuple[np.ndarray, float]:
    """TreeSHAP：返回 (n_sample × n_feature 的 SHAP 矩阵, base value)。

    LightGBM 的 `pred_contrib=True` 返回 n_feature+1 列，最后一列是 base value
    （即模型输出的基准 log-odds），必须去掉，否则会被误当成一个特征。
    """
    contrib = np.asarray(
        model.booster_.predict(X_sample, pred_contrib=True), dtype="float64"
    )
    if contrib.ndim == 1:
        contrib = contrib.reshape(1, -1)
    expected = X_sample.shape[1] + 1
    if contrib.shape[1] != expected:
        raise RuntimeError(
            f"pred_contrib 返回 {contrib.shape[1]} 列，期望 {expected} 列"
            f"（{X_sample.shape[1]} 特征 + 1 base value）；特征顺序可能不一致。"
        )
    shap = contrib[:, :-1]
    base_value = float(contrib[:, -1].mean())
    return shap, base_value


def _plot_importance(imp: pd.DataFrame, top: int, path: Path) -> None:
    d = imp.head(int(top)).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(d) + 1.6))
    ax.barh(d["feature"], d["share"] * 100.0, color="#2b6cb0")
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(r["share"] * 100.0, i, f" {r['share'] * 100:.1f}%", va="center", fontsize=7)
    ax.set_xlabel("share of mean |SHAP| (%)")
    ax.set_title(f"P3 SHAP importance (TreeSHAP, top {len(d)})")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_dependence(
    X_sample: pd.DataFrame,
    shap: np.ndarray,
    imp: pd.DataFrame,
    path: Path,
    seed: int,
) -> None:
    """mean|SHAP| 前 3 个特征：SHAP 值 vs 特征值；类别特征用抖动展示离散取值。"""
    feats = imp.head(3)["feature"].tolist()
    cols = list(X_sample.columns)
    rng = np.random.default_rng(int(seed))

    fig, axes = plt.subplots(1, max(len(feats), 1), figsize=(4.4 * max(len(feats), 1), 3.8))
    axes = np.atleast_1d(axes)
    for ax, feat in zip(axes, feats):
        ci = cols.index(feat)
        values = X_sample[feat]
        sv = shap[:, ci]
        if isinstance(values.dtype, pd.CategoricalDtype):
            codes = values.cat.codes.to_numpy(dtype="float64")
            jitter = rng.uniform(-0.32, 0.32, size=codes.size)
            ax.scatter(codes + jitter, sv, s=4, alpha=0.35, color="#2b6cb0", linewidths=0)
            cats = [str(c) for c in values.cat.categories]
            step = max(1, int(np.ceil(len(cats) / 10)))
            ticks = list(range(0, len(cats), step))
            ax.set_xticks(ticks, [cats[i][:18] for i in ticks], rotation=30, fontsize=6)
            ax.set_xlabel(f"{feat} (category code, jittered)")
        else:
            vals = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
            ax.scatter(vals, sv, s=4, alpha=0.35, color="#2b6cb0", linewidths=0)
            ax.set_xlabel(feat)
        ax.axhline(0.0, color="#718096", lw=1.0, ls="--")
        ax.set_ylabel("SHAP value (log-odds)")
        share = float(imp.loc[imp["feature"] == feat, "share"].iloc[0])
        ax.set_title(f"{feat}\nmean|SHAP| share={share:.1%}", fontsize=8)
    fig.suptitle(
        f"P3 SHAP dependence, top {len(feats)} (sampled n={len(X_sample):,}, OOF target encoding)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_interpret(
    cfg: dict,
    *,
    groups: list[str],
    out_dir: str | Path = "reports/eval",
    sample: int = 20000,
    seed: int | None = None,
    top: int = 20,
) -> dict:
    """训练全量模型 -> TreeSHAP -> 重要性 CSV + 两张图。"""
    if int(sample) < 1:
        raise ValueError(f"[interpret] --sample 必须 >= 1，收到 {sample}")
    seed = int(seed if seed is not None else cfg.get("seed", 42))

    X, y, _df, _schema = build_oof_features(cfg, groups)
    model = _make_lgbm(cfg, seed)

    t0 = time.time()
    model.fit(X, y)
    print(
        f"[interpret] 全量训练完成：{len(X):,} 行 × {X.shape[1]} 特征，"
        f"{time.time() - t0:.1f}s（注意：模型只用 OOF TE 特征，TE 本身来自折内 fit）"
    )

    n = len(X)
    k = min(int(sample), n)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=k, replace=False))
    X_sample = X.iloc[idx].reset_index(drop=True)

    shap, base_value = compute_shap(model, X_sample)
    mean_abs = np.abs(shap).mean(axis=0)
    total = float(mean_abs.sum())
    imp = pd.DataFrame(
        {
            "feature": [str(c) for c in X_sample.columns],
            "mean_abs_shap": mean_abs,
            "share": mean_abs / total if total > 0 else np.zeros_like(mean_abs),
        }
    ).sort_values("mean_abs_shap", ascending=False, kind="mergesort")
    imp.insert(0, "rank", np.arange(1, len(imp) + 1))
    imp = imp.reset_index(drop=True)

    out = _resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "shap_importance.csv"
    imp.to_csv(csv_path, index=False, encoding="utf-8-sig")

    _plot_importance(imp, top=top, path=out / "p3_shap_importance.png")
    _plot_dependence(X_sample, shap, imp, path=out / "p3_shap_dependence.png", seed=seed)

    print(f"[interpret] 采样 {k:,} 行  |  base value（log-odds）= {base_value:.4f}")
    print(f"[interpret] SHAP 数据 -> {csv_path}")
    print(f"[interpret] 图       -> {out / 'p3_shap_importance.png'} / {out / 'p3_shap_dependence.png'}")
    print("[interpret] top-10（mean |SHAP|）：")
    for _, r in imp.head(10).iterrows():
        print(f"  {int(r['rank']):>2}. {r['feature']:<32} {r['mean_abs_shap']:.5f}  ({r['share']:.2%})")
    print(
        "[interpret] 读法：SHAP 说明“模型怎么用特征”，不是因果；"
        "重要性高 ≠ 去掉一定掉分（那要看消融）。"
    )
    return {
        "csv": str(csv_path),
        "top10": imp.head(10).to_dict(orient="records"),
        "base_value": base_value,
        "n_sample": k,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="P3 可解释性：LightGBM TreeSHAP（不依赖 shap 包）")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--groups", default="base,C,T", help="逗号分隔特征组")
    ap.add_argument("--out", default="reports/eval")
    ap.add_argument("--sample", type=int, default=20000, help="SHAP 采样行数（70 万行别全量）")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--top", type=int, default=20, help="重要性图展示的特征数")
    args = ap.parse_args()

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    cfg = load_config(args.config)
    try:
        run_interpret(
            cfg,
            groups=groups,
            out_dir=args.out,
            sample=args.sample,
            seed=args.seed,
            top=args.top,
        )
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"[interpret] {e}") from None
    except RuntimeError as e:
        raise SystemExit(f"[interpret] {e}") from None


if __name__ == "__main__":
    main()
