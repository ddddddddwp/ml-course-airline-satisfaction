"""P2 学习曲线：单折 train/valid AUC 随 boosting 轮次变化（LightGBM）。

用于报告「训练、调参与过拟合控制」章节的必备图「学习曲线」：
- 用与主力模型完全一致的协议（默认 base+C+T、fold_v1、T 编码仅训练折 fit）
- n_estimators=1000 + early_stopping(50)，记录每轮 train/valid AUC
- 验证 train 与 valid 曲线未明显发散 → 过拟合受控

输出：<out>/p2_learning_curve.json（默认 <figures>/，供 figures_p2.py 画图）
不修改任何 P1/P3 代码，不依赖 run_cv。

路径约定：processed/figures 一律走 config + ROOT（path_of），输出走 --out（相对路径按仓库根解析），
所以在任何工作目录下运行结果都一致。

用法：
    python -m src.models.diag --config configs/base.yaml --groups C,T
    python -m src.models.diag --groups base,B,C,T --out reports/figures/_diag
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# ⚠️ 导入顺序不能动：Windows 上 lightgbm 先于 sklearn 加载会触发 OpenMP/DLL 冲突，
# 表现为训练时 `LGBM_DatasetSetField` access violation（0x0）。
# sklearn 必须先 import。train.py / predict.py 因为顶层先 import sklearn 所以没事。
import sklearn  # noqa: F401
import lightgbm as lgb
import pandas as pd

from src.common.config import ROOT, load_config, path_of
from src.common.split import load_folds
from src.data.schema import detect_schema
from src.features.build import FeatureBuilder, make_count_ref
from src.features.encode import TargetEncoder


def _resolve_out(out: str | Path | None, cfg: dict) -> Path:
    """--out 未给时用 config 的 figures 目录；给了就按仓库根解析相对路径。"""
    if out in (None, ""):
        return path_of(cfg, "figures")
    p = Path(out)
    if not p.is_absolute():
        p = ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="P2 学习曲线（单折 train/valid AUC）")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--groups", default="C,T", help="逗号分隔：base,A,B,C,T（base 隐含）")
    ap.add_argument("--out", default="", help="输出目录（默认 config 的 figures）")
    ap.add_argument("--seed", type=int, default=None, help="默认取 config 的 seed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    seed = int(args.seed if args.seed is not None else cfg["seed"])
    out_dir = _resolve_out(args.out, cfg)

    proc = path_of(cfg, "processed_dir")
    df = pd.read_parquet(proc / "train.parquet")
    schema = detect_schema(df, cfg, verbose=False)
    y = df[schema["target"]].to_numpy()

    # 与 train/predict 同一套特征：全量 fit 一次，transform 任意子集
    count_ref = None
    test_path = proc / "test.parquet"
    if "C" in set(groups) and test_path.exists():
        count_ref = make_count_ref(df, pd.read_parquet(test_path))
    builder = FeatureBuilder(schema, groups, count_ref=count_ref).fit(df)
    X = builder.transform(df)

    folds, _ = load_folds(cfg)
    tr, va = folds[0]
    Xtr, Xva = X.iloc[tr].copy(), X.iloc[va].copy()
    ytr, yva = y[tr], y[va]

    # 目标编码只在训练折 fit（防泄漏）
    if "T" in set(groups) and "Flight Distance" in Xtr.columns:
        te = TargetEncoder("Flight Distance", smooth=20.0).fit(Xtr, ytr)
        Xtr["Flight Distance__te"] = te.transform(Xtr)
        Xva["Flight Distance__te"] = te.transform(Xva)

    p = dict(cfg["model"]["lgbm"])
    p["random_state"] = seed
    p["n_estimators"] = 1000
    p["verbose"] = -1
    clf = lgb.LGBMClassifier(**p)
    clf.fit(
        Xtr, ytr,
        eval_set=[(Xtr, ytr), (Xva, yva)],
        eval_names=["train", "valid"],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    res = clf.evals_result_
    out = {
        "round": list(range(1, len(res["train"]["auc"]) + 1)),
        "train_auc": res["train"]["auc"],
        "valid_auc": res["valid"]["auc"],
        "best_iter": int(clf.best_iteration_),
        "valid_auc_best": float(res["valid"]["auc"][clf.best_iteration_ - 1]),
        "train_auc_end": float(res["train"]["auc"][-1]),
        "valid_auc_end": float(res["valid"]["auc"][-1]),
        "groups": "+".join(g for g in groups if g != "base") or "base",
        "seed": seed,
        "n_features": int(Xtr.shape[1]),
    }
    out_path = out_dir / "p2_learning_curve.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[diag] best_iter={out['best_iter']}  valid_auc@best={out['valid_auc_best']:.5f}"
          f"  train_auc_end={out['train_auc_end']:.5f}  valid_auc_end={out['valid_auc_end']:.5f}")
    print(f"[diag] -> {out_path}")


if __name__ == "__main__":
    main()
