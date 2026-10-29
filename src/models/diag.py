"""P2 学习曲线：单折 train/valid AUC 随 boosting 轮次变化（LightGBM）。

用于报告「训练、调参与过拟合控制」章节的必备图「学习曲线」：
- 用与主力模型完全一致的协议（base+C+T、fold_v1、T 编码仅训练折 fit）
- n_estimators=1000 + early_stopping(50)，记录每轮 train/valid AUC
- 验证 train 与 valid 曲线未明显发散 → 过拟合受控

输出：reports/figures/p2_learning_curve.json（供 figures_p2.py 画图）
不修改任何 P1/P3 代码，不依赖 run_cv。
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.common.config import load_config, path_of
from src.data.schema import detect_schema
from src.features.build import prepare_X
from src.features.encode import TargetEncoder
from src.common.split import load_folds


def main() -> None:
    cfg = load_config("configs/base.yaml")
    proc = path_of(cfg, "processed_dir")
    df = pd.read_parquet(proc / "train.parquet")
    schema = detect_schema(df, cfg, verbose=False)
    y = df[schema["target"]].to_numpy()

    # count_ref：与 train.py 一致（C 组用 train+test 的 Flight Distance/Age 分布）
    count_ref = None
    test = pd.read_parquet(proc / "test.parquet")
    cols = [c for c in ("Flight Distance", "Age") if c in test.columns]
    if cols:
        count_ref = pd.concat([df[cols], test[cols]], ignore_index=True)

    X = prepare_X(df, schema, ("C", "T"), count_ref=count_ref)
    folds, _ = load_folds(cfg)
    tr, va = folds[0]
    Xtr, Xva = X.iloc[tr].copy(), X.iloc[va].copy()
    ytr, yva = y[tr], y[va]

    # 目标编码只在训练折 fit（防泄漏）
    te = TargetEncoder("Flight Distance", smooth=20.0).fit(Xtr, ytr)
    Xtr["Flight Distance__te"] = te.transform(Xtr)
    Xva["Flight Distance__te"] = te.transform(Xva)

    p = dict(cfg["model"]["lgbm"])
    p["random_state"] = 42
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
    }
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    Path("reports/figures/p2_learning_curve.json").write_text(json.dumps(out, indent=2))
    print(f"[diag] best_iter={out['best_iter']}  valid_auc@best={out['valid_auc_best']:.5f}"
          f"  train_auc_end={out['train_auc_end']:.5f}  valid_auc_end={out['valid_auc_end']:.5f}")


if __name__ == "__main__":
    main()
