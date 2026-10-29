"""P2 交付：最终模型 → 测试集预测 → submission_final.csv（H3），及供 P1/P3 调用的接口（H2）。

设计原则（与 plan.md 铁律一致）：
- 不修改 run_cv（P1 消融依赖其契约）；本模块是 P2 在 src/models/ 内的独立交付。
- 测试集预测采用与 OOF 完全一致的 5 折协议：第 f 折在 4 折上 fit（含目标编码 T 仅在该 4 折 fit），
  对测试集预测时用该折的编码器 transform，最后对 5 个折模型的测试预测取平均 → 无泄漏。
- 频次编码 C 的 count_ref = train+test 全量统计（与 run_cv 一致）。
- 提交的是概率（satisfaction 列），不是 0/1 标签。

供 P1/P3 调用的接口：
    from src.models.predict import predict_test, train_final
    test_prob = predict_test(cfg, model="lgbm", groups=["base","C","T"])   # 写 submission_final.csv
    model, X_test, y_train, schema = train_final(cfg)                     # 供 SHAP / badcase

用法：
    python -m src.models.predict --config configs/base.yaml --model lgbm --groups base,C,T --tag final
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.common.config import load_config, path_of
from src.common.split import load_folds
from src.data.schema import detect_schema
from src.features.build import prepare_X
from src.features.encode import TargetEncoder
from src.models.train import _make_lgbm

_DIST = "Flight Distance"


def _add_target_encoding(X: pd.DataFrame, y, fit_on: pd.DataFrame) -> pd.DataFrame:
    """在 fit_on（仅训练数据）上 fit 目标编码，transform X。返回带 __te 列的 X。"""
    te = TargetEncoder(_DIST, smooth=20.0).fit(fit_on, y)
    X = X.copy()
    X[f"{_DIST}__te"] = te.transform(X)
    return X


def predict_test(cfg: dict, model: str = "lgbm", groups=("base", "C", "T"),
                 tag: str = "final", owner: str = "P2",
                 verbose: bool = True, seed: int | None = None) -> dict:
    """5 折协议对测试集做平均预测，写出 submission_final.csv；返回 {test_prob, cv_mean, oof}。"""
    if model != "lgbm":
        raise NotImplementedError("[predict] 目前只支持 lgbm（主力模型）")
    seed = int(seed if seed is not None else cfg["seed"])

    proc = path_of(cfg, "processed_dir")
    tr_path, te_path = proc / "train.parquet", proc / "test.parquet"
    if not tr_path.exists() or not te_path.exists():
        raise SystemExit("[predict] 找不到 processed 数据，先跑 make data")
    df, test = pd.read_parquet(tr_path), pd.read_parquet(te_path)
    schema = detect_schema(df, cfg, verbose=False)
    y = df[schema["target"]].to_numpy()

    # C 频次编码：train+test 全量统计（与 run_cv 一致）
    count_ref = None
    if "C" in set(groups):
        cols = [c for c in (_DIST, "Age") if c in test.columns and c in df.columns]
        if cols:
            count_ref = pd.concat([df[cols], test[cols]], ignore_index=True)

    folds, _ = load_folds(cfg)
    gset = set(groups)
    if verbose:
        gname = "+".join(g for g in groups if g != "base") or "base"
        print(f"[predict] model={model}  groups={gname}  |  {len(folds)} 折平均预测  |  seed={seed}")

    oof = np.full(len(df), np.nan, dtype="float64")
    test_preds, aucs = [], []
    for f, (tr, va) in enumerate(folds):
        Xtr = prepare_X(df.iloc[tr].copy(), schema, groups, count_ref=count_ref)
        Xva = prepare_X(df.iloc[va].copy(), schema, groups, count_ref=count_ref)
        Xte = prepare_X(test.copy(), schema, groups, count_ref=count_ref)
        ytr = y[tr]
        if "T" in gset and _DIST in Xtr.columns:
            Xtr = _add_target_encoding(Xtr, ytr, Xtr)
            Xva = _add_target_encoding(Xva, ytr, Xtr)   # 注意：用训练折的编码器
            Xte = _add_target_encoding(Xte, ytr, Xtr)
        m = _make_lgbm(cfg, seed)
        m.fit(Xtr, ytr)
        pva = m.predict_proba(Xva)[:, 1]
        oof[va] = pva
        aucs.append(float(roc_auc_score(y[va], pva)))
        test_preds.append(m.predict_proba(Xte)[:, 1])
        if verbose:
            print(f"  fold {f}: valid AUC {aucs[-1]:.5f}")

    cv_mean, cv_std = float(np.mean(aucs)), float(np.std(aucs))
    test_prob = (np.mean(test_preds, axis=0) if len(test_preds) > 1
                 else np.asarray(test_preds[0]))

    # 写 submission_final.csv（概率，非标签）
    out_dir = path_of(cfg, "reports")
    sub = pd.DataFrame({
        schema["id_col"]: test[schema["id_col"]].to_numpy(),
        schema["target"]: np.asarray(test_prob, dtype="float32"),
    })
    sub_path = out_dir / "submission_final.csv"
    sub.to_csv(sub_path, index=False)

    # 写最终 OOF（供 P3 评估 / 误差分析）
    oof_dir = Path(cfg["paths"]["oof_dir"])
    if not oof_dir.is_absolute():
        oof_dir = Path(__file__).resolve().parents[2] / oof_dir
    oof_dir.mkdir(parents=True, exist_ok=True)
    oof_path = oof_dir / "P2_final_oof.parquet"
    pd.DataFrame({"row": np.arange(len(oof)), "y": y, "pred": oof}).to_parquet(oof_path, index=False)

    if verbose:
        print(f"[predict] OOF AUC = {cv_mean:.5f} ± {cv_std:.5f}（应与 run_cv 的 0.96019 一致）")
        print(f"[predict] submission -> {sub_path}  ({len(sub):,} 行)")
        print(f"[predict] OOF        -> {oof_path}")
    return {"test_prob": test_prob, "cv_mean": cv_mean, "cv_std": cv_std, "oof": oof}


def train_final(cfg: dict, model: str = "lgbm", groups=("base", "C", "T"),
                verbose: bool = True, seed: int | None = None):
    """在 FULL train 上训练最终模型（T 编码仅在全量 train 上 fit），返回 (model, X_test, y_train, schema)。

    供 P3 做 SHAP / badcase / 可解释性分析时使用——直接 import 本函数即可，无需关心特征构造细节。
    """
    if model != "lgbm":
        raise NotImplementedError("[train_final] 目前只支持 lgbm（主力模型）")
    seed = int(seed if seed is not None else cfg["seed"])

    proc = path_of(cfg, "processed_dir")
    df, test = pd.read_parquet(proc / "train.parquet"), pd.read_parquet(proc / "test.parquet")
    schema = detect_schema(df, cfg, verbose=False)
    y = df[schema["target"]].to_numpy()

    count_ref = None
    if "C" in set(groups):
        cols = [c for c in (_DIST, "Age") if c in test.columns and c in df.columns]
        if cols:
            count_ref = pd.concat([df[cols], test[cols]], ignore_index=True)

    Xtr = prepare_X(df, schema, groups, count_ref=count_ref)
    Xte = prepare_X(test, schema, groups, count_ref=count_ref)
    if "T" in set(groups) and _DIST in Xtr.columns:
        Xtr = _add_target_encoding(Xtr, y, Xtr)
        Xte = _add_target_encoding(Xte, y, Xtr)
    m = _make_lgbm(cfg, seed)
    m.fit(Xtr, y)
    if verbose:
        print(f"[train_final] 在 {len(Xtr):,} 行 / {Xtr.shape[1]} 特征上训练完成；X_test 形状 {Xte.shape}")
    return m, Xte, y, schema


def main() -> None:
    ap = argparse.ArgumentParser(description="P2 最终模型：测试集预测 + submission_final.csv（H2/H3）")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--model", default="lgbm", choices=["lgbm"])
    ap.add_argument("--groups", default="base,C,T", help="逗号分隔：base,A,B,C,T")
    ap.add_argument("--tag", default="final")
    ap.add_argument("--owner", default="P2")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    cfg = load_config(args.config)
    predict_test(cfg, model=args.model, groups=groups, tag=args.tag,
                 owner=args.owner, seed=args.seed)


if __name__ == "__main__":
    main()
