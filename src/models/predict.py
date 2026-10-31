"""P2 交付：最终模型 → 测试集预测 → submission_final.csv（H3），及供 P1/P3 调用的接口（H2）。

设计原则（与 plan.md 铁律一致）：
- 不修改 run_cv（P1 消融依赖其契约）；本模块是 P2 在 src/models/ 内的独立交付。
- 测试集预测采用与 OOF 完全一致的 5 折协议：第 f 折在 4 折上 fit（含目标编码 T 仅在该 4 折 fit），
  对测试集预测时用该折的编码器 transform，最后对 5 个折模型的测试预测取平均 → 无泄漏。
- 特征构造与 run_cv 共用 `FeatureBuilder`：Flight Distance 的 dist_bin 边界在全量 train 上
  定一次（B 组），C 的频次参照是 train+test 全量。**不要**在每折/测试集上各自预处理，
  否则同一个人在 OOF 和 submission 里会落进不同的箱子。
- 记录可复现：run_id 含 model/feat/seed/tag，stage=final，CV 指标/params_hash/git_commit
  与 OOF 一并落盘（复用 train.log_run，不另写一套日志格式）。
- 提交的是概率（satisfaction 列），不是 0/1 标签。

供 P1/P3 调用的接口：
    from src.models.predict import predict_test, train_final
    res = predict_test(cfg, model="lgbm", groups=["base","C","T"])   # 写 submission_final.csv
    res["run_id"], res["cv_mean"], res["oof"]
    model, X_test, y_train, schema = train_final(cfg)                # 仅供预测，不供 SHAP

用法：
    python -m src.models.predict --config configs/base.yaml --model lgbm --groups base,C,T --tag final
"""

from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.common.config import git_commit, load_config, params_hash, path_of
from src.common.split import load_folds
from src.data.schema import detect_schema
from src.features.build import FeatureBuilder, make_count_ref
from src.features.encode import TargetEncoder
from src.models.train import _make_lgbm, log_run, make_run_id

_DIST = "Flight Distance"


def _feat_ver(groups) -> str:
    """exp_log 里的 feat_ver 口径与 train.run_cv 完全一致（base 隐含、B 不展开成 A+B）。"""
    gs = [g for g in groups if g != "base"]
    return "base" if not gs else "+".join(gs)


def _fit_te(X: pd.DataFrame, y):
    """目标编码只在给定数据上 fit（调用方保证这里是训练折/全量 train）。"""
    return TargetEncoder(_DIST, smooth=20.0).fit(X, y)


def _add_target_encoding(X: pd.DataFrame, y, fit_on: pd.DataFrame) -> pd.DataFrame:
    """在 fit_on（仅训练数据）上 fit 目标编码，transform X。返回带 __te 列的 X。"""
    X = X.copy()
    X[f"{_DIST}__te"] = _fit_te(fit_on, y).transform(X)
    return X


def predict_test(cfg: dict, model: str = "lgbm", groups=("base", "C", "T"),
                 tag: str = "final", owner: str = "P2",
                 verbose: bool = True, seed: int | None = None,
                 stage: str = "final", log: bool = True) -> dict:
    """5 折协议对测试集做平均预测，写出 submission_final.csv、OOF 与 exp_log 记录。

    返回 {run_id, test_prob, cv_mean, cv_std, oof}：
      run_id   —— 本次最终交付的唯一标识（含 model/feat/seed/tag）
      test_prob—— 测试集概率（5 折模型平均）
      oof      —— 与 run_cv 同协议的全量 OOF（长度 = train 行数）
    """
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

    # C 频次编码：train+test 全量统计（与 run_cv 完全同一份参照表）
    count_ref = make_count_ref(df, test) if "C" in set(groups) else None

    # 特征：全量 train 上 fit 一次（dist_bin 边界固化），逐折/测试集只做 transform
    builder = FeatureBuilder(schema, groups, count_ref=count_ref).fit(df)

    folds, meta = load_folds(cfg)
    gset = set(groups)
    feat_ver = _feat_ver(groups)
    run_id = make_run_id(owner=owner, stage=stage, model=model,
                         feat_ver=feat_ver, seed=seed, tag=tag)
    if verbose:
        print(f"[predict] model={model}  groups={'+'.join(g for g in groups if g != 'base') or 'base'}"
              f"  |  {len(folds)} 折平均预测  |  seed={seed}")
        print(f"[predict] run_id={run_id}")

    oof = np.full(len(df), np.nan, dtype="float64")
    test_preds, aucs = [], []
    n_features = 0
    Xte_base = builder.transform(test)      # 测试集特征与折无关，算一次；T 列逐折再加
    for f, (tr, va) in enumerate(folds):
        Xtr = builder.transform(df.iloc[tr])
        Xva = builder.transform(df.iloc[va])
        Xte = Xte_base.copy()
        ytr = y[tr]
        if "T" in gset and _DIST in Xtr.columns:
            te = _fit_te(Xtr, ytr)          # 只用本折训练数据 fit
            Xtr[f"{_DIST}__te"] = te.transform(Xtr)
            Xva[f"{_DIST}__te"] = te.transform(Xva)
            Xte[f"{_DIST}__te"] = te.transform(Xte)
        n_features = int(Xtr.shape[1])
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

    # 最终 OOF（带 id 列，供 P3 评估 / 误差分析）：run_id 命名 + 旧路径兼容
    oof_dir = path_of(cfg, "oof_dir")
    oof_df = pd.DataFrame({"row": np.arange(len(oof)), "y": y, "pred": oof})
    if schema["id_col"] and schema["id_col"] in df.columns:
        oof_df.insert(0, schema["id_col"], df[schema["id_col"]].to_numpy())
    oof_path = oof_dir / f"{run_id}.parquet"
    oof_df.to_parquet(oof_path, index=False)
    legacy_oof_path = oof_dir / "P2_final_oof.parquet"
    oof_df.to_parquet(legacy_oof_path, index=False)

    # 落账：复用 train.log_run，stage=final（提交历史行一律 append，不改旧行）
    log_path = None
    if log:
        params = dict(cfg["model"].get(model, {}))
        params.update({"features": feat_ver, "seed": seed, "n_features": n_features})
        row = {
            "run_id": run_id,
            "date": str(date.today()),
            "owner": owner,
            "stage": stage,
            "model": model,
            "feat_ver": feat_ver,
            "split_ver": meta.get("version", "v1"),
            "params_hash": params_hash(params),
            "cv_mean": round(cv_mean, 6),
            "cv_std": round(cv_std, 6),
            "public_lb": "",
            "git_commit": git_commit(),
            "notes": (f"n_feat={n_features}; {len(folds)} 折测试集平均预测"
                      f"; seed={seed}; tag={tag}"),
        }
        log_path = log_run(cfg, row, stage=stage)

    if verbose:
        print(f"[predict] OOF AUC = {cv_mean:.5f} ± {cv_std:.5f}（应与 run_cv 的同组结果一致）")
        print(f"[predict] submission -> {sub_path}  ({len(sub):,} 行)")
        print(f"[predict] OOF        -> {oof_path}")
        print(f"[predict] OOF(旧路径)-> {legacy_oof_path}")
        if log_path is not None:
            print(f"[predict] 登记       -> {log_path}")

    return {"run_id": run_id, "test_prob": test_prob,
            "cv_mean": cv_mean, "cv_std": cv_std, "oof": oof}


def train_final(cfg: dict, model: str = "lgbm", groups=("base", "C", "T"),
                verbose: bool = True, seed: int | None = None):
    """在 FULL train 上训练最终模型，返回 (model, X_test, y_train, schema)。

    ⚠️ **只能用于预测测试集，不得用于可解释性 / SHAP / badcase 分析。**

    原因：T 组的目标编码在这里是在完整 train（包含每一行自己的标签）上 fit，
    再 transform 同一张表，于是每一行的编码值都含有它自己的标签（自泄漏）。
    测 test 没问题（test 无标签，也从未参与 fit），但拿这个模型算 SHAP，
    会把"标签泄漏"解释成"Flight Distance 很重要"。

    该行为是有意保留的（predict 的语义就是"全量训练一个模型"）。
    做可解释性请用 P3 的 **OOF 目标编码**：每折用其余 4 折 fit、只 transform 本折，
    见 src/eval/interpret.py 的 OOF-TE 实现；不要直接复用本函数。
    """
    if model != "lgbm":
        raise NotImplementedError("[train_final] 目前只支持 lgbm（主力模型）")
    seed = int(seed if seed is not None else cfg["seed"])

    proc = path_of(cfg, "processed_dir")
    df, test = pd.read_parquet(proc / "train.parquet"), pd.read_parquet(proc / "test.parquet")
    schema = detect_schema(df, cfg, verbose=False)
    y = df[schema["target"]].to_numpy()

    count_ref = make_count_ref(df, test) if "C" in set(groups) else None

    builder = FeatureBuilder(schema, groups, count_ref=count_ref).fit(df)
    Xtr = builder.transform(df)
    Xte = builder.transform(test)
    if "T" in set(groups) and _DIST in Xtr.columns:
        Xtr = _add_target_encoding(Xtr, y, Xtr)   # ← 自泄漏：仅供预测，禁止 SHAP
        Xte = _add_target_encoding(Xte, y, Xtr)
    m = _make_lgbm(cfg, seed)
    m.fit(Xtr, y)
    if verbose:
        print(f"[train_final] 在 {len(Xtr):,} 行 / {Xtr.shape[1]} 特征上训练完成；X_test 形状 {Xte.shape}")
        print("[train_final] 注意：目标编码在完整 train 上 fit，仅供预测；SHAP 请用 P3 的 OOF-TE")
    return m, Xte, y, schema


def main() -> None:
    ap = argparse.ArgumentParser(description="P2 最终模型：测试集预测 + submission_final.csv（H2/H3）")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--model", default="lgbm", choices=["lgbm"])
    ap.add_argument("--groups", default="base,C,T", help="逗号分隔：base,A,B,C,T")
    ap.add_argument("--tag", default="final")
    ap.add_argument("--owner", default="P2")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--stage", default="final", help="写进 exp_log 的阶段名（默认 final）")
    ap.add_argument("--no-log", action="store_true", help="只写 submission/OOF，不追加 exp_log")
    args = ap.parse_args()
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    cfg = load_config(args.config)
    predict_test(cfg, model=args.model, groups=groups, tag=args.tag,
                 owner=args.owner, seed=args.seed, stage=args.stage,
                 log=not args.no_log)


if __name__ == "__main__":
    main()
