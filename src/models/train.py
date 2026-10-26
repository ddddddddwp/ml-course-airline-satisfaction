"""5 折训练骨架（最小可用实现）。

⚠️ 归属：本文件最终由 P2 拥有与替换。P1 只保证它的**接口契约定死**：
    run_cv(cfg, model, groups, fold_array=None, ...) -> dict
    必须读取 splits/fold_v1.json 取折、必须把每次运行写进 reports/exp_log.csv。
只要接口不变，P2 换成自己的训练器，P1 的消融矩阵不用改。

用法：
    python -m src.models.train --config configs/base.yaml --model lgbm --groups base,A,B,C
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.common.config import ROOT, git_commit, load_config, params_hash, path_of
from src.common.split import build_folds, load_folds
from src.data.schema import detect_schema
from src.features.build import prepare_X
from src.features.encode import TargetEncoder

HEADER = [
    "run_id", "date", "owner", "stage", "model", "feat_ver", "split_ver",
    "params_hash", "cv_mean", "cv_std", "public_lb", "git_commit", "notes",
]


def _make_lgbm(cfg: dict, seed: int):
    import lightgbm as lgb

    p = dict(cfg["model"]["lgbm"])
    p["random_state"] = seed
    return lgb.LGBMClassifier(**p)


def _make_lr(cfg: dict, seed: int, cat_cols: list[str], num_cols: list[str]):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cat = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=50)),
    ])
    num = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])
    ct = ColumnTransformer([("cat", cat, cat_cols), ("num", num, num_cols)])
    return Pipeline([
        ("prep", ct),
        ("clf", LogisticRegression(
            C=float(cfg["model"]["lr"]["C"]),
            max_iter=int(cfg["model"]["lr"]["max_iter"]),
            n_jobs=-1, random_state=seed,
        )),
    ])


def _append_log(cfg: dict, row: dict) -> Path:
    p = Path(cfg["paths"].get("exp_log", "reports/exp_log.csv"))
    if not p.is_absolute():
        p = ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row], columns=HEADER)
    df.to_csv(p, mode="a", header=not p.exists() or p.stat().st_size == 0,
              index=False, encoding="utf-8")
    return p


def run_cv(cfg: dict, model: str = "lgbm", groups=("base",), fold_array=None,
           sample: int | None = None, seed: int | None = None, log: bool = True,
           verbose: bool = True, tag: str = "", owner: str = "P1",
           notes: str = "", drop_cols: list[str] | None = None) -> dict:
    groups = [g for g in groups if g != "base"]
    seed = int(seed if seed is not None else cfg["seed"])
    proc = path_of(cfg, "processed_dir")
    tr_path = proc / "train.parquet"
    if not tr_path.exists():
        raise SystemExit(f"[train] 找不到 {tr_path}，先跑 make data")

    df = pd.read_parquet(tr_path)
    schema = detect_schema(df, cfg, verbose=False)
    if sample and int(sample) < len(df):
        df = df.sample(int(sample), random_state=seed).reset_index(drop=True)
        fold_array = None
        if verbose:
            print(f"[train] 采样 {len(df):,} 行（仅调试，不可写进报告）")
    y = df[schema["target"]].to_numpy()

    count_ref = None
    if "C" in set(groups):
        te_path = proc / "test.parquet"
        if te_path.exists():
            test = pd.read_parquet(te_path)
            cols = [c for c in ("Flight Distance", "Age") if c in test.columns]
            if cols:
                count_ref = pd.concat([df[cols], test[cols]], ignore_index=True)

    X = prepare_X(df, schema, groups, count_ref=count_ref)

    if drop_cols:
        present = [c for c in drop_cols if c in X.columns]
        absent = [c for c in drop_cols if c not in X.columns]
        if present:
            X = X.drop(columns=present)
        if verbose:
            print(f"[train] 剔除列 {present}" + (f"（不存在，已忽略: {absent}）" if absent else ""))
        if not present:
            print("[train][警告] 要剔除的列一个都不存在，本次实验等价于原组")

    if fold_array is not None:
        fold_array = np.asarray(fold_array)
        if len(fold_array) != len(df):
            raise SystemExit(
                f"[train] fold 数组长度 {len(fold_array)} 与数据 {len(df)} 不一致"
            )
        k = int(fold_array.max()) + 1
        folds = [(np.where(fold_array != f)[0], np.where(fold_array == f)[0]) for f in range(k)]
        split_ver = "v1"
    elif sample:
        fa = build_folds(df, cfg)
        k = int(fa.max()) + 1
        folds = [(np.where(fa != f)[0], np.where(fa == f)[0]) for f in range(k)]
        split_ver = "v1(resampled)"
    else:
        folds, meta = load_folds(cfg)
        split_ver = meta.get("version", "v1")

    if verbose:
        print(f"[train] model={model}  groups={'base' if not groups else '+'.join(groups)}")
        print(f"[train] 特征列 {X.shape[1]}  |  样本 {len(X):,}  |  {len(folds)} 折  |  seed={seed}")

    oof = np.full(len(X), np.nan, dtype="float64")
    aucs: list[float] = []

    for f, (tr, va) in enumerate(folds):
        Xtr, Xva = X.iloc[tr].copy(), X.iloc[va].copy()
        ytr = y[tr]

        # 目标编码：只在训练折上 fit（T 组，最容易写出泄漏的地方）
        if "T" in set(groups) and "Flight Distance" in Xtr.columns:
            te = TargetEncoder("Flight Distance", smooth=20.0).fit(Xtr, ytr)
            Xtr["Flight Distance__te"] = te.transform(Xtr)
            Xva["Flight Distance__te"] = te.transform(Xva)

        cat_cols = [c for c in Xtr.columns if str(Xtr[c].dtype) == "category"]
        num_cols = [c for c in Xtr.columns if c not in cat_cols]

        if model == "lr":
            for c in cat_cols:
                Xtr[c] = Xtr[c].astype(str)
                Xva[c] = Xva[c].astype(str)
            m = _make_lr(cfg, seed, cat_cols, num_cols)
        else:
            m = _make_lgbm(cfg, seed)

        m.fit(Xtr, ytr)
        p = m.predict_proba(Xva)[:, 1]
        oof[va] = p
        aucs.append(float(roc_auc_score(y[va], p)))
        if verbose:
            print(f"  fold {f}: AUC {aucs[-1]:.5f}   n_train={len(tr):,}  n_valid={len(va):,}")

    cv_mean = float(np.mean(aucs))
    cv_std = float(np.std(aucs))
    feat_ver = "base" if not groups else "+".join(groups)
    if drop_cols:
        feat_ver += "-drop"
    run_id = (f"{owner}_feat_{model}_{feat_ver.replace('+', '')}_{date.today():%m%d}"
              + (f"_{tag}" if tag else ""))

    params = dict(cfg["model"].get(model, {}))
    n_feat = int(X.shape[1]) + (1 if "T" in set(groups) else 0)
    params.update({"features": feat_ver, "seed": seed, "n_features": n_feat})

    row = {
        "run_id": run_id,
        "date": str(date.today()),
        "owner": owner,
        "stage": "feat",
        "model": model,
        "feat_ver": feat_ver,
        "split_ver": split_ver,
        "params_hash": params_hash(params),
        "cv_mean": round(cv_mean, 6),
        "cv_std": round(cv_std, 6),
        "public_lb": "",
        "git_commit": git_commit(),
        "notes": notes or (
            f"n_feat={n_feat}" + (f"; drop={','.join(drop_cols)}" if drop_cols else "")
        ),
    }

    if verbose:
        print(f"\n[train] OOF AUC = {cv_mean:.5f} ± {cv_std:.5f}  ({n_feat} 特征)")
        if cv_std > 0.01:
            print("[train][提示] 折间波动偏大，先查划分或数据泄漏，再谈提升")

    if log:
        p = _append_log(cfg, row)
        oof_dir = Path(cfg["paths"]["oof_dir"])
        if not oof_dir.is_absolute():
            oof_dir = ROOT / oof_dir
        oof_dir.mkdir(parents=True, exist_ok=True)
        out = pd.DataFrame({"row": np.arange(len(oof)), "y": y, "pred": oof})
        if schema["id_col"] and schema["id_col"] in df.columns:
            out.insert(0, schema["id_col"], df[schema["id_col"]].to_numpy())
        out.to_parquet(oof_dir / f"{run_id}.parquet", index=False)
        if verbose:
            print(f"[train] 登记 -> {p}")
            print(f"[train] OOF  -> {oof_dir / (run_id + '.parquet')}")

    return {
        "run_id": run_id, "cv_mean": cv_mean, "cv_std": cv_std,
        "fold_aucs": aucs, "n_features": int(X.shape[1]), "oof": oof,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="5 折训练")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--model", default="lgbm", choices=["lgbm", "lr"])
    ap.add_argument("--groups", default="base", help="逗号分隔：base,A,B,C")
    ap.add_argument("--sample", type=int, default=None, help="调试采样行数")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--owner", default="P1")
    ap.add_argument("--notes", default="")
    ap.add_argument("--drop-cols", default="",
                    help="逗号分隔，从最终特征矩阵里剔除这些列（用于事后特征实验）")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    drop = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    run_cv(load_config(args.config), model=args.model, groups=groups,
           sample=args.sample, seed=args.seed, log=not args.no_log,
           tag=args.tag, owner=args.owner, notes=args.notes,
           drop_cols=drop or None)


if __name__ == "__main__":
    main()
