"""划分策略：冻结 / 读取 / 三方案对比（P1-05）。

铁律：splits/fold_v1.json 一旦冻结就只读。所有实验必须通过 load_folds() 取折，
禁止自己 train_test_split，否则不同人的分数不可比。
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from src.common.config import ROOT, load_config, path_of


def _load_processed(cfg: dict) -> pd.DataFrame:
    p = path_of(cfg, "processed_dir") / "train.parquet"
    if not p.exists():
        raise SystemExit(f"[split] 找不到 {p}，先跑 make data")
    return pd.read_parquet(p)


def build_group_array(df: pd.DataFrame, cfg: dict) -> np.ndarray | None:
    """group 策略的实体列。distance_bins = 用 Flight Distance 分箱当伪实体。"""
    spec = cfg["split"].get("group_col")
    if not spec:
        return None
    if spec == "distance_bins":
        col = "Flight Distance"
        if col not in df.columns:
            raise SystemExit(f"[split] 没有 {col}，无法构造 distance_bins")
        n_bins = int(cfg["split"].get("group_bins", 50))
        r = df[col].rank(method="first")
        return pd.qcut(r, n_bins, labels=False).to_numpy()
    if spec not in df.columns:
        raise SystemExit(f"[split] 分组列 {spec} 不存在")
    return df[spec].to_numpy()


def build_folds(df: pd.DataFrame, cfg: dict, strategy: str | None = None) -> np.ndarray:
    """返回 fold_of_row：长度 n 的数组，取值 0..k-1。"""
    y = df[cfg["data"]["target"]].to_numpy()
    n = len(df)
    s = strategy or cfg["split"]["strategy"]
    k = int(cfg["split"]["n_splits"])
    seed = int(cfg["split"]["seed"])
    idx = np.arange(n)
    fold = np.full(n, -1, dtype=np.int16)

    if s == "time":
        col = cfg["split"].get("time_col")
        if not col or col not in df.columns:
            raise SystemExit("[split] time 策略需要 split.time_col 且该列存在")
        order = np.argsort(df[col].to_numpy(), kind="mergesort")
        for f, chunk in enumerate(np.array_split(order, k)):
            fold[chunk] = f
        return fold

    if s == "group":
        g = build_group_array(df, cfg)
        if g is None:
            raise SystemExit("[split] group 策略需要 split.group_col")
        it = GroupKFold(n_splits=k).split(idx, y, g)
    elif s == "kfold":
        it = KFold(n_splits=k, shuffle=True, random_state=seed).split(idx)
    elif s == "stratified":
        it = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(idx, y)
    else:
        raise SystemExit(f"[split] 未知策略: {s}")

    for f, (_, va) in enumerate(it):
        fold[va] = f
    assert (fold >= 0).all(), "有样本没被分到任何折"
    return fold


def load_folds(cfg: dict | None = None, path: str | Path | None = None):
    """读取冻结划分。返回 (folds, meta)，folds 是 [(train_idx, valid_idx), ...]。"""
    p = Path(path or cfg["paths"]["splits"])
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise SystemExit(f"[split] 找不到冻结划分 {p}\n        先跑 make split（P1-05 交付物）")
    meta = json.loads(p.read_text(encoding="utf-8"))
    arr = np.asarray(meta["fold_of_row"], dtype=np.int16)
    folds = [(np.where(arr != k)[0], np.where(arr == k)[0]) for k in range(meta["n_splits"])]
    return folds, meta


def freeze(cfg: dict) -> None:
    df = _load_processed(cfg)
    fold = build_folds(df, cfg)
    y = df[cfg["data"]["target"]].to_numpy()
    k = int(cfg["split"]["n_splits"])

    out = Path(cfg["paths"]["splits"])
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "version": "v1",
        "created_at": str(date.today()),
        "strategy": cfg["split"]["strategy"],
        "n_splits": k,
        "seed": int(cfg["split"]["seed"]),
        "target": cfg["data"]["target"],
        "group_col": cfg["split"].get("group_col"),
        "n_rows": len(df),
        "fold_counts": [int((fold == f).sum()) for f in range(k)],
        "positive_rate_by_fold": [round(float(y[fold == f].mean()), 6) for f in range(k)],
        "fold_of_row": fold.tolist(),
    }
    out.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    print(f"[split] 策略={meta['strategy']} 折数={k} seed={meta['seed']} 行数={meta['n_rows']:,}")
    print(f"[split] 每折正例率: {meta['positive_rate_by_fold']}")
    print(f"[split] 已冻结 -> {out}")
    print("[split] 从现在起该文件只读；任何人改它，其历史实验作废。")


def compare(cfg: dict, score: bool = False) -> None:
    df = _load_processed(cfg)
    y = df[cfg["data"]["target"]].to_numpy()
    k = int(cfg["split"]["n_splits"])
    rows = []

    for s in ("kfold", "stratified", "group"):
        try:
            fold = build_folds(df, cfg, strategy=s)
        except SystemExit as e:
            print(f"[compare] 跳过 {s}: {e}")
            continue
        rates = [float(y[fold == f].mean()) for f in range(k)]
        sizes = [int((fold == f).sum()) for f in range(k)]
        row = {
            "strategy": s,
            "n_splits": k,
            "pos_mean": round(float(np.mean(rates)), 6),
            "pos_std": round(float(np.std(rates)), 6),
            "fold_size_min": min(sizes),
            "fold_size_max": max(sizes),
        }
        if score:
            from src.models.train import run_cv

            res = run_cv(cfg, model="lgbm", groups=["base"], fold_array=fold,
                         log=False, verbose=False)
            row["cv_mean"] = round(res["cv_mean"], 6)
            row["cv_std"] = round(res["cv_std"], 6)
        rows.append(row)

    table = pd.DataFrame(rows)
    print("\n[compare] 三方案对比")
    print(table.to_string(index=False))

    out = Path(cfg["paths"]["reports"]) / "split_compare.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[compare] 已写入 {out}")
    print("[compare] 选型口径：正例率波动小 + CV 与 LB 差距最小；不要只挑分数最高。")


def main() -> None:
    ap = argparse.ArgumentParser(description="划分策略管理")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("freeze", help="冻结 fold_v1.json")
    f.add_argument("--config", default="configs/base.yaml")

    c = sub.add_parser("compare", help="三方案对比")
    c.add_argument("--config", default="configs/base.yaml")
    c.add_argument("--score", action="store_true", help="同时跑 lgbm base 打分")

    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.cmd == "freeze":
        freeze(cfg)
    else:
        compare(cfg, score=args.score)


if __name__ == "__main__":
    main()
